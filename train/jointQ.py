"""Joint decision head: P(the actor wins | state, play) fit on real labels (one per game) and world-model rollout labels (many per
state) with a trust weight on the rollout loss.

Real rows are the training decisions with the recorded winner. Rollout rows are the counterfactual menus of the training games
(train.counterfactual), each play labelled by the simulator in one of two ways: target 'win' is the simulated winner for the actor
as a soft label (one half for a draw), a claim about the level of P(win); target 'relative' is the play's return minus its group
mean, which the head must match with its logit minus the group's mean logit, a claim only about how the plays of one state compare.
The trust weight multiplies the loss of every rollout row (so trust one makes a rollout label worth one real label), optionally
scaled per game phase by the decision head's verdict weights, and is swept; trust zero is the real-only head of train.traceRl.
Evaluation is on the held-out games' real outcomes: log loss, accuracy, Brier score, the gap between the head's probability of the
recorded play in won and lost games, and the paired per-game log-loss difference to the real-only head with its standard error
over games; the head's within-group rank agreement with the simulator on the held-out games' menus says how much of the world
model it absorbed.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from train.counterfactual import cell_of,load_groups
from train.traceRl import Head,prepare,split,stack

LAMS=(0.3,1.0,3.0,10.0,30.0);TARGETS=('win','relative')


def rollout_rows(d,mask,groups):
    # rows of the counterfactual menus whose decision lies in mask, contiguous by group: record index, play, soft simulated win,
    # group-relative return, group id and phase, with the group offsets for batching by whole groups
    idx={c:i for i,c in enumerate(d['vocab'])};where={(g,int(i)):r for r,(g,i) in enumerate(zip(d['gid'],d['idx'])) if mask[r]};rows=[];gid=0
    for key,plays in groups.items():
        r=where.get(key)
        if r is None or len(plays)<2:continue
        rets=np.array([p[5] for p in plays]);hand=set(d['H'][r].tolist());added=0
        for (name,x,y,win,_,ret) in plays:
            if name in idx and idx[name] in hand:rows.append((r,idx[name],cell_of(x,y),0.5+0.5*win,ret-rets.mean(),gid,int(d['phase'][r])));added+=1
        gid+=added>0
    if not rows:return None
    cols=list(zip(*rows));g=torch.tensor(cols[5]);lens=torch.bincount(g,minlength=gid);ri=torch.tensor(cols[0])
    return {'ri':ri,'S':d['S'][ri],'card':torch.tensor(cols[1]),'cell':torch.tensor(cols[2]),'p':torch.tensor(cols[3],dtype=torch.float32),
            'rel':torch.tensor(cols[4],dtype=torch.float32),'gid':g,'phase':torch.tensor(cols[6]),'lens':lens,'starts':torch.cumsum(lens,0)-lens,'groups':gid}


def group_rows(roll,gsel):
    starts=roll['starts'][gsel];lens=roll['lens'][gsel];n=int(lens.sum())
    return torch.repeat_interleave(starts,lens)+torch.arange(n)-torch.repeat_interleave(torch.cumsum(lens,0)-lens,lens)


def rollout_loss(head,roll,rows,target):
    logit=head(roll['S'][rows],roll['card'][rows],roll['cell'][rows])
    if target=='win':return F.binary_cross_entropy_with_logits(logit,roll['p'][rows],reduction='none')
    _,inv=torch.unique(roll['gid'][rows],return_inverse=True);cnt=torch.bincount(inv).float()
    mean=torch.zeros(len(cnt)).index_add_(0,inv,logit)/cnt
    return (logit-mean[inv]-roll['rel'][rows])**2


def fit_joint(head,S,card,cell,y,roll=None,lam=0.0,target='win',trust=None,epochs=8,lr=2e-3,l2=1e-3,batch=4096,seed=0):
    # one pass over the real rows and one over the rollout groups per epoch; the rollout loss is summed with its weights and
    # divided by the real batch size, so a weight of one makes a rollout row count as one real row
    torch.manual_seed(seed);opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=l2);n=len(y);nb=(n+batch-1)//batch
    G=roll['groups'] if roll else 0;gb=(G+nb-1)//nb
    w=None if not G else (torch.full((len(roll['ri']),),float(lam)) if trust is None else lam*torch.tensor(trust,dtype=torch.float32)[roll['phase']])
    for _ in range(epochs):
        perm=torch.randperm(n);gperm=torch.randperm(G) if G else None
        for k in range(nb):
            b=perm[k*batch:(k+1)*batch];loss=F.binary_cross_entropy_with_logits(head(S[b],card[b],cell[b]),y[b])
            if G and lam:
                rows=group_rows(roll,gperm[k*gb:(k+1)*gb]);loss=loss+(w[rows]*rollout_loss(head,roll,rows,target)).sum()/len(b)
            opt.zero_grad();loss.backward();opt.step()
    return head


def evaluate_head(head,d,te):
    S,card,cell,y=(d[k][te] for k in ('S','card','cell','y'))
    with torch.no_grad():p=torch.sigmoid(head(S,card,cell)).clamp(1e-6,1-1e-6)
    ll=-(y*p.log()+(1-y)*(1-p).log());won=y>0.5;gids,inv=np.unique(d['gid'][te.numpy()],return_inverse=True)
    per_game=np.bincount(inv,weights=ll.numpy())/np.bincount(inv)
    out={'log_loss':float(ll.mean()),'accuracy':float(((p>0.5)==won).float().mean()),'brier':float(((p-y)**2).mean()),
         'gap':float(p[won].mean()-p[~won].mean()),'n':int(len(y)),'games':int(len(gids))}
    return {k:round(v,4) if isinstance(v,float) else v for k,v in out.items()},per_game


def ranks(v):
    # average ranks, so tied returns (two alternatives with the same simulated ending) do not decide the correlation
    order=np.argsort(v);r=np.empty(len(v));r[order]=np.arange(len(v));_,inv=np.unique(v,return_inverse=True)
    return (np.bincount(inv,weights=r)/np.bincount(inv))[inv]


def sim_agreement(head,roll):
    # within-group rank correlation of the head's logits with the simulated returns and top-play agreement, over groups of at least
    # three plays whose returns differ
    with torch.no_grad():logit=head(roll['S'],roll['card'],roll['cell'])
    rs=[];top=[]
    for g in range(roll['groups']):
        rows=torch.arange(roll['starts'][g],roll['starts'][g]+roll['lens'][g]);r=roll['rel'][rows].numpy();q=logit[rows].numpy()
        if len(rows)<3 or r.std()<1e-9 or q.std()<1e-9:continue
        rs.append(np.corrcoef(ranks(r),ranks(q))[0,1]);top.append(int(r.argmax())==int(q.argmax()))
    return {'groups':len(rs),'rank_corr':round(float(np.mean(rs)),4) if rs else None,'top1':round(float(np.mean(top)),4) if top else None}


def run(npzs,cf,holdout=0.25,out=None,lams=LAMS,targets=TARGETS,trust=None,head_epochs=8,head_l2=1e-3,hidden=128,batch=4096,seed=0,log=False,heads=None):
    # heads, when a dict is given, receives every fitted head and the prepared data for probing
    cols,X,games=stack(npzs);d=prepare(cols,X,games);tr,te=split(d['gid'],holdout);groups=load_groups(cf);n_state=d['n_state'];n_card=len(d['vocab'])
    roll=rollout_rows(d,tr.numpy(),groups);roll_te=rollout_rows(d,te.numpy(),groups)
    if heads is not None:heads.update({'data':d,'train':tr,'test':te})
    report={'records':int(len(d['y'])),'train':int(tr.sum()),'test':int(te.sum()),'games':len(set(d['gid'])),'test_games':len(set(d['gid'][te.numpy()])),
            'rollout_rows':int(len(roll['ri'])),'rollout_groups':roll['groups'],'heldout_rollout_groups':roll_te['groups'] if roll_te else 0,
            'settings':{'lams':list(lams),'targets':list(targets),'trust':trust,'head_epochs':head_epochs,'head_l2':head_l2,'hidden':hidden,'batch':batch,'seed':seed},
            'arms':{}}
    S,card,cell,y=(d[k][tr] for k in ('S','card','cell','y'))
    arms=[('real',0.0,'win',None,seed),('real:s1',0.0,'win',None,seed+1)]
    for target in targets:
        for lam in lams:
            arms.append((f'{target}:{lam:g}',lam,target,None,seed))
            if trust is not None:arms.append((f'{target}:{lam:g}:phase',lam,target,trust,seed))
    per_game={}
    for name,lam,target,tw,s in arms:
        t0=time.monotonic();torch.manual_seed(s);head=Head(n_state,n_card,hidden=hidden)
        fit_joint(head,S,card,cell,y,roll,lam,target,tw,head_epochs,l2=head_l2,batch=batch,seed=s)
        m,pg=evaluate_head(head,d,te);per_game[name]=pg;m['sim']=sim_agreement(head,roll_te) if roll_te else None;m['seconds']=round(time.monotonic()-t0,1)
        report['arms'][name]=m
        if heads is not None:heads[name]=head
        if log:print(name,m,flush=True)
    base=per_game['real']
    for name,pg in per_game.items():
        diff=pg-base;report['arms'][name]['vs_real']={'log_loss_diff':round(float(diff.mean()),5),'se':round(float(diff.std(ddof=1)/np.sqrt(len(diff))),5),
                                                        'games_better':round(float((diff<0).mean()),4)}
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=1)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--npz',nargs='+',required=True);ap.add_argument('--cf',required=True);ap.add_argument('--out')
    ap.add_argument('--holdout',type=float,default=0.25);ap.add_argument('--lams',type=float,nargs='+',default=list(LAMS))
    ap.add_argument('--targets',nargs='+',default=list(TARGETS));ap.add_argument('--trust',help='JSON list of four per-phase trust scales')
    ap.add_argument('--head_epochs',type=int,default=8);ap.add_argument('--head_l2',type=float,default=1e-3);ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--batch',type=int,default=4096);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--threads',type=int,default=0);a=ap.parse_args()
    if a.threads:torch.set_num_threads(a.threads)
    trust=json.loads(a.trust) if a.trust else None
    r=run(a.npz,a.cf,a.holdout,a.out,tuple(a.lams),tuple(a.targets),trust,a.head_epochs,a.head_l2,a.hidden,a.batch,a.seed,log=True)
    print(json.dumps({k:v for k,v in r.items() if k!='arms'},indent=1))
    print('arm log_loss accuracy brier gap diff se games_better sim_rank sim_top1')
    for name,m in r['arms'].items():
        v=m['vs_real'];sm=m['sim'] or {}
        print(name,m['log_loss'],m['accuracy'],m['brier'],m['gap'],v['log_loss_diff'],v['se'],v['games_better'],sm.get('rank_corr'),sm.get('top1'))


if __name__=='__main__':main()
