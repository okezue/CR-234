"""World-model bias measure: iterate the clipped update on the world model's advantages with refreshed anchors and watch where the
policy goes.

With the anchor refreshed every few epochs the clipped ratio no longer bounds the distance from the behaviour policy, so a fixed set
of advantages drives the policy as far as the epochs allow (the iterated trust region a world model permits because it can be
queried again). The same iteration on the real-outcome advantages (ppo1) and advantage-weighted regression are the reference, and
ppo1 on outcomes flipped at random per training game is the noise floor. At every checkpoint each arm is evaluated on the held-out
games (train.traceRl.evaluate) and its per-state preference shift, the log ratio of the policy to the behaviour policy over the menu
of (hand card, cell) plays, is kept on a fixed subset of held-out states. The direction agreement between two arms is the mean cosine
(and rank correlation) of their shifts over those states, at the full menu and marginalised to the card; the drift is read as a
function of the KL to the behaviour policy. An arm is named method[:seed], the seed changing only the batch order.
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from train.counterfactual import load_groups
from train.traceRl import Head,Policy,evaluate,fit_head,prepare,sim_advantages,split,stack,support,train_policy

ARMS=('simgroup','simgroup:1','ppo1','ppo1:1','awr','ppo1_shuffled')
DRIFT_KEYS=('kl_to_bc','winner_gap','logp','top1_won','top1_lost','q_policy','entropy','support_mass','winner_gap_120to180')


def shift(pol,ref,S,H,chunk=4096):
    # per-state preference shift: log ratio to the behaviour policy over the menu (flattened) and marginalised to the hand card;
    # empty hand slots are zero in both
    menu=[];card=[]
    with torch.no_grad():
        for i in range(0,len(S),chunk):
            s,h=S[i:i+chunk],H[i:i+chunk];ok=(h>=0).float();hc=h.clamp(min=0)
            menu.append(((pol.menu_logp(s,h)-ref.menu_logp(s,h))*ok[:,:,None]).flatten(1))
            lp,_=pol.card_logp(s,h);lr,_=ref.card_logp(s,h);card.append((lp.gather(1,hc)-lr.gather(1,hc))*ok)
    return torch.cat(menu),torch.cat(card)


def cosine(a,b):
    return float(((a*b).sum(1)/(a.norm(dim=1)*b.norm(dim=1)+1e-8)).mean())


def rank_corr(a,b):
    ra=a.argsort(1).argsort(1).float();rb=b.argsort(1).argsort(1).float()
    return cosine(ra-ra.mean(1,keepdim=True),rb-rb.mean(1,keepdim=True))


def agreement(x,y):
    return {'cos_menu':round(cosine(x[0],y[0]),4),'cos_card':round(cosine(x[1],y[1]),4),'rank_menu':round(rank_corr(x[0],y[0]),4)}


def shuffle_games(d,seed):
    # the recorded winner flipped for a random half of the games: both actors of a game keep opposite labels, the label keeps its
    # distribution and loses its relation to the states
    rng=np.random.default_rng(seed);gids=np.unique(d['gid']);flip=dict(zip(gids.tolist(),(rng.random(len(gids))<0.5).tolist()))
    f=torch.tensor([flip[g] for g in d['gid']]);y=d['y'].clone();y[f]=1-y[f]
    return {**d,'y':y}


def run(npzs,cf,holdout=0.25,out=None,checkpoints=(30,60,90),refresh=5,arms=ARMS,head_epochs=8,head_l2=1e-3,batch=4096,hidden=128,q_subset=50000,
        shift_subset=20000,min_support=5,seed=0,bc_epochs=30,log=False,policies=None):
    cols,X,games=stack(npzs);d=prepare(cols,X,games);tr,te=split(d['gid'],holdout);n_state=d['n_state'];n_card=len(d['vocab'])
    report={'records':int(len(d['y'])),'train':int(tr.sum()),'test':int(te.sum()),'games':len(set(d['gid'])),'test_games':len(set(d['gid'][te.numpy()])),
            'settings':{'checkpoints':list(checkpoints),'refresh':refresh,'arms':list(arms),'head_epochs':head_epochs,'head_l2':head_l2,'batch':batch,
                        'hidden':hidden,'q_subset':q_subset,'shift_subset':shift_subset,'seed':seed,'bc_epochs':bc_epochs},'arms':{}}
    V=fit_head(Head(n_state,hidden=hidden),d['S'][tr],d['y'][tr],epochs=head_epochs,l2=head_l2,seed=seed)
    Q=fit_head(Head(n_state,n_card,hidden=hidden),d['S'][tr],d['y'][tr],d['card'][tr],d['cell'][tr],epochs=head_epochs,l2=head_l2,seed=seed)
    sup=support(d,tr,min_support)
    bc=train_policy(Policy(n_state,n_card,hidden),d,tr,'bc',epochs=bc_epochs,batch=batch,seed=seed,log=log);bc.eval()
    for p in bc.parameters():p.requires_grad_(False)
    report['bc']=evaluate(bc,bc,Q,d,te,sup,q_subset,seed)
    adv=sim_advantages(d,tr,load_groups(cf),bc);report['simgroup_rows']=int(len(adv[2]))
    sub=torch.randperm(int(te.sum()),generator=torch.Generator().manual_seed(seed))[:shift_subset];S_sub=d['S'][te][sub];H_sub=d['H'][te][sub]
    shifts={};shuffled={}
    if policies is not None:policies.update({'data':d,'train':tr,'test':te,'bc':bc})
    for arm in arms:
        method,_,s=arm.partition(':');s=int(s) if s else seed;dd,Vv=d,V
        if method=='ppo1_shuffled':
            method='ppo1';dd=shuffle_games(d,s);Vv=fit_head(Head(n_state,hidden=hidden),dd['S'][tr],dd['y'][tr],epochs=head_epochs,l2=head_l2,seed=seed)
            shuffled[arm]=round(float((dd['y'][tr]!=d['y'][tr]).float().mean()),4)
        pol=Policy(n_state,n_card,hidden);pol.load_state_dict(bc.state_dict());report['arms'][arm]={};t0=time.monotonic()
        def snap(e,p,arm=arm):
            report['arms'][arm][str(e)]=evaluate(p,bc,Q,d,te,sup,q_subset,seed);shifts[f'{arm}@{e}']=shift(p,bc,S_sub,H_sub)
            if log:print(arm,e,{k:report['arms'][arm][str(e)][k] for k in ('kl_to_bc','winner_gap','q_policy')},f'{time.monotonic()-t0:.0f}s',flush=True)
        train_policy(pol,dd,tr,method,ref=bc,V=Vv,adv_cf=adv,epochs=max(checkpoints),batch=batch,seed=s,refresh=refresh,log=log,snap=snap,snap_at=checkpoints)
        if policies is not None:policies[arm]=pol
    report['shuffled_fraction']=shuffled
    report['drift']={arm:[{'epochs':e,**{k:r[k] for k in DRIFT_KEYS if k in r}} for e,r in ((int(e),r) for e,r in report['arms'][arm].items())] for arm in arms}
    keys=list(shifts);report['agreement']={f'{a}|{b}':agreement(shifts[a],shifts[b]) for i,a in enumerate(keys) for b in keys[i+1:]}
    report['same_epoch']={str(e):{f'{a}|{b}':report['agreement'][f'{a}@{e}|{b}@{e}'] for i,a in enumerate(arms) for b in arms[i+1:]} for e in checkpoints}
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=1)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--npz',nargs='+',required=True);ap.add_argument('--cf',required=True);ap.add_argument('--out')
    ap.add_argument('--holdout',type=float,default=0.25);ap.add_argument('--checkpoints',type=int,nargs='+',default=[30,60,90])
    ap.add_argument('--refresh',type=int,default=5);ap.add_argument('--arms',nargs='+',default=list(ARMS));ap.add_argument('--head_epochs',type=int,default=8)
    ap.add_argument('--head_l2',type=float,default=1e-3);ap.add_argument('--batch',type=int,default=4096);ap.add_argument('--hidden',type=int,default=128)
    ap.add_argument('--q_subset',type=int,default=50000);ap.add_argument('--shift_subset',type=int,default=20000);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--threads',type=int,default=0);a=ap.parse_args()
    if a.threads:torch.set_num_threads(a.threads)
    r=run(a.npz,a.cf,a.holdout,a.out,tuple(a.checkpoints),a.refresh,tuple(a.arms),a.head_epochs,a.head_l2,a.batch,a.hidden,a.q_subset,a.shift_subset,seed=a.seed,log=True)
    print(json.dumps({k:r[k] for k in ('records','train','test','test_games','simgroup_rows','shuffled_fraction')},indent=1))
    print('arm epochs',*DRIFT_KEYS)
    for arm,rows in r['drift'].items():
        for row in rows:print(arm,row['epochs'],*(row.get(k) for k in DRIFT_KEYS))
    for e,pairs in r['same_epoch'].items():
        for k,v in pairs.items():print(e,k,v)


if __name__=='__main__':main()
