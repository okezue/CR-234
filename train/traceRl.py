"""Policy learning from production traces with evaluative feedback, through Clash Royale.

A recorded game is one trajectory of a human policy through an environment nobody can recreate: the opponent was a human and the
simulator is a biased world model of the rest. The feedback is evaluative, one bit per game (who won), so environment replay (rerun
the policy from the recorded state, sample a group of plays, rank them) is unavailable in the real environment. This module compares
the ways of turning the single recorded trajectory into a policy update, on the decision records of train.decisionStates:

  bc         maximum likelihood on the recorded plays; the outcome is ignored (the behaviour policy estimate).
  reinforce  single-trajectory REINFORCE on the terminal outcome with the constant baseline one half: signed weights, no group,
             no constraint (the failure mode to measure against).
  ppo1       one sample per prompt: advantage y - V(s) from a state-value baseline fit on real outcomes, clipped importance ratio
             against the frozen behaviour policy, so the update stays where the data has support.
  awr        advantage-weighted regression: exp(A / beta) weights on the log-likelihood, clipped; never pushes mass off the record.
  simgroup   environment replay in the world model: group-relative advantages from counterfactual rollouts of the simulator
             (train.counterfactual) on the same decisions, with a trust weight per game phase.

Policy: a card from the recorded hand and a deploy cell (train.rl CELLS) given the state. Held out whole games: winner gap (mean
log-probability of the recorded play in won minus lost games), agreement with winners' and losers' plays, KL to the behaviour
policy, and the Q-eval, a P(win | state, play) model fit on real outcomes that scores each policy's play distribution over plays
with support in the data (a direct-method off-policy estimate, biased where the policy leaves the data).
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train.counterfactual import GH,GW,cell_of,load_groups
from train.decisionHead import PHASES,phase,relative
from train.decisionStates import load

N_CELLS=GW*GH;HAND=4


def stack(npzs):
    # several extraction files as one record set
    cols=None;Xs=[];games={}
    for p in npzs:
        c,X,g=load(p);Xs.append(X);games.update(g)
        cols=c if cols is None else {k:np.concatenate([cols[k],c[k]]) for k in cols}
    return cols,np.concatenate(Xs),games


def prepare(cols,X,games,vocab=None):
    # standardized state, hand indices, recorded card and cell, real label; drawn games are dropped
    keep=[i for i,b in enumerate(cols['bid']) if games[str(b)]['actual_winner'] is not None]
    hands=[str(h).split('|') for h in cols['hand'][keep]];cards=[str(c) for c in cols['card'][keep]]
    vocab=vocab or sorted({c for h in hands for c in h if c}|set(cards));idx={c:i for i,c in enumerate(vocab)}
    S=np.concatenate([X[keep],relative(X[keep])],1);mu=S.mean(0);sd=S.std(0)+1e-6;S=((S-mu)/sd).astype(np.float32)
    H=np.full((len(keep),HAND),-1,np.int64)
    for r,(h,c) in enumerate(zip(hands,cards)):
        menu=list(dict.fromkeys([c]+[x for x in h if x in idx]))[:HAND]
        for j,x in enumerate(menu):H[r,j]=idx[x]
    card=np.array([idx[c] for c in cards]);cell=np.array([cell_of(x,y) for x,y in zip(cols['x'][keep],cols['y'][keep])])
    teams=[str(t) for t in cols['team'][keep]]
    y=np.array([1.0 if games[str(b)]['actual_winner']==t else 0.0 for b,t in zip(cols['bid'][keep],teams)],np.float32)
    return {'S':torch.tensor(S),'H':torch.tensor(H),'card':torch.tensor(card),'cell':torch.tensor(cell),'y':torch.tensor(y),
            'phase':np.array([phase(float(t)) for t in cols['t'][keep]]),'gid':np.array([str(b) for b in cols['bid'][keep]]),
            'idx':np.array([int(i) for i in cols['idx'][keep]]),'vocab':vocab,'n_state':S.shape[1],'mu':mu,'sd':sd}


def split(gid,holdout=0.25):
    h=np.array([zlib.crc32(g.encode())%1000/1000.0 for g in gid])
    return torch.tensor(h>=holdout),torch.tensor(h<holdout)


class Policy(nn.Module):
    # card logits over the vocabulary masked to the hand, cell logits conditioned on the chosen card
    def __init__(self,n_state,n_card,hidden=128,emb=16):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(n_state,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU())
        self.card=nn.Linear(hidden,n_card);self.emb=nn.Embedding(n_card,emb);self.cell=nn.Linear(hidden+emb,N_CELLS)
    def card_logp(self,S,H):
        h=self.trunk(S);logits=self.card(h);mask=torch.zeros_like(logits,dtype=torch.bool)
        valid=H>=0;mask.scatter_(1,H.clamp(min=0),valid)
        return F.log_softmax(logits.masked_fill(~mask,-1e9),1),h
    def cell_logp(self,h,card):
        return F.log_softmax(self.cell(torch.cat([h,self.emb(card)],1)),1)
    def logp(self,S,H,card,cell):
        lc,h=self.card_logp(S,H)
        return lc.gather(1,card[:,None])[:,0]+self.cell_logp(h,card).gather(1,cell[:,None])[:,0]
    def menu_logp(self,S,H):
        # log-probability of every (hand card, cell) play: (n, HAND, N_CELLS), -inf for empty hand slots
        lc,h=self.card_logp(S,H);out=torch.full((len(S),HAND,N_CELLS),-1e9)
        for j in range(HAND):
            c=H[:,j];ok=c>=0
            if ok.any():out[ok,j]=lc[ok].gather(1,c[ok,None])+self.cell_logp(h[ok],c[ok])
        return out


class Head(nn.Module):
    # V(s) when n_card is zero, otherwise Q(s, card, cell) as P(actor wins), both fit on the recorded outcome
    def __init__(self,n_state,n_card=0,hidden=128,emb=16):
        super().__init__()
        self.n_card=n_card;extra=(emb+N_CELLS) if n_card else 0
        self.emb=nn.Embedding(max(n_card,1),emb)
        self.net=nn.Sequential(nn.Linear(n_state+extra,hidden),nn.GELU(),nn.Linear(hidden,1))
    def forward(self,S,card=None,cell=None):
        if self.n_card:S=torch.cat([S,self.emb(card),F.one_hot(cell,N_CELLS).float()],1)
        return self.net(S)[:,0]


def fit_head(head,S,y,card=None,cell=None,epochs=60,lr=2e-3,l2=1e-4,batch=4096,seed=0):
    torch.manual_seed(seed);opt=torch.optim.AdamW(head.parameters(),lr=lr,weight_decay=l2);n=len(y)
    for _ in range(epochs):
        perm=torch.randperm(n)
        for i in range(0,n,batch):
            b=perm[i:i+batch];logit=head(S[b],None if card is None else card[b],None if cell is None else cell[b])
            loss=F.binary_cross_entropy_with_logits(logit,y[b]);opt.zero_grad();loss.backward();opt.step()
    return head


def train_policy(pol,d,tr,method,ref=None,V=None,adv_cf=None,epochs=30,lr=1e-3,batch=2048,beta=0.5,clip=0.2,wmax=20.0,ent=0.0,seed=0,refresh=0,log=False):
    # one method's update on the training records; ref is the frozen behaviour policy (ppo1, simgroup), V the frozen value head;
    # refresh > 0 re-anchors the clipped ratio to the current policy every that many epochs (iterated trust region, as with a
    # world model that can be queried again), 0 keeps the behaviour policy as the anchor for the whole run (offline setting)
    torch.manual_seed(seed);opt=torch.optim.AdamW(pol.parameters(),lr=lr,weight_decay=1e-5)
    S,H,card,cell,y=(d[k][tr] for k in ('S','H','card','cell','y'))
    with torch.no_grad():
        if method in ('ppo1','awr'):
            A=y-torch.sigmoid(V(S));A=A/(A.std()+1e-6)
        if method=='reinforce':A=2*(y-0.5)
        if ref is not None:old=ref.logp(S,H,card,cell)
    if method=='simgroup':S,H,card,cell,A,old=adv_cf
    n=len(S);t0=time.monotonic()
    for e in range(epochs):
        if refresh and e and e%refresh==0 and method in ('ppo1','simgroup'):
            with torch.no_grad():old=pol.logp(S,H,card,cell)
        if log and e and e%5==0:print(f'{method} epoch {e}/{epochs} {time.monotonic()-t0:.0f}s',flush=True)
        perm=torch.randperm(n)
        for i in range(0,n,batch):
            b=perm[i:i+batch];lp=pol.logp(S[b],H[b],card[b],cell[b])
            if method=='bc':loss=-lp.mean()
            elif method=='reinforce':loss=-(A[b]*lp).mean()
            elif method=='awr':
                w=torch.exp(A[b]/beta).clamp(max=wmax);loss=-(w*lp).sum()/w.sum()
            else:
                r=torch.exp(lp-old[b]);loss=-torch.min(r*A[b],r.clamp(1-clip,1+clip)*A[b]).mean()
            if ent:
                lc,_=pol.card_logp(S[b],H[b]);loss=loss-ent*(-(lc.exp()*lc).sum(1)).mean()
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(pol.parameters(),1.0);opt.step()
    return pol


def sim_advantages(d,tr,groups,ref,trust=None):
    # (state, hand, card, cell, advantage, behaviour log-probability) rows from the counterfactual groups of training decisions: the
    # return of every rolled-out play minus the group mean, times the trust in the simulator for that phase
    idx={c:i for i,c in enumerate(d['vocab'])};rows=[]
    where={(g,int(i)):r for r,(g,i) in enumerate(zip(d['gid'],d['idx'])) if tr[r]}
    for (bid,i),plays in groups.items():
        r=where.get((bid,i))
        if r is None or len(plays)<2:continue
        rets=np.array([p[5] for p in plays]);adv=rets-rets.mean();w=1.0 if trust is None else trust[d['phase'][r]]
        for (name,x,y,_,_,_),a in zip(plays,adv):
            if name in idx and idx[name] in set(d['H'][r].tolist()):rows.append((r,idx[name],cell_of(x,y),float(a)*w))
    if not rows:return None
    ri=torch.tensor([r for r,_,_,_ in rows]);card=torch.tensor([c for _,c,_,_ in rows]);cell=torch.tensor([c for _,_,c,_ in rows])
    A=torch.tensor([a for _,_,_,a in rows],dtype=torch.float32);A=A/(A.std()+1e-6);S=d['S'][ri];H=d['H'][ri]
    with torch.no_grad():old=ref.logp(S,H,card,cell)
    return S,H,card,cell,A,old


def support(d,tr,min_n=5):
    # (card, cell) plays seen at least min_n times in training: the region where the Q-eval is trusted
    pairs,counts=np.unique(np.stack([d['card'][tr].numpy(),d['cell'][tr].numpy()],1),axis=0,return_counts=True)
    sup=torch.zeros(len(d['vocab']),N_CELLS,dtype=torch.bool)
    for (c,k),n in zip(pairs,counts):
        if n>=min_n:sup[c,k]=True
    return sup


def menu_q(Q,S,H,sup,chunk=256):
    # P(win) of every (hand card, cell) play and whether that play has support, in chunks of records
    n=len(S);q=torch.zeros(n,HAND,N_CELLS);ok=torch.zeros(n,HAND,N_CELLS,dtype=torch.bool);kk=torch.arange(N_CELLS)
    for i in range(0,n,chunk):
        s=S[i:i+chunk];h=H[i:i+chunk]
        for j in range(HAND):
            c=h[:,j];valid=c>=0
            if not valid.any():continue
            m=int(valid.sum());cc=c[valid].repeat_interleave(N_CELLS);ss=s[valid].repeat_interleave(N_CELLS,0)
            q[i:i+chunk][valid,j]=torch.sigmoid(Q(ss,cc,kk.repeat(m))).view(m,N_CELLS);ok[i:i+chunk][valid,j]=sup[c[valid]]
    return q,ok


def evaluate(pol,ref,Q,d,te,sup,q_subset=0,seed=0):
    # q_subset > 0 scores the Q-eval on that many held-out records drawn once per seed (the same records for every method)
    S,H,card,cell,y=(d[k][te] for k in ('S','H','card','cell','y'))
    with torch.no_grad():
        lp=pol.logp(S,H,card,cell);won=y>0.5;menu=pol.menu_logp(S,H);ref_menu=ref.menu_logp(S,H)
        p=menu.exp();pr=ref_menu.exp()
        kl=(p*(menu.clamp(min=-30)-ref_menu.clamp(min=-30))).sum((1,2))
        lc,_=pol.card_logp(S,H);top=lc.argmax(1)
        # Q over the menu, restricted to plays with support; the policy's mass is renormalized over that region
        sub=torch.arange(len(S)) if not q_subset or q_subset>=len(S) else torch.randperm(len(S),generator=torch.Generator().manual_seed(seed))[:q_subset]
        q,ok=menu_q(Q,S[sub],H[sub],sup)
        def qeval(prob):
            m=(prob*ok).sum((1,2));covered=m>1e-6
            return float((((prob*ok*q).sum((1,2)))[covered]/m[covered]).mean()),float(covered.float().mean()),float(m.mean())
        q_pol,cov,mass=qeval(p[sub]);q_ref,_,mass_ref=qeval(pr[sub])
        q_rec=torch.sigmoid(Q(S,card,cell))
    out={'logp':float(lp.mean()),'logp_won':float(lp[won].mean()),'logp_lost':float(lp[~won].mean()),
         'winner_gap':float(lp[won].mean()-lp[~won].mean()),'top1_won':float((top[won]==card[won]).float().mean()),
         'top1_lost':float((top[~won]==card[~won]).float().mean()),'kl_to_bc':float(kl.mean()),'entropy':float(-(p*menu.clamp(min=-30)).sum((1,2)).mean()),
         'q_policy':q_pol,'q_bc':q_ref,'q_recorded':float(q_rec.mean()),'q_recorded_won':float(q_rec[won].mean()),
         'q_recorded_lost':float(q_rec[~won].mean()),'support_mass':mass,'support_mass_bc':mass_ref,'support_coverage':cov,'q_records':int(len(sub))}
    for i,(_,_,label) in enumerate(PHASES):
        m=torch.tensor(d['phase'][te.numpy()]==i)
        if m.sum()>=50:out[f'winner_gap_{label}']=float(lp[m&won].mean()-lp[m&~won].mean())
    return {k:round(v,4) for k,v in out.items()}


def run(npzs,holdout=0.25,out=None,epochs=30,head_epochs=60,cf=None,trust=None,methods=('bc','reinforce','ppo1','awr','simgroup'),seed=0,
        hidden=128,min_support=5,policies=None,refresh=0,log=False,head_l2=1e-4,batch=2048,q_subset=0):
    # policies, when a dict is given, receives the trained policy of every method and the prepared data for probing; simgroup uses
    # the counterfactual groups with trust one, simgroup_trust the same groups weighted by the trust list per game phase
    cols,X,games=stack(npzs);d=prepare(cols,X,games);tr,te=split(d['gid'],holdout);n_state=d['n_state'];n_card=len(d['vocab'])
    if policies is not None:policies.update({'data':d,'train':tr,'test':te})
    report={'records':int(len(d['y'])),'train':int(tr.sum()),'test':int(te.sum()),'games':len(set(d['gid'])),'cards':n_card,'methods':{},
            'settings':{'epochs':epochs,'head_epochs':head_epochs,'head_l2':head_l2,'batch':batch,'hidden':hidden,'refresh':refresh,'trust':trust,'seed':seed}}
    V=fit_head(Head(n_state,hidden=hidden),d['S'][tr],d['y'][tr],epochs=head_epochs,l2=head_l2,seed=seed)
    Q=fit_head(Head(n_state,n_card,hidden=hidden),d['S'][tr],d['y'][tr],d['card'][tr],d['cell'][tr],epochs=head_epochs,l2=head_l2,seed=seed)
    with torch.no_grad():
        for name,h,args in (('value',V,(d['S'][te],)),('q',Q,(d['S'][te],d['card'][te],d['cell'][te]))):
            p=torch.sigmoid(h(*args));y=d['y'][te]
            report[f'{name}_head']={'accuracy':round(float(((p>0.5).float()==y).float().mean()),4),
                                    'log_loss':round(float(F.binary_cross_entropy(p.clamp(1e-6,1-1e-6),y)),4)}
    sup=support(d,tr,min_support);report['support_pairs']=int(sup.sum())
    bc=train_policy(Policy(n_state,n_card,hidden),d,tr,'bc',epochs=epochs,batch=batch,seed=seed,log=log);bc.eval()
    for p in bc.parameters():p.requires_grad_(False)
    report['methods']['bc']=evaluate(bc,bc,Q,d,te,sup,q_subset,seed)
    if policies is not None:policies['bc']=bc
    groups=load_groups(cf) if cf else None
    adv={'simgroup':sim_advantages(d,tr,groups,bc,None) if groups else None,
         'simgroup_trust':sim_advantages(d,tr,groups,bc,trust) if groups and trust is not None else None}
    if adv['simgroup']:report['simgroup_rows']=int(len(adv['simgroup'][2]))
    for method in methods:
        if method=='bc' or (method in adv and adv[method] is None):continue
        pol=Policy(n_state,n_card,hidden);pol.load_state_dict(bc.state_dict())
        for p in pol.parameters():p.requires_grad_(True)
        kind='simgroup' if method in adv else method
        train_policy(pol,d,tr,kind,ref=bc,V=V,adv_cf=adv.get(method),epochs=epochs,batch=batch,seed=seed,refresh=refresh,log=log)
        pol.eval();report['methods'][method]=evaluate(pol,bc,Q,d,te,sup,q_subset,seed)
        if policies is not None:policies[method]=pol
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=1)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--npz',nargs='+',required=True);ap.add_argument('--holdout',type=float,default=0.25)
    ap.add_argument('--out');ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--head_epochs',type=int,default=60)
    ap.add_argument('--cf',help='counterfactual rollout file from train/counterfactual.py')
    ap.add_argument('--trust',help='JSON list of four trust weights for the simulator by game phase (default all one)')
    ap.add_argument('--methods',nargs='+',default=['bc','reinforce','ppo1','awr','simgroup','simgroup_trust']);ap.add_argument('--seed',type=int,default=0)
    ap.add_argument('--hidden',type=int,default=128);ap.add_argument('--refresh',type=int,default=0,help='epochs between ratio anchor refreshes, 0 never')
    ap.add_argument('--threads',type=int,default=0,help='torch CPU threads, 0 for the default; small batches run faster with 8 to 16')
    ap.add_argument('--head_l2',type=float,default=1e-4);ap.add_argument('--batch',type=int,default=2048)
    ap.add_argument('--q_subset',type=int,default=0,help='held-out records scored by the Q-eval, 0 for all')
    a=ap.parse_args()
    if a.threads:torch.set_num_threads(a.threads)
    trust=json.loads(a.trust) if a.trust else None
    r=run(a.npz,a.holdout,a.out,a.epochs,a.head_epochs,a.cf,trust,tuple(a.methods),a.seed,a.hidden,refresh=a.refresh,log=True,head_l2=a.head_l2,
          batch=a.batch,q_subset=a.q_subset)
    print(json.dumps({k:v for k,v in r.items() if k!='methods'},indent=1))
    keys=('logp','winner_gap','top1_won','top1_lost','kl_to_bc','entropy','q_policy','q_bc','q_recorded','support_mass')
    print('method',*keys)
    for m,row in r['methods'].items():print(m,*(row[k] for k in keys))


if __name__=='__main__':main()
