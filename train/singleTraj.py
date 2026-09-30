"""Single-trajectory RL on production traces at scale: FlashREINFORCE, SAO and BPCO trained in battle-time order on the replay corpus.

A recorded game gives two trajectories, one per player: the actor's recorded decisions, each two tokens (a card from the forced hand,
then a deploy cell given the card), the opponent's plays in between as observations, and the recorded outcome as the only reward (1
won, 0 lost; drawn games are dropped). The behaviour policy is a human, so its probabilities were never stored: every importance
ratio and trust region below uses a behaviour-cloning estimate mu (the production-trace twist), and the estimate is varied (fit on a
disjoint half of the same period, on the same games, on earlier games only, online, and tempered). Games are consumed in battle-time
order in one pass, B games per batch, one actor update per batch, then discarded (the continual setting). Recipes as published, at
the decision level:

  flash  FlashREINFORCE (Hu, Zhang et al. 2026, flashreinforce/loss.py): advantage R - mean(R) over the batch, token ratios pi/mu
         detached, a whole trajectory admitted only if its mean sampled-token Bernoulli KL(mu || pi) is at most delta, token mean per
         trajectory then mean over the batch, no critic, no clipping.
  sao    SAO (Hou et al. 2026, arXiv 2607.07508): critic updated K=2 times per actor update at 5x the actor's learning rate,
         token-level GAE over the actor's own tokens only (skip-observation), lambda 1 - 1/(1.5 T), Monte Carlo critic target,
         advantages whitened as verl's GAE does, direct double-sided importance sampling (tokens with pi/mu outside
         (1 - eps_l, 1 + eps_h) masked, the rest weighted by the detached ratio), token mean.
  bpco   BPCO (Qi, Zhou and Lee 2026, arXiv 2608.23566; QPHutu/golden_critic): DPPO binary-TV mask on the sampled token's probability
         change in the advantage's direction (pi - mu <= eps when A > 0, >= -eps otherwise, gradient through the ratio), critic
         bounded to the reward range by a scaled arctangent, Monte Carlo target, unnormalised advantages, lambda 1 - 1/(0.4 T),
         sequence mean of token sums; the critic may see what the policy cannot: the opponent's hand and next card from the replay
         (hidden), the simulator's no-play rollout from the decision state (sim), or the simulator's verdict on the recorded
         continuation (simreplay: a per-game constant that depends on the actor's own plays, so it is not a valid baseline).

References: bc (the warm-up behaviour clone every arm starts from, frozen), bc_online (maximum likelihood in the same pass), reinforce
(centred outcome, no ratio, no gate), and the flip option (outcomes flipped for a random half of the games, both sides: the noise
floor). Evaluation on held-out games (the final day, half B; the Q model is fit on half A) and prequentially on each next unseen day
before training on it: winner gap, agreement with winners' and losers' cards, KL to bc, entropy, mass on plays with support in the
data seen so far, and the Q-eval three ways: restricted to supported plays and renormalised, pessimistic (off-support mass scored at
the state's worst supported play), and direct (unrestricted; the estimate the REINFORCE collapse fooled before).
Usage: python -m train.singleTraj pack|prep|arms ... (see main)
"""
import argparse
import copy
import json
import math
import time
import zlib
from datetime import datetime,timezone
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train.corpusStates import load as load_shard
from train.counterfactual import cell_of
from train.decisionHead import PHASES,phase,relative
from train.feats import FEAT_DIM
from train.traceRl import HAND,N_CELLS,Head,Policy,fit_head,menu_q

RECIPES={
    'bc_online':{'kind':'bc'},
    'reinforce':{'kind':'reinforce'},
    'flash':{'kind':'flash','delta':3e-3},
    'flash_d5':{'kind':'flash','delta':5e-3},
    'flash_nogate':{'kind':'flash','delta':math.inf},
    'sao':{'kind':'sao','eps_l':0.3,'eps_h':5.0,'alpha':1.5,'K':2,'bounded':False,'whiten':True,'priv':()},
    'sao_sym':{'kind':'sao','eps_l':0.2,'eps_h':0.2,'alpha':1.5,'K':2,'bounded':False,'whiten':True,'priv':()},
    'bpco':{'kind':'bpco','eps':0.2,'alpha':0.4,'K':1,'bounded':True,'whiten':False,'priv':()},
    'bpco_hidden':{'kind':'bpco','eps':0.2,'alpha':0.4,'K':1,'bounded':True,'whiten':False,'priv':('hidden',)},
    'bpco_sim':{'kind':'bpco','eps':0.2,'alpha':0.4,'K':1,'bounded':True,'whiten':False,'priv':('sim',)},
    'bpco_both':{'kind':'bpco','eps':0.2,'alpha':0.4,'K':1,'bounded':True,'whiten':False,'priv':('hidden','sim')},
    'bpco_simreplay':{'kind':'bpco','eps':0.2,'alpha':0.4,'K':1,'bounded':True,'whiten':False,'priv':('simreplay',)},
}
MUS=('cross','same','warm','online')
SPLITS=('warm','stream','heldA','heldB')


def utc(day):
    return datetime.strptime(day,'%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp()


def pack(states,out,warm_end='2026-09-02',held_start='2026-09-06'):
    # the shards as one time-ordered store: games by battle time, each game's blue then red trajectory, decisions in order; X stays
    # float16 on disk (memory-mapped), the standardisation statistics come from the warm-up games only
    states=Path(states);out=Path(out);out.mkdir(parents=True,exist_ok=True);shards=sorted(states.glob('shard*.npz'))
    vocab=set();n=0;G=0
    for p in shards:
        d=np.load(p);side=json.loads(p.with_suffix('.json').read_text());ok={g['bid'] for g in side['outcomes'] if g['actual_winner'] is not None}
        keep=np.isin(d['bid'],list(ok));n+=int(keep.sum());G+=len(ok)
        for c in ('hand','opp_hand'):vocab|={x for h in set(d[c][keep].tolist()) for x in h.split('|') if x}
        vocab|=set(d['card'][keep].tolist())|set(d['opp_next'][keep].tolist())
    vocab=sorted(vocab-{''});idx={c:i for i,c in enumerate(vocab)};X=np.lib.format.open_memmap(out/'X.npy','w+',np.float16,(n,FEAT_DIM))
    rec={k:np.zeros(s,t) for k,s,t in (('H',(n,HAND),np.int16),('card',n,np.int16),('cell',n,np.int8),('team',n,np.int8),('t',n,np.float32),
         ('game',n,np.int32),('pos',n,np.int16),('traj',n,np.int32),('OH',(n,HAND),np.int16),('ON',n,np.int16),('roll',(n,4),np.float16),
         ('rel',(n,6),np.float16),('y',n,np.int8))}
    games={k:[] for k in ('bid','ts','winner','sim_v','sim_dc','sim_ds','rolled','split','rstart','tstart','mode')}
    traj={k:[] for k in ('start','len','game','team','R')};w_end=utc(warm_end);h_start=utc(held_start);r0=0;s1=None;s2=None;nw=0
    for p in shards:
        cols,Xs,roll,outs=load_shard(p);bids=cols['bid'].astype(str);order={}
        for b in sorted({b for b in set(bids.tolist()) if outs[b]['actual_winner'] is not None},key=lambda b:(outs[b]['ts'],b)):order[b]=len(order)
        sel=[i for i,b in enumerate(bids) if b in order]
        sel.sort(key=lambda i:(order[bids[i]],cols['team'][i]!='blue',int(cols['idx'][i])))
        sel=np.array(sel,dtype=np.int64);m=len(sel);Xk=Xs[sel];X[r0:r0+m]=Xk;rel=relative(Xk.astype(np.float32))
        rec['rel'][r0:r0+m]=rel;rec['roll'][r0:r0+m]=roll[sel];rec['t'][r0:r0+m]=cols['t'][sel]
        teams=cols['team'][sel];rec['team'][r0:r0+m]=(teams=='red')
        for j,i in enumerate(sel):
            r=r0+j;h=[x for x in str(cols['hand'][i]).split('|') if x in idx];c=str(cols['card'][i])
            menu=list(dict.fromkeys([c]+h))[:HAND];rec['H'][r]=[idx[x] for x in menu]+[-1]*(HAND-len(menu));rec['card'][r]=idx[c]
            rec['cell'][r]=cell_of(cols['x'][i],cols['y'][i]);oh=[idx[x] for x in str(cols['opp_hand'][i]).split('|') if x in idx][:HAND]
            rec['OH'][r]=oh+[-1]*(HAND-len(oh));rec['ON'][r]=idx.get(str(cols['opp_next'][i]),-1)
        gb=bids[sel];start=0
        while start<m:
            b=gb[start];end=start
            while end<m and gb[end]==b:end+=1
            o=outs[b];g=len(games['bid']);ts=o['ts']
            split=0 if ts<w_end else 1 if ts<h_start else 2+zlib.crc32(b.encode())%2
            sv=0.0 if o['sim_winner'] not in ('blue','red') else (1.0 if o['sim_winner']=='blue' else -1.0)
            games['bid'].append(b);games['ts'].append(ts);games['winner'].append(0 if o['actual_winner']=='blue' else 1);games['sim_v'].append(sv)
            games['sim_dc'].append(float((o['sim_bc'] or 0)-(o['sim_rc'] or 0)));games['sim_ds'].append(float(o['sim_blue']-o['sim_red']))
            games['rolled'].append(bool(o.get('rolled')));games['split'].append(split);games['rstart'].append(r0+start);games['tstart'].append(len(traj['start']))
            games['mode'].append(o.get('mode',''))
            k=start
            while k<end:
                tm=teams[k];e=k
                while e<end and teams[e]==tm:e+=1
                ti=len(traj['start']);won=float((o['actual_winner']==tm))
                traj['start'].append(r0+k);traj['len'].append(e-k);traj['game'].append(g);traj['team'].append(int(tm=='red'));traj['R'].append(won)
                rec['pos'][r0+k:r0+e]=np.arange(e-k);rec['traj'][r0+k:r0+e]=ti;rec['game'][r0+k:r0+e]=g;rec['y'][r0+k:r0+e]=won;k=e
            if split==0:
                S=np.concatenate([Xk[start:end].astype(np.float64),rel[start:end]],1);nw+=len(S)
                s1=S.sum(0) if s1 is None else s1+S.sum(0);s2=(S**2).sum(0) if s2 is None else s2+(S**2).sum(0)
            start=end
        r0+=m
    X.flush();games['rstart'].append(r0);games['tstart'].append(len(traj['start']))
    mu=s1/nw;sd=np.sqrt(np.maximum(s2/nw-mu**2,0))+1e-6
    np.savez(out/'rec.npz',**rec,**{'g_'+k:np.array(v) for k,v in games.items() if k not in ('bid','mode')},**{'t_'+k:np.array(v) for k,v in traj.items()},
             mu=mu.astype(np.float32),sd=sd.astype(np.float32))
    split=np.array(games['split']);meta={'vocab':vocab,'records':int(r0),'games':len(games['bid']),'trajectories':len(traj['start']),'warm_end':warm_end,
          'held_start':held_start,'split_games':{s:int((split==i).sum()) for i,s in enumerate(SPLITS)},'rolled_games':int(np.sum(games['rolled'])),
          'bid':games['bid'],'mode':games['mode'],'shards':len(shards)}
    (out/'meta.json').write_text(json.dumps(meta)+'\n')
    return {k:v for k,v in meta.items() if k not in ('bid','mode','vocab')}


class Corpus:
    # the packed store: X memory-mapped, per-record, per-trajectory and per-game arrays in memory
    def __init__(self,d):
        d=Path(d);self.dir=d;self.X=np.load(d/'X.npy',mmap_mode='r');a=np.load(d/'rec.npz');self.a={k:a[k] for k in a.files}
        self.meta=json.loads((d/'meta.json').read_text());self.vocab=self.meta['vocab'];self.n_card=len(self.vocab)
        self.mu=self.a['mu'];self.sd=self.a['sd'];self.n_state=len(self.mu);self.gr=self.a['g_rstart'];self.gt=self.a['g_tstart']
        self.split=self.a['g_split'];self.day=((self.a['g_ts']-utc('2026-01-01'))//86400).astype(int)
    def S(self,a,b,rows=None):
        x=self.X[a:b] if rows is None else self.X[rows];r=self.a['rel'][a:b] if rows is None else self.a['rel'][rows]
        return torch.from_numpy(((np.concatenate([x.astype(np.float32),r.astype(np.float32)],1)-self.mu)/self.sd).astype(np.float32))
    def games(self,split):
        return np.where(self.split==SPLITS.index(split))[0]
    def records(self,gs):
        return np.concatenate([np.arange(self.gr[g],self.gr[g+1]) for g in gs]) if len(gs) else np.zeros(0,np.int64)
    def priv(self,kinds,rows):
        # critic-only inputs: opponent hand and next card (multi-hot over the vocabulary), no-play rollout features, recorded-continuation
        # verdict for the actor (sign flipped for red)
        out=[];a=self.a
        for k in kinds:
            if k=='hidden':
                Z=np.zeros((len(rows),2*self.n_card),np.float32);oh=a['OH'][rows];on=a['ON'][rows];ri=np.arange(len(rows))
                for j in range(HAND):
                    ok=oh[:,j]>=0;Z[ri[ok],oh[ok,j]]=1
                ok=on>=0;Z[ri[ok],self.n_card+on[ok]]=1;out.append(Z)
            elif k=='sim':
                r=a['roll'][rows].astype(np.float32);out.append(np.stack([2*r[:,0],2*r[:,1],2*(r[:,0]-r[:,1]),r[:,2],r[:,3]/10],1))
            elif k=='simreplay':
                g=a['game'][rows];s=1-2*a['team'][rows].astype(np.float32)
                out.append(np.stack([s*a['g_sim_v'][g],s*a['g_sim_dc'][g]/3,2*s*a['g_sim_ds'][g]],1).astype(np.float32))
        return torch.from_numpy(np.concatenate(out,1)) if out else None
    def priv_dim(self,kinds):
        return sum({'hidden':2*self.n_card,'sim':5,'simreplay':3}[k] for k in kinds)


def batch(c,g0,g1):
    # one batch of consecutive games: records, their trajectory within the batch, token positions, rewards
    a,b=int(c.gr[g0]),int(c.gr[g1]);ta,tb=int(c.gt[g0]),int(c.gt[g1]);A=c.a
    return {'S':c.S(a,b),'H':torch.from_numpy(A['H'][a:b].astype(np.int64)),'card':torch.from_numpy(A['card'][a:b].astype(np.int64)),
            'cell':torch.from_numpy(A['cell'][a:b].astype(np.int64)),'tr':torch.from_numpy((A['traj'][a:b]-ta).astype(np.int64)),
            'pos':torch.from_numpy(A['pos'][a:b].astype(np.int64)),'L':torch.from_numpy(A['t_len'][ta:tb].astype(np.int64)),
            'R':torch.from_numpy(A['t_R'][ta:tb].astype(np.float32)),'n':tb-ta,'rows':(a,b),'trajs':(ta,tb),'Z':None}


def extend(c,base,priv=(),mu=None,flip=None):
    # an arm's view of a batch: its critic-only inputs, its behaviour estimate and, for the noise floor, its flipped outcomes
    (a,b),(ta,tb)=base['rows'],base['trajs'];out=dict(base)
    if priv:out['Z']=c.priv(priv,np.arange(a,b))
    if mu is not None:out['mu']=torch.from_numpy(np.asarray(mu[a:b],np.float32))
    if flip is not None:out['R']=torch.where(torch.from_numpy(flip[c.a['t_game'][ta:tb]]),1-base['R'],base['R'])
    return out


def token_logp(pol,S,H,card,cell,T=1.0,temps=None):
    # (n, 2) log-probabilities of the card token and the cell token given the card at temperature T, or a list of them for temps
    h=pol.trunk(S);logits=pol.card(h);mask=torch.zeros_like(logits,dtype=torch.bool);mask.scatter_(1,H.clamp(min=0),H>=0)
    cl=pol.cell(torch.cat([h,pol.emb(card)],1));out=[]
    for t in temps or (T,):
        lc=F.log_softmax((logits/t).masked_fill(~mask,-1e9),1).gather(1,card[:,None])[:,0]
        out.append(torch.stack([lc,F.log_softmax(cl/t,1).gather(1,cell[:,None])[:,0]],1))
    return out if temps else out[0]


class Critic(nn.Module):
    # value at the card token (state) and at the cell token (state and chosen card); bounded maps the head through the scaled
    # arctangent into the reward range (0, 1), unbounded is the usual linear head (offset to start at one half)
    def __init__(self,n_in,n_card,hidden=128,emb=16,bounded=True):
        super().__init__();self.bounded=bounded
        self.trunk=nn.Sequential(nn.Linear(n_in,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,hidden),nn.GELU())
        self.vc=nn.Linear(hidden,1);self.emb=nn.Embedding(n_card,emb);self.vk=nn.Linear(hidden+emb,1)
    def out(self,z):
        return 0.5+torch.atan(z)/math.pi if self.bounded else z+0.5
    def forward(self,S,card,Z=None):
        h=self.trunk(S if Z is None else torch.cat([S,Z],1))
        return torch.stack([self.out(self.vc(h)[:,0]),self.out(self.vk(torch.cat([h,self.emb(card)],1))[:,0])],1)


def la_lambda(T,alpha):
    # length-adaptive GAE parameter 1 - 1/(alpha T) of the token count, clamped to [0, 1] for very short trajectories
    return (1-1/(alpha*T.float())).clamp(0,1)


def gae(v,R,T,lam,gamma=1.0):
    # token-level GAE over the actor's own tokens: the value after a token is the actor's next token, so the opponent's plays in
    # between are skipped (the skip-observation estimator); terminal reward R, per-trajectory lambda; v is (n, Tmax) padded
    n,Tm=v.shape;adv=torch.zeros(n,Tm);last=torch.zeros(n);nxt=torch.zeros(n);z=torch.zeros(n)
    for t in range(Tm-1,-1,-1):
        ok=t<T;cur=torch.where(t==T-1,R,z)+gamma*nxt-v[:,t]+gamma*lam*last
        last=torch.where(ok,cur,last);nxt=torch.where(ok,v[:,t],nxt);adv[:,t]=torch.where(ok,cur,z)
    return adv


def to_mat(x,tr,pos,n,T):
    # (records, 2) token values into (trajectories, 2 * max decisions) with tokens card0, cell0, card1, ...
    m=torch.zeros(n,2*int(T.max()));m[tr,2*pos]=x[:,0];m[tr,2*pos+1]=x[:,1];return m


def from_mat(m,tr,pos):
    return torch.stack([m[tr,2*pos],m[tr,2*pos+1]],1)


def bern_kl(lmu,lp):
    p=lmu.exp().clamp(1e-6,1-1e-6);q=lp.exp().clamp(1e-6,1-1e-6)
    return p*(p.log()-q.log())+(1-p)*(torch.log1p(-p)-torch.log1p(-q))


def flash_loss(lp,lmu,A,tr,T,delta):
    # FlashREINFORCE on flattened tokens: lp, lmu (m,), A (n,) trajectory advantages, tr (m,) trajectory index, T (n,) token counts
    n=len(A)
    with torch.no_grad():
        D=(torch.zeros(n).index_add(0,tr,bern_kl(lmu,lp.detach()))/T).clamp(min=0);adm=D<=delta;act=adm[tr]
        w=torch.where(act,lp.detach()-lmu,torch.full_like(lp,-math.inf)).exp()*A[tr]
    return -(torch.zeros(n).index_add(0,tr,w*lp)/T).mean(),{'admitted':float(adm.float().mean()),'kl_proxy':float(D.mean())}


def sao_loss(lp,lmu,A,eps_l,eps_h):
    # SAO direct double-sided importance sampling on flattened tokens, A per token, token mean
    with torch.no_grad():
        r=(lp.detach()-lmu).exp();keep=(r>1-eps_l)&(r<1+eps_h);w=torch.where(keep,r,torch.zeros_like(r))*A
    return -(w*lp).mean(),{'masked':float(1-keep.float().mean())}


def dppo_loss(lp,lmu,A,tr,n,eps):
    # DPPO binary-TV (golden_critic compute_policy_loss_dppo_tv) with seq-mean-token-sum aggregation, A per token
    ratio=(lp-lmu).clamp(-20,20).exp()
    with torch.no_grad():
        dp=lp.detach().exp()-lmu.exp();valid=torch.where(A>0,dp<=eps,dp>=-eps).float()
    return torch.zeros(n).index_add(0,tr,-A*ratio*valid).mean(),{'masked':float(1-valid.mean())}


def parse(spec):
    # recipe[:mu=cross|same|warm|online][:T=temperature][:flip][:seed=k][:lr=x]
    parts=spec.split(':');cfg=dict(RECIPES[parts[0]]);cfg.update({'recipe':parts[0],'mu':'cross','T':1.0,'flip':False,'seed':0,'lr':None})
    for p in parts[1:]:
        k,_,v=p.partition('=')
        if k=='flip':cfg['flip']=True
        elif k=='mu':cfg['mu']=v
        elif k=='T':cfg['T']=float(v)
        elif k=='seed':cfg['seed']=int(v)
        elif k=='lr':cfg['lr']=float(v)
    return cfg


class Arm:
    # one recipe's actor (from the warm-up clone) and critic (pretrained on the warm-up games), updated once per batch
    def __init__(self,spec,init,critics,lr=1e-4,critic_mult=5.0):
        self.spec=spec;self.cfg=parse(spec);torch.manual_seed(self.cfg['seed']);self.pol=copy.deepcopy(init)
        for p in self.pol.parameters():p.requires_grad_(True)
        self.opt=torch.optim.Adam(self.pol.parameters(),lr=self.cfg['lr'] or lr);self.critic=None;self.stats={}
        if self.cfg['kind'] in ('sao','bpco'):
            self.critic=copy.deepcopy(critics[critic_key(self.cfg)]);self.copt=torch.optim.Adam(self.critic.parameters(),lr=(self.cfg['lr'] or lr)*critic_mult)
    def log(self,**kv):
        for k,v in kv.items():s=self.stats.setdefault(k,[0.0,0]);s[0]+=float(v);s[1]+=1
    def step(self,b):
        cfg=self.cfg;kind=cfg['kind'];tr,pos,T=b['tr'],b['pos'],2*b['L'];R=b['R'];n=b['n']
        lp=token_logp(self.pol,b['S'],b['H'],b['card'],b['cell'])
        if kind=='bc':
            loss=-lp.sum(1).mean()
        elif kind in ('reinforce','flash'):
            A=R-R.mean();tri=tr.repeat_interleave(2);lpf=lp.reshape(-1)
            if kind=='reinforce':loss=-(torch.zeros(n).index_add(0,tri,A[tri]*lpf)/T).mean()
            else:
                loss,st=flash_loss(lpf,b['mu'].reshape(-1),A,tri,T.float(),cfg['delta']);self.log(**st)
        else:
            with torch.no_grad():
                V=self.critic(b['S'],b['card'],b['Z']);vm=to_mat(V,tr,pos,n,b['L']);adv=from_mat(gae(vm,R,T,la_lambda(T,cfg['alpha'])),tr,pos)
                Rt=R[tr][:,None].expand(-1,2);ev=1-float((Rt-V).var()/(Rt.var()+1e-8));self.log(critic_ev=ev,adv_abs=adv.abs().mean())
                if cfg['whiten']:adv=(adv-adv.mean())/(adv.std()+1e-8)
            for _ in range(cfg['K']):
                cl=F.mse_loss(self.critic(b['S'],b['card'],b['Z']),Rt);self.copt.zero_grad();cl.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(),1.0);self.copt.step()
            if kind=='sao':loss,st=sao_loss(lp.reshape(-1),b['mu'].reshape(-1),adv.reshape(-1),cfg['eps_l'],cfg['eps_h'])
            else:loss,st=dppo_loss(lp.reshape(-1),b['mu'].reshape(-1),adv.reshape(-1),tr.repeat_interleave(2),n,cfg['eps'])
            self.log(**st)
        if 'mu' in b:
            with torch.no_grad():self.log(ratio=(lp-b['mu']).exp().mean(),abs_logratio=(lp-b['mu']).abs().mean())
        self.opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.pol.parameters(),1.0);self.opt.step()
    def take_stats(self):
        s={k:round(v[0]/max(v[1],1),5) for k,v in self.stats.items()};self.stats={};return s


def critic_key(cfg):
    return ('b' if cfg['bounded'] else 'u')+'|'+'+'.join(cfg['priv'])


def fit_bc(pol,c,rows,epochs=1,lr=1e-3,batch_size=4096,chunk=400000,seed=0,log=None):
    # maximum likelihood on the recorded plays of the given records, shuffled within contiguous chunks of the memory-mapped store
    torch.manual_seed(seed);rng=np.random.default_rng(seed);opt=torch.optim.AdamW(pol.parameters(),lr=lr,weight_decay=1e-5);A=c.a
    for e in range(epochs):
        blocks=[rows[i:i+chunk] for i in range(0,len(rows),chunk)];rng.shuffle(blocks)
        for bl in blocks:
            bl=np.sort(bl);S=c.S(0,0,bl);H=torch.from_numpy(A['H'][bl].astype(np.int64));card=torch.from_numpy(A['card'][bl].astype(np.int64))
            cell=torch.from_numpy(A['cell'][bl].astype(np.int64));perm=torch.randperm(len(bl))
            for i in range(0,len(bl),batch_size):
                j=perm[i:i+batch_size];loss=-pol.logp(S[j],H[j],card[j],cell[j]).mean();opt.zero_grad();loss.backward();opt.step()
        if log:log(f'bc epoch {e+1}/{epochs}')
    return pol


def fit_critic(cr,c,rows,priv,epochs=2,lr=1e-3,batch_size=4096,chunk=400000,seed=0):
    # Monte Carlo pretraining of a critic on the given records: both token values regress to the actor's recorded outcome
    torch.manual_seed(seed);rng=np.random.default_rng(seed);opt=torch.optim.Adam(cr.parameters(),lr=lr);A=c.a
    for _ in range(epochs):
        blocks=[rows[i:i+chunk] for i in range(0,len(rows),chunk)];rng.shuffle(blocks)
        for bl in blocks:
            bl=np.sort(bl);S=c.S(0,0,bl);card=torch.from_numpy(A['card'][bl].astype(np.int64));Z=c.priv(priv,bl) if priv else None
            y=torch.from_numpy(A['y'][bl].astype(np.float32));perm=torch.randperm(len(bl))
            for i in range(0,len(bl),batch_size):
                j=perm[i:i+batch_size];loss=F.mse_loss(cr(S[j],card[j],None if Z is None else Z[j]),y[j][:,None].expand(-1,2))
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(cr.parameters(),1.0);opt.step()
    return cr


def critic_ev(cr,c,rows,priv):
    # explained variance of the critic's state value against the recorded outcome on the given records
    parts=np.array_split(rows,max(1,len(rows)//50000))
    with torch.no_grad():V=torch.cat([cr(c.S(0,0,r),torch.from_numpy(c.a['card'][r].astype(np.int64)),c.priv(priv,r) if priv else None)[:,0] for r in parts])
    y=torch.from_numpy(c.a['y'][rows].astype(np.float32))
    return {'ev':round(1-float((y-V).var()/y.var()),4),'mse':round(float(((y-V)**2).mean()),4)}


def behaviour(c,pols,rows,temps=(1.0,),chunk=200000):
    # token log-probabilities of the recorded plays (the given records; zero elsewhere) under each behaviour estimate: pols maps a
    # name to a policy or, per record, to one of two policies chosen by a boolean array (the cross-fit halves)
    out={f'{k}|{T:g}':np.zeros((len(c.a['y']),2),np.float32) for k in pols for T in temps};A=c.a
    for i in range(0,len(rows),chunk):
        r=rows[i:i+chunk];S=c.S(0,0,r);H=torch.from_numpy(A['H'][r].astype(np.int64));card=torch.from_numpy(A['card'][r].astype(np.int64))
        cell=torch.from_numpy(A['cell'][r].astype(np.int64))
        with torch.no_grad():
            for k,p in pols.items():
                if isinstance(p,tuple):
                    a,b_,side=p;sel=torch.from_numpy(side[r])[:,None]
                    vs=[torch.where(sel,x,y) for x,y in zip(token_logp(b_,S,H,card,cell,temps=temps),token_logp(a,S,H,card,cell,temps=temps))]
                else:vs=token_logp(p,S,H,card,cell,temps=temps)
                for T,v in zip(temps,vs):out[f'{k}|{T:g}'][r]=v.numpy()
    return out


def eval_rows(c,gs,n,seed=0):
    rng=np.random.default_rng(seed);rows=c.records(gs)
    return np.sort(rng.choice(rows,min(n,len(rows)),replace=False)) if len(rows) else rows


def eval_set(c,rows):
    A=c.a
    return {'rows':rows,'S':c.S(0,0,rows),'H':torch.from_numpy(A['H'][rows].astype(np.int64)),'card':torch.from_numpy(A['card'][rows].astype(np.int64)),
            'cell':torch.from_numpy(A['cell'][rows].astype(np.int64)),'y':torch.from_numpy(A['y'][rows].astype(np.float32)),
            'phase':np.array([phase(float(t)) for t in A['t'][rows]])}


def support(c,rows,min_n=5):
    cnt=np.zeros((c.n_card,N_CELLS),np.int64);np.add.at(cnt,(c.a['card'][rows].astype(np.int64),c.a['cell'][rows].astype(np.int64)),1)
    return torch.from_numpy(cnt>=min_n)


def evaluate(pol,ref,Q,ev,sup,q_n=30000,chunk=20000):
    # held-out metrics of one policy against the reference (bc): see the module docstring; the Q-eval uses the first q_n records
    S,H,card,cell,y=ev['S'],ev['H'],ev['card'],ev['cell'],ev['y'];n=len(y);won=y>0.5;acc={k:[] for k in ('lp','kl','ent','top','mass')};qs=[]
    with torch.no_grad():
        for i in range(0,n,chunk):
            s,h=S[i:i+chunk],H[i:i+chunk];lp=token_logp(pol,s,h,card[i:i+chunk],cell[i:i+chunk]).sum(1);menu=pol.menu_logp(s,h);rm=ref.menu_logp(s,h)
            p=menu.exp();ok=sup[h.clamp(min=0)]&(h>=0)[:,:,None];mc=menu.clamp(min=-30)
            acc['lp'].append(lp);acc['kl'].append((p*(mc-rm.clamp(min=-30))).sum((1,2)));acc['ent'].append(-(p*mc).sum((1,2)))
            lc,_=pol.card_logp(s,h);acc['top'].append(lc.argmax(1));acc['mass'].append((p*ok).sum((1,2)))
            if i<q_n:
                m=min(chunk,q_n-i);q,okq=menu_q(Q,s[:m],h[:m],sup);pm=p[:m];qs.append((pm,okq,q,(h[:m]>=0)[:,:,None].expand(-1,-1,N_CELLS)))
        a={k:torch.cat(v) for k,v in acc.items()};lp=a['lp']
        pm=torch.cat([x[0] for x in qs]);okq=torch.cat([x[1] for x in qs]);q=torch.cat([x[2] for x in qs]);valid=torch.cat([x[3] for x in qs])
        mass=(pm*okq).sum((1,2));cov=mass>1e-6;qsup=((pm*okq*q).sum((1,2))[cov]/mass[cov]).mean()
        qmin=torch.where(okq,q,torch.full_like(q,2.0)).amin((1,2));qpess=((pm*okq*q).sum((1,2))+(1-mass)*qmin)[cov].mean()
        qdir=(pm*q*valid).sum((1,2)).mean()
    out={'logp':float(lp.mean()),'winner_gap':float(lp[won].mean()-lp[~won].mean()),'top1_won':float((a['top'][won]==card[won]).float().mean()),
         'top1_lost':float((a['top'][~won]==card[~won]).float().mean()),'kl_to_bc':float(a['kl'].mean()),'entropy':float(a['ent'].mean()),
         'support_mass':float(a['mass'].mean()),'q_support':float(qsup),'q_pess':float(qpess),'q_direct':float(qdir),'records':n,'q_records':int(len(pm))}
    for i,(_,_,label) in enumerate(PHASES):
        m=torch.from_numpy(ev['phase']==i)
        if (m&won).sum()>=50 and (m&~won).sum()>=50:out[f'winner_gap_{label}']=float(lp[m&won].mean()-lp[m&~won].mean())
    return {k:round(v,5) if isinstance(v,float) else v for k,v in out.items()}


def prep(pack_dir,out=None,bc_epochs=4,critic_epochs=2,hidden=128,q_epochs=8,eval_n=100000,day_n=20000,rolled_pre=8000,log=print,threads=0,fit_batch=4096):
    # everything the arms share: the warm-up clone (actor init and the 'warm' estimate), the same-data and cross-fit clones, the online
    # clone's pre-update probabilities, the held-out Q model, the pretrained critics, the evaluation rows and supports
    if threads:torch.set_num_threads(threads)
    c=Corpus(pack_dir);out=Path(out or Path(pack_dir)/'prep');out.mkdir(parents=True,exist_ok=True);t0=time.monotonic();A=c.a
    warm=c.games('warm');stream=c.games('stream');held=[c.games('heldA'),c.games('heldB')];rw=c.records(warm);rs=c.records(stream)
    say=lambda m:log(f'[prep {time.monotonic()-t0:.0f}s] {m}',flush=True)
    torch.manual_seed(0);bc=fit_bc(Policy(c.n_state,c.n_card,hidden),c,rw,bc_epochs,batch_size=fit_batch,log=say);bc.eval();torch.save(bc.state_dict(),out/'bc_warm.pt')
    half=np.array([zlib.crc32(b.encode())%2 for b in c.meta['bid']],bool)
    def tune(rows,seed):
        p=copy.deepcopy(bc);p.train();fit_bc(p,c,rows,1,batch_size=fit_batch,seed=seed);p.eval();return p
    same=tune(rs,1);ca=tune(c.records(stream[~half[stream]]),2);cb=tune(c.records(stream[half[stream]]),3);say('same and cross clones')
    for k,m in (('same',same),('cross_a',ca),('cross_b',cb)):torch.save(m.state_dict(),out/f'bc_{k}.pt')
    online=np.zeros((len(A['y']),2),np.float32);ob=copy.deepcopy(bc);ob.train();opt=torch.optim.Adam(ob.parameters(),lr=1e-4)
    for g0,g1 in batches(c,stream,64):
        bb=batch(c,g0,g1);a,b=bb['rows'];lp=token_logp(ob,bb['S'],bb['H'],bb['card'],bb['cell']);online[a:b]=lp.detach().numpy()
        loss=-lp.sum(1).mean();opt.zero_grad();loss.backward();opt.step()
    say('online clone')
    # the cross estimate of a record comes from the clone fit on the other half: records of half-1 games use the half-0 clone (ca)
    mu=behaviour(c,{'warm':bc,'same':same,'cross':(cb,ca,half[A['game']])},rs,temps=(1.0,0.8,1.25));mu['online|1']=online
    for k,v in mu.items():np.save(out/f"mu_{k.replace('|','_')}.npy",v)
    say('behaviour estimates')
    rq=c.records(held[0]);Q=Head(c.n_state,c.n_card,hidden=hidden)
    fit_head(Q,c.S(0,0,rq),torch.from_numpy(A['y'][rq].astype(np.float32)),torch.from_numpy(A['card'][rq].astype(np.int64)),
             torch.from_numpy(A['cell'][rq].astype(np.int64)),epochs=q_epochs,l2=1e-3,batch=fit_batch);Q.eval();torch.save(Q.state_dict(),out/'q.pt')
    rows={'heldB':eval_rows(c,held[1],eval_n)};ev=eval_set(c,rows['heldB'])
    with torch.no_grad():
        pq=torch.sigmoid(Q(ev['S'],ev['card'],ev['cell'])).clamp(1e-6,1-1e-6)
        qrep={'accuracy':round(float(((pq>0.5).float()==ev['y']).float().mean()),4),'log_loss':round(float(F.binary_cross_entropy(pq,ev['y'])),4),
              'constant_log_loss':round(float(F.binary_cross_entropy(torch.full_like(pq,float(ev['y'].mean())),ev['y'])),4),'fit_records':int(len(rq))}
    say(f'Q head {qrep}')
    days=sorted(set(c.day[stream].tolist()));sup={'final':support(c,np.concatenate([rw,rs]))}
    for d in days:
        rows[f'day{d}']=eval_rows(c,stream[c.day[stream]==d],day_n,seed=d);sup[f'day{d}']=support(c,np.concatenate([rw,c.records(stream[c.day[stream]<d])]))
    torch.save({'rows':rows,'sup':sup},out/'evals.pt')
    rolled=stream[A['g_rolled'][stream]];pre=rolled[:rolled_pre];crit={};cev={}
    for bounded,priv,src in ((False,(),'warm'),(True,(),'warm'),(True,('hidden',),'warm'),(True,('simreplay',),'warm'),
                             (True,(),'rolled'),(True,('hidden',),'rolled'),(True,('sim',),'rolled'),(True,('hidden','sim'),'rolled'),(True,('simreplay',),'rolled')):
        if src=='rolled' and not len(pre):continue
        key=('b' if bounded else 'u')+'|'+'+'.join(priv);name=f'{src}:{key}';torch.manual_seed(0)
        cr=Critic(c.n_state+c.priv_dim(priv),c.n_card,hidden,bounded=bounded)
        fit_critic(cr,c,rw if src=='warm' else c.records(pre),priv,critic_epochs,batch_size=fit_batch)
        crit[name]=cr.state_dict();cev[name]=critic_ev(cr,c,rows['heldB'],priv);say(f'critic {name} {cev[name]}')
    torch.save(crit,out/'critics.pt')
    rep={'meta':{k:v for k,v in c.meta.items() if k not in ('bid','mode','vocab')},'days':days,'q_head':qrep,'critic_pretrain_heldB':cev,
         'rolled_stream_games':int(len(rolled)),'rolled_pretrain_games':int(len(pre)),'settings':{'bc_epochs':bc_epochs,'critic_epochs':critic_epochs,
         'hidden':hidden,'q_epochs':q_epochs,'eval_n':eval_n,'day_n':day_n},'seconds':round(time.monotonic()-t0,1)}
    rep['bc_heldB']=evaluate(bc,bc,Q,ev,sup['final'])
    for k,m in (('same',same),('cross_a',ca),('cross_b',cb)):rep[f'{k}_heldB']=evaluate(m,bc,Q,ev,sup['final'])
    (out/'prep.json').write_text(json.dumps(rep,indent=1)+'\n');say('done')
    return rep


def batches(c,gs,size):
    # consecutive runs of the given (time-ordered) games, size games per batch
    gs=np.asarray(gs);cuts=[0]+[i for i in range(1,len(gs)) if gs[i]!=gs[i-1]+1]+[len(gs)];out=[]
    for a,b in zip(cuts[:-1],cuts[1:]):
        for i in range(a,b,size):out.append((int(gs[i]),int(gs[min(i+size,b)-1])+1))
    return out


def run_arms(pack_dir,specs,slice_='all',out=None,batch_games=64,lr=1e-4,hidden=128,threads=4,prequential=True,log=print,prep_dir=None):
    # train the given arms side by side on one pass over the chosen stream games (all, last:N, or rolled) and evaluate them
    torch.set_num_threads(threads);c=Corpus(pack_dir);P=Path(prep_dir or Path(pack_dir)/'prep');A=c.a;t0=time.monotonic()
    bc=Policy(c.n_state,c.n_card,hidden);bc.load_state_dict(torch.load(P/'bc_warm.pt'));bc.eval()
    for p in bc.parameters():p.requires_grad_(False)
    Q=Head(c.n_state,c.n_card,hidden=hidden);Q.load_state_dict(torch.load(P/'q.pt'));Q.eval();E=torch.load(P/'evals.pt',weights_only=False);sup=E['sup']
    evs={k:eval_set(c,r) for k,r in E['rows'].items()};stream=c.games('stream')
    if slice_=='rolled':gs=stream[A['g_rolled'][stream]];src='rolled';gs=gs[json.loads((P/'prep.json').read_text())['rolled_pretrain_games']:]
    elif slice_.startswith('last:'):gs=stream[-int(slice_[5:]):];src='warm'
    else:gs=stream;src='warm'
    critics={}
    for k,sd in torch.load(P/'critics.pt').items():
        s,key=k.split(':',1)
        if s!=src:continue
        priv=tuple(x for x in key[2:].split('+') if x);cr=Critic(c.n_state+c.priv_dim(priv),c.n_card,hidden,bounded=key[0]=='b')
        cr.load_state_dict(sd);critics[key]=cr
    arms=[Arm(s,bc,critics,lr) for s in specs];mus={};flips={}
    for a in arms:
        k=f"{a.cfg['mu']}_{a.cfg['T']:g}"
        if k not in mus:mus[k]=np.load(P/f'mu_{k}.npy',mmap_mode='r')
        if a.cfg['flip']:flips.setdefault(a.cfg['seed'],np.random.default_rng(100+a.cfg['seed']).random(len(c.split))<0.5)
    rep={'slice':slice_,'games':int(len(gs)),'records':int(sum(c.gr[g+1]-c.gr[g] for g in gs)),'batch_games':batch_games,'lr':lr,'arms':{},
         'prequential':{a.spec:{} for a in arms},'train_stats':{a.spec:{} for a in arms}}
    bl=batches(c,gs,batch_games);cur=None
    for nb,(g0,g1) in enumerate(bl):
        d=int(c.day[g0])
        if d!=cur:
            if cur is not None:
                for a in arms:rep['train_stats'][a.spec][f'day{cur}']=a.take_stats()
            if prequential and slice_=='all' and f'day{d}' in evs:
                for a in arms:
                    a.pol.eval();rep['prequential'][a.spec][f'day{d}']=evaluate(a.pol,bc,Q,evs[f'day{d}'],sup[f'day{d}'],q_n=10000);a.pol.train()
                gaps=' '.join(f"{a.spec}:{rep['prequential'][a.spec][f'day{d}']['winner_gap']}" for a in arms)
                log(f'[{time.monotonic()-t0:.0f}s] prequential day{d} {gaps}',flush=True)
            cur=d
        base=batch(c,g0,g1)
        for a in arms:
            cfg=a.cfg;a.step(extend(c,base,cfg.get('priv',()),mus[f"{cfg['mu']}_{cfg['T']:g}"],flips.get(cfg['seed']) if cfg['flip'] else None))
        if (nb+1)%500==0:log(f'[{time.monotonic()-t0:.0f}s] batch {nb+1}/{len(bl)}',flush=True)
    for a in arms:
        rep['train_stats'][a.spec][f'day{cur}']=a.take_stats();a.pol.eval();rep['arms'][a.spec]=evaluate(a.pol,bc,Q,evs['heldB'],sup['final'])
        if a.critic is not None:rep['arms'][a.spec]['critic_heldB']=critic_ev(a.critic,c,evs['heldB']['rows'],a.cfg['priv'])
    rep['bc']=evaluate(bc,bc,Q,evs['heldB'],sup['final']);rep['seconds']=round(time.monotonic()-t0,1)
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(rep,indent=1)+'\n')
    return rep


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('pack');a.add_argument('--states',required=True);a.add_argument('--out',required=True)
    a.add_argument('--warm_end',default='2026-09-02');a.add_argument('--held_start',default='2026-09-06')
    a=sp.add_parser('prep');a.add_argument('--pack',required=True);a.add_argument('--threads',type=int,default=16);a.add_argument('--bc_epochs',type=int,default=4)
    a.add_argument('--critic_epochs',type=int,default=2)
    a=sp.add_parser('arms');a.add_argument('--pack',required=True);a.add_argument('--arms',nargs='+',required=True);a.add_argument('--slice',default='all')
    a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=4);a.add_argument('--batch_games',type=int,default=64)
    a.add_argument('--lr',type=float,default=1e-4);a=ap.parse_args()
    if a.cmd=='pack':print(json.dumps(pack(a.states,a.out,a.warm_end,a.held_start),indent=1))
    elif a.cmd=='prep':prep(a.pack,threads=a.threads,bc_epochs=a.bc_epochs,critic_epochs=a.critic_epochs)
    else:
        r=run_arms(a.pack,a.arms,a.slice,a.out,a.batch_games,a.lr,threads=a.threads)
        keys=('winner_gap','top1_won','top1_lost','kl_to_bc','entropy','support_mass','q_support','q_pess','q_direct')
        print('arm',*keys);print('bc',*(r['bc'][k] for k in keys))
        for m,row in r['arms'].items():print(m,*(row[k] for k in keys))


if __name__=='__main__':main()
