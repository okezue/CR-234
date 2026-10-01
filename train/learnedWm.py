"""Learned world model from production traces as the RL environment: group-relative training inside a model fit on the replay
corpus, against the model-free single-trajectory recipes (train.singleTraj) and the hand-built simulator (simgroup).

The model works at decision resolution along one player's trajectory. A decision state (train.feats, the actor's view, standardised
as in train.singleTraj) is encoded to a latent; the actor's play (card, cell) is followed by the opponent's first play before the
actor's next decision (a vocabulary card, its cell and delay, or none), and the dynamics map latent, play and opponent play to the
latent of the actor's next decision. Heads read from a latent a summary of the state (time, elixir rate, elixir, crowns, the six
towers' health, unit counts by owner, lane and half, unit health by owner and half), the recorded outcome (V of the latent; Q of the
latent and the play, the one-step learned Q), the opponent's play and delay given the actor's play, the actor's next delay and
whether the game ends before it, and the actor's play given its hand (the behaviour in the latent). Training unrolls K decisions
along recorded trajectories with both sides' recorded plays (summary, opponent, timing and play losses at every step, outcome
losses on the encoded recorded state only), and each unrolled latent is pulled toward the encoding of the recorded next state (stop
gradient, as in EfficientZero), so rollouts stay where the heads were fit. Only warm-up and stream games are used. Ensemble members
differ in seed; one member is kept out of every arm for the exploitation check.

A rollout starts from a recorded state and a given play and runs H decisions: the actor's later plays come from the actor heads
(the behaviour) or, in an arm, from copies distilled from its current policy at every update, the opponent's from the opponent
heads; each play is sampled once from the members' mean distribution and shared by all members, so their spread is the models' own.
The game ends softly by the done probability (terminal value the one-step Q) and V bootstraps the rest; H = 0 is the one-step Q.
The hand cycles as in the game: a played card goes to the back of the queue and the queue's front enters, so the next K - 1 cards
to enter are the recorded ones whatever the actor plays.

Arms (side by side with the merged recipes in train.singleTraj.run_arms, one pass over the stream, 64 games per batch):
  wmgroup       at a quarter of each batch's recorded states, G = 8 plays sampled from the policy, each rolled out H = 4 decisions,
                advantages the value minus the group mean (scaled by their batch spread), BPCO's binary-TV mask against the frozen
                warm-up clone on both tokens (anchor=clip: the ppo1/simgroup clipped ratio against it).
  wmgroup_mopo  the same with the MOPO penalty (Yu et al. 2020): value minus lambda times the ensemble disagreement (the summed
                largest member distance to the mean predicted summary), lambda set on warm-up states so that the penalty's spread
                within groups equals the value's.
  qgroup        H = 0: the members' one-step Q, no dynamics.
  :flip         V and Q heads refit on outcomes flipped for a random half of the games (both sides): the noise floor.
Diagnostics: teacher-forced one- and multi-step prediction error on held-out games against persistence, opponent-play and timing
error against marginal baselines, the counterfactual direction (menu advantages against the held-out real-outcome Q model, and
policy-shift cosines as in train.worldModelBias, against the flipped-outcome floor) and exploitation (support of the favourite
plays; in-model value gain under the training members and under the held-out member against the held-out real-outcome gain).
Usage: python -m train.learnedWm aux|fit|refit|lam|arms|menu|diag ... (see main)
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from train.counterfactual import load_groups,own_cells,cell_of
from train.feats import FLAT_DIM,GAME_FEAT,MAX_UNITS,N_TOWERS,TOWER_FEAT,UNIT_FEAT
from train.singleTraj import Corpus,run_arms,simgroup,token_logp
from train.traceRl import HAND,N_CELLS,Head,Policy
from train.worldModelBias import agreement,shift

SUM=('t','erate','elixir_own','elixir_opp','crowns_own','crowns_opp','king_own','left_own','right_own','king_opp','left_opp','right_opp',
     'n_own_left_home','n_own_left_away','n_own_right_home','n_own_right_away','n_opp_left_home','n_opp_left_away','n_opp_right_home',
     'n_opp_right_away','hp_own_home','hp_own_away','hp_opp_home','hp_opp_away')
GROUPS={'time':(0,1),'elixir':(2,3),'crowns':(4,5),'towers':tuple(range(6,12)),'units':tuple(range(12,24))}
W={'dec':1.0,'cons':1.0,'v':1.0,'q':1.0,'oc':0.5,'ocell':0.5,'odt':0.2,'ndt':0.2,'done':0.5,'act':0.5}
ARMS={'wmgroup':{'H':4,'lam':0.0},'wmgroup_mopo':{'H':4,'lam':'auto'},'qgroup':{'H':0,'lam':0.0},'wmgroup_h1':{'H':1,'lam':0.0}}
MEMBERS=(0,1,2,3);HELD_OUT=4


def summary(X,red):
    # actor-relative summary of raw decision states (train.feats layout); red marks red actors, whose home half is y >= 16
    X=np.asarray(X,np.float32);n=len(X);red=np.asarray(red).astype(bool)[:,None]
    T=X[:,GAME_FEAT:GAME_FEAT+N_TOWERS*TOWER_FEAT].reshape(n,N_TOWERS,TOWER_FEAT);hp=T[:,:,2]*T[:,:,0];own=T[:,:,3]>0.5;cx=T[:,:,4]
    king=np.abs(cx-0.5)<0.1;left=cx<0.5
    tw=[(hp*((own==o)&m)).sum(1) for o in (True,False) for m in (king,~king&left,~king&~left)]
    U=X[:,FLAT_DIM:FLAT_DIM+2*MAX_UNITS*UNIT_FEAT].reshape(n,2*MAX_UNITS,UNIT_FEAT);occ=U[:,:,2]>0;mine=U[:,:,8]>0.5;ul=U[:,:,0]<0.5
    home=np.where(red,U[:,:,1]>=0.5,U[:,:,1]<0.5)
    cnt=[(occ&(mine==o)&(ul==lane)&(home==h)).sum(1)/10 for o in (True,False) for lane in (True,False) for h in (True,False)]
    hps=[(U[:,:,2]*(occ&(mine==o)&(home==h))).sum(1)/10 for o in (True,False) for h in (True,False)]
    return np.stack([X[:,0],X[:,7],X[:,8],X[:,9],X[:,10],X[:,11]]+tw+cnt+hps,1).astype(np.float32)


def aux(c,K=4,chunk=500000):
    # per record of the packed store: summary, the opponent's first play after the actor's play and before the actor's next decision
    # (vocabulary card or -1 for none, cell, delay), the count of opponent plays in that interval, the actor's next delay, the
    # last-decision flag, and the queue (the next K - 1 cards to enter the actor's hand); ties in time follow the replay's order
    A=c.a;n=len(A['y']);g=A['game'].astype(np.int64);t=A['t'].astype(np.float64);o=np.lexsort((A['idx'].astype(np.int64),g))
    go=g[o];tmo=A['team'][o].astype(np.int64);p=np.arange(n);nxt=[];pv={}
    for v in (0,1):
        pv[v]=p[tmo==v];k=np.searchsorted(pv[v],p,side='right')
        q=np.where(k<len(pv[v]),pv[v][np.minimum(k,max(len(pv[v])-1,0))],n) if len(pv[v]) else np.full(n,n)
        nxt.append(np.where((q<n)&(go[np.minimum(q,n-1)]==go),q,n))
    own=np.where(tmo==0,nxt[0],nxt[1]);opp=np.where(tmo==0,nxt[1],nxt[0]);has=opp<own
    bound=np.where(own<n,own,np.searchsorted(go,go,side='right'));cnt=np.zeros(n,np.int64)
    for v in (0,1):
        m=tmo!=v;cnt[m]=np.searchsorted(pv[v],bound[m])-np.searchsorted(pv[v],p[m],side='right')
    ro=o[np.minimum(opp,n-1)];oc=np.full(n,-1,np.int16);ocl=np.full(n,-1,np.int8);odt=np.zeros(n,np.float32);nop=np.zeros(n,np.int8)
    oc[o]=np.where(has,A['card'][ro],-1);ocl[o]=np.where(has,A['cell'][ro],-1);odt[o]=np.where(has,t[ro]-t[o],0);nop[o]=np.minimum(cnt,127)
    nself=np.full(n,-1,np.int64);nself[o]=np.where(own<n,o[np.minimum(own,n-1)],-1)
    tl=A['t_len'][A['traj']].astype(np.int64);pos=A['pos'].astype(np.int64);done=pos>=tl-1;r1=np.minimum(p+1,n-1)
    ndt=np.where(done,0,t[r1]-t).astype(np.float32);Hh=A['H'].astype(np.int64);Hn=Hh[r1]
    new=~((Hn[:,:,None]==Hh[:,None,:]).any(2)|(Hn<0));enter=np.where(new.any(1)&~done,Hn[p,new.argmax(1)],-1);qu=np.full((n,max(K-1,1)),-1,np.int16)
    for j in range(qu.shape[1]):qu[:,j]=np.where(pos+j<tl-1,enter[np.minimum(p+j,n-1)],-1)
    zs=np.zeros((n,len(SUM)),np.float32)
    for a in range(0,n,chunk):zs[a:a+chunk]=summary(c.X[a:a+chunk],A['team'][a:a+chunk])
    w=c.records(c.games('warm'));zm=zs[w].mean(0);zd=zs[w].std(0)+1e-3
    return {'zs':zs.astype(np.float16),'z_mu':zm.astype(np.float32),'z_sd':zd.astype(np.float32),'oc':oc,'ocl':ocl,'odt':odt,'nop':nop,'ndt':ndt,
            'done':done,'qu':qu,'next_self':nself,'K':np.array(K)}


def load_aux(path):
    a=np.load(path);return {k:a[k] for k in a.files}


def mlp(i,h,o):
    return nn.Sequential(nn.Linear(i,h),nn.GELU(),nn.LayerNorm(h),nn.Linear(h,o))


def hand_mask(H,n):
    # vocabulary mask of the hand; empty slots (-1) repeat the first card present, so they never mask a real card
    ok=H>=0;fb=H.gather(1,ok.float().argmax(1,keepdim=True)).clamp(min=0);m=torch.zeros(len(H),n,dtype=torch.bool)
    m.scatter_(1,torch.where(ok,H,fb),True);return m


class Actor(nn.Module):
    # a play from the latent and the hand: a card of the hand, then a cell given the card
    def __init__(self,latent,n_card,hidden,emb=16):
        super().__init__();self.n_card=n_card;self.emb=nn.Embedding(n_card,emb);self.body=nn.Sequential(nn.Linear(latent+emb,hidden),nn.GELU())
        self.card=nn.Linear(hidden,n_card);self.cell=nn.Linear(hidden+emb,N_CELLS)
    def card_logp(self,h,H):
        ok=(H>=0).float();he=(self.emb(H.clamp(min=0))*ok[...,None]).sum(1)/ok.sum(1,keepdim=True).clamp(min=1);z=self.body(torch.cat([h,he],1))
        return z,F.log_softmax(self.card(z).masked_fill(~hand_mask(H,self.n_card),-1e9),1)
    def logp(self,h,H,card,cell):
        z,lc=self.card_logp(h,H)
        return lc.gather(1,card[:,None])[:,0]+F.log_softmax(self.cell(torch.cat([z,self.emb(card)],1)),1).gather(1,cell[:,None])[:,0]
    def menu_logp(self,h,H):
        # (n, HAND, N_CELLS) like train.traceRl.Policy.menu_logp, -1e9 for empty slots
        z,lc=self.card_logp(h,H);out=torch.full((len(h),HAND,N_CELLS),-1e9)
        for j in range(H.shape[1]):
            c=H[:,j].clamp(min=0);v=lc.gather(1,c[:,None])+F.log_softmax(self.cell(torch.cat([z,self.emb(c)],1)),1)
            out[:,j]=torch.where((H[:,j]>=0)[:,None],v,torch.full_like(v,-1e9))
        return out


class WorldModel(nn.Module):
    # one ensemble member; card index n_card and cell index N_CELLS stand for no opponent play
    def __init__(self,n_state,n_card,n_sum=len(SUM),hidden=256,latent=128,emb=16):
        super().__init__();self.n_card=n_card;self.latent=latent;self.cfg={'n_state':n_state,'n_card':n_card,'n_sum':n_sum,'hidden':hidden,'latent':latent,'emb':emb}
        self.enc=nn.Sequential(nn.Linear(n_state,hidden),nn.GELU(),nn.LayerNorm(hidden),nn.Linear(hidden,latent))
        self.card=nn.Embedding(n_card+1,emb);self.cell=nn.Embedding(N_CELLS+1,emb);self.dyn=mlp(latent+4*emb,hidden,latent);self.dec=mlp(latent,hidden,n_sum)
        self.v=mlp(latent,hidden,1);self.q=mlp(latent+2*emb,hidden,1);self.ob=nn.Sequential(nn.Linear(latent+2*emb,hidden),nn.GELU())
        self.oc=nn.Linear(hidden,n_card+1);self.ocell=nn.Linear(hidden+emb,N_CELLS);self.odt=nn.Linear(hidden,1);self.tim=mlp(latent+4*emb,hidden,2)
        self.actor=Actor(latent,n_card,hidden,emb)
    def encode(self,S):
        return F.layer_norm(self.enc(S),(self.latent,))
    def play(self,card,cell):
        return torch.cat([self.card(card),self.cell(cell)],1)
    def step(self,h,a,o):
        return F.layer_norm(h+self.dyn(torch.cat([h,a,o],1)),(self.latent,))
    def opp(self,h,a):
        z=self.ob(torch.cat([h,a],1));return z,self.oc(z)
    def opp_cell(self,z,oc):
        return self.ocell(torch.cat([z,self.card(oc)],1))
    def qv(self,h,a):
        return self.q(torch.cat([h,a],1))[:,0]


def steps(c,ax,S_of,r,K):
    # tensors of the K + 1 decisions of each trajectory from store records r: S_of maps a record index array to standardised states
    A=c.a;tl=A['t_len'][A['traj'][r]];pos=A['pos'][r];j=np.arange(K+1)[:,None];ok=pos[None,:]+j<tl[None,:];R=np.where(ok,r[None,:]+j,r[None,:])
    T=lambda x,d=torch.long:torch.from_numpy(np.ascontiguousarray(x)).to(d)
    return {'S':S_of(R),'ok':T(ok,torch.bool),'y':T(A['y'][R],torch.float32),'card':T(A['card'][R]),'cell':T(A['cell'][R]),'H':T(A['H'][R]),
            'z':T((ax['zs'][R].astype(np.float32)-ax['z_mu'])/ax['z_sd'],torch.float32),'oc':T(ax['oc'][R]),'ocl':T(ax['ocl'][R]),
            'odt':T(ax['odt'][R],torch.float32),'ndt':T(ax['ndt'][R],torch.float32),'done':T(ax['done'][R],torch.float32)}


def unroll(m,b,K,w=W):
    # the K-step loss along recorded trajectories (b from steps): the weighted total and its parts
    ok=b['ok'];nc=m.n_card;L={k:[torch.zeros(()),torch.zeros(())] for k in w};h=m.encode(b['S'][0])
    with torch.no_grad():tgt=m.encode(b['S'][1:].flatten(0,1)).view(K,-1,m.latent) if K else None
    def add(k,v,mask):
        mask=mask.float();L[k][0]=L[k][0]+(v*mask).sum();L[k][1]=L[k][1]+mask.sum()
    bce=lambda x,y:F.binary_cross_entropy_with_logits(x,y,reduction='none')
    # the outcome heads see encoded recorded states only, so a rollout's value depends on the predicted state and not on which plays
    # led there (outcome labels on unrolled latents would teach them the recorded plays' association with the outcome)
    for j in range(K+1):
        mk=ok[j];add('dec',((m.dec(h)-b['z'][j])**2).mean(1),mk)
        if j==0:add('v',bce(m.v(h)[:,0],b['y'][j]),mk)
        if j==K:break
        card,cell=b['card'][j],b['cell'][j];a=m.play(card,cell);oc=b['oc'][j];has=oc>=0;ot=torch.where(has,oc,nc)
        o=m.play(ot,torch.where(has,b['ocl'][j].long(),N_CELLS))
        if j==0:add('q',bce(m.qv(h,a),b['y'][j]),mk)
        z,lo=m.opp(h,a);add('oc',F.cross_entropy(lo,ot,reduction='none'),mk)
        add('ocell',F.cross_entropy(m.opp_cell(z,ot),b['ocl'][j].long().clamp(min=0),reduction='none'),mk&has)
        add('odt',(m.odt(z)[:,0]-torch.log1p(b['odt'][j]))**2,mk&has);tm=m.tim(torch.cat([h,a,o],1));dn=b['done'][j]
        add('ndt',(tm[:,0]-torch.log1p(b['ndt'][j]))**2,mk&(dn<0.5));add('done',bce(tm[:,1],dn),mk);add('act',-m.actor.logp(h,b['H'][j],card,cell),mk)
        h=m.step(h,a,o);add('cons',((h-tgt[j])**2).mean(1),ok[j+1]);h=0.5*h+0.5*h.detach()
    parts={k:s/n.clamp(min=1) for k,(s,n) in L.items()}
    return sum(w[k]*v for k,v in parts.items()),{k:float(v.detach()) for k,v in parts.items()}


def train_rows(c):
    # start records of the warm-up and stream games only
    return c.records(np.concatenate([c.games('warm'),c.games('stream')]))


def fit(c,ax,rows,seed=0,K=4,epochs=1,hidden=256,latent=128,lr=1e-3,batch_size=1024,chunk=200000,log=print,max_steps=0):
    # one member on the given start records, contiguous blocks of the memory-mapped store shuffled per epoch, cosine learning rate
    torch.manual_seed(seed);rng=np.random.default_rng(seed);m=WorldModel(c.n_state,c.n_card,len(SUM),hidden,latent);rows=np.sort(np.asarray(rows))
    opt=torch.optim.AdamW(m.parameters(),lr=lr,weight_decay=1e-5);total=max_steps or epochs*math.ceil(len(rows)/batch_size);n=len(c.a['y'])
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:0.1+0.45*(1+math.cos(math.pi*min(s/total,1.0))));st=0;t0=time.monotonic();agg={};hist=[]
    for e in range(epochs):
        blocks=[rows[i:i+chunk] for i in range(0,len(rows),chunk)];rng.shuffle(blocks)
        for bl in blocks:
            lo=int(bl[0]);S=c.S(lo,min(int(bl[-1])+K+1,n));perm=rng.permutation(len(bl))
            for i in range(0,len(bl),batch_size):
                b=steps(c,ax,lambda R:S[torch.from_numpy(R-lo)],bl[perm[i:i+batch_size]],K);loss,parts=unroll(m,b,K)
                opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step();sched.step();st+=1
                for k,v in parts.items():agg[k]=agg.get(k,0.0)+v
                if st%500==0 or st==total:
                    nb=st-500*((st-1)//500);row={'step':st,'seconds':round(time.monotonic()-t0,1),**{k:round(v/nb,4) for k,v in agg.items()}}
                    hist.append(row);agg={};log(json.dumps(row),flush=True)
                if st>=total:return m,hist
    if agg:
        nb=st-500*((st-1)//500);hist.append({'step':st,'seconds':round(time.monotonic()-t0,1),**{k:round(v/nb,4) for k,v in agg.items()}})
    return m,hist


def save_member(m,path,**extra):
    torch.save({'state':m.state_dict(),'cfg':m.cfg,**extra},path)


def load_member(path):
    d=torch.load(path,weights_only=False);cfg=d['cfg'];m=WorldModel(cfg['n_state'],cfg['n_card'],cfg['n_sum'],cfg['hidden'],cfg['latent'],cfg['emb'])
    m.load_state_dict(d['state']);m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m


def flips(c,seed):
    # the games whose outcome train.singleTraj.run_arms flips for a flip arm of this seed
    return np.random.default_rng(100+seed).random(len(c.split))<0.5


def refit_outcome(m,c,rows,flip,seed=0,epochs=1,lr=1e-3,batch_size=4096,chunk=400000):
    # V and Q heads reinitialised and refit on the encoded states of the given records with the outcomes of the flip games flipped;
    # the encoder, dynamics and other heads stay as trained
    m=copy.deepcopy(m);torch.manual_seed(seed);rng=np.random.default_rng(seed);A=c.a;y=A['y'].astype(np.float32)
    for head in (m.v,m.q):
        for layer in head:
            if hasattr(layer,'reset_parameters'):layer.reset_parameters()
    ps=list(m.v.parameters())+list(m.q.parameters())
    for p in ps:p.requires_grad_(True)
    opt=torch.optim.Adam(ps,lr=lr);rows=np.sort(np.asarray(rows))
    for _ in range(epochs):
        blocks=[rows[i:i+chunk] for i in range(0,len(rows),chunk)];rng.shuffle(blocks)
        for bl in blocks:
            with torch.no_grad():
                h=m.encode(c.S(0,0,bl));a=m.play(torch.from_numpy(A['card'][bl].astype(np.int64)),torch.from_numpy(A['cell'][bl].astype(np.int64)))
            yy=torch.from_numpy(np.where(flip[A['game'][bl]],1-y[bl],y[bl]));perm=torch.from_numpy(rng.permutation(len(bl)))
            for i in range(0,len(bl),batch_size):
                j=perm[i:i+batch_size];loss=F.binary_cross_entropy_with_logits(m.v(h[j])[:,0],yy[j])+F.binary_cross_entropy_with_logits(m.qv(h[j],a[j]),yy[j])
                opt.zero_grad();loss.backward();opt.step()
    m.eval()
    for p in m.parameters():p.requires_grad_(False)
    return m


def sample(menu,H,G,gen=None):
    # G plays per state from (n, HAND, N_CELLS) log-probabilities: (n, G) hand cards and cells
    i=torch.multinomial(menu.flatten(1).exp()+1e-12,G,replacement=True,generator=gen);return H.gather(1,i//N_CELLS),i%N_CELLS


def rollout(models,hs,hand,queue,card,cell,H,actors=None,gen=None):
    # (M, n) values of the plays (card, cell) at root latents hs (one per member) after H decisions of the models, and (n,) the
    # disagreement: summed over steps, the largest member distance to the mean predicted summary over the root of its size
    M=len(models);n=len(card);u=torch.zeros(n)
    if H==0:return torch.stack([torch.sigmoid(m.qv(h,m.play(card,cell))) for m,h in zip(models,hs)]),u
    ret=torch.zeros(M,n);alive=torch.ones(M,n);hand=hand.clone();hs=list(hs);ar=torch.arange(n);nc=models[0].n_card;acts=actors or [None]*M
    for k in range(H):
        if k:
            menu=torch.stack([(a if a is not None else m.actor).menu_logp(h,hand) for m,h,a in zip(models,hs,acts)]).exp().mean(0)
            i=torch.multinomial(menu.flatten(1)+1e-12,1,generator=gen)[:,0];card=hand[ar,i//N_CELLS].clamp(min=0);cell=i%N_CELLS
        a=[m.play(card,cell) for m in models];zo=[m.opp(h,x) for m,h,x in zip(models,hs,a)]
        oc=torch.multinomial(torch.stack([F.softmax(lg,1) for _,lg in zo]).mean(0),1,generator=gen)[:,0]
        ocl=torch.multinomial(torch.stack([F.softmax(m.opp_cell(z,oc),1) for m,(z,_) in zip(models,zo)]).mean(0),1,generator=gen)[:,0]
        ocl=torch.where(oc==nc,N_CELLS,ocl);zp=[]
        for i_,(m,h,x) in enumerate(zip(models,hs,a)):
            o=m.play(oc,ocl);d=torch.sigmoid(m.tim(torch.cat([h,x,o],1))[:,1]);ret[i_]+=alive[i_]*d*torch.sigmoid(m.qv(h,x));alive[i_]=alive[i_]*(1-d)
            hs[i_]=m.step(h,x,o);zp.append(m.dec(hs[i_]))
        zp=torch.stack(zp);u=u+(zp-zp.mean(0)).norm(dim=2).max(0).values/math.sqrt(zp.shape[2])
        if k<H-1:
            hit=hand==card[:,None];hr=hit.any(1);nx=queue[:,k] if k<queue.shape[1] else torch.full((n,),-1,dtype=hand.dtype)
            hand[ar[hr],hit.float().argmax(1)[hr]]=nx[hr]
    return ret+alive*torch.stack([torch.sigmoid(m.v(h)[:,0]) for m,h in zip(models,hs)]),u


def values(models,S,H_,queue,card,cell,H,n_roll=1,actors=None,gen=None,chunk=4096):
    # mean over n_roll rollouts of the (M, n) values and (n,) disagreement of plays at standardised states S (one play per row)
    vs=[];us=[]
    with torch.no_grad():
        for i in range(0,len(S),chunk):
            s=slice(i,i+chunk);hs=[m.encode(S[s]) for m in models];acc=0;au=0
            for _ in range(n_roll):
                v,u=rollout(models,hs,H_[s],queue[s],card[s],cell[s],H,actors,gen);acc=acc+v;au=au+u
            vs.append(acc/n_roll);us.append(au/n_roll)
    return torch.cat(vs,1),torch.cat(us)


def parse_wm(spec):
    # recipe[:H=k][:G=k][:lam=x][:roots=x][:eps=x][:anchor=tv|clip][:cont=pi|mu][:flip][:seed=k][:lr=x]
    parts=spec.split(':');cfg={'recipe':parts[0],'kind':'wm','G':8,'roots':0.25,'eps':0.2,'anchor':'tv','cont':'pi','mu':'cross','T':1.0,'flip':False,
                               'seed':0,'lr':None,'priv':(),**ARMS[parts[0]]}
    for p in parts[1:]:
        k,_,v=p.partition('=')
        if k=='flip':cfg['flip']=True
        elif k in ('H','G','seed'):cfg[k]=int(v)
        elif k in ('roots','eps','lr'):cfg[k]=float(v)
        elif k=='lam':cfg[k]=v if v=='auto' else float(v)
        else:cfg[k]=v
    return cfg


class WmArm:
    # an arm trained inside the learned model; the run loop of train.singleTraj drives it like its own recipes
    def __init__(self,spec,init,models,queue,lam=0.0,lr=1e-4):
        self.spec=spec;self.cfg=cfg=parse_wm(spec);torch.manual_seed(cfg['seed']);self.gen=torch.Generator().manual_seed(cfg['seed'])
        self.bc=init;self.pol=copy.deepcopy(init);self.models=models;self.queue=queue;self.critic=None;self.stats={}
        for p in self.pol.parameters():p.requires_grad_(True)
        self.opt=torch.optim.Adam(self.pol.parameters(),lr=cfg['lr'] or lr);self.lam=lam if cfg['lam']=='auto' else cfg['lam']
        self.actors=[copy.deepcopy(m.actor) for m in models] if cfg['cont']=='pi' and cfg['H']>1 else None
        if self.actors:
            ps=[p for a in self.actors for p in a.parameters()]
            for p in ps:p.requires_grad_(True)
            self.aopt=torch.optim.Adam(ps,lr=1e-3)
    def log(self,**kv):
        for k,v in kv.items():s=self.stats.setdefault(k,[0.0,0]);s[0]+=float(v.detach() if torch.is_tensor(v) else v);s[1]+=1
    def take_stats(self):
        s={k:round(v[0]/max(v[1],1),5) for k,v in self.stats.items()};self.stats={};return s
    def advantages(self,S,H,rows):
        # G plays per state from the policy, their values in the models and the group-relative advantages (penalised for MOPO)
        cfg=self.cfg;G=cfg['G']
        with torch.no_grad():
            menu=self.pol.menu_logp(S,H);card,cell=sample(menu,H,G,self.gen);rep=lambda x:x.repeat_interleave(G,0)
            qu=torch.from_numpy(self.queue[rows].astype(np.int64))
            v,u=values(self.models,rep(S),rep(H),rep(qu),card.reshape(-1),cell.reshape(-1),cfg['H'],1,self.actors,self.gen)
            Q=v.mean(0).view(-1,G);U=u.view(-1,G);Qt=Q-self.lam*U;A=Qt-Qt.mean(1,keepdim=True)
            self.log(adv_abs=A.abs().mean(),q_spread=Q.std(1).mean(),disagree=U.mean(),value=Q.mean())
            A=A/(A.std()+1e-8)
        return menu,card,cell,A
    def step(self,b):
        cfg=self.cfg;n=len(b['S']);k=max(1,int(round(cfg['roots']*n)));sel=torch.randperm(n,generator=self.gen)[:k].sort().values
        S=b['S'][sel];H=b['H'][sel];rows=np.arange(*b['rows'])[sel.numpy()];G=cfg['G'];menu,card,cell,A=self.advantages(S,H,rows)
        Sg=S.repeat_interleave(G,0);Hg=H.repeat_interleave(G,0);cg=card.reshape(-1);xg=cell.reshape(-1);Ag=A.reshape(-1)
        lp=token_logp(self.pol,Sg,Hg,cg,xg)
        with torch.no_grad():lb=token_logp(self.bc,Sg,Hg,cg,xg)
        if cfg['anchor']=='clip':
            r=(lp.sum(1)-lb.sum(1)).exp();loss=-torch.min(r*Ag,r.clamp(1-cfg['eps'],1+cfg['eps'])*Ag).mean();masked=((r-1).abs()>cfg['eps']).float().mean()
        else:
            dp=lp.detach().exp()-lb.exp();valid=torch.where(Ag[:,None]>0,dp<=cfg['eps'],dp>=-cfg['eps']).float()
            loss=-(Ag[:,None]*(lp-lp.detach()).exp()*valid).sum(1).mean();masked=1-valid.mean()
        self.opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(self.pol.parameters(),1.0);self.opt.step();self.log(masked=masked)
        if self.actors:
            with torch.no_grad():hs=[m.encode(S) for m in self.models];tgt=menu.exp()
            dl=sum(-(tgt*a.menu_logp(h,H).clamp(min=-30)).sum((1,2)).mean() for a,h in zip(self.actors,hs))/len(self.actors)
            self.aopt.zero_grad();dl.backward();self.aopt.step();self.log(distill=dl)


def make_wm(wm_dir,members=MEMBERS):
    # arm factory for train.singleTraj.run_arms: a WmArm for the recipes of ARMS, None for the merged ones
    wm_dir=Path(wm_dir);queue=load_aux(wm_dir/'aux.npz')['qu'];cache={};lam_p=wm_dir/'lam.json'
    def get(flip,seed):
        if (flip,seed) not in cache:cache[(flip,seed)]=[load_member(wm_dir/(f'member{i}_flip{seed}.pt' if flip else f'member{i}.pt')) for i in members]
        return cache[(flip,seed)]
    def make(spec,bc,lr):
        if spec.split(':')[0] not in ARMS:return None
        cfg=parse_wm(spec);lam=0.0
        if cfg['lam']=='auto':lam=json.loads(lam_p.read_text())['lam']
        return WmArm(spec,bc,get(cfg['flip'],cfg['seed']),queue,lam,lr)
    return make


def calibrate(c,ax,models,bc,rows,G=8,H=4,seed=0,chunk=500):
    # the MOPO weight on warm-up states: the within-group spread of the values over that of the disagreement, plays from the clone
    gen=torch.Generator().manual_seed(seed);qs=[];us=[];A=c.a
    for i in range(0,len(rows),chunk):
        r=rows[i:i+chunk];S=c.S(0,0,r);Hh=torch.from_numpy(A['H'][r].astype(np.int64));qu=torch.from_numpy(ax['qu'][r].astype(np.int64))
        with torch.no_grad():card,cell=sample(bc.menu_logp(S,Hh),Hh,G,gen)
        rep=lambda x:x.repeat_interleave(G,0);v,u=values(models,rep(S),rep(Hh),rep(qu),card.reshape(-1),cell.reshape(-1),H,1,None,gen)
        qs.append(v.mean(0).view(-1,G));us.append(u.view(-1,G))
    q=torch.cat(qs);u=torch.cat(us);dq=q-q.mean(1,keepdim=True);du=u-u.mean(1,keepdim=True)
    return {'lam':float(dq.std()/(du.std()+1e-12)),'value_within_sd':float(dq.std()),'disagree_within_sd':float(du.std()),'disagree_mean':float(u.mean()),
            'states':int(len(rows)),'G':G,'H':H}


def tensors(c,rows):
    A=c.a;T=lambda k:torch.from_numpy(A[k][rows].astype(np.int64))
    return c.S(0,0,rows),T('H'),T('card'),T('cell'),torch.from_numpy(A['y'][rows].astype(np.float32))


def predict(c,ax,models,rows,K=4,chunk=4096):
    # teacher-forced unroll from the given start records: summary error by group and step (each member, the ensemble mean,
    # persistence of the start's summary), opponent play and delays against marginal baselines from the training records, done and
    # outcome log loss against constants
    A=c.a;tr=train_rows(c);ocT=np.where(ax['oc'][tr]>=0,ax['oc'][tr],c.n_card).astype(np.int64)
    freq=np.bincount(ocT,minlength=c.n_card+1)+1.0;freq=freq/freq.sum();cf=np.bincount(ax['ocl'][tr][ax['oc'][tr]>=0].astype(np.int64),minlength=N_CELLS)+1.0
    cf=cf/cf.sum();has=ax['oc'][tr]>=0;odt_med=float(np.median(ax['odt'][tr][has]));nd=~ax['done'][tr];ndt_med=float(np.median(ax['ndt'][tr][nd]))
    done_rate=float(ax['done'][tr].mean());y_rate=float(A['y'][tr].mean());acc={}
    def add(k,v,mask=None):
        v=v if mask is None else v[mask];s=acc.setdefault(k,[0.0,0]);s[0]+=float(v.sum());s[1]+=int(v.numel())
    zsd=torch.from_numpy(ax['z_sd'])
    for i in range(0,len(rows),chunk):
        r=rows[i:i+chunk];uq=np.unique(np.minimum(r[None,:]+np.arange(K+1)[:,None],len(A['y'])-1));Su=c.S(0,0,uq)
        b=steps(c,ax,lambda X:Su[torch.from_numpy(np.searchsorted(uq,X))],r,K);ok=b['ok']
        with torch.no_grad():
            hs=[m.encode(b['S'][0]) for m in models];nc=c.n_card
            for k in range(K+1):
                mk=ok[k];zk=b['z'][k];preds=torch.stack([m.dec(h) for m,h in zip(models,hs)]);ens=preds.mean(0)
                for gname,ix in GROUPS.items():
                    ix=list(ix);add(f'sum_{gname}_ens_k{k}',((ens[:,ix]-zk[:,ix])**2).mean(1),mk);add(f'sum_{gname}_persist_k{k}',((b['z'][0][:,ix]-zk[:,ix])**2).mean(1),mk)
                    add(f'sum_{gname}_member_k{k}',((preds[:,:,ix]-zk[None,:,ix])**2).mean(2).mean(0),mk)
                    add(f'mae_{gname}_ens_k{k}',((ens[:,ix]-zk[:,ix]).abs()*zsd[ix]).mean(1),mk);add(f'mae_{gname}_persist_k{k}',((b['z'][0][:,ix]-zk[:,ix]).abs()*zsd[ix]).mean(1),mk)
                pv=torch.stack([torch.sigmoid(m.v(h)[:,0]) for m,h in zip(models,hs)]).mean(0).clamp(1e-6,1-1e-6);y=b['y'][k]
                add(f'v_logloss_k{k}',-(y*pv.log()+(1-y)*(1-pv).log()),mk);add(f'v_acc_k{k}',((pv>0.5).float()==y).float(),mk)
                pe=torch.stack([torch.sigmoid(m.v(m.encode(b['S'][k]))[:,0]) for m in models]).mean(0).clamp(1e-6,1-1e-6)
                add(f'v_encoded_logloss_k{k}',-(y*pe.log()+(1-y)*(1-pe).log()),mk);add(f'const_logloss_k{k}',-(y*math.log(y_rate)+(1-y)*math.log(1-y_rate)),mk)
                if k==K:break
                card,cell=b['card'][k],b['cell'][k];oc=b['oc'][k];hv=oc>=0;ot=torch.where(hv,oc,nc);ocl=torch.where(hv,b['ocl'][k].long(),N_CELLS)
                pq=torch.stack([torch.sigmoid(m.qv(h,m.play(card,cell))) for m,h in zip(models,hs)]).mean(0).clamp(1e-6,1-1e-6)
                add(f'q_logloss_k{k}',-(y*pq.log()+(1-y)*(1-pq).log()),mk)
                po=[];pc=[];od=[];tm=[]
                for m,h in zip(models,hs):
                    a=m.play(card,cell);z,lg=m.opp(h,a);po.append(F.softmax(lg,1));pc.append(F.softmax(m.opp_cell(z,ot),1));od.append(m.odt(z)[:,0])
                    tm.append(m.tim(torch.cat([h,a,m.play(ot,ocl)],1)))
                po=torch.stack(po).mean(0);pc=torch.stack(pc).mean(0);od=torch.stack(od).mean(0);tm=torch.stack(tm).mean(0)
                add(f'opp_nll_k{k}',-po.gather(1,ot[:,None])[:,0].clamp(min=1e-9).log(),mk);add(f'opp_nll_marg_k{k}',-torch.from_numpy(np.log(freq))[ot].float(),mk)
                add(f'opp_top1_k{k}',(po.argmax(1)==ot).float(),mk);add(f'opp_top1_marg_k{k}',(ot==int(freq.argmax())).float(),mk)
                add(f'opp_top5_k{k}',(po.topk(5,1).indices==ot[:,None]).any(1).float(),mk)
                add(f'oppcell_nll_k{k}',-pc.gather(1,ocl.clamp(max=N_CELLS-1)[:,None])[:,0].clamp(min=1e-9).log(),mk&hv)
                add(f'oppcell_nll_marg_k{k}',-torch.from_numpy(np.log(cf))[ocl.clamp(max=N_CELLS-1)].float(),mk&hv)
                add(f'odt_mae_k{k}',(torch.expm1(od)-b['odt'][k]).abs(),mk&hv);add(f'odt_mae_median_k{k}',(odt_med-b['odt'][k]).abs(),mk&hv)
                dn=b['done'][k];nd_=mk&(dn<0.5);add(f'ndt_mae_k{k}',(torch.expm1(tm[:,0])-b['ndt'][k]).abs(),nd_);add(f'ndt_mae_median_k{k}',(ndt_med-b['ndt'][k]).abs(),nd_)
                pd=torch.sigmoid(tm[:,1]).clamp(1e-6,1-1e-6);add(f'done_logloss_k{k}',-(dn*pd.log()+(1-dn)*(1-pd).log()),mk)
                add(f'done_const_k{k}',-(dn*math.log(done_rate)+(1-dn)*math.log(1-done_rate)),mk)
                hs=[m.step(h,m.play(card,cell),m.play(ot,ocl)) for m,h in zip(models,hs)]
    return {k:round(s/max(n,1),5) for k,(s,n) in acc.items()}


def menus(c,rows,seed=0):
    # the recorded play and three alternatives per record, as train.counterfactual.alternatives builds them: other hand cards at the
    # recorded cell, then the recorded card at other cells on the actor's side (any cell for spells)
    from sim.cards import card as card_def,key
    A=c.a;rng=np.random.default_rng(seed);spell=np.zeros(c.n_card,bool)
    for i,name in enumerate(c.vocab):
        try:spell[i]=card_def(key(name) or name)['kind']=='spell'
        except (KeyError,TypeError):pass
    cards=np.zeros((len(rows),4),np.int64);cells=np.zeros((len(rows),4),np.int64)
    for j,r in enumerate(rows):
        cd,cl=int(A['card'][r]),int(A['cell'][r]);h=[int(x) for x in A['H'][r] if x>=0 and x!=cd];rng.shuffle(h);alts=[(x,cl) for x in h[:3]]
        pool=[k for k in (range(N_CELLS) if spell[cd] else own_cells('red' if A['team'][r] else 'blue')) if k!=cl]
        while len(alts)<3:alts.append((cd,int(rng.choice(pool))))
        cards[j]=[cd]+[a for a,_ in alts];cells[j]=[cl]+[b for _,b in alts]
    return cards,cells


def menu_adv(models,S,H_,qu,cards,cells,H,n_roll,gen):
    # (n, 4) values of each menu play minus the menu mean, ensemble mean
    n,k=cards.shape;rep=lambda x:x.repeat_interleave(k,0);v,_=values(models,rep(S),rep(H_),rep(qu),cards.reshape(-1),cells.reshape(-1),H,n_roll,None,gen)
    v=v.mean(0).view(n,k);return v-v.mean(1,keepdim=True)


def cos_rows(a,b):
    na=a.norm(dim=1);nb=b.norm(dim=1);ok=(na>1e-9)&(nb>1e-9);return float(((a*b).sum(1)[ok]/(na[ok]*nb[ok])).mean())


def corr(a,b):
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64);return float(np.corrcoef(a,b)[0,1]) if a.std()>0 and b.std()>0 else 0.0


def menu_values(c,ax,models,cf,H=4,n_roll=8,seed=0):
    # the learned model's value of every play of the simulator's counterfactual menus (train.counterfactual groups), as a cf file whose
    # ret is the model's value, so train.singleTraj.simgroup trains on it exactly as on the simulator's returns
    side=json.loads(Path(cf).read_text());gid={b:g for g,b in enumerate(c.meta['bid'])};A=c.a;vi={x:i for i,x in enumerate(c.vocab)};where={};keep=[]
    for g in {gid[str(r[0])] for r in side['rows'] if str(r[0]) in gid}:
        for r in range(int(c.gr[g]),int(c.gr[g+1])):where[(c.meta['bid'][g],int(A['idx'][r]))]=r
    for k,row in enumerate(side['rows']):
        r=where.get((str(row[0]),int(row[1])))
        if r is not None and row[4] in vi:keep.append((k,r,vi[row[4]],cell_of(row[5],row[6])))
    ri=np.array([x[1] for x in keep]);S=c.S(0,0,ri);T=lambda x:torch.from_numpy(np.asarray(x).astype(np.int64))
    v,_=values(models,S,T(A['H'][ri]),T(ax['qu'][ri]),T([x[2] for x in keep]),T([x[3] for x in keep]),H,n_roll,None,torch.Generator().manual_seed(seed))
    v=v.mean(0).numpy();rows=[list(r) for r in side['rows']];out=[]
    for (k,_,_,_),val in zip(keep,v):rows[k][9]=round(float(val),6);out.append(rows[k])
    return {**{k:x for k,x in side.items() if k!='rows'},'rows':out,'source':'learnedWm','H':H,'n_roll':n_roll}


def cf_direction(c,ax,models,flipped,Q,cf,H=4,n_roll=4,seed=0):
    # on the simulator's menus: per-state cosine of menu advantages of the simulator, the learned model and its flipped-outcome copy
    # against the held-out real-outcome Q model, and their correlation with the recorded outcome at the recorded play
    groups=load_groups(cf);gid={b:g for g,b in enumerate(c.meta['bid'])};A=c.a;vi={x:i for i,x in enumerate(c.vocab)};where={}
    for g in {gid[b] for b,_ in groups if b in gid}:
        for r in range(int(c.gr[g]),int(c.gr[g+1])):where[(c.meta['bid'][g],int(A['idx'][r]))]=r
    rows=[];cards=[];cells=[];sim=[]
    for key_,plays in groups.items():
        r=where.get(key_)
        if r is None or len(plays)!=4 or any(p[0] not in vi for p in plays):continue
        rows.append(r);cards.append([vi[p[0]] for p in plays]);cells.append([cell_of(p[1],p[2]) for p in plays]);sim.append([p[5] for p in plays])
    rows=np.array(rows);S,Hh,_,_,y=tensors(c,rows);qu=torch.from_numpy(ax['qu'][rows].astype(np.int64));cards=torch.tensor(cards);cells=torch.tensor(cells)
    gen=torch.Generator().manual_seed(seed);sim=torch.tensor(sim,dtype=torch.float32);sim=sim-sim.mean(1,keepdim=True)
    with torch.no_grad():rq=torch.sigmoid(Q(S.repeat_interleave(4,0),cards.reshape(-1),cells.reshape(-1))).view(-1,4);rq=rq-rq.mean(1,keepdim=True)
    wm=menu_adv(models,S,Hh,qu,cards,cells,H,n_roll,gen);w0=menu_adv(models,S,Hh,qu,cards,cells,0,1,gen);fl=menu_adv(flipped,S,Hh,qu,cards,cells,H,n_roll,gen)
    return {'decisions':int(len(rows)),'cos_sim_realQ':cos_rows(sim,rq),'cos_wm_realQ':cos_rows(wm,rq),'cos_wmH0_realQ':cos_rows(w0,rq),
            'cos_wmflip_realQ':cos_rows(fl,rq),
            'cos_wm_sim':cos_rows(wm,sim),'cos_wmflip_sim':cos_rows(fl,sim),'corr_rec_y_sim':corr(sim[:,0],y),'corr_rec_y_wm':corr(wm[:,0],y),
            'corr_rec_y_realQ':corr(rq[:,0],y),'corr_rec_y_wmflip':corr(fl[:,0],y)}


def favourites(pol,S,H,cnt,ref,chunk=8192):
    # support of each state's most probable play: training count of its (card, cell), and its log-probability under the clone
    cs=[];lr=[]
    with torch.no_grad():
        for i in range(0,len(S),chunk):
            s,h=S[i:i+chunk],H[i:i+chunk];menu=pol.menu_logp(s,h);j=menu.flatten(1).argmax(1);card=h.gather(1,(j//N_CELLS)[:,None])[:,0];cell=j%N_CELLS
            cs.append(cnt[card,cell]);lr.append(ref.menu_logp(s,h).flatten(1).gather(1,j[:,None])[:,0])
    cs=torch.cat(cs).float();lr=torch.cat(lr)
    return {'fav_count_lt5':float((cs<5).float().mean()),'fav_count_lt100':float((cs<100).float().mean()),'fav_count_median':float(cs.median()),
            'fav_bc_logp':float(lr.mean())}


def model_gain(models,S,H_,qu,pol,ref,G=16,H=4,seed=0):
    # in-model value of the policy's plays minus the clone's at the given states (both sampled, continuation the behaviour)
    gen=torch.Generator().manual_seed(seed);out=[]
    for p in (pol,ref):
        with torch.no_grad():card,cell=sample(p.menu_logp(S,H_),H_,G,gen)
        rep=lambda x:x.repeat_interleave(G,0);v,_=values(models,rep(S),rep(H_),rep(qu),card.reshape(-1),cell.reshape(-1),H,1,None,gen)
        out.append(float(v.mean()))
    return out[0]-out[1]


def load_policies(c,runs,hidden=128):
    # every saved arm of the given run reports (their .pt sidecars), by spec; simgroup reports keep theirs the same way
    pols={}
    for p in runs:
        pt=Path(p).with_suffix('.pt')
        if not pt.exists():continue
        for spec,sd in torch.load(pt).items():
            pol=Policy(c.n_state,c.n_card,hidden);pol.load_state_dict(sd);pol.eval();pols[f'{Path(p).stem}/{spec}']=pol
    return pols


def diagnose(pack,prep,wm_dir,runs,out=None,n_pred=20000,n_dir=20000,n_gain=4000,K=4,H=4,seed=0,cf=None,hidden=128,log=print):
    c=Corpus(pack);wm_dir=Path(wm_dir);ax=load_aux(wm_dir/'aux.npz');P=Path(prep);rng=np.random.default_rng(seed);A=c.a;t0=time.monotonic()
    say=lambda m:log(f'[diag {time.monotonic()-t0:.0f}s] {m}',flush=True)
    train=[load_member(wm_dir/f'member{i}.pt') for i in MEMBERS];held=load_member(wm_dir/f'member{HELD_OUT}.pt')
    flipped=[load_member(wm_dir/f'member{i}_flip0.pt') for i in MEMBERS]
    hb=c.records(c.games('heldB'));rep={'settings':{'n_pred':n_pred,'n_dir':n_dir,'n_gain':n_gain,'K':K,'H':H,'seed':seed}}
    pr=np.sort(rng.choice(hb,min(n_pred,len(hb)),replace=False));rep['predict']=predict(c,ax,train,pr,K);rep['predict_held_out_member']=predict(c,ax,[held],pr,K)
    say('prediction errors')
    bc=Policy(c.n_state,c.n_card,hidden);bc.load_state_dict(torch.load(P/'bc_warm.pt'));bc.eval();Q=Head(c.n_state,c.n_card,hidden=hidden);Q.load_state_dict(torch.load(P/'q.pt'));Q.eval()
    dr=np.sort(rng.choice(hb,min(n_dir,len(hb)),replace=False));S,Hh,card,cell,y=tensors(c,dr);qu=torch.from_numpy(ax['qu'][dr].astype(np.int64))
    cards,cells=menus(c,dr,seed);cards=torch.from_numpy(cards);cells=torch.from_numpy(cells);gen=torch.Generator().manual_seed(seed)
    with torch.no_grad():rq=torch.sigmoid(Q(S.repeat_interleave(4,0),cards.reshape(-1),cells.reshape(-1))).view(-1,4);rq=rq-rq.mean(1,keepdim=True)
    adv={'wm':menu_adv(train,S,Hh,qu,cards,cells,H,4,gen),'wmH1':menu_adv(train,S,Hh,qu,cards,cells,1,4,gen),'wmH0':menu_adv(train,S,Hh,qu,cards,cells,0,1,gen),
         'wmflip':menu_adv(flipped,S,Hh,qu,cards,cells,H,4,gen),'wm_members01':menu_adv(train[:2],S,Hh,qu,cards,cells,H,4,gen),
         'wm_members23':menu_adv(train[2:],S,Hh,qu,cards,cells,H,4,gen),'wm_held_out':menu_adv([held],S,Hh,qu,cards,cells,H,4,gen)}
    rep['menu_direction']={**{f'cos_{k}_realQ':cos_rows(v,rq) for k,v in adv.items()},'cos_wm_wmH0':cos_rows(adv['wm'],adv['wmH0']),
                           'cos_members01_members23':cos_rows(adv['wm_members01'],adv['wm_members23']),'cos_wm_held_out':cos_rows(adv['wm'],adv['wm_held_out']),
                           **{f'corr_rec_y_{k}':corr(v[:,0],y) for k,v in adv.items()},'corr_rec_y_realQ':corr(rq[:,0],y),'decisions':int(len(dr))}
    say('menu direction')
    if cf:rep['sim_menu_direction']=cf_direction(c,ax,train,flipped,Q,cf,H,4,seed);say('simulator menus')
    pols=load_policies(c,runs,hidden);sub=torch.from_numpy(rng.choice(len(dr),min(n_dir,len(dr)),replace=False))
    shifts={k:shift(p,bc,S[sub],Hh[sub]) for k,p in pols.items()}
    keys=list(shifts);rep['policy_direction']={f'{a}|{b}':agreement(shifts[a],shifts[b]) for i,a in enumerate(keys) for b in keys[i+1:]}
    say(f'policy direction ({len(keys)} policies)')
    tr=train_rows(c);cnt=torch.zeros(c.n_card,N_CELLS,dtype=torch.long)
    cnt.index_put_((torch.from_numpy(A['card'][tr].astype(np.int64)),torch.from_numpy(A['cell'][tr].astype(np.int64))),torch.ones(len(tr),dtype=torch.long),accumulate=True)
    gi=torch.from_numpy(rng.choice(len(dr),min(n_gain,len(dr)),replace=False));ex={'bc':favourites(bc,S,Hh,cnt,bc)}
    for k,p in pols.items():
        ex[k]={**favourites(p,S,Hh,cnt,bc),'gain_train_H4':model_gain(train,S[gi],Hh[gi],qu[gi],p,bc,16,H,seed),'gain_held_out_H4':model_gain([held],S[gi],Hh[gi],qu[gi],p,bc,16,H,seed),
               'gain_train_H0':model_gain(train,S[gi],Hh[gi],qu[gi],p,bc,16,0,seed)}
    rep['exploitation']=ex;rep['seconds']=round(time.monotonic()-t0,1);say('exploitation')
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(rep,indent=1)+'\n')
    return rep


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('aux');a.add_argument('--pack',required=True);a.add_argument('--wm',required=True);a.add_argument('--K',type=int,default=4)
    a=sp.add_parser('fit');a.add_argument('--pack',required=True);a.add_argument('--wm',required=True);a.add_argument('--member',type=int,required=True)
    a.add_argument('--epochs',type=int,default=1);a.add_argument('--threads',type=int,default=6);a.add_argument('--max_steps',type=int,default=0)
    a.add_argument('--batch',type=int,default=1024);a.add_argument('--hidden',type=int,default=256);a.add_argument('--latent',type=int,default=128)
    a=sp.add_parser('refit');a.add_argument('--pack',required=True);a.add_argument('--wm',required=True);a.add_argument('--member',type=int,required=True)
    a.add_argument('--seed',type=int,default=0);a.add_argument('--threads',type=int,default=6)
    a=sp.add_parser('lam');a.add_argument('--pack',required=True);a.add_argument('--prep',required=True);a.add_argument('--wm',required=True)
    a.add_argument('--n',type=int,default=2000);a.add_argument('--threads',type=int,default=8)
    a=sp.add_parser('arms');a.add_argument('--pack',required=True);a.add_argument('--prep',required=True);a.add_argument('--wm',required=True)
    a.add_argument('--arms',nargs='+',required=True);a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=8);a.add_argument('--slice',default='all')
    a=sp.add_parser('menu');a.add_argument('--pack',required=True);a.add_argument('--wm',required=True);a.add_argument('--cf',required=True)
    a.add_argument('--out',required=True);a.add_argument('--prep');a.add_argument('--threads',type=int,default=8)
    a=sp.add_parser('diag');a.add_argument('--pack',required=True);a.add_argument('--prep',required=True);a.add_argument('--wm',required=True)
    a.add_argument('--runs',nargs='*',default=[]);a.add_argument('--out',required=True);a.add_argument('--cf');a.add_argument('--threads',type=int,default=16)
    a=ap.parse_args();torch.set_num_threads(getattr(a,'threads',8));wm=Path(a.wm);wm.mkdir(parents=True,exist_ok=True)
    if a.cmd=='aux':
        c=Corpus(a.pack);t0=time.monotonic();d=aux(c,a.K);np.savez(wm/'aux.npz',**d)
        print(json.dumps({'records':int(len(d['oc'])),'opp_play_rate':round(float((d['oc']>=0).mean()),4),'done_rate':round(float(d['done'].mean()),4),
                          'queue_known':round(float((d['qu']>=0).mean()),4),'seconds':round(time.monotonic()-t0,1),'z_mu':dict(zip(SUM,d['z_mu'].round(4).tolist()))}))
    elif a.cmd=='fit':
        c=Corpus(a.pack);ax=load_aux(wm/'aux.npz');K=int(ax['K']);m,hist=fit(c,ax,train_rows(c),a.member,K,a.epochs,a.hidden,a.latent,batch_size=a.batch,max_steps=a.max_steps)
        save_member(m,wm/f'member{a.member}.pt',hist=hist,K=K)
    elif a.cmd=='refit':
        c=Corpus(a.pack);m=load_member(wm/f'member{a.member}.pt');f=refit_outcome(m,c,train_rows(c),flips(c,a.seed),seed=a.seed)
        save_member(f,wm/f'member{a.member}_flip{a.seed}.pt',flip_seed=a.seed)
    elif a.cmd=='lam':
        c=Corpus(a.pack);ax=load_aux(wm/'aux.npz');bc=Policy(c.n_state,c.n_card,128);bc.load_state_dict(torch.load(Path(a.prep)/'bc_warm.pt'));bc.eval()
        rows=np.sort(np.random.default_rng(0).choice(c.records(c.games('warm')),a.n,replace=False))
        r=calibrate(c,ax,[load_member(wm/f'member{i}.pt') for i in MEMBERS],bc,rows);(wm/'lam.json').write_text(json.dumps(r)+'\n');print(json.dumps(r))
    elif a.cmd=='arms':
        r=run_arms(a.pack,a.arms,a.slice,a.out,threads=a.threads,prep_dir=a.prep,make=make_wm(wm))
        print(json.dumps({k:{x:v[x] for x in ('winner_gap','kl_to_bc','entropy','support_mass','q_support','q_pess','q_direct')} for k,v in r['arms'].items()}))
    elif a.cmd=='menu':
        c=Corpus(a.pack);ax=load_aux(wm/'aux.npz');out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
        cfw=menu_values(c,ax,[load_member(wm/f'member{i}.pt') for i in MEMBERS],a.cf);(out/'cfLearned.json').write_text(json.dumps(cfw)+'\n')
        for name,cf in (('simgroup',a.cf),('wmmenu',out/'cfLearned.json')):simgroup(a.pack,cf,out/f'{name}.json',threads=a.threads,prep_dir=a.prep,name=name)
    else:
        diagnose(a.pack,a.prep,wm,a.runs,a.out,cf=a.cf)


if __name__=='__main__':main()
