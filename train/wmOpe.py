"""Model-based off-policy evaluation checked against the truth: can a world model learnt from production traces evaluate policies?

The ninth step (train.simTruth) made the simulator the production world: the tempered corpus clone played 78,000 games against itself,
written as corpus traces, and the policies trained on them have true win rates against that behaviour from 10,000 seeded games each.
Here every policy is evaluated from the traces alone, as a recorded corpus would have to be used, and the estimates are checked
against the truth. The estimand J(pi) is the policy's chance of beating the behaviour from the start of a game; the starts are the
first decisions of the held-out games' trajectories (the world fixes when players act, and the opponent's plays before the actor's
first decision do not depend on the actor, so they sample the policy's start distribution).

Estimators:
  model  rollouts in a learned world model (train.learnedWm members fit on the first N warm-up and stream games): the first play from
         the policy on the recorded state, every later actor play from the policy distilled into each member's latent actor (fit to
         the policy's play distribution on encoded training states), the opponent's plays, the delays and the game's end from the
         model; the value after k decisions is the ended games' one-step Q plus the running games' V (k = 0 the one-step Q, the
         largest k the full rollout); the penalised value subtracts lambda times the members' disagreement on each predicted decision
         state weighted by the chance that the game reaches it (MOPO's penalty, Yu et al. 2020). The world model has no hand: in
         cycle the hand cycles the actor's recorded deck and the menu is the hand's cards affordable at the own elixir of the recorded
         trajectory's decision at the same depth (in this world a player decides once its elixir reaches a threshold redrawn after
         every play, or under threat, so the elixir at a decision barely depends on the policy; the model's own elixir prediction is
         far less accurate); in recorded the menus are the recorded trajectory's (open loop: a card the policy favours can recur).
  fqe    fitted-Q evaluation (Le, Voloshin and Yue 2019) on the same traces: Q_0 the outcome model of the play, then rounds
         Q_{k+1}(s, a) <- the outcome at the trajectory's last decision, else the policy's expected Q_k at the actor's next decision;
         J_k is the policy's expected Q_k at the starts (the policy for k + 1 decisions, then the behaviour).
  is     trajectory importance sampling with the stored behaviour probabilities or an estimate, plain and self-normalised (wis), and
         per-decision doubly robust (dr, Jiang and Li 2016) and weighted doubly robust (wdr, Thomas and Brunskill 2016) with an FQE Q.
Usage: python -m train.wmOpe fit|lam|mb|q0|fqe|report ... (see main)
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train.learnedWm as LW
from train.simTruth import rank_corr
from train.singleTraj import Corpus
from train.traceRl import HAND,N_CELLS,Head,Policy

MARKS=(0,1,2,4,8,16,32,64)
LAMS=(0.0,0.1,0.3,1.0,3.0)


def T(x):
    return torch.from_numpy(np.ascontiguousarray(x).astype(np.int64))


def load_policy(spec,c):
    # a policy saved by train.singleTraj.run_arms or train.simTruth.grpo, by its agents-file spec ('pack=DIR;file.pt[#key]')
    f=spec.split(';',1)[1] if spec.startswith('pack=') else spec;f,_,k=f.partition('#');sd=torch.load(f,weights_only=False);sd=sd[k] if k else sd
    p=Policy(c.n_state,c.n_card,sd['trunk.0.weight'].shape[0]);p.load_state_dict(sd);p.eval()
    for q in p.parameters():q.requires_grad_(False)
    return p


def registry(files):
    # name -> spec of every packed-store policy of the agents files (the behaviour files, with their own vocabulary, are left out)
    out={}
    for f in files:out.update({k:v for k,v in json.loads(Path(f).read_text()).items() if v.startswith('pack=')})
    return out


def train_games(c,n):
    # the first n warm-up and stream games in battle-time order: the traces a volume of n games provides
    return np.concatenate([c.games('warm'),c.games('stream')])[:n]


def starts(c,splits=('heldA','heldB')):
    # the first decision of every trajectory of the given splits' games
    gs=np.concatenate([c.games(s) for s in splits]);tr=np.concatenate([np.arange(c.gt[g],c.gt[g+1]) for g in gs])
    return c.a['t_start'][tr].astype(np.int64)


def decks(states,c,rows):
    # each record's actor deck as vocabulary indices (-1 outside the vocabulary) from the shards' per-game decks (blue, red)
    want={c.meta['bid'][int(g)] for g in c.a['game'][rows]};dk={};vi={x:i for i,x in enumerate(c.vocab)};out=np.full((len(rows),8),-1,np.int64)
    for p in sorted(Path(states).glob('shard*.json')):
        for o in json.loads(p.read_text())['outcomes']:
            if o['bid'] in want:dk[o['bid']]=o['decks']
    for j,r in enumerate(rows):
        v=[vi.get(x,-1) for x in dk[c.meta['bid'][int(c.a['game'][r])]][int(c.a['team'][r])]][:8];out[j,:len(v)]=v
    return out


def costs(vocab):
    # elixir cost of every vocabulary card (the median for a card the card data does not know)
    from sim.cards import card
    out=np.full(len(vocab),np.nan,np.float32)
    for i,x in enumerate(vocab):
        try:out[i]=card(x)['cost'] or 0
        except KeyError:pass
    return torch.from_numpy(np.nan_to_num(out,nan=float(np.nanmedian(out))))


class Cycle:
    # the actor's hand in a rollout: the start's menu filled to HAND with the deck's other cards in a seeded random order, the rest of
    # the deck queued behind them in that order; a play sends its card to the back of the queue and the queue's front takes its slot;
    # the menu of decision k is the hand, or with cost and elixir (n, depth) the hand's cards whose cost elixir[:, k] covers (the last
    # column after the depth; the cheapest card when none is, as the world waits until one is affordable), known cards first as the
    # store writes menus
    def __init__(self,menu,deck,seed=0,cost=None,elixir=None):
        rng=np.random.default_rng(seed);n=len(menu);hand=np.full((n,HAND),-1,np.int64);queue=np.full((n,8),-1,np.int64)
        for j in range(n):
            m=[int(x) for x in menu[j] if x>=0];rest=[int(x) for x in deck[j] if x>=0 and x not in m];rest=[rest[i] for i in rng.permutation(len(rest))]
            k=max(HAND-len(m),0);h=(m+rest[:k])[:HAND];q=rest[k:][:8];hand[j,:len(h)]=h;queue[j,:len(q)]=q
        self.hand=torch.from_numpy(hand);self.queue=torch.from_numpy(queue);self.cost=cost
        self.el=None if elixir is None else torch.as_tensor(elixir,dtype=torch.float32)
    def __call__(self,k,card,z=None):
        n=len(card);ar=torch.arange(n);hit=self.hand==card[:,None];row=hit.any(1);j=hit.float().argmax(1);front=self.queue[:,0].clone()
        q=torch.cat([self.queue[:,1:],torch.full((n,1),-1,dtype=torch.long)],1);q[ar,(q>=0).sum(1).clamp(max=q.shape[1]-1)]=card
        self.hand=torch.where(row[:,None]&(torch.arange(HAND)[None,:]==j[:,None]),front[:,None],self.hand);self.queue=torch.where(row[:,None],q,self.queue)
        return self.menu(k+1)
    def menu(self,k=None):
        h=self.hand
        if self.cost is not None and self.el is not None and k is not None:
            el=self.el[:,min(k,self.el.shape[1]-1)];cc=torch.where(h>=0,self.cost[h.clamp(min=0)],torch.full(h.shape,1e9))
            ok=cc<=el[:,None]+1e-6;ok[torch.arange(len(h)),cc.argmin(1)]=True;h=torch.where(ok&(h>=0),h,torch.full_like(h,-1))
        return h.gather(1,torch.argsort((h<0).int(),dim=1,stable=True))


def ahead(c,rows,H):
    # the record of each start's trajectory at depths 0..H (the trajectory's last record after it ends)
    A=c.a;tr=A['traj'][rows];end=A['t_start'][tr].astype(np.int64)+A['t_len'][tr].astype(np.int64)-1
    return np.minimum(np.asarray(rows)[:,None]+np.arange(H+1)[None,:],end[:,None])


def own_elixir(c,idx):
    # the actor's own elixir at the given records (train.feats keeps it over 10 at index 8)
    return np.asarray(c.X[idx.reshape(-1)][:,8],np.float32).reshape(idx.shape)*10


class Recorded:
    # the menus of the recorded trajectory: decision k + 1 of a rollout gets the menu of the start's (k + 1)-th next recorded decision,
    # the trajectory's last menu after it ends
    def __init__(self,c,rows,H):
        self.m=T(c.a['H'][ahead(c,rows,H)])
    def __call__(self,k,card,z=None):
        return self.m[:,k+1]


def distil(models,pol,c,rows,steps=1500,batch=512,lr=1e-3,seed=0):
    # one latent actor per member, started from the member's actor head (the behaviour) and fit to the policy's play distribution on the
    # encoded states of the given records (cross-entropy to the policy's menu probabilities, as train.learnedWm.WmArm distils)
    rng=np.random.default_rng(seed);torch.manual_seed(seed);acts=[copy.deepcopy(m.actor) for m in models];ps=[p for a in acts for p in a.parameters()]
    for p in ps:p.requires_grad_(True)
    opt=torch.optim.Adam(ps,lr=lr);A=c.a;rows=np.asarray(rows)
    for _ in range(steps):
        r=np.sort(rng.choice(rows,batch));S=c.S(0,0,r);H=T(A['H'][r])
        with torch.no_grad():tgt=pol.menu_logp(S,H).exp();hs=[m.encode(S) for m in models]
        loss=sum(-(tgt*a.menu_logp(h,H).clamp(min=-30)).sum((1,2)).mean() for a,h in zip(acts,hs))/len(acts);opt.zero_grad();loss.backward();opt.step()
    for a in acts:
        a.eval()
        for p in a.parameters():p.requires_grad_(False)
    return acts


def fidelity(models,acts,pol,S,H):
    # mean KL(policy || each member's distilled actor) and KL(policy || the member's own actor head) on the given states
    with torch.no_grad():
        lp=pol.menu_logp(S,H).clamp(min=-30);p=lp.exp();out={'kl_actor':[],'kl_head':[]}
        for m,a in zip(models,acts):
            h=m.encode(S);out['kl_actor'].append(float((p*(lp-a.menu_logp(h,H).clamp(min=-30))).sum((1,2)).mean()))
            out['kl_head'].append(float((p*(lp-m.actor.menu_logp(h,H).clamp(min=-30))).sum((1,2)).mean()))
    return out


def model_values(models,acts,pol,S,H_,nxt_of,Hmax,marks=MARKS,n_roll=2,seed=0,chunk=2048):
    # per start, means over n_roll rollouts of the members' mean value after each mark's decisions, the penalty's disagreement and the
    # chance that the game is still running; the first play is the policy's on the recorded state (the members' actor heads when pol is
    # None, the model's own behaviour), nxt_of(slice, roll) the menus of the rollout's later decisions
    gen=torch.Generator().manual_seed(seed);n=len(S);ks=[k for k in marks if k<=Hmax];out={x:np.zeros((len(ks),n)) for x in ('value','disagree','running')}
    with torch.no_grad():
        for i in range(0,n,chunk):
            s=slice(i,min(i+chunk,n));hs=[m.encode(S[s]) for m in models];Hs=H_[s]
            if pol is not None:menu=pol.menu_logp(S[s],Hs)
            else:menu=torch.stack([m.actor.menu_logp(h,Hs) for m,h in zip(models,hs)]).exp().mean(0).clamp(min=1e-30).log()
            for r in range(n_roll):
                card,cell=LW.sample(menu,Hs,1,gen);card,cell=card[:,0].clamp(min=0),cell[:,0];tr={'marks':set(ks)-{0}}
                if Hmax>0:LW.rollout(models,hs,Hs,None,card,cell,Hmax,acts,gen,nxt=nxt_of(s,r),trace=tr)
                if 0 in ks:
                    w=s.stop-s.start;tr[0]=(torch.stack([torch.sigmoid(m.qv(h,m.play(card,cell))) for m,h in zip(models,hs)]),torch.zeros(w),torch.ones(w))
                for j,k in enumerate(ks):
                    v,u,a=tr[k];out['value'][j,s]+=v.mean(0).numpy()/n_roll;out['disagree'][j,s]+=u.numpy()/n_roll;out['running'][j,s]+=a.numpy()/n_roll
    out['marks']=np.array(ks);return out


def menu_qv(Q,S,H,chunk=512):
    # P(win) of every (hand card, cell) play under a train.traceRl.Head, (n, HAND, N_CELLS): its first layer split into the state, card
    # and cell parts, so a state's 192 plays share one state product (train.traceRl.menu_q without the support mask)
    W=Q.net[0].weight;b=Q.net[0].bias;ns=S.shape[1];e=Q.emb.embedding_dim;E=Q.emb.weight@W[:,ns:ns+e].T;C=W[:,ns+e:].T;out=torch.zeros(len(S),HAND,N_CELLS)
    for i in range(0,len(S),chunk):
        s=slice(i,i+chunk);pre=(S[s]@W[:,:ns].T+b)[:,None,None,:]+E[H[s].clamp(min=0)][:,:,None,:]+C[None,None]
        out[s]=torch.sigmoid(Q.net[2](Q.net[1](pre))[...,0])
    return out


def fit_q0(c,rows,steps=3000,batch=2048,lr=2e-3,l2=1e-3,hidden=128,seed=0):
    # the outcome model of the play, P(actor wins | state, play), on minibatches of the given records (the Q-eval's model class)
    torch.manual_seed(seed);rng=np.random.default_rng(seed);Q=Head(c.n_state,c.n_card,hidden=hidden);opt=torch.optim.AdamW(Q.parameters(),lr=lr,weight_decay=l2)
    A=c.a;rows=np.asarray(rows)
    for _ in range(steps):
        r=np.sort(rng.choice(rows,batch));y=torch.from_numpy(A['y'][r].astype(np.float32))
        loss=F.binary_cross_entropy_with_logits(Q(c.S(0,0,r),T(A['card'][r]),T(A['cell'][r])),y);opt.zero_grad();loss.backward();opt.step()
    Q.eval();return Q


def fqe(c,pol,rows,S0,H0,Q0,K=32,steps=300,batch=2048,lr=1e-3,seed=0,keep=(),sampled=False):
    # fitted-Q evaluation from the given training records, each round warm-started from the last for a fixed number of minibatch steps;
    # the target of a trajectory's last decision is its outcome, of any other the policy's expected Q_k at the actor's next decision (or
    # Q_k of one play sampled from the policy when sampled); returns J_k for k = 0..K, the per-start values and the models of the rounds
    # in keep
    torch.manual_seed(seed);rng=np.random.default_rng(seed);A=c.a;rows=np.asarray(rows)
    with torch.no_grad():P0=pol.menu_logp(S0,H0).exp()
    def J(Q):
        with torch.no_grad():return (P0*menu_qv(Q,S0,H0)).sum((1,2))
    Q=copy.deepcopy(Q0);v=J(Q);out={'J':[float(v.mean())],'per':{0:v.numpy()}};kept={0:copy.deepcopy(Q0)} if 0 in keep else {}
    for p in Q.parameters():p.requires_grad_(True)
    for k in range(1,K+1):
        Qk=copy.deepcopy(Q).eval();opt=torch.optim.Adam(Q.parameters(),lr=lr)
        for _ in range(steps):
            r=np.sort(rng.choice(rows,batch));last=A['pos'][r]>=A['t_len'][A['traj'][r]]-1;y=A['y'][r].astype(np.float32);rn=r[~last]+1
            if len(rn):
                Sn=c.S(0,0,rn);Hn=T(A['H'][rn])
                with torch.no_grad():
                    if sampled:
                        cd,cl=LW.sample(pol.menu_logp(Sn,Hn),Hn,1);y[~last]=torch.sigmoid(Qk(Sn,cd[:,0],cl[:,0])).numpy()
                    else:y[~last]=(pol.menu_logp(Sn,Hn).exp()*menu_qv(Qk,Sn,Hn)).sum((1,2)).numpy()
            loss=F.binary_cross_entropy_with_logits(Q(c.S(0,0,r),T(A['card'][r]),T(A['cell'][r])),torch.from_numpy(y));opt.zero_grad();loss.backward();opt.step()
        v=J(Q);out['J'].append(float(v.mean()));out['per'][k]=v.numpy()
        if k in keep:kept[k]=copy.deepcopy(Q).eval()
    return out,kept


def play_logp(pol,S,H,card,cell,chunk=8192):
    # log-probability of each recorded play (card, cell) under a policy's menu distribution
    out=[];slot=(H==card[:,None]).float().argmax(1)
    with torch.no_grad():
        for i in range(0,len(S),chunk):
            s=slice(i,i+chunk);m=pol.menu_logp(S[s],H[s]);out.append(m[torch.arange(len(m)),slot[s],cell[s]])
    return torch.cat(out)


def q_and_v(Q,pol,S,H,card,cell,chunk=4096):
    # the model's Q of each recorded play and its V under the policy (the policy's expected Q over the menu)
    qs=[];vs=[]
    with torch.no_grad():
        for i in range(0,len(S),chunk):
            s=slice(i,i+chunk);q=menu_qv(Q,S[s],H[s]);slot=(H[s]==card[s][:,None]).float().argmax(1)
            qs.append(q[torch.arange(len(q)),slot,cell[s]]);vs.append((pol.menu_logp(S[s],H[s]).exp()*q).sum((1,2)))
    return torch.cat(qs).numpy(),torch.cat(vs).numpy()


def trajectory_ope(tr,pos,lr,y,q=None,v=None,idx=None):
    # IS, WIS, per-decision DR (Jiang and Li 2016) and weighted DR (Thomas and Brunskill 2016) from per-record trajectory index tr
    # (0..n-1), decision position, log ratio log pi - log mu of the recorded play, and per-trajectory outcome y (the only reward, at the
    # last decision); q, v the model's Q of the recorded play and V of the state under the policy; ended trajectories are padded with
    # ratio 1 and zero values; idx resamples trajectories (a bootstrap draw); also the effective sample size of the final weights
    tr=np.asarray(tr);pos=np.asarray(pos);n=int(tr.max())+1;L=int(pos.max())+1;y=np.asarray(y,np.float64)
    lw=np.zeros((n,L));lw[tr,pos]=lr;lw=np.cumsum(lw,1);last=np.zeros(n,np.int64);np.maximum.at(last,tr,pos)
    R=np.zeros((n,L));R[np.arange(n),last]=y
    if idx is not None:lw,R,last,y=lw[idx],R[idx],last[idx],y[idx]
    # unnormalised weights are capped at e^300 so a far policy's estimate stays finite (it is meaningless there anyway)
    m=len(y);wf=lw[:,-1];sw=np.exp(wf-wf.max());out={'is':float(np.mean(np.exp(np.minimum(wf,300))*y)),'wis':float((sw*y).sum()/sw.sum()),
                                                   'ess':float(sw.sum()**2/(sw**2).sum())}
    if q is not None:
        Q=np.zeros((n,L));V=np.zeros((n,L));Q[tr,pos]=q;V[tr,pos]=v
        if idx is not None:Q,V=Q[idx],V[idx]
        w=np.exp(np.minimum(lw,300));wp=np.concatenate([np.ones((m,1)),w[:,:-1]],1)
        out['dr']=float((w*R-w*Q+wp*V).sum(1).mean())
        nw=np.exp(lw-lw.max(0));nw=nw/nw.sum(0);nwp=np.concatenate([np.full((m,1),1/m),nw[:,:-1]],1)
        out['wdr']=float((nw*R-nw*Q+nwp*V).sum())
    return out


def compare(est,truth,kl,far=0.3,ref='bc',per=None,boots=0,seed=0):
    # one estimator against the truth over the trained policies (every name but ref): est maps a name to its estimate of J (or of the
    # gain over ref when ref is absent), truth to {'wr': (mean, lo, hi), 'dwr': (mean, lo, hi)} in win-rate units; Spearman and Kendall
    # of the estimated and true gains over ref, the mean absolute and signed error of the gain and of the level (J) in points, the error
    # on the far policies (KL to ref above far), the pick (largest estimated gain), whether it is the true best by the ninth step's rule
    # (its true gain at least the lower end of the true best's interval), its regret, and Spearman's correlation of the absolute gain
    # error with KL; per (name -> per-start values) adds Spearman's 95% interval over boots resamples of the starts
    names=[a for a in truth if a!=ref and a in est];g=np.array([est[a]-(est[ref] if ref in est else 0.0) for a in names])
    t=np.array([truth[a]['dwr'][0] for a in names])
    sp,tau=rank_corr(g,t);err=g-t;k=np.array([kl[a] for a in names]);best=int(np.argmax(t));pick=int(np.argmax(g));lo=truth[names[best]]['dwr'][1]
    r={'policies':len(names),'spearman':sp,'kendall':tau,'mae_gain':float(np.abs(err).mean()*100),'bias_gain':float(err.mean()*100),
       'pick':names[pick],'true_best':names[best],'picks_best':bool(t[pick]>=lo),'regret':float((t[best]-t[pick])*100),
       'err_kl_spearman':rank_corr(np.abs(err),k)[0],'far':{a:float(e*100) for a,e,x in zip(names,err,k) if x>far}}
    r['mae_far']=float(np.mean(np.abs(list(r['far'].values())))) if r['far'] else None
    near=k<=far;r['spearman_near']=rank_corr(g[near],t[near])[0] if near.sum()>2 else None
    if ref in est:r['mae_level']=float(np.mean([abs(est[a]-truth[a]['wr'][0]) for a in names+[ref] if 'wr' in truth[a]])*100);r['level_ref']=float(est[ref])
    if per is not None and boots:
        rng=np.random.default_rng(seed);P=np.stack([per[a] for a in names]);P0=per[ref];n=P.shape[1];bs=[]
        for _ in range(boots):
            i=rng.integers(n,size=n);bs.append(rank_corr(P[:,i].mean(1)-P0[i].mean(),t)[0])
        r['spearman_boot']=[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))]
    return r


def truth_of(report,opp='behaviour'):
    # the ninth step's true win rates against an opponent and the Q-eval's proxies, by policy name, from train.simTruth.report's json
    rep=json.loads(Path(report).read_text());tr={};kl={};qe={}
    for a,r in rep['rows'].items():
        if opp in r['true']:tr[a]={'wr':r['true'][opp]['wr'],'dwr':r['true'][opp]['dwr']}
        if 'proxy' in r:kl[a]=r['proxy']['kl'];qe[a]={k:r['proxy'][k] for k in ('dq_pess','dq_support','dq_direct','d_winner_gap')}
    return tr,kl,qe


# ---- jobs

def fit_volume(pack,aux,out,games,member,max_steps=4079,threads=3,log=print):
    # one member on the first games warm-up and stream games, as many gradient steps whatever the volume (epochs as needed)
    torch.set_num_threads(threads);c=Corpus(pack);ax=LW.load_aux(aux);rows=c.records(train_games(c,games));K=int(ax['K'])
    ep=max(1,math.ceil(max_steps/math.ceil(len(rows)/1024)));t0=time.monotonic()
    m,hist=LW.fit(c,ax,rows,member,K,ep,max_steps=max_steps,log=log);Path(out).parent.mkdir(parents=True,exist_ok=True)
    LW.save_member(m,out,hist=hist,K=K,games=games,records=int(len(rows)),epochs=ep,seconds=round(time.monotonic()-t0,1))


def mb_job(pack,states,wms,configs,spec,name,games,out,Hmax=64,marks=MARKS,n_roll=2,distil_steps=1500,menus='cycle',start='held',n_starts=0,
           seed=0,log=print):
    # one policy (spec, or 'model' for the members' own behaviour) in one volume's members: distil it into every member, then roll it out
    # from the starts under each ensemble configuration (lists of member indices into wms); per-start arrays to out (npz) and a json
    c=Corpus(pack);t0=time.monotonic();A=c.a;models=[LW.load_member(p) for p in wms]
    say=lambda m:log(f'[mb {time.monotonic()-t0:.0f}s] {m}',flush=True)
    # held: every held-out trajectory's first decision; mid: decisions drawn from held-out B (the behaviour's state distribution)
    rows=starts(c) if start=='held' else c.records(c.games('heldB'))
    if n_starts or start!='held':rows=np.sort(np.random.default_rng(seed).choice(rows,min(n_starts or 8000,len(rows)),replace=False))
    S=c.S(0,0,rows);H=T(A['H'][rows]);rep={'name':name,'spec':spec,'games':games,'starts':int(len(rows)),'start':start,'menus':menus,'Hmax':Hmax,'n_roll':n_roll,
                                           'members':[str(p) for p in wms],'configs':configs}
    pol=None if spec=='model' else load_policy(spec,c);acts=None
    if pol is not None:
        tg=c.records(train_games(c,games));acts=distil(models,pol,c,tg,distil_steps,seed=seed)
        fr=np.sort(np.random.default_rng(seed+1).choice(rows,min(4000,len(rows)),replace=False));rep['fidelity']=fidelity(models,acts,pol,c.S(0,0,fr),T(A['H'][fr]))
        say(f"distilled: KL actor {np.mean(rep['fidelity']['kl_actor']):.4f} head {np.mean(rep['fidelity']['kl_head']):.4f}")
    if menus=='cycle':
        dk=decks(states,c,rows);cost=costs(c.vocab);el=own_elixir(c,ahead(c,rows,Hmax));nxt=lambda s,r:Cycle(A['H'][rows[s]],dk[s],seed*1000+r,cost,el[s])
    else:nxt=lambda s,r:Recorded(c,rows[s],Hmax)
    res={}
    for ci,cfg in enumerate(configs):
        ms=[models[i] for i in cfg];ac=None if acts is None else [acts[i] for i in cfg]
        v=model_values(ms,ac,pol,S,H,nxt,Hmax,marks,n_roll,seed)
        for k,x in v.items():res[f'c{ci}_{k}']=x
        rep[f'c{ci}']={'J':dict(zip(map(int,v['marks']),v['value'].mean(1).round(5).tolist())),'running':dict(zip(map(int,v['marks']),v['running'].mean(1).round(4).tolist())),
                       'disagree':dict(zip(map(int,v['marks']),v['disagree'].mean(1).round(4).tolist()))};say(f"config {cfg}: J {rep[f'c{ci}']['J']}")
    rep['seconds']=round(time.monotonic()-t0,1);Path(out).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,rows=rows,**res)
    Path(out).with_suffix('.json').write_text(json.dumps(rep)+'\n');return rep


def lam_job(pack,prep,aux,wms,key,out,n=2000,seed=0):
    # the penalty's weight for one ensemble, by train.learnedWm.calibrate's rule (on warm-up states, the clone's plays, H = 4)
    c=Corpus(pack);ax=LW.load_aux(aux);bc=load_policy(str(Path(prep)/'bc_warm.pt'),c)
    rows=np.sort(np.random.default_rng(seed).choice(c.records(c.games('warm')),n,replace=False))
    r=LW.calibrate(c,ax,[LW.load_member(p) for p in wms],bc,rows,seed=seed);Path(out).write_text(json.dumps({'key':key,**r})+'\n');return r


def q0_job(pack,games,out,steps=3000,seed=0,log=print):
    c=Corpus(pack);t0=time.monotonic();rows=c.records(train_games(c,games));Q=fit_q0(c,rows,steps,seed=seed);hb=c.records(c.games('heldB'))
    r=np.sort(np.random.default_rng(seed).choice(hb,min(100000,len(hb)),replace=False));A=c.a
    with torch.no_grad():p=torch.sigmoid(Q(c.S(0,0,r),T(A['card'][r]),T(A['cell'][r]))).clamp(1e-6,1-1e-6)
    y=torch.from_numpy(A['y'][r].astype(np.float32));rep={'games':games,'records':int(len(rows)),'steps':steps,'heldB_logloss':float(F.binary_cross_entropy(p,y)),
                                                       'heldB_acc':float(((p>0.5).float()==y).float().mean()),'seconds':round(time.monotonic()-t0,1)}
    torch.save({'state':Q.state_dict(),**rep},out);log(json.dumps(rep),flush=True);return rep


def fqe_job(pack,spec,name,games,q0,out,K=32,steps=300,batch=2048,keep=(0,8,32),mu_est=None,sampled=False,boots=200,seed=0,log=print):
    # FQE of one policy on one volume's traces from the volume's Q_0, then IS, WIS, DR and WDR on the held-out trajectories with the
    # stored behaviour probabilities (and with the estimate mu_est, a policy file, when given) and the kept rounds' Q
    c=Corpus(pack);t0=time.monotonic();A=c.a;say=lambda m:log(f'[fqe {time.monotonic()-t0:.0f}s] {m}',flush=True)
    pol=load_policy(spec,c);rows=c.records(train_games(c,games));s0=starts(c);S0=c.S(0,0,s0);H0=T(A['H'][s0])
    Q0=Head(c.n_state,c.n_card,hidden=128);Q0.load_state_dict(torch.load(q0,weights_only=False)['state']);Q0.eval()
    res,kept=fqe(c,pol,rows,S0,H0,Q0,K,steps,batch,seed=seed,keep=keep,sampled=sampled);say(f"J {[round(x,4) for x in res['J']]}")
    rep={'name':name,'spec':spec,'games':games,'K':K,'steps':steps,'batch':batch,'sampled':sampled,'J':res['J']}
    gs=np.concatenate([c.games('heldA'),c.games('heldB')]);hr=c.records(gs);S=c.S(0,0,hr);H=T(A['H'][hr]);cd=T(A['card'][hr]);cl=T(A['cell'][hr])
    tr=A['traj'][hr].astype(np.int64);_,tr=np.unique(tr,return_inverse=True);pos=A['pos'][hr].astype(np.int64);last=pos==A['t_len'][A['traj'][hr]]-1
    y=np.zeros(int(tr.max())+1);y[tr[last]]=A['y'][hr][last];lp=play_logp(pol,S,H,cd,cl).numpy().astype(np.float64)
    P=Path(pack)/'prep';mus={'stored':np.load(P/'mu_true_1.npy',mmap_mode='r')[hr].sum(1).astype(np.float64)}
    if mu_est:mus['estimated']=play_logp(load_policy(mu_est,c),S,H,cd,cl).numpy().astype(np.float64)
    qv={k:q_and_v(Q,pol,S,H,cd,cl) for k,Q in kept.items()};rng=np.random.default_rng(seed);n=len(y)
    for mk,lm in mus.items():
        for k,(q,v) in qv.items():
            e=trajectory_ope(tr,pos,lp-lm,y,q,v);rep[f'{mk}|q{k}']=e
            if boots:
                bs=[trajectory_ope(tr,pos,lp-lm,y,q,v,rng.integers(n,size=n)) for _ in range(boots)]
                rep[f'{mk}|q{k}']['boot_sd']={x:float(np.std([b[x] for b in bs])) for x in ('is','wis','dr','wdr')}
    np.savez_compressed(out,starts=s0,**{f'J{k}':v for k,v in res['per'].items()});rep['seconds']=round(time.monotonic()-t0,1)
    Path(out).with_suffix('.json').write_text(json.dumps(rep)+'\n');say('done');return rep


# ---- report

def estimates(mb_dir,fqe_dir):
    # every estimator's J (or penalised J) by policy, and per-start values for the bootstrap: keys like
    # 'model|v70k|M4|cycle|held|H64|lam0', 'fqe|v70k|K32', 'stored|q32|wdr|v70k'
    est={};per={};lam={json.loads(p.read_text())['key']:json.loads(p.read_text())['lam'] for p in Path(mb_dir).glob('lam_*.json')}
    for p in sorted(Path(mb_dir).glob('*.json')):
        if p.name.startswith('lam_'):continue
        r=json.loads(p.read_text());d=np.load(p.with_suffix('.npz'))
        for ci,cfg in enumerate(r['configs']):
            ks=d[f'c{ci}_marks'];v=d[f'c{ci}_value'];u=d[f'c{ci}_disagree'];base=lam.get(f"v{r['games']//1000}k|M{len(cfg)}")
            for j,k in enumerate(ks):
                for lm in LAMS:
                    if lm and (k==0 or base is None):continue
                    key=f"model|v{r['games']//1000}k|M{len(cfg)}|{r['menus']}|{r['start']}|H{int(k)}|lam{lm:g}";x=v[j]-lm*(base or 0)*u[j]
                    est.setdefault(key,{})[r['name']]=float(x.mean());per.setdefault(key,{})[r['name']]=x
    for p in sorted(Path(fqe_dir).glob('*.json')):
        r=json.loads(p.read_text());d=np.load(p.with_suffix('.npz'));vol=f"v{r['games']//1000}k"
        for k in sorted({0,1,2,4,8,16,32,r['K']}):
            if k<len(r['J']):
                key=f'fqe|{vol}|K{k}';est.setdefault(key,{})[r['name']]=r['J'][k]
                if f'J{k}' in d.files:per.setdefault(key,{})[r['name']]=d[f'J{k}']
        for mk in [x for x in r if '|q' in x]:
            for e in ('is','wis','dr','wdr'):
                if e in r[mk] and (e in ('dr','wdr') or mk.endswith('|q0')):
                    key=f"{mk.split('|')[0]}|{e}|{vol}"+(f"|{mk.split('|')[1]}" if e in ('dr','wdr') else '');est.setdefault(key,{})[r['name']]=r[mk][e]
    return est,per


def report(truth,mb_dir,fqe_dir,out=None,boots=1000,opp='behaviour'):
    tr,kl,qe=truth_of(truth,opp);est,per=estimates(mb_dir,fqe_dir);rows={}
    names=[a for a in tr if a in kl or a=='bc']
    rows['qeval|dq_pess']=compare({a:qe[a]['dq_pess'] for a in names if a in qe},tr,kl)
    rows['qeval|dq_support']=compare({a:qe[a]['dq_support'] for a in names if a in qe},tr,kl)
    for k,e in est.items():
        if not all(a in e for a in names):continue
        rows[k]=compare(e,{a:tr[a] for a in names},kl,per=per.get(k),boots=boots if k in per else 0)
    model=[k for k in est if k.startswith('model') and 'model' in est[k]]
    rep={'opponent':opp,'rows':rows,'model_behaviour':{k:est[k]['model'] for k in model},'policies':names,'kl':kl}
    if out:Path(out).write_text(json.dumps(rep,indent=1)+'\n');Path(out).with_suffix('.md').write_text(markdown(rep)+'\n')
    return rep


def markdown(rep):
    f=lambda v,d=3:'' if v is None else f'{v:+.{d}f}' if isinstance(v,float) else str(v)
    out=['| estimator | Spearman [boot] | Kendall | Spearman near | MAE gain (pts) | bias (pts) | MAE level (pts) | J(bc) | MAE far | qgroup err | pick | best '
         '| regret | abs err vs KL |',
         '|'+'---|'*14]
    for k,r in rep['rows'].items():
        b=r.get('spearman_boot');sp=f"{r['spearman']:+.3f}"+(f' [{b[0]:+.3f}, {b[1]:+.3f}]' if b else '')
        out.append('| '+' | '.join([k,sp,f(r['kendall']),f(r.get('spearman_near')),f(r['mae_gain'],1) if not k.startswith('qeval') else '',
                                    f(r['bias_gain'],1) if not k.startswith('qeval') else '',f(r.get('mae_level'),1),f(r.get('level_ref')),
                                    f(r.get('mae_far'),1) if not k.startswith('qeval') else '',
                                    f(r['far'].get('wm/qgroup'),1) if not k.startswith('qeval') else '',
                                    r['pick'],str(r['picks_best']),f(r['regret'],1),f(r['err_kl_spearman'])])+' |')
    return '\n'.join(out)


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('fit');a.add_argument('--pack',required=True);a.add_argument('--aux',required=True);a.add_argument('--out',required=True)
    a.add_argument('--games',type=int,required=True);a.add_argument('--member',type=int,required=True);a.add_argument('--max_steps',type=int,default=4079)
    a.add_argument('--threads',type=int,default=3)
    a=sp.add_parser('mb');a.add_argument('--pack',required=True);a.add_argument('--states',required=True)
    a.add_argument('--wm',nargs='+',required=True);a.add_argument('--configs',required=True,help='json list of member index lists')
    a.add_argument('--agents',nargs='+')
    a.add_argument('--policy',required=True);a.add_argument('--games',type=int,required=True);a.add_argument('--out',required=True);a.add_argument('--H',type=int,default=64)
    a.add_argument('--n_roll',type=int,default=2);a.add_argument('--distil_steps',type=int,default=1500);a.add_argument('--menus',default='cycle')
    a.add_argument('--start',default='held');a.add_argument('--n_starts',type=int,default=0);a.add_argument('--threads',type=int,default=1)
    a=sp.add_parser('lam');a.add_argument('--pack',required=True);a.add_argument('--prep',required=True);a.add_argument('--aux',required=True)
    a.add_argument('--wm',nargs='+',required=True);a.add_argument('--key',required=True);a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=4)
    a=sp.add_parser('q0');a.add_argument('--pack',required=True);a.add_argument('--games',type=int,required=True);a.add_argument('--out',required=True)
    a.add_argument('--steps',type=int,default=3000);a.add_argument('--threads',type=int,default=1)
    a=sp.add_parser('fqe');a.add_argument('--pack',required=True);a.add_argument('--agents',nargs='+',required=True);a.add_argument('--policy',required=True)
    a.add_argument('--games',type=int,required=True);a.add_argument('--q0',required=True);a.add_argument('--out',required=True);a.add_argument('--K',type=int,default=32)
    a.add_argument('--steps',type=int,default=300);a.add_argument('--batch',type=int,default=2048);a.add_argument('--mu_est');a.add_argument('--sampled',action='store_true')
    a.add_argument('--boots',type=int,default=200);a.add_argument('--threads',type=int,default=1)
    a=sp.add_parser('report');a.add_argument('--truth',required=True);a.add_argument('--mb',required=True);a.add_argument('--fqe',required=True)
    a.add_argument('--out',required=True);a.add_argument('--opp',default='behaviour')
    a=ap.parse_args();torch.set_num_threads(getattr(a,'threads',1))
    if a.cmd=='fit':fit_volume(a.pack,a.aux,a.out,a.games,a.member,a.max_steps,a.threads)
    elif a.cmd=='mb':
        spec='model' if a.policy=='model' else registry(a.agents)[a.policy]
        r=mb_job(a.pack,a.states,a.wm,json.loads(a.configs),spec,a.policy,a.games,a.out,a.H,n_roll=a.n_roll,distil_steps=a.distil_steps,menus=a.menus,
                 start=a.start,n_starts=a.n_starts)
        print(json.dumps({k:v for k,v in r.items() if k.startswith('c') or k in ('seconds','fidelity')}))
    elif a.cmd=='lam':print(json.dumps(lam_job(a.pack,a.prep,a.aux,a.wm,a.key,a.out)))
    elif a.cmd=='q0':q0_job(a.pack,a.games,a.out,a.steps)
    elif a.cmd=='fqe':
        reg=registry(a.agents);fqe_job(a.pack,reg[a.policy],a.policy,a.games,a.q0,a.out,a.K,a.steps,a.batch,mu_est=reg.get(a.mu_est,a.mu_est),sampled=a.sampled,boots=a.boots)
    else:
        r=report(a.truth,a.mb,a.fqe,a.out,opp=a.opp);print(markdown(r))


if __name__=='__main__':main()
