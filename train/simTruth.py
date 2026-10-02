"""Ground truth for single-trajectory RL: the simulator as the production world.

Every trace-RL verdict so far rests on proxies measured on recorded games (the support-restricted Q-eval of an outcome model fit on
held-out outcomes, and the winner gap), because the real environment cannot be replayed. Here the simulator plays the production
world. A behaviour policy (the clone of the recorded corpus, tempered so it explores) plays both sides of sim.game games at decision
resolution in the action space of train.singleTraj (a card from the menu, then one of the 48 deploy cells given the card); the games
are written as train.corpusStates shards (one trajectory per player, the outcome as the only reward), with the behaviour's token
log-probabilities stored beside them and used only by the arms told to (mu=true; the default estimate is the cross-fitted clone, as
with human traces). The shipped pack, prep and run_arms train on them exactly as on the corpus. Because this world can be replayed,
environment-replay GRPO is the upper bound: from a recorded decision state, G continuations (the first play and the actor's later
plays from the current policy, the opponent's from the behaviour) to the end of the game, advantages the outcome minus the group mean
over the group's spread, every actor token of a continuation weighted by its advantage; its budget is counted in simulated ticks, the
unit the traces cost. Every policy's true win rate is then measured by playing it against the behaviour and a fixed rule bot.

World rules, the same for every policy, so policies differ only in which card and cell they pick:
  timing  a player decides once a second has passed since its last play, it can afford a hand card, and either its elixir has
          reached a threshold drawn uniformly from 3..10 after each play or an enemy troop stands on its half;
  menu    the hand cards it can afford (written as the record's hand, so the packed menu H is the menu);
  cell    the cell's centre tile (row offset mirrored for red, so the two grids correspond); a troop or building whose tile is not
          deployable takes the nearest deployable tile of the cell, then of the cell mirrored to its own half, then a fixed tile in
          front of its king; the record keeps the chosen cell;
  decks   a pair drawn per game from a pool of recorded decks (evolutions and heroes as recorded, every card at level 11).
Usage: python -m train.simTruth decks|behaviour|gen|stored|prep|grpo|wins|report ... (see main)
"""
import argparse
import copy
import json
import random
import time
from datetime import datetime,timezone
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from sim.cards import card as card_def,key
from sim.game import Game,card_info
from sim.units import Building,Troop
from train.corpusStates import EXTRA,ROLL,clone,rollout
from train.counterfactual import GH,GW,cell_of
from train.decisionHead import relative
from train.decisionStates import COLS
from train.feats import FEAT_DIM,featurize
from train.traceRl import HAND,Policy

TEAMS=('blue','red');COOL=1.0;THR=(3,10);LEVEL=11;HALF=16.0;TOL=1e-3  # TOL: the store keeps times as float32, ticks are 0.05 s apart
FALLBACK={'blue':(8,7),'red':(8,24)}
DAY=86400.0;T0=datetime(2026,8,30,tzinfo=timezone.utc).timestamp()


def opp(team):
    return 'red' if team=='blue' else 'blue'


def cell_tile(c,team):
    gy,gx=divmod(int(c),GW);return 3*gx+1,4*gy+(2 if team=='blue' else 1)


def place(g,team,name,c):
    # the world's deploy tile for a card on a cell, None when nothing is deployable
    tx,ty=cell_tile(c,team)
    if card_info(name)['deploy_anywhere']:return tx,ty
    gy,gx=divmod(int(c),GW)
    for cc in (int(c),(GH-1-gy)*GW+gx):
        cy,cx=divmod(cc,GW);ox,oy=cell_tile(cc,team)
        for x,y in sorted(((x,y) for y in range(4*cy,4*cy+4) for x in range(3*cx,3*cx+3)),key=lambda p:((p[0]-ox)**2+(p[1]-oy)**2,p[1],p[0])):
            if g._valid_deploy(team,x,y):return x,y
    x,y=FALLBACK[team];return (x,y) if g._valid_deploy(team,x,y) else None


def threat(g,team):
    return any(u.alive and not getattr(u,'is_building',False) and ((u.y<HALF)==(team=='blue')) for u in g.players[opp(team)].troops)


def cost(name):
    return card_info(name)['cost']


class NullReplay:
    # the in-game recorder is not needed here and costs a snapshot every other tick
    snaps=()
    def snap(self,g):
        pass


NULL=NullReplay()


def side_cfg(d):
    return {'deck':list(d['cards']),'evolutions':list(d.get('evo',())),'heroes':list(d.get('hero',())),'king_lvl':LEVEL}


def sample(lp,rng):
    p=np.exp(np.asarray(lp,np.float64));c=np.cumsum(p);return int(min(np.searchsorted(c,rng.random()*c[-1],side='right'),len(p)-1))


class Agent:
    # a train.traceRl.Policy over its own vocabulary and standardisation, sampled at temperature T; act returns the play and the
    # token log-probabilities at T (the stored behaviour probabilities when this agent generated the traces)
    def __init__(self,state,vocab,mu,sd,T=1.0):
        self.vocab=list(vocab);self.idx={c:i for i,c in enumerate(self.vocab)};self.mu=np.asarray(mu,np.float32);self.sd=np.asarray(sd,np.float32);self.T=T
        self.pol=Policy(len(self.mu),len(self.vocab),state['trunk.0.weight'].shape[0]);self.pol.load_state_dict(state);self.pol.eval()
    def S(self,X):
        X=np.asarray(X,np.float32).reshape(-1,FEAT_DIM);return torch.from_numpy(((np.concatenate([X,relative(X)],1)-self.mu)/self.sd).astype(np.float32))
    def act(self,x,menu,rng,g=None,team=None):
        known=[c for c in menu if c in self.idx][:HAND]
        if not known:return None
        H=torch.tensor([self.idx[c] for c in known])
        with torch.no_grad():
            h=self.pol.trunk(self.S(x));pc=F.log_softmax(self.pol.card(h)[0,H]/self.T,0);i=sample(pc.numpy(),rng)
            lx=F.log_softmax(self.pol.cell(torch.cat([h,self.pol.emb(H[i:i+1])],1))[0]/self.T,0);j=sample(lx.numpy(),rng)
        return known[i],j,float(pc[i]),float(lx[j])


class Bot:
    # the fixed rule opponent: a random affordable card; spells on the cell holding most enemy units (at least two, else the weaker
    # enemy princess tower when only spells are affordable), buildings in the middle of its half, troops onto the nearest enemy troop
    # on its half or else at the bridge of the weaker enemy princess tower's lane
    def act(self,x,menu,rng,g=None,team=None):
        foes=[u for u in g.players[opp(team)].troops if u.alive];cnt={}
        for u in foes:c=cell_of(u.x,u.y);cnt[c]=cnt.get(c,0)+1
        hot=max(cnt,key=lambda c:(cnt[c],-c)) if cnt and max(cnt.values())>=2 else None
        spells=[c for c in menu if card_def(c)['kind']=='spell'];rest=[c for c in menu if c not in spells]
        pool=rest+(spells if hot is not None else []) or menu;name=pool[int(rng.integers(len(pool)))];kind=card_def(name)['kind']
        tw=[t for t in g.arena.towers if t.team==opp(team) and t.ttype=='princess' and t.alive];blue=team=='blue'
        lane=min(tw,key=lambda t:(t.hp,t.cx)).cx<9 if tw else True
        if kind=='spell':
            if hot is not None:return name,hot,0.0,0.0
            t=min(tw,key=lambda t:(t.hp,t.cx)) if tw else next(t for t in g.arena.towers if t.team==opp(team) and t.ttype=='king')
            return name,cell_of(t.cx,t.cy),0.0,0.0
        if kind=='building':return name,(2 if blue else 5)*GW+(2 if lane else 3),0.0,0.0
        home=[u for u in foes if not getattr(u,'is_building',False) and (u.y<HALF)==blue]
        if home:
            u=min(home,key=lambda u:abs(u.y-(3.0 if blue else 29.0)));gy=min(int(u.y)//4,3) if blue else max(int(u.y)//4,4)
            return name,gy*GW+min(max(int(u.x)//3,0),GW-1),0.0,0.0
        return name,(3 if blue else 4)*GW+(1 if lane else 4),0.0,0.0


class Match:
    # one game under the world's rules; random is seeded from the game seed and every policy draw comes from a numpy generator, so a
    # game is reproduced by replaying its plays at their times
    def __init__(self,decks,seed):
        random.seed(int(seed));Troop._n=Building._n=0
        self.g=Game(p1=side_cfg(decks[0]),p2=side_cfg(decks[1]));self.g.replay=NULL;self.st={tm:{'thr':0,'last':-9.0} for tm in TEAMS};self.ticks=0;self.n=0
    def copy(self):
        m=copy.copy(self);m.g=clone(self.g);m.g.replay=NULL;m.st=copy.deepcopy(self.st);return m
    def draw(self,tm,rng):
        self.st[tm]['thr']=int(rng.integers(THR[0],THR[1]+1))
    def menu(self,tm):
        g=self.g;p=g.players[tm];h=p.deck.hand if p.deck else []
        if not h or g.t-self.st[tm]['last']<COOL-1e-9:return None
        costs=[cost(c) for c in h]
        if p.elixir<min(costs):return None
        if p.elixir>=self.st[tm]['thr'] or threat(g,tm):return [c for c,k in zip(h,costs) if k<=p.elixir]
        return None
    def play(self,tm,name,c):
        g=self.g;xy=place(g,tm,name,c)
        if xy is None:return False
        ok,_=g.play_card(tm,name,xy[0]+0.5,xy[1]+0.5)
        if ok:self.st[tm]['last']=g.t
        return ok
    def decide(self,tm,agent,rng,menu=None,out=None,horizon=0.0):
        # one decision of team tm by agent; a record (corpusStates layout plus the token log-probabilities) goes to out when given
        g=self.g;menu=menu or self.menu(tm)
        if not menu:return None
        # agents act on the float16 state the store keeps, so stored and recomputed probabilities agree
        x=featurize(g,tm).astype(np.float16);a=agent.act(x.astype(np.float32),menu,rng,g,tm)
        if a is None:return None
        name,c,lpc,lpx=a;p=g.players[tm];k=key(name) or name;evo=bool(k in p.evolutions and p.evolution_ready(k));hero=k in p.heroes
        d=g.players[opp(tm)].deck;oh='|'.join(d.hand) if d else '';on=(d.nxt or '') if d else '';erate=g._erate()
        roll=rollout(g,tm,horizon) if horizon else (0.0,)*len(ROLL);t=g.t
        if not self.play(tm,name,c):return None
        tx,ty=cell_tile(c,tm);self.draw(tm,rng)
        r={'t':t,'team':tm,'card':k,'x':tx+0.5,'y':ty+0.5,'evolved':evo,'hero':hero,'erate':erate,'hand':'|'.join(menu),'state':x,
           'opp_hand':oh,'opp_next':on,'roll':roll,'mu':(lpc,lpx),'idx':self.n}
        self.n+=1
        if out is not None:out.append(r)
        return r
    def tick(self):
        self.g.tick();self.ticks+=1
    def finish(self,agents,rng,record=(),out=None,horizon=0.0):
        # play to the end: agents maps a team to its agent, record the teams whose decisions go to out
        while not self.g.ended:
            for tm in TEAMS:self.decide(tm,agents[tm],rng,out=out if tm in record else None,horizon=horizon if tm in record else 0.0)
            self.tick()
        return self.g.winner


def score(winner,team):
    return 0.5 if winner not in TEAMS else float(winner==team)


class SimWorld:
    # the production world: a deck pool, the match rules above, generation, replay of recorded plays and branching
    def __init__(self,decks):
        self.decks=decks if isinstance(decks,list) else json.loads(Path(decks).read_text())['decks']
    def deal(self,seed):
        i,j=np.random.default_rng([int(seed),7]).integers(len(self.decks),size=2);return self.decks[i],self.decks[j]
    def play(self,seed,agents,record=TEAMS,horizon=0.0,rseed=None):
        m=Match(self.deal(seed),seed);rng=np.random.default_rng([int(seed) if rseed is None else int(rseed),1]);recs=[]
        for tm in TEAMS:m.draw(tm,rng)
        w=m.finish(agents,rng,record,recs,horizon);g=m.g
        dk=[d['cards'] for d in self.deal(seed)]
        return {'winner':w,'end_t':g.t,'bc':g.players['blue'].crowns,'rc':g.players['red'].crowns,'ticks':m.ticks,'decks':dk},recs
    def duel(self,seed,agent,other):
        # one evaluation game: decks dealt by the seed, the agent holding the first deck and playing blue on even seeds; its score
        tm=TEAMS[int(seed)%2];m=Match(self.deal(seed) if tm=='blue' else self.deal(seed)[::-1],seed);rng=np.random.default_rng([int(seed),3])
        for x in TEAMS:m.draw(x,rng)
        return score(m.finish({tm:agent,opp(tm):other},rng),tm),m.ticks
    def replay(self,seed,plays,stop=None):
        # the match after replaying recorded plays (t, team, card, cell) in record order, or at record stop's decision before its play
        m=Match(self.deal(seed),seed);n=len(plays) if stop is None else stop
        for k in range(n):
            t,tm,name,c=plays[k]
            while abs(t-m.g.t)>TOL and not m.g.ended:m.tick()
            if not m.play(tm,name,c):raise RuntimeError(f'replayed play {k} of game {seed} failed')
            m.n+=1
        if stop is not None and stop<len(plays):
            while abs(plays[stop][0]-m.g.t)>TOL and not m.g.ended:m.tick()
        return m
    def branch(self,seed,plays,stops,actor,opponent,G,rseed):
        # GRPO groups: for each stop, G continuations from the recorded state; the actor (the team of that record) plays its current
        # menu now and every later decision with actor, the opponent with opponent, both with fresh thresholds; returns per stop the
        # group's (actor records, outcome) and the ticks spent (replay and continuations)
        m=Match(self.deal(seed),seed);k=0;out=[]
        for s in sorted(stops):
            while k<s:
                while abs(plays[k][0]-m.g.t)>TOL and not m.g.ended:m.tick()
                t,tm,name,c=plays[k]
                if not m.play(tm,name,c):raise RuntimeError(f'replayed play {k} of game {seed} failed')
                m.n+=1;k+=1
            while abs(plays[s][0]-m.g.t)>TOL and not m.g.ended:m.tick()
            tm=plays[s][1];group=[]
            for j in range(G):
                h=m.copy();h.ticks=0;rng=np.random.default_rng([int(rseed),s,j]);random.seed(int(rseed)*100003+s*1009+j);recs=[]
                menu=h.menu(tm) or [c for c in h.g.players[tm].deck.hand if cost(c)<=h.g.players[tm].elixir]
                for x in TEAMS:h.draw(x,rng)
                h.decide(tm,actor,rng,menu=menu,out=recs);w=h.finish({tm:actor,opp(tm):opponent},rng,(tm,),recs)
                group.append((recs,score(w,tm),h.ticks))
            out.append((s,tm,group))
        return out,m.ticks+sum(x[2] for _,_,gr in out for x in gr)


# ---- generation

def days(n_warm,n_stream,n_held):
    # battle times by generation index: warm-up over 30 August to 1 September, the stream over 2 to 5 September, held-out 6 September
    w=T0+3*DAY*(np.arange(n_warm)+0.5)/max(n_warm,1);s=T0+3*DAY+4*DAY*(np.arange(n_stream)+0.5)/max(n_stream,1)
    return np.concatenate([w,s,T0+7*DAY+DAY*(np.arange(n_held)+0.5)/max(n_held,1)])


_W={}


def _init(world,agents,threads=1):
    torch.set_num_threads(threads);_W['world']=world;_W['agents']=agents;_W['cache']={}


class Workers:
    # a pool of jobs workers initialised with init, or this process when jobs is 0 (tests)
    def __init__(self,jobs,init):
        self.pool=Pool(jobs,initializer=_init,initargs=init) if jobs else None
        if not jobs:_init(*init)
    def map(self,fn,tasks,ordered=True):
        return map(fn,tasks) if self.pool is None else (self.pool.imap if ordered else self.pool.imap_unordered)(fn,tasks)
    def __enter__(self):
        return self
    def __exit__(self,*a):
        if self.pool is not None:self.pool.terminate()


def agent_of(spec):
    if spec not in _W['cache']:_W['cache'][spec]=load_agent(spec)
    return _W['cache'][spec]


def _gen(args):
    seed,horizon=args;a=agent_of(_W['agents']['behaviour']);t0=time.monotonic()
    out,recs=_W['world'].play(seed,{'blue':a,'red':a},TEAMS,horizon);out['seconds']=time.monotonic()-t0;return seed,out,recs


def write_shard(path,games,recs):
    # a train.corpusStates shard: npz with the record columns, states, rollout features and the stored behaviour log-probabilities,
    # and the json sidecar of per-game outcomes
    X=np.stack([r['state'] for r in recs]) if recs else np.zeros((0,FEAT_DIM),np.float16)
    cols={c:[r[k] for r in recs] for c,k in zip(COLS+EXTRA,('bid','idx','t','team','card','x','y','evolved','hero','erate','hand','opp_hand','opp_next'))}
    assert set(cols)==set(COLS+EXTRA)
    roll=np.array([r['roll'] for r in recs],np.float32).reshape(-1,len(ROLL));mu=np.array([r['mu'] for r in recs],np.float32).reshape(-1,2)
    np.savez_compressed(path,X=X,roll=roll,mu=mu,**{c:np.array(v) for c,v in cols.items()})
    Path(path).with_suffix('.json').write_text(json.dumps({'games':len(games),'records':len(recs),'outcomes':games})+'\n')


def generate(world,behaviour,out,n_warm,n_stream,n_held,jobs=2,per=2000,roll=(0,0),horizon=10.0,offset=0,log=print):
    # self-play traces of the behaviour, shard by shard in generation order; games with index in [roll[0], roll[1]) get the no-play
    # rollout features (the simulator critic's input); existing shards are kept, so a killed run resumes
    out=Path(out);out.mkdir(parents=True,exist_ok=True);n=n_warm+n_stream+n_held;ts=days(n_warm,n_stream,n_held);t0=time.monotonic()
    (out/'generation.json').write_text(json.dumps({'warm':n_warm,'stream':n_stream,'held':n_held,'per':per,'roll':list(roll),'horizon':horizon,'offset':offset,
                                                   'behaviour':behaviour,'decks':len(world.decks),'rules':{'cool':COOL,'thr':THR,'level':LEVEL}})+'\n')
    todo=[k for k in range(0,n,per) if not ((out/f'shard{k//per:03d}.npz').exists() and (out/f'shard{k//per:03d}.json').exists())]
    tasks=[(offset+i,horizon if roll[0]<=i<roll[1] else 0.0) for k in todo for i in range(k,min(k+per,n))]
    with Workers(jobs,(world,{'behaviour':behaviour})) as W:
        it=W.map(_gen,tasks)
        for k in todo:
            p=out/f'shard{k//per:03d}.npz';games=[];recs=[];sec=0.0
            for seed,o,rs in (next(it) for _ in range(k,min(k+per,n))):
                i=seed-offset;bid=f'S{seed:08d}';w=o['winner']
                for r in rs:r['bid']=bid
                recs+=rs;sec+=o['seconds']
                games.append({'bid':bid,'actual_winner':w,'sim_winner':w,'end_t':o['end_t'],'premature':False,'actual_bc':o['bc'],'actual_rc':o['rc'],
                              'sim_bc':o['bc'],'sim_rc':o['rc'],'n':len(rs),'sim_blue':0.0,'sim_red':0.0,'ts':float(ts[i]),'mode':'sim',
                              'rolled':bool(roll[0]<=i<roll[1] and horizon),'ticks':o['ticks'],'seconds':round(o['seconds'],3),'decks':o['decks']})
            write_shard(p,games,recs)
            log(json.dumps({'shard':k//per,'games':len(games),'records':len(recs),'cpu_s':round(sec,1),'wall_s':round(time.monotonic()-t0,1),
                            'draws':sum(g['actual_winner'] is None for g in games),'blue':sum(g['actual_winner']=='blue' for g in games)}),flush=True)


def trace_ticks(states,pack,split='stream'):
    # simulated ticks the traces of a split cost to generate (draws included: they were simulated too)
    from train.singleTraj import SPLITS,utc
    meta=json.loads((Path(states)/'generation.json').read_text());ts=days(meta['warm'],meta['stream'],meta['held']);w=utc('2026-09-02');h=utc('2026-09-06');tot=0
    for p in sorted(Path(states).glob('shard*.json')):
        for g in json.loads(p.read_text())['outcomes']:
            s=0 if g['ts']<w else 1 if g['ts']<h else 2
            if s==min(SPLITS.index(split),2):tot+=g['ticks']
    return tot,len(ts)


def stored(states,pack,prep_dir=None):
    # the behaviour's stored token log-probabilities aligned with the packed store, saved as the prep's mu_true_1.npy; every store
    # record must be the shard record it came from (same card and cell)
    from train.singleTraj import Corpus
    c=Corpus(pack);A=c.a;gid={b:i for i,b in enumerate(c.meta['bid'])};mu=np.full((len(A['y']),2),np.nan,np.float32);vi=c.vocab
    for p in sorted(Path(states).glob('shard*.npz')):
        d=np.load(p);bid=d['bid'].astype(str);card=d['card'];cell=np.array([cell_of(x,y) for x,y in zip(d['x'],d['y'])]);m=d['mu'];start={}
        for i,b in enumerate(bid):start.setdefault(b,i)
        for b,s in start.items():
            g=gid.get(b)
            if g is None:continue
            rows=np.arange(c.gr[g],c.gr[g+1]);src=s+A['idx'][rows].astype(np.int64)
            if not all(bid[src]==b) or any(vi[A['card'][r]]!=card[q] for r,q in zip(rows,src)) or (A['cell'][rows]!=cell[src]).any():
                raise RuntimeError(f'record mismatch in game {b}')
            mu[rows]=m[src]
    if np.isnan(mu).any():raise RuntimeError('store records without a stored probability')
    out=Path(prep_dir or Path(pack)/'prep');out.mkdir(parents=True,exist_ok=True);np.save(out/'mu_true_1.npy',mu);return mu


# ---- agents

def warm_stats(states,warm_end='2026-09-02'):
    # train.singleTraj.pack's standardisation statistics (the warm-up games' records, drawn games dropped) from the corpus shards
    from train.corpusStates import load as load_shard
    from train.singleTraj import utc
    w=utc(warm_end);s1=s2=None;n=0
    for p in sorted(Path(states).glob('shard*.npz')):
        side=json.loads(p.with_suffix('.json').read_text());ok={g['bid'] for g in side['outcomes'] if g['actual_winner'] is not None and g['ts']<w}
        if not ok:continue
        cols,X,_,_=load_shard(p);keep=np.isin(cols['bid'].astype(str),list(ok))
        if not keep.any():continue
        Xk=X[keep].astype(np.float32);S=np.concatenate([Xk.astype(np.float64),relative(Xk)],1);n+=len(S)
        s1=S.sum(0) if s1 is None else s1+S.sum(0);s2=(S**2).sum(0) if s2 is None else s2+(S**2).sum(0)
    mu=s1/n;return mu.astype(np.float32),(np.sqrt(np.maximum(s2/n-mu**2,0))+1e-6).astype(np.float32),n


def export_behaviour(prep_dir,meta,states,out):
    # the corpus clone with the vocabulary and standardisation it was trained with, as one file the workers load
    mu,sd,n=warm_stats(states);vocab=json.loads(Path(meta).read_text())['vocab']
    torch.save({'state':torch.load(Path(prep_dir)/'bc_warm.pt'),'vocab':vocab,'mu':mu,'sd':sd,'hidden':128,'warm_records':n},out);return n


def load_agent(spec):
    # 'bot'; 'file.pt[@T]' (an export_behaviour file); 'pack=DIR;file.pt[#key][@T]' (a policy trained on a packed store: a state dict
    # or a dict of them by arm spec, with that store's vocabulary and standardisation)
    if spec=='bot':return Bot()
    if spec.startswith('py:'):
        import importlib
        mod,_,attr=spec[3:].partition(':');return getattr(importlib.import_module(mod),attr)()
    T=1.0
    if '@' in spec.rsplit('/',1)[-1]:spec,_,t=spec.rpartition('@');T=float(t)
    if spec.startswith('pack='):
        pk,_,f=spec[5:].partition(';');f,_,k=f.partition('#');sd=torch.load(f,weights_only=False);sd=sd[k] if k else sd
        meta=json.loads((Path(pk)/'meta.json').read_text());a=np.load(Path(pk)/'rec.npz');return Agent(sd,meta['vocab'],a['mu'],a['sd'],T)
    d=torch.load(spec,weights_only=False);return Agent(d['state'],d['vocab'],d['mu'],d['sd'],T)


def build_decks(dataset,vocab,out,n=400,modes=('Ranked','Ladder')):
    # the most played complete recorded decks whose cards the behaviour knows and the simulator builds (no Mirror: its cost depends on
    # the previous play)
    import pyarrow.parquet as pq
    from sim.replay import norm
    cols=[f'{p}_card_{i}' for p in ('team','opp') for i in range(8)]+['gameMode_name']
    t=pq.read_table(Path(dataset)/'battles.parquet',columns=cols).to_pydict();cnt={};known=set(vocab)
    for r in range(len(t['gameMode_name'])):
        if t['gameMode_name'][r] not in modes:continue
        for p in ('team','opp'):
            d=[norm(t[f'{p}_card_{i}'][r] or '') for i in range(8)]
            if all(x[0] for x in d) and len({x[0] for x in d})==8:k=tuple(sorted(d));cnt[k]=cnt.get(k,0)+1
    keep=[]
    for k,v in sorted(cnt.items(),key=lambda kv:(-kv[1],kv[0])):
        d={'cards':[x[0] for x in k],'evo':[x[0] for x in k if x[1]],'hero':[x[0] for x in k if x[2]],'count':v}
        if 'mirror' in d['cards'] or not set(d['cards'])<=known:continue
        try:Game(p1=side_cfg(d),p2=side_cfg(d))
        except (AssertionError,KeyError,TypeError,ValueError):continue
        keep.append(d)
        if len(keep)>=n:break
    Path(out).write_text(json.dumps({'decks':keep,'modes':list(modes),'candidates':len(cnt)})+'\n');return keep


# ---- environment-replay GRPO

def _branch(args):
    seed,plays,stops,state,G,rseed=args;w=_W['world'];actor=_W['cache'].get('actor')
    if actor is None:actor=_W['cache']['actor']=load_agent(_W['agents']['actor'])
    actor.pol.load_state_dict(state);opponent=agent_of(_W['agents']['behaviour']);t0=time.monotonic()
    out,ticks=w.branch(seed,plays,stops,actor,opponent,G,rseed);vi=actor.idx;res=[]
    for s,tm,group in out:
        for recs,R,_ in group:
            X=np.stack([r['state'] for r in recs]) if recs else np.zeros((0,FEAT_DIM),np.float16)
            # the menu as the store writes it: the played card first
            H=np.array([([vi[r['card']]]+[vi[c] for c in r['hand'].split('|') if c in vi and c!=r['card']]+[-1]*HAND)[:HAND] for r in recs],np.int64)
            H=H.reshape(-1,HAND)
            res.append((s,X,H,np.array([vi[r['card']] for r in recs],np.int64),np.array([cell_of(r['x'],r['y']) for r in recs],np.int64),
                        np.array([r['mu'] for r in recs],np.float32).reshape(-1,2),R))
    return seed,res,ticks,time.monotonic()-t0


def game_plays(c,g):
    # a packed game's recorded plays in record order: (time, team, card, cell)
    A=c.a;rows=np.arange(c.gr[g],c.gr[g+1]);rows=rows[np.argsort(A['idx'][rows],kind='stable')]
    return [(float(A['t'][r]),'red' if A['team'][r] else 'blue',c.vocab[A['card'][r]],int(A['cell'][r])) for r in rows]


def grpo_loss(pol,S,H,card,cell,old,tr,A,clip=0.2):
    # GRPO's clipped surrogate per token (both tokens of a decision), mean over each continuation's tokens, then over continuations
    from train.singleTraj import token_logp
    lp=token_logp(pol,S,H,card,cell);r=(lp-old).exp();a=A[tr][:,None];n=len(A)
    tok=-torch.min(r*a,r.clamp(1-clip,1+clip)*a).sum(1);cnt=torch.zeros(n).index_add(0,tr,torch.full((len(tr),),2.0))
    return (torch.zeros(n).index_add(0,tr,tok)/cnt.clamp(min=1)).mean()


def group_adv(R,gi,eps=1e-6):
    # the outcome minus its group's mean over the group's standard deviation; a group with one outcome gets zero
    R=torch.as_tensor(R,dtype=torch.float32);gi=torch.as_tensor(gi);n=int(gi.max())+1;cnt=torch.zeros(n).index_add(0,gi,torch.ones(len(R)))
    m=torch.zeros(n).index_add(0,gi,R)/cnt;v=torch.zeros(n).index_add(0,gi,(R-m[gi])**2)/(cnt-1).clamp(min=1)
    return (R-m[gi])/(v.sqrt()[gi]+eps)


def grpo(pack,world,behaviour,out,budget,G=8,batch_prompts=64,lr=1e-4,steps=4,jobs=2,seed=0,marks=(0.125,0.25,0.5,1.0),hidden=128,log=print,prep_dir=None):
    # environment-replay GRPO from the stream's recorded states: each batch takes one decision of each of batch_prompts stream games
    # (games in a seeded order, decision uniform), branches G continuations per prompt in the world, and makes steps clipped updates
    # against the sampling policy; it stops when the simulated ticks reach budget and saves the policy at the given fractions of it
    from train.singleTraj import Corpus
    c=Corpus(pack);P=Path(prep_dir or Path(pack)/'prep');rng=np.random.default_rng(seed);torch.manual_seed(seed);t0=time.monotonic()
    pol=Policy(c.n_state,c.n_card,hidden);pol.load_state_dict(torch.load(P/'bc_warm.pt'));opt=torch.optim.Adam(pol.parameters(),lr=lr)
    stream=c.games('stream');order=stream[rng.permutation(len(stream))];actor=f'pack={pack};{P/"bc_warm.pt"}'
    mu=torch.from_numpy(c.mu);sd=torch.from_numpy(c.sd);spent=0;nb=0;hist=[];saved={};todo=sorted(marks);gi=0;rep={'budget':int(budget),'G':G,'batch_prompts':batch_prompts,'lr':lr}
    with Workers(jobs,(world,{'behaviour':behaviour,'actor':actor})) as W:
        while spent<budget and gi<len(order):
            gs=order[gi:gi+batch_prompts];gi+=len(gs);state={k:v.detach().clone() for k,v in pol.state_dict().items()};tasks=[]
            for g in gs:
                plays=game_plays(c,g);s=int(rng.integers(len(plays)));tasks.append((int(c.meta['bid'][g][1:]),plays,[s],state,G,int(rng.integers(2**31))))
            Xs=[];Hs=[];cs=[];xs=[];olds=[];tr=[];R=[];grp=[];cpu=0.0;ng=0
            for _,res,ticks,sec in W.map(_branch,tasks,False):
                spent+=ticks;cpu+=sec
                for s,X,H,cd,cl,old,r in res:
                    if len(X):Xs.append(X);Hs.append(H);cs.append(cd);xs.append(cl);olds.append(old);tr.append(np.full(len(X),len(R)))
                    R.append(r);grp.append(ng)
                    if len(R)%G==0:ng+=1
            X=np.concatenate(Xs).astype(np.float32);S=((torch.from_numpy(np.concatenate([X,relative(X)],1))-mu)/sd).float()
            A=group_adv(R,grp);tri=torch.from_numpy(np.concatenate(tr));pol.train();H=torch.from_numpy(np.concatenate(Hs));cd=torch.from_numpy(np.concatenate(cs))
            cl=torch.from_numpy(np.concatenate(xs));old=torch.from_numpy(np.concatenate(olds))
            for _ in range(steps):
                loss=grpo_loss(pol,S,H,cd,cl,old,tri,A);opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(pol.parameters(),1.0);opt.step()
            nb+=1
            Rt=torch.tensor(R).view(-1,G);row={'batch':nb,'ticks':int(spent),'mean_R':round(float(Rt.mean()),4),'mixed_groups':round(float((Rt.std(1)>0).float().mean()),4),
                                               'tokens':int(len(X)),'cpu_s':round(cpu,1),'wall_s':round(time.monotonic()-t0,1)}
            hist.append(row)
            if nb%10==0:log(json.dumps(row),flush=True)
            while todo and spent>=todo[0]*budget:
                f=todo.pop(0);saved[f'grpo_{f:g}']={k:v.detach().clone() for k,v in pol.state_dict().items()}
                rep.setdefault('marks',{})[f'{f:g}']={'batch':nb,'ticks':int(spent)}
    if todo:saved[f'grpo_{todo[-1]:g}']=pol.state_dict()
    rep.update({'batches':nb,'steps':steps,'ticks':int(spent),'prompts':int(gi),'seconds':round(time.monotonic()-t0,1),'history':hist})
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(rep)+'\n');torch.save(saved,Path(out).with_suffix('.pt'))
    return rep,saved


# ---- true win rates

def _wins(args):
    spec,ospec,seeds=args;w=_W['world'];a=agent_of(spec);o=agent_of(ospec);res=[];ticks=0;t0=time.monotonic()
    for s in seeds:r,tk=w.duel(s,a,o);res.append(r);ticks+=tk
    return spec,ospec,seeds,res,ticks,time.monotonic()-t0


def wins(world,agents,opponents,n,jobs=2,seed0=10**7,chunk=50,out=None,log=print,every=400):
    # every agent against every opponent over the same n games (game s: decks dealt by seed s, the agent holding the first deck and
    # playing blue on even s, red on odd s, the sim's random seeded by s), so results pair across agents; agents and opponents map a
    # name to a load_agent spec; finished games go to out.partial every that many tasks, and a rerun skips the games found there
    seeds=list(range(seed0,seed0+n));inv={v:k for k,v in agents.items()};oinv={v:k for k,v in opponents.items()};res={};ticks=0;t0=time.monotonic();done=0
    part=Path(str(out)+'.partial') if out else None
    if part and part.exists():
        res={o:{a:{int(s):r for s,r in d.items()} for a,d in v.items()} for o,v in json.loads(part.read_text())['res'].items()}
    have=lambda a,o,ss:all(s in res.get(oinv[o],{}).get(inv[a],{}) for s in ss)
    tasks=[(a,o,seeds[i:i+chunk]) for a in agents.values() for o in opponents.values() for i in range(0,n,chunk) if not have(a,o,seeds[i:i+chunk])]
    with Workers(jobs,(world,{})) as W:
        for a,o,ss,r,tk,_ in W.map(_wins,tasks,False):
            d=res.setdefault(oinv[o],{}).setdefault(inv[a],{});d.update(zip(ss,r));ticks+=tk;done+=1
            if done%200==0:log(json.dumps({'tasks':done,'of':len(tasks),'wall_s':round(time.monotonic()-t0,1)}),flush=True)
            if part and done%every==0:part.write_text(json.dumps({'res':res})+'\n')
    outcomes={o:{a:[d[s] for s in seeds] for a,d in v.items() if a in agents} for o,v in res.items() if o in opponents}
    rep={'n':n,'seed0':seed0,'agents':agents,'opponents':opponents,'ticks':int(ticks),'seconds':round(time.monotonic()-t0,1),'outcomes':outcomes}
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(rep)+'\n')
    return rep


def paired(x,y=None,z=1.96):
    # mean and normal 95% interval of x, or of x - y paired by game
    d=np.asarray(x,np.float64)-(0 if y is None else np.asarray(y,np.float64));m=float(d.mean());h=z*float(d.std(ddof=1))/np.sqrt(len(d))
    return m,m-h,m+h


def rank_corr(a,b):
    # Spearman's correlation (average ranks for ties) and Kendall's tau-b
    a=np.asarray(a,np.float64);b=np.asarray(b,np.float64)
    def rk(v):
        o=np.argsort(v,kind='stable');r=np.empty(len(v));r[o]=np.arange(len(v));u,inv=np.unique(v,return_inverse=True)
        return np.array([r[inv==i].mean() for i in range(len(u))])[inv]
    ra,rb=rk(a),rk(b);sp=float(np.corrcoef(ra,rb)[0,1]) if ra.std()>0 and rb.std()>0 else 0.0
    s=np.sign(a[:,None]-a[None,:]);t=np.sign(b[:,None]-b[None,:]);num=(s*t).sum();den=np.sqrt((s*s).sum()*(t*t).sum())
    return sp,float(num/den) if den>0 else 0.0


def proxies(pack,pt,out=None,prep_dir=None):
    # the real rounds' held-out measures (train.singleTraj.evaluate on half B with the half-A Q model and both support masks) of saved
    # policies (a .pt of state dicts by name), beside the clone's, in run_arms's report layout
    from train.singleTraj import Corpus,Head,eval_set,evaluate,support
    c=Corpus(pack);P=Path(prep_dir or Path(pack)/'prep');bc=Policy(c.n_state,c.n_card,128);bc.load_state_dict(torch.load(P/'bc_warm.pt'));bc.eval()
    Q=Head(c.n_state,c.n_card,hidden=128);Q.load_state_dict(torch.load(P/'q.pt'));Q.eval();E=torch.load(P/'evals.pt',weights_only=False)
    ev=eval_set(c,E['rows']['heldB']);strict=support(c,np.concatenate([c.records(c.games('warm')),c.records(c.games('stream'))]),100)
    rep={'arms':{},'train_stats':{},'bc':evaluate(bc,bc,Q,ev,E['sup']['final'],strict=strict)}
    for k,sd in torch.load(pt,weights_only=False).items():
        q=Policy(c.n_state,c.n_card,sd['trunk.0.weight'].shape[0]);q.load_state_dict(sd);q.eval();rep['arms'][k]=evaluate(q,bc,Q,ev,E['sup']['final'],strict=strict)
    if out:Path(out).write_text(json.dumps(rep,indent=1)+'\n')
    return rep


def proxy_row(m,b):
    # changes against the clone on held-out B, as the real rounds report them
    return {'dq_pess':m['q_pess']-b['q_pess'],'dq_support':m['q_support']-b['q_support'],'dq_direct':m['q_direct']-b['q_direct'],
            'd_winner_gap':m['winner_gap']-b['winner_gap'],'kl':m['kl_to_bc'],'entropy':m['entropy'],'d_support':m['support_mass']-b['support_mass'],
            'd_support100':(m.get('support_mass_strict') or 0)-(b.get('support_mass_strict') or 0),'top_q':m.get('q_top_mass')}


def ratio(x,y,base,z=1.96):
    # mean(x - base) / mean(y - base) paired by game, with a delta-method interval
    a=np.asarray(x,np.float64)-np.asarray(base,np.float64);g=np.asarray(y,np.float64)-np.asarray(base,np.float64);n=len(a);r=a.mean()/g.mean()
    v=(a.var(ddof=1)+r*r*g.var(ddof=1)-2*r*np.cov(a,g)[0,1])/(n*g.mean()**2);h=z*float(np.sqrt(max(v,0)));return float(r),float(r-h),float(r+h)


PAIRS=(('cross/flash','true/flash:mu=true'),('cross/flash_nogate','true/flash_nogate:mu=true'),('cross/sao','true/sao:mu=true'),('cross/bpco','true/bpco:mu=true'))
VOLUME=(('bpco',('vol6k/bpco','vol20k/bpco','cross/bpco')),('sao',('vol6k/sao','vol20k/sao','cross/sao')),
        ('flash:mu=true',('vol6k/flash:mu=true','vol20k/flash:mu=true','true/flash:mu=true')))


def report(runs,wins_files,out=None,ref='bc',top='grpo/grpo_1',boots=1000,seed=0):
    # true win rates against every opponent beside the proxies of the registered plan (f5369): verdicts, rank agreement (with a game
    # bootstrap of Spearman's correlation over the games every policy played), recovery of the GRPO gain, stored against estimated
    # behaviour probabilities and trace volume; agent names stem/spec take their proxies from runs/stem.json
    reps={Path(p).stem:json.loads(Path(p).read_text()) for p in runs};W={}
    for f in wins_files:
        for o,v in json.loads(Path(f).read_text())['outcomes'].items():
            for a,x in v.items():
                if len(x)>len(W.setdefault(o,{}).get(a,[])):W[o][a]=x
    opps=list(W);rows={}
    for a in W[opps[0]]:
        r={'true':{}}
        for o in opps:
            x=W[o][a];b=W[o][ref];n=min(len(x),len(b));r['true'][o]={'n':len(x),'wr':paired(x),'dwr':paired(x[:n],b[:n])}
        if '/' in a:
            stem,spec=a.split('/',1)
            if stem in reps and spec in reps[stem]['arms']:r['proxy']=proxy_row(reps[stem]['arms'][spec],reps[stem]['bc'])
        rows[a]=r
    trained=[a for a,r in rows.items() if 'proxy' in r]
    F=max([0.0005]+[rows[a]['proxy']['dq_pess'] for a in trained if a.endswith(':flip')])
    for a in trained:
        p=rows[a]['proxy'];p['verdict']='improves' if p['dq_pess']>2*F and p['d_support']>=-0.01 and p['dq_direct']-p['dq_pess']<=0.002 else 'no'
    for a,r in rows.items():
        for o,t in r['true'].items():t['verdict']='improves' if t['dwr'][1]>0 else 'hurts' if t['dwr'][2]<0 else 'no change'
    rng=np.random.default_rng(seed);rank={}
    for o in opps:
        n=min(len(W[o][a]) for a in trained+[ref]);X=np.array([W[o][a][:n] for a in trained],np.float64)-np.array(W[o][ref][:n],np.float64)
        idx=rng.integers(n,size=(boots,n));D=np.array([X[:,i].mean(1) for i in idx]);dm=np.array([rows[a]['true'][o]['dwr'][0] for a in trained])
        best=int(np.argmax(dm));lo=rows[trained[best]]['true'][o]['dwr'][1]
        for k in ('dq_pess','dq_support','d_winner_gap'):
            pv=np.array([rows[a]['proxy'][k] for a in trained]);sp,tau=rank_corr(pv,dm);bs=np.array([rank_corr(pv,d)[0] for d in D])
            ptop=trained[int(np.argmax(pv))]
            rank[f'{o}|{k}']={'spearman':sp,'kendall':tau,'spearman_boot':[float(np.percentile(bs,2.5)),float(np.percentile(bs,97.5))],
                              'proxy_top':ptop,'true_top':trained[best],'ranks_correctly':bool(sp>=0.7 and rows[ptop]['true'][o]['dwr'][0]>=lo),
                              'policies':len(trained),'games':n}
        rank[f'{o}|agreement']=sum((rows[a]['proxy']['verdict']=='improves')==(rows[a]['true'][o]['verdict']=='improves') for a in trained)
    rec={}
    if top in rows:
        for o in opps:
            g=W[o][top];b=W[o][ref]
            if rows[top]['true'][o]['verdict']!='improves':continue
            for a in trained:
                if a==top:continue
                x=W[o][a];n=min(len(x),len(g),len(b));rec[f'{o}|{a}']=ratio(x[:n],g[:n],b[:n])
    mu={f'{o}|{c}->{t}':paired(W[o][t][:min(len(W[o][t]),len(W[o][c]))],W[o][c][:min(len(W[o][t]),len(W[o][c]))]) for c,t in PAIRS for o in opps
        if c in W[o] and t in W[o]}
    vol={f'{o}|{k}':[rows[a]['true'][o]['dwr'] if a in rows else None for a in names] for k,names in VOLUME for o in opps}
    rep={'F':F,'rows':rows,'rank':rank,'recovery':rec,'stored_vs_estimated':mu,'volume':vol,'opponents':opps,'reference':ref}
    if out:Path(out).write_text(json.dumps(rep,indent=1)+'\n')
    return rep


def markdown(rep):
    # the report's main table and summaries as markdown
    o0,o1=(rep['opponents']+[None])[:2];f=lambda v,d=4:'' if v is None else f'{v:+.{d}f}';ci=lambda t:f"{t[0]:+.3f} [{t[1]:+.3f}, {t[2]:+.3f}]"
    out=[f"F (largest flipped dq_pess, min .0005) = {rep['F']:.4f}",'',
         '| policy | n | WR vs '+o0+' | dWR vs '+o0+' | true | '+(f'WR vs {o1} | dWR vs {o1} | ' if o1 else '')
         +'dq pess | dq support | d winner gap | KL | entropy | d support | proxy |',
         '|'+'---|'*(13 if o1 else 10)]
    for a,r in rep['rows'].items():
        t=r['true'][o0];p=r.get('proxy',{});u=r['true'].get(o1) if o1 else None
        cells=[a,str(t['n']),f"{t['wr'][0]:.3f}",ci(t['dwr']),t['verdict']]+([f"{u['wr'][0]:.3f}",ci(u['dwr'])] if u else [])
        cells+=[f(p.get('dq_pess')),f(p.get('dq_support')),f(p.get('d_winner_gap'),3),'' if 'kl' not in p else f"{p['kl']:.3f}",
                '' if 'entropy' not in p else f"{p['entropy']:.3f}",f(p.get('d_support')),p.get('verdict','')]
        out.append('| '+' | '.join(cells)+' |')
    out+=['',"Rank agreement (Spearman over each policy's full evaluation, its 95% game bootstrap over the games every policy played, Kendall; "
          'ranks correctly by the registered rule):']
    for k,v in rep['rank'].items():
        out.append(f'- {k}: {v}' if not isinstance(v,dict) else f"- {k}: {v['spearman']:+.3f}, bootstrap over {v['games']} games "
                   f"[{v['spearman_boot'][0]:+.3f}, {v['spearman_boot'][1]:+.3f}], tau {v['kendall']:+.3f}; proxy top {v['proxy_top']}, "
                   f"true top {v['true_top']}; ranks correctly {v['ranks_correctly']}")
    out+=['','Recovery of the GRPO gain (ratio [delta 95%]):']+[f'- {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]' for k,v in rep['recovery'].items()]
    out+=['','Stored minus estimated behaviour probabilities (paired dWR):']+[f'- {k}: {ci(v)}' for k,v in rep['stored_vs_estimated'].items()]
    out+=['','Volume (dWR at 6k / 20k / 60k stream games):']+[f"- {k}: "+' / '.join('' if v is None else ci(v) for v in vs) for k,vs in rep['volume'].items()]
    return '\n'.join(out)


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('decks');a.add_argument('--dataset',required=True);a.add_argument('--behaviour',required=True);a.add_argument('--out',required=True)
    a.add_argument('--n',type=int,default=400)
    a=sp.add_parser('behaviour');a.add_argument('--prep',required=True);a.add_argument('--meta',required=True);a.add_argument('--states',required=True)
    a.add_argument('--out',required=True)
    a=sp.add_parser('gen');a.add_argument('--decks',required=True);a.add_argument('--behaviour',required=True);a.add_argument('--out',required=True)
    for k in ('warm','stream','held'):a.add_argument(f'--{k}',type=int,required=True)
    a.add_argument('--jobs',type=int,default=2);a.add_argument('--per',type=int,default=2000);a.add_argument('--roll',type=int,nargs=2,default=[0,0])
    a.add_argument('--horizon',type=float,default=10.0);a.add_argument('--offset',type=int,default=0)
    a=sp.add_parser('stored');a.add_argument('--states',required=True);a.add_argument('--pack',required=True)
    a=sp.add_parser('prep');a.add_argument('--states',required=True);a.add_argument('--pack',required=True);a.add_argument('--threads',type=int,default=16)
    a.add_argument('--rolled_pre',type=int,default=4000)
    a=sp.add_parser('grpo');a.add_argument('--pack',required=True);a.add_argument('--decks',required=True);a.add_argument('--behaviour',required=True)
    a.add_argument('--out',required=True);a.add_argument('--states',required=True)
    a.add_argument('--frac',type=float,default=1.0,help='budget as a fraction of the ticks the stream traces cost')
    a.add_argument('--G',type=int,default=8)
    a.add_argument('--batch',type=int,default=64);a.add_argument('--jobs',type=int,default=2);a.add_argument('--seed',type=int,default=0)
    a.add_argument('--steps',type=int,default=4)
    a=sp.add_parser('wins');a.add_argument('--decks',required=True);a.add_argument('--agents',required=True,help='json file: name -> agent spec')
    a.add_argument('--opponents',required=True,help='json file: name -> agent spec');a.add_argument('--n',type=int,required=True)
    a.add_argument('--jobs',type=int,default=2);a.add_argument('--out',required=True);a.add_argument('--seed0',type=int,default=10**7)
    a=sp.add_parser('proxies');a.add_argument('--pack',required=True);a.add_argument('--pt',required=True);a.add_argument('--out',required=True)
    a=sp.add_parser('report');a.add_argument('--runs',nargs='+',required=True);a.add_argument('--wins',nargs='+',required=True);a.add_argument('--out',required=True)
    a=ap.parse_args()
    if a.cmd=='decks':print(len(build_decks(a.dataset,torch.load(a.behaviour,weights_only=False)['vocab'],a.out,a.n)))
    elif a.cmd=='behaviour':print(export_behaviour(a.prep,a.meta,a.states,a.out))
    elif a.cmd=='gen':generate(SimWorld(a.decks),a.behaviour,a.out,a.warm,a.stream,a.held,a.jobs,a.per,tuple(a.roll),a.horizon,a.offset)
    elif a.cmd=='stored':stored(a.states,a.pack)
    elif a.cmd=='prep':
        from train.singleTraj import prep
        prep(a.pack,threads=a.threads,rolled_pre=a.rolled_pre);stored(a.states,a.pack)
    elif a.cmd=='grpo':grpo(a.pack,SimWorld(a.decks),a.behaviour,a.out,a.frac*trace_ticks(a.states,a.pack)[0],a.G,a.batch,steps=a.steps,jobs=a.jobs,seed=a.seed)
    elif a.cmd=='wins':wins(SimWorld(a.decks),json.loads(Path(a.agents).read_text()),json.loads(Path(a.opponents).read_text()),a.n,a.jobs,a.seed0,out=a.out)
    elif a.cmd=='proxies':torch.set_num_threads(8);proxies(a.pack,a.pt,a.out)
    else:
        r=report(a.runs,a.wins,a.out);md=markdown(r);Path(a.out).with_suffix('.md').write_text(md+'\n');print(md)


if __name__=='__main__':main()
