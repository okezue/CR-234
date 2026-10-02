import json
import math

import numpy as np
import torch

import train.simTruth as STr
from sim.game import Game
from train.counterfactual import GH,GW,cell_of
from train.feats import FEAT_DIM,featurize
from train.simTruth import (TEAMS,Agent,SimWorld,cell_tile,generate,group_adv,grpo,grpo_loss,load_agent,paired,place,rank_corr,stored,
                            trace_ticks,wins)
from train.singleTraj import Corpus,Policy,pack,prep,run_arms,token_logp

# The simulator as a known world. Two kinds of test: the real world's rules (cells correspond across sides and land on deployable
# tiles, a game is reproduced by replaying its plays, a GRPO group starts from the recorded state) on short sim.game games, and a
# planted world in which the best policy is known (the right card is written in the state; the winner is drawn from the two sides'
# shares of right plays) driven through the shipped generation, pack, prep, stored probabilities, run_arms, GRPO and win-rate
# functions: outcome-driven training must raise the true win rate, flipped outcomes must not, and the stored probabilities must be
# the behaviour's.

DECKS=[{'cards':['knight','archers','fireball','giant','musketeer','valkyrie','bomber','arrows']},
       {'cards':['hog_rider','minions','zap','goblins','skeletons','mini_pekka','baby_dragon','cannon']}]
VOC=('knight','archers','fireball','giant');N_DEC=8;BEH='py:tests.simTruth:ToyBehaviour'
quiet=lambda *a,**k:None


def short(monkeypatch):
    monkeypatch.setattr(Game,'REG',20.0);monkeypatch.setattr(Game,'OT',10.0);monkeypatch.setattr(Game,'END',30.0)


def random_agent(seed=0):
    torch.manual_seed(seed);vocab=sorted({c for d in DECKS for c in d['cards']})
    return Agent(Policy(FEAT_DIM+6,len(vocab),16).state_dict(),vocab,np.zeros(FEAT_DIM+6),np.ones(FEAT_DIM+6))


def toy_state(seed,k):
    r=np.random.default_rng([int(seed),2,k]);x=np.zeros(FEAT_DIM,np.float32);x[:8]=r.normal(0,1,8);x[40]=3.0 if r.random()<0.5 else -3.0;return x


def right(x):
    return 'knight' if float(x[40])>0 else 'archers'


class ToyBehaviour:
    # follows the planted rule 55 percent of the time, otherwise one of the other cards; cells uniform
    def act(self,x,menu,rng,g=None,team=None):
        p=np.array([0.55 if c==right(x) else 0.15 for c in menu]);i=int(rng.choice(len(menu),p=p))
        return menu[i],int(rng.integers(GW*GH)),float(np.log(p[i])),float(-np.log(GW*GH))


class Oracle:
    def act(self,x,menu,rng,g=None,team=None):
        return right(x),0,0.0,0.0


class ToyWorld:
    # decisions alternate blue and red, one tick each, states drawn from the seed and decision index alone; the winner is drawn from
    # the difference of the two sides' counts of right plays
    decks=[{'cards':list(VOC)}]
    def rec(self,seed,k,a,rng):
        tm=TEAMS[k%2];x=toy_state(seed,k);name,c,lpc,lpx=a.act(x,list(VOC),rng);tx,ty=cell_tile(c,tm)
        return {'t':float(k),'team':tm,'card':name,'x':tx+0.5,'y':ty+0.5,'evolved':False,'hero':False,'erate':1,'hand':'|'.join(VOC),
                'state':x.astype(np.float16),'opp_hand':'minions|goblins|zap|hog_rider','opp_next':'the_log','roll':(0.0,)*4,'mu':(lpc,lpx),'idx':k}
    def end(self,recs,rng):
        q=sum((r['card']==right(r['state']))*(1 if r['team']=='blue' else -1) for r in recs)/(N_DEC/2)
        return 'blue' if rng.random()<1/(1+math.exp(-4*q)) else 'red'
    def play(self,seed,agents,record=TEAMS,horizon=0.0,rseed=None):
        rng=np.random.default_rng([int(seed),1]);recs=[self.rec(seed,k,agents[TEAMS[k%2]],rng) for k in range(N_DEC)];w=self.end(recs,rng)
        return {'winner':w,'end_t':float(N_DEC),'bc':int(w=='blue'),'rc':int(w=='red'),'ticks':N_DEC,'decks':[]},[r for r in recs if r['team'] in record]
    def branch(self,seed,plays,stops,actor,opponent,G,rseed):
        out=[];ticks=0
        for s in sorted(stops):
            tm=plays[s][1];past=[{'card':p[2],'team':p[1],'state':toy_state(seed,k)} for k,p in enumerate(plays[:s])];group=[]
            for j in range(G):
                rng=np.random.default_rng([int(rseed),s,j]);recs=[self.rec(seed,k,actor if TEAMS[k%2]==tm else opponent,rng) for k in range(s,N_DEC)]
                group.append(([r for r in recs if r['team']==tm],STr.score(self.end(past+recs,rng),tm),N_DEC-s));ticks+=N_DEC-s
            out.append((s,tm,group))
        return out,ticks
    def duel(self,seed,agent,other):
        tm=TEAMS[int(seed)%2];o,_=self.play(seed,{tm:agent,STr.opp(tm):other},());return STr.score(o['winner'],tm),N_DEC


def t_cells_correspond_across_sides_and_land_on_deployable_tiles():
    g=Game(p1=STr.side_cfg(DECKS[0]),p2=STr.side_cfg(DECKS[1]))
    for c in range(GW*GH):
        gy,gx=divmod(c,GW);m=(GH-1-gy)*GW+gx;bx,by=cell_tile(c,'blue');rx,ry=cell_tile(m,'red')
        assert (bx,by)==(rx,31-ry) and cell_of(bx+0.5,by+0.5)==c and cell_of(rx+0.5,ry+0.5)==m
        for tm in TEAMS:
            x,y=place(g,tm,'knight',c);assert g._valid_deploy(tm,x,y) and (y<15)==(tm=='blue'),(c,tm,x,y)
            assert place(g,tm,'fireball',c)==cell_tile(c,tm)
    # with every enemy tower up a troop sent to an enemy cell lands on the mirrored cell of its own half; a felled princess opens its pocket
    c=4*GW+1;x,y=place(g,'blue','knight',c);assert cell_of(x+0.5,y+0.5)==3*GW+1
    t=g.arena.get_tower('red','princess','left');t.take_damage(t.hp);assert place(g,'blue','knight',c)==cell_tile(c,'blue')


def t_a_game_is_reproduced_by_replaying_its_plays_and_groups_start_from_the_recorded_state(monkeypatch):
    short(monkeypatch);w=SimWorld(DECKS);a=random_agent()
    out,recs=w.play(3,{'blue':a,'red':a});out2,recs2=w.play(3,{'blue':a,'red':a})
    assert len(recs)>8 and {r['team'] for r in recs}==set(TEAMS) and out==out2 and all((r['state']==q['state']).all() for r,q in zip(recs,recs2))
    assert [r['idx'] for r in recs]==list(range(len(recs))) and all(r['card'] in r['hand'].split('|') for r in recs)
    # the packed store keeps times as float32
    plays=[(float(np.float32(r['t'])),r['team'],r['card'],cell_of(r['x'],r['y'])) for r in recs]
    for k in (0,len(recs)//2,len(recs)-1):
        m=w.replay(3,plays,k);assert (featurize(m.g,recs[k]['team']).astype(np.float16)==recs[k]['state']).all(),k
    k=len(recs)//2;res,ticks=w.branch(3,plays,[k],a,a,3,7);(s,tm,group),=res
    assert s==k and tm==recs[k]['team'] and len(group)==3 and ticks>sum(x[2] for x in group)>0
    for rs,R,_ in group:
        assert (rs[0]['state']==recs[k]['state']).all() and rs[0]['hand']==recs[k]['hand'] and all(r['team']==tm for r in rs) and R in (0.0,0.5,1.0)
    again,_=w.branch(3,plays,[k],a,a,3,7);assert [x[1] for x in again[0][2]]==[x[1] for x in group]
    assert all(len(x[0])==len(y[0]) and all((p['state']==q['state']).all() for p,q in zip(x[0],y[0])) for x,y in zip(again[0][2],group))


def t_group_advantages_and_the_grpo_surrogate():
    A=group_adv([1,0,1,1,1,1,0,0],[0,0,0,0,1,1,2,2])
    # a group with one outcome gets zero; otherwise the outcome minus the group mean over the group's spread
    assert torch.allclose(A[4:],torch.zeros(4)) and abs(float(A[:4].mean()))<1e-6 and A[0]>0>A[1] and abs(float(A[:4].std())-1)<1e-4
    torch.manual_seed(0);pol=Policy(8,5,16);S=torch.randn(6,8);H=torch.tensor([[0,1,2,-1]]*6);card=torch.tensor([0,1,0,2,1,0]);cell=torch.arange(6)
    tr=torch.tensor([0,0,0,1,1,2]);A=torch.tensor([1.0,-1.0,0.5])
    with torch.no_grad():old=token_logp(pol,S,H,card,cell)
    loss=grpo_loss(pol,S,H,card,cell,old,tr,A);loss.backward();g=[p.grad.clone() for p in pol.parameters()];pol.zero_grad()
    # at the sampling policy the ratio is one: the gradient is that of minus the mean over continuations of A times the token mean of
    # the log-probabilities (continuations of three, two and one decisions)
    lp=token_logp(pol,S,H,card,cell).sum(1);ref=-torch.stack([A[i]*lp[tr==i].sum()/(2*int((tr==i).sum())) for i in range(3)]).mean();ref.backward()
    assert abs(loss.item()+float(A.mean()))<1e-6 and all(torch.allclose(x,p.grad,atol=1e-6) for x,p in zip(g,pol.parameters()))
    m,lo,hi=paired([1,0,1,1],[0,0,1,0]);assert abs(m-0.5)<1e-9 and lo<m<hi
    sp,tau=rank_corr([1,2,3,4],[10,20,30,40]);assert sp==1 and tau==1
    sp,tau=rank_corr([1,2,3,4],[4,3,2,1]);assert sp==-1 and tau==-1


def t_in_a_planted_world_training_on_traces_raises_the_true_win_rate_and_flipped_outcomes_do_not(tmp_path):
    w=ToyWorld();st=tmp_path/'states';pk=tmp_path/'pack'
    generate(w,BEH,st,300,600,300,jobs=0,per=300,log=quiet);pack(st,pk);c=Corpus(pk)
    assert c.meta['games']==1200 and c.meta['records']==1200*N_DEC and c.meta['split_games']['stream']==600
    assert trace_ticks(st,pk)[0]==600*N_DEC
    prep(pk,hidden=32,bc_epochs=10,critic_epochs=10,q_epochs=10,eval_n=4000,day_n=500,rolled_pre=0,log=quiet,fit_batch=256);mu=stored(st,pk)
    # the stored probabilities are the behaviour's own: 0.55 for the right card, 0.15 for the others, one cell in 48
    ok=np.array([c.vocab[k]==('knight' if x>0 else 'archers') for k,x in zip(c.a['card'],c.X[:,40])])
    assert np.allclose(mu[:,1],-np.log(48),atol=1e-6) and np.allclose(mu[:,0],np.where(ok,np.log(0.55),np.log(0.15)),atol=1e-6)
    assert np.allclose(np.load(pk/'prep'/'mu_true_1.npy'),mu)
    arms=('bpco:mu=true','flash_nogate:mu=true','flash_nogate:mu=true:flip','bpco:mu=true:flip')
    run_arms(pk,arms,out=tmp_path/'runs'/'a.json',batch_games=4,lr=3e-3,hidden=32,threads=1,log=quiet)
    rep,_=grpo(pk,w,BEH,tmp_path/'grpo.json',600*N_DEC,G=4,batch_prompts=8,lr=3e-3,jobs=0,hidden=32,log=quiet)
    assert rep['ticks']>=600*N_DEC>rep['marks']['0.5']['ticks'] and rep['marks']['1']['batch']==rep['batches']
    spec=lambda f,k='':f'pack={pk};{f}'+(f'#{k}' if k else '')
    agents={'bc':spec(pk/'prep'/'bc_warm.pt'),'grpo':spec(tmp_path/'grpo.pt','grpo_1'),'oracle':'py:tests.simTruth:Oracle'}
    agents.update({a:spec(tmp_path/'runs'/'a.pt',a) for a in arms})
    r=wins(w,agents,{'behaviour':BEH},1500,jobs=0,log=quiet);o=r['outcomes']['behaviour'];wr={k:float(np.mean(v)) for k,v in o.items()}
    d={k:paired(o[k],o['bc']) for k in agents};assert r['n']==1500 and len(o['bc'])==1500
    # the clone of the behaviour wins about half its games against it; the always-right policy wins most; the outcome-driven recipes and
    # environment replay beat the clone by more than their intervals; flipped outcomes give the critic-free recipe nothing, while BPCO's
    # flipped arm keeps part of its gain through the critic pretrained on the true warm-up outcomes
    assert abs(wr['bc']-0.5)<0.06 and wr['oracle']>0.75,wr
    for k in ('bpco:mu=true','flash_nogate:mu=true','grpo'):assert d[k][1]>0.08,(k,d[k],wr)
    assert d['flash_nogate:mu=true:flip'][1]<0<d['flash_nogate:mu=true:flip'][2] or abs(d['flash_nogate:mu=true:flip'][0])<0.03,(d,wr)
    assert 0<d['bpco:mu=true:flip'][1] and d['bpco:mu=true:flip'][0]<d['bpco:mu=true'][0],(d,wr)
    json.dumps(r)
    assert load_agent(spec(tmp_path/'grpo.pt','grpo_1')).pol.trunk[0].weight.shape[0]==32


def t_win_rates_pair_games_across_agents_and_resume_from_a_partial_file(tmp_path):
    w=ToyWorld();ag={'oracle':'py:tests.simTruth:Oracle','behaviour':BEH};out=tmp_path/'w.json'
    r=wins(w,ag,{'behaviour':BEH},200,jobs=0,chunk=20,out=out,log=quiet,every=1);o=r['outcomes']['behaviour']
    # game s is the same game for every agent: the behaviour against itself gives the same outcomes as its own duel on that seed
    ref=[w.duel(s,ToyBehaviour(),ToyBehaviour())[0] for s in range(10**7,10**7+200)]
    assert (tmp_path/'w.json.partial').exists() and len(o['oracle'])==200 and o['behaviour']==ref
    assert np.mean(o['oracle'])>0.7 and abs(np.mean(o['behaviour'])-0.5)<0.1
    again=wins(w,ag,{'behaviour':BEH},200,jobs=0,chunk=20,out=out,log=quiet);assert again['outcomes']==r['outcomes'] and again['ticks']==0
