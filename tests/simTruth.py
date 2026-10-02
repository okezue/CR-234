import json
import math

import numpy as np
import torch

import train.simTruth as STr
from sim.cards import create,key
from sim.game import Game
from train.counterfactual import GH,GW,cell_of
from train.feats import FEAT_DIM,featurize
from train.simTruth import (LEVEL,TEAMS,Agent,SimWorld,cell_tile,generate,group_adv,grpo,grpo_loss,load_agent,markdown,paired,place,rank_corr,
                            report,stored,trace_ticks,wins)
from train.singleTraj import Corpus,Policy,pack,prep,run_arms,token_logp

# The simulator as a known world. Three kinds of test: the real world's rules (cells correspond across sides and land on deployable
# tiles, a game is reproduced by replaying its plays, a GRPO group starts from the recorded state, an evaluation game seats the policy
# by its seed) on short sim.game games; a planted world in which the best policy is known (the right card is written in the state;
# the winner is drawn from the two sides' shares of right plays) driven through the shipped generation, pack, prep, stored
# probabilities, run_arms, GRPO and win-rate functions: outcome-driven training must raise the true win rate, flipped outcomes must
# not, and the stored probabilities must be the behaviour's; and the registered decision rules applied by report to hand-built win
# files and run reports with known answers.

DECKS=[{'cards':['knight','archers','fireball','giant','musketeer','valkyrie','bomber','arrows']},
       {'cards':['hog_rider','minions','zap','goblins','skeletons','mini_pekka','baby_dragon','cannon']}]
VOC=('knight','archers','fireball','giant');N_DEC=8;BEH='py:tests.simTruth:ToyBehaviour'
quiet=lambda *a,**k:None


def short(monkeypatch,reg=20.0):
    monkeypatch.setattr(Game,'REG',reg);monkeypatch.setattr(Game,'OT',10.0);monkeypatch.setattr(Game,'END',reg+10.0)


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


class Spy:
    # an agent that records the sides it is asked to play
    def __init__(self,a):
        self.a=a;self.teams=set()
    def act(self,x,menu,rng,g=None,team=None):
        self.teams.add(team);return self.a.act(x,menu,rng,g,team)


class Seat:
    # plays its first menu card on cell 0 and records the side and the deck it plays
    def __init__(self):
        self.seen=set()
    def act(self,x,menu,rng,g=None,team=None):
        self.seen.add((team,tuple(sorted(g.players[team].deck.all))));return menu[0],0,0.0,0.0


class Who:
    # a world whose branch records the actor and opponent GRPO's worker passes it
    def branch(self,seed,plays,stops,actor,opponent,G,rseed):
        self.got=(actor,opponent);return [],0


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


def t_cells_correspond_across_sides_land_on_deployable_tiles_and_an_enemy_troop_on_its_half_makes_a_player_decide():
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
    # below its elixir threshold a player decides only when an enemy troop stands on its own half
    m=STr.Match(DECKS,5);m.st['blue']['thr']=m.st['red']['thr']=10;assert m.menu('blue') is None and m.menu('red') is None
    k=create('knight',LEVEL,'red',9,10);m.g.deploy('red',k);assert m.menu('blue')==m.g.players['blue'].deck.hand and m.menu('red') is None
    k.y=25.0;assert m.menu('blue') is None


def t_a_game_is_reproduced_by_replaying_its_plays_and_groups_start_from_the_recorded_state(monkeypatch):
    # games of up to 50 s: past 32 s the store's float32 times can be more than 1e-6 off the tick times
    short(monkeypatch,40.0);w=SimWorld(DECKS);a=random_agent()
    out,recs=w.play(3,{'blue':a,'red':a});out2,recs2=w.play(3,{'blue':a,'red':a})
    assert len(recs)>8 and {r['team'] for r in recs}==set(TEAMS) and out==out2 and all((r['state']==q['state']).all() for r,q in zip(recs,recs2))
    assert [r['idx'] for r in recs]==list(range(len(recs))) and all(r['card'] in r['hand'].split('|') for r in recs)
    # the packed store keeps times as float32
    plays=[(float(np.float32(r['t'])),r['team'],r['card'],cell_of(r['x'],r['y'])) for r in recs]
    err=[abs(p[0]-r['t']) for p,r in zip(plays,recs)];kf=int(np.argmax(err));assert err[kf]>1e-6,max(err)
    for k in (0,len(recs)//2,kf,len(recs)-1):
        m=w.replay(3,plays,k);assert (featurize(m.g,recs[k]['team']).astype(np.float16)==recs[k]['state']).all(),k
    k=len(recs)//2;ac,op=Spy(a),Spy(a);res,ticks=w.branch(3,plays,[k],ac,op,3,7);(s,tm,group),=res
    assert s==k and tm==recs[k]['team'] and len(group)==3 and ticks>sum(x[2] for x in group)>0
    for rs,R,_ in group:
        assert (rs[0]['state']==recs[k]['state']).all() and rs[0]['hand']==recs[k]['hand'] and all(r['team']==tm for r in rs) and R in (0.0,0.5,1.0)
    again,_=w.branch(3,plays,[k],a,a,3,7);assert [x[1] for x in again[0][2]]==[x[1] for x in group]
    assert all(len(x[0])==len(y[0]) and all((p['state']==q['state']).all() for p,q in zip(x[0],y[0])) for x,y in zip(again[0][2],group))
    # the actor plays the prompt's side, the opponent the other; GRPO's worker passes the behaviour as the opponent, not the actor
    assert ac.teams=={tm} and op.teams=={STr.opp(tm)}
    who=Who();STr._init(who,{'behaviour':BEH});STr._W['cache']['actor']=a;STr._branch((3,plays,[k],a.pol.state_dict(),3,7))
    assert who.got[0] is a and type(who.got[1]).__name__=='ToyBehaviour'


def t_an_evaluation_game_seats_the_policy_blue_with_the_first_deck_on_even_seeds_and_red_with_it_on_odd_seeds(monkeypatch):
    short(monkeypatch);w=SimWorld(DECKS);dk=lambda d:tuple(sorted(key(c) or c for c in d['cards']))
    for p in (0,1):
        s=next(s for s in range(p,60,2) if w.deal(s)[0] is not w.deal(s)[1]);a,b=Seat(),Seat();r,ticks=w.duel(s,a,b);d0,d1=map(dk,w.deal(s));tm=TEAMS[p]
        assert a.seen=={(tm,d0)} and b.seen=={(STr.opp(tm),d1)} and r in (0.0,0.5,1.0) and ticks>0,(s,a.seen,b.seen)


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
    # a normal 95% interval: 1.96 standard errors of the paired differences (1, 0, 0, 1) either side
    m,lo,hi=paired([1,0,1,1],[0,0,1,0]);assert abs(m-0.5)<1e-9 and abs(hi-m-0.98*math.sqrt(1/3))<1e-12 and abs(m-lo-0.98*math.sqrt(1/3))<1e-12
    sp,tau=rank_corr([1,2,3,4],[10,20,30,40]);assert sp==1 and tau==1
    sp,tau=rank_corr([1,2,3,4],[4,3,2,1]);assert sp==-1 and tau==-1


def t_the_report_applies_the_registered_rules_to_hand_built_win_files_and_run_reports(tmp_path):
    # six trained policies and bc over 400 paired games (the flipped copy over 200, from a second file): gains of 80, 40, 32, 20, -24
    # and 0 games over bc; the flipped copy's dq pess .001 sets F; bpco clears 2F but loses two points of support; flash clears F but
    # not 2F; dq support has the true top at Spearman 1/7, the winner gap Spearman 33/35 with another top
    up=lambda k:[int(i%2==0 or i<2*k) for i in range(400)];bc=up(0);tr=lambda a:R[a]['true']['behaviour']
    arm=lambda q,s,wg,sup=0.0:{'q_pess':0.5+q,'q_support':0.5+s,'q_direct':0.5+q,'winner_gap':wg,'kl_to_bc':0.1,'entropy':3.5,'support_mass':0.99+sup}
    runs={'grpo':{'grpo_1':arm(.006,.006,.05)},'flip':{'sao:flip':arm(.001,.003,.02)},
          'cross':{'sao':arm(.003,.001,.06),'bpco':arm(.004,.0005,.04,-.02),'flash':arm(.0015,.004,.03),'flash_nogate':arm(-.001,.002,.01)}}
    for k,v in runs.items():(tmp_path/f'{k}.json').write_text(json.dumps({'bc':arm(0,0,0),'arms':v}))
    core={'bc':bc,'grpo/grpo_1':up(80),'cross/sao':up(40),'cross/bpco':up(32),'cross/flash':up(20),
          'cross/flash_nogate':[int(i%2==0 and i>=48) for i in range(400)],'true/sao:mu=true':up(48)}
    (tmp_path/'core.json').write_text(json.dumps({'outcomes':{'behaviour':core}}))
    (tmp_path/'sec.json').write_text(json.dumps({'outcomes':{'behaviour':{'bc':bc[:200],'flip/sao:flip':bc[:200]}}}))
    files=[tmp_path/f'{k}.json' for k in runs],[tmp_path/'core.json',tmp_path/'sec.json']
    rep=report(*files,out=tmp_path/'rep.json',boots=200);R=rep['rows'];pol=('grpo/grpo_1','cross/sao','cross/bpco','cross/flash','cross/flash_nogate','flip/sao:flip')
    assert abs(rep['F']-0.001)<1e-12 and [R[a]['proxy']['verdict'] for a in pol]==['improves','improves','no','no','no','no']
    assert [tr(a)['verdict'] for a in pol]==['improves']*4+['hurts','no change'] and tr('bc')['n']==400 and tr('flip/sao:flip')['n']==200
    m,lo,hi=tr('cross/sao')['dwr'];assert abs(m-0.1)<1e-12 and abs(hi-m-0.588/math.sqrt(399))<1e-12 and abs(m-lo-0.588/math.sqrt(399))<1e-12
    rk=rep['rank'];p,s,g=(rk[f'behaviour|{k}'] for k in ('dq_pess','dq_support','d_winner_gap'))
    assert rk['behaviour|agreement']==4 and p['games']==200 and p['policies']==6 and -1<=p['spearman_boot'][0]<=p['spearman_boot'][1]<=1
    assert abs(p['spearman']-33/35)<1e-12 and p['proxy_top']==p['true_top']=='grpo/grpo_1' and p['ranks_correctly']
    assert abs(s['spearman']-1/7)<1e-12 and s['proxy_top']=='grpo/grpo_1' and not s['ranks_correctly']
    assert abs(g['spearman']-33/35)<1e-12 and g['proxy_top']=='cross/sao' and not g['ranks_correctly']
    # recovery is the gain over bc as a share of GRPO's gain over bc, here 40/80, with a delta-method interval
    r,lo,hi=rep['recovery']['behaviour|cross/sao'];assert abs(r-0.5)<1e-12 and abs(hi-r-1.96*math.sqrt(20/399/16))<1e-12 and abs(r-lo-(hi-r))<1e-12
    assert len(rep['recovery'])==5 and report(*files,top='flip/sao:flip',boots=10)['recovery']=={}
    (k,v),=rep['stored_vs_estimated'].items();assert k=='behaviour|cross/sao->true/sao:mu=true' and abs(v[0]-0.02)<1e-12
    assert rep['volume']['behaviour|bpco']==[None,None,tr('cross/bpco')['dwr']] and json.loads((tmp_path/'rep.json').read_text())['F']==rep['F']
    assert 'bootstrap over 200 games' in markdown(rep)


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
