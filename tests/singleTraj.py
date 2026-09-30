import copy
import json
import math
from datetime import datetime,timezone

import numpy as np
import torch

from train.corpusStates import EXTRA
from train.counterfactual import CELLS,own_cells
from train.decisionStates import COLS
from train.feats import FEAT_DIM
from train.singleTraj import (Corpus,Critic,Policy,batch,dppo_loss,eval_set,evaluate,flash_loss,from_mat,gae,la_lambda,pack,prep,run_arms,sao_loss,
                              to_mat,token_logp)

# Single-trajectory RL on synthetic production traces through the shipped pack, prep and single-pass functions. Games plant the
# decision rule of tests/traceRl.py (feature 40 says which of knight and archers is right; the behaviour follows it 60 percent of the
# time; the winner is drawn from the two players' shares of right plays), a hidden variable the policy cannot see (the opponent holds
# minions, which costs the actor), and a rollout feature that tracks the winner. Outcome-driven recipes must move toward the right card
# in one pass, flipped outcomes must not, the privileged critics must explain more outcome variance, and the pessimistic Q-eval must not
# reward mass moved off the data.

HANDS='knight|archers|fireball|giant'
T0=datetime(2026,8,30,tzinfo=timezone.utc).timestamp()


def right_card(c):
    return 'knight' if c>0 else 'archers'


def shards(tmp_path,n_games=900,n_dec=10,follow=0.6,seed=0,per=300):
    # corpus shards in train.corpusStates format, games spread over 2026-08-30 .. 2026-09-07 in time order
    rng=np.random.default_rng(seed);out=tmp_path/'states';out.mkdir(parents=True,exist_ok=True);ts=np.sort(rng.uniform(T0,T0+8*86400,n_games))
    for k in range(0,n_games,per):
        rows={c:[] for c in COLS+EXTRA};X=[];roll=[];games=[]
        for g in range(k,min(k+per,n_games)):
            bid=f'G{g:05d}';share={'blue':0,'red':0};hid={'blue':rng.random()<0.5,'red':rng.random()<0.5};recs=[]
            for i in range(n_dec):
                team='blue' if i%2==0 else 'red';s=np.zeros(FEAT_DIM,np.float32);s[:16]=rng.normal(0,1,16);c=1.0 if rng.random()<0.5 else -1.0;s[40]=c*3.0
                right=right_card(c);card=right if rng.random()<follow else rng.choice([h for h in HANDS.split('|') if h!=right])
                share[team]+=card==right;x,y=CELLS[int(rng.choice(own_cells(team)))][0];X.append(s.astype(np.float16))
                opp=('minions' if hid[team] else 'musketeer')+'|goblins|zap|hog'
                recs.append((bid,i,i*12.5,team,card,float(x)+0.5,float(y)+0.5,False,False,1,HANDS,opp,'log'))
            q=(share['blue']-share['red'])/(n_dec/2)+0.25*(hid['red']-hid['blue'])
            winner='blue' if rng.random()<1/(1+np.exp(-6*q)) else 'red'
            for r in recs:
                for c_,v in zip(COLS+EXTRA,r):rows[c_].append(v)
                lead=0.1 if r[3]==winner else -0.1;roll.append((0.4+lead+rng.normal(0,0.02),0.4-lead+rng.normal(0,0.02),0.0,10.0))
            games.append({'bid':bid,'actual_winner':winner,'sim_winner':winner if rng.random()<0.6 else ('red' if winner=='blue' else 'blue'),'end_t':300.0,
                          'premature':False,'actual_bc':1,'actual_rc':0,'sim_bc':1,'sim_rc':0,'n':n_dec,'sim_blue':0.4,'sim_red':0.3,'ts':float(ts[g]),'mode':'Ranked',
                          'rolled':bool(g>=n_games*0.6)})
        p=out/f'shard{k//per:03d}.npz';np.savez_compressed(p,X=np.stack(X),roll=np.array(roll,np.float32),**{c:np.array(v) for c,v in rows.items()})
        p.with_suffix('.json').write_text(json.dumps({'outcomes':games})+'\n')
    return out


def packed(tmp_path,**kw):
    d=tmp_path/'pack';pack(shards(tmp_path,**kw),d);return d


def right_rate(pol,c,ev):
    idx={x:i for i,x in enumerate(c.vocab)};right=torch.tensor([idx[right_card(v)] for v in np.sign(c.X[ev['rows'],40].astype(np.float32))])
    with torch.no_grad():lc,_=pol.card_logp(ev['S'],ev['H'])
    return float(lc.exp().gather(1,right[:,None]).mean())


def t_gae_skips_nothing_of_the_actor_and_reduces_to_monte_carlo_and_td():
    v=torch.tensor([[0.2,0.4,0.5,0.0],[0.3,0.6,0.0,0.0]]);R=torch.tensor([1.0,0.0]);T=torch.tensor([3,2])
    mc=gae(v,R,T,torch.ones(2));td=gae(v,R,T,torch.zeros(2))
    # lambda one: the Monte Carlo return minus the value at every actor token; lambda zero: one-step TD over the actor's next token
    assert torch.allclose(mc,torch.tensor([[0.8,0.6,0.5,0.0],[-0.3,-0.6,0.0,0.0]]))
    assert torch.allclose(td,torch.tensor([[0.2,0.1,0.5,0.0],[0.3,-0.6,0.0,0.0]]))
    lam=la_lambda(torch.tensor([1,2,10,100]),0.4);assert lam[0]==0 and lam[1]==0 and abs(float(lam[2])-0.75)<1e-6 and abs(float(lam[3])-0.975)<1e-6
    x=torch.tensor([[1.0,2.0],[3.0,4.0],[5.0,6.0]]);tr=torch.tensor([0,0,1]);pos=torch.tensor([0,1,0]);m=to_mat(x,tr,pos,2,torch.tensor([2,1]))
    assert m.tolist()==[[1,2,3,4],[5,6,0,0]] and torch.equal(from_mat(m,tr,pos),x)


def t_flash_admits_whole_trajectories_and_weights_by_the_detached_ratio():
    lmu=torch.log(torch.tensor([0.5,0.5,0.5,0.5,0.9,0.1]));tr=torch.tensor([0,0,1,1,2,2]);T=torch.tensor([2.0,2.0,2.0]);A=torch.tensor([0.5,-0.5,0.5])
    lp=torch.log(torch.tensor([0.5,0.52,0.5,0.49,0.3,0.6])).requires_grad_(True)
    loss,st=flash_loss(lp,lmu,A,tr,T,3e-3);loss.backward()
    # the third trajectory drifted (Bernoulli KL far above delta) and contributes nothing; the others get -A rho / (B T) per token
    assert abs(st['admitted']-2/3)<1e-6 and (lp.grad[4:]==0).all()
    rho=(lp.detach()-lmu).exp();want=-(A[tr]*rho/T[tr])/3;assert torch.allclose(lp.grad[:4],want[:4],atol=1e-6)
    lp2=lp.detach().clone().requires_grad_(True);flash_loss(lp2,lmu,A,tr,T,math.inf)[0].backward();assert (lp2.grad[4:]!=0).all()


def t_sao_masks_ratios_outside_the_band_and_dppo_masks_probability_moves_with_the_advantage():
    lmu=torch.log(torch.tensor([0.5,0.5,0.5,0.5]));lp=torch.log(torch.tensor([0.5,0.9,0.2,0.4])).requires_grad_(True);A=torch.tensor([1.0,1.0,-1.0,-1.0])
    loss,st=sao_loss(lp,lmu,A,0.3,0.5);loss.backward()
    # ratios 1, 1.8, 0.4, 0.8: the second and third leave (0.7, 1.5) and are masked
    assert st['masked']==0.5 and lp.grad[1]==0 and lp.grad[2]==0 and lp.grad[0]<0 and lp.grad[3]>0
    lp=torch.log(torch.tensor([0.75,0.6,0.2,0.45])).requires_grad_(True);tr=torch.tensor([0,0,1,1])
    loss,st=dppo_loss(lp,lmu,A,tr,2,0.2);loss.backward()
    # probability moves +0.25 (A>0: masked), +0.1 (kept), -0.3 (A<0: masked), -0.05 (kept); the kept gradient is -A * ratio / B
    assert st['masked']==0.5 and lp.grad[0]==0 and lp.grad[2]==0 and abs(float(lp.grad[1])+0.6)<1e-5 and abs(float(lp.grad[3])-0.45)<1e-5


def t_the_bounded_critic_stays_inside_the_reward_range():
    cr=Critic(8,5,hidden=16,bounded=True);S=torch.randn(64,8)*100;v=cr(S,torch.randint(0,5,(64,)))
    assert v.shape==(64,2) and (v>0).all() and (v<1).all()
    with torch.no_grad():cr.vc.bias.fill_(1e6)
    assert (cr(S,torch.zeros(64,dtype=torch.long))[:,0]<1).all()


def t_pack_orders_by_battle_time_and_makes_one_trajectory_per_player(tmp_path):
    d=packed(tmp_path,n_games=60,per=25);c=Corpus(d);A=c.a
    assert c.meta['games']==60 and c.meta['trajectories']==120 and c.meta['records']==600 and set(c.meta['split_games'])=={'warm','stream','heldA','heldB'}
    assert (np.diff(A['g_ts'])>=0).all() and (np.diff(A['t_start'])==10//2).all() and (A['t_len']==5).all()
    # every trajectory is one team's decisions in order, labelled with that team's recorded outcome
    for ti in range(0,120,17):
        r=np.arange(A['t_start'][ti],A['t_start'][ti]+A['t_len'][ti]);assert (A['team'][r]==A['t_team'][ti]).all() and (A['pos'][r]==np.arange(5)).all()
        assert (A['y'][r]==A['t_R'][ti]).all() and A['t_R'][2*(ti//2)]+A['t_R'][2*(ti//2)+1]==1
    b=batch(c,3,7);assert b['n']==8 and b['S'].shape==(40,c.n_state) and int(b['tr'].max())==7 and (b['H'][:,0]==b['card']).all()
    idx={x:i for i,x in enumerate(c.vocab)};assert idx['minions'] in set(A['OH'].ravel().tolist()) and (A['ON']==idx['log']).all()


def t_single_pass_recipes_learn_the_planted_rule_and_flipped_outcomes_do_not(tmp_path):
    d=packed(tmp_path);log=lambda *a,**k:None
    prep(d,hidden=32,bc_epochs=20,critic_epochs=10,q_epochs=10,eval_n=4000,day_n=500,rolled_pre=40,log=log,fit_batch=256)
    arms=('bc_online','reinforce','flash:mu=warm','flash_nogate','sao:mu=warm','bpco:mu=warm','bpco','flash:mu=warm:flip','bpco:mu=warm:flip','bpco_hidden:mu=warm')
    r=run_arms(d,arms,batch_games=2,lr=3e-3,hidden=32,threads=1,log=log)
    c=Corpus(d);E=torch.load(d/'prep'/'evals.pt',weights_only=False);ev=eval_set(c,E['rows']['heldB'])
    bc=Policy(c.n_state,c.n_card,32);bc.load_state_dict(torch.load(d/'prep'/'bc_warm.pt'));base=right_rate(bc,c,ev);m=r['arms']
    assert 0.5<base<0.7 and r['bc']['kl_to_bc']==0 and set(r['prequential']['bpco:mu=warm'])=={f'day{x}' for x in (244,245,246,247)}
    # the outcome-driven single-pass updates prefer the right card more than the behaviour clone on held-out games; noise does not
    g={a:m[a]['winner_gap']-r['bc']['winner_gap'] for a in arms}
    for a in ('flash:mu=warm','sao:mu=warm','bpco:mu=warm','bpco_hidden:mu=warm'):assert g[a]>0.02,(a,g)
    for a in ('flash:mu=warm:flip','bpco:mu=warm:flip'):assert g[a]<min(g['flash:mu=warm'],g['bpco:mu=warm'])/2,(a,g)
    assert abs(g['bc_online'])<g['bpco:mu=warm']
    # against a fixed behaviour estimate the sequence trust region admits most trajectories at first and closes as the actor drifts
    st=r['train_stats']['flash:mu=warm'];assert st['day244']['admitted']>st['day247']['admitted']>=0
    # with the actor's own starting point as the estimate the first day's ratios start at one; the estimate fit on other data does not
    st=r['train_stats'];assert st['bpco:mu=warm']['day244']['abs_logratio']<st['bpco']['day244']['abs_logratio']
    assert m['flash_nogate']['kl_to_bc']>=m['flash:mu=warm']['kl_to_bc'] and 'critic_heldB' in m['bpco:mu=warm']


def t_privileged_critics_explain_more_of_the_outcome(tmp_path):
    d=packed(tmp_path);rep=prep(d,hidden=32,bc_epochs=2,critic_epochs=30,q_epochs=2,eval_n=4000,day_n=200,rolled_pre=60,log=lambda *a,**k:None,fit_batch=256)
    ev=rep['critic_pretrain_heldB']
    # the opponent's hidden hand and the rollout lead carry outcome information the actor's state lacks
    assert ev['warm:b|hidden']['ev']>ev['warm:b|']['ev']+0.01 and ev['rolled:b|sim']['ev']>ev['rolled:b|']['ev']+0.1
    assert ev['rolled:b|hidden+sim']['ev']>=ev['rolled:b|hidden']['ev']


def t_the_pessimistic_q_eval_does_not_reward_leaving_the_data(tmp_path):
    d=packed(tmp_path,n_games=200,per=100);c=Corpus(d);rows=np.arange(400);ev=eval_set(c,rows);torch.manual_seed(0)
    bc=Policy(c.n_state,c.n_card,16);sup=torch.ones(c.n_card,48,dtype=torch.bool);far=47;sup[:,far]=False

    class Q(torch.nn.Module):
        # an outcome model that extrapolates: the unsupported cell looks like a sure win, supported cells differ a little
        def forward(self,S,card,cell):return torch.where(cell==far,torch.full(cell.shape,5.0),(cell%5-2).float()*0.3)
    leave=copy.deepcopy(bc)
    with torch.no_grad():leave.cell.bias[far]+=8.0
    a=evaluate(bc,bc,Q(),ev,sup,q_n=400);b=evaluate(leave,bc,Q(),ev,sup,q_n=400)
    assert b['support_mass']<a['support_mass']-0.5 and b['q_direct']>a['q_direct']+0.3
    assert b['q_pess']<a['q_pess'] and abs(b['q_support']-a['q_support'])<1e-3
    lp=token_logp(bc,ev['S'][:5],ev['H'][:5],ev['card'][:5],ev['cell'][:5]).sum(1)
    assert torch.allclose(lp,bc.logp(ev['S'][:5],ev['H'][:5],ev['card'][:5],ev['cell'][:5]),atol=1e-5)
