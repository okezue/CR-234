import copy
import json
import math
from datetime import datetime,timezone

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import train.singleTraj as ST
from train.corpusStates import EXTRA
from train.counterfactual import CELLS,own_cells
from train.decisionStates import COLS
from train.feats import FEAT_DIM
from train.singleTraj import (HAND,N_CELLS,Arm,Corpus,Critic,Policy,batch,bern_kl,critic_key,dppo_loss,eval_set,evaluate,flash_loss,from_mat,gae,
                              la_lambda,pack,parse,prep,run_arms,sao_loss,simgroup,to_mat,token_logp)

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


def shards(tmp_path,n_games=900,n_dec=10,follow=0.6,seed=0,per=300,ids=None):
    # corpus shards in train.corpusStates format, games spread over 2026-08-30 .. 2026-09-07 in time order; ids maps a game's time rank
    # to its battle id (default: ids in time order)
    rng=np.random.default_rng(seed);out=tmp_path/'states';out.mkdir(parents=True,exist_ok=True);ts=np.sort(rng.uniform(T0,T0+8*86400,n_games))
    for k in range(0,n_games,per):
        rows={c:[] for c in COLS+EXTRA};X=[];roll=[];games=[]
        for g in range(k,min(k+per,n_games)):
            bid=ids(g) if ids else f'G{g:05d}';share={'blue':0,'red':0};hid={'blue':rng.random()<0.5,'red':rng.random()<0.5};recs=[]
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


def toy(L=(3,2,4,3,2,4),R=None,n_state=8,n_card=6,seed=0):
    # a batch in the layout of singleTraj.batch with one trajectory per entry of L (decision counts), outcomes alternating won and
    # lost unless given, the behaviour estimate set to the starting actor's own probabilities
    torch.manual_seed(seed);pol=Policy(n_state,n_card,16);L=torch.tensor(L);n=len(L);m=int(L.sum())
    H=torch.stack([torch.randperm(n_card)[:HAND] for _ in range(m)]);card=H[:,0].clone();cell=torch.randint(0,N_CELLS,(m,));S=torch.randn(m,n_state)
    tr=torch.arange(n).repeat_interleave(L);pos=torch.cat([torch.arange(int(x)) for x in L]);R=torch.arange(n).remainder(2).float() if R is None else R
    with torch.no_grad():mu=token_logp(pol,S,H,card,cell)
    return pol,{'S':S,'H':H,'card':card,'cell':cell,'tr':tr,'pos':pos,'L':L,'R':R,'n':n,'Z':None,'mu':mu}


def with_critic(spec,lr=1e-4,**kw):
    pol,b=toy(**kw);cfg=parse(spec)
    return Arm(spec,pol,{critic_key(cfg):Critic(b['S'].shape[1],pol.card.out_features,16,bounded=cfg['bounded'])},lr),b


def params(m):
    return [p.detach().clone() for p in m.parameters()]


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


def t_flash_gates_on_the_trajectory_mean_of_the_behaviour_first_divergence():
    # one token alone above delta, its trajectory's mean below: the whole trajectory is admitted, that token included
    lmu=torch.log(torch.tensor([0.5,0.5]));lp=torch.log(torch.tensor([0.5,0.545])).requires_grad_(True);T=torch.tensor([2.0]);A=torch.tensor([1.0])
    kl=bern_kl(lmu,lp.detach());assert kl[1]>3e-3>=kl.mean()
    loss,st=flash_loss(lp,lmu,A,torch.tensor([0,0]),T,3e-3);loss.backward();rho=(lp.detach()-lmu).exp()
    assert st['admitted']==1 and lp.grad[1]<0 and torch.allclose(lp.grad,-rho/2,atol=1e-6)
    # KL(mu || pi): mu .5, pi .01 is 1.61 (rejected at delta 1) although KL(pi || mu) is 0.64; the mirrored trajectory is admitted
    lmu=torch.log(torch.tensor([0.5,0.01]));lp=torch.log(torch.tensor([0.01,0.5])).requires_grad_(True)
    assert bern_kl(lmu[:1],lp[:1].detach())>1.0>bern_kl(lp[:1].detach(),lmu[:1])
    loss,st=flash_loss(lp,lmu,torch.tensor([1.0,1.0]),torch.tensor([0,1]),torch.ones(2),1.0);loss.backward()
    assert st['admitted']==0.5 and lp.grad[0]==0 and lp.grad[1]<0


def t_flash_and_reinforce_centre_the_outcome_on_the_batch_mean():
    # every trajectory won: the centred advantage is zero everywhere, so the one update leaves the actor exactly where it started
    for spec in ('flash:mu=warm','flash_nogate:mu=warm','reinforce'):
        pol,b=toy(R=torch.ones(6));a=Arm(spec,pol,{});w=params(a.pol);a.step(b)
        assert all(torch.equal(p,q) for p,q in zip(a.pol.parameters(),w)),spec
        pol,b=toy();a=Arm(spec,pol,{});w=params(a.pol);a.step(b)
        assert not all(torch.equal(p,q) for p,q in zip(a.pol.parameters(),w)),spec


def t_sao_masks_ratios_outside_the_band_and_dppo_masks_probability_moves_with_the_advantage():
    lmu=torch.log(torch.tensor([0.5,0.5,0.5,0.5]));lp=torch.log(torch.tensor([0.5,0.9,0.2,0.4])).requires_grad_(True);A=torch.tensor([1.0,1.0,-1.0,-1.0])
    loss,st=sao_loss(lp,lmu,A,0.3,0.5);loss.backward()
    # ratios 1, 1.8, 0.4, 0.8: the second and third leave (0.7, 1.5) and are masked
    assert st['masked']==0.5 and lp.grad[1]==0 and lp.grad[2]==0 and lp.grad[0]<0 and lp.grad[3]>0
    lp=torch.log(torch.tensor([0.75,0.6,0.2,0.45])).requires_grad_(True);tr=torch.tensor([0,0,1,1])
    loss,st=dppo_loss(lp,lmu,A,tr,2,0.2);loss.backward()
    # probability moves +0.25 (A>0: masked), +0.1 (kept), -0.3 (A<0: masked), -0.05 (kept); the kept gradient is -A * ratio / B
    assert st['masked']==0.5 and lp.grad[0]==0 and lp.grad[2]==0 and abs(float(lp.grad[1])+0.6)<1e-5 and abs(float(lp.grad[3])-0.45)<1e-5


def t_sao_weights_kept_tokens_by_the_detached_ratio():
    lmu=torch.log(torch.full((4,),0.5));lp=torch.log(torch.tensor([0.6,0.45,0.9,0.3])).requires_grad_(True);A=torch.tensor([1.0,-2.0,0.5,1.0])
    loss,st=sao_loss(lp,lmu,A,0.3,5.0);loss.backward();r=(lp.detach()-lmu).exp()
    # ratios 1.2, 0.9, 1.8 inside (0.7, 6), 0.6 outside: the gradient is -A r / m on kept tokens and zero on the masked one
    assert st['masked']==0.25 and lp.grad[3]==0 and torch.allclose(lp.grad[:3],-(A*r)[:3]/4,rtol=1e-6,atol=0)


def t_the_critic_steps_at_five_times_the_actor_rate_and_twice_per_update_for_sao():
    for spec in ('sao','bpco'):
        a,b=with_critic(spec);seen=[];step=a.copt.step
        a.copt.step=lambda *x,**k:(seen.append(a.copt.param_groups[0]['lr']),step(*x,**k))[1]
        a.step(b);assert seen and seen==pytest.approx([5*a.opt.param_groups[0]['lr']]*len(seen)),(spec,seen)
        if spec=='sao':assert len(seen)==2


def t_critic_targets_are_the_outcome_and_only_sao_whitens_its_advantages(monkeypatch):
    # the critic regresses both tokens on the recorded outcome (Monte Carlo, lambda_V 1); the policy loss gets the length-adaptive GAE
    # of the pre-update critic, raw for BPCO (alpha 0.4) and whitened over the batch for SAO (alpha 1.5)
    mse=F.mse_loss;targets=[];got={}
    monkeypatch.setattr(F,'mse_loss',lambda x,y,**k:(targets.append(y.detach().clone()),mse(x,y,**k))[1])
    for name in ('dppo_loss','sao_loss'):
        f=getattr(ST,name);monkeypatch.setattr(ST,name,lambda lp,lmu,A,*x,f=f,name=name:(got.__setitem__(name,A.detach().clone()),f(lp,lmu,A,*x))[1])
    for spec,alpha,name in (('bpco',0.4,'dppo_loss'),('sao',1.5,'sao_loss')):
        a,b=with_critic(spec);targets.clear();tr,pos,L,R=b['tr'],b['pos'],b['L'],b['R'];T=2*L
        with torch.no_grad():V=a.critic(b['S'],b['card'],None)
        g=from_mat(gae(to_mat(V,tr,pos,b['n'],L),R,T,la_lambda(T,alpha)),tr,pos).reshape(-1);a.step(b)
        assert targets and all(torch.equal(y,R[tr][:,None].expand(-1,2)) for y in targets),spec
        assert la_lambda(T,alpha).max()<1 and not torch.allclose(V+g.reshape(-1,2),R[tr][:,None].expand(-1,2),atol=1e-3)
        want=g if spec=='bpco' else (g-g.mean())/g.std();assert torch.allclose(got[name],want,atol=1e-5),spec


def t_the_bounded_critic_stays_inside_the_reward_range():
    cr=Critic(8,5,hidden=16,bounded=True);S=torch.randn(64,8)*100;v=cr(S,torch.randint(0,5,(64,)))
    assert v.shape==(64,2) and (v>0).all() and (v<1).all()
    with torch.no_grad():cr.vc.bias.fill_(1e6)
    assert (cr(S,torch.zeros(64,dtype=torch.long))[:,0]<1).all()


def t_pack_orders_by_battle_time_and_makes_one_trajectory_per_player(tmp_path):
    # battle ids run against time, so only the battle time can give the order
    d=packed(tmp_path,n_games=60,per=25,ids=lambda g:f'G{59-g:05d}');c=Corpus(d);A=c.a
    assert c.meta['games']==60 and c.meta['trajectories']==120 and c.meta['records']==600 and set(c.meta['split_games'])=={'warm','stream','heldA','heldB'}
    assert (np.diff(A['g_ts'])>=0).all() and (np.diff(A['t_start'])==10//2).all() and (A['t_len']==5).all() and c.meta['bid']!=sorted(c.meta['bid'])
    # the standardisation statistics are the warm-up games' own, not the whole corpus's
    Xw=c.X[c.records(c.games('warm'))].astype(np.float64);Xa=np.asarray(c.X,np.float64);assert np.abs(Xa.mean(0)-Xw.mean(0)).max()>1e-2
    assert np.allclose(c.mu[:FEAT_DIM],Xw.mean(0),atol=1e-5) and np.allclose(c.sd[:FEAT_DIM],Xw.std(0)+1e-6,atol=1e-4)
    # every trajectory is one team's decisions in order, labelled with that team's recorded outcome
    for ti in range(0,120,17):
        r=np.arange(A['t_start'][ti],A['t_start'][ti]+A['t_len'][ti]);assert (A['team'][r]==A['t_team'][ti]).all() and (A['pos'][r]==np.arange(5)).all()
        assert (A['y'][r]==A['t_R'][ti]).all() and A['t_R'][2*(ti//2)]+A['t_R'][2*(ti//2)+1]==1 and (A['idx'][r]==2*A['pos'][r]+A['t_team'][ti]).all()
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


def t_privileged_critics_explain_more_of_the_outcome(tmp_path,monkeypatch):
    d=packed(tmp_path);fit=ST.fit_head;seen=[];monkeypatch.setattr(ST,'fit_head',lambda Q,S,*a,**k:(seen.append(S.clone()),fit(Q,S,*a,**k))[1])
    rep=prep(d,hidden=32,bc_epochs=2,critic_epochs=30,q_epochs=2,eval_n=4000,day_n=200,rolled_pre=60,log=lambda *a,**k:None,fit_batch=256)
    # the Q model is fit on the states of held-out half A only (half B is the one scored)
    c=Corpus(d);assert len(seen)==1 and torch.equal(seen[0],c.S(0,0,c.records(c.games('heldA'))))
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
    strict=sup.clone();strict[:,:24]=False;s=evaluate(bc,bc,Q(),ev,sup,q_n=400,strict=strict)
    assert s['support_mass_strict']<s['support_mass'] and s['q_pess_strict']<=s['q_pess']+1e-6 and 0<s['q_top_mass']<1
    lp=token_logp(bc,ev['S'][:5],ev['H'][:5],ev['card'][:5],ev['cell'][:5]).sum(1)
    assert torch.allclose(lp,bc.logp(ev['S'][:5],ev['H'][:5],ev['card'][:5],ev['cell'][:5]),atol=1e-5)


def t_the_world_model_as_an_environment_learns_from_an_oracle_menu(tmp_path):
    d=packed(tmp_path);prep(d,hidden=32,bc_epochs=20,critic_epochs=1,q_epochs=10,eval_n=4000,day_n=200,rolled_pre=10,log=lambda *a,**k:None,fit_batch=256)
    c=Corpus(d);A=c.a;rows=[]
    # a perfect world model: in every stream decision's menu the right card returns +1 and the others -1
    for g in c.games('stream')[:200]:
        bid=c.meta['bid'][g]
        for r in range(int(c.gr[g]),int(c.gr[g+1])):
            right=right_card(float(c.X[r,40]));team='red' if A['team'][r] else 'blue'
            for k,name in enumerate(('knight','archers','fireball')):
                rows.append((bid,int(A['idx'][r]),k,team,name,4.5,4.5,0.0,0.0,1.0 if name==right else -1.0))
    cf=tmp_path/'cf.json';cf.write_text(json.dumps({'rows':rows})+'\n')
    rep=simgroup(d,cf,epochs=(1,5),batch_size=256,lr=3e-3,hidden=32,threads=1,log=lambda *a,**k:None)
    assert rep['games']==200 and rep['rows']==200*10*3
    assert rep['arms']['simgroup:epochs=5']['winner_gap']>rep['arms']['simgroup:epochs=1']['winner_gap']>rep['bc']['winner_gap']


def t_a_short_menu_never_masks_the_first_vocabulary_card():
    # a hand with an empty slot and vocabulary card 0 in it: the card keeps its probability under both log-probability paths
    torch.manual_seed(0);pol=Policy(8,6,hidden=16);S=torch.randn(3,8);H=torch.tensor([[3,0,-1,-1],[0,2,5,-1],[4,1,0,2]]);card=torch.tensor([0,0,0]);cell=torch.tensor([1,2,3])
    lp=token_logp(pol,S,H,card,cell)
    assert torch.isfinite(lp).all() and (lp[:,0]>-20).all() and torch.allclose(lp.sum(1),pol.logp(S,H,card,cell),atol=1e-5)
    menu=pol.menu_logp(S,H).exp().sum((1,2));assert torch.allclose(menu,torch.ones(3),atol=1e-4)
