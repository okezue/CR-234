import json

import numpy as np
import torch

from train.counterfactual import CELLS,cell_of,own_cells
from train.decisionStates import COLS
from train.feats import FEAT_DIM
from train.traceRl import HAND,N_CELLS,Policy,prepare,run,stack


# Learning from one recorded trajectory per game with only the winner as feedback. Synthetic games plant a decision rule the
# behaviour policy follows only 60 percent of the time (feature 40 says which of two cards is right), and the winner is drawn from
# the two players' shares of right plays. Outcome-driven updates must move the policy toward the right card; the constrained ones
# must stay close to the behaviour policy; a perfect world model's group advantages must do the same, and zero trust must do nothing.

HANDS='knight|archers|fireball|giant'


def right_card(c):
    return 'knight' if c>0 else 'archers'


def synthetic(tmp_path,n_games=600,seed=0,follow=0.6,n_dec=8):
    rng=np.random.default_rng(seed);rows={c:[] for c in COLS};X=[];games=[];truth={}
    for g in range(n_games):
        bid=f'G{g:05d}';share={'blue':0,'red':0};recs=[]
        for i in range(n_dec):
            team='blue' if i%2==0 else 'red';s=rng.normal(0,1,FEAT_DIM).astype(np.float32);c=1.0 if rng.random()<0.5 else -1.0;s[40]=c*3.0
            right=right_card(c);hand=HANDS.split('|')
            card=right if rng.random()<follow else rng.choice([h for h in hand if h!=right])
            share[team]+=card==right;cell=int(rng.choice(own_cells(team)));x,y=CELLS[cell][0]
            X.append(s.astype(np.float16));recs.append((bid,i,i*12.5,team,card,float(x)+0.5,float(y)+0.5,False,False,1,HANDS));truth[(bid,i)]=right
        q=(share['blue']-share['red'])/(n_dec/2);winner='blue' if rng.random()<1/(1+np.exp(-6*q)) else 'red'
        games.append({'bid':bid,'actual_winner':winner,'sim_winner':winner,'end_t':300.0,'premature':False,'actual_bc':1,'actual_rc':0,'sim_bc':1,'sim_rc':0,'n':n_dec})
        for r in recs:
            for c,v in zip(COLS,r):rows[c].append(v)
    npz=tmp_path/'synthetic.npz';np.savez_compressed(npz,X=np.stack(X),**{c:np.array(v) for c,v in rows.items()})
    (tmp_path/'synthetic.json').write_text(json.dumps({'outcomes':games})+'\n')
    return npz,truth


def oracle_groups(tmp_path,npz,truth,every=1):
    # a perfect world model: every menu play returns +1 when it is the right card and -1 otherwise
    cols,X,games=stack([npz]);rows=[]
    for j,(bid,i,card,x,y,team) in enumerate(zip(cols['bid'],cols['idx'],cols['card'],cols['x'],cols['y'],cols['team'])):
        if int(i)%every:continue
        menu=[(str(card),float(x),float(y))]+[(c,float(x),float(y)) for c in HANDS.split('|') if c!=str(card)][:2]
        for k,(name,px,py) in enumerate(menu):
            r=1.0 if name==truth[(str(bid),int(i))] else -1.0;rows.append((str(bid),int(i),k,str(team),name,px,py,r,0.0,r))
    out=tmp_path/'cf.json';out.write_text(json.dumps({'rows':rows})+'\n')
    return out


def right_rate(pol,d,te):
    # probability the policy puts on the right card of each held-out state
    S,H=d['S'][te],d['H'][te];c=torch.tensor(np.sign(d['S'][te][:,40].numpy()))
    with torch.no_grad():lc,_=pol.card_logp(S,H)
    idx={c_:i for i,c_ in enumerate(d['vocab'])}
    right=torch.tensor([idx[right_card(v)] for v in c.tolist()])
    return float(lc.exp().gather(1,right[:,None]).mean())


def t_prepare_builds_hand_menus_cells_and_labels(tmp_path):
    npz,_=synthetic(tmp_path,n_games=30)
    cols,X,games=stack([npz]);d=prepare(cols,X,games)
    assert d['H'].shape==(240,HAND) and (d['H']>=0).all() and all(d['H'][r,0]==d['card'][r] for r in range(240))
    assert d['cell'].min()>=0 and d['cell'].max()<N_CELLS and set(d['y'].tolist())=={0.0,1.0}
    blue=torch.tensor([t=='blue' for t in cols['team']]);assert all(cell_of(x,y)<N_CELLS//2 for x,y in zip(cols['x'][blue],cols['y'][blue]))
    pol=Policy(d['n_state'],len(d['vocab']),hidden=32);lp=pol.logp(d['S'][:5],d['H'][:5],d['card'][:5],d['cell'][:5])
    menu=pol.menu_logp(d['S'][:5],d['H'][:5])
    assert lp.shape==(5,) and menu.shape==(5,HAND,N_CELLS) and torch.allclose(menu.exp().sum((1,2)),torch.ones(5),atol=1e-4)


def t_outcome_updates_move_toward_the_right_card_and_the_constraints_hold(tmp_path):
    npz,truth=synthetic(tmp_path);pols={}
    r=run([npz],holdout=0.25,epochs=40,head_epochs=40,methods=('bc','reinforce','ppo1','awr'),hidden=64,policies=pols)
    d,te=pols['data'],pols['test'];rates={m:right_rate(pols[m],d,te) for m in ('bc','reinforce','ppo1','awr')}
    # behaviour cloning reproduces the 60 percent habit; every outcome-driven update improves on it from the single trajectory
    assert 0.5<rates['bc']<0.7,rates
    assert rates['ppo1']>rates['bc']+0.01 and rates['awr']>rates['bc']+0.01 and rates['reinforce']>rates['bc'],rates
    m=r['methods']
    # the winner gap of the outcome-driven policies exceeds the behaviour policy's, and the constrained updates stay near it
    assert m['ppo1']['winner_gap']>m['bc']['winner_gap'] and m['awr']['winner_gap']>m['bc']['winner_gap'],m
    assert m['ppo1']['kl_to_bc']<m['reinforce']['kl_to_bc'] and m['awr']['kl_to_bc']<m['reinforce']['kl_to_bc'],m
    assert m['bc']['kl_to_bc']==0.0 and m['bc']['q_policy']==m['bc']['q_bc']
    # the Q-eval scores a policy with more right plays higher than the behaviour policy
    assert m['ppo1']['q_policy']>m['bc']['q_bc'] or m['awr']['q_policy']>m['bc']['q_bc'],m


def t_world_model_group_advantages_learn_the_rule_and_zero_trust_changes_nothing(tmp_path):
    npz,truth=synthetic(tmp_path,n_games=300);cf=oracle_groups(tmp_path,npz,truth,every=2)
    pols={};r=run([npz],holdout=0.25,epochs=25,head_epochs=20,cf=cf,methods=('bc','simgroup'),hidden=64,policies=pols,refresh=5)
    d,te=pols['data'],pols['test']
    assert r['simgroup_rows']>0 and right_rate(pols['simgroup'],d,te)>right_rate(pols['bc'],d,te)+0.1
    frozen={};r0=run([npz],holdout=0.25,epochs=10,head_epochs=5,cf=cf,trust=[0,0,0,0],methods=('bc','simgroup_trust'),hidden=64,policies=frozen)
    same=right_rate(frozen['simgroup_trust'],frozen['data'],frozen['test'])-right_rate(frozen['bc'],frozen['data'],frozen['test'])
    assert r0['methods']['simgroup_trust']['kl_to_bc']<1e-4 and abs(same)<1e-3 and 'simgroup' not in r0['methods']
