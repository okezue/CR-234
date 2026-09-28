import copy

import numpy as np
import torch

from tests.traceRl import oracle_groups,right_card,synthetic
from train.counterfactual import load_groups
from train.jointQ import fit_joint,group_rows,ranks,rollout_rows,run,sim_agreement
from train.traceRl import Head,prepare,split,stack


# The joint head learns P(win | state, play) from real labels and world-model rollout labels with a trust weight on the rollout loss.
# Synthetic games plant the rule of tests/traceRl.py and a perfect world model labels every menu play; the head must absorb the
# world model's ranking as the trust grows, prefer the right card more, improve on the real outcomes (the world model is right here),
# and reduce exactly to the real-only head at zero trust.


def right_pref(head,d,te):
    # mean logit of the right card minus the wrong alternative at the recorded cell on held-out states
    S,cell=d['S'][te],d['cell'][te];idx={c:i for i,c in enumerate(d['vocab'])};c=np.sign(S[:,40].numpy())
    right=torch.tensor([idx[right_card(v)] for v in c]);wrong=torch.tensor([idx['archers' if right_card(v)=='knight' else 'knight'] for v in c])
    with torch.no_grad():return float((head(S,right,cell)-head(S,wrong,cell)).mean())


def t_rollout_rows_are_contiguous_groups_from_the_hand(tmp_path):
    npz,truth=synthetic(tmp_path,n_games=60);cf=oracle_groups(tmp_path,npz,truth,every=2);cols,X,games=stack([npz]);d=prepare(cols,X,games)
    tr,te=split(d['gid'],0.25);roll=rollout_rows(d,tr.numpy(),load_groups(cf))
    n=len(roll['ri']);assert roll['groups']==int(tr.sum())//2 and int(roll['lens'].sum())==n and (roll['lens']==3).all() and roll['S'].shape==(n,d['n_state'])
    assert (roll['gid'][1:]>=roll['gid'][:-1]).all() and set(roll['p'].tolist())=={0.0,1.0}
    assert all(int(c) in set(d['H'][int(r)].tolist()) for r,c in zip(roll['ri'],roll['card']))
    # the group-relative return is centred and the oracle's right card is the top of every group
    rel=roll['rel'];assert abs(float(rel.view(-1,3).sum(1).abs().max()))<1e-5 and (rel.view(-1,3).max(1).values>1.0).all()
    rows=group_rows(roll,torch.tensor([2,0]));assert rows.tolist()==[6,7,8,0,1,2]
    assert rollout_rows(d,np.zeros(len(d['y']),bool),load_groups(cf)) is None
    assert ranks(np.array([2.0,-1.0,-1.0])).tolist()==[2.0,0.5,0.5] and ranks(np.array([0.3,0.1,0.2])).tolist()==[2.0,0.0,1.0]


def t_trust_grows_the_world_model_share_and_zero_trust_is_the_real_head(tmp_path):
    npz,truth=synthetic(tmp_path,n_games=400);cf=oracle_groups(tmp_path,npz,truth,every=1);heads={}
    r=run([npz],cf,holdout=0.25,lams=(0.3,10.0),targets=('win','relative'),trust=[0,0,0,0],head_epochs=6,hidden=16,head_l2=1e-2,batch=256,heads=heads)
    d,te=heads['data'],heads['test'];a=r['arms'];real=a['real']
    assert real['vs_real']['log_loss_diff']==0.0 and real['games']==r['test_games'] and r['heldout_rollout_groups']==real['n']
    # the same seed and batch order with every rollout weight zero reproduces the real-only head to the bit; another seed does not
    assert a['win:0.3:phase']['log_loss']==real['log_loss'] and a['relative:10:phase']['vs_real']['log_loss_diff']==0.0
    assert a['real:s1']['vs_real']['log_loss_diff']!=0.0 and a['real:s1']['vs_real']['se']>0
    for target in ('win','relative'):
        prefs=[right_pref(heads[k],d,te) for k in ('real',f'{target}:0.3',f'{target}:10')]
        assert prefs[0]+0.05<prefs[1]<prefs[2],(target,prefs)
        assert a[f'{target}:10']['log_loss']<real['log_loss']-0.05 and a[f'{target}:10']['vs_real']['games_better']>0.6,(target,a[f'{target}:10'])
    sim,base=a['relative:0.3']['sim'],real['sim'];assert sim['top1']>base['top1']+0.05 and sim['rank_corr']>base['rank_corr'] and sim['groups']==base['groups']


def t_relative_target_constrains_only_the_comparison_within_a_state_and_phase_trust_scales_it(tmp_path):
    npz,truth=synthetic(tmp_path,n_games=100);cf=oracle_groups(tmp_path,npz,truth,every=1);cols,X,games=stack([npz]);d=prepare(cols,X,games)
    tr,te=split(d['gid'],0.25);roll=rollout_rows(d,tr.numpy(),load_groups(cf));S,card,cell,y=(d[k][tr] for k in ('S','card','cell','y'))
    torch.manual_seed(0);h=Head(d['n_state'],len(d['vocab']),hidden=16);h0=copy.deepcopy(h)
    fit_joint(h,S,card,cell,y*0+0.5,roll,lam=30.0,target='relative',epochs=15,batch=64)
    # with the real labels flat at one half the level stays near zero while the plays inside a group spread to the returns
    with torch.no_grad():logit=h(roll['S'],roll['card'],roll['cell'])
    rel=(logit.view(-1,3)-logit.view(-1,3).mean(1,keepdim=True)).flatten()
    assert abs(float(logit.mean()))<0.3 and float(np.corrcoef(rel.numpy(),roll['rel'].numpy())[0,1])>0.9 and sim_agreement(h,roll)['top1']>0.9
    h1=copy.deepcopy(h0);fit_joint(h1,S,card,cell,y*0+0.5,roll,lam=30.0,target='relative',trust=[1,1,1,1],epochs=15,batch=64)
    assert all(torch.equal(p,q) for p,q in zip(h.parameters(),h1.parameters()))
    h2=copy.deepcopy(h0);fit_joint(h2,S,card,cell,y*0+0.5,roll,lam=30.0,target='relative',trust=[0.5,0.5,0.5,0.5],epochs=15,batch=64)
    assert not all(torch.equal(p,q) for p,q in zip(h.parameters(),h2.parameters()))
