import copy

import numpy as np
import torch

from tests.traceRl import HANDS,oracle_groups,right_rate,synthetic
from train.traceRl import Policy,prepare,stack
from train.worldModelBias import agreement,cosine,rank_corr,run,shift,shuffle_games


# The bias measure iterates the clipped update with refreshed anchors and compares the direction of the policy's preference shift
# under the world model with the direction under real outcomes. Synthetic games plant the rule of tests/traceRl.py; a perfect world
# model's iteration must keep drifting from the behaviour policy in the direction of the real-outcome update, while the update on
# flipped outcomes must not share that direction.


def t_shift_measures_the_menu_log_ratio_and_agreement_reads_its_direction(tmp_path):
    npz,_=synthetic(tmp_path,n_games=20);cols,X,games=stack([npz]);d=prepare(cols,X,games);torch.manual_seed(0)
    a=Policy(d['n_state'],len(d['vocab']),hidden=32);b=copy.deepcopy(a);k=d['vocab'].index('knight')
    with torch.no_grad():b.card.bias[k]+=2.0
    zero=shift(a,a,d['S'],d['H']);assert float(zero[0].abs().max())==0.0 and float(zero[1].abs().max())==0.0
    menu,card=shift(b,a,d['S'],d['H']);slot=(d['H']==k).nonzero();others=card.clone();others[slot[:,0],slot[:,1]]=0
    # the raised card gains log-probability in every state and every other hand card loses it; the menu shift is the card shift spread over its cells
    assert (card[slot[:,0],slot[:,1]]>0).all() and (others<0).sum()==3*len(d['S']) and card.shape==(len(d['S']),HANDS.count('|')+1)
    assert torch.allclose(menu.view(len(d['S']),-1,48)[slot[:,0],slot[:,1]],card[slot[:,0],slot[:,1],None].expand(-1,48),atol=1e-5)
    same=agreement((menu,card),(menu,card));flipped=agreement((menu,card),(-menu,-card))
    # the 48 equal cell entries of a card-only shift are ties, so the rank measure of the negation is below one in magnitude
    assert same=={'cos_menu':1.0,'cos_card':1.0,'rank_menu':1.0} and flipped['cos_menu']==-1.0 and flipped['cos_card']==-1.0 and flipped['rank_menu']<-0.7
    x=torch.randn(50,8);assert abs(cosine(x,torch.randn(50,8)))<0.3 and abs(rank_corr(x,x)-1.0)<1e-5 and abs(rank_corr(x,-x)+1.0)<1e-5


def t_shuffle_flips_whole_games_and_about_half_of_them(tmp_path):
    npz,_=synthetic(tmp_path,n_games=200);cols,X,games=stack([npz]);d=prepare(cols,X,games);s=shuffle_games(d,seed=3)
    flipped=(s['y']!=d['y']).numpy()
    for g in np.unique(d['gid']):
        m=d['gid']==g;assert flipped[m].all() or not flipped[m].any()
    assert 0.35<flipped.mean()<0.65 and abs(float(s['y'].mean()-d['y'].mean()))<0.1


def t_oracle_world_model_drifts_in_the_real_direction_and_flipped_outcomes_do_not(tmp_path):
    npz,truth=synthetic(tmp_path,n_games=400);cf=oracle_groups(tmp_path,npz,truth,every=1);pols={}
    r=run([npz],cf,holdout=0.25,checkpoints=(8,16),refresh=4,arms=('simgroup','ppo1','ppo1_shuffled'),head_epochs=10,batch=256,hidden=32,q_subset=0,
          shift_subset=500,bc_epochs=15,policies=pols)
    d,te=pols['data'],pols['test'];sg=r['arms']['simgroup']
    assert set(sg)=={'8','16'} and sg['16']['kl_to_bc']>sg['8']['kl_to_bc']>0 and r['drift']['simgroup'][1]['epochs']==16
    assert right_rate(pols['simgroup'],d,te)>right_rate(pols['bc'],d,te)+0.1
    # every arm shares a component from fitting the recorded plays further (the flipped arm's cosines are the floor), so the oracle's
    # agreement with the real-outcome direction is read above that floor
    same=r['same_epoch']['16'];assert same['simgroup|ppo1']['cos_card']>0.3 and same['simgroup|ppo1']['cos_card']>same['ppo1|ppo1_shuffled']['cos_card']+0.05
    assert same['simgroup|ppo1']['cos_card']>same['simgroup|ppo1_shuffled']['cos_card']+0.05 and 0.3<r['shuffled_fraction']['ppo1_shuffled']<0.7
    assert len(r['agreement'])==15 and r['bc']['kl_to_bc']==0.0
