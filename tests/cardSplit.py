import json

import numpy as np
import torch

from train.cardSplit import compare,perm_p,recorded_advantages,run,table,variant
from train.counterfactual import CELLS,load_groups,own_cells
from train.decisionStates import COLS
from train.feats import FEAT_DIM
from train.traceRl import prepare,stack


# The per-card split averages the world model's advantage of the recorded play and the real advantage per card variant and asks
# whether their gap follows the calibrated head's distrust. Synthetic games have no real structure (the winner is a coin) and a world
# model whose judgement of the recorded play carries a planted offset per card variant, so the gap must recover the offsets and rank
# with a head table built from them.

CARDS=('knight','archers','fireball','giant','musketeer','valkyrie')
OFFSET={'knight-ev1':1.0,'knight':0.0,'archers':-1.0,'fireball':0.5,'giant':-0.5,'musketeer':0.25,'valkyrie':-0.25}


def synthetic(tmp_path,n_games=300,n_dec=8,seed=0):
    rng=np.random.default_rng(seed);rows={c:[] for c in COLS};X=[];games=[];cf=[]
    for g in range(n_games):
        bid=f'G{g:05d}';winner='blue' if rng.random()<0.5 else 'red'
        games.append({'bid':bid,'actual_winner':winner,'sim_winner':winner,'end_t':300.0,'premature':False,'actual_bc':1,'actual_rc':0,'sim_bc':1,'sim_rc':0,'n':n_dec})
        for i in range(n_dec):
            team='blue' if i%2==0 else 'red';hand=list(rng.choice(CARDS,4,replace=False));card=hand[0];evolved=card=='knight' and rng.random()<0.5
            cell=int(rng.choice(own_cells(team)));x,y=CELLS[cell][0];X.append(rng.normal(0,1,FEAT_DIM).astype(np.float16))
            for c,v in zip(COLS,(bid,i,i*12.5,team,card,float(x)+0.5,float(y)+0.5,bool(evolved),False,1,'|'.join(hand))):rows[c].append(v)
            # the recorded play's simulated return carries the card's offset; the alternatives are noise around zero
            menu=[(card,float(x)+0.5,float(y)+0.5)]+[(c,float(x)+0.5,float(y)+0.5) for c in hand[1:3]]
            for k,(name,px,py) in enumerate(menu):
                ret=(OFFSET[variant(card,evolved)] if k==0 else 0.0)+rng.normal(0,0.3);cf.append((bid,i,k,team,name,px,py,1.0 if ret>0 else -1.0,0.0,ret))
    npz=tmp_path/'synthetic.npz';np.savez_compressed(npz,X=np.stack(X),**{c:np.array(v) for c,v in rows.items()})
    (tmp_path/'synthetic.json').write_text(json.dumps({'outcomes':games})+'\n')
    cfp=tmp_path/'cf.json';cfp.write_text(json.dumps({'rows':cf})+'\n')
    head=tmp_path/'head.json';head.write_text(json.dumps({'cards':{c:{'bias':o/5,'correction':-o/5} for c,o in OFFSET.items()}})+'\n')
    return npz,cfp,head


def t_variant_naming_matches_the_calibrated_head():
    assert variant('archer_queen')=='archer-queen' and variant('knight',evolved=True)=='knight-ev1' and variant('ice_golem',hero=True)=='ice-golem-hero'
    assert variant('the_log',False,False)=='the-log' and variant('knight',True,True)=='knight-ev1'


def t_recorded_advantages_and_table_read_the_planted_offsets(tmp_path):
    npz,cf,head=synthetic(tmp_path,n_games=200);cols,X,games=stack([npz]);d=prepare(cols,X,games)
    ri,adv,win=recorded_advantages(d,load_groups(cf));assert len(ri)==len(d['y'])==1600 and set(np.sign(win).tolist())<={-1.0,1.0}
    ti=torch.tensor(ri);names=np.array([variant(d['vocab'][c],e,h) for c,e,h in zip(d['card'][ti].tolist(),d['evolved'][ri],d['hero'][ri])])
    y=d['y'][ti].numpy();t=table(names,adv,y-0.5,y,min_n=20)
    assert set(t)==set(OFFSET) and all(t[c]['n']>=20 for c in t)
    order=sorted(t,key=lambda c:t[c]['adv_wm']);assert order[0]=='archers' and order[-1]=='knight-ev1' and t['knight']['adv_wm']<t['fireball']['adv_wm']
    # the recorded arm's advantage is two thirds of its offset: the group mean contains it once in three plays
    assert abs(t['knight-ev1']['adv_wm']-2/3)<0.15 and abs(t['archers']['adv_wm']+2/3)<0.15
    c=compare(t,json.loads(head.read_text())['cards'],perms=300)
    assert c['cards']==7 and c['cards_in_head']==7 and c['spearman_gap_vs_bias']>0.9 and c['spearman_gap_vs_correction']<-0.9
    assert c['p_gap_vs_bias']<0.05 and c['gap_most_distrusted']>c['gap_most_trusted']
    a=np.arange(30.0);assert perm_p(a,a,perms=200)<0.01 and perm_p(a,np.random.default_rng(0).permutation(a),perms=200)>0.05


def t_run_fits_the_baseline_outside_the_rollout_games_and_reports_the_split(tmp_path):
    npz,cf,head=synthetic(tmp_path,n_games=300);groups=json.loads(cf.read_text())['rows']
    # rollouts for the first 100 games only, so the value baseline must come from the other 200
    cf.write_text(json.dumps({'rows':[r for r in groups if int(r[0][1:])<100]})+'\n')
    r=run([npz],cf,head,out=tmp_path/'split.json',min_n=20,head_epochs=5,hidden=16,perms=200)
    assert r['rollout_games']==100 and r['baseline_games']==200 and r['groups']==800 and abs(r['baseline_mean']-0.5)<0.1
    assert r['compare']['spearman_gap_vs_bias']>0.8 and r['compare']['spearman_gap_vs_correction']<-0.8 and abs(r['decision_corr_wm_vs_real'])<0.1
    saved=json.loads((tmp_path/'split.json').read_text());assert list(saved['cards'])[0]=='knight-ev1' and saved['cards']['knight-ev1']['bias']==0.2
