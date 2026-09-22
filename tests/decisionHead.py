import json

import numpy as np

from train.decisionHead import PHASES,design,phase,run,split,verdict
from train.decisionStates import COLS
from train.feats import FEAT_DIM


# The decision head learns P(actor wins | state, play) from recorded winners, with the simulator's verdict as a witness weighted per
# game phase. Synthetic records plant the structure the head must recover: a board feature that predicts the winner only late in the
# game, and a simulator verdict that agrees with the recorded winner far more often than chance.


def synthetic(tmp_path,n_games=400,seed=0,agree=0.85):
    rng=np.random.default_rng(seed);rows={c:[] for c in COLS};X=[];games=[]
    for g in range(n_games):
        bid=f'G{g:05d}';winner='blue' if rng.random()<0.5 else 'red'
        sim=winner if rng.random()<agree else ('red' if winner=='blue' else 'blue')
        games.append({'bid':bid,'actual_winner':winner,'sim_winner':sim,'end_t':300.0,'premature':False,'actual_bc':1,'actual_rc':0,'sim_bc':1,'sim_rc':0,'n':0})
        for i in range(24):
            t=i*12.5;team='blue' if i%2==0 else 'red';s=rng.normal(0,1,FEAT_DIM).astype(np.float32)
            # feature 30 carries the actor's tower lead, but only once the game is past two minutes
            lead=(1.0 if winner==team else -1.0)
            s[30]=lead*2.0+rng.normal(0,0.5) if t>=120 else rng.normal(0,1)
            X.append(s.astype(np.float16))
            for c,v in zip(COLS,(bid,i,t,team,'knight' if i%3 else 'fireball',9.0,10.0 if team=='blue' else 21.0,False,False,1)):rows[c].append(v)
    npz=tmp_path/'synthetic.npz';np.savez_compressed(npz,X=np.stack(X),**{c:np.array(v) for c,v in rows.items()})
    (tmp_path/'synthetic.json').write_text(json.dumps({'outcomes':games})+'\n')
    return npz


def t_phase_and_verdict_helpers():
    assert [phase(t) for t in (0,59.9,60,119.9,120,179.9,180,299)]==[0,0,1,1,2,2,3,3] and len(PHASES)==4
    g={'sim_winner':'blue'};assert verdict(g,'blue')==1.0 and verdict(g,'red')==-1.0 and verdict({'sim_winner':None},'blue')==0.0


def t_design_drops_draws_and_splits_games_whole(tmp_path):
    npz=synthetic(tmp_path,n_games=40)
    from train.decisionStates import load
    cols,X,games=load(npz);games['G00000']['actual_winner']=None
    d=design(cols,X,games)
    assert len(d['y'])==39*24 and 'G00000' not in set(d['gid'])
    assert d['state'].shape[1]==d['n_state']+d['n_card']+4+len(PHASES) and d['both'].shape[1]==d['state'].shape[1]+len(PHASES)
    tr,te=split(d['gid'],0.3)
    assert not (set(d['gid'][tr])&set(d['gid'][te])) and tr.sum()+te.sum()==len(d['y'])


def t_head_finds_the_late_board_signal_and_weights_the_verdict_early(tmp_path):
    npz=synthetic(tmp_path);r=run(npz,holdout=0.25,epochs=150)
    pp=r['per_phase']
    # the state alone is near chance before two minutes and strong after, so the earliest phase reaching 70% is the third
    assert pp['0to60']['state']['accuracy']<0.6 and pp['120to180']['state']['accuracy']>0.85,pp
    assert r['earliest_phase']['0.7']=='120to180'
    # the verdict carries the game early and adds little once the board decides it: its weight falls from the first phase to the third
    w=r['verdict_weight_by_phase'];assert w['0to60']>0.5 and w['120to180']<w['0to60'],w
    # the combined head beats the bare simulator verdict late, where the board is informative, and is at least as good as the state alone
    assert pp['120to180']['both']['log_loss']<pp['120to180']['simulator']['log_loss']
    assert r['models']['both']['log_loss']<=r['models']['state']['log_loss']+1e-3


def t_witness_table_reads_the_planted_late_lead(tmp_path):
    from train.decisionHead import witness
    from train.decisionStates import load
    npz=synthetic(tmp_path,n_games=200);cols,X,games=load(npz);d=design(cols,X,games)
    w=witness(d,d['raw'])
    # feature 30 is not a tower slot, so the synthetic tower lead is random noise: near chance against both labels in every phase, while
    # the planted 85% verdict agreement is read back within sampling error
    for label,row in w.items():
        assert abs(row['verdict_predicts_recorded']-0.85)<0.06,row
        assert row['lead_predicts_recorded'] is None or abs(row['lead_predicts_recorded']-0.5)<0.1,row
    assert set(w)=={'0to60','60to120','120to180','overtime'}
