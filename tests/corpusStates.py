import random

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

import train.corpusStates as CS
from sim import replay as R
from train.corpusStates import PrivRecorder,exclusions,extract,select
from train.decisionStates import Recorder
from train.decisionStates import extract as plain_extract


# The corpus extractor wraps the shipped decision recorder: the same states as train.decisionStates, plus the opponent's replay hand
# and next card and a no-play rollout of the simulator from each decision state that must leave the replay itself unchanged.


def row(name,team,ticks,x=9,y=None):
    return {'card':name,'team':team,'time':ticks,'tile_x':x,'tile_y':(10 if team=='blue' else 21) if y is None else y,'card_type':'normal','ability':0}


PLAYS=[row('hog-rider','blue',40,x=3,y=15),row('archers','red',120,x=3,y=26),row('musketeer','blue',160,x=4,y=8),row('knight','red',200,x=3,y=20),
       row('fireball','blue',260,x=3,y=24),row('valkyrie','red',300,x=14,y=24)]
OUT={'result':'W','tc':1,'oc':0,'b_deck':['hog-rider','musketeer','fireball','arrows','knight','archers','minions','valkyrie'],
     'r_deck':['archers','knight','valkyrie','goblins','zap','hog-rider','musketeer','giant']}


def t_rollouts_leave_the_replay_unchanged_and_the_states_match_the_shipped_recorder():
    g0,r0=plain_extract('synthetic',PLAYS,OUT);g1,r1=extract('synthetic',PLAYS,OUT,horizon=0.0);g2,r2=extract('synthetic',PLAYS,OUT,horizon=10.0)
    assert len(r0)==len(r1)==len(r2)==6 and all(g0[k]==g1[k]==g2[k] for k in g0) and g1['sim_blue']==g2['sim_blue']
    assert all((a['state']==b['state']).all() and (a['state']==c['state']).all() and a['card']==c['card'] for a,b,c in zip(r0,r1,r2))
    assert R.Game is not PrivRecorder and PrivRecorder.records is None and Recorder.records is None
    # without a horizon the rollout features are zero; with one they are tower shares in (0, 0.5], a crown count and the seconds run
    assert all(r['roll']==(0.0,0.0,0.0,0.0) for r in r1)
    assert all(0<r['roll'][0]<=0.5 and 0<r['roll'][1]<=0.5 and 0<r['roll'][3]<=10.0+1e-6 for r in r2)


def t_a_rollout_that_draws_random_numbers_restores_the_random_state(monkeypatch):
    # a graveyard is still dropping skeletons at random points when red decides, so the forecast consumes random numbers; the replay
    # must go on from the random state it had before the forecast
    plays=[row('graveyard','blue',40,x=3,y=26)]+PLAYS[1:];out=dict(OUT,b_deck=['graveyard']+OUT['b_deck'][1:])
    uni=random.uniform;roll=CS.rollout;draws=[0];seen=[]
    def count(*a):
        draws[0]+=1;return uni(*a)
    def spy(g,team,h):
        st=random.getstate();n=draws[0];r=roll(g,team,h);seen.append((draws[0]>n,random.getstate()==st));return r
    monkeypatch.setattr(random,'uniform',count);monkeypatch.setattr(CS,'rollout',spy)
    g0,r0=extract('synthetic',plays,out,horizon=0.0);g1,r1=extract('synthetic',plays,out,horizon=10.0)
    assert seen and any(d for d,_ in seen) and all(ok for _,ok in seen),seen
    assert all(g0[k]==g1[k] for k in g0) and all((a['state']==b['state']).all() for a,b in zip(r0,r1))


def t_the_rollout_forecasts_the_board_and_stops_on_an_empty_field():
    _,recs=extract('synthetic',PLAYS,OUT,horizon=10.0)
    # the first decision faces an empty field: the rollout stops after its first second and nothing changes
    assert recs[0]['roll'][3]<=1.1 and recs[0]['roll'][0]==recs[0]['roll'][1]
    # red's first decision comes with a blue hog rider at the bridge: in red's no-play forecast red's own towers lose health
    red=recs[1];assert red['team']=='red' and red['roll'][0]<red['roll'][1]


def t_the_opponent_hand_is_the_replays_reconstructed_cycle():
    _,recs=extract('synthetic',PLAYS,OUT)
    red_first={p['card'].replace('-','_') for p in PLAYS if p['team']=='red'}
    for r in recs:
        hand=r['opp_hand'].split('|');assert len(hand)==4 and r['opp_next'] not in hand and r['opp_next']
    # the replay starts the opponent's hand with its first distinct recorded plays, so blue's first decision sees red's first three cards
    assert red_first<=set(recs[0]['opp_hand'].split('|'))|{recs[0]['opp_next']}


def t_selection_orders_by_time_and_drops_modes_duplicates_and_excluded_ids(tmp_path):
    rows=[('A1',30.0,'Ranked',True,True),('A2',10.0,'Ladder',True,True),('A3',20.0,'C.H.A.O.S Draft',True,True),('A4',5.0,'Ranked',False,True),
          ('A5',15.0,'Ranked',True,False),('A6',25.0,'1v1 Showdown',True,True),('#A7',1.0,'Ranked',True,True),('A8',None,'Ranked',True,True)]
    cols=list(zip(*rows));t=pa.table({'replayTag':list(cols[0]),'battle_ts':list(cols[1]),'gameMode_name':list(cols[2]),'one_v_one':list(cols[3]),'has_replay':list(cols[4])})
    pq.write_table(t,tmp_path/'battles.parquet');(tmp_path/'ex.txt').write_text('A6\n#A9\n')
    (tmp_path/'ex.csv').write_text('replayTag,x\n#A1,1\n')
    ex=exclusions([tmp_path/'ex.txt',tmp_path/'ex.csv']);tags,ts=select(tmp_path,ex)
    assert ex=={'A6','A9','A1'} and tags==['A2'] and ts=={'A2':10.0}
    tags,ts=select(tmp_path,set());assert tags==['A2','A6','A1'] and np.all(np.diff([ts[k] for k in tags])>0)
