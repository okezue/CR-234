import numpy as np

from sim import replay as R
from train.decisionStates import COLS,Recorder,extract,label
from train.feats import FEAT_DIM,GAME_FEAT


# The decision-state extractor wraps the shipped replay path: one record per recorded card play, taken from the acting player's side
# just before the play is submitted, labelled with the recorded and the simulated outcome of the whole game.


def row(name,team,ticks,x=9,y=None):
    return {'card':name,'team':team,'time':ticks,'tile_x':x,'tile_y':(10 if team=='blue' else 21) if y is None else y,'card_type':'normal','ability':0}


def t_one_record_per_recorded_card_play_from_the_actors_side():
    plays=[row('knight','blue',40),row('archers','red',100),row('musketeer','blue',400),row('ability-royal-ghost','red',420)]
    plays[-1]['ability']=1
    outcome={'result':'W','tc':1,'oc':0}
    game,recs=extract('synthetic',plays,outcome)
    assert game['n']==len(recs)==3 and game['actual_winner']=='blue' and game['sim_winner'] in ('blue','red',None)
    assert [r['team'] for r in recs]==['blue','red','blue'] and [r['card'] for r in recs]==['knight','archers','musketeer']
    assert [r['t'] for r in recs]==[2.0,5.0,20.0]
    assert all(r['state'].shape==(FEAT_DIM,) and r['state'].dtype==np.float16 for r in recs)
    # the state is the actor's view: the clock feature is the recorded time, and the acting player's elixir is the forced ten
    assert abs(float(recs[2]['state'][0])-20.0/300.0)<1e-3
    assert float(recs[2]['state'][8])==1.0
    # the recorded card sits in the forced hand of four, the menu counterfactual plays are drawn from
    assert all(r['card'] in r['hand'].split('|') and len(r['hand'].split('|'))==4 for r in recs)
    assert R.Game is not Recorder and Recorder.records is None and COLS[-1]=='hand'


def t_a_relocated_placement_is_one_decision_at_its_recorded_tile():
    # a blue troop recorded on the enemy side is rejected and relocated by the replay; the retries add no record
    plays=[row('knight','blue',40,x=9,y=25),row('archers','blue',200)]
    game,recs=extract('synthetic',plays,{'result':'W','tc':1,'oc':0})
    assert game['n']==len(recs)==2 and (recs[0]['x'],recs[0]['y'])==(9,25) and recs[1]['card']=='archers'


def t_labels_follow_the_recorded_winner_and_draws_carry_none():
    assert label({'actual_winner':'blue'},'blue')==1.0 and label({'actual_winner':'blue'},'red')==0.0
    assert label({'actual_winner':None},'blue') is None


def t_the_state_precedes_the_play_and_tower_features_are_read_from_the_arena():
    plays=[row('giant','blue',20)]
    game,recs=extract('synthetic',plays,{'result':'L','tc':0,'oc':1})
    s=recs[0]['state'].astype(np.float32)
    # no troop has been placed when the first decision is taken, so the unit block is empty and all six towers stand at full health
    unit_block=s[GAME_FEAT+6*6+20:GAME_FEAT+6*6+20+24*2*12]
    assert not unit_block.any()
    towers=s[GAME_FEAT:GAME_FEAT+36].reshape(6,6)
    assert (towers[:,0]==1).all() and (towers[:,2]==1).all() and towers[:,3].sum()==3
    assert len(COLS)==11 and game['actual_winner']=='red'
