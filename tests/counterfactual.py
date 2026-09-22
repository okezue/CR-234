import random

from sim import replay as R
from train.counterfactual import CELLS,FIELDS,GH,GW,_work,alternatives,cell_of,decision_plays,own_cells,score
from train.decisionStates import extract
from train.rl import CELLS as RL_CELLS


# Counterfactual rollouts replay a recorded game with one play replaced by another hand card or another tile and score the
# simulated result for the actor; the recorded play rolled out the same way must reproduce the plain replay exactly.


def row(name,team,ticks,x=9,y=None):
    return {'card':name,'team':team,'time':ticks,'tile_x':x,'tile_y':(10 if team=='blue' else 21) if y is None else y,'card_type':'normal','ability':0}


def t_cells_and_menus():
    assert CELLS==RL_CELLS
    assert cell_of(0,0)==0 and cell_of(17.9,31.9)==GW*GH-1 and cell_of(9,10)==2*GW+3 and all(cell_of(*t)==i for i,c in enumerate(CELLS) for t in c)
    assert len(own_cells('blue'))==len(own_cells('red'))==GW*GH//2 and all(i//GW<4 for i in own_cells('blue'))
    rec={'card':'knight','x':9.0,'y':10.0,'team':'blue','hand':'knight|archers|fireball|giant'}
    menu=alternatives(rec,random.Random(1),3)
    # other hand cards at the recorded tile first, then the recorded card somewhere else on the actor's side
    assert sorted(c for c,_,_ in menu)==['archers','fireball','giant'] and all((x,y)==(9.0,10.0) for _,x,y in menu)
    menu=alternatives({**rec,'hand':'knight|archers'},random.Random(1),3)
    assert [c for c,_,_ in menu]==['archers','knight','knight'] and all(cell_of(x,y) in own_cells('blue') and cell_of(x,y)!=cell_of(9,10) for _,x,y in menu[1:])
    spell=alternatives({**rec,'card':'fireball','hand':'fireball'},random.Random(2),4)
    assert all(c=='fireball' for c,_,_ in spell) and any(cell_of(x,y) not in own_cells('blue') for _,x,y in spell)


def t_recorded_arm_reproduces_the_plain_replay_and_alternatives_change_the_play():
    plays=[row('knight','blue',40),row('archers','red',100),row('ability-royal-ghost','red',120),row('musketeer','blue',400)]
    plays[2]['ability']=1
    outcome={'result':'W','tc':1,'oc':0,'b_deck':['knight','musketeer','fireball','giant'],'r_deck':['archers','goblins','arrows','valkyrie']}
    assert decision_plays(plays)==[0,1,3]
    game,recs=extract('synthetic',plays,outcome)
    g,_=R.replay_battle('synthetic',plays,outcome);plain=score(g,'blue')
    rec={'idx':2,'team':'blue','card':'musketeer','x':9.0,'y':10.0,'hand':recs[2]['hand']}
    rows=_work(('synthetic',plays,outcome,None,[rec],2,0))
    assert len(rows)==3 and all(len(r)==len(FIELDS) for r in rows) and [r[2] for r in rows]==[0,1,2]
    bid,idx,alt,team,name,x,y,win,margin,ret=rows[0]
    assert (name,x,y)==('musketeer',9.0,10.0) and (win,margin,ret)==(plain['win'],plain['margin'],plain['ret'])
    hand=recs[2]['hand'].split('|')
    for r in rows[1:]:
        assert r[4] in hand and (r[4]!='musketeer' or (r[5],r[6])!=(9.0,10.0))
    # the recorded plays are untouched by the rollouts
    assert plays[3]['card']=='musketeer' and plays[3]['tile_x']==9
