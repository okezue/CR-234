import math
from sim.cards import card,create
from sim.game import Game
from tests.util import Dummy,quiet


# The evolved Cannon's deploy barrage: 9 cannonballs in 2 rows, 5 on the top row and 4 on the bottom (wiki Cannon/Evolution); the far row
# of 5 lands 7 tiles in front of the cannon and the near row of 4 is aligned with its front, the barrage hits air and ground and a unit
# covered by several circles takes damage once (RoyaleAPI's Cannon Evolution post, read through search summaries since the page refuses
# automated fetches; Theria Games' guide states the air and ground targeting and the single hit directly). Radius 2, 282 at level 11, 89
# to crown towers and the 1 tile knockback are unchanged. The balls land on fixed arena columns whatever the cannon's x, near row at x 1.5,
# 6.5, 11.5 and 16.5 and far row at x 1, 5, 9, 13 and 17 (the sources say the spread is fixed and spans the arena; the columns are read
# from the wiki's preview animation calibrated on the bridges, the lane path and the river pillar, local/evoCannonRound2/findings.md).


def barrage(cx=9.0,cy=8.0,units=(),team='blue'):
    # an evolved cannon fires its barrage at (cx, cy) among still red dummies given as (x, y, air); towers out of the way
    g=Game()
    for t in g.arena.towers:t.alive=False
    ds=[]
    for x,y,air in units:
        d=Dummy('red' if team=='blue' else 'blue',x,y,hp=50000,spd=0,dmg=0)
        if air:d.transport='Air'
        g.deploy(d.team,d);ds.append(d)
    cn=create('cannon',11,team,cx,cy,evolved=True);g.deploy(team,cn);g.run(0.1)
    return g,cn,ds


def t_evo_cannon_record_carries_the_sourced_rows_and_targets():
    v=card('cannon')['evo']['skills']['volley'];src=card('cannon')['src']
    assert (v['projectileCount'],v['radius'],v['knockback'])==(9,2.0,1),v
    assert v['farRowCount']==5 and v['farRowDistance']==7.0 and v['targets']==['air','ground'],v
    for f in ('farRowCount','farRowDistance','targets'):assert src[f'evo.skills.volley.{f}']=='patch:2026-10-01ec',src
    ev=next(c for c in create('cannon',11,'blue',9,8,evolved=True).components if type(c).__name__=='EvoCannon')
    assert (len(ev.nearX)+len(ev.farX),len(ev.farX),ev.far,ev.air,ev.r,ev.dmg,ev.ct)==(9,5,7.0,True,2.0,282,89),vars(ev)


def t_evo_cannon_record_carries_the_fixed_ball_columns():
    v=card('cannon')['evo']['skills']['volley'];src=card('cannon')['src']
    assert v['nearRowX']==[1.5,6.5,11.5,16.5] and v['farRowX']==[1.0,5.0,9.0,13.0,17.0],v
    assert len(v['nearRowX'])+len(v['farRowX'])==v['projectileCount'] and len(v['farRowX'])==v['farRowCount'],v
    for f in ('nearRowX','farRowX'):assert src[f'evo.skills.volley.{f}']=='patch:2026-10-01ex',src
    for x in (3.5,9.0,14.5):
        ev=next(c for c in create('cannon',11,'blue',x,8,evolved=True).components if type(c).__name__=='EvoCannon')
        assert (ev.nearX,ev.farX)==(v['nearRowX'],v['farRowX']),vars(ev)


def t_balls_span_the_arena_whatever_the_cannon_x():
    # a cannon in the left lane reaches the right edge columns and one in the right lane the left edge columns, on both rows
    g,cn,ds=barrage(cx=3.5,units=((16.5,8.6,False),(13,15,False),(17,15,False)))
    assert [50000-d.hp for d in ds]==[282,282,282],[50000-d.hp for d in ds]
    g,cn,ds=barrage(cx=14.5,units=((1.5,8.6,False),(1,15,False),(5,15,False)))
    assert [50000-d.hp for d in ds]==[282,282,282],[50000-d.hp for d in ds]


def t_moving_the_cannon_does_not_move_the_balls():
    # cannons at x 8 and 10 (y 6, far row at y 13) both push an air unit at (9, 13.5) straight ahead: the far ball at x 9 hit it each time
    for cx in (8.0,10.0):
        g,cn,(d,)=barrage(cx=cx,cy=6.0,units=((9,13.5,True),))
        assert 50000-d.hp==282 and abs(d.x-9.0)<1e-6 and abs(d.y-14.5)<1e-6,(cx,d.x,d.y)


def t_red_cannon_uses_the_same_columns():
    # a red cannon in its right lane (14.5, 24) lands its far row at y 17 on x 1 and its near row at y 23.4 on x 1.5
    g,cn,ds=barrage(cx=14.5,cy=24.0,units=((1,17,False),(1.5,23.4,False)),team='red')
    assert [50000-d.hp for d in ds]==[282,282],[50000-d.hp for d in ds]


def t_far_row_lands_seven_tiles_ahead_and_nothing_between_the_rows():
    # cannon at (9, 8): near row through its front at y 8.6, far row at y 15; a dummy (collision radius 0.5) is reached within 2.5
    g,cn,(far,between,front)=barrage(units=((9,15,False),(9,11.8,False),(6.5,8.6,False)))
    assert cn.collision_r==0.6
    assert 50000-far.hp==282,50000-far.hp
    assert between.hp==50000,50000-between.hp
    assert 50000-front.hp==282,50000-front.hp


def t_red_cannon_fires_toward_blue():
    g,cn,(far,between)=barrage(cy=24.0,units=((9,17,False),(9,20.2,False)),team='red')
    assert 50000-far.hp==282 and between.hp==50000,(50000-far.hp,50000-between.hp)


def t_barrage_hits_air_units():
    g,cn,(air_far,air_front)=barrage(units=((9,15,True),(6.5,8.6,True)))
    assert 50000-air_far.hp==282 and 50000-air_front.hp==282,(50000-air_far.hp,50000-air_front.hp)


def t_a_unit_under_several_circles_takes_one_ball():
    # (7, 15) is reached by the far balls at x 5 and 9 (2 from each) and (3, 15) by those at x 1 and 5
    g,cn,ds=barrage(units=((7,15,False),(7,15,True),(3,15,False)))
    assert [50000-d.hp for d in ds]==[282,282,282],[50000-d.hp for d in ds]


def t_the_knockback_comes_from_the_ball_that_hit():
    # the only ball reaching (7, 10.5) is the near-row ball at (6.5, 8.6): one 1 tile push straight away from it
    g,cn,(d,)=barrage(units=((7,10.5,False),))
    ux,uy=0.5/math.hypot(0.5,1.9),1.9/math.hypot(0.5,1.9)
    assert abs(d.x-(7+ux))<1e-6 and abs(d.y-(10.5+uy))<1e-6,(d.x,d.y)


def t_crown_tower_takes_one_ball_once():
    # the far row of a cannon at (3.5, 18.5) lands on the red left princess tower (3.5, 25.5): the balls at x 1 and 5 reach it
    g=quiet(Game());tw=g.arena.get_tower('red','princess','left');ini=tw.hp
    cn=create('cannon',11,'blue',3.5,18.5,evolved=True);g.deploy('blue',cn);g.run(0.1)
    assert ini-tw.hp==89,ini-tw.hp
