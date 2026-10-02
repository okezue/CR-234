import math
from sim.cards import card,create
from sim.game import Game
from tests.util import Dummy,quiet


# The evolved Cannon's deploy barrage: 9 cannonballs in 2 rows, 5 on the top row and 4 on the bottom (wiki Cannon/Evolution); the far row
# of 5 lands 7 tiles in front of the cannon and the near row of 4 is aligned with its front, the barrage hits air and ground and a unit
# covered by several circles takes damage once (RoyaleAPI's Cannon Evolution post, read through search summaries since the page refuses
# automated fetches; Theria Games' guide states the air and ground targeting and the single hit directly). Radius 2, 282 at level 11, 89
# to crown towers and the 1 tile knockback are unchanged. The x spread stays the engine's approximation: 2 tiles apart, centred on the
# cannon (the sources say it is fixed and spans the arena but give no positions).


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
    assert (ev.n,ev.nfar,ev.far,ev.air,ev.r,ev.dmg,ev.ct)==(9,5,7.0,True,2.0,282,89),vars(ev)


def t_far_row_lands_seven_tiles_ahead_and_nothing_between_the_rows():
    # cannon at (9, 8): near row through its front at y 8.6, far row at y 15; a dummy (collision radius 0.5) is reached within 2.5
    g,cn,(far,between,front)=barrage(units=((9,15,False),(9,11.8,False),(9,8.6,False)))
    assert cn.collision_r==0.6
    assert 50000-far.hp==282,50000-far.hp
    assert between.hp==50000,50000-between.hp
    assert 50000-front.hp==282,50000-front.hp


def t_red_cannon_fires_toward_blue():
    g,cn,(far,between)=barrage(cy=24.0,units=((9,17,False),(9,20.2,False)),team='red')
    assert 50000-far.hp==282 and between.hp==50000,(50000-far.hp,50000-between.hp)


def t_barrage_hits_air_units():
    g,cn,(air_far,air_front)=barrage(units=((9,15,True),(9,8.6,True)))
    assert 50000-air_far.hp==282 and 50000-air_front.hp==282,(50000-air_far.hp,50000-air_front.hp)


def t_a_unit_under_several_circles_takes_one_ball():
    # (9, 10.5) is reached by the near balls at x 8 and 10 (2.15 from each); (8, 15) by the far balls at x 7 and 9
    g,cn,ds=barrage(units=((9,10.5,False),(8,15,False),(8,15,True)))
    assert [50000-d.hp for d in ds]==[282,282,282],[50000-d.hp for d in ds]


def t_the_knockback_comes_from_the_ball_that_hit():
    # nearest ball to (8.5, 10.5) is the near-row ball at (8, 8.6): one 1 tile push straight away from it
    g,cn,(d,)=barrage(units=((8.5,10.5,False),))
    ux,uy=0.5/math.hypot(0.5,1.9),1.9/math.hypot(0.5,1.9)
    assert abs(d.x-(8.5+ux))<1e-6 and abs(d.y-(10.5+uy))<1e-6,(d.x,d.y)


def t_crown_tower_takes_one_ball_once():
    # the far row of a cannon at (3.5, 18.5) lands on the red left princess tower (3.5, 25.5): the balls at x 1.5, 3.5 and 5.5 reach it
    g=quiet(Game());tw=g.arena.get_tower('red','princess','left');ini=tw.hp
    cn=create('cannon',11,'blue',3.5,18.5,evolved=True);g.deploy('blue',cn);g.run(0.1)
    assert ini-tw.hp==89,ini-tw.hp
