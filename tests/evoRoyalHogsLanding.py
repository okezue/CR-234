import random

from sim import fx
from sim.cards import create
from sim.game import Game
from tests.util import Dummy, quiet


# Evolved Royal Hogs fly until they fall to the ground, dealing landing damage around them. Supercell's May 2026 note cut the landing
# damage from 84 to 43 at level 11 ("it'll now take the landing of two Evolved Royal Hogs to take out Skeletons"); the game data export's
# RoyalHog_EV1 runs its Fall_To_Ground action on its first attack (onAttackActionData) as well as when hurt below 99% (onStartingActionData),
# and the wiki says "Upon attacking, or getting hurt, the hogs will fall to the ground". The engine landed them only when hurt, so a hog
# that reached an air-blind building or the tower unhurt kept hitting from the air.


def bare():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


def dummy(g,x,y,hp):
    d=Dummy('red',x,y,hp=hp,spd=0,dmg=0);d._settled=True;g.deploy('red',d);return d


def landing(u):
    return next(c for c in u.components if isinstance(c,fx.EvoRoyalHogs))


def t_landing_damage_is_43_at_level_11():
    hogs=create('royal_hogs',11,'blue',9,10,evolved=True)
    assert [landing(u).ldmg for u in hogs]==[43]*4
    assert landing(create('royal_hogs',16,'blue',9,10,evolved=True)[0]).ldmg==69


def t_two_landings_take_out_a_skeleton():
    # two hurt hogs land one after the other beside an 81-hitpoint unit (a level 11 Skeleton): the first leaves it at 38, the second kills it
    random.seed(1);g=bare();a,b=create('royal_hogs',11,'blue',9,10,evolved=True)[:2]
    for u in (a,b):u.x,u.y=9.0,10.0;g.deploy('blue',u);u._settled=True
    sk=dummy(g,9.0,11.0,81)
    a.hp-=1;landing(a).on_tick(a,g)
    assert a.transport=='Ground' and sk.alive and sk.hp==38,(a.transport,sk.hp)
    b.hp-=1;landing(b).on_tick(b,g)
    assert not sk.alive


def first_hit(g,hog,tgt,until=8.0):
    while g.t<until:
        hp0=tgt.hp;air=hog.transport;g.tick()
        if hp0-tgt.hp>5:return air
    raise AssertionError(f'no hit on {tgt.name} by {until} s')


def t_a_flying_hog_lands_on_its_first_attack():
    # one evolved hog against a Cannon, which cannot shoot air: it is never hurt, yet its first hit brings it down, and the landing hits
    # an 81-hitpoint unit beside the Cannon
    random.seed(1);g=bare();h=create('royal_hogs',11,'blue',9,17,evolved=True)[0];g.deploy('blue',h)
    cn=create('cannon',11,'red',9,22);g.deploy('red',cn);cn.dmg=cn.ct_dmg=0
    sk=dummy(g,9.8,21.0,81)
    assert first_hit(g,h,cn)=='Air'
    assert h.transport=='Ground' and h.hp==h.max_hp,(h.transport,h.hp)
    assert sk.hp==81-43,sk.hp


def t_a_flying_hog_lands_on_its_first_tower_hit():
    random.seed(1);g=quiet(Game());h=create('royal_hogs',11,'blue',3.5,20,evolved=True)[0];g.deploy('blue',h)
    tw=next(t for t in g.arena.towers if t.team=='red' and t.ttype=='princess' and t.cx<9)
    assert first_hit(g,h,tw)=='Air'
    assert h.transport=='Ground' and h.hp==h.max_hp,(h.transport,h.hp)


def t_an_unhurt_hog_flies_until_it_attacks():
    # control: walking to its target unhurt, the hog stays in the air and lands no damage
    random.seed(1);g=bare();h=create('royal_hogs',11,'blue',9,10,evolved=True)[0];g.deploy('blue',h)
    cn=create('cannon',11,'red',9,24);g.deploy('red',cn);cn.dmg=cn.ct_dmg=0
    sk=dummy(g,12.0,14.0,81)
    g.run(1.5)
    assert h.alive and h.transport=='Air' and sk.hp==81,(h.transport,sk.hp)


def t_plain_royal_hogs_are_unchanged():
    # control: the base card has no flight and no landing
    hogs=create('royal_hogs',11,'blue',9,10)
    assert all(u.transport=='Ground' and not any(isinstance(c,fx.EvoRoyalHogs) for c in u.components) for u in hogs)
