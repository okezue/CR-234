import random

from sim.cards import create
from sim.game import Game
from tests.util import Dummy, quiet


# Hero Musketeer's Trusty Turret reaches 4 tiles: Supercell's February 2026 note cut it from 5.5 to 3.5 tiles ("making it harder for Turret
# to lock on to Crown Towers") and the March 2026 note raised it to 4 ("no longer being able to reach King Towers"). The engine kept
# ClashStrategic's 5.5. The turret's hitpoints, damage and deploy splash have no primary source, so they are not tested here.


def bare():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


def turret_of(g,x,y):
    # a rooted, harmless hero Musketeer casts her ability; the turret stands 3 tiles in front of her after the 1 s cast
    m=create('musketeer',11,'blue',x,y,hero=True);g.deploy('blue',m);m.spd=0;m.rng=0;m.dmg=0
    g.players['blue'].elixir=10;m.ability.cd=0;g.activate_ability('blue',m);g.run(1.2)
    return next(u for u in g.players['blue'].troops if u.name=='AutoTurret')


def t_turret_range_is_4_tiles():
    m=create('musketeer',11,'blue',9,10,hero=True)
    assert m.ability.turret_cfg['rng']==4.0
    assert m.rng==6.0


def t_turret_shoots_a_troop_at_3_9_tiles_not_at_4_6():
    for gap,hit in ((3.9,True),(4.6,False)):
        random.seed(1);g=bare();tu=turret_of(g,9,6)
        d=Dummy('red',tu.x,tu.y+tu.collision_r+0.5+gap,hp=50000,spd=0,dmg=0);d._settled=True;g.deploy('red',d)
        assert abs(g._dist(tu,d)-gap)<1e-9
        g.run(3)
        assert (d.hp<50000)==hit,(gap,50000-d.hp)


def t_turret_reaches_a_crown_tower_at_3_5_tiles_not_at_4_5():
    # with no enemy troop on the board the turret takes the nearest princess tower; it shoots only within its range
    for gap,hit in ((3.5,True),(4.5,False)):
        random.seed(1);g=quiet(Game())
        tw=next(t for t in g.arena.towers if t.team=='red' and t.ttype=='princess' and t.cx<9);h0=tw.hp
        tu=turret_of(g,tw.cx,tw.cy-tw.collision_r-0.5-gap-3.0)
        assert abs(g._dist(tu,tw)-gap)<1e-6,g._dist(tu,tw)
        g.run(3)
        assert (tw.hp<h0)==hit,(gap,h0-tw.hp)
