import pytest

from sim import fx
from sim.cards import create
from sim.game import Game
from tests.util import Dummy, quiet

# A held target is kept while it is being hit; out of reach the unit takes the nearest target in sight. Wiki Mega Knight: "Similar to the
# Bandit, the Mega Knight will re-target onto any troop placed nearby him if his original target ends up being farther away than the newly
# spawned troop"; wiki Battle Ram: slower units "will eventually lose sight of the Battle Ram and will not be kited the whole way"; wiki
# Little Prince: Spear Goblins "locked onto the Little Prince can be pushed out of range by the Royal Rescue knockback, causing them to
# retarget"; wiki Inferno Dragon: a Tornado "will not retarget" it while its target is still in range. Red attacks toward blue's towers.
HEAVY = ('mega_knight', 'pekka', 'prince')


def unit(g, name, team, x, y):
    u = create(name, 11, team, x, y)
    us = u if isinstance(u, list) else [u]
    for v in us:
        g.deploy(team, v)
    return us


def dummy(g, team, x, y, hp=10**6):
    # heavy enough that the Mega Knight's deploy knockback leaves it in place
    d = Dummy(team, x, y, hp=hp, spd=0, dmg=0, mass=20)
    g.deploy(team, d)
    return d


def until(g, cond, t):
    end = g.t + t
    while g.t < end and not cond():
        g.tick()
    return cond()


@pytest.mark.parametrize('name', HEAVY)
def t_walking_troop_takes_a_nearer_troop(name):
    g = quiet(Game())
    a, = unit(g, name, 'red', 9, 14)
    d = dummy(g, 'blue', 9, 9.75)
    assert until(g, lambda: a.tgt is d, 2)
    g.run(0.3)
    assert g._dist(a, d) > a.rng
    sk = unit(g, 'skeletons', 'blue', a.x + 1.6, a.y - 0.6)
    g.run(0.15)
    assert a.tgt in sk, (getattr(a.tgt, 'name', None), round(g._dist(a, d), 2), min(round(g._dist(a, s), 2) for s in sk))


@pytest.mark.parametrize('name', HEAVY)
def t_attacking_troop_keeps_its_target(name):
    g = quiet(Game())
    a, = unit(g, name, 'red', 9, 12)
    d = dummy(g, 'blue', 9, 10.45)
    g.run(2.5)
    assert a.tgt is d and g._dist(a, d) <= a.rng and d.hp < 10**6
    sk = unit(g, 'skeletons', 'blue', a.x + 0.9, a.y - 0.2)
    g.run(0.15)
    assert a.tgt is d and not any(a.tgt is s for s in sk)


@pytest.mark.parametrize('name', HEAVY)
def t_attacking_tower_keeps_it(name):
    g = quiet(Game())
    tw = next(t for t in g.arena.towers if t.team == 'blue' and t.ttype == 'princess' and t.cx < 9)
    a, = unit(g, name, 'red', 3.5, 9.5)
    assert until(g, lambda: tw.hp < tw.max_hp, 6)
    unit(g, 'skeletons', 'blue', a.x + 1.2, a.y + 0.3)
    g.run(0.15)
    assert a.tgt is tw


@pytest.mark.parametrize('name', HEAVY)
def t_walking_to_the_tower_takes_skeletons_in_sight(name):
    g = quiet(Game())
    a, = unit(g, name, 'red', 3.5, 18.5)
    assert until(g, lambda: a.y <= 14.5, 6)
    sk = unit(g, 'skeletons', 'blue', 3.5, a.y - 3.0)
    g.run(0.2)
    assert a.tgt in sk


def kite(name, kiter):
    g = quiet(Game())
    a, = unit(g, name, 'red', 3.5, 12.5)
    k, = unit(g, kiter, 'blue', 9.6, 12.5)
    assert until(g, lambda: a.tgt is k, 1.5)
    return g, a, k


def t_chase_ends_when_the_target_leaves_sight():
    # the Battle Ram charges away from a P.E.K.K.A (sight 5.0); once it is out of her sight she stops following while it still lives
    g, a, k = kite('pekka', 'battle_ram')
    seen = []
    while k.alive and g.t < 8:
        g.tick()
        gap = g._dist(a, k)
        if gap > a.sight_r + 0.3:
            seen.append(a.tgt is k)
    assert seen and not any(seen[2:]), (len(seen), sum(seen))


@pytest.mark.parametrize('name', ('pekka', 'knight', 'valkyrie'))
def t_kite_within_sight_crosses_the_river(name):
    # wiki Ice Golem: melee troops chase the Ice Golem to the other lane; it stays within sight of these chasers
    g, a, k = kite(name, 'ice_golem')
    held = []
    while g.t < 15 and k.alive:
        g.tick()
        held.append(a.tgt is k)
    assert all(held) and a.y > 16.5 and a.x > 12, (held.count(False), round(a.x, 1), round(a.y, 1))


def t_pushed_out_of_reach_takes_the_nearest():
    g = quiet(Game())
    a, = unit(g, 'knight', 'red', 9, 12)
    d1 = dummy(g, 'blue', 9, 10.7)
    assert until(g, lambda: d1.hp < 10**6, 4)
    d2 = dummy(g, 'blue', 11.2, 13.5)
    fx.push(a, a.x, a.y - 1, 2.0)
    g.tick()
    assert g._dist(a, d2) < g._dist(a, d1) and g._dist(a, d1) > a.rng
    assert a.tgt is d2


def t_pushed_out_of_reach_keeps_a_target_that_is_still_nearest():
    g = quiet(Game())
    a, = unit(g, 'knight', 'red', 9, 12)
    d1 = dummy(g, 'blue', 9, 10.7)
    assert until(g, lambda: d1.hp < 10**6, 4)
    fx.push(a, a.x, a.y - 1, 2.0)
    g.tick()
    assert g._dist(a, d1) > a.rng and a.tgt is d1
