import pytest

from sim.cards import create
from sim.game import Game
from sim.units import Troop

# The Ice Spirit and the Heal Spirit launch themselves at their target and are destroyed on impact (wiki Ice Spirit, Heal Spirit;
# export kamikaze): one jump each. At 8 tiles per second the jump outlasts their 0.3 s hit speed from about 2.4 tiles centre to centre.

DECK = ['ice_spirit', 'heal_spirit', 'electro_spirit', 'knight', 'archers', 'fireball', 'giant', 'arrows']


def quiet():
    g = Game(p1={'deck': DECK, 'drag_del': 0}, p2={'deck': DECK, 'drag_del': 0})
    for tower in g.arena.towers:
        tower.rng = 0
        if tower.troop:
            tower.troop.RNG = 0
    return g


def target(g, team, x, y, hp=5000):
    t = Troop(team, x, y, {'hp': hp, 'dmg': 0, 'spd': 0, 'hspd': 1, 'rng': 0, 'name': 'Stationary control'})
    t._settled = True
    g.deploy(team, t)
    return t


def spirit(g, card, team, x, y, level=11):
    s = create(card, level, team, x, y)
    s._settled = True
    g.deploy(team, s)
    return s


def releases(g, unit):
    # times at which the unit puts a projectile in the air
    out = []
    fire = g._fire

    def wrap(tr, tgt):
        n = len(g.projs)
        r = fire(tr, tgt)
        if tr is unit and len(g.projs) > n:
            out.append(round(g.t, 2))
        return r
    g._fire = wrap
    return out


@pytest.mark.parametrize('card', ('ice_spirit', 'heal_spirit'))
@pytest.mark.parametrize('team', ('blue', 'red'))
@pytest.mark.parametrize('level', (11, 16))
def t_long_range_jump_is_one_attack(card, team, level):
    g = quiet()
    y, f = (10.5, 1) if team == 'blue' else (21.5, -1)
    s = spirit(g, card, team, 9.5, y, level)
    t = target(g, g._opp(team), 9.5, y + f * 3.2)
    rel = releases(g, s)
    g.run(3)
    assert len(rel) == 1, rel
    assert t.hp == 5000 - s.dmg
    assert not s.alive


def t_ice_spirit_freezes_once_per_jump():
    g = quiet()
    s = spirit(g, 'ice_spirit', 'blue', 9.5, 10.5)
    t = target(g, 'red', 9.5, 13.7)
    hits = []
    take = t.take_damage

    def note(a):
        hits.append(round(g.t, 2))
        return take(a)
    t.take_damage = note
    stuns = set()
    while g.t < 3:
        g.tick()
        stuns |= {id(st) for st in t.statuses if st.kind == 'stun'}
    assert len(hits) == 1 and len(stuns) == 1
    assert t.hp == 5000 - s.dmg and not s.alive


def t_launched_spirit_does_not_jump_again_when_its_target_dies_in_flight():
    g = quiet()
    s = spirit(g, 'ice_spirit', 'blue', 9.5, 10.5)
    first = target(g, 'red', 9.5, 13.7)
    second = target(g, 'red', 10.5, 13.9)
    rel = releases(g, s)
    while not rel and g.t < 2:
        g.tick()
    first.take_damage(first.hp)
    g.run(2)
    assert len(rel) == 1
    assert second.hp >= 5000 - s.dmg
    assert not s.alive


@pytest.mark.parametrize('card', ('ice_spirit', 'heal_spirit', 'electro_spirit'))
def t_short_jump_and_electro_spirit_unchanged(card):
    g = quiet()
    s = spirit(g, card, 'blue', 9.5, 10.5)
    t = target(g, 'red', 9.5, 12.6)
    rel = releases(g, s)
    g.run(3)
    assert len(rel) == 1
    assert t.hp < 5000 and not s.alive


@pytest.mark.parametrize('card', ('ice_spirit', 'heal_spirit', 'electro_spirit'))
def t_spirit_leaves_the_field_at_launch(card):
    # a jumping spirit no longer counts as a troop; it becomes similar to a projectile (wiki Ice, Heal and Electro Spirit)
    g = quiet()
    s = spirit(g, card, 'blue', 9.5, 10.5)
    t = target(g, 'red', 9.5, 13.7)
    while not g.projs and g.t < 2:
        g.tick()
    assert not s.alive and t.hp == 5000
    g._proc_deaths()
    assert s not in g.players['blue'].troops
    g.run(1)
    assert t.hp == 5000 - s.dmg


def t_ordinary_shooter_keeps_firing_while_its_shot_flies():
    g = quiet()
    m = create('musketeer', 11, 'blue', 9.5, 10.5)
    m._settled = True
    g.deploy('blue', m)
    target(g, 'red', 9.5, 16.0)
    rel = releases(g, m)
    g.run(4)
    assert len(rel) >= 3
