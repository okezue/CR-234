import math

import pytest

from sim import fx
from sim.cards import create
from sim.game import Game
from tests.util import Dummy

# Wiki Tornado: "For most melee troops, placing the Tornado on the tile directly in front of the King's Tower and two tiles into the
# defending lane will cause an activation" (blue's tile (10.5, 5.5), mirrored (7.5, 5.5)); the GrandWiz05 blog linked there: light melee
# troops "move fast and are therefore easier to be pulled by the Tornado". Recorded: LCQ game 09YP9UPGQ2YU, Wallace's Tornado (70.80 s,
# raw 7.5, 5.5) pulls the first drill Goblin from beside his left princess tower to the Tornado's centre by 71.75 to 71.80 (4.75 tiles,
# badge track at 0.05 s), the Goblin hits the King at 72.70 and the King fires at 76.70.
MELEE = ('knight', 'valkyrie', 'mini_pekka', 'pekka', 'hog_rider', 'lumberjack', 'royal_ghost', 'goblins', 'barbarians', 'elite_barbarians',
         'prince', 'dark_prince', 'bandit', 'miner', 'mega_knight', 'goblin_cage', 'guards', 'skeletons', 'fisherman', 'battle_ram', 'ram_rider')


def attackers(g, name, lx):
    if name == 'goblin_cage':
        cage = create(name, 11, 'red', lx, 12.5)
        g._place('red', cage, 0)
        cage.take_damage(cage.hp)
        g._proc_deaths()
        return [u for u in g.players['red'].troops if u.alive]
    x, y = (lx + (1.5 if lx > 9 else -1.5), 9.5) if name == 'miner' else (lx, 14.5)
    r = create(name, 11, 'red', x, y)
    rs = r if isinstance(r, list) else [r]
    for u in rs:
        g._place('red', u, 0)
    return rs


def wiki_tornado(name, lx):
    # the attacker walks from the bridge to blue's princess tower; blue's Tornado lands on the wiki tile when the tower is first hit
    g = Game()
    rs = attackers(g, name, lx)
    king = g.arena.get_tower('blue', 'king')
    tower = next(t for t in g.arena.towers if t.team == 'blue' and t.ttype == 'princess' and t.cx == lx)
    tx = 10.5 if lx > 9 else 7.5
    while tower.hp == tower.max_hp and g.t < 20 and any(u.alive for u in rs):
        g.tick()
    if tower.hp == tower.max_hp:
        return None
    cast = g.t
    g._cast('blue', create('tornado', 11, 'blue', tx, 5.5), tx, 5.5)
    while king.hp == king.max_hp and g.t < cast + 6:
        g.tick()
    return round(g.t - cast, 2) if king.hp < king.max_hp else None


@pytest.mark.parametrize('lx', (14.5, 3.5))
def t_wiki_tile_activates_the_king_for_most_melee_troops(lx):
    got = {n: wiki_tornado(n, lx) for n in MELEE}
    hit = [n for n, t in got.items() if t is not None]
    assert len(hit) > len(MELEE) / 2, got
    # the light melee troops the blog names and the Hog Rider it shows
    assert all(got[n] is not None for n in ('hog_rider', 'lumberjack', 'royal_ghost', 'goblin_cage')), got


@pytest.mark.parametrize('name,spd', (('goblins', 2.4), ('knight', 1.2), ('mini_pekka', 1.8), ('giant', 0.9), ('baby_dragon', 1.8)))
def t_pull_is_360_percent_of_the_troops_own_walking_speed(name, spd):
    g = Game()
    r = create(name, 11, 'red', 11, 10)
    u = r[0] if isinstance(r, list) else r
    u.x, u.y = 11.0, 10.0
    g.deploy('red', u)
    assert math.isclose(u.spd, spd)
    sp = create('tornado', 11, 'blue', 9, 10)
    sp.apply(g)
    sp.tick(g.DT, g)
    assert math.isclose(u.x, 11 - sp.pull_str * spd * g.DT) and u.y == 10


def t_pull_uses_the_walking_speed_not_a_charge_or_windup():
    g = Game()
    u = Dummy('red', 11, 10, spd=1.2)
    u.base_spd = 1.2
    g.deploy('red', u)
    sp = create('tornado', 11, 'blue', 9, 10)
    sp.apply(g)
    u.spd = 2.4
    sp.tick(g.DT, g)
    assert math.isclose(u.x, 11 - sp.pull_str * 1.2 * g.DT)
    u.spd = 0
    x = u.x
    sp.tick(g.DT, g)
    assert math.isclose(u.x, x - sp.pull_str * 1.2 * g.DT)


def t_pull_stops_at_the_centre():
    g = Game()
    u = Dummy('red', 9.3, 10, spd=2.4)
    g.deploy('red', u)
    sp = create('tornado', 11, 'blue', 9, 10)
    sp.apply(g)
    sp.tick(g.DT, g)
    assert (u.x, u.y) == (9.0, 10.0)


def t_game_one_drill_goblin_activates_the_king_near_72_7(monkeypatch):
    # the replay's timeline with game time = 65.35 + g.t: drill placed on the 65.40 tick at (3.5, 23.5), Wallace (red) Tornado cast on the
    # 70.85 tick at (7.5, 26.5); the first Goblin is born on that tick at the replay's spot (3.5, 23.5), the free tile nearest the
    # unjittered spawn point on the princess tower
    monkeypatch.setattr(fx.random, 'uniform', lambda a, b: 0.0)
    g = Game()
    g._place('blue', create('goblin_drill', 11, 'blue', 3.5, 23.5), 0)
    t0 = g.t
    while g.t < t0 + 5.45 - 1e-9:
        g.tick()
    g._cast('red', create('tornado', 11, 'red', 7.5, 26.5), 7.5, 26.5)
    king = g.arena.get_tower('red', 'king')
    g.tick()
    gob = next(u for u in g.players['blue'].troops if u.name == 'Goblin')
    assert (gob.x, gob.y) == (3.5, 23.5) and round(g.t - t0, 2) == 5.5
    while king.hp == king.max_hp and g.t < t0 + 12:
        g.tick()
    assert king.hp < king.max_hp
    # recorded hit 72.70
    assert abs(g.t - t0 - 7.35) <= 0.5, round(g.t - t0, 2)
