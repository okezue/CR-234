import math

from sim import fx
from sim.cards import create
from sim.game import Game
from tests.util import quiet

# The Fisherman's hook drags a troop until it is drag_margin (0.2 tiles) from him edge to edge, and him to a building the same way
# (RoyaleAPI FishermanProjectile drag_margin 200). Recording: Ian77 G7cTO29IaQA game 1 (battle 09YP9U9P90GQ), both pulls of a Hog Rider
# off the princess tower end with the Hog touching him in front of the King (1.2 to 1.6 tiles centre to centre), and the Hog then hits
# the King Tower (7221, then 6439 and 5932); the engine had stopped the drag at his 1.2 attack range and the Hog went back to the princess.


def one(r):return r[0] if isinstance(r, list) else r


def unit(name, team, x, y, g):
    u = one(create(name, 11, team, x, y));u.x, u.y = float(x), float(y);u._settled = True;g.deploy(team, u);return u


def hook(f):return next(c for c in f.components if isinstance(c, fx.Hook))


def t_hook_drags_a_hog_until_it_touches_the_fisherman():
    g = quiet(Game());f = unit('fisherman', 'blue', 8.8, 5.9, g);hog = unit('hog_rider', 'red', 4.3, 8.6, g)
    hk = hook(f);hk._hit(f, hog, g)
    for _ in range(100):
        hk.on_tick(f, g)
        if hk.pull is None:break
    d = math.hypot(hog.x - f.x, hog.y - f.y)
    assert hk.pull is None and abs(g._dist(f, hog) - 0.2) < 1e-6 and abs(d - 1.3) < 1e-6, d


def t_hog_hooked_off_the_princess_tower_takes_the_king_tower():
    # the recorded geometry: the Fisherman stands in front of the King (sim (8.8, 5.9)) and hooks the Hog hitting the princess tower
    g = quiet(Game());f = unit('fisherman', 'blue', 8.8, 5.9, g);f.spd = 0;f.cd = 99
    unit('hog_rider', 'red', 4.3, 8.6, g)
    king = g.arena.get_tower('blue', 'king');pr = next(t for t in g.arena.towers if t.team == 'blue' and t.ttype == 'princess' and t.cx < 9)
    hk = hook(f);pulled = False
    while g.t < 4.0 and not (pulled and hk.pull is None):
        g.tick();pulled = pulled or hk.pull is not None
    hp = pr.hp
    g.run(6.0)
    assert pulled and king.hp < king.max_hp and king.active and pr.hp == hp, (king.hp, pr.hp, hp)


def t_dragged_troop_is_not_held_after_the_drag():
    # the stun only covers the drag, so the troop picks its next target as soon as it lands
    g = quiet(Game());f = unit('fisherman', 'blue', 8.8, 5.9, g);f.spd = 0;f.cd = 99;hog = unit('hog_rider', 'red', 4.3, 8.6, g)
    hk = hook(f);pulled = False
    while g.t < 4.0 and not (pulled and hk.pull is None):
        g.tick();pulled = pulled or hk.pull is not None
    assert pulled and not any(s.kind == 'stun' and s.dur > g.DT for s in hog.statuses), [(s.kind, s.dur) for s in hog.statuses]


def t_fisherman_hooked_to_a_building_stops_at_the_drag_margin():
    g = quiet(Game());f = unit('fisherman', 'blue', 14.5, 17.5, g)
    tw = next(t for t in g.arena.towers if t.team == 'red' and t.ttype == 'princess' and t.cx > 9)
    hk = hook(f);hk._hit(f, tw, g)
    for _ in range(100):
        hk.on_tick(f, g)
        if hk.pull is None:break
    assert hk.pull is None and abs(g._dist(f, tw) - 0.2) < 1e-6, g._dist(f, tw)
