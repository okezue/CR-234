import pytest

from sim.cards import create
from sim.game import Game
from tests.util import quiet

# A ground unit snapped from the river to the bank beside a side fence is settled on the nearest walkable tile centre instead of
# standing on the fence (review t2118 finding 1); a snap in open water keeps the old bank point.


def one(r):return r[0] if isinstance(r, list) else r


def unit(name, team, x, y, g, lvl=11, **kw):
    u = one(create(name, lvl, team, x, y, **kw));u.x, u.y = float(x), float(y);u._settled = True;g.deploy(team, u);return u


def stands(g, u):
    return 0 <= u.x < 18 and 0 <= u.y < 32 and g._walkable(u.x, u.y, False)


# review t2118 finding 1: a knight left in the river at the side edges was snapped onto the fence tile (0|17, 14|17) and stayed there
@pytest.mark.parametrize('x,y,tx,ty', [(0.5, 15.5, 3.5, 25.5), (17.5, 16.2, 14.5, 6.5), (0.6, 15.2, 3.5, 25.5)])
def t_river_snap_beside_a_side_fence_lands_on_walkable_ground(x, y, tx, ty):
    g = quiet(Game());k = unit('knight', 'blue', x, y, g)
    g._move(k, k.spd, tx, ty)
    assert stands(g, k), (k.x, k.y)
    p = (k.x, k.y)
    for _ in range(40):g._move(k, k.spd, tx, ty)
    assert stands(g, k) and (k.x, k.y) != p, (p, k.x, k.y)


@pytest.mark.parametrize('x,y,bank', [(8.5, 15.5, 14.9), (8.5, 16.4, 17.0), (12.0, 15.6, 14.9)])
def t_river_snap_in_open_water_is_unchanged(x, y, bank):
    g = quiet(Game());k = unit('knight', 'blue', x, y, g)
    g._move(k, k.spd, 8.5, 25.5 if y < 16 else 6.5)
    assert (k.x, round(k.y, 6)) == (x, bank), (k.x, k.y)
