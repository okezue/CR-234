import pytest

from sim import fx
from sim.cards import create
from sim.game import Game
from tests.util import Dummy, quiet

# The evolved P.E.K.K.A picks her heal by the victim's hitpoints against the two thresholds of her level (wiki P.E.K.K.A./Evolution
# level table: 990 and 1991 at level 11, x1.1 per level), not fixed 500 and 1500. A ground unit snapped from the river to the bank
# beside a side fence is settled on the nearest walkable tile centre instead of standing on the fence.


def one(r):return r[0] if isinstance(r, list) else r


def unit(name, team, x, y, g, lvl=11, **kw):
    u = one(create(name, lvl, team, x, y, **kw));u.x, u.y = float(x), float(y);u._settled = True;g.deploy(team, u);return u


def comp(u, cls):return next(c for c in u.components if isinstance(c, cls))


def heal_for(mhp, lvl=11):
    g = quiet(Game());pk = unit('pekka', 'blue', 9.0, 10.0, g, lvl=lvl, evolved=True);ep = comp(pk, fx.EvoPekka)
    pk.hp = 1000;d = Dummy('red', 9.0, 11.0, hp=mhp);d.alive = False
    ep.on_attack(pk, d, g)
    return pk.hp - 1000, ep


# level 11 heals 168 / 320 / 606 (data); a Musketeer (721) and an 800 hitpoint victim are below 990, a Knight (1766) and a Hog Rider
# (1697) between 990 and 1991
@pytest.mark.parametrize('mhp,tier', [(721, 's'), (800, 's'), (990, 's'), (1697, 'm'), (1766, 'm'), (1991, 'm')])
def t_evo_pekka_heal_tier_follows_the_level_11_thresholds(mhp, tier):
    h, ep = heal_for(mhp)
    assert h == ep.tiers[tier], (mhp, h, ep.tiers)


@pytest.mark.parametrize('mhp,tier', [(400, 's'), (2100, 'l'), (3968, 'l')])
def t_evo_pekka_heal_tier_unchanged_outside_the_band(mhp, tier):
    h, ep = heal_for(mhp)
    assert h == ep.tiers[tier], (mhp, h, ep.tiers)


def t_evo_pekka_thresholds_scale_with_her_level():
    _, ep = heal_for(100, lvl=13)
    assert ep.cuts == [1195, 2404], ep.cuts
    assert heal_for(1195, lvl=13)[0] == ep.tiers['s'] and heal_for(1196, lvl=13)[0] == ep.tiers['m']
    assert heal_for(2404, lvl=13)[0] == ep.tiers['m'] and heal_for(2405, lvl=13)[0] == ep.tiers['l']


def t_evo_pekka_kill_in_combat_heals_the_small_tier_for_an_800_hitpoint_victim():
    g = quiet(Game());pk = unit('pekka', 'blue', 9.0, 10.0, g, evolved=True);ep = comp(pk, fx.EvoPekka)
    pk.hp = 1000;d = Dummy('red', 9.0, 11.2, hp=800, dmg=0, spd=0);g.deploy('red', d)
    for _ in range(200):
        g.tick()
        if not d.alive:break
    assert not d.alive and pk.hp == 1000 + ep.tiers['s'], (d.alive, pk.hp, ep.tiers)


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
