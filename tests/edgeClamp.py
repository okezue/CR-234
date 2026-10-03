import math

import pytest

from sim import fx
from sim.cards import card, create, hero
from sim.game import Game
from tests.util import Dummy, quiet

# Units never stand outside the 18 by 32 arena or on a tower footprint, a fence or (unless they hover or jump the river) water. Every
# path that moves a ground unit ends on such a point: shoves, pulls and rolls stop where their path leaves standable ground, like a
# knockback; dashes, jumps, throws and lane switches are settled like a deploy (nearest walkable tile centre).


def one(r):return r[0] if isinstance(r, list) else r


def unit(name, team, x, y, g, **kw):
    u = one(create(name, 11, team, x, y, **kw));u.x, u.y = float(x), float(y);u._settled = True;g.deploy(team, u);return u


def stands(g, u):
    rj = getattr(u, 'hovering', False) or any(isinstance(c, fx.RiverJump) for c in getattr(u, 'components', []))
    return 0 <= u.x < 18 and 0 <= u.y < 32 and g._walkable(u.x, u.y, rj)


def comp(u, cls):return next(c for c in u.components if isinstance(c, cls))


def tower(g, team, side):return next(t for t in g.arena.towers if t.team == team and t.ttype == 'princess' and (t.cx < 9) == (side == 'left'))


def monk_shove(g, monk, tgt):
    mc = comp(monk, fx.MonkCombo)
    for _ in range(mc.cycle):mc.on_attack(monk, tgt, g)


# Monk third-hit shove (1.8 tiles): off the right, left, front and back edges, into a princess footprint and into the river
@pytest.mark.parametrize('mx,my,tx,ty', [(16.0, 10.0, 17.0, 10.0), (2.0, 10.0, 1.0, 10.0), (8.5, 30.0, 8.5, 31.0), (6.5, 1.8, 6.5, 0.8),
                                         (3.5, 10.0, 3.5, 9.0), (8.5, 12.5, 8.5, 13.6)])
def t_monk_shove_ends_on_standable_ground(mx, my, tx, ty):
    g = quiet(Game());monk = unit('monk', 'blue', mx, my, g);k = unit('knight', 'red', tx, ty, g)
    monk_shove(g, monk, k)
    assert stands(g, k), (k.x, k.y)
    # it still moves away from the monk as far as the ground allows
    assert math.hypot(k.x - mx, k.y - my) > math.hypot(tx - mx, ty - my), (k.x, k.y)


def t_monk_shove_in_the_open_moves_the_full_distance():
    g = quiet(Game());monk = unit('monk', 'blue', 9.0, 20.0, g);k = unit('knight', 'red', 9.0, 21.0, g)
    monk_shove(g, monk, k)
    assert (k.x, k.y) == (9.0, 22.8)


# The Log's 0.7 tile roll pushback: off the front edge (row 31 is open between x 6 and 12) and into the corner fence
@pytest.mark.parametrize('x,y', [(8.5, 31.6), (3.5, 30.7)])
def t_log_pushback_ends_on_standable_ground(x, y):
    g = quiet(Game());k = unit('knight', 'red', x, y, g)
    sp = create('the_log', 11, 'blue', x, 29.0);sp._sweep(g, 0, sp.rng)
    assert k.hp < k.max_hp and stands(g, k), (k.x, k.y)


def t_log_pushback_in_the_open_moves_the_full_distance():
    g = quiet(Game());k = unit('knight', 'red', 9.0, 20.0, g)
    sp = create('the_log', 11, 'blue', 9.0, 18.0);sp._sweep(g, 0, sp.rng)
    assert (k.x, k.y) == (9.0, 20.7)


def t_barbarian_barrel_hero_roll_ends_on_standable_ground():
    # Rowdy Reroll rolls the Barbarian 3 tiles forward: from y 30 that passes the front edge
    g = quiet(Game());b = unit('knight', 'blue', 8.5, 30.0, g);c = card('barbarian_barrel')
    hero(c, c['hero']['ability'], 11, b).activate(b, g)
    assert stands(g, b) and b.y > 30.0, (b.x, b.y)


def t_evolved_snowball_roll_ends_on_standable_ground():
    # the evolved Giant Snowball carries what it hits 4 tiles forward over 0.75 s: from y 30 that passes the front edge
    g = quiet(Game());k = unit('knight', 'red', 6.5, 30.0, g)
    sp = create('giant_snowball', 11, 'blue', 6.5, 30.0, evolved=True);sp.apply(g)
    while sp.active:
        sp.tick(g.DT, g);assert stands(g, k), (k.x, k.y)


def t_firecracker_recoil_ends_on_standable_ground():
    # a 1 tile recoil from row 17 towards blue's side lands in the river off the bridges
    g = quiet(Game());fc = unit('firecracker', 'blue', 8.5, 16.6 + 1.0, g);t = Dummy('red', 8.5, 22.0, hp=5000, spd=0, dmg=0);g.deploy('red', t)
    comp(fc, fx.Recoil).on_attack(fc, t, g)
    assert stands(g, fc) and fc.y < 17.6, (fc.x, fc.y)


def t_boss_bandit_getaway_ends_on_standable_ground():
    # 6 tiles back from y 2 stops at y 0, on the corner fence
    g = quiet(Game());bb = unit('boss_bandit', 'blue', 3.5, 2.0, g)
    bb.ability.activate(bb, g)
    assert stands(g, bb), (bb.x, bb.y)


def t_elite_archer_hero_dash_ends_on_standable_ground():
    g = quiet(Game());ma = unit('magic_archer', 'blue', 3.5, 1.5, g, hero=True)
    ma.ability.activate(ma, g)
    assert stands(g, ma), (ma.x, ma.y)


def t_giant_hero_hurl_ends_on_standable_ground():
    # a 9 tile throw from x 5.5 lands at x 14.5, inside red's right princess tower
    g = quiet(Game());gi = unit('giant', 'blue', 4.5, 25.0, g, hero=True);k = unit('knight', 'red', 5.5, 25.0, g)
    gi.ability.activate(gi, g)
    assert k.x > 5.5 and stands(g, k), (k.x, k.y)


def t_mighty_miner_lane_switch_ends_on_standable_ground():
    # from the footprint of red's fallen left tower to the same spot on the right, where red's tower still stands
    g = quiet(Game());tower(g, 'red', 'left').alive = False;mm = unit('mighty_miner', 'blue', 3.2, 25.0, g)
    mm.ability.activate(mm, g)
    assert mm.x > 9 and stands(g, mm), (mm.x, mm.y)


def t_evolved_mega_knight_uppercut_ends_on_standable_ground():
    # the uppercut throws a red troop 4 tiles towards red's King Tower: from y 22.6 that is inside red's right princess tower
    g = quiet(Game());mk = unit('mega_knight', 'blue', 14.5, 21.4, g, evolved=True);k = unit('knight', 'red', 14.5, 22.6, g)
    up = comp(mk, fx.EvoMegaKnight)
    for _ in range(up.every):up.on_attack(mk, k, g)
    assert k.y > 22.6 and stands(g, k), (k.x, k.y)


def t_evolved_valkyrie_pull_ends_on_standable_ground():
    # pulled up to 0.75 tiles towards her from the far bank, the troop stops at the water's edge
    g = quiet(Game());v = unit('valkyrie', 'blue', 8.5, 14.6, g, evolved=True);k = unit('knight', 'red', 8.5, 17.6, g)
    comp(v, fx.EvoValkyrie).on_attack(v, k, g)
    assert k.y < 17.6 and stands(g, k), (k.x, k.y)


@pytest.mark.parametrize('cx,cy,kx,ky', [(8.5, 15.5, 8.5, 17.4), (14.5, 25.5, 14.5, 22.4)])
def t_tornado_pull_ends_on_standable_ground(cx, cy, kx, ky):
    # a Tornado on the river or on a standing princess tower pulls a troop to its centre; when it ends the troop is set beside it
    g = quiet(Game());k = unit('knight', 'red', kx, ky, g);sp = create('tornado', 11, 'blue', cx, cy);sp.apply(g)
    while sp.active:sp.tick(g.DT, g)
    assert stands(g, k) and math.hypot(k.x - cx, k.y - cy) < math.hypot(kx - cx, ky - cy), (k.x, k.y)


def t_tornado_pull_in_the_open_is_unchanged():
    g = quiet(Game());k = unit('knight', 'red', 11.0, 10.0, g);sp = create('tornado', 11, 'blue', 9.0, 10.0);sp.apply(g)
    sp.tick(g.DT, g)
    assert (k.x, k.y) == (11.0 - 3.6 * k.spd * g.DT, 10.0)


@pytest.mark.parametrize('team', ('blue', 'red'))
def t_golden_knight_dash_onto_a_tower_ends_beside_it_on_his_side(team):
    # of the four tile centres 2 tiles from the tower's centre he takes the one toward where he came from, for either team
    g = quiet(Game());y0 = 21.5 if team == 'blue' else 32 - 21.5;gk = unit('golden_knight', team, 14.5, y0, g)
    tw = tower(g, g._opp(team), 'right');gk.ability.activate(gk, g);gk.ability.tick(g.DT, gk, g)
    assert tw.hp < tw.max_hp and stands(g, gk) and (gk.x, gk.y) == (14.5, 23.5 if team == 'blue' else 8.5), (gk.x, gk.y)


def t_wizard_hero_flight_lands_on_standable_ground():
    # Fiery Flight lifts the hero Wizard for 5 s; a flight that ends over the fence beside the river lands him on walkable ground
    g = quiet(Game());w = unit('wizard', 'blue', 16.5, 13.5, g, hero=True);ab = w.ability
    ab.activate(w, g);w.x, w.y = 17.15, 17.1
    while ab.active:ab.tick(g.DT, w, g)
    assert w.transport == 'Ground' and stands(g, w), (w.x, w.y)


def t_valkyrie_hero_dash_ends_on_standable_ground():
    # the dash stops 1 tile short of the nearest ground enemy: from y 13 to a troop at y 17.2 that is y 16.2, in the river
    g = quiet(Game());v = unit('valkyrie', 'blue', 8.5, 13.0, g, hero=True);unit('knight', 'red', 8.5, 17.2, g)
    v.ability.activate(v, g)
    assert v.y > 13.0 and stands(g, v), (v.x, v.y)


def t_bandit_dash_ends_on_standable_ground():
    # the dash crosses the river and stops in reach of a troop standing on the far bank
    g = quiet(Game());b = unit('bandit', 'blue', 8.5, 12.0, g);t = unit('knight', 'red', 8.5, 17.1, g);t.spd = 0
    bd = comp(b, fx.BanditDash);bd.dashing = True;bd.dtgt = t;bd.to = (t.x, t.y)
    while bd.dashing:bd.on_tick(b, g)
    assert t.hp < t.max_hp and stands(g, b), (b.x, b.y)


def t_fisherman_drag_ends_on_standable_ground():
    # a troop hooked across the river is dragged into the Fisherman's reach, which from y 13.8 is inside the river
    g = quiet(Game());f = unit('fisherman', 'blue', 8.5, 13.8, g);t = unit('knight', 'red', 8.5, 19.0, g)
    hk = comp(f, fx.Hook);hk._hit(f, t, g)
    for _ in range(100):
        hk.on_tick(f, g)
        if hk.pull is None:break
    assert hk.pull is None and stands(g, t), (t.x, t.y)


def t_troop_dragged_when_the_fisherman_falls_stands_on_standable_ground():
    g = quiet(Game());f = unit('fisherman', 'blue', 8.5, 13.8, g);t = unit('knight', 'red', 8.5, 19.0, g)
    hk = comp(f, fx.Hook);hk._hit(f, t, g)
    while t.y >= 17.0:hk.on_tick(f, g)
    f.take_damage(f.hp);g._proc_deaths()
    assert not f.alive and stands(g, t), (t.x, t.y)


def t_mega_knight_landing_ends_on_standable_ground():
    # he comes down on a Hog Rider jumping the river; the hog may stand in the water, he may not
    g = quiet(Game());mk = unit('mega_knight', 'blue', 8.5, 11.0, g);hog = unit('hog_rider', 'red', 8.5, 15.5, g);hog.spd = 0
    j = comp(mk, fx.MKJump);j.airborne = True;j.timer = 0;j.jtgt = hog
    j.on_tick(mk, g)
    assert stands(g, hog) and stands(g, mk), (mk.x, mk.y)


@pytest.mark.parametrize('x,y', [(22.0, 10.3), (-4.5, 20.2), (9.3, 35.0), (6.4, -4.2)])
def t_unit_born_far_outside_is_settled_inside(x, y):
    g = quiet(Game());k = one(create('knight', 11, 'blue', x, y));g._place('blue', k, 1.0)
    assert stands(g, k), (k.x, k.y)


def t_goblinstein_doctor_deployed_at_the_back_stands_inside():
    # the Doctor stands the summon radius behind the Monster, beyond the back edge when the card is played on the last row
    g = quiet(Game())
    for u in create('goblinstein', 11, 'red', 9.0, 31.5):g._place('red', u, 1.0)
    assert all(stands(g, u) for u in g.players['red'].troops), [(u.name, u.x, u.y) for u in g.players['red'].troops]
