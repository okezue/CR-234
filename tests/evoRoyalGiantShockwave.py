import pytest

from sim.cards import create, load
from sim.game import Game
from tests.util import quiet


# evo skills pushback: radius 2.5, distance 1.0, damage 81 at level 11 (data/cards.json); wiki Royal Giant/Evolution: damage in a
# 2.5 tile radius and a one tile knockback of enemy ground troops, air troops immune to the recoil


def giant_and_target(g):
    rg=create('royal_giant',11,'red',9,20,evolved=True);rg.spd=0;g.deploy('red',rg)
    cannon=create('cannon',11,'blue',9,14.5);g.deploy('blue',cannon)
    return rg,cannon


def t_shockwave_data_is_sourced():
    pb=load()['cards']['royal_giant']['evo']['skills']['pushback']
    assert (pb['radius'],pb['distance'],pb['damage'][10])==(2.5,1.0,81)


def t_air_units_are_immune_to_the_recoil():
    g=quiet(Game());rg,cannon=giant_and_target(g)
    bat=create('bats',11,'blue',9.6,20.8)[0];bat.spd=0;bat.dmg=0;g.deploy('blue',bat);hp=bat.hp;pos=(bat.x,bat.y)
    g.run(3.0)
    assert bat.alive and bat.hp==hp and (bat.x,bat.y)==pos


def t_ground_troop_takes_recoil_damage_and_a_one_tile_shove_with_a_swing_reset():
    g=quiet(Game());rg,cannon=giant_and_target(g)
    knight=create('knight',11,'blue',9,21.6);knight.spd=0;knight.dmg=0;g.deploy('blue',knight);hp=knight.hp;y0=knight.y
    shoved=False
    for _ in range(int(round(3.0/g.DT))):
        g.tick();shoved=shoved or any(s.kind=='knockback' for s in knight.statuses)
    assert hp-knight.hp>=81 and (hp-knight.hp)%81==0
    assert knight.y>y0+0.9
    # the shove goes through the knockback rules, so the knight's swing restarts like any knocked-back troop
    assert shoved


def t_shockwave_hits_a_building_that_is_not_its_target_without_moving_it():
    g=quiet(Game());rg=create('royal_giant',11,'red',9,20,evolved=True);rg.spd=0;g.deploy('red',rg)
    near=create('cannon',11,'blue',9,18.3);g.deploy('blue',near)
    tomb=create('tombstone',11,'blue',11.2,20.9);g.deploy('blue',tomb);y0=tomb.y;hp=tomb.hp
    for _ in range(int(round(3.0/g.DT))):
        g.tick();assert rg.tgt is None or rg.tgt is near
    # the Tombstone sits inside the 2.5 tile recoil radius but is not the shot's target: damaged, never displaced
    assert tomb.alive and tomb.y==y0 and hp-tomb.hp>81


@pytest.mark.parametrize('card',('pekka','prince'))
def t_knockback_immune_or_heavy_troops_take_damage_but_stand(card):
    g=quiet(Game());rg,cannon=giant_and_target(g)
    t=create(card,11,'blue',9,21.6);t.spd=0;t.dmg=0;g.deploy('blue',t);hp=t.hp;pos=(t.x,t.y)
    g.run(3.0)
    assert t.hp<hp and (t.x,t.y)==pos


def t_unevolved_royal_giant_has_no_shockwave():
    g=quiet(Game());rg=create('royal_giant',11,'red',9,20);rg.spd=0;g.deploy('red',rg)
    cannon=create('cannon',11,'blue',9,14.5);g.deploy('blue',cannon)
    knight=create('knight',11,'blue',9,21.6);knight.spd=0;knight.dmg=0;g.deploy('blue',knight);hp=knight.hp;y0=knight.y
    g.run(3.0)
    assert knight.hp==hp and knight.y==y0


def t_recoil_is_generated_by_the_shot_not_the_cannonball_landing():
    # the shockwave appears around him every time he attacks; the cannonball still needs its flight to reach the building
    g=quiet(Game());rg,cannon=giant_and_target(g)
    knight=create('knight',11,'blue',9,21.6);knight.spd=0;knight.dmg=0;g.deploy('blue',knight);hp=knight.hp;chp=cannon.hp
    recoil_t=None;impact_t=None
    for _ in range(int(round(3.0/g.DT))):
        g.tick()
        if recoil_t is None and knight.hp<hp:recoil_t=g.t
        if impact_t is None and cannon.hp<chp-cannon.decay*g.t-1:impact_t=g.t
    assert recoil_t is not None and impact_t is not None
    assert recoil_t<impact_t
    assert (hp-knight.hp)%81==0 and hp-knight.hp>=81
