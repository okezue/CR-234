import pytest

from sim.cards import create, load
from sim.game import Game
from tests.util import quiet


def t_goblin_hut_lifetime_is_thirty_seconds():
    # game data GoblinHut lifeTime 30000; the May 2025 update raised it from 29
    c=load()['cards']['goblin_hut'];assert c['lifetime']==30.0
    hut=create('goblin_hut',11,'blue',9,10)
    assert hut.lifetime==30.0 and hut.decay==pytest.approx(hut.max_hp/30.0)


def t_goblin_hut_decays_to_zero_at_thirty_seconds_not_twenty_nine():
    g=quiet(Game());hut=create('goblin_hut',11,'blue',9,10);g.deploy('blue',hut)
    # the lifetime decays from placement; a 29 second hut is gone before the 29.5 second mark
    g.run(29.5)
    assert hut.alive and hut.hp>0
    g.run(0.6)
    assert not hut.alive


@pytest.mark.parametrize('name,lifetime',(('cannon',30.0),('tesla',25.0),('goblin_cage',20.0),('tombstone',30.0)))
def t_other_building_lifetimes_unchanged(name,lifetime):
    assert load()['cards'][name]['lifetime']==lifetime


def t_battle_ram_carries_its_own_geometry_not_the_barbarians():
    # game data BattleRam range 500 and collisionRadius 750; its death-spawned Barbarian has range 700 and collision 500
    c=load()['cards'];ram=create('battle_ram',11,'blue',9,10);ram=ram[0] if isinstance(ram,list) else ram
    assert (c['battle_ram']['range'],c['battle_ram']['collisionRadius'])==(0.5,0.75)
    assert (ram.rng,ram.collision_r)==(0.5,0.75)
    barbs=c['battle_ram']['units']['barbarian'];assert barbs['range']==0.7
    assert (c['barbarians']['range'],c['barbarians']['collisionRadius'])==(0.7,0.5)


def cannon_r():
    return create('cannon',11,'red',0,0).collision_r


def t_battle_ram_reach_is_edge_to_edge_with_its_wider_body():
    g=quiet(Game());ram=create('battle_ram',11,'blue',9,10);ram=ram[0] if isinstance(ram,list) else ram;g.deploy('blue',ram)
    cannon=create('cannon',11,'red',9,10+0.75+cannon_r()+0.5);g.deploy('red',cannon)
    # centres ram radius plus cannon radius plus 0.5 apart: edge distance 0.5 equals the ram's range, so it is in reach without moving
    assert g._dist(ram,cannon)==pytest.approx(0.5)
    assert g._dist(ram,cannon)<=ram.rng
