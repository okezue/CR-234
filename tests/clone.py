import math

from sim.cards import create
from sim.game import Game
from tests.util import Dummy


def copied(name,team='blue',level=11):
    g=Game()
    for tower in g.arena.towers:tower.alive=False
    troops=create(name,level,team,9,10)
    original=troops[0] if isinstance(troops,list) else troops
    g.deploy(team,original)
    spell=create('clone',level,team,9,10);spell.apply(g)
    clone=next(t for t in g.players[team].troops if t is not original)
    return g,original,clone


def t_clone_ranged_attack_keeps_projectile_speed():
    for name in ('musketeer','archers','spear_goblins','baby_dragon'):
        for team in ('blue','red'):
            for level in (11,16):
                g,original,clone=copied(name,team,level)
                assert clone.proj_spd==original.proj_spd>0
                assert clone.hp==clone.max_hp==1 and clone.dmg==original.dmg
                target=Dummy(g._opp(team),clone.x,clone.y+3,hp=5000,spd=0,dmg=0)
                g.deploy(target.team,target);hp=target.hp
                g._fire(clone,target)
                assert target.hp==hp and len(g.projs)==1
                steps=math.ceil(3/(original.proj_spd*g.DT))
                for _ in range(steps-1):g._proc_projs();assert target.hp==hp
                g._proc_projs()
                assert target.hp==hp-clone.dmg and not g.projs
                assert original.hp==original.max_hp and original.proj_spd==clone.proj_spd


def t_clone_does_not_change_parent_projectile_or_existing_shot():
    g=Game()
    for tower in g.arena.towers:tower.alive=False
    original=create('musketeer',11,'blue',9,10);g.deploy('blue',original)
    target=Dummy('red',9,15,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    before=(original.hp,original.dmg,original.proj_spd,original.cd)
    g._fire(original,target);shot=g.projs[0]
    spell=create('clone',11,'blue',9,10);spell.apply(g)
    assert (original.hp,original.dmg,original.proj_spd,original.cd)==before
    assert g.projs==[shot] and target.hp==5000
    clone=next(t for t in g.players['blue'].troops if t is not original)
    g._fire(clone,target)
    assert len(g.projs)==2 and g.projs[0] is shot
    for _ in range(20):g._proc_projs()
    assert target.hp==5000-2*original.dmg and not g.projs
    assert clone.hp==clone.max_hp==1


def t_clone_homing_shot_survives_shooter_death():
    g,original,clone=copied('musketeer')
    target=Dummy('red',9,14.5,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    g._fire(clone,target)
    assert len(g.projs)==1 and target.hp==5000
    shot=g.projs[0];target.x=11
    g._proc_projs()
    assert shot.homing and (shot.tx,shot.ty)==(target.x,target.y)
    clone.take_damage(1);g._proc_deaths()
    assert not clone.alive and clone not in g.players['blue'].troops and original.alive
    for _ in range(30):g._proc_projs()
    assert target.hp==5000-clone.dmg and not g.projs
    for _ in range(4):g._proc_projs()
    assert target.hp==5000-clone.dmg


def t_clone_preserves_explicit_nonhoming_projectile():
    g=Game();original=create('musketeer',11,'blue',9,10)
    original.proj_homing=False;g.deploy('blue',original)
    create('clone',11,'blue',9,10).apply(g)
    clone=next(t for t in g.players['blue'].troops if t is not original)
    target=Dummy('red',9,14.5,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    g._fire(clone,target)
    assert len(g.projs)==1 and target.hp==5000
    shot=g.projs[0]
    assert not shot.homing
    target.x=15
    for _ in range(20):g._proc_projs()
    assert (shot.tx,shot.ty)==(9,14.5) and target.hp==5000 and not g.projs
    assert original.proj_homing is False


def t_clone_melee_and_duplicate_exclusion_unchanged():
    g,original,clone=copied('knight')
    assert original.proj_spd==clone.proj_spd==0
    target=Dummy('red',9,11,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    g._fire(clone,target)
    assert target.hp==5000-clone.dmg and not g.projs
    clone_spell=create('clone',11,'blue',9,10);clone_spell.apply(g)
    assert len(g.players['blue'].troops)==3
    clone_spell.apply(g)
    assert len(g.players['blue'].troops)==3
    clone.take_damage(1)
    assert not clone.alive and original.hp==original.max_hp


def t_clone_spirit_projectile_consumes_once_on_impact():
    g,original,clone=copied('fire_spirit')
    assert clone.is_suicide and clone.proj_spd==original.proj_spd>0
    target=Dummy('red',9,12.5,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    g._fire(clone,target)
    assert clone.alive and target.hp==5000 and len(g.projs)==1
    for _ in range(20):g._proc_projs()
    assert not clone.alive and target.hp==5000-clone.dmg
    g._proc_deaths()
    assert clone not in g.players['blue'].troops
    for _ in range(5):g._proc_projs()
    assert target.hp==5000-clone.dmg


def t_clone_automatic_ranged_attack_has_flight_time():
    g,original,clone=copied('musketeer')
    g.players['blue'].troops.remove(original)
    target=Dummy('red',clone.x,clone.y+5,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    while not g.projs and g.t<1:g.tick()
    assert g.projs and target.hp==5000
    launch=g.t
    while target.hp==5000 and g.t<2:g.tick()
    assert target.hp==5000-clone.dmg and g.t-launch>=0.2


def t_clone_preserves_special_shooter_delivery_config():
    for name in ('hunter','firecracker','executioner','bowler'):
        _,original,clone=copied(name)
        assert (clone.proj_spd,clone.dmg,clone.ct_dmg)==(original.proj_spd,original.dmg,original.ct_dmg)
        assert clone.proj_spd>0
        assert [type(c) for c in clone.components]==[type(c) for c in original.components]
