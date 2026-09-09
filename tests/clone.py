import math

from sim.cards import create
from sim.game import Game
from sim.units import has
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


def shielded_source(name,team,level):
    g=Game()
    for tower in g.arena.towers:tower.alive=False
    if name=='royal_delivery':
        create(name,level,team,9,10).apply(g)
        original=g.players[team].troops[0]
    else:
        troops=create(name,level,team,9,10)
        original=troops[0] if isinstance(troops,list) else troops
        original.x,original.y=9,10;g.deploy(team,original)
    return g,original


def t_clone_live_shield_absorbs_one_hit():
    for name in ('guards','dark_prince','royal_recruits','royal_delivery'):
        for team in ('blue','red'):
            for level in (11,16):
                g,original=shielded_source(name,team,level)
                before=(original.hp,original.shield_hp,original.max_shield_hp,original.cd,original.dmg,original.proj_spd)
                create('clone',level,team,original.x,original.y).apply(g)
                clone=next(t for t in g.players[team].troops if t is not original)
                assert clone.hp==clone.max_hp==clone.shield_hp==clone.max_shield_hp==1
                clone.take_damage(10000)
                assert clone.alive and clone.hp==1 and clone.shield_hp==0
                assert (original.hp,original.shield_hp,original.max_shield_hp,original.cd,original.dmg,original.proj_spd)==before
                clone.take_damage(1)
                assert not clone.alive and original.alive


def t_clone_shielded_evolution_leader_keeps_one_hit_layer():
    g=Game();troops=create('skeleton_army',11,'blue',9,10,evolved=True)
    original=next(t for t in troops if t.shield_hp>0)
    original.x,original.y=9,10;g.deploy('blue',original)
    shield=original.shield_hp
    create('clone',11,'blue',9,10).apply(g)
    clone=next(t for t in g.players['blue'].troops if t is not original)
    assert clone.hp==clone.max_hp==clone.shield_hp==clone.max_shield_hp==1
    clone.take_damage(10000)
    assert clone.alive and clone.hp==1 and clone.shield_hp==0
    clone.take_damage(1)
    assert not clone.alive and original.shield_hp==shield


def t_clone_damaged_shield_is_one_but_broken_shield_is_absent():
    for remaining in ('half','one','broken'):
        g,original=shielded_source('guards','blue',11)
        initial=original.shield_hp
        original.take_damage(initial//2 if remaining=='half' else initial-1 if remaining=='one' else initial)
        hp,shield=original.hp,original.shield_hp
        assert original.max_shield_hp==initial
        create('clone',11,'blue',9,10).apply(g)
        clone=next(t for t in g.players['blue'].troops if t is not original)
        expected=0 if remaining=='broken' else 1
        assert clone.shield_hp==clone.max_shield_hp==expected
        clone.take_damage(10000)
        assert clone.alive==(remaining!='broken')
        assert (original.hp,original.shield_hp)==(hp,shield)


def t_unshielded_clone_and_damaged_original_stay_one_hit():
    for kind in ('knight','dummy'):
        g=Game();original=create('knight',11,'blue',9,10) if kind=='knight' else Dummy('blue',9,10)
        original.hp=1;g.deploy('blue',original)
        create('clone',11,'blue',9,10).apply(g)
        clone=next(t for t in g.players['blue'].troops if t is not original)
        assert clone.hp==clone.max_hp==1 and clone.shield_hp==clone.max_shield_hp==0
        clone.take_damage(1)
        assert not clone.alive and original.alive


def t_clone_shield_does_not_prevent_spell_status_or_second_hit():
    for spell in ('zap','fireball'):
        g,original=shielded_source('guards','blue',11)
        create('clone',11,'blue',9,10).apply(g)
        clone=next(t for t in g.players['blue'].troops if t is not original)
        create(spell,11,'red',clone.x,clone.y).apply(g)
        assert clone.alive and clone.hp==1 and clone.shield_hp==0
        if spell=='zap':assert has(clone,'stun')
        create('zap',11,'red',clone.x,clone.y).apply(g)
        assert not clone.alive


def t_clone_shield_tower_hits_and_reclone_exclusion():
    g,original=shielded_source('guards','blue',11)
    spell=create('clone',11,'blue',9,10);spell.apply(g)
    clone=next(t for t in g.players['blue'].troops if t is not original)
    spell.apply(g);assert len(g.players['blue'].troops)==2
    other=create('clone',11,'blue',9,10);other.apply(g)
    assert len(g.players['blue'].troops)==3
    tower=g.arena.get_tower('red','princess','left')
    g._tower_hit(tower,clone,tower.troop.dmg)
    assert clone.alive and clone.hp==1 and clone.shield_hp==0
    g._tower_hit(tower,clone,tower.troop.dmg)
    assert not clone.alive
    other_clone=next(t for t in g.players['blue'].troops if t is not original and t is not clone)
    assert other_clone.shield_hp==1 and other_clone.hp==1


def t_shielded_clone_survives_first_tower_projectile():
    g=Game();original=create('guards',11,'blue',3.5,20)[0]
    original.x,original.y=3.5,20;g.deploy('blue',original)
    create('clone',11,'blue',3.5,20).apply(g)
    clone=next(t for t in g.players['blue'].troops if t is not original)
    g.players['blue'].troops.remove(original);clone.spd=0;clone.dmg=0
    g.tick()
    assert g.projs and clone.shield_hp==1
    for _ in range(20):
        g.tick()
        if clone.shield_hp==0:break
    assert clone.alive and clone.hp==1 and clone.shield_hp==0 and clone in g.players['blue'].troops
    for _ in range(30):
        g.tick()
        if not clone.alive:break
    assert not clone.alive and clone not in g.players['blue'].troops
    assert original.hp==original.max_hp and original.shield_hp==original.max_shield_hp


def t_cloning_wizard_shield_does_not_consume_original_burst():
    from sim.fx import ShieldBurst
    g=Game();original=create('wizard',11,'blue',9,10,evolved=True);g.deploy('blue',original)
    component=next(c for c in original.components if isinstance(c,ShieldBurst))
    enemy=Dummy('red',10,10,hp=5000,spd=0,dmg=0);g.deploy('red',enemy)
    original_state=(original.hp,original.shield_hp,original.max_shield_hp,component.done)
    create('clone',11,'blue',9,10).apply(g)
    clone=next(t for t in g.players['blue'].troops if t is not original)
    assert clone.shield_hp==clone.max_shield_hp==1
    clone.take_damage(10000)
    assert clone.alive and clone.hp==1 and clone.shield_hp==0
    assert (original.hp,original.shield_hp,original.max_shield_hp,component.done)==original_state
    before=enemy.hp;original.take_damage(original.shield_hp)
    assert component.done and enemy.hp==before-component.dmg


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
