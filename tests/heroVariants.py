import pytest

from sim.cards import create
from sim.fx import EvoRoyalGhost,Fade,FrostyFella
from sim.game import Game
from sim.units import Status,has
from tests.util import Dummy,quiet


def snowmen(g,team):return [t for t in g.players[team].troops if t.name=='Snowman']


def arm(g,tr):
    tr.ability.cd=0;g.players[tr.team].elixir=10
    assert g.activate_ability(tr.team,tr)==(True,'ok')
    g.run(1.2)


@pytest.mark.parametrize('team',('blue','red'))
def t_ice_wizard_waits_for_next_slowing_impact(team):
    g=quiet(Game());hero=create('ice_wizard',11,team,9,10,hero=True);hero.spd=0;g.deploy(team,hero)
    assert isinstance(hero.ability,FrostyFella)
    arm(g,hero);assert not snowmen(g,team)
    assert hero.ability.active and hero.ability.uses==0 and not hero.ability.can_use()
    target=Dummy(g._opp(team),9,14,hp=5000,spd=0,dmg=0);g.deploy(target.team,target)
    hero.tgt=target;g._fire(hero,target)
    assert not snowmen(g,team)
    g._proc_projs()
    assert not snowmen(g,team)
    for _ in range(20):g._proc_projs()
    assert has(target,'slow') and len(snowmen(g,team))==1
    snow=snowmen(g,team)[0]
    assert (snow.x,snow.y)==(9,15) and not hero.ability.active
    hp=target.hp;g._do_attack(hero,target)
    assert len(snowmen(g,team))==1 and target.hp==hp-hero.dmg
    g.tick();assert has(target,'freeze')


def t_ice_wizard_uses_hit_target_not_new_target():
    g=quiet(Game());hero=create('ice_wizard',11,'blue',9,10,hero=True);hero.spd=0;g.deploy('blue',hero);arm(g,hero)
    hit=Dummy('red',9,14,hp=5000,spd=0,dmg=0);other=Dummy('red',13,10,hp=5000,spd=0,dmg=0)
    for t in (hit,other):g.deploy('red',t)
    hero.tgt=other;g._do_attack(hero,hit)
    assert len(snowmen(g,'blue'))==1 and (snowmen(g,'blue')[0].x,snowmen(g,'blue')[0].y)==(9,15)


@pytest.mark.parametrize('blocked',('dead_hero','immune_target','no_slow','dead_target'))
def t_ice_wizard_requires_living_hero_and_valid_slow(blocked):
    g=quiet(Game());hero=create('ice_wizard',11,'blue',9,10,hero=True);hero.spd=0;g.deploy('blue',hero);arm(g,hero)
    target=Dummy('red',9,14,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    if blocked=='dead_hero':hero.alive=False
    elif blocked=='immune_target':target.statuses.append(Status('invincible',10))
    elif blocked=='no_slow':hero.slow_dur=0
    else:target.hp=1
    g._do_attack(hero,target)
    assert not snowmen(g,'blue')


def t_ice_wizard_can_trigger_on_crown_tower():
    g=Game();hero=create('ice_wizard',11,'blue',14.5,20,hero=True);hero.spd=0;g.deploy('blue',hero)
    hero.ability.activate(hero,g)
    tower=g.arena.get_tower('red','princess','right');hp=tower.hp
    g._do_attack(hero,tower)
    assert len(snowmen(g,'blue'))==1 and snowmen(g,'blue')[0].y==26.5
    assert tower.hp==hp-hero.dmg-hero.ability.ct and has(tower,'slow')


def t_ice_wizard_clone_cannot_consume_armed_original():
    g=quiet(Game());hero=create('ice_wizard',11,'blue',9,10,hero=True);hero.spd=0;g.deploy('blue',hero);arm(g,hero)
    create('clone',11,'blue',9,10).apply(g)
    clone=next(t for t in g.players['blue'].troops if t is not hero)
    target=Dummy('red',9,14,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    clone.slow_dur=hero.slow_dur;clone.slow_val=hero.slow_val
    g._do_attack(clone,target)
    assert hero.ability.active and not snowmen(g,'blue')
    g._do_attack(hero,target);assert len(snowmen(g,'blue'))==1


def t_ice_wizard_projectile_in_flight_before_arming():
    g=quiet(Game());hero=create('ice_wizard',11,'blue',9,10,hero=True);g.deploy('blue',hero)
    target=Dummy('red',9,14,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    g._fire(hero,target);assert not snowmen(g,'blue')
    hero.ability.activate(hero,g)
    for _ in range(20):g._proc_projs()
    assert len(snowmen(g,'blue'))==1


def t_ice_wizard_shield_hit_triggers_but_dead_primary_does_not():
    g=quiet(Game());hero=create('ice_wizard',11,'blue',9,10,hero=True);g.deploy('blue',hero);hero.ability.activate(hero,g)
    target=create('knight',11,'red',9,14);target.shield_hp=5000;g.deploy('red',target);hp=target.hp
    g._do_attack(hero,target);assert len(snowmen(g,'blue'))==1 and target.hp==hp
    g=quiet(Game());hero=create('ice_wizard',11,'blue',9,10,hero=True);g.deploy('blue',hero);hero.ability.activate(hero,g)
    dead=Dummy('red',9,14,hp=1,spd=0,dmg=0);nearby=Dummy('red',10,14,hp=5000,spd=0,dmg=0)
    for t in (dead,nearby):g.deploy('red',t)
    g._do_attack(hero,dead)
    assert not dead.alive and has(nearby,'slow') and hero.ability.active and not snowmen(g,'blue')


def t_ice_wizard_target_dies_before_projectile_arrives():
    g=quiet(Game());hero=create('ice_wizard',11,'blue',9,10,hero=True);g.deploy('blue',hero);hero.ability.activate(hero,g)
    target=Dummy('red',9,14,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    g._fire(hero,target);target.take_damage(5000)
    for _ in range(20):g._proc_projs()
    assert hero.ability.active and not snowmen(g,'blue')


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('level',(11,16))
def t_royal_ghost_two_souldiers_attack_after_reveal(team,level):
    g=quiet(Game());ghost=create('royal_ghost',level,team,9,10,evolved=True);ghost.spd=0;g.deploy(team,ghost)
    g.tick();assert has(ghost,'invisible')
    enemy=Dummy(g._opp(team),10,10,hp=50000,spd=0,dmg=0);g.deploy(enemy.team,enemy)
    g._do_attack(ghost,enemy);g.tick()
    souls=[t for t in g.players[team].troops if t is not ghost]
    assert len(souls)==2 and all(t.name=='Souldier' for t in souls)
    assert len({id(next(c for c in t.components if isinstance(c,Fade))) for t in souls})==2
    for soul in souls:
        soul.spd=0;soul.x,soul.y=9,10
    ghost.x,ghost.y=1,1;hp=enemy.hp
    g.run(1.2)
    assert enemy.hp<=hp-sum(t.dmg for t in souls)
    assert all(t.alive for t in souls)
    before=len(souls);g._do_attack(ghost,enemy)
    evo=next(c for c in ghost.components if isinstance(c,EvoRoyalGhost));evo.on_tick(ghost,g)
    assert len(g.players[team].troops)==before+1
    ghost.statuses.append(Status('invisible',10));evo.on_tick(ghost,g)
    g._do_attack(ghost,enemy);evo.on_tick(ghost,g)
    assert len(g.players[team].troops)==before+3
    enemy.alive=False;ghost.alive=False;g._proc_deaths()
    for tower in g.arena.towers:tower.alive=False
    g.run(2.5)
    assert not any(t.alive for t in g.players[team].troops)
