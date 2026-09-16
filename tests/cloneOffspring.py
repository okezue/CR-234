import pytest

from sim.cards import create
from sim.fx import DeathSpawn,SpawnTimer
from sim.game import Game
from tests.util import Dummy


def setup(name,team='blue',cloned=True):
    g=Game()
    for t in g.arena.towers:t.alive=False
    parent=create(name,11,team,9,10);parent.spd=0;g.deploy(team,parent)
    if not cloned:return g,parent,None
    create('clone',11,team,9,10).apply(g)
    clone=next(t for t in g.players[team].troops if t is not parent);clone.spd=0
    return g,parent,clone


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('name',('golem','lava_hound','goblin_giant','skeleton_barrel','battle_ram'))
def t_cloned_death_spawner_produces_cloned_children(name,team):
    g,parent,clone=setup(name,team);parent.x,parent.y=1,1
    effect=next(c for c in clone.components if isinstance(c,DeathSpawn))
    clone.take_damage(1);g._proc_deaths();g.run(.7)
    children=[t for t in g.players[team].troops if t is not parent]
    assert len(children)==effect.count
    assert all(t.hp==t.max_hp==1 and t.is_clone for t in children)
    assert parent.hp==parent.max_hp and not getattr(parent,'is_clone',False)
    target=Dummy(g._opp(team),9,12,hp=100000,spd=0,dmg=0);g.deploy(target.team,target)
    hp=target.hp;g._do_attack(children[0],target)
    assert target.hp==hp-effect.cfg['dmg']
    for t in children:t.take_damage(1)
    assert not any(t.alive for t in children)


@pytest.mark.parametrize('name',('golem','lava_hound','skeleton_barrel'))
def t_wounded_normal_spawner_children_remain_normal(name):
    g,parent,_=setup(name,cloned=False);effect=next(c for c in parent.components if isinstance(c,DeathSpawn))
    parent.hp=1;parent.take_damage(1);g._proc_deaths();g.run(.7)
    children=g.players['blue'].troops
    assert len(children)==effect.count
    assert all(t.max_hp==effect.cfg['hp'] and t.max_hp>1 and not getattr(t,'is_clone',False) for t in children)


@pytest.mark.parametrize('name',('witch','night_witch'))
@pytest.mark.parametrize('team',('blue','red'))
def t_periodic_clone_children_are_fragile_and_original_timer_is_independent(name,team):
    g,parent,clone=setup(name,team)
    original=next(c for c in parent.components if isinstance(c,SpawnTimer));copied=next(c for c in clone.components if isinstance(c,SpawnTimer))
    assert original is not copied and original.timer==copied.timer
    original_before=original.timer;copied.on_tick(clone,g);assert original.timer==original_before
    for _ in range(120):g.tick()
    normal=[t for t in g.players[team].troops if getattr(t,'_spawner',None) is parent]
    descendants=[t for t in g.players[team].troops if getattr(t,'_spawner',None) is clone]
    assert normal and descendants
    assert all(t.max_hp==original.cfg['hp'] and not getattr(t,'is_clone',False) for t in normal)
    assert all(t.hp==t.max_hp==1 and t.is_clone for t in descendants)
    target=Dummy(g._opp(team),9,13,hp=100000,spd=0,dmg=0);g.deploy(target.team,target)
    hp=target.hp;g._do_attack(descendants[0],target);assert target.hp==hp-copied.cfg['dmg']
    before=list(g.players[team].troops)
    parent.x,parent.y=1,1;clone.x,clone.y=1,2
    for t in normal:t.x,t.y=1,3
    target=descendants[0];target.x,target.y=9,10
    create('clone',11,team,9,10).apply(g)
    assert g.players[team].troops==before


def t_cloning_partway_through_spawn_timer_preserves_original_cadence():
    g,parent,_=setup('witch',cloned=False)
    original=next(c for c in parent.components if isinstance(c,SpawnTimer))
    g.run(.4);phase=original.timer
    create('clone',11,'blue',parent.x,parent.y).apply(g)
    clone=next(t for t in g.players['blue'].troops if t.name==parent.name and t is not parent)
    copied=next(c for c in clone.components if isinstance(c,SpawnTimer))
    assert copied is not original and copied.timer==original.timer==phase
    times={}
    for _ in range(100):
        g.tick()
        for owner in (parent,clone):
            if owner not in times and any(getattr(t,'_spawner',None) is owner for t in g.players['blue'].troops):times[owner]=g.t
        if len(times)==2:break
    assert len(times)==2 and times[parent]==times[clone]


def t_clone_identity_survives_multiple_death_generations():
    g,parent,clone=setup('elixir_golem');parent.x,parent.y=1,1
    clone.take_damage(1);g._proc_deaths();g.run(.5)
    middle=[t for t in g.players['blue'].troops if t is not parent]
    assert len(middle)==2 and all(t.is_clone and t.max_hp==1 for t in middle)
    for t in middle:t.take_damage(1)
    g._proc_deaths();g.run(.5)
    final=[t for t in g.players['blue'].troops if t is not parent]
    assert len(final)==4 and all(t.is_clone and t.hp==t.max_hp==1 for t in final)


def t_cloned_offspring_shield_keeps_one_hit_without_overflow():
    g,parent,clone=setup('golem');effect=next(c for c in clone.components if isinstance(c,DeathSpawn))
    effect.cfg=dict(effect.cfg,shield_hp=200,max_shield_hp=200)
    clone.take_damage(1);g._proc_deaths();g.run(.5)
    child=next(t for t in g.players['blue'].troops if t is not parent)
    assert child.hp==child.max_hp==child.shield_hp==child.max_shield_hp==1
    child.take_damage(100);assert child.alive and child.hp==1 and child.shield_hp==0
    child.take_damage(1);assert not child.alive


def t_cloned_golemite_retains_death_damage():
    g,parent,clone=setup('golem');parent.x,parent.y=1,1;clone.take_damage(1);g._proc_deaths();g.run(.5)
    child=next(t for t in g.players['blue'].troops if t is not parent);child.x,child.y=9,10
    target=Dummy('red',9,11,hp=100000,spd=0,dmg=0);g.deploy('red',target)
    hp=target.hp;damage=child.death_dmg;assert damage>0
    child.take_damage(1);g._proc_deaths()
    assert target.hp==hp-damage
