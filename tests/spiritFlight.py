import pytest

from sim.cards import create
from sim.fx import CurseOnHit,SoulCollect
from sim.game import Game
from sim.units import Troop


DECK=['fire_spirit','knight','archers','fireball','giant','musketeer','bomber','arrows']


def quiet():
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0})
    for tower in g.arena.towers:
        tower.rng=0
        if tower.troop:tower.troop.RNG=0
    return g


def target(g,team,x,y):
    t=Troop(team,x,y,{'hp':5000,'dmg':0,'spd':0,'hspd':1,'rng':0,'name':'Stationary control'})
    g.deploy(team,t)
    return t


def placed(team='blue',level=11):
    g=quiet();p=g.players[team];p.deck.hand=DECK[:4];p.elixir=10
    y=10.5 if team=='blue' else 21.5
    t=target(g,g._opp(team),9.5,y+(2.9 if team=='blue' else -2.9))
    p.card_levels['fire_spirit']=level
    assert g.play_card(team,'fire_spirit',9.5,y)==(True,'ok')
    g.tick();spirit=next(t for t in p.troops if t.name=='Fire Spirit')
    return g,spirit,t


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('level',(11,16))
def t_fire_spirit_converts_to_one_projectile_before_damage(team,level):
    g,spirit,t=placed(team,level)
    while not g.projs and g.t<2:g.tick()
    assert len(g.projs)==1 and t.hp==t.max_hp
    assert not spirit.alive and spirit not in g.players[team].troops
    release=g.t;g.run(1)
    assert t.hp==t.max_hp-spirit.dmg and not g.projs
    assert release<g.t


def t_fire_spirit_flight_has_no_clone_or_spell_target_body():
    g,spirit,t=placed()
    while not g.projs and g.t<2:g.tick()
    create('clone',11,'blue',spirit.x,spirit.y).apply(g)
    create('lightning',11,'red',spirit.x,spirit.y).apply(g)
    assert not any(x.name=='Fire Spirit' for x in g.players['blue'].troops)
    assert len(g.projs)==1 and t.hp==5000
    g.run(1);assert t.hp==5000-spirit.dmg


def t_fire_spirit_projectile_damages_primary_and_splash_once_after_cleanup():
    g,spirit,t=placed();secondary=target(g,'red',11,13.4)
    while not g.projs and g.t<2:g.tick()
    assert t.hp==secondary.hp==5000
    g._proc_deaths();g._proc_deaths();g.run(1)
    assert t.hp==secondary.hp==5000-spirit.dmg


def t_retired_fire_spirit_cannot_launch_again():
    g,spirit,t=placed()
    while not g.projs and g.t<2:g.tick()
    g._fire(spirit,t)
    assert len(g.projs)==1
    g.run(1);assert t.hp==5000-spirit.dmg


def t_prelaunch_fire_spirit_death_produces_no_projectile_or_damage():
    g,spirit,t=placed();spirit.take_damage(spirit.hp);g.run(2)
    assert not g.projs and t.hp==5000


def t_cloned_fire_spirit_also_has_one_projectile_and_no_body():
    g=quiet();spirit=create('fire_spirit',11,'blue',9.5,10.5);g.deploy('blue',spirit)
    create('clone',11,'blue',spirit.x,spirit.y).apply(g)
    clone=next(t for t in g.players['blue'].troops if getattr(t,'is_clone',False))
    spirit.take_damage(spirit.hp);g._proc_deaths()
    t=target(g,'red',clone.x,clone.y+2.9)
    while not g.projs and g.t<1:g.tick()
    assert not clone.alive and clone not in g.players['blue'].troops and t.hp==5000
    g.run(1);assert t.hp==5000-clone.dmg


def t_furnace_fire_spirit_uses_same_projectile_body_transition():
    g=quiet();furnace=create('furnace',11,'blue',9.5,10.5);g.deploy('blue',furnace)
    while not any(t.name=='Fire Spirit' for t in g.players['blue'].troops) and g.t<20:g.tick()
    spirit=next(t for t in g.players['blue'].troops if t.name=='Fire Spirit')
    furnace.take_damage(furnace.hp);g._proc_deaths()
    t=target(g,'red',spirit.x,spirit.y+2.9)
    while not g.projs and g.t<22:g.tick()
    assert not spirit.alive and spirit not in g.players['blue'].troops and t.hp==5000
    g.run(1);assert t.hp==5000-spirit.dmg


@pytest.mark.parametrize('self_destruct',(False,True))
def t_cursed_fire_spirit_only_becomes_hog_on_external_death(self_destruct):
    g,spirit,t=placed();witch=create('mother_witch',11,'red',16,26);g.deploy('red',witch)
    curse=next(c for c in witch.components if isinstance(c,CurseOnHit))
    g._do_attack(witch,spirit);assert spirit.alive and curse.marks
    if not self_destruct:spirit.take_damage(spirit.hp)
    g.run(2)
    hogs=[x for x in g.players['red'].troops if x.name=='Cursed Hog']
    assert len(hogs)==(0 if self_destruct else 1)
    assert t.hp==(5000-spirit.dmg if self_destruct else 5000)


def t_fire_spirit_self_destruction_still_supplies_a_soul():
    g,spirit,_=placed();king=create('skeleton_king',11,'red',16,26);g.deploy('red',king)
    souls=next(c for c in king.components if isinstance(c,SoulCollect))
    g.tick();assert spirit in souls._prev and souls.souls==0
    g.run(2);assert souls.souls==1


@pytest.mark.parametrize('card,evolved',(('ice_spirit',False),('ice_spirit',True),('wall_breakers',True),('battle_ram',False)))
def t_other_suicide_units_keep_existing_release_behavior(card,evolved):
    g=quiet();units=create(card,11,'blue',9.5,10.5,evolved=evolved)
    tr=units[0] if isinstance(units,list) else units;g.deploy('blue',tr)
    t=target(g,'red',9.5,13.4)
    if tr.proj_spd>0:
        g._fire(tr,t)
        assert tr.alive and len(g.projs)==1 and t.hp==5000
    else:
        g._fire(tr,t)
        assert t.hp<5000
    assert not getattr(tr,'_self_destructed',False)


def t_incoming_attack_cannot_destroy_launched_fire_spirit_projectile():
    g,spirit,t=placed()
    while not g.projs and g.t<2:g.tick()
    hit=[]
    def impact(game,projectile):
        spirit.take_damage(5000);hit.append(game.t)
    g._shoot('red',spirit.x,spirit.y-1,8,spirit,impact)
    g.run(1)
    assert len(hit)==1 and t.hp==5000-spirit.dmg and spirit._self_destructed


def t_two_witch_marks_are_discarded_after_fire_spirit_launch():
    g,spirit,t=placed(level=16);marks=[]
    for x in (15,17):
        witch=create('mother_witch',9,'red',x,26);g.deploy('red',witch)
        g._do_attack(witch,spirit)
        marks.append(next(c for c in witch.components if isinstance(c,CurseOnHit)))
    assert spirit.alive and all(c.marks for c in marks)
    g.run(2)
    assert not any(x.name=='Cursed Hog' for x in g.players['red'].troops)
    assert all(not c.marks for c in marks) and t.hp==5000-spirit.dmg


def t_fire_spirit_projectile_can_damage_crown_after_body_cleanup():
    g=quiet();spirit=create('fire_spirit',11,'blue',3.5,20);g.deploy('blue',spirit)
    tower=g.arena.get_tower('red','princess','left');before=tower.hp
    while not g.projs and g.t<10:g.tick()
    assert len(g.projs)==1 and not spirit.alive and tower.hp==before
    g.run(1);assert tower.hp==before-spirit.dmg


def t_environment_action_removes_spirit_body_before_impact():
    import numpy as np
    from sim.env import CREnv,GAME_FEAT,PLAYER_FEAT,N_TOWERS,TOWER_FEAT,TROOP_FEAT
    env=CREnv(blue_deck=DECK,decision_freq=1);env.reset(seed=7);g=env.game;p=g.players['blue']
    p.deck.hand=DECK[:4];p.elixir=10;p.drag_del=0
    for tw in g.arena.towers:
        tw.rng=0
        if tw.troop:tw.troop.RNG=0
    t=target(g,'red',9.5,13.4)
    obs,*_=env.step({'card':0,'x':9.5,'y':10.5})
    while not g.projs and g.t<2:obs,*_=env.step(4)
    start=GAME_FEAT+2*PLAYER_FEAT+N_TOWERS*TOWER_FEAT
    assert env.observation_space.contains(obs) and not np.any(obs[start:start+TROOP_FEAT])
    assert t.hp==5000 and len(g.projs)==1
    for _ in range(20):obs,*_=env.step(4)
    assert t.hp==4793 and env.observation_space.contains(obs)
    env.close()


def t_ordinary_projectile_shooter_survives_release():
    g=quiet();musketeer=create('musketeer',11,'blue',9.5,10.5);g.deploy('blue',musketeer)
    t=target(g,'red',9.5,15)
    while not g.projs and g.t<2:g.tick()
    assert musketeer.alive and musketeer in g.players['blue'].troops and t.hp==5000
    g.run(0.25);assert t.hp==5000-musketeer.dmg
