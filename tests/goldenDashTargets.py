import pytest

from sim.cards import create
from sim.game import Game


DECK=['golden_knight','knight','archers','fireball','giant','musketeer','bomber','arrows']


def setup(team='blue'):
    g=Game(p1={'deck':DECK,'drag_del':0,'ability_del':.05,'ability_std':0},p2={'deck':DECK,'drag_del':0,'ability_del':.05,'ability_std':0})
    for tw in g.arena.towers:
        tw.rng=0
        if tw.troop:tw.troop.RNG=0
    p=g.players[team];p.deck.hand=DECK[:4];p.elixir=10
    y=10.5 if team=='blue' else 21.5
    assert g.play_card(team,'golden_knight',9.5,y)==(True,'ok');g.run(2.2)
    knight=next(t for t in p.troops if t.name=='Golden Knight');knight.spd=0
    return g,knight


def target(g,card,team,x,y):
    t=create(card,11,team,x,y)
    if isinstance(t,list):t=t[0]
    t.spd=0;t.dmg=0;t.rng=0;g.deploy(team,t)
    return t


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('air_card',('baby_dragon','minions','balloon'))
def t_dash_ignores_air_targets_even_when_they_are_closest(team,air_card):
    g,knight=setup(team);opp=g._opp(team);direction=1 if team=='blue' else -1
    air=target(g,air_card,opp,knight.x,knight.y+2*direction)
    ground=target(g,'giant',opp,knight.x+3,knight.y+2*direction)
    air_hp=air.hp;ground_hp=ground.hp;p=g.players[team];p.elixir=10
    assert g.activate_ability(team,knight)==(True,'ok');g.run(1.4)
    assert air.hp==air_hp and air not in knight.ability.hit
    assert ground.hp<ground_hp and ground in knight.ability.hit
    assert knight.ability.dashes==1


@pytest.mark.parametrize('team',('blue','red'))
def t_air_only_does_not_become_dash_destination_or_take_damage(team):
    g,knight=setup(team);opp=g._opp(team)
    air=target(g,'baby_dragon',opp,knight.x+2,knight.y);before=air.hp;position=knight.x,knight.y
    g.players[team].elixir=10;assert g.activate_ability(team,knight)==(True,'ok');g.run(1.4)
    assert air.hp==before and knight.ability.dashes==0 and not knight.ability.hit
    assert (knight.x,knight.y)==position


@pytest.mark.parametrize('card,eligible',(('battle_healer',True),('baby_dragon',False)))
def t_cloned_ground_and_air_targets_keep_dash_eligibility(card,eligible):
    g,knight=setup();original=target(g,card,'red',knight.x,knight.y+3)
    create('clone',11,'red',original.x,original.y).apply(g)
    clone=next(t for t in g.players['red'].troops if getattr(t,'is_clone',False))
    original.take_damage(original.hp);g._proc_deaths()
    g.players['blue'].elixir=10;assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.4)
    assert (clone in knight.ability.hit)==eligible
    assert clone.alive is not eligible


@pytest.mark.parametrize('card,eligible',(('golem',True),('lava_hound',False)))
def t_spawned_ground_and_air_children_keep_dash_eligibility(card,eligible):
    g,knight=setup();parent=target(g,card,'red',knight.x,knight.y+3)
    parent.take_damage(parent.hp);g._proc_deaths();children=list(g.players['red'].troops)
    assert children
    for child in children:child.spd=0;child.dmg=0;child.rng=0
    g.players['blue'].elixir=10;assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.4)
    assert any(child in knight.ability.hit for child in children)==eligible


def t_hovering_ground_healer_remains_a_dash_target():
    g,knight=setup();healer=target(g,'battle_healer','red',knight.x,knight.y+3)
    assert healer.hovering and healer.transport=='Ground'
    g.players['blue'].elixir=10;assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.4)
    assert healer in knight.ability.hit


def t_environment_dash_preserves_air_target_health():
    from sim.env import CREnv
    env=CREnv(blue_deck=DECK,decision_freq=1);env.reset(seed=4);g=env.game;p=g.players['blue']
    p.deck.hand=DECK[:4];p.elixir=10;p.drag_del=0
    for tw in g.arena.towers:
        tw.rng=0
        if tw.troop:tw.troop.RNG=0
    env.step({'card':0,'x':9.5,'y':10.5})
    for _ in range(44):env.step(4)
    knight=next(t for t in p.troops if t.name=='Golden Knight');knight.spd=0
    air=target(g,'baby_dragon','red',knight.x+2,knight.y)
    ground=target(g,'giant','red',knight.x+3,knight.y+2);hp=air.hp;p.elixir=10
    assert g.activate_ability('blue',knight)==(True,'ok')
    for _ in range(30):
        obs,*_=env.step(4);assert env.observation_space.contains(obs)
    assert air.hp==hp and air not in knight.ability.hit and ground in knight.ability.hit
    env.close()


def t_dash_keeps_ground_buildings_eligible():
    g,knight=setup();building=target(g,'cannon','red',knight.x,knight.y+3)
    before=building.hp;g.players['blue'].elixir=10;assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.4)
    assert building in knight.ability.hit and building.hp<before-knight.ability.dd
