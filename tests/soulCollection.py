import pytest

from sim.cards import create
from sim.fx import SoulCollect
from sim.game import Game
from sim.units import Status


DECK=['skeleton_king','knight','archers','fireball','giant','musketeer','bomber','arrows']


def setup(team='blue'):
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0})
    for tower in g.arena.towers:
        tower.rng=0
        if tower.troop:tower.troop.RNG=0
    p=g.players[team];p.deck.hand=DECK[:4];p.elixir=10
    assert g.play_card(team,'skeleton_king',16.5,10.5 if team=='blue' else 21.5)==(True,'ok')
    g.tick();king=next(t for t in p.troops if t.name=='Skeleton King')
    sc=next(c for c in king.components if isinstance(c,SoulCollect))
    return g,king,sc


def deploy(g,card,team):
    t=create(card,11,team,2.5,10.5 if team=='blue' else 21.5)
    if isinstance(t,list):t=t[0]
    g.deploy(team,t)
    return t


@pytest.mark.parametrize('owner',('blue','red'))
@pytest.mark.parametrize('victim',('blue','red'))
def t_souls_include_both_teams_without_prior_snapshot(owner,victim):
    g,king,sc=setup(owner);t=deploy(g,'knight',victim)
    assert any(s.kind=='deploying' for s in king.statuses)
    t.take_damage(t.hp);g._proc_deaths()
    assert sc.souls==1
    g._proc_deaths();g.run(0.1);assert sc.souls==1


@pytest.mark.parametrize('card',('cannon','elixir_collector','cannon_cart','golem','lava_hound','elixir_golem','battle_ram'))
def t_buildings_and_intermediate_bodies_do_not_supply_souls(card):
    g,_,sc=setup();g.run(1.1);t=deploy(g,card,'red');g.tick()
    t.take_damage(t.hp);g._proc_deaths();g.tick()
    assert sc.souls==0


@pytest.mark.parametrize('card',('golem','lava_hound','elixir_golem','battle_ram','goblin_giant'))
def t_final_offspring_supply_souls_with_goblin_giant_body_exception(card):
    g,_,sc=setup();parent=deploy(g,card,'red');parent.take_damage(parent.hp);g._proc_deaths()
    assert sc.souls==(1 if card=='goblin_giant' else 0)
    while g.players['red'].troops:
        for t in list(g.players['red'].troops):t.take_damage(t.hp)
        g._proc_deaths()
    expected={'golem':2,'lava_hound':6,'elixir_golem':4,'battle_ram':2,'goblin_giant':3}[card]
    assert sc.souls==expected


def t_clones_do_not_supply_souls():
    g,_,sc=setup();g.run(1.1);original=deploy(g,'knight','red')
    create('clone',11,'red',original.x,original.y).apply(g)
    clone=next(t for t in g.players['red'].troops if getattr(t,'is_clone',False));g.tick()
    clone.take_damage(clone.hp);g._proc_deaths();g.tick();assert sc.souls==0
    original.take_damage(original.hp);g._proc_deaths();assert sc.souls==1


def t_cloned_collector_does_not_double_credit_original():
    g,king,sc=setup();g.run(2.2)
    create('clone',11,'blue',king.x,king.y).apply(g)
    clone=next(t for t in g.players['blue'].troops if getattr(t,'is_clone',False))
    assert next(c for c in clone.components if isinstance(c,SoulCollect)) is sc
    t=deploy(g,'knight','red');t.take_damage(t.hp);g._proc_deaths()
    assert sc.souls==1


def t_two_living_kings_collect_each_eligible_death():
    g,_,first=setup();other=deploy(g,'skeleton_king','red')
    second=next(c for c in other.components if isinstance(c,SoulCollect))
    t=deploy(g,'knight','red');t.take_damage(t.hp);g._proc_deaths()
    assert first.souls==second.souls==1


def t_soul_cap_reset_and_new_deaths_use_real_ability_submission():
    g,king,sc=setup();g.run(2.2)
    for _ in range(13):
        t=deploy(g,'knight','blue');t.take_damage(t.hp);g._proc_deaths()
    assert sc.cap==sc.souls==10
    g.players['blue'].elixir=10
    assert g.activate_ability('blue',king)==(True,'ok')
    assert sc.souls==10
    g.run(1.5)
    assert sc.souls==0 and king.ability.active
    t=deploy(g,'knight','red');t.take_damage(t.hp);g._proc_deaths();assert sc.souls==1


def t_ability_skeletons_do_not_refill_either_kings_souls():
    g,king,sc=setup();g.run(2.2);other=deploy(g,'skeleton_king','red')
    other.x,other.y=2.5,25.5;other.spd=0
    second=next(c for c in other.components if isinstance(c,SoulCollect))
    g.players['blue'].elixir=10;assert g.activate_ability('blue',king)==(True,'ok');g.run(3)
    skeletons=[t for t in g.players['blue'].troops if t is not king]
    assert len(skeletons)==6
    for t in skeletons:t.take_damage(t.hp)
    g._proc_deaths();g.tick();assert sc.souls==second.souls==0


@pytest.mark.parametrize('destroy_egg',(False,True))
def t_phoenix_hatching_is_removal_but_egg_destruction_is_death(destroy_egg):
    g,_,sc=setup();phoenix=deploy(g,'phoenix','red');phoenix.take_damage(phoenix.hp);g._proc_deaths()
    assert sc.souls==0
    egg=next(t for t in g.players['red'].troops if t.name=='Phoenix Egg')
    if destroy_egg:
        egg.take_damage(egg.hp);g._proc_deaths();assert sc.souls==1
    else:
        g.run(5);assert sc.souls==0
        reborn=next(t for t in g.players['red'].troops if t.name=='PhoenixNoRespawn')
        reborn.take_damage(reborn.hp);g._proc_deaths();assert sc.souls==1


def t_cascading_death_damage_counts_each_eligible_body_once():
    g,_,sc=setup();bomb=deploy(g,'giant_skeleton','red');victim=deploy(g,'knight','blue')
    victim.x,victim.y=bomb.x,bomb.y;victim.hp=1
    bomb.take_damage(bomb.hp);g._proc_deaths();assert sc.souls==1
    g.run(4);assert not victim.alive and sc.souls==2
    g._proc_deaths();assert sc.souls==2


@pytest.mark.parametrize('kind',('summoned','cloned','ordinary'))
@pytest.mark.parametrize('team',('blue','red'))
def t_cursed_skeleton_and_hog_preserve_soul_eligibility(kind,team):
    from sim.fx import CurseOnHit
    g,king,sc=setup(team);g.run(2.2);opp=g._opp(team)
    other=deploy(g,'skeleton_king',opp);other.x,other.y=16.5,27.5 if opp=='red' else 4.5;other.spd=0
    second=next(c for c in other.components if isinstance(c,SoulCollect))
    if kind=='summoned':
        g.players[team].elixir=10;assert g.activate_ability(team,king)==(True,'ok');g.run(2)
        victim=next(t for t in g.players[team].troops if t is not king)
    else:
        victim=deploy(g,'skeletons',team)
        if kind=='cloned':
            create('clone',11,team,victim.x,victim.y).apply(g)
            victim=next(t for t in g.players[team].troops if getattr(t,'is_clone',False))
    witch=deploy(g,'mother_witch',opp);curse=next(c for c in witch.components if isinstance(c,CurseOnHit))
    counts=sc.souls,second.souls
    g._do_attack(witch,victim)
    if victim.alive:victim.take_damage(victim.hp)
    g._proc_deaths();curse.on_tick(witch,g)
    hog=next(t for t in g.players[opp].troops if t.name=='Cursed Hog')
    hog.take_damage(hog.hp);g._proc_deaths()
    expected=2 if kind=='ordinary' else 0
    assert (sc.souls,second.souls)==(counts[0]+expected,counts[1]+expected)


@pytest.mark.parametrize('team',('blue','red'))
def t_cloned_phoenix_hatch_keeps_soul_exclusion(team):
    g,_,sc=setup('red' if team=='blue' else 'blue');g.run(2.2)
    original=deploy(g,'phoenix',team);create('clone',11,team,original.x,original.y).apply(g)
    clone=next(t for t in g.players[team].troops if getattr(t,'is_clone',False))
    clone.take_damage(clone.hp);g._proc_deaths();g.run(5)
    reborn=next(t for t in g.players[team].troops if t.name=='PhoenixNoRespawn')
    reborn.take_damage(reborn.hp);g._proc_deaths();assert sc.souls==0
    assert original.alive


def t_environment_observation_contains_soul_charged_summons():
    from sim.env import CREnv,GAME_FEAT,PLAYER_FEAT,N_TOWERS,TOWER_FEAT,TROOP_FEAT,MAX_TROOPS
    env=CREnv(blue_deck=DECK,decision_freq=1);env.reset(seed=4);g=env.game;p=g.players['blue']
    p.deck.hand=DECK[:4];p.elixir=10;p.drag_del=0
    for tower in g.arena.towers:
        tower.rng=0
        if tower.troop:tower.troop.RNG=0
    env.step({'card':0,'x':9.5,'y':10.5})
    for _ in range(44):env.step(4)
    king=next(t for t in p.troops if t.name=='Skeleton King')
    ally=deploy(g,'knight','blue');ally.take_damage(ally.hp);env.step(4)
    sc=next(c for c in king.components if isinstance(c,SoulCollect));assert sc.souls==1
    p.elixir=10;assert g.activate_ability('blue',king)==(True,'ok')
    for _ in range(85):obs,*_=env.step(4)
    assert len(p.troops)==8 and env.observation_space.contains(obs)
    start=GAME_FEAT+2*PLAYER_FEAT+N_TOWERS*TOWER_FEAT
    slots=obs[start:start+MAX_TROOPS*TROOP_FEAT].reshape(MAX_TROOPS,TROOP_FEAT)
    assert sum(row[2]>0 for row in slots)==8
    env.close()


@pytest.mark.parametrize('team',('blue','red'))
def t_reentrant_shield_burst_deaths_each_supply_one_soul(team):
    from sim.fx import Timer
    g,king,sc=setup(team);opp=g._opp(team);king.x,king.y=16.5,26.5
    ice=create('ice_golem',11,team,8,10);balloon=create('balloon',11,team,9,10)
    wizard=create('wizard',11,opp,9,11,evolved=True)
    for t in (ice,balloon,wizard):g.deploy(t.team,t)
    wizard.shield_hp=1;balloon.hp=1;ice.take_damage(ice.hp);g._proc_deaths()
    assert sc.souls==2 and not balloon.alive
    bombs=[s for s in g.spells if isinstance(s,Timer) and s.name==balloon.name]
    assert len(bombs)==1
    g._proc_deaths();assert sc.souls==2 and len(g.spells)==1


def t_surviving_cloned_king_does_not_collect_for_dead_original():
    g,king,sc=setup();g.run(2.2)
    create('clone',11,'blue',king.x,king.y).apply(g)
    clone=next(t for t in g.players['blue'].troops if getattr(t,'is_clone',False))
    king.take_damage(king.hp);g._proc_deaths();before=sc.souls
    victim=deploy(g,'knight','red');victim.take_damage(victim.hp);g._proc_deaths()
    assert clone.alive and sc.souls==before


def t_cannon_cart_becomes_building_without_soul_then_dies_without_soul():
    from sim.fx import Breakdown
    g,_,sc=setup();cart=deploy(g,'cannon_cart','red')
    breakdown=next(c for c in cart.components if isinstance(c,Breakdown))
    cart.take_damage(cart.max_hp*(1-breakdown.pct));g.tick()
    assert cart.alive and cart.is_building and sc.souls==0
    cart.take_damage(cart.hp);g._proc_deaths();assert sc.souls==0


def t_dead_king_does_not_collect_and_freeze_does_not_stop_collection():
    g,king,sc=setup();king.statuses.append(Status('freeze',3))
    t=deploy(g,'knight','red');t.take_damage(t.hp);g._proc_deaths();assert sc.souls==1
    king.take_damage(king.hp);t=deploy(g,'knight','red');t.take_damage(t.hp);g._proc_deaths()
    assert sc.souls==1
