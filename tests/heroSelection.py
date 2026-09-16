import pytest

from sim.cards import create
from sim.game import Game
from sim import replay as R


@pytest.mark.parametrize('team',('blue','red'))
def t_distinct_hero_cards_can_activate_without_changing_other_slot(team):
    deck=['golden_knight','bowler','knight','archers','fireball','giant','goblins','arrows']
    g=Game(p1={'deck':deck,'heroes':['bowler'],'drag_del':0},p2={'deck':deck,'heroes':['bowler'],'drag_del':0})
    p=g.players[team]
    for name in ('golden_knight','bowler'):
        p.deck.hand=[name]+[c for c in deck if c!=name][:3];p.elixir=10
        assert g.play_card(team,name,9,10 if team=='blue' else 21)==(True,'ok')
        g.run(2)
    golden=next(t for t in p.troops if t.name=='Golden Knight');bowler=next(t for t in p.troops if t.name=='Bowler')
    for t in (golden,bowler):t.spd=0;t.ability.cd=0
    p.elixir=10
    assert g.activate_ability(team,bowler)==(True,'ok')
    assert g.activate_ability(team,golden)==(True,'ok')
    assert {x.troop for x in g.pending_ab}=={golden,bowler}
    g.run(1.3)
    assert golden.ability.uses==bowler.ability.uses==0


@pytest.mark.parametrize('team',('blue','red'))
def t_latest_same_card_is_selected_and_older_copy_cannot_use_ability(team):
    g=Game();p=g.players[team]
    old=create('golden_knight',11,team,4,6);new=create('golden_knight',11,team,5,6)
    for t in (old,new):g.deploy(team,t);t.ability.cd=0
    p.elixir=10
    assert R._ability_troop(g,team,'golden_knight') is new
    assert g.activate_ability(team,old)==(False,'not active champion')
    assert g.activate_ability(team,new)==(True,'ok')


@pytest.mark.parametrize('name,hero',[('bowler',True),('golden_knight',False)])
def t_distinct_hero_cast_refund_does_not_require_default_slot(name,hero):
    g=Game(p1={'ability_del':0,'ability_std':0});p=g.players['blue']
    first=create('golden_knight',11,'blue',4,6);second=create(name,11,'blue',5,6,hero=hero)
    for t in (first,second):g.deploy('blue',t);t.ability.cd=0
    p.elixir=5;assert g.activate_ability('blue',second)==(True,'ok')
    g.tick();assert second.ability.casting and p.active_champ is first
    hp=p.elixir;second.take_damage(second.hp);g._proc_deaths()
    assert p.elixir==min(10,hp+second.ability.cost) and not second.ability.casting
    assert p.active_champ is first
    balance=p.elixir;g._proc_deaths();assert p.elixir==balance


def t_default_ability_targets_latest_copy_of_default_card():
    g=Game();p=g.players['blue']
    older=create('golden_knight',11,'blue',9,10);latest=create('golden_knight',11,'blue',10,10)
    for t in (older,latest):g.deploy('blue',t);t.ability.cd=0
    p.elixir=10
    assert g.activate_ability('blue')==(True,'ok')
    assert g.pending_ab[-1].troop is latest and older.ability.can_use()


@pytest.mark.parametrize('other_hero',(False,True))
def t_recorded_banner_survives_dead_default_pointer_and_other_hero(other_hero):
    g=Game(p1={'ability_del':0,'ability_std':0});p=g.players['blue']
    goblins=create('goblins',11,'blue',9,10,hero=True)
    for t in goblins:g.deploy('blue',t)
    if other_hero:g.deploy('blue',create('golden_knight',11,'blue',10,10))
    for t in goblins:t.take_damage(t.hp)
    g._proc_deaths();banner=goblins[0].ability
    assert banner in p.pending_abilities and banner.can_use()
    default=p.active_champ;p.elixir=5
    assert R.submit_recorded_ability(g,'blue','goblins')==(True,'ok')
    assert p.active_champ is default and g.pending_ab[-1].ability is banner
    g.tick()
    assert len([t for t in p.troops if t.name==banner.base_cfg['name']])==banner.spawn_cnt and banner.uses==0


def t_banner_entry_point_cannot_charge_or_spawn_for_opponent():
    g=Game(p1={'ability_del':0,'ability_std':0});goblins=create('goblins',11,'blue',9,10,hero=True)
    for t in goblins:g.deploy('blue',t);t.take_damage(t.hp)
    g._proc_deaths();ab=goblins[0].ability
    g.players['red'].elixir=5
    assert not g.activate_banner('red',ab)[0] and g.players['red'].elixir==5
    g.players['blue'].elixir=5
    assert g.activate_banner('blue',ab)==(True,'ok') and g.pending_ab[0].troop is None
    g.tick()
    assert len(g.players['blue'].troops)==ab.spawn_cnt and not g.players['red'].troops


def t_explicit_foreign_dead_or_unregistered_ability_cannot_execute():
    g=Game();p=g.players['blue'];p.elixir=10
    own=create('golden_knight',11,'blue',4,6);g.deploy('blue',own);own.ability.cd=0
    for team,deploy,alive in [('red',True,True),('blue',False,True),('blue',True,False)]:
        t=create('archer_queen',11,team,5,6);t.ability.cd=0
        if deploy:g.deploy(team,t)
        t.alive=alive
        before=p.elixir;assert not g.activate_ability('blue',t)[0] and p.elixir==before
