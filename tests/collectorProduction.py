import pytest

from sim.cards import create
from sim.fx import ElixirProd
from sim.game import Game
from sim.units import Status


DECK=['elixir_collector','knight','archers','fireball','giant','musketeer','bomber','arrows']

def setup(team='blue'):
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0});p=g.players[team]
    p.deck.hand=['elixir_collector','knight','archers','fireball'];p.elixir=10
    assert g.play_card(team,'elixir_collector',9,10 if team=='blue' else 21)==(True,'ok')
    g.run(1.1);collector=next(t for t in p.troops if t.name=='Elixir Collector')
    return g,collector,next(c for c in collector.components if isinstance(c,ElixirProd))


@pytest.mark.parametrize('team',('blue','red'))
def t_collector_real_deployment_lasts93_and_produces7_plus_death(team):
    g,t,prod=setup(team);p=g.players[team]
    assert t.lifetime==93 and prod.interval==13 and prod.amount==1
    generated=0;living=0;death=0
    while t.alive and g.t<96:
        p.elixir=0;g.tick();produced=p.elixir-g.DT*g._erate()/g.EBASE;generated+=produced
        if t.alive:living+=produced
        else:death+=produced
        if g.t<90:assert t.alive
    assert not t.alive and g.t>93 and generated==pytest.approx(8)
    assert living==pytest.approx(7) and death==pytest.approx(1)
    balance=p.elixir;g._proc_deaths();assert p.elixir==balance


@pytest.mark.parametrize('team',('blue','red'))
def t_collector_destroyed_early_pays_one_death_elixir(team):
    g,t,prod=setup(team);p=g.players[team];p.elixir=0
    t.take_damage(t.hp);g._proc_deaths()
    assert p.elixir==1
    g._proc_deaths();assert p.elixir==1
    prod.on_death(t,g);assert p.elixir==1


def t_collector_holds_ready_production_until_player_spends():
    g,t,prod=setup();p=g.players['blue'];p.elixir=10
    g.run(14)
    assert prod.timer==0 and p.elixir==10
    g.run(10);assert prod.timer==0 and p.elixir==10
    p.deck.hand=['knight','archers','fireball','arrows']
    assert g.play_card('blue','knight',3,10)==(True,'ok')
    before=p.elixir;g.tick()
    assert p.elixir==pytest.approx(before+1+g.DT/g.EBASE)
    assert prod.timer==prod.interval
    previous=p.elixir;g.tick();assert p.elixir==pytest.approx(previous+g.DT/g.EBASE)


def t_ready_stored_elixir_is_delivered_even_while_frozen():
    g,t,prod=setup();p=g.players['blue'];p.elixir=10;g.run(14)
    assert prod.timer==0
    create('freeze',11,'red',t.x,t.y).apply(g)
    p.elixir=5;g.tick()
    assert p.elixir==pytest.approx(6+g.DT/g.EBASE)
    assert prod.timer==prod.interval
    g.tick();assert prod.timer==prod.interval


def t_death_payout_does_not_exceed_player_capacity():
    g,t,_=setup();p=g.players['blue'];p.elixir=9.8
    t.take_damage(t.hp);g._proc_deaths()
    assert p.elixir==10


def t_unfinished_frozen_collector_production_stays_paused():
    g,t,prod=setup();remaining=prod.timer;t.statuses=[Status('freeze',3)]
    g.run(1);assert prod.timer==remaining


@pytest.mark.parametrize('frozen',(False,True))
def t_collector_death_at_capacity_does_not_leave_later_credit(frozen):
    g,t,prod=setup();p=g.players['blue'];p.elixir=10
    if frozen:t.statuses=[Status('freeze',10)]
    t.take_damage(t.hp);g._proc_deaths();assert p.elixir==10
    p.elixir=0;g.tick();assert p.elixir==pytest.approx(g.DT/g.EBASE)
    assert prod.dead


def t_frozen_collector_death_with_space_pays_exactly_one():
    g,t,prod=setup();p=g.players['blue'];p.elixir=0;t.statuses=[Status('freeze',10)]
    t.take_damage(t.hp);g._proc_deaths();assert p.elixir==1
    prod.on_death(t,g);assert p.elixir==1


def t_ready_stored_production_is_not_a_second_death_payout():
    g,t,prod=setup();p=g.players['blue'];p.elixir=10;g.run(14)
    assert prod.timer==0
    p.elixir=0;t.take_damage(t.hp);g._proc_deaths()
    assert p.elixir==1
    g.tick();assert p.elixir==pytest.approx(1+g.DT/g.EBASE)


def t_ready_production_and_death_are_distinct_once_only_credits():
    g,t,prod=setup();p=g.players['blue'];p.elixir=0
    prod.timer=g.DT/2;t.hp=t.decay*g.DT/2
    g.tick()
    assert not t.alive and p.elixir==pytest.approx(2+g.DT/g.EBASE)
    before=p.elixir;g._proc_deaths();assert p.elixir==before


def t_collector_patch_is_part_of_rebuild_inputs():
    import json
    from pathlib import Path
    patch=json.loads((Path(__file__).resolve().parents[1]/'data/patches/2026-09-16.json').read_text())
    assert any(r['card']=='elixir_collector' and r['path']=='lifetime' and r['value']==93 for r in patch)
    assert any(r['card']=='elixir_collector' and r['path']=='skills.produceElixir.deathAmount' and r['value']==1 for r in patch)
    from data.build import set_path
    from sim.cards import card,load
    assert {'name':'data/patches/2026-09-16.json','tag':'patch:2026-09-16'} in load()['meta']['sources']
    config={'lifetime':65,'skills':{'produceElixir':{'interval':13,'amount':1}},'src':{}}
    for change in patch:set_path(config,change['path'],change['value'],'patch:2026-09-16')
    assert config['lifetime']==card('elixir_collector')['lifetime']
    assert config['skills']['produceElixir']==card('elixir_collector')['skills']['produceElixir']
