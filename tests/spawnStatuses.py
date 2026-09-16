import pytest

from sim.cards import create
from sim.fx import SpawnTimer,ElixirProd,RiderAttack
from sim.game import Game
from sim.units import Status


def game_with(name,team='blue'):
    g=Game()
    for t in g.arena.towers:t.alive=False
    tr=create(name,11,team,9,10);tr.spd=0;g.deploy(team,tr)
    return g,tr


@pytest.mark.parametrize('name',('witch','night_witch','tombstone','elixir_collector'))
@pytest.mark.parametrize('effect',('freeze','stun'))
def t_paused_spawner_does_not_count_down_or_produce(name,effect):
    g,tr=game_with(name);component=next(c for c in tr.components if isinstance(c,(SpawnTimer,ElixirProd)))
    tr.statuses=[Status(effect,2)];before=component.timer
    for _ in range(10):g.tick()
    assert component.timer==before
    assert g.players['blue'].troops==[tr]
    tr.statuses=[];g.tick();assert component.timer!=before


@pytest.mark.parametrize('name',('witch','night_witch','tombstone','elixir_collector'))
@pytest.mark.parametrize('status,value,rate',(('slow',.7,.7),('rage',.35,1.35),('mslow',.2,1)))
def t_spawn_clock_uses_spawn_speed_not_movement_only_slow(name,status,value,rate):
    g,tr=game_with(name);component=next(c for c in tr.components if isinstance(c,(SpawnTimer,ElixirProd)))
    component.timer=5;tr.statuses=[Status(status,10,value)]
    g.tick()
    assert component.timer==pytest.approx(5-g.DT*rate)


def t_real_freeze_pauses_witch_and_resumes_without_resetting_timer():
    g,tr=game_with('witch');clock=next(c for c in tr.components if isinstance(c,SpawnTimer))
    g.run(.4);remaining=clock.timer
    create('freeze',11,'red',9,10).apply(g)
    while any(s.kind=='freeze' for s in tr.statuses):
        before=clock.timer;g.tick()
        if any(s.kind=='freeze' for s in tr.statuses):assert clock.timer==before
    assert not any(t is not tr for t in g.players['blue'].troops)
    assert clock.timer<=remaining
    g.run(remaining+.1)
    assert len([t for t in g.players['blue'].troops if t is not tr])==4


def t_building_lifetime_continues_during_paused_production():
    g,tr=game_with('elixir_collector');clock=next(c for c in tr.components if isinstance(c,ElixirProd))
    before=tr.hp;timer=clock.timer;tr.statuses=[Status('freeze',10)]
    g.run(1)
    assert tr.hp==pytest.approx(before-tr.decay*1) and clock.timer==timer


def t_collector_produces_once_after_pause_with_normal_elixir_generation():
    g,tr=game_with('elixir_collector');clock=next(c for c in tr.components if isinstance(c,ElixirProd))
    g.players['blue'].elixir=0;clock.timer=.1;tr.statuses=[Status('stun',.2)]
    g.run(.15);assert g.players['blue'].elixir==pytest.approx(.15/g.EBASE)
    g.run(.2);assert g.players['blue'].elixir==pytest.approx(1+.35/g.EBASE)


def t_ram_rider_snare_does_not_slow_attack_or_spawn_clock():
    g,tr=game_with('witch');rider=create('ram_rider',11,'red',9,14);rider.spd=0;g.deploy('red',rider)
    shot=next(c for c in rider.components if isinstance(c,RiderAttack));shot.cd=0;shot.on_tick(rider,g)
    halt,attack_rate,movement_rate=g._status_mods(tr)
    assert not halt and attack_rate==1 and movement_rate<1
    clock=next(c for c in tr.components if isinstance(c,SpawnTimer));before=clock.timer;g.tick()
    assert clock.timer==pytest.approx(before-g.DT)


@pytest.mark.parametrize('existing',('mslow','slow','none','snare'))
def t_ram_rider_deprioritizes_own_snare_not_other_slows(existing):
    g=Game()
    for t in g.arena.towers:t.alive=False
    rider=create('ram_rider',11,'blue',9,10);g.deploy('blue',rider)
    near=create('knight',11,'red',9,12);far=create('knight',11,'red',9,14)
    g.deploy('red',near);g.deploy('red',far)
    if existing!='none':near.statuses=[Status(existing,5,.3)]
    attack=next(c for c in rider.components if isinstance(c,RiderAttack));attack.cd=0
    attack.on_tick(rider,g)
    assert attack.tgt is (far if existing=='snare' else near)
    victim=attack.tgt;halt,rate,movement=g._status_mods(victim)
    assert not halt and rate==(.3 if existing=='slow' else 1) and movement<1


def t_staggered_wave_already_released_keeps_existing_delay():
    g,tr=game_with('tombstone');g.tick()
    assert len(g.players['blue'].troops)==2
    tr.statuses=[Status('freeze',3)];g.run(.55)
    assert len(g.players['blue'].troops)==3
