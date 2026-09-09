import math

from sim.cards import create
from sim.fx import RampUp
from sim.game import Game
from sim.units import Status
from tests.util import Dummy


NAMES=('inferno_dragon','inferno_tower','mighty_miner')


def setup(name,team='blue',evolved=False):
    g=Game()
    for tower in g.arena.towers:tower.alive=False
    tr=create(name,11,team,9,10,evolved=evolved);tr.spd=0;g.deploy(team,tr)
    ramp=next(c for c in tr.components if isinstance(c,RampUp))
    target=Dummy(g._opp(team),9,10+tr.rng+2,hp=50000,spd=0,dmg=0);g.deploy(target.team,target)
    return g,tr,ramp,target


def first_hit(g,tr,target):
    hp=target.hp;start=g.t
    while target.hp==hp and g.t-start<1:g.tick()
    assert target.hp<hp
    return hp-target.hp,round(g.t-start,6)


def t_ordinary_ramp_cannot_precharge_out_of_range():
    for name in NAMES:
        for team in ('blue','red'):
            g,tr,ramp,target=setup(name,team);g.run(5)
            assert tr.tgt is target and target.hp==50000
            assert tr.dmg==ramp.stages[0] and ramp.elapsed==0 and ramp.cur_tgt is None
            target.y=10+tr.rng
            damage,delay=first_hit(g,tr,target)
            assert damage==ramp.stages[0] and delay==round(math.ceil(tr.fhspd/g.DT)*g.DT,6)


def t_ordinary_ramp_does_not_charge_without_target():
    for name in NAMES:
        g,tr,ramp,target=setup(name);g.players[target.team].troops=[];g.run(5)
        assert tr.tgt is None and ramp.cur_tgt is None and ramp.elapsed==0
        assert tr.dmg==ramp.stages[0]


def t_range_exit_resets_before_same_target_returns():
    for name in NAMES:
        g,tr,ramp,target=setup(name);target.y=10+tr.rng
        g.run(3.5);assert tr.dmg==ramp.stages[-1]
        target.y=10+tr.rng+2;hp=target.hp;g.run(1)
        assert target.hp==hp and ramp.elapsed==0 and ramp.cur_tgt is None and tr.dmg==ramp.stages[0]
        target.y=10+tr.rng
        damage,_=first_hit(g,tr,target)
        assert damage==ramp.stages[0]


def t_ramp_attack_range_uses_body_edges_and_minimum_range():
    for name in NAMES:
        g,tr,ramp,target=setup(name)
        target.y=math.nextafter(tr.y+tr.rng+tr.collision_r+target.collision_r,tr.y)
        tr.tgt=target
        ramp.on_tick(tr,g)
        assert ramp.cur_tgt is target and ramp.elapsed==0
        ramp.on_tick(tr,g)
        assert ramp.elapsed==g.DT
        target.y+=0.001;ramp.on_tick(tr,g)
        assert ramp.cur_tgt is None and ramp.elapsed==0
        tr.min_rng=1;target.y=tr.y+tr.collision_r+target.collision_r+0.5
        ramp.on_tick(tr,g)
        assert ramp.cur_tgt is None and ramp.elapsed==0
        target.y+=0.5;tr.min_rng=g._dist(tr,target);ramp.on_tick(tr,g)
        assert ramp.cur_tgt is target


def t_ramp_tower_range_uses_tower_edge_not_center():
    g,tr,ramp,_=setup('inferno_dragon')
    tower=g.arena.get_tower('red','princess','left');tower.alive=True
    tr.x=tower.cx;tr.y=tower.cy-tower.collision_r-tr.rng-tr.collision_r;tr.tgt=tower
    assert math.hypot(tr.x-tower.cx,tr.y-tower.cy)>tr.rng
    ramp.on_tick(tr,g);ramp.on_tick(tr,g)
    assert ramp.cur_tgt is tower and ramp.elapsed==g.DT
    tr.y-=0.01;ramp.on_tick(tr,g)
    assert ramp.cur_tgt is None and ramp.elapsed==0


def t_invalid_target_clears_ramp_without_changing_status_rules():
    for name in NAMES:
        for invalid in ('dead','invisible','burrowed','stun','freeze'):
            g,tr,ramp,target=setup(name);target.y=10+tr.rng;g.run(2)
            assert tr.dmg==ramp.stages[1]
            if invalid=='dead':target.take_damage(target.hp)
            elif invalid in ('invisible','burrowed'):target.statuses.append(Status(invalid,1))
            else:tr.statuses.append(Status(invalid,1))
            ramp.on_tick(tr,g)
            assert ramp.cur_tgt is None and ramp.elapsed==0 and tr.dmg==ramp.stages[0]


def t_uninterrupted_ramp_keeps_configured_timing():
    for name in NAMES:
        g,tr,ramp,target=setup(name);target.y=10+tr.rng;tr.tgt=target
        stage_times=list(ramp.durations);stages=list(ramp.stages);cooldown=tr.cd
        ramp.on_tick(tr,g);assert ramp.elapsed==0
        ticks=round(stage_times[0]/g.DT)
        for _ in range(ticks-1):ramp.on_tick(tr,g);assert tr.dmg==stages[0]
        ramp.on_tick(tr,g);assert tr.dmg==stages[1]
        for _ in range(round(stage_times[1]/g.DT)-1):ramp.on_tick(tr,g);assert tr.dmg==stages[1]
        ramp.on_tick(tr,g);assert tr.dmg==stages[-1]
        assert ramp.stages==stages and ramp.durations==stage_times and tr.cd==cooldown


def t_walking_inferno_starts_at_low_damage():
    g,tr,ramp,target=setup('inferno_dragon');tr.spd=1.2;target.y=22
    hp=target.hp;steps=0
    while target.hp==hp and g.t<10:
        if g._dist(tr,target)>tr.rng:
            assert tr.dmg==ramp.stages[0] and ramp.elapsed==0
        g.tick();steps+=1
    assert steps>60 and hp-target.hp==ramp.stages[0]


def t_evolution_retention_path_is_not_cleared_by_ordinary_gate():
    g,tr,ramp,target=setup('inferno_dragon',evolved=True);target.y=10+tr.rng;g.run(2)
    assert tr.dmg==ramp.stages[1]
    target.y=10+tr.rng+2;elapsed=ramp.elapsed;g.tick()
    assert ramp.cur_tgt is target and ramp.elapsed>elapsed and tr.dmg==ramp.stages[1]
