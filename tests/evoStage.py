import copy
import math

import pytest

from sim.cards import create,card
from sim.fx import RampUp,EvoInfernoDragon
from sim.game import Game
from sim.units import Status
from tests.util import Dummy


def setup(team='blue',level=11):
    g=Game()
    for tower in g.arena.towers:tower.alive=False
    tr=create('inferno_dragon',level,team,9,10,evolved=True);g.deploy(team,tr)
    target=Dummy(g._opp(team),9,12,hp=1000000,spd=0,dmg=0);g.deploy(target.team,target)
    ramp=next(c for c in tr.components if isinstance(c,RampUp))
    evo=next(c for c in tr.components if isinstance(c,EvoInfernoDragon))
    return g,tr,ramp,evo,target


def components_tick(g,tr):
    for c in tr.components:c.on_tick(tr,g)


def t_evolved_construction_separates_final_stage():
    source=copy.deepcopy(card('inferno_dragon'))
    for team in ('blue','red'):
        for level in (11,16):
            _,tr,ramp,evo,_=setup(team,level)
            expected=[row[level-1] for row in source['evo']['skills']['rampingDamage']['damageTiers']]
            assert sum(isinstance(c,RampUp) for c in tr.components)==1
            assert sum(isinstance(c,EvoInfernoDragon) for c in tr.components)==1
            assert tr.components.index(ramp)<tr.components.index(evo)
            assert len(ramp.stages)==3 and ramp.stages==expected[:3]
            assert tr.ramp_stages==ramp.stages
            assert ramp.durations==tr.ramp_durations==[1.5,1.5]
            assert (evo.s4_dmg,evo.s4_time,evo.retain)==(expected[-1],20.0,7.0)
            assert tr.ct_dmg==expected[0]
    assert card('inferno_dragon')==source


@pytest.mark.parametrize('team',('blue','red'))
def t_evolved_does_not_select_fourth_tier_at_base_ramp_end(team):
    g,tr,ramp,evo,target=setup(team);hp=target.hp;hits=[]
    for _ in range(399):
        g.tick()
        if target.hp!=hp:hits.append(hp-target.hp);hp=target.hp
        assert not evo.s4_active and tr.dmg<=ramp.stages[2]
    assert set(hits)==set(ramp.stages)
    assert tr.dmg==422 and evo.total_beam<evo.s4_time
    g.run(0.2)
    assert evo.s4_active and tr.dmg==844
    hp=target.hp;g.run(0.4)
    assert hp-target.hp==844 and tr.dmg==844


def t_final_stage_threshold_and_following_ticks():
    g,tr,ramp,evo,target=setup();tr.tgt=target;ramp.cur_tgt=target;ramp.elapsed=5
    evo.total_beam=math.nextafter(evo.s4_time-2*g.DT,math.inf)
    components_tick(g,tr)
    assert evo.total_beam<evo.s4_time and not evo.s4_active and tr.dmg==422
    components_tick(g,tr)
    assert evo.total_beam>=evo.s4_time and evo.s4_active and tr.dmg==844
    for _ in range(8):
        components_tick(g,tr)
        assert tr.dmg==844
        hp=target.hp;g._do_attack(tr,target)
        assert hp-target.hp==844


@pytest.mark.parametrize('status',('stun','freeze'))
@pytest.mark.parametrize('charged',(False,True))
def t_disabling_status_invalidates_final_stage(status,charged):
    g,tr,ramp,evo,target=setup();tr.tgt=target;ramp.cur_tgt=target;ramp.elapsed=5
    evo.total_beam=evo.s4_time if charged else evo.s4_time-g.DT
    evo.s4_active=charged;tr.dmg=844 if charged else 422
    tr.statuses.append(Status(status,1))
    for _ in range(4):
        components_tick(g,tr)
        assert tr.dmg==35 and ramp.elapsed==0 and evo.total_beam==0 and not evo.s4_active
    tr.statuses=[]
    for _ in range(10):components_tick(g,tr)
    assert not evo.s4_active and tr.dmg<844
    hp=target.hp;g._do_attack(tr,target)
    assert hp-target.hp<844


def t_last_stage_survives_lower_ramp_target_reset():
    g,tr,ramp,evo,target=setup();tr.tgt=target
    evo.total_beam=evo.s4_time;evo.s4_active=True
    components_tick(g,tr);assert tr.dmg==844
    other=Dummy('red',10,12,hp=1000000,spd=0,dmg=0);g.deploy('red',other)
    target.alive=False;tr.tgt=other
    components_tick(g,tr)
    assert ramp.cur_tgt is other and ramp.elapsed==0 and tr.dmg==844
    hp=other.hp;g._do_attack(tr,other)
    assert hp-other.hp==844


def t_final_stage_idle_expiry_and_new_deployment():
    g,tr,ramp,evo,target=setup();tr.tgt=None;evo.total_beam=evo.s4_time;evo.s4_active=True
    evo.idle_timer=evo.retain-2*g.DT
    components_tick(g,tr);assert evo.s4_active
    evo.idle_timer=evo.retain
    components_tick(g,tr)
    assert not evo.s4_active and evo.total_beam==0 and tr.dmg==35
    tr.tgt=target
    components_tick(g,tr)
    assert not evo.s4_active and tr.dmg<844
    tr.take_damage(tr.hp);g._proc_deaths()
    fresh=create('inferno_dragon',11,'blue',9,10,evolved=True)
    fresh_evo=next(c for c in fresh.components if isinstance(c,EvoInfernoDragon))
    assert fresh_evo.total_beam==0 and not fresh_evo.s4_active and fresh.dmg==35


def t_ordinary_ramps_and_crown_routing_stay_unchanged():
    for name,stages in (('inferno_dragon',[35,120,422]),('inferno_tower',[43,158,847]),('mighty_miner',[43,204,409])):
        tr=create(name,11,'blue',9,10)
        ramp=next(c for c in tr.components if isinstance(c,RampUp))
        assert ramp.stages==stages and ramp.durations==[1.5,1.5]
        assert tr.ct_dmg==stages[0]
    g,tr,_,evo,_=setup();tower=g.arena.get_tower('red','princess','left');tower.alive=True
    tr.dmg=evo.s4_dmg;hp=tower.hp;g._do_attack(tr,tower)
    assert hp-tower.hp==35
