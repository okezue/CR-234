import math

import pytest

from sim.cards import create
from sim.fx import Component,EvoTesla
from sim.game import Game
from sim.units import Status,has
from tests.util import Dummy,quiet


def setup(team='blue',level=11):
    g=quiet(Game());tesla=create('tesla',level,team,9,10,evolved=True);g.deploy(team,tesla)
    pulse=next(c for c in tesla.components if isinstance(c,EvoTesla))
    pulse.on_tick(tesla,g)
    assert pulse.done
    return g,tesla,pulse


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('level',(11,16))
@pytest.mark.parametrize('cause',('damage','expiry'))
def t_evolved_tesla_death_has_no_damage_or_stun(team,level,cause):
    g,tesla,pulse=setup(team,level);enemy=g._opp(team)
    ground=Dummy(enemy,10,10,hp=5000,spd=0,dmg=0)
    air=Dummy(enemy,9,12,hp=5000,spd=0,dmg=0);air.transport='Air'
    shield=create('guards',level,enemy,9,11)[0]
    building=create('cannon',level,enemy,11,12)
    ally=Dummy(team,8,10,hp=5000,spd=0,dmg=0)
    targets=(ground,air,shield,building,ally)
    for t in targets:g.deploy(t.team,t)
    before=[(t.hp,getattr(t,'shield_hp',0),list(t.statuses)) for t in targets]
    if cause=='damage':tesla.take_damage(tesla.hp)
    else:
        tesla.hp=tesla.decay*g.DT/2
        g.players[enemy].troops=[];g._proc_troops();g.players[enemy].troops=list(targets[:-1])
        assert not tesla.alive
    g._proc_deaths();g._proc_deaths()
    assert tesla not in g.players[team].troops and not tesla.alive
    assert before==[(t.hp,getattr(t,'shield_hp',0),list(t.statuses)) for t in targets]
    assert all(not has(t,'stun') for t in targets) and pulse.done
    assert all(p.crowns==0 for p in g.players.values())


@pytest.mark.parametrize('team',('blue','red'))
def t_tesla_full_tick_expiry_with_enemy_present(team):
    g,tesla,_=setup(team);target=Dummy(g._opp(team),10,10,hp=5000,spd=0,dmg=0)
    g.deploy(target.team,target);tesla.hp=tesla.decay*g.DT/2
    g.tick()
    assert not tesla.alive and tesla not in g.players[team].troops
    assert target.hp==5000 and not has(target,'stun')


@pytest.mark.parametrize('team',('blue','red'))
def t_tesla_destroyed_during_deployment_does_not_pulse(team):
    g=quiet(Game());tesla=create('tesla',11,team,9,10,evolved=True);g._place(team,tesla,1)
    pulse=next(c for c in tesla.components if isinstance(c,EvoTesla))
    target=Dummy(g._opp(team),10,10,hp=5000,spd=0,dmg=0);g.deploy(target.team,target)
    tesla.hp=1;create('arrows',11,target.team,9,10).apply(g);g.tick()
    assert not tesla.alive and tesla not in g.players[team].troops and not pulse.done
    assert target.hp==5000 and not has(target,'stun')


def t_tesla_death_preserves_existing_status_and_crown_state():
    g,tesla,_=setup();target=Dummy('red',10,10,hp=5000,spd=0,dmg=0)
    stun=Status('stun',0.2);target.statuses=[stun];g.deploy('red',target)
    tower=g.arena.get_tower('red','king');tesla.x,tesla.y=tower.cx,tower.cy
    target.x,target.y=tower.cx-1,tower.cy;hp=tower.hp
    tesla.take_damage(tesla.hp);g._proc_deaths()
    assert target.hp==5000 and target.statuses==[stun] and stun.dur==0.2
    assert tower.hp==hp and not tower.active and all(p.crowns==0 for p in g.players.values())


class DeathCount(Component):
    def __init__(self):self.n=0
    def on_death(self,tr,g):self.n+=1


def t_tesla_removal_keeps_other_death_hooks_and_first_pulse():
    g=quiet(Game());tesla=create('tesla',11,'blue',9,10,evolved=True);g.deploy('blue',tesla)
    target=Dummy('red',10,10,hp=5000,spd=0,dmg=0);g.deploy('red',target)
    pulse=next(c for c in tesla.components if isinstance(c,EvoTesla));counter=DeathCount();tesla.components.append(counter)
    pulse.on_tick(tesla,g);assert target.hp==5000-pulse.dmg and has(target,'stun')
    hp=target.hp;statuses=list(target.statuses)
    pulse.on_tick(tesla,g);assert target.hp==hp and target.statuses==statuses
    tesla.take_damage(tesla.hp);g._proc_deaths();g._proc_deaths()
    assert counter.n==1 and target.hp==hp and target.statuses==statuses


def t_tesla_killer_continues_without_death_stun():
    g,tesla,_=setup();tesla.hp=1
    knight=create('knight',11,'red',10,10);g.deploy('red',knight);hp=knight.hp
    g._do_attack(knight,tesla);g._proc_deaths();before=(knight.x,knight.y)
    g.tick()
    assert not tesla.alive and knight.hp==hp and not has(knight,'stun')
    assert math.dist(before,(knight.x,knight.y))>0
