import math

import pytest

from sim.cards import create
from sim.fx import Component,Enchanted,ShieldBurst
from sim.game import Game
from sim.units import Status
from tests.util import Dummy,quiet


def setup(team='blue',evolved=False,level=11):
    g=quiet(Game());tr=create('firecracker',level,team,9,10,evolved=evolved);g.deploy(team,tr)
    target=create('dark_prince',level,g._opp(team),9,14);g.deploy(target.team,target)
    return g,tr,target


def record_damage(target):
    calls=[];damage=target.take_damage
    def hit(amount):
        before=(target.hp,getattr(target,'shield_hp',0))
        damage(amount)
        calls.append((amount,before,(target.hp,getattr(target,'shield_hp',0))))
    target.take_damage=hit
    return calls


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('evolved',(False,True))
def t_firecracker_packets_break_shield_then_damage_body(team,evolved):
    g,tr,target=setup(team,evolved);calls=record_damage(target)
    assert (target.hp,target.shield_hp)==(1200,240) and tr.dmg==320
    g._fire(tr,target)
    assert not calls and len(g.projs)==1 and (tr.x,tr.y)==(9,9)
    for _ in range(7):g._proc_projs();assert not calls
    g._proc_projs()
    assert [c[0] for c in calls]==[64]*5
    assert [c[2][1] for c in calls]==[176,112,48,0,0]
    assert [c[2][0] for c in calls]==[1200,1200,1200,1200,1136]
    assert not g.projs and (tr.x,tr.y)==(9,9)


@pytest.mark.parametrize('shield,body_loss',((0,320),(1,256),(64,256),(65,192),(240,64),(320,0),(400,0)))
def t_only_shield_breaking_packet_discards_overflow(shield,body_loss):
    g,tr,target=setup();target.shield_hp=target.max_shield_hp=shield;hp=target.hp
    g._do_attack(tr,target)
    assert hp-target.hp==body_loss
    assert target.shield_hp==max(0,shield-320)


def t_level_scaled_packets_and_launch_snapshot():
    for level in (11,16):
        g,tr,target=setup(level=level);calls=record_damage(target);damage=tr.dmg;hp=target.hp;shield=target.shield_hp
        g._fire(tr,target)
        tr.dmg=9999;tr.ct_dmg=9998;tr.take_damage(tr.hp);g._proc_deaths()
        for _ in range(8):g._proc_projs()
        assert [c[0] for c in calls]==[damage/5]*5
        shield_packets=math.ceil(shield/(damage/5))
        assert hp-target.hp==(5-shield_packets)*damage/5


def t_packetization_preserves_unshielded_troop_building_and_tower_totals():
    for kind in ('troop','building','princess','king'):
        g,tr,_=setup()
        if kind=='troop':target=Dummy('red',9,14,hp=5000,spd=0)
        elif kind=='building':target=create('cannon',11,'red',9,14)
        else:target=g.arena.get_tower('red',kind,'left' if kind=='princess' else None)
        target.hp=5000;calls=record_damage(target)
        tr.ct_dmg=105
        g._do_attack(tr,target)
        expected=105 if kind in ('princess','king') else 320
        assert 5000-target.hp==expected and [c[0] for c in calls]==[expected/5]*5


class Hooks(Component):
    def __init__(self):self.pre=0;self.attacks=0;self.taken=0
    def pre_damage(self,tr,att,dmg,g):self.pre+=1;return dmg+1
    def on_attack(self,tr,tgt,g):self.attacks+=1
    def on_take_damage(self,tr,att,g):self.taken+=1


def t_hooks_and_existing_enchantment_run_once_per_volley():
    g,tr,target=setup();target.shield_hp=0;target.hp=5000
    attacker=Hooks();defender=Hooks();tr.components.append(attacker);target.components.append(defender)
    source=create('rune_giant',11,'blue',9,9);g.deploy('blue',source)
    enchant=Enchanted(100,3,source,5);enchant.n=2;tr.components.append(enchant)
    calls=record_damage(target)
    g._do_attack(tr,target)
    assert attacker.attacks==defender.pre==defender.taken==1
    assert enchant.n==3 and len(calls)==6
    assert calls[-1][0]==100 and [c[0] for c in calls[:5]]==[65,64,64,64,64]
    assert 5000-target.hp==421


def t_cloned_shielded_target_dies_once_and_secondary_fan_still_runs():
    g,tr,target=setup();target.hp=target.max_hp=target.shield_hp=target.max_shield_hp=1
    deaths=[]
    class Death(Component):
        def on_death(self,tr,g):deaths.append(tr)
    target.components=[Death()];calls=record_damage(target)
    behind=Dummy('red',9,16,hp=5000,spd=0,dmg=0);g.deploy('red',behind)
    g._do_attack(tr,target);g._proc_deaths();g._proc_deaths()
    assert not target.alive and len(calls)==2 and deaths==[target]
    assert behind.hp==5000-192


def t_shield_burst_cannot_cancel_already_released_shrapnel():
    g,tr,_=setup();g.players['red'].troops=[]
    target=create('wizard',11,'red',9,12,evolved=True);g.deploy('red',target)
    burst=next(c for c in target.components if isinstance(c,ShieldBurst))
    tr.hp=1;target.shield_hp=64;hp=target.hp;breaks=[]
    callback=target.on_shield_break
    def shield_break():breaks.append(target);callback()
    target.on_shield_break=shield_break
    g._fire(tr,target)
    for _ in range(4):g._proc_projs()
    assert burst.done and not tr.alive and target.hp==hp-256
    assert target.shield_hp==0 and breaks==[target]


def t_primary_packets_finish_crowns_once():
    for kind in ('princess','king'):
        g,tr,_=setup();tower=g.arena.get_tower('red',kind,'left' if kind=='princess' else None)
        tower.hp=100;calls=record_damage(tower)
        g._do_attack(tr,tower)
        assert len(calls)==2 and not tower.alive and tower.down
        assert g.players['blue'].crowns==(3 if kind=='king' else 1)
        g._tower_down(tower)
        assert g.players['blue'].crowns==(3 if kind=='king' else 1)
        if kind=='king':assert g.ended and g.winner=='blue'


def t_primary_shrapnel_does_not_split_other_attacks():
    for name in ('knight','musketeer','hunter','magic_archer','bowler'):
        g=Game();tr=create(name,11,'blue',9,10);g.deploy('blue',tr)
        target=Dummy('red',9,11,hp=50000,spd=0);g.deploy('red',target);calls=record_damage(target)
        g._do_attack(tr,target)
        if name=='hunter':assert calls and all(c[0]==tr.dmg for c in calls)
        else:assert len(calls)==1 and calls[0][0]==tr.dmg


@pytest.mark.parametrize('shield', (0,65,321))
def t_nondivisible_total_has_exact_death_and_shield_boundaries(shield):
    g,tr,target=setup();target.hp=321;target.shield_hp=shield;target.components=[Hooks()]
    breaks=[];target.on_shield_break=lambda:breaks.append(target)
    calls=record_damage(target);g._do_attack(tr,target)
    assert [c[0] for c in calls]==[65,64,64,64,64]
    if shield==0:assert target.hp==0 and not target.alive and not breaks
    elif shield==65:assert target.hp==65 and target.alive and target.shield_hp==0 and breaks==[target]
    else:assert target.hp==321 and target.alive and target.shield_hp==0 and breaks==[target]


class ReplaceDamage(Component):
    def __init__(self,amount):self.amount=amount
    def pre_damage(self,tr,att,dmg,g):return self.amount


@pytest.mark.parametrize('amount', (0,-1,0.5))
def t_nonpositive_or_fractional_transforms_keep_single_hit(amount):
    g,tr,target=setup();target.shield_hp=0;target._dmg_reduction=0.3
    target.components=[ReplaceDamage(amount)];calls=record_damage(target);hp=target.hp
    g._do_attack(tr,target)
    assert [c[0] for c in calls]==[amount] and hp-target.hp==1


def t_integer_packet_reduction_and_zero_sized_piece_handling():
    g,tr,target=setup();target.shield_hp=0;target._dmg_reduction=0.3;hp=target.hp
    calls=record_damage(target);g._do_attack(tr,target)
    assert [c[0] for c in calls]==[64]*5 and hp-target.hp==220
    target.components=[ReplaceDamage(3)];hp=target.hp;calls.clear();g._do_attack(tr,target)
    assert [c[0] for c in calls]==[1,1,1] and hp-target.hp==3


def t_tower_packet_override_is_snapshotted_at_launch():
    g,tr,_=setup();tower=g.arena.get_tower('red','princess','left');calls=record_damage(tower)
    hp=tower.hp;tr.ct_dmg=105;g._fire(tr,tower);tr.ct_dmg=500;tr.dmg=1000
    assert not calls
    for _ in range(40):g._proc_projs()
    assert [c[0] for c in calls]==[21]*5 and hp-tower.hp==105


def t_sparks_attribute_on_unrelated_component_does_not_split():
    class Other(Component):sparks=5
    g=Game();tr=create('musketeer',11,'blue',9,10);tr.components.append(Other())
    target=Dummy('red',9,14,hp=5000,spd=0);calls=record_damage(target)
    g._do_attack(tr,target)
    assert [c[0] for c in calls]==[tr.dmg]


def t_immune_target_remains_immune_to_each_packet():
    g,tr,target=setup();target.statuses.append(Status('invincible',2));hp=target.hp;shield=target.shield_hp
    g._do_attack(tr,target)
    assert (target.hp,target.shield_hp)==(hp,shield)
