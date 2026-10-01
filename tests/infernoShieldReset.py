import pytest

from sim.cards import create
from sim.game import Game


# The Inferno Tower's and Inferno Dragon's "damage would now reset after breaking a shield" (wiki Inferno Tower, Inferno Dragon, Dark
# Prince, Guards and Cannon Cart histories, 12/12/2017); the Guards strategy adds the Mighty Miner ("These three cards will reset their
# charge after they destroy a shield"). The evolved dragon keeps its charge: "Guards, Royal Recruits, or Dark Prince are also unable to
# reset its charge damage when their shields are down" (wiki Inferno Dragon/Evolution). Level 11 tiers and the 2 s stage as in
# tests/rampStageTime.py.
TIERS={'inferno_dragon':[35,120,422],'inferno_tower':[43,158,847],'mighty_miner':[43,204,409]}
STAGE=2.0


def beam(name,seconds,evolved=False,shield=True,strip_at=None):
    # a still Dark Prince that neither moves nor attacks stands just inside the attacker's reach; returns the lock time and the hits as
    # (time, loss of shield plus hitpoints, shield left); strip_at removes the shield by another source at that time
    g=Game()
    for tw in g.arena.towers:tw.alive=False
    tr=create(name,11,'blue',9,6,evolved=evolved);tr.spd=0;g.deploy('blue',tr)
    v=create('dark_prince',11,'red',9,6+tr.rng+0.5);v.spd=0;v.dmg=0;v.targets=[];v.hp=v.max_hp=50000
    if not shield:v.shield_hp=v.max_shield_hp=0
    g.deploy('red',v)
    last=v.hp+v.shield_hp;hits=[];lock=None
    for _ in range(int(round(seconds/g.DT))):
        g.tick()
        if strip_at is not None and lock is not None and g.t-lock>=strip_at and v.shield_hp>0:v.shield_hp=0;last=v.hp
        if lock is None and tr.tgt is v:lock=g.t
        if v.hp+v.shield_hp!=last:hits.append((round(g.t,2),last-v.hp-v.shield_hp,v.shield_hp));last=v.hp+v.shield_hp
    return lock,hits


def broken(hits):
    i=next(i for i,h in enumerate(hits) if h[2]==0)
    return hits[i][0],hits[i+1:]


@pytest.mark.parametrize('name',sorted(TIERS))
def t_breaking_a_shield_restarts_the_ramp(name):
    t1,t2,t3=TIERS[name]
    lock,hits=beam(name,9)
    tb,after=broken(hits)
    assert tb-lock>=STAGE-0.05,f"{name}: the shield should break at the second stage, at {tb-lock:.2f} s after the lock"
    assert after[0][1]==t1,f"{name}: the first hit after the break should be the first tier {t1}, got {after[0][1]} ({after[:3]})"
    first2=next(t for t,d,_ in after if d==t2);first3=next(t for t,d,_ in after if d==t3)
    assert STAGE-0.05<=first2-tb<=STAGE+0.45,f"{name}: the second tier should return {STAGE} s after the break, came {first2-tb:.2f} s"
    assert 2*STAGE-0.05<=first3-tb<=2*STAGE+0.45,f"{name}: the third tier should return {2*STAGE} s after the break, came {first3-tb:.2f} s"


@pytest.mark.parametrize('name',sorted(TIERS))
def t_unbroken_or_stripped_shield_leaves_the_ramp_running(name):
    # without a shield, or with the shield taken off by something else between two hits, the damage never steps back
    for kw in ({'shield':False},{'strip_at':STAGE+0.2}):
        lock,hits=beam(name,6,**kw)
        ds=[d for _,d,_ in hits if d>0]
        assert ds==sorted(ds) and TIERS[name][2] in ds,f"{name} {kw}: the ramp should keep climbing, {ds}"


def t_evolved_inferno_dragon_keeps_its_stage_through_a_shield_break():
    lock,hits=beam('inferno_dragon',6,evolved=True)
    tb,after=broken(hits)
    assert after[0][1]==TIERS['inferno_dragon'][1],f"The evolved dragon should keep the second tier after the break, {after[:3]}"
