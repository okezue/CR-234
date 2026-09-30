import pytest

from sim.cards import card, create
from sim.fx import RampUp
from sim.game import Game
from tests.util import Dummy


# level 11 tiers (wiki stat tables; the Mighty Miner's first tier follows the export) and the stage time: two seconds for all three
# (RoyaleAPI's extraction of the client data: InfernoDragon and InfernoTower variable_damage_time 2000 ms; wiki Mighty Miner 8/1/2025)
TIERS={'inferno_dragon':[35,120,422],'inferno_tower':[43,158,847],'mighty_miner':[43,204,409]}
STAGE=2.0


def beam(name,team,seconds,evolved=False,target='troop'):
    g=Game()
    for tw in g.arena.towers:
        tw.rng=0
        if tw.troop:tw.troop.RNG=0
    opp=g._opp(team);sign=1 if team=='blue' else -1
    if target=='tower':
        v=next(t for t in g.arena.towers if t.team==opp and t.ttype=='princess');v.hp=v.max_hp=100000;x,y=v.cx,v.cy-sign*2.5
    else:
        # dead towers block nothing; both bodies stay off the river rows
        for tw in g.arena.towers:tw.alive=False
        x,y=9,6 if team=='blue' else 26
    tr=create(name,11,team,x,y,evolved=evolved);tr.spd=0;g.deploy(team,tr)
    if target!='tower':v=Dummy(opp,9,y+sign*(tr.rng+0.5),hp=500000,spd=0,dmg=0);g.deploy(opp,v)
    last=v.hp;hits=[];lock=None
    for _ in range(int(round(seconds/g.DT))):
        g.tick()
        if lock is None and tr.tgt is v:lock=g.t
        if v.hp!=last:hits.append((round(g.t-lock,2),last-v.hp));last=v.hp
    return tr,hits


def onsets(hits,tiers):
    # the time after the lock of the first hit at each tier
    return [next(at for at,d in hits if d==t) for t in tiers]


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('name,target',(('inferno_dragon','tower'),('inferno_dragon','troop'),('inferno_tower','troop'),('mighty_miner','tower')))
def t_beam_climbs_one_stage_every_two_seconds(name,target,team):
    tr,hits=beam(name,team,6.0,target=target);tiers=TIERS[name]
    assert [d for _,d in hits]==sorted(d for _,d in hits) and {d for _,d in hits}==set(tiers)
    s1,s2,s3=onsets(hits,tiers);hs=tr.hspd
    # each stage opens at its two-second mark, on the next swing
    assert s2-STAGE>=-1e-6 and s2-STAGE<hs+1e-6
    assert s3-2*STAGE>=-1e-6 and s3-2*STAGE<hs+1e-6
    assert all(at<STAGE for at,d in hits if d==tiers[0]) and all(STAGE<=at<2*STAGE for at,d in hits if d==tiers[1])
    assert sum(d==tiers[0] for _,d in hits)==sum(d==tiers[1] for _,d in hits)==round(STAGE/hs)


@pytest.mark.parametrize('team',('blue','red'))
def t_evolved_dragon_keeps_the_base_stage_time_and_its_fourth_stage(team):
    tr,hits=beam('inferno_dragon',team,21.0,evolved=True,target='tower')
    tiers=TIERS['inferno_dragon']+[card('inferno_dragon')['evo']['skills']['rampingDamage']['damageTiers'][3][10]]
    s1,s2,s3,s4=onsets(hits,tiers)
    assert STAGE<=s2<STAGE+tr.hspd+1e-6 and 2*STAGE<=s3<2*STAGE+tr.hspd+1e-6 and 19.5<s4<=20.5
    assert all(d==tiers[2] for at,d in hits if 2*STAGE+tr.hspd<=at<19.5)


def t_records_carry_the_two_second_stage_time():
    db={k:card(k) for k in TIERS}
    assert db['inferno_dragon']['skills']['rampingDamage']['rampInterval']==STAGE
    assert db['inferno_dragon']['evo']['skills']['rampingDamage']['rampInterval']==STAGE
    assert db['inferno_tower']['skills']['rampingDamage']['rampInterval']==STAGE
    assert db['mighty_miner']['skills']['ability']['skills']['rampingDamage']['rampInterval']==STAGE
    for k,t in TIERS.items():
        ramp=next(c for c in create(k,11,'blue',9,10).components if isinstance(c,RampUp))
        assert ramp.durations==[STAGE,STAGE] and ramp.stages==t
