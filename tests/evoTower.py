import math
import random

from sim import fx
from sim.cards import create
from sim.game import Game
from sim.spells import CloneSpell
from tests.util import Dummy, quiet


# Five tower-damage defects of t2125 (local/towerDamageRound/findings.md), one group of tests per part, all at level 11:
# p1 the evolved Knight keeps his 60% reduction while attacking (wiki Knight/Evolution: lost once he starts attacking; export Knight_EV1
#    buffWhenNotAttacking damageReduction 60);
# p2 the evolved Lumberjack's ghost hits Crown Towers for 256, unraged, and stays after its Rage (Supercell June 2026: Crown Tower Damage
#    256 -> 128; wiki Lumberjack/Evolution: ghost_crown_11 128, gone shortly after leaving the initial Rage, about 5.5 s);
# p3 the evolved Inferno Dragon ramps while flying and drops its stage on every new target (export InfernoDragon_EV1 IncrementAttackCount on
#    attack; wiki Inferno Dragon/Evolution: the stage is kept 7 s without a hit, the fourth stage after 20 s of damage);
# p4 the Lumberjack's death Rage deals nothing (export BarbarianRageDamage 70 at level 1, crownTowerDamagePercent -70; wiki 179 and 54);
# p5 the dash deals twice the hit, 388 and 490, where the record has 389 and 491 (export dashDamage 152 and 192 at level 1).


def bare():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


def dummy(g,team,x,y,hp=50000,air=False):
    d=Dummy(team,x,y,hp=hp,spd=0,dmg=0);d._settled=True
    if air:d.transport='Air'
    g.deploy(team,d);return d


def red_left(g):
    return next(t for t in g.arena.towers if t.team=='red' and t.ttype=='princess' and t.cx<9)


def taken(u,amount=100):
    hp=u.hp;u.take_damage(amount);d=hp-u.hp;u.hp=hp;return d


def t_p1_evo_knight_takes_full_tower_shots_once_he_hits():
    # alone against a princess tower (109 a shot): 43 a shot until his first hit, then 109 while he stands hitting it
    random.seed(42);g=Game();k=create('knight',11,'blue',3.5,14.5,evolved=True);g._place('blue',k,1.0);tw=red_left(g)
    hp,thp,first,shots=k.hp,tw.hp,None,[]
    while g.t<30 and k.alive and tw.alive:
        g.tick()
        if tw.hp<thp and first is None:first=g.t
        thp=tw.hp
        if k.hp<hp:shots.append((g.t,hp-k.hp));hp=k.hp
    before=[d for t,d in shots if t<=first];after=[d for t,d in shots if t>first]
    assert first is not None and set(before)=={43} and len(before)>=5,(first,before)
    # the last shot takes what is left of him
    assert set(after[:-1])=={109} and after[-1]<=109 and len(after)>=10,after
    assert not k.alive and tw.alive,(k.alive,tw.hp)


def t_p1_evo_knight_is_shielded_while_deploying():
    random.seed(1);g=bare();k=create('knight',11,'blue',9,10,evolved=True);g._place('blue',k,1.0);g.run(0.5)
    assert fx.has(k,'deploying') and taken(k)==40


def t_p1_evo_knight_shield_returns_when_his_target_dies():
    random.seed(1);g=bare();k=create('knight',11,'blue',9,10,evolved=True);g.deploy('blue',k);k._settled=True
    near=dummy(g,'red',9,11.5,hp=300);dummy(g,'red',9,20)
    g.tick()
    assert taken(k)==40
    while near.hp==300:g.tick()
    g.tick()
    assert near.alive and taken(k)==100
    while near.alive:g.tick()
    for _ in range(4):g.tick()
    assert k.tgt is not near and taken(k)==40


def lj_at_tower(g,evolved=True):
    lj=create('lumberjack',11,'blue',3.5,22.0,evolved=evolved);g.deploy('blue',lj);lj._settled=True;tw=red_left(g);hp=tw.hp
    while tw.hp==hp:g.tick()
    lj.take_damage(lj.hp);g.tick()
    return lj,tw


def ghosts(g):
    return [u for u in g.players['blue'].troops if u.name=='Lumberjack Ghost']


def t_p2_ghost_hits_crown_towers_for_128():
    random.seed(1);g=quiet(Game());_,tw=lj_at_tower(g);hits=[];hp=tw.hp
    for _ in range(80):
        g.tick()
        if tw.hp<hp:hits.append(hp-tw.hp);hp=tw.hp
    assert len(hits)>=5 and set(hits)=={128},hits


def t_p2_ghost_is_raged_unseen_and_gone_with_its_rage():
    random.seed(1);g=quiet(Game());_,tw=lj_at_tower(g);gh,=ghosts(g);t0=g.t;hits=[];hp=tw.hp
    while gh.alive:
        g.tick()
        assert not gh.alive or fx.hidden(gh)
        if tw.hp<hp:hits.append(g.t);hp=tw.hp
    assert 5.45<=g.t-t0<=5.55,g.t-t0
    gaps=[round(b-a,2) for a,b in zip(hits,hits[1:])]
    assert gaps and max(gaps)<=0.65,gaps


def t_p2_ghost_leaves_its_rage_and_disappears_1s_later():
    random.seed(1);g=bare();lj=create('lumberjack',11,'blue',9,10,evolved=True);g.deploy('blue',lj);lj._settled=True
    far=dummy(g,'red',9,19.5);lj.take_damage(lj.hp);g.tick();gh,=ghosts(g);out=None
    while gh.alive and g.t<8:
        g.tick()
        if out is None and math.hypot(gh.x-9,gh.y-10)>3:out=g.t
    # it walks toward the bridge, out of the 3-tile Rage, and is gone 1 s after its last tick inside
    assert out is not None and 0.9<=g.t-out<=1.0,(out,g.t)
    assert far.hp==50000


def t_p2_ghost_takes_no_damage():
    random.seed(1);g=bare();lj=create('lumberjack',11,'blue',9,10,evolved=True);g.deploy('blue',lj)
    lj.take_damage(lj.hp);g.tick();gh,=ghosts(g)
    gh.take_damage(5000);create('fireball',11,'red',gh.x,gh.y).apply(g);g.tick()
    assert gh.alive and gh.hp==1


def beam_hits(g,tw,n,until=40):
    hits=[];hp=tw.hp
    while len(hits)<n and g.t<until:
        g.tick()
        if tw.hp<hp:hits.append((g.t,hp-tw.hp));hp=tw.hp
    return hits


def t_p3_evo_inferno_first_tower_beam_after_a_flight_ramps_from_stage_1():
    # recorded at level 16 (08PY89989G80, WWJv_aWCnT0 966.7-971.5 s, 10 fps labels read by t2124): a Dragon flying in from the bridge
    # without a beam before deals 5 ticks of 57, 5 of 192, then 674, one per 0.4 s; the base card the same
    for ev in (False,True):
        random.seed(42);g=quiet(Game());d=create('inferno_dragon',16,'blue',3.5,14.5,evolved=ev);g._place('blue',d,1.0)
        tw=red_left(g);tw.hp=tw.max_hp=50000;hits=beam_hits(g,tw,13)
        assert [x for _,x in hits]==[57]*5+[192]*5+[674]*3,(ev,hits)
        assert {round(b[0]-a[0],2) for a,b in zip(hits,hits[1:])}=={0.4}


def kill_then_tower(wait=0.0):
    # the dragon kills a 2000-hitpoint troop at its third stage, then finds the princess tower
    random.seed(42);g=quiet(Game());d=create('inferno_dragon',11,'blue',3.5,19.0,evolved=True);g._place('blue',d,0.0);d.statuses=[]
    v=dummy(g,'red',3.5,21.5,hp=2000);tw=red_left(g);hp=tw.hp
    while v.alive:g.tick()
    assert d.dmg==422 and tw.hp==hp
    return g,tw


def t_p3_evo_inferno_carries_its_stage_from_a_kill_to_the_tower():
    g,tw=kill_then_tower()
    assert [x for _,x in beam_hits(g,tw,3)]==[422,422,422]


def staged(wait):
    random.seed(1);g=bare();d=create('inferno_dragon',11,'blue',9,10,evolved=True);g.deploy('blue',d);d._settled=True
    v=dummy(g,'red',9,12,hp=2000)
    while v.alive:g.tick()
    t=g.t
    while g.t<t+wait:g.tick()
    w=dummy(g,'red',9,12);hp=w.hp
    while w.hp==hp:g.tick()
    return hp-w.hp


def t_p3_evo_inferno_keeps_its_stage_7s_without_a_beam():
    assert staged(6.5)==422
    assert staged(7.5)==35


def t_p3_evo_inferno_fourth_stage_after_20s_of_beam_not_of_flight():
    random.seed(42);g=quiet(Game());d=create('inferno_dragon',11,'blue',3.5,8.0,evolved=True);g._place('blue',d,1.0)
    tw=red_left(g);tw.hp=tw.max_hp=50000
    hits=beam_hits(g,tw,200,until=45);t0=hits[0][0];s4=[t for t,x in hits if x==844]
    # the beam clock starts when the tower comes in range, its first hit 0.4 s later
    assert t0>9 and s4 and 19.5<=s4[0]-t0<=20.1,(t0,s4[:1])


def rage_scene(evolved):
    random.seed(1);g=quiet(Game());lj=create('lumberjack',11,'blue',3.5,22.0,evolved=evolved);g.deploy('blue',lj);lj._settled=True
    tw=red_left(g);ds={'ground':dummy(g,'red',5.0,22.0),'air':dummy(g,'red',2.0,21.5,air=True),'far':dummy(g,'red',3.5,17.5),
                       'ally':dummy(g,'blue',4.5,21.0)}
    for d in ds.values():d.dmg=0
    hp={k:d.hp for k,d in ds.items()};thp=tw.hp
    lj.take_damage(lj.hp);g.tick()
    return {k:hp[k]-d.hp for k,d in ds.items()},thp-tw.hp


def t_p4_lumberjack_death_rage_hits_troops_179_and_towers_54():
    for ev in (False,True):
        dealt,tower=rage_scene(ev)
        assert dealt=={'ground':179,'air':179,'far':0,'ally':0} and tower==54,(ev,dealt,tower)


def dash_hit(name,clone=False):
    random.seed(1);g=bare();b=create(name,11,'blue',9,10);g.deploy('blue',b);b._settled=True
    if clone:
        CloneSpell('blue',9,10,{'radius':1.0}).apply(g);b.alive=False;g.players['blue'].troops.remove(b)
    d=dummy(g,'red',9,15);hp=d.hp
    while d.hp==hp and g.t<5:g.tick()
    return hp-d.hp


def t_p5_dashes_land_the_record_damage():
    assert dash_hit('bandit')==389
    assert dash_hit('boss_bandit')==491


def t_p5_a_cloned_bandit_dashes_for_389():
    assert dash_hit('bandit',clone=True)==389
