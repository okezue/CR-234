import json
from pathlib import Path

import pytest

from sim.cards import create
from sim.fx import CurseOnHit,EvoIceSpirit,Timer
from sim.game import Game
from sim.units import Troop

# A jumping spirit no longer counts as a troop; it becomes similar to a projectile (wiki Fire, Ice and Heal Spirit, Ice Spirit/Evolution).
# The evolved Ice Spirit's jump infuses its target with an Ice Blast: 3 s after the hit it deals the same damage and 1.1 s freeze within 2 tiles
# (wiki Ice Spirit/Evolution; export IceSpiritsAOE_EV1 hitSpeed 3000, radius 2000, buffTime 1100, damage base 43 like the jump).

DECK=['ice_spirit','heal_spirit','fire_spirit','knight','archers','fireball','giant','arrows']


def quiet():
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0})
    for tower in g.arena.towers:
        tower.rng=0
        if tower.troop:tower.troop.RNG=0
    return g


def target(g,team,x,y,hp=5000,spd=0):
    t=Troop(team,x,y,{'hp':hp,'dmg':0,'spd':spd,'hspd':1,'rng':0,'name':'Stationary control'})
    t._settled=True
    g.deploy(team,t)
    return t


def spirit(g,card,team,x,y,level=11,evolved=False):
    s=create(card,level,team,x,y,evolved=evolved)
    s._settled=True
    g.deploy(team,s)
    return s


def launch(g,s):
    while not g.projs and g.t<3:g.tick()
    assert len(g.projs)==1


def hits(g,unit):
    # (time, damage) of every hit the unit takes
    out=[];take=unit.take_damage

    def note(a):
        out.append((round(g.t,2),a))
        return take(a)
    unit.take_damage=note
    return out


SPIRITS=[('ice_spirit',False),('ice_spirit',True),('heal_spirit',False)]


@pytest.mark.parametrize('card,evolved',SPIRITS)
@pytest.mark.parametrize('team',('blue','red'))
def t_ice_and_heal_spirit_bodies_leave_the_field_at_launch(card,evolved,team):
    g=quiet();y,f=(10.5,1) if team=='blue' else (21.5,-1)
    s=spirit(g,card,team,9.5,y,evolved=evolved);t=target(g,g._opp(team),9.5,y+f*3.2)
    launch(g,s)
    assert not s.alive and s._self_destructed and t.hp==5000
    g._proc_deaths()
    assert s not in g.players[team].troops
    g.run(1)
    assert t.hp==5000-s.dmg and not g.projs


@pytest.mark.parametrize('card,evolved',SPIRITS)
def t_launched_spirit_offers_no_body_to_spells_or_attackers(card,evolved):
    g=quiet();s=spirit(g,card,'blue',9.5,10.5,evolved=evolved);t=target(g,'red',9.5,13.7)
    launch(g,s);g._proc_deaths()
    knight=create('knight',11,'red',s.x+0.5,s.y);knight._settled=True;g.deploy('red',knight)
    create('clone',11,'blue',s.x,s.y).apply(g)
    assert not any(x.name==s.name for x in g.players['blue'].troops)
    g.tick()
    assert knight.tgt is not s
    g.run(1)
    assert t.hp==5000-s.dmg


@pytest.mark.parametrize('card,evolved',SPIRITS)
def t_retired_spirit_cannot_launch_again(card,evolved):
    g=quiet();s=spirit(g,card,'blue',9.5,10.5,evolved=evolved);t=target(g,'red',9.5,13.7)
    launch(g,s)
    g._fire(s,t)
    assert len(g.projs)==1
    g.run(1)
    assert t.hp==5000-s.dmg


def t_retired_heal_spirit_still_damages_and_heals_on_impact():
    g=quiet();s=spirit(g,'heal_spirit','blue',9.5,10.5);t=target(g,'red',9.5,13.7);other=target(g,'red',10.5,13.7)
    ally=target(g,'blue',8.5,13.7,hp=1000);ally.hp=hp=400
    launch(g,s)
    assert not s.alive and t.hp==other.hp==5000 and ally.hp==hp
    g.run(1)
    assert t.hp==other.hp==5000-s.dmg
    assert ally.hp==hp+401==801


@pytest.mark.parametrize('card',('ice_spirit','heal_spirit'))
def t_cursed_spirit_launch_makes_no_hog(card):
    g=quiet();s=spirit(g,card,'blue',9.5,10.5);t=target(g,'red',9.5,13.7)
    witch=create('mother_witch',11,'red',16,26);g.deploy('red',witch)
    curse=next(c for c in witch.components if isinstance(c,CurseOnHit))
    g._do_attack(witch,s);assert s.alive and curse.marks
    g.run(2)
    assert not any(x.name=='Cursed Hog' for x in g.players['red'].troops)
    assert t.hp==5000-s.dmg


def t_evolved_ice_spirit_blasts_its_target_three_seconds_after_impact():
    g=quiet();s=spirit(g,'ice_spirit','blue',9.5,10.5,evolved=True);t=target(g,'red',9.5,13.7)
    got=hits(g,t);stuns=[]
    while g.t<6:
        g.tick()
        stuns+=[st for st in t.statuses if st.kind=='stun' and st not in stuns]
    assert len(got)==2,got
    (t1,d1),(t2,d2)=got
    assert d1==s.dmg==110 and d2==110
    assert 3.0-1e-6<=t2-t1<=3.0+0.05+1e-6
    assert len(stuns)==2


def t_ice_blast_follows_the_infused_target():
    g=quiet();s=spirit(g,'ice_spirit','blue',9.5,10.5,evolved=True);t=target(g,'red',9.5,13.7)
    while t.hp==5000 and g.t<3:g.tick()
    assert t.hp==5000-s.dmg
    start=(t.x,t.y);t.x+=6
    near=target(g,'red',t.x+1.4,t.y);left=target(g,'red',*start)
    g.run(3.2)
    assert t.hp==5000-2*s.dmg
    assert near.hp==5000-s.dmg
    assert left.hp==5000


def t_ice_blast_radius_is_two_tiles():
    g=quiet();s=spirit(g,'ice_spirit','blue',9.5,10.5,evolved=True);t=target(g,'red',9.5,13.7)
    while t.hp==5000 and g.t<3:g.tick()
    # placed after the jump, so only the blast can reach them; the edge counts the body's collision circle like every splash
    inside=target(g,'red',t.x+1.9,t.y);outside=target(g,'red',t.x+2.6,t.y)
    g.run(3.2)
    assert inside.hp==5000-s.dmg and outside.hp==5000


def t_ice_blast_freezes_for_the_jump_duration():
    g=quiet();s=spirit(g,'ice_spirit','blue',9.5,10.5,evolved=True);t=target(g,'red',9.5,13.7)
    while t.hp==5000 and g.t<3:g.tick()
    late=target(g,'red',t.x+1,t.y);got=hits(g,late)
    while not got and g.t<7:g.tick()
    st=[x for x in late.statuses if x.kind=='stun']
    assert len(st)==1 and abs(st[0].dur-s.stun_dur)<1e-9 and abs(s.stun_dur-1.1)<1e-9


def t_ice_blast_reaches_a_crown_tower():
    g=quiet();s=spirit(g,'ice_spirit','blue',3.5,20,evolved=True)
    tower=g.arena.get_tower('red','princess','left');got=hits(g,tower)
    g.run(6)
    assert [d for _,d in got]==[s.dmg,s.dmg]
    assert got[1][0]-got[0][0]>=3.0-1e-6


def t_ice_blast_fires_where_a_killed_target_fell():
    g=quiet();s=spirit(g,'ice_spirit','blue',9.5,10.5,evolved=True);t=target(g,'red',9.5,13.7,hp=50)
    while t.alive and g.t<3:g.tick()
    assert not t.alive
    fell=(t.x,t.y);late=target(g,'red',fell[0]+1,fell[1]);away=target(g,'red',fell[0]+6,fell[1])
    g.run(3.2)
    assert late.hp==5000-s.dmg and away.hp==5000


def t_evolved_ice_spirit_killed_before_its_jump_leaves_no_blast():
    g=quiet();s=spirit(g,'ice_spirit','blue',9.5,10.5,evolved=True);t=target(g,'red',9.5,13.7)
    s.take_damage(s.hp);g.run(5)
    assert not g.projs and t.hp==5000
    assert not any(isinstance(x,Timer) for x in g.spells)


def t_base_ice_spirit_has_no_blast():
    g=quiet();s=spirit(g,'ice_spirit','blue',9.5,10.5);t=target(g,'red',9.5,13.7)
    got=hits(g,t);g.run(6)
    assert [d for _,d in got]==[s.dmg]


@pytest.mark.parametrize('level',(11,16))
def t_evolution_parameters_come_from_the_sources(level):
    s=create('ice_spirit',level,'blue',9.5,10.5,evolved=True)
    e=next(c for c in s.components if isinstance(c,EvoIceSpirit))
    assert e.delay==3 and e.radius==2.0 and e.freeze==1.1
    assert e.dmg==s.dmg==(110 if level==11 else 175)


def t_card_data_ice_blast_damage_equals_jump_damage():
    c=json.loads((Path(__file__).resolve().parents[1]/'data'/'cards.json').read_text())['cards']['ice_spirit']
    assert c['evo']['skills']['poison']['damage']==c['evo']['stats']['damage']
