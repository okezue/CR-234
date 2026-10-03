import pytest

from sim.cards import create
from sim.fx import CurseOnHit,SoulCollect
from sim.game import Game
from sim.units import Troop

# A jumping spirit no longer counts as a troop; it becomes similar to a projectile (wiki Electro Spirit, Touchdown trivia), and the Electro
# Spirit "is immediately sacrificed when it attacks" (wiki strategy). Its jump still strikes the target and up to 8 more within 3 tiles of each
# other and stuns them for 0.5 s (wiki intro; export ElectroSpiritProjectile chainedHitCount 9, buffTime 500 ZapFreeze; chain range 3 tiles
# since the 2026-08-26 balance change, data/patches/2026-08-26.json).

DECK=['electro_spirit','knight','archers','fireball','giant','musketeer','bomber','arrows']


def quiet():
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0})
    for tower in g.arena.towers:
        tower.rng=0
        if tower.troop:tower.troop.RNG=0
    return g


def target(g,team,x,y,hp=5000):
    t=Troop(team,x,y,{'hp':hp,'dmg':0,'spd':0,'hspd':1,'rng':0,'name':'Stationary control'})
    t._settled=True
    g.deploy(team,t)
    return t


def spirit(g,team,x,y,level=11):
    s=create('electro_spirit',level,team,x,y)
    s._settled=True
    g.deploy(team,s)
    return s


def launch(g):
    while not g.projs and g.t<3:g.tick()
    assert len(g.projs)==1


def impact(g,t):
    while t.hp==t.max_hp and g.t<4:g.tick()


def stuns(u):
    return [st for st in u.statuses if st.kind=='stun']


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('level',(11,16))
def t_electro_spirit_body_leaves_the_field_at_launch(team,level):
    g=quiet();y,f=(10.5,1) if team=='blue' else (21.5,-1)
    s=spirit(g,team,9.5,y,level);t=target(g,g._opp(team),9.5,y+f*3.2)
    launch(g)
    assert not s.alive and s._self_destructed and t.hp==5000
    g._proc_deaths()
    assert s not in g.players[team].troops
    impact(g,t)
    assert t.hp==5000-s.dmg and len(stuns(t))==1
    g.run(1)
    assert t.hp==5000-s.dmg and not g.projs


def t_launched_electro_spirit_offers_no_body_to_spells_or_attackers():
    g=quiet();s=spirit(g,'blue',9.5,10.5);t=target(g,'red',9.5,13.7)
    launch(g)
    knight=create('knight',11,'red',s.x+0.5,s.y);knight._settled=True;g.deploy('red',knight)
    create('clone',11,'blue',s.x,s.y).apply(g)
    assert not any(x.name=='Electro Spirit' for x in g.players['blue'].troops)
    g.tick()
    assert knight.tgt is not s
    g.run(1)
    assert t.hp==5000-s.dmg


def t_retired_electro_spirit_cannot_launch_again():
    g=quiet();s=spirit(g,'blue',9.5,10.5);t=target(g,'red',9.5,13.7)
    launch(g)
    g._fire(s,t)
    assert len(g.projs)==1
    g.run(1)
    assert t.hp==5000-s.dmg


def t_retired_electro_spirit_chains_nine_targets_from_the_impact():
    # the target and a line of enemies 1 tile apart: the jump strikes nine of them, one every 0.25 s (Supercell December 2025 Chain Speed
    # 0.25 s), each for the full damage with a 0.5 s stun from its own hit
    g=quiet();s=spirit(g,'blue',4,10.5)
    line=[target(g,'red',4+i,13.7) for i in range(10)]
    launch(g)
    assert not s.alive and all(u.hp==5000 for u in line)
    impact(g,line[0]);seen={}
    while len(seen)<9 and g.t<6:
        seen.update({i:[st.dur for st in stuns(u)] for i,u in enumerate(line) if u.hp<5000 and i not in seen})
        if len(seen)<9:g.tick()
    g.run(0.5)
    assert [5000-u.hp for u in line]==[s.dmg]*9+[0]
    assert all(len(seen[i])==1 and 0.4<seen[i][0]<=0.5 for i in range(9)) and not stuns(line[9])


def t_retired_electro_spirit_chain_reach_is_measured_from_the_target():
    # 2.9 tiles beyond the target and more than 3 tiles from the launch point: reached; 3.1 tiles beside the target: not reached
    g=quiet();s=spirit(g,'blue',9.5,10.5);t=target(g,'red',9.5,13.7)
    near=target(g,'red',9.5,16.6);far=target(g,'red',12.6,13.7)
    launch(g)
    assert ((near.x-s.x)**2+(near.y-s.y)**2)**0.5>3
    g.run(1)
    assert t.hp==near.hp==5000-s.dmg and far.hp==5000


def t_retired_electro_spirit_chain_reaches_a_crown_tower():
    g=quiet();tower=g.arena.get_tower('red','princess','left');before=tower.hp
    s=spirit(g,'blue',tower.cx,tower.cy-5.5);t=target(g,'red',tower.cx,tower.cy-2.5)
    launch(g)
    assert not s.alive
    g.run(1)
    assert t.hp==5000-s.dmg and tower.hp==before-s.dmg


def t_electro_spirit_killed_before_its_jump_strikes_nothing():
    g=quiet();s=spirit(g,'blue',9.5,10.5);t=target(g,'red',9.5,13.7)
    s.take_damage(s.hp);g.run(2)
    assert not g.projs and t.hp==5000 and not stuns(t)


def t_cursed_electro_spirit_launch_makes_no_hog():
    g=quiet();s=spirit(g,'blue',9.5,10.5);t=target(g,'red',9.5,13.7)
    witch=create('mother_witch',11,'red',16,26);g.deploy('red',witch)
    curse=next(c for c in witch.components if isinstance(c,CurseOnHit))
    g._do_attack(witch,s);assert s.alive and curse.marks
    g.run(2)
    assert not any(x.name=='Cursed Hog' for x in g.players['red'].troops)
    assert t.hp==5000-s.dmg


def t_retired_electro_spirit_still_supplies_a_soul():
    g=quiet();s=spirit(g,'blue',9.5,10.5);target(g,'red',9.5,13.7)
    king=create('skeleton_king',11,'red',16,26);g.deploy('red',king)
    souls=next(c for c in king.components if isinstance(c,SoulCollect))
    g.tick();assert s.alive and souls.souls==0
    g.run(2);assert souls.souls==1
