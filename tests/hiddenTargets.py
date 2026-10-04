import pytest

from sim import fx
from sim.cards import create
from sim.game import Game
from sim.units import Status,has
from tests.util import Dummy

# Wiki Royal Ghost: while invisible "he will not be targeted by opposing units, but can still be hit by area damage or spells", and
# "any projectiles that have already been fired to target him will deal damage to him". A burrowed unit is also immune (sim.units.immune).
KINDS=('invisible','burrowed')


def arena():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


def aim(g,att,tgt,t):
    end=g.t+t
    while att.tgt is not tgt and g.t<end:g.tick()
    return att.tgt is tgt


def watch(g,att,tgt,ticks,attr='tgt'):
    out=[]
    for _ in range(ticks):g.tick();out.append(getattr(att,attr) is tgt)
    return out


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('name',('cannon','inferno_tower'))
def t_defender_building_drops_a_target_that_hides(name,kind):
    g=arena();b=create(name,11,'red',9,23);b.decay=0;g.deploy('red',b)
    d=Dummy('blue',9,19,hp=50000,spd=0,dmg=0);g.deploy('blue',d)
    assert aim(g,b,d,3);g.run(1.5);assert d.hp<50000
    d.statuses.append(Status(kind,3.0))
    # 0.6 s for a cannonball already in the air to land, then 1.5 s with nothing more
    held=watch(g,b,d,12);hp=d.hp;held+=watch(g,b,d,30)
    assert not any(held) and d.hp==hp and b.aggro_tgt is not d,(held.count(True),hp-d.hp)


@pytest.mark.parametrize('kind',KINDS)
def t_building_targeting_troop_drops_a_building_that_hides(kind):
    g=arena();b=create('cannon',11,'red',9,21.5);b.dmg=0;b.decay=0;g.deploy('red',b)
    gi=create('giant',11,'blue',9,18);g.deploy('blue',gi)
    assert aim(g,gi,b,6);g.run(2.0);assert b.hp<b.max_hp
    b.statuses.append(Status(kind,3.0));hp=b.hp
    held=watch(g,gi,b,40)
    assert not any(held) and b.hp==hp and gi.aggro_tgt is not b,(held.count(True),hp-b.hp)


@pytest.mark.parametrize('name',('cannon','inferno_tower'))
def t_defender_building_drops_the_royal_ghost_when_he_turns_invisible(name):
    g=arena();b=create(name,11,'red',9,24);b.decay=0;g.deploy('red',b)
    gh=create('royal_ghost',11,'blue',9,20);gh.spd=0;gh.hp=gh.max_hp=10**6;g.deploy('blue',gh)
    v=Dummy('red',9,21,hp=1,spd=0,dmg=0);g.deploy('red',v)
    g.run(0.2);assert has(gh,'invisible') and b.tgt is not gh
    # his first swing kills the victim and reveals him; the building takes him; 2 s without an attack and he is invisible again
    assert aim(g,b,gh,2) and not v.alive and not has(gh,'invisible')
    while not has(gh,'invisible') and g.t<6:g.tick()
    assert has(gh,'invisible')
    held=watch(g,b,gh,12);hp=gh.hp;held+=watch(g,b,gh,30)
    assert has(gh,'invisible') and not any(held) and gh.hp==hp,(held.count(True),hp-gh.hp)


@pytest.mark.parametrize('name,xy',(('cannon',(9,23)),('inferno_tower',(9,23)),('giant',(9,22)),('knight',(9,21.5))))
def t_defenders_drop_the_evolved_goblin_drill_while_it_is_submerged(name,xy):
    # review t2032 finding 6: a Cannon, an Inferno Tower and a Giant held the submerged drill for the whole 1 s hide; a Knight turned away
    g=arena();d=create(name,11,'red',*xy);d.decay=0;g._place('red',d,0)
    gd=create('goblin_drill',11,'blue',9,20,evolved=True);g._place('blue',gd,1.0)
    while has(gd,'burrowed') or has(gd,'deploying'):g.tick()
    while d.tgt is not gd and g.t<30:gd.hp=gd.max_hp;g.tick()
    assert d.tgt is gd
    gd.hp=int(gd.max_hp*0.6);g.tick()
    assert has(gd,'burrowed')
    held=[]
    while has(gd,'burrowed'):held.append(d.tgt is gd);g.tick()
    assert len(held)>=19 and not any(held),(len(held),held.count(True))


@pytest.mark.parametrize('kind',KINDS)
def t_troop_drops_a_target_that_hides(kind):
    g=arena();k=create('knight',11,'red',9,22);g.deploy('red',k)
    d=Dummy('blue',9,20,hp=50000,spd=0,dmg=0);g.deploy('blue',d)
    assert aim(g,k,d,3);g.run(1.0);assert d.hp<50000
    d.statuses.append(Status(kind,3.0));hp=d.hp
    held=watch(g,k,d,40)
    assert not any(held) and d.hp==hp


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('name',('knight','cannon'))
def t_attacker_with_nothing_else_to_aim_at_forgets_a_target_that_hides(name,kind):
    # the held target is dropped, not kept for when it reappears (wiki Royal Ghost: "enemy troops will stop chasing him")
    g=arena();a=create(name,11,'red',9,22);a.decay=0;g.deploy('red',a)
    d=Dummy('blue',9,20,hp=50000,spd=0,dmg=0);g.deploy('blue',d)
    assert aim(g,a,d,3) and a.aggro_tgt is d
    d.statuses.append(Status(kind,3.0));g.tick()
    assert a.tgt is not d and a.aggro_tgt is None


@pytest.mark.parametrize('kind',KINDS)
def t_crown_tower_drops_a_target_that_hides(kind):
    g=Game();tw=g.arena.get_tower('red','princess','left')
    d=Dummy('blue',tw.cx,tw.cy-6,hp=50000,spd=0,dmg=0);g.deploy('blue',d)
    assert any(watch(g,tw.troop,d,40,'lock'));g.run(1.0);assert d.hp<50000
    d.statuses.append(Status(kind,3.0))
    held=watch(g,tw.troop,d,12,'lock');hp=d.hp;held+=watch(g,tw.troop,d,30,'lock')
    assert not any(held) and d.hp==hp


def t_area_damage_hits_invisible_units_but_not_underground_ones():
    g=arena();a=create('knight',11,'red',9,20);b=create('knight',11,'red',9.6,20);c=create('knight',11,'red',8.4,20)
    for u in (a,b,c):g.deploy('red',u)
    a.statuses.append(Status('invisible',5));b.statuses.append(Status('burrowed',5))
    create('fireball',11,'blue',9,20).apply(g)
    assert a.hp<a.max_hp and b.hp==b.max_hp and c.hp<c.max_hp
    # a splash attack on a visible target reaches the invisible troop beside it (wiki Royal Ghost history 8/4/2022)
    a.hp=a.max_hp;v=create('valkyrie',11,'blue',9,19)
    fx.SplashAttack().on_attack(v,c,g)
    assert a.hp==a.max_hp-v.dmg and b.hp==b.max_hp


def t_projectile_fired_before_the_target_hides_still_lands():
    g=arena();m=create('musketeer',11,'blue',9,10);g.deploy('blue',m)
    d=Dummy('red',9,15,hp=50000,spd=0,dmg=0);g.deploy('red',d)
    while not g.projs and g.t<3:g.tick()
    assert g.projs and d.hp==50000
    d.statuses.append(Status('invisible',3.0))
    held=watch(g,m,d,30)
    assert d.hp==50000-m.dmg and not any(held)


def lightning_or_vines(name):
    g=arena()
    under,inv,a,b,c=(Dummy('red',9+dx,20,hp=hp,spd=0,dmg=0) for dx,hp in ((0,9000),(0.5,8000),(-0.5,7000),(1,6000),(-1,5000)))
    for d in (under,inv,a,b,c):g.deploy('red',d)
    under.statuses.append(Status('burrowed',5));inv.statuses.append(Status('invisible',5))
    sp=create(name,11,'blue',9,20);sp.apply(g);g.spells.append(sp);g.run(1.5)
    return under,inv,a,b,c


@pytest.mark.parametrize('name',('lightning','vines'))
def t_targeting_spells_strike_invisible_troops_but_not_underground_ones(name):
    # Lightning and Vines pick the three highest-hp bodies: the underground one is not among them, the invisible one is (wiki Lightning:
    # it "can hit the Archer Queen even with her ability active"); Vines roots only troops
    under,inv,a,b,c=lightning_or_vines(name)
    assert under.hp==under.max_hp and inv.hp<inv.max_hp and a.hp<a.max_hp and b.hp<b.max_hp and c.hp==c.max_hp
    assert not has(under,'stun','freeze')


def _pair(g,team,hid_xy,vis_xy,kind,hp=(50000,50000)):
    hid=Dummy(team,*hid_xy,hp=hp[0],spd=0,dmg=0);vis=Dummy(team,*vis_xy,hp=hp[1],spd=0,dmg=0)
    for d in (hid,vis):g.deploy(team,d)
    hid.statuses.append(Status(kind,10));return hid,vis


def _ewiz(g,kind):
    w=create('electro_wizard',11,'blue',9,10);g.deploy('blue',w);p=Dummy('red',9,13,hp=50000,spd=0,dmg=0);g.deploy('red',p)
    hid,vis=_pair(g,'red',(9.5,11),(9,14.5),kind)
    fx.DualTarget().on_attack(w,p,g);return hid,vis


def _chain(g,kind):
    e=create('electro_dragon',11,'blue',9,6);g.deploy('blue',e);p=Dummy('red',9,9,hp=50000,spd=0,dmg=0);g.deploy('red',p)
    hid,vis=_pair(g,'red',(9,10),(9,12.5),kind)
    fx.chain(e,p,g);return hid,vis


def _rocket(g,kind):
    m=create('goblin_machine',11,'blue',9,10);g.deploy('blue',m);rl=next(c for c in m.components if isinstance(c,fx.RocketLauncher))
    hid,vis=_pair(g,'red',(9,13),(9,6),kind)
    for _ in range(100):rl.on_tick(m,g)
    return hid,vis


def _dash(g,kind):
    k=create('golden_knight',11,'blue',9,10);g.deploy('blue',k);ab=k.ability
    hid,vis=_pair(g,'red',(9,12),(9,14),kind)
    ab.activate(k,g)
    for _ in range(5):ab.tick(g.DT,k,g)
    return hid,vis


def _rescue(g,kind):
    lp=create('little_prince',11,'blue',9,10);g.deploy('blue',lp)
    hid,vis=_pair(g,'red',(9,11.5),(9,13),kind)
    lp.ability.activate(lp,g);return hid,vis


def _hurl(g,kind):
    gi=create('giant',11,'blue',9,10,hero=True);g.deploy('blue',gi)
    hid,vis=_pair(g,'red',(9,11.5),(9.5,11.5),kind,hp=(60000,50000))
    gi.ability.activate(gi,g);return hid,vis


def _warp(g,kind):
    mm=create('mega_minion',11,'blue',9,10,hero=True);g.deploy('blue',mm)
    hid,vis=_pair(g,'red',(9,14),(9,18),kind,hp=(40000,50000))
    mm.ability.activate(mm,g);return hid,vis


def _cadets(g,kind):
    b=create('balloon',11,'blue',9,10,hero=True);g.deploy('blue',b)
    hid,vis=_pair(g,'red',(9,12),(9,14),kind)
    b.ability.activate(b,g);return hid,vis


def _whirl(g,kind):
    v=create('valkyrie',11,'blue',9,10,hero=True);g.deploy('blue',v)
    hid,vis=_pair(g,'red',(9,12),(9,14),kind)
    v.ability.activate(v,g)
    # she lands a tile short of the troop she picked
    return hid,vis,abs(v.y-13)<0.01


def _cage(g,kind):
    c=create('goblin_cage',11,'blue',9,10,evolved=True);g.deploy('blue',c);cg=next(x for x in c.components if isinstance(x,fx.EvoGoblinCage))
    hid,vis=_pair(g,'red',(9,11),(9,12.5),kind)
    cg.on_tick(c,g);return hid,vis,cg.trapped is vis


PICKERS={'electro_wizard_second':_ewiz,'goblin_machine_rocket':_rocket,'golden_knight_dash':_dash,
         'little_prince_rescue':_rescue,'giant_hero_hurl':_hurl,'mega_minion_hero_warp':_warp,'balloon_hero_cadets':_cadets,
         'valkyrie_hero_whirlwind':_whirl,'evolved_goblin_cage':_cage}


@pytest.mark.parametrize('kind',KINDS)
@pytest.mark.parametrize('who',sorted(PICKERS))
def t_attacks_that_pick_their_own_target_pass_over_hidden_units(who,kind):
    r=PICKERS[who](arena(),kind)
    if len(r)==3:
        hid,vis,ok=r;assert ok and hid.hp==hid.max_hp and not has(hid,'stun'),who
    else:
        hid,vis=r;assert hid.hp==hid.max_hp and vis.hp<vis.max_hp and not has(hid,'stun'),(who,hid.hp,vis.hp)


@pytest.mark.parametrize('kind',KINDS)
def t_chain_bounce_reaches_invisible_units_but_not_underground_ones(kind):
    # wiki Royal Ghost: "An Electro Dragon or Electro Spirit's attack can also chain onto him"; wiki Tesla: underground it "cannot be
    # targeted by any troops". The chain goes on to the visible unit either way.
    hid,vis=_chain(arena(),kind)
    if kind=='invisible':assert hid.hp<hid.max_hp and has(hid,'stun') and vis.hp<vis.max_hp,(hid.hp,vis.hp)
    else:assert hid.hp==hid.max_hp and not has(hid,'stun') and vis.hp<vis.max_hp,(hid.hp,vis.hp)


@pytest.mark.parametrize('kind',KINDS)
def t_bandit_drops_a_dash_target_that_hides_in_the_wind_up(kind):
    g=arena();b=create('bandit',11,'blue',9,5);g.deploy('blue',b);bd=next(c for c in b.components if isinstance(c,fx.BanditDash))
    d=Dummy('red',9,10,hp=50000,spd=0,dmg=0);g.deploy('red',d);g.deploy('red',Dummy('red',9,13,hp=50000,spd=0,dmg=0))
    while not bd.charging and g.t<2:g.tick()
    assert bd.charging and bd.dtgt is d
    d.statuses.append(Status(kind,3.0));g.tick()
    assert not bd.charging and not bd.dashing and bd.dtgt is None
    g.run(1.5);assert d.hp==50000
