import pytest

from sim.cards import create, load
from sim.game import Game


DECK=['inferno_dragon','inferno_tower','mighty_miner','fireball','giant','musketeer','bomber','arrows']
# level 11 tiers: wiki stat tables for the Inferno Dragon and Inferno Tower; the Mighty Miner's first tier follows the game data export
# (17 x 2.56 = 43) where the wiki still shows 40. No crownTowerDamagePercent exists on any of the three in the export.
TIERS={'inferno_dragon':[35,120,422],'inferno_tower':[43,158,847],'mighty_miner':[43,204,409]}


def quiet(g):
    for tw in g.arena.towers:
        tw.rng=0
        if tw.troop:tw.troop.RNG=0


def setup(team='blue'):
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0})
    quiet(g)
    opp=g._opp(team)
    tw=next(t for t in g.arena.towers if t.team==opp and t.ttype=='princess')
    return g,tw


def beam(g,tw,name,team,dy,seconds,evolved=False):
    t=create(name,11,team,tw.cx,tw.cy+dy,evolved=evolved)
    if isinstance(t,list):t=t[0]
    t.spd=0;g.deploy(team,t)
    last=tw.hp;hits=[]
    for _ in range(int(round(seconds/g.DT))):
        g.tick()
        if tw.hp!=last:hits.append((round(g.t,2),last-tw.hp,last));last=tw.hp
        if not tw.alive:break
    return t,hits


@pytest.mark.parametrize('team,dy',(('blue',-2.5),('red',2.5)))
@pytest.mark.parametrize('name',('inferno_dragon','inferno_tower','mighty_miner'))
def t_ramp_tiers_reach_crown_towers_in_full(name,team,dy):
    g,tw=setup(team);start=tw.hp
    t,hits=beam(g,tw,name,team,dy,6.0)
    tiers=TIERS[name]
    def tier(at):return tiers[0] if at<1.5 else tiers[1] if at<3.0 else tiers[2]
    assert [d for at,d,_ in hits if at<1.5] and [d for at,d,_ in hits if 1.5<at<3.0] and [d for at,d,_ in hits if at>3.0]
    # the final blow is capped by the tower's remaining health; every other hit lands the full tier
    assert [(at,d) for at,d,before in hits]==[(at,min(tier(at),before)) for at,d,before in hits]
    assert t.ct_dmg==tiers[2]
    assert start-tw.hp==sum(d for _,d,_ in hits)
    # a level 11 Princess Tower cannot survive six seconds of a locked ramp at these tiers
    assert not tw.alive


def t_ramp_reset_returns_tower_damage_to_the_first_tier():
    g,tw=setup();t,hits=beam(g,tw,'inferno_dragon','blue',-2.5,3.5)
    assert hits[-1][1]==422 and t.ct_dmg==422
    ramp=next(c for c in t.components if type(c).__name__=='RampUp')
    ramp._reset(t)
    assert t.dmg==35 and t.ct_dmg==35


def t_stun_resets_tower_damage_through_the_shipped_tick():
    from sim.units import Status
    g,tw=setup();t,hits=beam(g,tw,'inferno_tower','blue',-2.5,3.5)
    assert hits[-1][1]==847
    t.statuses.append(Status('stun',0.5));last=tw.hp;after=[]
    for _ in range(int(round(1.2/g.DT))):
        g.tick()
        if tw.hp!=last:after.append(last-tw.hp);last=tw.hp
    # the first hit after the stun clears is tier one again, for the tower as well as the beam
    assert after and after[0]==43
    assert t.ct_dmg==t.dmg


def t_king_tower_takes_ramped_damage_too():
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0});quiet(g)
    king=next(t for t in g.arena.towers if t.team=='red' and t.ttype=='king');king.active=True
    t,hits=beam(g,king,'inferno_dragon','blue',-3.0,4.0)
    assert [d for at,d,_ in hits if at>3.0] and set(d for at,d,_ in hits if at>3.0)=={422}


def t_ramp_tower_damage_matches_troop_damage_at_every_tier():
    # the same beam against a troop and a tower must deal the same per-hit numbers, since no reduction is sourced
    g,tw=setup();t,tower_hits=beam(g,tw,'inferno_tower','blue',-2.5,3.6)
    assert tw.alive
    g2=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0});quiet(g2)
    victim=create('golem',11,'red',9,14);victim.spd=0;victim.dmg=0;g2.deploy('red',victim)
    t2=create('inferno_tower',11,'blue',9,11.5);g2.deploy('blue',t2)
    last=victim.hp;troop_hits=[]
    for _ in range(int(round(3.6/g2.DT))):
        g2.tick()
        if victim.hp!=last:troop_hits.append(last-victim.hp);last=victim.hp
    assert victim.alive
    assert [d for _,d,_ in tower_hits]==troop_hits
    assert 847 in troop_hits


def t_reduced_spell_tower_damage_is_untouched():
    # Fireball keeps its sourced -75 percent Crown Tower reduction: the ramp change must not leak into spells
    c=load()['cards']['fireball'];lvl=11
    assert c['stats']['towerDamage'][lvl-1]<c['stats']['damage'][lvl-1]
    g,tw=setup();before=tw.hp
    create('fireball',lvl,'blue',tw.cx,tw.cy).apply(g)
    assert before-tw.hp==c['stats']['towerDamage'][lvl-1]


def t_non_ramp_troop_tower_damage_unchanged():
    g,tw=setup();t,hits=beam(g,tw,'musketeer','blue',-4.0,3.0)
    dmg=load()['cards']['musketeer']['stats']['damage'][10]
    assert hits and set(d for _,d,_ in hits)=={dmg}


def t_evolved_inferno_dragon_fourth_tier_reaches_towers():
    # the tower is given extra health only so the 20 second fourth tier can be reached on a single lock
    g,tw=setup();tw.hp=tw.max_hp=50000
    t,hits=beam(g,tw,'inferno_dragon','blue',-2.5,21.0,evolved=True)
    evo=next(c for c in t.components if type(c).__name__=='EvoInfernoDragon')
    assert evo.s4_active
    assert hits[-1][1]==844 and t.ct_dmg==844
    third=[d for at,d,_ in hits if 3.0<at<19.5]
    assert third and set(third)=={422}
    assert tw.alive

