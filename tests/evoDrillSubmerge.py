import math

from sim.cards import create, load
from sim.game import Game
from sim.units import has, hidden, immune
from tests.util import Dummy, quiet


# Export GoblinDrill_EV1_relocate: hideTime 1000, hideHpThresholds [66, 33]. Wiki Goblin Drill/Evolution: at 66% and 33% the drill
# resurfaces leaving Goblins behind, a quarter turn around the Crown Tower it was deployed next to, else in the same spot; history
# 8/10/2024 removed the submerges' damage and pushback and gave the left Goblins a deploy time. LCQ game 09YP9UPGQ2YU (hd_2611, End
# clip): placed on the inner side of a princess tower it hides for 1 s, comes back in front of the tower, then on its outer side.


def surfaced(team,x,y,g=None):
    g=g or quiet(Game());gd=create('goblin_drill',11,team,x,y,evolved=True);g._place(team,gd,1.0)
    while has(gd,'burrowed') or has(gd,'deploying'):g.tick()
    return g,gd


def goblins(g,team):return [t for t in g.players[team].troops if t.name=='Goblin' and t.alive]


def resurface(g,gd,frac):
    gd.hp=int(gd.max_hp*frac);g.tick()
    assert has(gd,'burrowed')
    while has(gd,'burrowed'):g.tick()
    return round(gd.x,6),round(gd.y,6)


def t_hide_time_and_relocation_range_are_patched_from_the_export():
    c=load()['cards']['goblin_drill'];b=c['evo']['skills']['burrow']
    assert b['hideTime']==1.0 and b['relocateRange']==2.0 and b['resurfacePercent']==[66,33]
    assert c['src']['evo.skills.burrow.hideTime']=='patch:2026-10-01u' and c['src']['evo.skills.burrow.relocateRange']=='patch:2026-10-01u'


def t_submerged_drill_is_hidden_and_immune_for_one_second():
    g,gd=surfaced('blue',9,20)
    gd.hp=int(gd.max_hp*0.6);g.tick()
    assert has(gd,'burrowed') and hidden(gd) and immune(gd)
    hp=gd.hp;gd.take_damage(500);assert gd.hp==hp
    n=0
    while has(gd,'burrowed'):n+=1;g.tick()
    assert n==round(1.0/g.DT),n
    assert math.isclose(gd.x,9) and math.isclose(gd.y,20) and not hidden(gd)


def t_submerges_deal_no_damage_or_pushback():
    # a flying dummy is in the spawn blast's reach but never a Goblin's target; the arrival blast still lands
    g=quiet(Game());d=Dummy('red',9,21,hp=50000,dmg=0,spd=0);d.transport='Air';g.deploy('red',d)
    g,gd=surfaced('blue',9,20,g)
    assert 50000-d.hp==84
    xy=(d.x,d.y)
    for frac in (0.6,0.3):
        gd.hp=int(gd.max_hp*frac);g.tick();assert has(gd,'burrowed') and d.hp==50000-84 and (d.x,d.y)==xy
        while has(gd,'burrowed'):g.tick()
    assert d.hp==50000-84


def t_goblins_are_left_behind_and_deploy():
    g,gd=surfaced('blue',9,20)
    gd.hp=int(gd.max_hp*0.6);g.tick();first=goblins(g,'blue')
    assert len(first)==2 and all(has(t,'deploying') and math.hypot(t.x-9,t.y-20)<=0.75 for t in first)
    while has(gd,'burrowed'):g.tick()
    gd.hp=int(gd.max_hp*0.3);g.tick();second=[t for t in goblins(g,'blue') if t not in first]
    assert len(second)==1 and has(second[0],'deploying')


def t_beside_a_crown_tower_it_turns_a_quarter_round_it_from_the_inner_side_toward_the_river():
    # the recorded geometry (raw frame of game 09YP9UPGQ2YU): offset (-2.5, 0.5) from the tower centre -> (0.5, 2.5) -> (2.5, -0.5)
    g,gd=surfaced('red',12,7)
    assert resurface(g,gd,0.6)==(15.0,9.0) and resurface(g,gd,0.3)==(17.0,6.0)
    # the same placements seen from the other side and mirrored: the turn keeps to the tower's inner side, toward the river
    for x,turns in ((12,((15.0,23.0),(17.0,26.0))),(6,((3.0,23.0),(1.0,26.0)))):
        g,gd=surfaced('blue',x,25)
        assert (resurface(g,gd,0.6),resurface(g,gd,0.3))==turns


def t_away_from_the_towers_or_once_the_tower_is_down_it_comes_back_in_place():
    g,gd=surfaced('blue',9,20)
    assert resurface(g,gd,0.6)==(9.0,20.0)
    g,gd=surfaced('blue',12,25);tw=g.arena.get_tower('red','princess','right');tw.take_damage(tw.hp)
    assert not tw.alive and resurface(g,gd,0.6)==(12.0,25.0)


def t_one_blow_past_both_thresholds_submerges_once():
    g,gd=surfaced('blue',12,25)
    assert resurface(g,gd,0.3)==(15.0,23.0) and len(goblins(g,'blue'))==1
    gd.hp=int(gd.max_hp*0.2);g.tick()
    assert not has(gd,'burrowed') and len(goblins(g,'blue'))==1


def t_lifetime_runs_on_while_hidden():
    g,gd=surfaced('blue',9,20)
    gd.hp=h=int(gd.max_hp*0.6);g.tick()
    while has(gd,'burrowed'):g.tick()
    assert gd.decay*(1.0-g.DT)<=h-gd.hp<=gd.decay*(1.0+2*g.DT),(h-gd.hp)
