import math
import random

from sim.cards import create
from sim.game import Game,Pending
from tests.util import quiet


# Royal Hogs deploy in a horizontal formation (wiki), four bodies touching: the game data's summonRadius of one thousandth of a tile
# stacks them on the point, and the recording of LCQ game 09YP9UPGJGU8 shows every deployment as a line of four badges about 1.1 tiles
# apart. The engine placed them on a circle of that radius, so collisions scattered them into a diamond with one hog a tile ahead.


def xs_ys(units):
    return sorted(round(u.x,6) for u in units),sorted(round(u.y,6) for u in units)


def t_royal_hogs_stand_on_a_horizontal_touching_line():
    random.seed(1);hogs=create('royal_hogs',11,'blue',9,10)
    assert len(hogs)==4
    xs,ys=xs_ys(hogs)
    r=hogs[0].collision_r
    assert r==0.6
    assert ys==[10.0]*4,ys
    assert xs==[round(9+(i-1.5)*2*r,6) for i in range(4)],xs


def t_evolved_royal_hogs_keep_the_line():
    random.seed(1);hogs=create('royal_hogs',11,'blue',9,10,evolved=True)
    xs,ys=xs_ys(hogs)
    assert ys==[10.0]*4 and xs==[7.2,8.4,9.6,10.8],(xs,ys)
    assert all(u.transport=='Air' for u in hogs)


def t_the_line_faces_the_same_way_for_both_teams():
    # a horizontal line has no front, so the red player's hogs stand on the same offsets
    b=create('royal_hogs',11,'blue',9,10);r=create('royal_hogs',11,'red',9,22)
    assert xs_ys(b)[0]==xs_ys(r)[0] and xs_ys(r)[1]==[22.0]*4


def deployed(g,team,x,y,evolved=False):
    # the shipped deploy path: the pending card spawns its units with the game data's 0.1 s stagger and 1 s deploy
    g.pending.append(Pending(team,'royal_hogs',x,y,g.DT,evolved,False,1.0));g.tick()
    hogs=[]
    while len(hogs)<4:
        g.tick();hogs=[u for u in g.players[team].troops if u.name=='Royal Hogs']
    return hogs


def t_placed_hogs_cross_the_river_abreast():
    # the four hogs placed on the bank leave together: one second into the run their front-to-back spread is under a body width
    random.seed(2);g=quiet(Game());hogs=deployed(g,'blue',9,14.5)
    g.run(2.5)
    ys=[u.y for u in hogs]
    assert all(u.alive for u in hogs) and min(ys)>15.5,ys
    assert max(ys)-min(ys)<1.0,ys


def arrivals(g,hogs):
    arrive={}
    while len(arrive)<len(hogs) and g.t<20:
        g.tick()
        for u in hogs:
            if u.id not in arrive and u.tgt is not None and g._dist(u,u.tgt)<=u.rng+0.05:arrive[u.id]=g.t
    return arrive


def t_all_four_line_hogs_reach_the_tower():
    # at these three placements the diamond left its rear hog wedged behind the other three out of range for the whole run; the line
    # brings all four into range. The bound is the 1 s deploy and 0.3 s stagger, about 9.6 tiles at 2.4 tiles/s, and the crowd delay
    # of up to two seconds the outer hogs still pay while they queue behind the inner pair (the recording shows three within 0.5 s)
    for x,y in ((3.5,12.5),(5,12.5),(2.5,14.5)):
        random.seed(2);g=quiet(Game());hogs=deployed(g,'blue',x,y)
        arrive=arrivals(g,hogs)
        assert len(arrive)==4,((x,y),arrive)
        assert max(arrive.values())<8.0,((x,y),arrive)


def t_a_landing_evolved_hog_on_the_fence_corner_settles_and_runs():
    # the airborne outer hog of a line at the bank's lane tile hangs over the fence corner (0,14) or (17,17); hurt there during its
    # deploy it lands, and a landed hog must be settled on a walkable tile like a born one or it stands on the fence for the game
    for team,x,y,cx in (('blue',2.5,14.5,0.7),('red',15.5,17.5,17.3)):
        random.seed(2);g=quiet(Game());hogs=deployed(g,team,x,y,evolved=True)
        outer=min(hogs,key=lambda u:abs(u.x-cx))
        assert outer.transport=='Air' and not g._walkable(outer.x,outer.y,True)
        outer.hp-=1
        # components do not act during the deploy, so the landing comes with the hog's first action
        while outer.transport=='Air' and g.t<3.0:g.tick()
        assert outer.transport=='Ground' and g._walkable(outer.x,outer.y,True),(outer.x,outer.y,g.t)
        arrive=arrivals(g,hogs)
        assert outer.id in arrive and outer.alive,(team,arrive)


def t_a_line_hog_born_on_the_fence_settles_on_the_row():
    # placed at the recording's tile the outer hog would stand on the fence corner; the walkable-tile rule keeps it beside the line
    random.seed(2);g=quiet(Game());hogs=deployed(g,'blue',2.5,14.5)
    for u in hogs:
        assert g._walkable(u.x,u.y,True),(u.x,u.y)
        assert abs(u.y-14.5)<0.3,(u.x,u.y)
    xs=sorted(u.x for u in hogs)
    assert xs[-1]-xs[0]>2.5,xs


def t_other_summons_keep_their_circles_and_lines():
    random.seed(3)
    barbs=create('barbarians',11,'blue',9,10)
    assert len(barbs)==5 and all(abs(math.hypot(u.x-9,u.y-10)-0.7)<1e-6 for u in barbs)
    recruits=create('royal_recruits',11,'blue',9,10)
    assert xs_ys(recruits)[0]==[4.0,6.0,8.0,10.0,12.0,14.0]
    archers=create('archers',11,'blue',9,10)
    assert xs_ys(archers)==([8.5,9.5],[10.0,10.0])
