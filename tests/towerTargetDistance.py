from sim.cards import create
from sim.game import Game
from tests.util import quiet


# Range and sight are edge to edge, both collision radii counted (wiki Giant Skeleton: range 2 -> 0.8 with its 1.2 collision radius left the
# effective range unchanged). Game._find_target measured troops and buildings that way but a Crown Tower from the attacker's centre to the
# tower's edge, so a building-targeting troop preferred a building by up to its own radius (08PY8998G90L: hut 6.54 against tower 7.39).


def hog_between(cx=9.1):
    # a Hog Rider (radius 0.6) 5.5 tiles from the centre of red's right princess tower (radius 1.0): 3.9 tiles edge to edge, 4.5 without his
    # radius; a Cannon (radius 0.6) 5.4 tiles away is 4.2 edge to edge
    g=quiet(Game());h=create('hog_rider',11,'blue',14.5,20.0);g.deploy('blue',h)
    c=create('cannon',11,'red',cx,20.0);g.deploy('red',c)
    return g,h,c,g.arena.get_tower('red','princess','right')


def t_building_targeter_compares_tower_and_building_edge_to_edge():
    g,h,c,tw=hog_between()
    t,d=g._find_target(h)
    assert t is tw and abs(d-3.9)<1e-9,f"The tower 3.9 tiles away edge to edge should beat the Cannon at 4.2, got {getattr(t,'name',t)} {d:.3f}"


def t_building_still_wins_when_nearer_edge_to_edge():
    # the Cannon 4.9 tiles away is 3.7 edge to edge, nearer than the tower's 3.9
    g,h,c,tw=hog_between(cx=9.6)
    t,d=g._find_target(h)
    assert t is c and abs(d-3.7)<1e-9,(getattr(t,'name',t),d)


def t_default_tower_target_is_edge_to_edge():
    g,h,c,tw=hog_between()
    t,d=g._default_target(h)
    assert t is tw and abs(d-3.9)<1e-9,(t,d)


def t_giant_starts_on_a_tower_at_edge_to_edge_reach():
    # the reach was already edge to edge through the held target; the Giant (radius 0.75, range 1.2) stops 2.95 tiles from the tower centre
    g=quiet(Game());gi=create('giant',11,'blue',14.5,21.0);g.deploy('blue',gi);tw=g.arena.get_tower('red','princess','right')
    hp=tw.hp
    while tw.hp==hp and g.t<10:g.tick()
    d=tw.dist(gi.x,gi.y)-gi.collision_r
    assert tw.hp<hp and 1.2-0.1<d<=1.2+1e-9,f"The Giant should hit from 1.2 tiles edge to edge, {d:.3f}"
