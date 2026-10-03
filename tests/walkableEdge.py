import pytest

from sim.cards import create
from sim.game import Game

# Game._walkable reads the tile under a point with floor: a point up to 1 tile outside the left or back edge (x or y in -1..0) is off the
# arena, not on column or row 0, so a ground unit born there is settled on the nearest walkable tile centre.


def bare():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


@pytest.mark.parametrize('x,y',[(-0.8,8),(-0.2,20),(9,-0.8),(7.5,-0.3)])
def t_point_just_outside_the_left_or_back_edge_is_not_walkable(x,y):
    g=bare();assert not g._walkable(x,y,False) and not g._walkable(x,y,True)


@pytest.mark.parametrize('x,y',[(0.2,8),(17.8,20),(9,0.2),(9,31.8)])
def t_point_just_inside_the_edge_is_walkable(x,y):
    assert bare()._walkable(x,y,False)


@pytest.mark.parametrize('x,y,to',[(-0.8,8.3,(0.5,8.5)),(9.3,-0.8,(9.5,0.5))])
def t_ground_unit_born_just_outside_is_settled_inside(x,y,to):
    g=bare();k=create('knight',11,'blue',x,y);g._place('blue',k,1.0)
    assert (k.x,k.y)==to
