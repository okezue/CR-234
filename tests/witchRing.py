import math

import pytest

from sim import fx
from sim.cards import create
from sim.game import Game
from tests.util import Dummy

# wiki Witch: "Every 7 seconds, the Witch will passively summon a group of four Skeletons surrounding her", the first wave 1 s after deploy;
# wiki Night Witch: "Every 5 seconds, the Night Witch will passively summon a group of two Bats surrounding her"; RoyaleAPI cr-api-data
# characters: Witch spawn_radius 2000, spawn_angle_shift 0; DarkWitch spawn_radius 1500, spawn_angle_shift 90. Goblin Hut, Tombstone,
# Barbarian Hut and Furnace carry no spawn radius there and keep the wave in front.


def bare():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


def first_wave(name,team='blue',evolved=False,x=9,y=8):
    # the spawner stands still; returns it and the offsets of its first wave at the tick it appears
    g=bare();s=create(name,11,team,x,y,evolved=evolved);s=s[0] if isinstance(s,list) else s;s._settled=True;g.deploy(team,s);s.spd=0
    while g.t<3:
        g.tick()
        wave=[t for t in g.players[team].troops if getattr(t,'_spawner',None) is s]
        if wave:return s,[(t.x-s.x,t.y-s.y) for t in wave]
    return s,[]


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('evolved',(False,True))
def t_witch_wave_stands_on_a_two_tile_circle_around_her(team,evolved):
    w,off=first_wave('witch',team,evolved)
    assert len(off)==4 and all(abs(math.hypot(dx,dy)-2.0)<1e-6 for dx,dy in off)
    ang=sorted(math.degrees(math.atan2(dy,dx))%360 for dx,dy in off)
    assert all(abs((b-a)-90)<1e-6 for a,b in zip(ang,ang[1:]))
    # the first stands at her front (toward the enemy side with no target), so one Skeleton is behind her and one to each side
    f=1 if team=='blue' else -1
    assert sorted((round(dx,6)+0.0,round(dy*f,6)+0.0) for dx,dy in off)==[(-2.0,0.0),(0.0,-2.0),(0.0,2.0),(2.0,0.0)]


@pytest.mark.parametrize('team',('blue','red'))
def t_night_witch_bats_stand_at_her_sides(team):
    w,off=first_wave('night_witch',team)
    assert len(off)==2 and sorted((round(dx,6)+0.0,round(dy,6)+0.0) for dx,dy in off)==[(-1.5,0.0),(1.5,0.0)]


def t_witch_wave_faces_her_target():
    g=bare();w=create('witch',11,'blue',9,8);w._settled=True;g.deploy('blue',w);w.spd=0
    d=Dummy('red',13,11,hp=10**6,spd=0,dmg=0);d._settled=True;g.deploy('red',d)
    while not [t for t in g.players['blue'].troops if getattr(t,'_spawner',None) is w]:g.tick()
    off=[(t.x-w.x,t.y-w.y) for t in g.players['blue'].troops if getattr(t,'_spawner',None) is w]
    front=max(off,key=lambda o:o[0]*0.8+o[1]*0.6)
    assert abs(front[0]-1.6)<1e-6 and abs(front[1]-1.2)<1e-6


@pytest.mark.parametrize('team,x,y',[('blue',0.6,8),('blue',1.2,8),('blue',1.5,8),('blue',9,1.2),('red',1.2,24)])
def t_ring_skeleton_off_the_arena_is_settled_inside(team,x,y,monkeypatch):
    # births up to 1 tile outside the left or back edge are off the arena too; the old front jitter is pinned so a wave in front (code
    # without the ring) lands inside on every run
    monkeypatch.setattr(fx.random,'uniform',lambda a,b:0.0)
    w,off=first_wave('witch',team,x=x,y=y)
    assert len(off)==4 and all(0<=w.x+dx<18 and 0<=w.y+dy<32 for dx,dy in off)


@pytest.mark.parametrize('name',('goblin_hut','tombstone','barbarian_hut','furnace'))
def t_spawners_without_a_spawn_radius_have_no_ring(name):
    s=create(name,11,'blue',9,8)
    assert next(c for c in s.components if isinstance(c,fx.SpawnTimer)).radius==0


@pytest.mark.parametrize('name',('tombstone','furnace'))
def t_spawners_without_a_spawn_radius_keep_the_wave_in_front(name):
    _,off=first_wave(name)
    assert off and all(1.0<=dy<=3.0 and abs(dx)<=1.0 for dx,dy in off)
