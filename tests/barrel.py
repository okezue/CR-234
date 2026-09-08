import random
from unittest.mock import patch

from sim.cards import create
from sim.game import Game
from sim.units import has
from tests.util import quiet


def legal(game,troop):
    a=game.arena
    return 0<=troop.x<a.W and 0<=troop.y<a.H and not a.blocked(int(troop.x),int(troop.y))


def t_centered_barrel_spawns_outside_intact_towers():
    for team in ('blue','red'):
        for side in ('left','right','king'):
            for seed in (0,42,100):
                random.seed(seed);g=quiet(Game());enemy=g._opp(team)
                tower=g.arena.get_tower(enemy,'king' if side=='king' else 'princess',None if side=='king' else side)
                barrel=create('goblin_barrel',11,team,tower.cx,tower.cy);barrel.apply(g)
                troops=g.players[team].troops
                assert len(troops)==3 and all(legal(g,t) for t in troops)
                assert len({(int(t.x),int(t.y)) for t in troops})==3
                assert all(has(t,'deploying') and t.hp==202 and t.dmg==125 and t.proj_spd==0 for t in troops)
                for _ in range(50):
                    g.tick()
                    assert all(legal(g,t) for t in troops if t.alive)


def t_barrel_legal_samples_and_rng_are_unchanged():
    for evolved in (False,True):
        for x,y in ((9,10),(5.5,25.5),(3.5,25.5)):
            g=Game();random.seed(17)
            centers=[(x,y)]+([(g.arena.W-x,y)] if evolved else [])
            expected=[]
            for cx,cy in centers:
                expected.extend((cx+random.uniform(-1,1),cy+random.uniform(-1,1)) for _ in range(3))
            state=random.getstate();random.seed(17)
            barrel=create('goblin_barrel',11,'blue',x,y,evolved=evolved);barrel.apply(g)
            assert random.getstate()==state
            troops=g.players['blue'].troops
            assert len(troops)==len(expected)
            for t,(px,py) in zip(troops,expected):
                if 0<=px<18 and 0<=py<32 and not g.arena.blocked(int(px),int(py)):
                    assert (t.x,t.y)==(px,py)
                assert legal(g,t)
            points=[(t.x,t.y) for t in troops];barrel.apply(g)
            assert [(t.x,t.y) for t in troops]==points and len(g.players['blue'].troops)==len(expected)
            assert random.getstate()==state


def t_barrel_edges_river_and_decoy_lanes():
    points=((0,0),(17.9,31.9),(9,15.5),(3.5,15.5),(0.1,14.5),(17.9,17.5),(3.5,25.5),(14.5,6.5))
    for team in ('blue','red'):
        for x,y in points:
            random.seed(42);g=Game();barrel=create('goblin_barrel',11,team,x,y,evolved=True);barrel.apply(g)
            real,decoys=g.players[team].troops[:3],g.players[team].troops[3:]
            assert len(real)==len(decoys)==3 and all(legal(g,t) for t in real+decoys)
            assert all(t.hp==202 and t.dmg==125 for t in real)
            assert all(t.hp==81 and t.dmg==66 and t.proj_spd==0 for t in decoys)
            if abs(x-9)>4:
                assert all((t.x<9)==(x<9) for t in real)
                assert all((t.x<9)!=(x<9) for t in decoys)


def t_destroyed_tower_footprint_remains_available():
    g=Game();tower=g.arena.get_tower('red','princess','left');tower.take_damage(tower.hp)
    random.seed(42);expected=[(tower.cx+random.uniform(-1,1),tower.cy+random.uniform(-1,1)) for _ in range(3)]
    random.seed(42);barrel=create('goblin_barrel',11,'blue',tower.cx,tower.cy);barrel.apply(g)
    assert [(t.x,t.y) for t in g.players['blue'].troops]==expected
    assert all(legal(g,t) for t in g.players['blue'].troops)


def t_barrel_relocation_reserves_later_valid_points():
    g=Game();barrel=create('goblin_barrel',11,'blue',4.5,25.5)
    offsets=[0,0,1,0,1,0.2]
    with patch('sim.spells.random.uniform',side_effect=offsets):barrel.apply(g)
    troops=g.players['blue'].troops
    assert all(legal(g,t) for t in troops)
    assert [(t.x,t.y) for t in troops[1:]]==[(5.5,25.5),(5.5,25.7)]
    assert (int(troops[0].x),int(troops[0].y))!=(5,25)


def t_barrel_equal_distance_relocation_is_repeatable():
    results=[]
    for _ in range(2):
        g=Game();barrel=create('goblin_barrel',11,'blue',3.5,25.5)
        with patch('sim.spells.random.uniform',return_value=0):barrel.apply(g)
        troops=g.players['blue'].troops
        assert all(legal(g,t) for t in troops)
        assert len({(t.x,t.y) for t in troops})==3
        results.append([(t.x,t.y) for t in troops])
    assert results[0]==results[1]==[(3.5,23.5),(1.5,25.5),(5.5,25.5)]
