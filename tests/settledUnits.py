import random

from sim.cards import create
from sim.game import Game
from tests.util import quiet


# The walkable-tile rule of placement (a ground unit is never left standing on a tower footprint, a fence, open water or outside the
# arena) now covers units that spells and components add directly: Graveyard Skeletons, clones, death and ability spawns.


def blocked(g,t):
    # the engine's own walkability rule (footprints and fences block, water blocks off the bridges)
    return not g._walkable(t.x,t.y,False)


def t_graveyard_skeletons_never_stand_on_the_tower_footprint_or_water():
    # a Graveyard centred on the river drops part of its ring onto open water; the Skeletons are frozen so that only the settle rule,
    # not their first step to the bank, can take them off it. Spells tick before troops, so a birth is settled within its own tick
    random.seed(3);g=quiet(Game())
    spell=create('graveyard',11,'blue',9,16);spell.tcfg['spd']=0;spell.apply(g);g.spells.append(spell)
    seen=set()
    for _ in range(int(round(10.0/g.DT))):
        g.tick()
        for t in g.players['blue'].troops:
            if t.name=='Skeleton' and t.id not in seen:
                seen.add(t.id)
                assert not blocked(g,t),(t.x,t.y)
    assert len(seen)>=10


def t_clone_beside_a_tower_moves_off_the_footprint():
    g=quiet(Game());tw=g.arena.get_tower('red','princess','left')
    # the clone appears half a tile behind the original; a Knight just past the footprint's far edge puts that spot on the footprint
    knight=create('knight',11,'blue',tw.cx,tw.cy+1.8);knight.spd=0;g.deploy('blue',knight);g.tick()
    spell=create('clone',11,'blue',knight.x,knight.y);spell.apply(g)
    clone=next(t for t in g.players['blue'].troops if t is not knight)
    g.tick()
    assert clone.is_clone and not blocked(g,clone) and not blocked(g,knight)


def t_death_spawn_on_a_free_tile_stays_where_it_dies():
    g=quiet(Game());gs=create('goblin_gang',11,'blue',9,10)
    for t in gs:t.spd=0;g.deploy('blue',t)
    g.tick()
    assert all(getattr(t,'_settled',False) and not blocked(g,t) for t in gs)
    # an air death spawn is never moved: the pups appear around the hound's spot and stay within its spread
    ll=create('lava_hound',11,'blue',5,12);ll.spd=0;g.deploy('blue',ll);g.tick();ll.hp=0;ll.alive=False;g.tick();g.tick()
    pups=[t for t in g.players['blue'].troops if t.name=='LavaPups']
    assert len(pups)==6 and all(abs(t.x-5)<=2 and abs(t.y-12)<=2 and t._settled for t in pups)


def t_evolved_drill_resurfacing_on_a_tower_leaves_walking_goblins():
    # the evolved drill's resurface adds its Goblins directly around itself; born on the tower footprint they must be settled off it
    g=quiet(Game());tw=g.arena.get_tower('red','princess','left')
    gd=create('goblin_drill',11,'blue',tw.cx,tw.cy,evolved=True);g._place('blue',gd,1.0)
    for _ in range(int(round(6.0/g.DT))):g.tick()
    gd.hp=int(gd.max_hp*0.6);g.tick();g.tick()
    gobs=[t for t in g.players['blue'].troops if t.name=='Goblin' and t.alive]
    assert len(gobs)>=2 and all(not blocked(g,t) for t in gobs),[(round(t.x,2),round(t.y,2)) for t in gobs]
