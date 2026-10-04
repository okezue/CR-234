from sim import fx
from sim.cards import create
from sim.game import Game
from sim.units import has
from tests.util import Dummy,quiet


# The Ice Golem's death nova reaches a body it touches, like the death bombs, spells and splash (fx.tdist: the troop's collision circle;
# wiki Bomb Tower: its blast "can hit troops whose hitboxes are touching the Crown Towers"). The nova measured centres before, so a body
# whose edge lay inside the 2 tile radius escaped (09PP9JRPYR9Y: the nova missed the Mighty Miner by 0.21 tile, 1.71 to his edge).


def died(bodies):
    g=quiet(Game());ig=create('ice_golem',11,'blue',9,10);g.deploy('blue',ig)
    ds=[]
    for x,y,r in bodies:
        d=Dummy('red',x,y,hp=5000,spd=0,dmg=0);d.collision_r=r;g.deploy('red',d);ds.append(d)
    ig.hp=0;ig.alive=False;ig.on_death(g)
    return ds


def t_ice_golem_nova_reaches_a_body_whose_edge_is_inside_the_radius():
    # 2.3 tiles centre to centre is 1.8 to a 0.5 radius body's edge; 2.6 is 2.1, outside; a 1.0 radius body at 2.9 is 1.9, inside
    inside,outside,big=died([(9,12.3,0.5),(11.6,10,0.5),(9,7.1,1.0)])
    assert inside.hp==5000-84 and big.hp==5000-84,f"Bodies touching the 2 tile nova should take 84, {inside.hp} {big.hp}"
    assert has(inside,'slow') and has(big,'slow'),"and be slowed"
    assert outside.hp==5000 and not has(outside,'slow'),f"A body 2.1 tiles from its edge is outside, {outside.hp}"


def t_ice_golem_nova_and_death_bombs_share_the_reach():
    # the nova reaches exactly the bodies fx.near gives for its radius, the rule DeathDamage uses
    g=quiet(Game());ig=create('ice_golem',11,'blue',9,10);g.deploy('blue',ig)
    ds=[Dummy('red',9+dx,10+dy,hp=5000,spd=0,dmg=0) for dx,dy in ((2.4,0),(0,2.6),(1.8,1.6),(-1.9,-1.7))]
    for d in ds:g.deploy('red',d)
    want={id(e) for e in fx.near(g,'blue',ig.x,ig.y,ig.death_splash_r,towers=False)}
    ig.hp=0;ig.alive=False;ig.on_death(g)
    assert {id(d) for d in ds if d.hp<5000}==want and want,want
