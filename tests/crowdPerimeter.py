import statistics

from sim.cards import create
from sim.game import Game
from tests.util import quiet


# Swarms surround their target in the game (a Skeleton Army engulfs a P.E.K.K.A). With the export's collision radii (Skeleton 500,
# Knight 500, Giant and P.E.K.K.A 750) and the Skeletons' 0.5 melee range, the ring in range of a Knight has circumference 2*pi*1.5
# and room for nine Skeletons of diameter one, and the ring around a P.E.K.K.A room for ten. The engine's movement walked every
# body straight at the target, so late arrivals stalled behind the first and never reached the ring; a stalled body now steps along
# the ring away from the body ahead of it, and the collision separation of a stalled body slides it along the ring too. The rule
# checked here: a swarm larger than its ring fills at least half of the ring's capacity, and a group that fits engages entirely.


def engaged(card,target,seconds=8.0,after=3.0):
    g=quiet(Game());k=create(target,11,'red',9,12.5);k.spd=0;k.dmg=0;k.hp=k.max_hp=99999;g.deploy('red',k)
    us=create(card,11,'blue',9,10.5)
    for u in us:g.deploy('blue',u)
    counts=[]
    for _ in range(int(round(seconds/g.DT))):
        g.tick()
        if g.t>after:counts.append(sum(1 for u in us if g._dist(u,k)<=u.rng+1e-9))
    return len(us),statistics.mean(counts)


def ring_capacity(target_r,body_r,rng):
    import math
    return int(2*math.pi*(target_r+body_r+rng)//(2*body_r))


def t_skeleton_army_fills_half_the_ring_around_a_knight():
    n,mean=engaged('skeleton_army','knight');cap=ring_capacity(0.5,0.5,0.5)
    assert n==15 and cap==9 and mean>=cap/2,(mean,cap)


def t_skeleton_army_fills_half_the_ring_around_a_pekka():
    n,mean=engaged('skeleton_army','pekka');cap=ring_capacity(0.75,0.5,0.5)
    assert n==15 and cap==10 and mean>=cap/2,(mean,cap)


def t_barbarians_all_reach_a_knight():
    # five Barbarians fit the ring around a Knight (capacity ten), so nearly all of them engage
    n,mean=engaged('barbarians','knight')
    assert n==5 and mean>=4.5,mean


def t_small_groups_are_unaffected():
    # three Skeletons and a Goblin Gang already fitted; they still all engage
    assert engaged('skeletons','knight')[1]>=2.9
    assert engaged('goblin_gang','knight')[1]>=5.9


def t_buildings_never_slide():
    g=quiet(Game());tomb=create('tombstone',11,'red',9,14);g.deploy('red',tomb);pos=(tomb.x,tomb.y)
    for u in create('barbarians',11,'blue',9,12.5):g.deploy('blue',u)
    g.run(4.0)
    assert (tomb.x,tomb.y)==pos


def t_no_body_sinks_into_the_target():
    # the target's own push is never redirected, so no Skeleton ends up overlapping the P.E.K.K.A's body
    import math
    g=quiet(Game());k=create('pekka',11,'red',9,12.5);k.spd=0;k.dmg=0;k.hp=k.max_hp=99999;g.deploy('red',k)
    us=create('skeleton_army',11,'blue',9,10.5)
    for u in us:g.deploy('blue',u)
    worst=0.0
    for _ in range(int(round(8.0/g.DT))):
        g.tick()
        if g.t>3.0:worst=max(worst,max(k.collision_r+u.collision_r-math.hypot(u.x-k.x,u.y-k.y) for u in us))
    assert worst<0.2,worst
