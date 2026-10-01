from sim.cards import card,create
from sim.game import Game
from tests.util import Dummy,quiet


# Explosive Escape leaves a bomb where the miner stood, "dealing medium area damage to enemies around it after 1 second. His bomb affects
# both ground and air units and will also knock them back 1.8 tiles" (wiki Mighty Miner). The bomb is the building MightyMinerBomb:
# deployTime 1000 in the game data export and in RoyaleAPI's extraction, which also gives death_damage_radius 3000 beside collision_radius
# 450 (the 0.45 ClashStrategic lists) and death_push_back 1800.


def escape(x=14.5,y=12.0,bodies=(),troops=()):
    # a still miner in the right lane escapes to x = 18 - x; the towers are quiet, the bodies neither move nor attack and the enemy troops
    # stand still; returns the game, the miner, the bodies then the troops, and the escape time
    g=quiet(Game(p1={'ability_std':0}))
    mm=create('mighty_miner',11,'blue',x,y);mm.spd=0;g.deploy('blue',mm)
    out=[]
    for bx,by,air in bodies:
        d=Dummy('red',bx,by,hp=5000,spd=0,dmg=0);d.targets=[]
        if air:d.transport='Air'
        g.deploy('red',d);out.append(d)
    for name,tx,ty in troops:
        t=create(name,11,'red',tx,ty);t.spd=0;g.deploy('red',t);out.append(t)
    g.run(0.5)
    g.players['blue'].elixir=10;mm.ability.cd=0
    assert g.activate_ability('blue',mm)[0]
    while mm.x==x and g.t<5:g.tick()
    assert abs(mm.x-(18-x))<1e-9,mm.x
    return g,mm,out,g.t


def t_mighty_miner_bomb_record_carries_the_export_fuse_and_the_sourced_radius():
    c=card('mighty_miner');s=c['skills']['ability']['skills'];dd=s['areaDamageOnDeath'];src=c['src']
    assert dd['fuse']==1.0 and src['skills.ability.skills.areaDamageOnDeath.fuse'].startswith('gd:'),(dd,src)
    assert dd['radius']==3.0 and src['skills.ability.skills.areaDamageOnDeath.radius']=='patch:2026-10-01',(dd,src)
    assert s['pushback']['distance']==1.8 and src['skills.ability.skills.pushback.distance']=='patch:2026-10-01',(s['pushback'],src)
    ab=create('mighty_miner',11,'blue',9,10).ability
    assert (ab.fuse,ab.bomb_r,ab.bomb_dmg,ab.kb)==(1.0,3.0,332,1.8),(ab.fuse,ab.bomb_r,ab.bomb_dmg,ab.kb)


def t_escape_bomb_goes_off_one_second_after_the_escape():
    # an air body over the miner's spot: the miner cannot hit it, so only the bomb can
    g,mm,(a,),te=escape(bodies=[(14.5,12.0,True)])
    assert a.hp==5000,f"The bomb should not go off in the escape tick, hp {a.hp}"
    while a.hp==5000 and g.t<te+2:g.tick()
    assert round(g.t-te,2)==1.0 and a.hp==5000-332,(round(g.t-te,2),a.hp)


def t_escape_bomb_reaches_three_tiles_around_the_spot_on_air_and_ground():
    # air bodies 2.5 and 4 tiles from the spot and a ground body 2.7 tiles out, beyond the miner's 1.6 reach (edge to edge 1.7), so each
    # loses exactly the bomb or nothing
    g,mm,(near,far,gnd),te=escape(bodies=[(14.5,9.5,True),(14.5,8.0,True),(17.2,12.0,False)])
    g.run_to(te+1.05)
    assert near.hp==5000-332 and gnd.hp==5000-332,f"Bodies within 3 tiles should take the bomb: air {near.hp}, ground {gnd.hp}"
    assert far.hp==5000,f"A body 4 tiles out is beyond the blast, hp {far.hp}"


def t_escape_bomb_knocks_bodies_back_and_spares_the_immune():
    # a light ground body 2.7 tiles out and an air body 2 tiles out are pushed 1.8 tiles away from the spot; the Monk (knockback immune),
    # the P.E.K.K.A (mass 10) and a Cannon (building) take the bomb where they stand
    g,mm,(gnd,air,monk,pk,can),te=escape(bodies=[(11.8,12.0,False),(14.5,10.0,True)],
                                         troops=[('monk',13.0,10.5),('pekka',16.5,10.5),('cannon',16.5,13.5)])
    st=[(u.x,u.y,u.hp) for u in (monk,pk,can)]
    g.run_to(te+1.05)
    assert abs(gnd.x-10.0)<0.05 and abs(gnd.y-12.0)<0.05,f"Ground body should be pushed to x 10.0, at ({gnd.x:.2f}, {gnd.y:.2f})"
    assert abs(air.x-14.5)<0.05 and abs(air.y-8.2)<0.05,f"Air body should be pushed to y 8.2, at ({air.x:.2f}, {air.y:.2f})"
    for u,(x,y,hp) in zip((monk,pk,can),st):
        assert abs(u.x-x)<0.05 and abs(u.y-y)<0.05,f"{u.name} should stand, moved from ({x:.2f}, {y:.2f}) to ({u.x:.2f}, {u.y:.2f})"
        assert hp-u.hp>=332,f"{u.name} should be inside the blast, lost {hp-u.hp}"


def t_escape_bomb_outlives_the_miner():
    # the bomb is a separate object at the old spot: it still goes off when the miner dies during the fuse
    g,mm,(a,),te=escape(bodies=[(14.5,12.0,True)])
    mm.hp=0;mm.alive=False
    g.run_to(te+1.05)
    assert a.hp==5000-332,a.hp
