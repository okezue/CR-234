import math

from sim.cards import create, load
from sim.game import Game
from tests.util import quiet


# Game data export debfe1e2, MotherWitch projectile VoodooCurse deathSpawnData VoodooHog: hitpoints 246 (the Witch's own record 207),
# damage 21, hitSpeed 1200, loadTime 950, deployTime 200, collisionRadius 600, range 750, speed 120. On the card curve 246 is 629 at
# level 11, the wiki Mother Witch page's Cursed Hog Hitpoints; the ClashStrategic snapshot had copied the Witch's 529.


def hog_from(g,team='blue'):
    witch=create('mother_witch',11,team,9,10);g.deploy(team,witch)
    victim=create('knight',11,g._opp(team),9,11.5);victim.spd=0;victim.dmg=0;victim.hp=1;g.deploy(victim.team,victim)
    for _ in range(int(round(3.0/g.DT))):
        g.tick()
        hogs=[t for t in g.players[team].troops if t.name=='Cursed Hog']
        if hogs:return witch,hogs[0]
    raise AssertionError('no hog')


def t_cursed_hog_record_is_sourced_on_the_patch():
    c=load()['cards']['mother_witch'];sk=c['skills']['spawnOnKill'];u=c['units']['cursed_hog']
    assert sk['hitpoints'][10]==629 and sk['hitpoints'][15]==1006 and sk['damage'][10]==53
    assert (sk['hitSpeed'],sk['loadTime'],sk['deployTime'],sk['collisionRadius'],sk['range'],sk['speed'])==(1.2,0.95,0.2,0.6,0.75,120)
    assert (u['hitSpeed'],u['loadTime'],u['hitpoints'][10])==(1.2,0.95,629)
    assert c['src']['skills.spawnOnKill.hitpoints']=='patch:2026-09-17g' and c['stats']['hitpoints'][10]==529


def t_spawned_hog_carries_the_export_values_and_the_witch_keeps_hers():
    g=quiet(Game());witch,hog=hog_from(g)
    assert (hog.hp,hog.max_hp,hog.dmg)==(629,629,53) and witch.max_hp==529
    assert (hog.hspd,round(hog.fhspd,2),hog.collision_r,hog.rng)==(1.2,0.25,0.6,0.75)
    assert hog.targets==['Buildings'] and round(hog.spd*50)==120


def t_hog_stands_for_its_deploy_time_then_runs():
    g=quiet(Game());witch,hog=hog_from(g);x0,y0=hog.x,hog.y
    assert any(s.kind=='deploying' for s in hog.statuses)
    g.run(0.2)
    # the Witch walking past may nudge it by collision, but it does not run (2.4 tiles per second) until the deploy ends
    assert not any(s.kind=='deploying' for s in hog.statuses) and math.dist((hog.x,hog.y),(x0,y0))<0.2
    g.run(0.5)
    assert math.dist((hog.x,hog.y),(x0,y0))>0.8


def t_hog_born_on_a_tower_footprint_is_placed_beside_it():
    # the cursed troop dies against the tower it attacks; the hog appears where it died, which may be the footprint edge
    g=quiet(Game());witch=create('mother_witch',11,'blue',9,10);g.deploy('blue',witch)
    tw=g.arena.get_tower('red','princess','left')
    victim=create('knight',11,'red',tw.cx,tw.cy);victim.spd=0;victim.dmg=0;victim.hp=1;g.deploy('red',victim)
    # the victim is pinned on the footprint for the test (the settle rule would otherwise move it off) so the hog is born there
    victim._settled=True
    # the Witch stands off the tower's axis so the Knight on the tower centre is her nearest target by a clear margin
    witch.x,witch.y=tw.cx+2.5,tw.cy-3
    for _ in range(int(round(3.0/g.DT))):
        g.tick()
        hogs=[t for t in g.players['blue'].troops if t.name=='Cursed Hog']
        if hogs:break
    hog=hogs[0]
    assert g.arena.blocked(int(victim.x),int(victim.y)) and not g.arena.blocked(int(hog.x),int(hog.y))


def t_hog_jumps_the_river_like_the_royal_hogs():
    # game data VoodooHog jumpHeight 4000 and jumpSpeed 160, the Royal Hogs' river jump
    from sim.fx import RiverJump
    g=quiet(Game());witch,hog=hog_from(g)
    assert any(isinstance(c,RiverJump) for c in hog.components)
    hog.x,hog.y=6.5,14.5
    for _ in range(int(round(2.0/g.DT))):g.tick()
    # from a tile before the river and away from both bridges it crosses straight over instead of walking to a bridge
    assert hog.y>17.5 and abs(hog.x-6.5)<1.5
