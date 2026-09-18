import math

from sim.cards import create, load
from sim.fx import Burrow
from sim.game import Game
from sim.units import has
from tests.util import quiet


# Wiki Goblin Drill: burrowing speed 300 since the 7/6/2022 balance update (from 400); the first Goblin comes 1 s after deployment
# (0.8 s from 14/5/2024, back to 1 s on 6/10/2025). Game data export: the surfaced building (spawnPathfindMorphData) deploys for
# 1000 ms. In the 5 September 2026 LCQ recording (game 09YP9UPGQ2YU) the drill placed at 96.85 s shows its trail from the King Tower
# at 97.5, reaches a tower about 21 tiles away at 100.5 and shows its deploy clock there until 101.5. Speed 300 in the engine's
# convention is 300 / 50 = 6 tiles/s.


def surfaced_at(g,unit,limit):
    while g.t<limit:
        g.tick()
        if not has(unit,'burrowed'):return round(g.t,2)
    return None


def t_drill_record_is_sourced_on_the_patch():
    c=load()['cards']['goblin_drill']
    assert c['skills']['burrow']['speed']==300 and c['skills']['burrow']['surfaceDeploy']==1.0 and c['skills']['periodicSpawn']['firstDelay']==1.0
    assert c['src']['skills.burrow.speed']=='patch:2026-09-17h' and c['src']['skills.burrow.surfaceDeploy']=='patch:2026-09-17h'
    assert c['src']['skills.periodicSpawn.firstDelay']=='legacy:spawn'
    m=load()['cards']['miner']['skills']['burrow']
    assert m['speed']==650 and 'surfaceDeploy' not in m


def t_drill_travels_at_six_tiles_per_second_then_deploys_for_a_second():
    # from the blue King Tower at (9, 3) to a tower placement at (14.5, 25.5): 23.2 tiles, 3.85 s of travel
    g=quiet(Game());gd=create('goblin_drill',11,'blue',14.5,25.5);g._place('blue',gd,1.0)
    travel=math.hypot(14.5-9,25.5-3)/6.0
    assert has(gd,'burrowed') and not has(gd,'deploying') and (gd.x,gd.y)==(9.0,3.0)
    ts=surfaced_at(g,gd,10)
    assert abs(ts-travel)<=g.DT+1e-9,(ts,travel)
    assert has(gd,'deploying') and abs(gd.x-14.5)<0.05 and abs(gd.y-25.5)<0.05
    g.run(1.0)
    assert not has(gd,'deploying')


def t_drill_is_untouchable_underground_and_targetable_while_surfacing():
    g=quiet(Game());gd=create('goblin_drill',11,'blue',14.5,24.0);g._place('blue',gd,1.0)
    tw=g.arena.get_tower('red','princess','right')
    for _ in range(int(round(2.0/g.DT))):
        g.tick()
        assert getattr(tw.troop,'lock',None) is None
    surfaced_at(g,gd,10)
    assert gd.hp==gd.max_hp
    g.run(0.5)
    # the tower locks the deploying drill (it stands on the field) while the lifetime drain waits for the deploy to end
    assert tw.troop.lock is gd and has(gd,'deploying') and gd.hp==gd.max_hp
    g.run(0.6)
    assert not has(gd,'deploying') and gd.hp<gd.max_hp


def t_first_goblin_comes_one_second_after_the_surfacing_deploy():
    g=quiet(Game());gd=create('goblin_drill',11,'blue',9,15);g._place('blue',gd,1.0)
    ts=surfaced_at(g,gd,10);births=[]
    for _ in range(int(round(6.0/g.DT))):
        g.tick()
        for t in g.players['blue'].troops:
            if t.name=='Goblin' and t.id not in [b[1] for b in births]:births.append((round(g.t,2),t.id))
    first=births[0][0]
    # the spawn timer runs from the tick that ends the surfacing deploy (the engine's timer convention, as for the Furnace), so the
    # first Goblin is born one tick short of arrival + 1 s deploy + 1 s first delay
    assert abs(first-(ts+1.0+1.0-g.DT))<=g.DT/2+1e-9,(ts,first)
    assert abs(births[1][0]-first-3.0)<=g.DT/2+1e-9,births[:2]


def t_miner_rule_is_unchanged_and_replay_placement_adds_no_second_deploy():
    # the Miner's single deploy is folded into the travel: max(1.0, distance / 13)
    g=quiet(Game());mn=create('miner',11,'blue',9,25);g._place('blue',mn,1.0)
    assert has(mn,'burrowed') and not has(mn,'deploying')
    ts=surfaced_at(g,mn,5)
    assert abs(ts-max(1.0,22.0/13.0))<=g.DT+1e-9,ts
    assert not has(mn,'deploying') and abs(mn.y-25)<0.1
    g2=quiet(Game());mn2=create('miner',11,'blue',9,5);g2._place('blue',mn2,1.0)
    assert abs(surfaced_at(g2,mn2,5)-1.0)<=g2.DT+1e-9


def t_evolved_drill_keeps_the_travel_speed():
    gd=create('goblin_drill',11,'blue',9,20,evolved=True)
    b=next(c for c in gd.components if isinstance(c,Burrow))
    assert b.spd==6.0 and b.surface==1.0


def t_drill_near_its_king_still_takes_its_burrowing_deploy_before_surfacing():
    # both export deploy fields apply: the burrowing form's 1 s overlaps the travel (0.5 s for 3 tiles), then the surfaced form deploys
    g=quiet(Game());gd=create('goblin_drill',11,'blue',9,6);g._place('blue',gd,1.0)
    ts=surfaced_at(g,gd,5)
    assert abs(ts-1.0)<=g.DT/2+1e-9 and has(gd,'deploying'),ts
    g.run(1.0)
    assert not has(gd,'deploying')
