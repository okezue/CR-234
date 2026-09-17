import pytest

from sim.cards import create, load
from sim.game import Game
from tests.util import quiet


# Game data export: the reworked Furnace's spawn is its onStartingActionData (Furnace_rework_continuous_spawn, interval 5000), an action
# that starts with the character; in the 5 September 2026 LCQ recording a Fire Spirit stands beside the Furnace the frame after its
# deploy ends (placed 122.10 s, spirit visible 124.25 s) and the next one five seconds later. The engine waited a full interval first.


def spirits(g,team):
    return [t for t in g.players[team].troops if t.name=='Fire Spirit']


def t_first_delay_is_sourced_on_the_patch():
    c=load()['cards']['furnace']['skills']['periodicSpawn']
    assert c['firstDelay']==0 and c['pauseTime']==5
    assert load()['cards']['furnace']['src']['skills.periodicSpawn.firstDelay']=='patch:2026-09-17f'


@pytest.mark.parametrize('evolved',(False,True))
def t_first_spirit_appears_as_soon_as_the_furnace_has_deployed(evolved):
    g=quiet(Game());furnace=create('furnace',11,'blue',9,10,evolved=evolved);g._place('blue',furnace,1.0)
    seen=None
    for _ in range(int(round(1.5/g.DT))):
        g.tick()
        if seen is None and spirits(g,'blue'):seen=round(g.t,2)
    # nothing during the Furnace's own one second deploy, a spirit on the tick that ends it
    assert seen==pytest.approx(1.0,abs=g.DT/2),seen
    s=spirits(g,'blue')[0]
    assert any(st.kind=='deploying' for st in s.statuses)


def t_second_spirit_follows_five_seconds_after_the_first():
    g=quiet(Game());furnace=create('furnace',11,'blue',9,10);g.deploy('blue',furnace)
    births=[];seen=set()
    for _ in range(int(round(11.0/g.DT))):
        g.tick()
        for s in spirits(g,'blue'):
            if s.id not in seen:seen.add(s.id);births.append(round(g.t,2))
    # spirits that have already jumped at the tower are gone by the next spawn, so the births are tracked by unit id
    assert births==pytest.approx([0.05,5.05,10.05],abs=g.DT/2),births


def t_goblin_hut_first_delay_unchanged():
    # the hut sleeps until an enemy is within its spawn range and then waits its own 0.5 s (August 2025 balance change)
    c=load()['cards']['goblin_hut']['skills']['periodicSpawn']
    assert c['firstDelay']==0.5 and c['range']==6
    g=quiet(Game());hut=create('goblin_hut',11,'blue',9,8);g.deploy('blue',hut)
    for t in g.arena.towers:t.alive=False
    g.run(1.5)
    assert not [t for t in g.players['blue'].troops if t is not hut]


def t_spirit_spawned_behind_the_king_tower_is_placed_on_a_walkable_tile():
    # a Furnace at the back spawns forward onto the King Tower footprint; the spirit must not be born inside the tower (it could never walk)
    g=quiet(Game());furnace=create('furnace',11,'blue',9.5,0.5);g.deploy('blue',furnace)
    g.run(0.1)
    s=spirits(g,'blue')[0]
    assert not g.arena.blocked(int(s.x),int(s.y))
    g.run(3.0)
    # it walks toward the enemy side once its own deploy is over
    assert s.y>2.0 or not s.alive


def t_free_spot_leaves_air_units_and_valid_ground_placements_alone():
    g=quiet(Game());bat=create('bats',11,'blue',9,3)[0];bp=(bat.x,bat.y)
    assert g.arena.blocked(int(bat.x),int(bat.y))
    g._place('blue',bat,0)
    knight=create('knight',11,'blue',5,8);g._place('blue',knight,0)
    assert (bat.x,bat.y)==bp and (knight.x,knight.y)==(5,8)
