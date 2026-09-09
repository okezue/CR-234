import math

from sim.cards import create
from sim.fx import Charge,Recoil
from sim.game import Game
from sim.units import Status,has
from tests.util import Dummy,quiet


def setup(team='blue',evolved=False):
    g=Game()
    for tower in g.arena.towers:tower.alive=False
    tr=create('firecracker',11,team,9,10,evolved=evolved);g.deploy(team,tr)
    target=Dummy(g._opp(team),9,16 if team=='blue' else 4,hp=5000,spd=0,dmg=0)
    g.deploy(target.team,target)
    return g,tr,target


def t_firecracker_recoils_at_launch_not_impact():
    for team in ('blue','red'):
        for evolved in (False,True):
            g,tr,target=setup(team,evolved)
            origin=(tr.x,tr.y);expected_y=9 if team=='blue' else 11
            g._fire(tr,target)
            assert (tr.x,tr.y)==(9,expected_y) and target.hp==5000
            assert len(g.projs)==1 and (g.projs[0].x,g.projs[0].y)==origin
            for _ in range(11):g._proc_projs();assert target.hp==5000
            g._proc_projs()
            assert target.hp==5000-320 and not g.projs and (tr.x,tr.y)==(9,expected_y)
            for _ in range(3):g._proc_projs()
            assert target.hp==5000-320 and (tr.x,tr.y)==(9,expected_y)


def t_recoil_uses_launch_direction_even_when_target_moves():
    g,tr,target=setup();g._fire(tr,target)
    assert (tr.x,tr.y)==(9,9)
    target.x=12
    for _ in range(30):g._proc_projs()
    assert target.hp==5000-320 and (tr.x,tr.y)==(9,9)


def t_recoil_occurs_when_nonhoming_projectile_misses():
    g,tr,target=setup();tr.proj_homing=False;g._fire(tr,target)
    assert (tr.x,tr.y)==(9,9)
    target.x=15
    for _ in range(20):g._proc_projs()
    assert target.hp==5000 and not g.projs and (tr.x,tr.y)==(9,9)


def t_post_launch_status_or_death_does_not_repeat_recoil():
    for kind in ('stun','freeze','death'):
        g,tr,target=setup();g._fire(tr,target)
        assert (tr.x,tr.y)==(9,9)
        if kind=='death':tr.take_damage(tr.hp);g._proc_deaths()
        else:tr.statuses.append(Status(kind,2))
        for _ in range(20):g._proc_projs()
        assert target.hp==5000-320 and (tr.x,tr.y)==(9,9)
        assert tr.alive==(kind!='death')


def t_recoil_waits_for_actual_attack_release():
    for kind in ('deploying','stun','freeze','knockback'):
        g,tr,target=setup();tr.cd=0;tr.statuses.append(Status(kind,0.5))
        for _ in range(5):g._proc_troops()
        assert (tr.x,tr.y)==(9,10) and not g.projs and target.hp==5000
    g,tr,target=setup();tr.cd=0.2
    for _ in range(3):g._proc_troops();assert not g.projs and (tr.x,tr.y)==(9,10)
    g._proc_troops()
    assert g.projs and (tr.x,tr.y)==(9,9) and target.hp==5000


def t_recoil_does_not_move_damage_effects_to_launch():
    g,tr,target=setup();behind=Dummy('red',9,18,hp=5000,spd=0,dmg=0);g.deploy('red',behind)
    g._fire(tr,target)
    assert target.hp==behind.hp==5000
    for _ in range(12):g._proc_projs()
    assert target.hp==5000-320 and 5000-320<behind.hp<5000
    g=quiet(Game());tr=create('firecracker',11,'blue',3.5,19.5);g.deploy('blue',tr)
    tower=g.arena.get_tower('red','princess','left');hp=tower.hp;g._fire(tr,tower)
    assert tr.y==18.5 and tower.hp==hp
    for _ in range(12):g._proc_projs()
    assert tower.hp==hp-320


def t_target_death_during_flight_does_not_repeat_recoil():
    g,tr,target=setup();g._fire(tr,target)
    target.take_damage(target.hp);g._proc_deaths()
    assert target not in g.players[target.team].troops and (tr.x,tr.y)==(9,9)
    for _ in range(20):g._proc_projs()
    assert not g.projs and (tr.x,tr.y)==(9,9)


def t_evolved_firecracker_slow_remains_impact_time():
    g,tr,target=setup(evolved=True)
    nearby=Dummy('red',9,12,hp=5000,spd=0,dmg=0);g.deploy('red',nearby)
    g._fire(tr,target)
    assert (tr.x,tr.y)==(9,9) and not has(nearby,'slow')
    for _ in range(11):g._proc_projs();assert not has(nearby,'slow')
    g._proc_projs()
    assert has(nearby,'slow') and (tr.x,tr.y)==(9,9)


def t_melee_battle_ram_recoil_keeps_damage_and_charge_order():
    g=quiet(Game());tr=create('battle_ram',11,'blue',3.5,24,evolved=True);g.deploy('blue',tr)
    tower=g.arena.get_tower('red','princess','left');hp=tower.hp;origin=(tr.x,tr.y)
    charge=next(c for c in tr.components if isinstance(c,Charge))
    recoil=next(c for c in tr.components if isinstance(c,Recoil))
    g._fire(tr,tower)
    assert tr.alive and not g.projs and tower.hp==hp-tr.dmg
    assert math.isclose(math.dist(origin,(tr.x,tr.y)),recoil.dist)
    assert charge.moved==charge.dist


def t_cloned_firecracker_recoils_independently_at_launch():
    g,tr,target=setup();create('clone',11,'blue',9,10).apply(g)
    clone=next(t for t in g.players['blue'].troops if t is not tr);origin=(clone.x,clone.y)
    g._fire(clone,target)
    assert (tr.x,tr.y)==(9,10) and (clone.x,clone.y)==(origin[0],origin[1]-1)
    assert (g.projs[0].x,g.projs[0].y)==origin and target.hp==5000
    for _ in range(20):g._proc_projs()
    assert (tr.x,tr.y)==(9,10) and (clone.x,clone.y)==(origin[0],origin[1]-1)
    assert target.hp==5000-320
