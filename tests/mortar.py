import pytest

from sim.cards import create
from sim.game import Game
from sim.units import Status
from tests.util import Dummy,quiet


def setup(team='blue',level=11,evolved=False):
    g=Game()
    for tower in g.arena.towers:tower.alive=False
    mortar=create('mortar',level,team,9,10,evolved=evolved);g.deploy(team,mortar)
    near=Dummy(g._opp(team),9,12,hp=50000,spd=0,dmg=0)
    far=Dummy(g._opp(team),9,18,hp=50000,spd=0,dmg=0)
    g.deploy(near.team,near);g.deploy(far.team,far)
    return g,mortar,near,far


@pytest.mark.parametrize('team',['blue','red'])
@pytest.mark.parametrize('level',[11,16])
@pytest.mark.parametrize('evolved',[False,True])
def t_mortar_ignores_blind_spot_for_first_target(team,level,evolved):
    g,mortar,near,far=setup(team,level,evolved)
    assert g._dist(mortar,near)<mortar.min_rng<g._dist(mortar,far)<mortar.rng
    assert g._find_target(mortar)[0] is far
    g.run(3)
    assert far.hp<far.max_hp and near.hp==near.max_hp


@pytest.mark.parametrize('team',['blue','red'])
@pytest.mark.parametrize('evolved',[False,True])
def t_mortar_releases_target_that_enters_blind_spot(team,evolved):
    g,mortar,near,far=setup(team,evolved=evolved)
    near.y=16
    assert g._find_target(mortar)[0] is near
    near.y=12
    assert g._find_target(mortar)[0] is far
    g.run(3)
    assert far.hp<far.max_hp and near.hp==near.max_hp


@pytest.mark.parametrize('team',['blue','red'])
def t_mortar_blind_spot_only_waits_then_reacquires(team):
    g,mortar,near,far=setup(team)
    far.alive=False
    assert g._find_target(mortar)[0] is None
    g.run(2)
    assert not g.projs and mortar.cd==mortar.fhspd and near.hp==near.max_hp
    near.y=16
    assert g._find_target(mortar)[0] is near
    g.run(3)
    assert near.hp<near.max_hp


def t_mortar_minimum_range_boundary_uses_existing_attack_distance():
    g,mortar,near,far=setup()
    near.y=15
    mortar.min_rng=g._dist(mortar,near)
    assert g._find_target(mortar)[0] is near
    near.y-=0.001
    assert g._find_target(mortar)[0] is far
    near.y+=0.002;mortar.aggro_tgt=None
    assert g._find_target(mortar)[0] is near


@pytest.mark.parametrize('team',['blue','red'])
def t_mortar_excludes_blind_spot_tower_from_candidates_and_default(team):
    g,mortar,near,far=setup(team)
    near.alive=far.alive=False
    tower=g.arena.get_tower(g._opp(team),'princess','left');tower.alive=True
    mortar.x=tower.cx;mortar.y=tower.cy+2
    assert tower.dist(mortar.x,mortar.y)<mortar.min_rng
    assert g._default_target(mortar)[0] is None
    assert g._find_target(mortar)[0] is None
    mortar.retarget_cd=0.1
    assert g._find_target(mortar)[0] is None
    mortar.retarget_cd=0;mortar.y=tower.cy+7
    assert g._find_target(mortar)[0] is tower


@pytest.mark.parametrize('evolved',[False,True])
def t_mortar_deployment_ignores_near_giant_and_hits_tower(evolved):
    deck=['mortar','knight','archers','goblins','zap','fireball','bats','skeletons']
    g=quiet(Game(p1={'deck':deck,'drag_del':0,'drag_std':0}))
    g.players['blue'].deck.hand=['mortar','knight','archers','goblins']
    g.players['blue'].elixir=10
    assert g.play_card('blue','mortar',3,14,evolved=evolved)[0]
    giant=create('giant',11,'red',3,17);giant.spd=0;giant.dmg=0;g.deploy('red',giant)
    g.run_to(3.5)
    mortar=next(t for t in g.players['blue'].troops if t.name=='Mortar')
    tower=g.arena.get_tower('red','princess','left')
    assert g._dist(mortar,giant)<mortar.min_rng<tower.dist(mortar.x,mortar.y)<mortar.rng
    assert g._find_target(mortar)[0] is tower
    g.run_to(7)
    assert tower.hp<tower.max_hp


def t_mortar_keeps_eligible_lock_and_ignores_dead_air_hidden_alternatives():
    g,mortar,near,far=setup()
    near.y=16;mortar.aggro_tgt=far
    assert g._find_target(mortar)[0] is far
    for invalid in ('dead','air','invisible','burrowed'):
        mortar.aggro_tgt=None;near.alive=True;near.transport='Ground';near.statuses=[]
        if invalid=='dead':near.alive=False
        elif invalid=='air':near.transport='Air'
        else:near.statuses.append(Status(invalid,1))
        assert g._find_target(mortar)[0] is far


def t_mortar_explicit_taunt_override_is_unchanged():
    g,mortar,near,far=setup();mortar._taunt_target=near
    assert g._find_target(mortar)[0] is near
    g.run(3)
    assert near.hp==near.max_hp and far.hp==far.max_hp and not g.projs


@pytest.mark.parametrize('team',['blue','red'])
def t_mortar_releases_locked_tower_inside_blind_spot(team):
    g,mortar,near,far=setup(team)
    tower=g.arena.get_tower(g._opp(team),'princess','left');tower.alive=True
    mortar.x=tower.cx;mortar.y=tower.cy+2
    near.alive=False;far.x=mortar.x;far.y=mortar.y-8
    mortar.aggro_tgt=tower
    assert g._dist(mortar,tower)<mortar.min_rng
    assert g._find_target(mortar)[0] is far


@pytest.mark.parametrize('kind',['troop','tower'])
@pytest.mark.parametrize('delta',[-0.001,0,0.001])
def t_mortar_configured_minimum_range_threshold(kind,delta):
    g,mortar,near,far=setup();far.alive=False
    assert mortar.min_rng==3.5
    if kind=='troop':
        mortar.collision_r=near.collision_r=0.5
        near.y=mortar.y+mortar.min_rng+mortar.collision_r+near.collision_r+delta
        distance=g._dist(mortar,near);target=near
    else:
        near.alive=False;target=g.arena.get_tower('red','princess','left');target.alive=True
        mortar.x=target.cx;mortar.y=target.cy-mortar.min_rng-target.collision_r-delta
        distance=target.dist(mortar.x,mortar.y)
    assert abs(distance-(3.5+delta))<1e-12
    assert g._find_target(mortar)[0] is (None if delta<0 else target)


def t_mortar_released_projectile_can_still_hit_inside_blind_spot():
    g,mortar,near,far=setup();near.alive=False
    g.run_to(1)
    assert any(p.tgt is far for p in g.projs) and far.hp==far.max_hp
    far.y=12;g.run_to(2)
    assert far.max_hp-far.hp==mortar.dmg
    assert mortar.tgt is None and not g.projs


def t_mortar_splash_can_still_damage_an_unselected_blind_spot_enemy():
    g,mortar,near,far=setup();near.y=14;far.y=15.5
    assert g._dist(mortar,near)<mortar.min_rng<g._dist(mortar,far)
    assert g._find_target(mortar)[0] is far
    g.run_to(2.5)
    assert far.max_hp-far.hp==near.max_hp-near.hp==mortar.dmg
    assert mortar.tgt is far


@pytest.mark.parametrize('name',['cannon','tesla','x_bow','knight'])
def t_zero_minimum_range_selection_is_unchanged(name):
    g,mortar,near,far=setup()
    tr=create(name,11,'blue',9,10);g.players['blue'].troops=[tr]
    assert getattr(tr,'min_rng',0)==0
    assert g._find_target(tr)[0] is near
