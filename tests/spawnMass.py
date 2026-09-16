import pytest

from sim.cards import create
from sim.game import Game
from sim.fx import push
from tests.util import Dummy


def setup():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('mass',(1,2.5))
def t_skeleton_descendants_inherit_configured_parent_mass(team,mass):
    g=setup();roots=create('skeletons',11,team,9,10,evolved=True)
    for t in roots:g.deploy(team,t)
    parent=roots[0];parent.mass=mass
    target=Dummy(g._opp(team),9,11,hp=100000,spd=0,dmg=0);g.deploy(target.team,target)
    for _ in range(3):
        old=list(g.players[team].troops);g._do_attack(parent,target)
        child=next(t for t in g.players[team].troops if t not in old)
        assert child.mass==mass
        parent=child


@pytest.mark.parametrize('name',('skeletons','goblins','giant','pekka','royal_ghost'))
@pytest.mark.parametrize('team',('blue','red'))
def t_clones_preserve_parent_collision_attributes(name,team):
    g=setup();group=create(name,11,team,9,10);original=group[0] if isinstance(group,list) else group
    g.deploy(team,original)
    create('clone',11,team,9,10).apply(g)
    clone=next(t for t in g.players[team].troops if t is not original)
    assert (clone.mass,clone.collision_r,clone.sight_r)==(original.mass,original.collision_r,original.sight_r)
    assert clone.hp==clone.max_hp==1 and original.hp==original.max_hp


def collision_step(unit):
    g=setup();unit.x,unit.y=9,10;unit.id=1
    partner=create('knight',11,g._opp(unit.team),9.6,10);partner.id=2
    g.players[unit.team].troops=[unit];g.players[partner.team].troops=[partner]
    g._resolve_collisions()
    return (unit.x,unit.y,partner.x,partner.y)


def t_descendant_collision_matches_factory_character():
    g=setup();roots=create('skeletons',11,'blue',9,10,evolved=True)
    for t in roots:g.deploy('blue',t)
    target=Dummy('red',9,11,hp=100000,spd=0,dmg=0);g.deploy('red',target);g._do_attack(roots[0],target)
    child=next(t for t in g.players['blue'].troops if t not in roots)
    reference=create('skeletons',11,'blue',9,10,evolved=True)[0]
    assert collision_step(child)==pytest.approx(collision_step(reference))


@pytest.mark.parametrize('team',('blue','red'))
def t_clone_collision_matches_original_character(team):
    g=setup();original=create('giant',11,team,9,10);g.deploy(team,original)
    create('clone',11,team,9,10).apply(g);clone=next(t for t in g.players[team].troops if t is not original)
    reference=create('giant',11,team,9,10)
    assert collision_step(clone)==pytest.approx(collision_step(reference))


@pytest.mark.parametrize('name',('giant','pekka'))
def t_heavy_clone_preserves_parent_knockback_response(name):
    g=setup();original=create(name,11,'blue',9,10);g.deploy('blue',original)
    create('clone',11,'blue',9,10).apply(g);clone=next(t for t in g.players['blue'].troops if t is not original)
    clone.x,clone.y=original.x,original.y
    before=(original.x,original.y)
    for t in (original,clone):push(t,8,10,1)
    assert (clone.x,clone.y)==(original.x,original.y)==before


def t_cloned_evolved_skeleton_descendant_keeps_source_mass():
    g=setup();root=create('skeletons',11,'blue',9,10,evolved=True)[0];g.deploy('blue',root)
    create('clone',11,'blue',9,10).apply(g);clone=next(t for t in g.players['blue'].troops if t is not root)
    target=Dummy('red',10,10,hp=100000,spd=0,dmg=0);g.deploy('red',target)
    before=list(g.players['blue'].troops);g._do_attack(clone,target)
    child=next(t for t in g.players['blue'].troops if t not in before)
    assert root.mass==clone.mass==child.mass

