import pytest

from sim.cards import create
from sim.fx import EvoRoyalGhost,Fade,Stealth
from sim.game import Game
from sim.units import has
from tests.util import Dummy


def arena():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


def duplicate(g,original):
    before=set(g.players[original.team].troops)
    create('clone',original.lvl,original.team,original.x,original.y).apply(g)
    return next(t for t in g.players[original.team].troops if t not in before)


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('level',(11,16))
def t_cloned_evolved_ghost_is_base_and_cannot_summon(team,level):
    g=arena();ghost=create('royal_ghost',level,team,9,10,evolved=True);g.deploy(team,ghost);g.tick()
    clone=duplicate(g,ghost)
    assert not any(isinstance(c,EvoRoyalGhost) for c in clone.components)
    assert not getattr(clone,'evolved',False) and clone.hp==clone.max_hp==1
    assert clone.lvl==ghost.lvl==level and clone.dmg==ghost.dmg and clone.hovering and clone.ability is None
    ghost.x,ghost.y=1,1;clone.spd=0;g.tick()
    target=Dummy(g._opp(team),10,10,hp=50000,spd=0,dmg=0);g.deploy(target.team,target)
    g._do_attack(clone,target);g.tick()
    assert not any(t.name=='Souldier' for t in g.players[team].troops)
    assert has(ghost,'invisible') and not has(clone,'invisible')
    ghost.x,ghost.y=9,10;g._do_attack(ghost,target);g.tick()
    assert len([t for t in g.players[team].troops if t.name=='Souldier'])==2


@pytest.mark.parametrize('evolved',(False,True))
@pytest.mark.parametrize('attacker',('original','clone'))
def t_ghost_attacks_do_not_change_other_stealth_timer(evolved,attacker):
    g=arena();ghost=create('royal_ghost',11,'blue',9,10,evolved=evolved);g.deploy('blue',ghost);g.tick()
    clone=duplicate(g,ghost)
    parent_timer=next(c for c in ghost.components if isinstance(c,Stealth))
    clone_timer=next(c for c in clone.components if isinstance(c,Stealth))
    assert parent_timer is not clone_timer
    a,b=(ghost,clone) if attacker=='original' else (clone,ghost)
    b.x,b.y=1,1;a.spd=b.spd=0;g.tick()
    state=next(c for c in b.components if isinstance(c,Stealth));idle=state.idle
    target=Dummy('red',10,10,hp=50000,spd=0,dmg=0);g.deploy('red',target)
    g._do_attack(a,target)
    assert state.idle==idle and has(b,'invisible')
    for _ in range(5):g.tick()
    assert has(b,'invisible')


def t_cloned_ghost_has_its_own_initial_stealth_state():
    g=arena();ghost=create('royal_ghost',11,'blue',9,10,evolved=True);g.deploy('blue',ghost)
    timer=next(c for c in ghost.components if isinstance(c,Stealth));timer.idle=0.4
    clone=duplicate(g,ghost);copy=next(c for c in clone.components if isinstance(c,Stealth))
    assert copy is not timer and copy.idle==copy.after and timer.idle==0.4
    copy.on_tick(clone,g);assert has(clone,'invisible') and not has(ghost,'invisible')
    assert timer.idle==0.4


def t_cloned_souldier_keeps_damage_without_sharing_fade_state():
    g=arena();ghost=create('royal_ghost',11,'blue',9,10,evolved=True);g.deploy('blue',ghost);g.tick()
    target=Dummy('red',10,10,hp=50000,spd=0,dmg=0);g.deploy('red',target)
    g._do_attack(ghost,target);g.tick()
    souls=[t for t in g.players['blue'].troops if t.name=='Souldier'];assert len(souls)==2
    soul=souls[0];soul.x,soul.y=5,10;ghost.x,ghost.y=1,1;souls[1].x,souls[1].y=1,5
    fade=next(c for c in soul.components if isinstance(c,Fade));fade.idle=0.75
    clone=duplicate(g,soul);copied=next(c for c in clone.components if isinstance(c,Fade))
    assert copied is not fade and copied.idle==fade.idle==0.75
    assert clone.hp==clone.max_hp==1 and clone.dmg==soul.dmg==81
    hp=target.hp;g._do_attack(clone,target)
    assert target.hp==hp-81 and copied.idle==0 and fade.idle==0.75
    fade.on_tick(soul,g);assert copied.idle==0
    assert not any(isinstance(c,EvoRoyalGhost) for c in clone.components)


def t_multiple_clones_cannot_rearm_original_summon_latch():
    g=arena();ghost=create('royal_ghost',11,'blue',9,10,evolved=True);g.deploy('blue',ghost);g.tick()
    a=duplicate(g,ghost);b=duplicate(g,ghost)
    assert a is not b and len(g.players['blue'].troops)==3
    state=next(c for c in ghost.components if isinstance(c,EvoRoyalGhost))
    target=Dummy('red',10,10,hp=50000,spd=0,dmg=0);g.deploy('red',target)
    g._do_attack(ghost,target);g.tick();count=len([t for t in g.players['blue'].troops if t.name=='Souldier'])
    assert count==2 and not state.was_invis
    for c in (a,b):g._do_attack(c,target)
    for _ in range(4):g.tick()
    assert len([t for t in g.players['blue'].troops if t.name=='Souldier'])==count


def t_suspicious_bush_clone_stealth_is_independent():
    g=arena();bush=create('suspicious_bush',11,'blue',9,10);g.deploy('blue',bush)
    original=next(c for c in bush.components if isinstance(c,Stealth));original.idle=0.75
    clone=duplicate(g,bush);copied=next(c for c in clone.components if isinstance(c,Stealth))
    assert original is not copied and copied.after==original.after and copied.idle==original.idle==0.75
    copied.idle=3;assert original.idle==0.75
