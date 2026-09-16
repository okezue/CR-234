import pytest

from sim.cards import create
from sim.fx import EvoSkeletons
from sim.game import Game
from sim.units import Status
from tests.util import Dummy


def encounter(team='blue',level=11,evolved=True):
    g=Game()
    for tower in g.arena.towers:tower.alive=False
    parents=create('skeletons',level,team,9,10,evolved=evolved)
    for t in parents:g.deploy(team,t)
    target=Dummy(g._opp(team),9,11,hp=100000,spd=0,dmg=0);g.deploy(target.team,target)
    return g,parents,target


def latest_child(g,team,previous):
    children=[t for t in g.players[team].troops if t not in previous]
    assert len(children)==1
    return children[0]


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('level',(11,16))
def t_each_generation_retains_skeleton_evolution(team,level):
    g,parents,target=encounter(team,level);attacker=parents[0]
    for expected in range(4,9):
        previous=list(g.players[team].troops);hp=target.hp
        attacker.hp=attacker.max_hp/2;attacker.statuses=[Status('slow',2,0.7)];attacker.tgt=target
        g._do_attack(attacker,target)
        assert target.hp==hp-attacker.dmg
        child=latest_child(g,team,previous)
        assert len(g.players[team].troops)==expected
        component=next(c for c in child.components if isinstance(c,EvoSkeletons))
        assert component.mx==8 and component is not next(c for c in attacker.components if isinstance(c,EvoSkeletons))
        assert child.evolved and child.lvl==level and child.hp==child.max_hp==attacker.max_hp
        assert child.dmg==attacker.dmg and child.spd==attacker.spd and child.proj_spd==0
        assert child.statuses==[] and child.tgt is None and child.cd==child.fhspd
        attacker=child
    g._do_attack(attacker,target)
    assert len(g.players[team].troops)==8


@pytest.mark.parametrize('team',('blue','red'))
def t_descendants_continue_attacking_after_originals_die(team):
    g,parents,target=encounter(team);g._do_attack(parents[0],target)
    child=latest_child(g,team,parents)
    for t in parents:t.take_damage(t.hp)
    g._proc_deaths();assert g.players[team].troops==[child]
    g.run(3)
    descendants=g.players[team].troops
    assert len(descendants)>1 and all(any(isinstance(c,EvoSkeletons) for c in t.components) for t in descendants)
    assert all(t.evolved for t in descendants) and target.hp<target.max_hp


def t_real_third_card_deployment_has_replicating_descendants():
    deck=['skeletons','knight','archers','fireball','giant','musketeer','bomber','arrows']
    g=Game(p1={'deck':deck,'evolutions':['skeletons'],'drag_del':0})
    for tower in g.arena.towers:tower.alive=False
    p=g.players['blue']
    for i in range(3):
        p.deck.hand=['skeletons','knight','archers','fireball'];p.elixir=10
        assert g.play_card('blue','skeletons',9,10)==(True,'ok')
        g.run(1.1)
        if i<2:
            assert not any(getattr(t,'evolved',False) for t in p.troops)
            for t in p.troops:t.take_damage(t.hp)
            g._proc_deaths()
    roots=list(p.troops);assert len(roots)==3 and all(t.evolved for t in roots)
    target=Dummy('red',9,11,hp=100000,spd=0,dmg=0);g.deploy('red',target)
    while len(p.troops)==3 and g.t<5:g.tick()
    children=[t for t in p.troops if t not in roots];assert children
    for t in roots:t.take_damage(t.hp)
    g._proc_deaths();n=len(p.troops);g.run(2)
    assert len(p.troops)>n and all(t.evolved for t in p.troops)


def t_ordinary_skeleton_attacks_do_not_multiply():
    g,parents,target=encounter(evolved=False)
    g.run(3)
    assert g.players['blue'].troops==parents and target.hp<target.max_hp


def t_descendant_death_frees_existing_cap_capacity():
    g,parents,target=encounter();attacker=parents[0]
    while len(g.players['blue'].troops)<8:
        previous=list(g.players['blue'].troops);g._do_attack(attacker,target);attacker=latest_child(g,'blue',previous)
    victim=next(t for t in g.players['blue'].troops if t not in parents and t is not attacker);victim.take_damage(victim.hp)
    assert len([t for t in g.players['blue'].troops if t.alive])==7
    g._do_attack(attacker,target)
    assert len(g.players['blue'].troops)==9 and sum(t.alive for t in g.players['blue'].troops)==8
    g._do_attack(attacker,target);assert len(g.players['blue'].troops)==9
    g._proc_deaths();assert len(g.players['blue'].troops)==8 and victim not in g.players['blue'].troops


def t_ready_parents_cannot_create_same_tick_infinite_descendants():
    g,parents,target=encounter()
    for t in parents:t.x,t.y=9,10.5;t.cd=0
    hp=target.hp;g.tick()
    assert target.hp==hp-sum(t.dmg for t in parents)
    assert len(g.players['blue'].troops)==6
    for _ in range(50):
        g.tick()
        assert len([t for t in g.players['blue'].troops if t.alive])<=8
