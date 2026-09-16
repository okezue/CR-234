import pytest

from sim.cards import create
from sim.fx import CurseOnHit,SoulCollect
from sim.game import Game


DECK=['skeleton_king','knight','archers','fireball','giant','musketeer','bomber','arrows']


def setup(team='blue',level=11):
    g=Game(p1={'deck':DECK,'drag_del':0,'card_levels':{'skeleton_king':level}},p2={'deck':DECK,'drag_del':0,'card_levels':{'skeleton_king':level}})
    for tower in g.arena.towers:
        tower.rng=0
        if tower.troop:tower.troop.RNG=0
    p=g.players[team];p.deck.hand=DECK[:4];p.elixir=10
    assert g.play_card(team,'skeleton_king',9.5,10.5 if team=='blue' else 21.5)==(True,'ok')
    g.run(2.2);king=next(t for t in p.troops if t.name=='Skeleton King');p.elixir=10
    assert g.activate_ability(team,king)==(True,'ok');g.run(3)
    children=[t for t in p.troops if t is not king]
    assert len(children)==6
    return g,king,children


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('level',(11,16))
def t_ability_skeletons_have_one_hp_and_clone_identity(team,level):
    g,_,children=setup(team,level)
    ordinary=create('skeletons',level,team,9,10)[0]
    assert all(t.hp==t.max_hp==1 and t.is_clone for t in children)
    assert all(t.dmg==ordinary.dmg and t.hspd==ordinary.hspd for t in children)
    victim=children[0];victim.take_damage(1);g._proc_deaths();assert victim not in g.players[team].troops


def t_ability_children_leave_templates_parent_and_sibling_lists_unchanged():
    g,king,children=setup();template=king.ability.scfg
    assert template['hp']>1 and king.max_hp>1 and not getattr(king,'is_clone',False)
    assert all(t.components is not template['components'] for t in children)
    assert len({id(t.components) for t in children})==len(children)
    assert all(t.hp==t.max_hp==1 and t.is_clone for t in children)
    assert all(t.dmg==template['dmg'] for t in children)
    assert len(g.players['blue'].troops)==7


def t_clone_spell_cannot_copy_ability_skeletons_but_can_copy_king():
    g,king,children=setup();before=set(g.players['blue'].troops)
    for child in children:create('clone',11,'blue',child.x,child.y).apply(g)
    new=[t for t in g.players['blue'].troops if t not in before]
    assert new and all(t.name==king.name for t in new)
    assert len([t for t in g.players['blue'].troops if t.name=='Skeleton'])==6


def t_real_witch_projectile_converts_one_summon_only_once():
    g,_,children=setup()
    for unit in g.players['blue'].troops:unit.x,unit.y,unit.spd=2.5,4.5,0
    witch=create('mother_witch',11,'red',9.5,13.5);g.deploy('red',witch);witch.spd=0
    victim=children[0];victim.x,victim.y=9.5,10.5
    seen=False
    for _ in range(80):
        g.tick();seen=seen or any(p.tgt is victim and p.team=='red' for p in g.projs)
        hogs=[t for t in g.players['red'].troops if t.name=='Cursed Hog']
        if hogs:break
    assert seen and not victim.alive and len(hogs)==1
    hog=hogs[0];assert hog.hp==hog.max_hp==1 and hog.is_clone
    curse=next(c for c in witch.components if isinstance(c,CurseOnHit))
    for _ in range(2):g._proc_deaths();curse.on_tick(witch,g)
    assert [t for t in g.players['red'].troops if t.name=='Cursed Hog']==[hog]
    hog.take_damage(1);g._proc_deaths();assert hog not in g.players['red'].troops


@pytest.mark.parametrize('kind',('ability','clone','ordinary','wounded','no_soul'))
@pytest.mark.parametrize('team',('blue','red'))
def t_cursed_clone_skeleton_produces_one_hp_clone_hog(kind,team):
    g,king,children=setup(team);opp=g._opp(team)
    if kind=='ability':victim=children[0]
    else:
        victim=create('skeletons',11,team,2.5,10.5 if team=='blue' else 21.5)[0];g.deploy(team,victim)
        if kind=='clone':
            create('clone',11,team,victim.x,victim.y).apply(g)
            victim=next(t for t in g.players[team].troops if t is not victim and getattr(t,'is_clone',False) and t.name==victim.name)
        elif kind=='wounded':victim.hp=1
        elif kind=='no_soul':victim.no_soul=True
    witch=create('mother_witch',11,opp,2.5,25.5 if opp=='red' else 6.5);g.deploy(opp,witch)
    curse=next(c for c in witch.components if isinstance(c,CurseOnHit));g._do_attack(witch,victim)
    if victim.alive:victim.take_damage(victim.hp)
    g._proc_deaths();curse.on_tick(witch,g)
    hog=next(t for t in g.players[opp].troops if t.name=='Cursed Hog')
    assert hog.dmg==curse.cfg['dmg']
    if kind in ('ordinary','wounded','no_soul'):assert hog.hp==hog.max_hp>1 and not getattr(hog,'is_clone',False)
    else:
        assert hog.hp==hog.max_hp==1 and hog.is_clone
        before=len(g.players[opp].troops);create('clone',11,opp,hog.x,hog.y).apply(g)
        assert len([t for t in g.players[opp].troops if t.name=='Cursed Hog'])==1
        assert len(g.players[opp].troops)>=before
    sc=next(c for c in king.components if isinstance(c,SoulCollect));before=sc.souls
    hog.take_damage(hog.hp);g._proc_deaths();assert sc.souls==before+(kind in ('ordinary','wounded'))
