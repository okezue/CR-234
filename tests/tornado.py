import math

import pytest

from sim.cards import create
from sim.fx import Breakdown
from sim.game import Game
from sim.units import Status
from tests.util import Dummy,quiet


@pytest.mark.parametrize('name',('cannon','tesla','goblin_drill','goblin_cage','bomb_tower','inferno_tower','elixir_collector'))
def t_tornado_keeps_buildings_stationary(name):
    for team in ('blue','red'):
        for level in (11,16):
            g=Game();building=create(name,level,g._opp(team),11,10)
            building.x,building.y=11,10;g.deploy(building.team,building)
            assert building.is_building
            before=(building.x,building.y,building.cd);hp=building.hp
            spell=create('tornado',level,team,9,10);spell.apply(g)
            for _ in range(30):
                if not spell.active:break
                spell.tick(g.DT,g)
                assert (building.x,building.y,building.cd)==before
            assert not spell.active and spell.ticks_left==0
            if name=='goblin_drill':assert building.hp==hp
            else:assert hp-building.hp==spell.ticks*spell.tick_dmg


def t_tornado_skips_building_pull_without_skipping_other_targets():
    for team in ('blue','red'):
        g=Game();enemy=g._opp(team)
        building=create('cannon',11,enemy,11,10);g.deploy(enemy,building)
        hidden=create('tesla',11,enemy,11,11);hidden.statuses.append(Status('burrowed',2));g.deploy(enemy,hidden)
        troop=Dummy(enemy,11,10,hp=5000,spd=0);g.deploy(enemy,troop)
        hp=troop.hp;spell=create('tornado',11,team,9,10);spell.apply(g);spell.tick(g.DT,g)
        assert (building.x,building.y)==(11,10) and (hidden.x,hidden.y)==(11,11)
        assert math.isclose(troop.x,11-spell.pull_str*g.DT) and troop.y==10
        assert hp-troop.hp==spell.tick_dmg


@pytest.mark.parametrize('name',('knight','baby_dragon','battle_healer','furnace','cannon_cart'))
def t_tornado_keeps_existing_troop_pull(name):
    g=Game();troop=create(name,11,'red',11,10);g.deploy('red',troop)
    assert not getattr(troop,'is_building',False)
    if name=='baby_dragon':assert troop.transport=='Air'
    if name=='battle_healer':assert troop.hovering
    if name=='cannon_cart':assert troop.kb_immune
    target=Dummy('blue',11,11);troop.tgt=troop.aggro_tgt=target
    state=(troop.spd,troop.cd,troop.tgt,troop.aggro_tgt,list(troop.statuses));hp=troop.hp
    spell=create('tornado',11,'blue',9,10);spell.apply(g);spell.tick(g.DT,g)
    assert math.isclose(troop.x,11-spell.pull_str*g.DT) and troop.y==10
    assert hp-troop.hp==spell.tick_dmg
    assert (troop.spd,troop.cd,troop.tgt,troop.aggro_tgt,troop.statuses)==state


def t_cannon_cart_stops_being_pulled_after_breakdown():
    g=Game();cart=create('cannon_cart',11,'red',11,10);g.deploy('red',cart)
    breakdown=next(c for c in cart.components if isinstance(c,Breakdown))
    spell=create('tornado',11,'blue',9,10);spell.apply(g);spell.tick(g.DT,g)
    assert cart.x<11 and not getattr(cart,'is_building',False)
    cart.take_damage(cart.hp-cart.max_hp*breakdown.pct);breakdown.on_tick(cart,g)
    assert breakdown.on and cart.is_building and cart.spd==0
    frozen=(cart.x,cart.y);hp=cart.hp
    for _ in range(15):spell.tick(g.DT,g)
    assert (cart.x,cart.y)==frozen and hp-cart.hp==spell.tick_dmg


def t_tornado_uses_current_classification_not_hitpoint_threshold():
    g=Game();cart=create('cannon_cart',11,'red',11,10);g.deploy('red',cart)
    cart.hp=cart.max_hp/2
    assert not getattr(cart,'is_building',False)
    spell=create('tornado',11,'blue',9,10);spell.apply(g);spell.tick(g.DT,g)
    assert cart.x<11


def t_tornado_preserves_team_dead_and_range_boundaries():
    g=Game();spell=create('tornado',11,'blue',9,10)
    ally=Dummy('blue',11,10,hp=5000);dead=Dummy('red',11,11,hp=5000);dead.alive=False
    center=Dummy('red',9,10,hp=5000);outside=Dummy('red',9+spell.radius+0.01,10,hp=5000)
    edge=Dummy('red',9+spell.radius,10,hp=5000)
    for t in (ally,dead,center,outside,edge):g.deploy(t.team,t)
    spell.apply(g);spell.tick(g.DT,g)
    assert (ally.x,ally.y,ally.hp)==(11,10,5000)
    assert (dead.x,dead.y,dead.hp)==(11,11,5000)
    assert (center.x,center.y)==(9,10) and center.hp==5000-spell.tick_dmg
    assert outside.x==9+spell.radius+0.01 and outside.hp==5000-spell.tick_dmg
    assert math.isclose(edge.x,9+spell.radius-spell.pull_str*g.DT)


def t_tornado_crown_damage_activation_and_lifetime_are_unchanged():
    for team in ('blue','red'):
        g=Game();tower=g.arena.get_tower(g._opp(team),'king');xy=(tower.cx,tower.cy);hp=tower.hp
        spell=create('tornado',11,team,tower.cx-1,tower.cy);spell.apply(g)
        assert not tower.active and tower.hp==hp
        hits=[];elapsed=0
        while spell.active and elapsed<2:
            before=tower.hp;spell.tick(g.DT,g);elapsed+=g.DT
            if tower.hp!=before:hits.append((round(elapsed,6),before-tower.hp))
            assert (tower.cx,tower.cy)==xy
        assert hits==[(0.05,25),(0.6,25)] and tower.active
        assert spell.ticks_left==0 and not spell.active and math.isclose(elapsed,1.05)
        spell.apply(g);assert tower.hp==hp-50
        lethal=create('tornado',11,team,tower.cx-1,tower.cy);tower.hp=1
        lethal.apply(g);lethal.tick(g.DT,g)
        assert not tower.alive and g.ended and g.players[team].crowns==3


@pytest.mark.parametrize('name',('cannon','tesla','goblin_drill','goblin_cage','bomb_tower','inferno_tower','elixir_collector'))
def t_exposed_buildings_take_normal_ticks_without_moving(name):
    for team in ('blue','red'):
        for level in (11,16):
            g=Game();building=create(name,level,g._opp(team),11,10)
            # Isolate exposed-state spell damage, not Drill travel or Tesla hiding.
            building.x,building.y=11,10;building.statuses=[];g.deploy(building.team,building)
            spell=create('tornado',level,team,9,10);spell.apply(g)
            hp=building.hp;events=[]
            for tick in range(21):
                before=building.hp;spell.tick(g.DT,g)
                if building.hp!=before:events.append((tick+1,before-building.hp))
                assert (building.x,building.y)==(11,10)
            assert events==[(1,spell.tick_dmg),(12,spell.tick_dmg)]
            assert hp-building.hp==2*spell.tick_dmg and not spell.active


@pytest.mark.parametrize('name',('tesla','goblin_drill'))
@pytest.mark.parametrize('kind',('burrowed','invincible'))
def t_building_immunity_is_checked_on_each_damage_tick(name,kind):
    g=Game();building=create(name,11,'red',11,10)
    building.x,building.y=11,10;building.statuses=[Status(kind,2)];g.deploy('red',building)
    hp=building.hp;spell=create('tornado',11,'blue',9,10);spell.apply(g)
    spell.tick(g.DT,g);assert building.hp==hp
    building.statuses=[]
    for _ in range(10):spell.tick(g.DT,g);assert building.hp==hp
    spell.tick(g.DT,g)
    assert building.hp==hp-spell.tick_dmg and (building.x,building.y)==(11,10)


def t_tornado_building_team_range_and_shield_boundaries():
    g=Game();spell=create('tornado',11,'blue',9,10)
    ally=create('cannon',11,'blue',11,10)
    dead=create('cannon',11,'red',11,11);dead.take_damage(dead.hp)
    shielded=create('cannon',11,'red',11,12);shielded.shield_hp=1
    edge=create('cannon',11,'red',9,10)
    edge.x=spell.x+spell.radius+edge.collision_r-0.001
    outside=create('cannon',11,'red',edge.x+0.002,10)
    for b in (ally,dead,shielded,edge,outside):g.deploy(b.team,b)
    before={b:(b.hp,b.x,b.y) for b in (ally,dead,shielded,edge,outside)}
    spell.apply(g);spell.tick(g.DT,g)
    assert ally.hp==before[ally][0] and not dead.alive
    assert shielded.shield_hp==0 and shielded.hp==before[shielded][0]
    assert edge.hp==before[edge][0]-84 and outside.hp==before[outside][0]
    for _ in range(11):spell.tick(g.DT,g)
    assert shielded.hp==before[shielded][0]-84
    assert all((b.x,b.y)==before[b][1:] for b in before)


def t_tornado_rooted_cart_preserves_damage_reduction():
    g=Game();cart=create('cannon_cart',11,'red',11,10);g.deploy('red',cart)
    breakdown=next(c for c in cart.components if isinstance(c,Breakdown))
    cart.hp=cart.max_hp*breakdown.pct;breakdown.on_tick(cart,g)
    cart._dmg_reduction=0.5;hp=cart.hp
    spell=create('tornado',11,'blue',9,10);spell.apply(g)
    for _ in range(21):spell.tick(g.DT,g)
    assert cart.is_building and (cart.x,cart.y)==(11,10)
    assert hp-cart.hp==2*int(spell.tick_dmg*0.5)


def t_tornado_lethal_building_runs_death_spawn_once():
    g=quiet(Game());cage=create('goblin_cage',11,'red',11,10);cage.hp=50;g.deploy('red',cage)
    g._cast('blue',create('tornado',11,'blue',9,10),9,10);g.tick()
    assert not cage.alive and cage not in g.players['red'].troops
    brawlers=[t for t in g.players['red'].troops if t.name=='GoblinBrawler']
    assert len(brawlers)==1
    first=brawlers[0]
    g._proc_deaths();g.run(1.2)
    assert [t for t in g.players['red'].troops if t.name=='GoblinBrawler']==[first]
    assert all(p.crowns==0 for p in g.players.values())


def t_tornado_full_game_preserves_building_location_and_decay():
    g=quiet(Game());building=create('cannon',11,'red',11,10);g.deploy('red',building)
    troop=Dummy('red',12,12,hp=5000,spd=0,dmg=0);g.deploy('red',troop)
    hp=building.hp;g._cast('blue',create('tornado',11,'blue',9,10),9,10)
    g.run(1.5)
    assert (building.x,building.y)==(11,10)
    assert math.isclose(hp-building.hp,building.decay*1.5+2*84,abs_tol=1e-6)
    assert troop.x<12 and troop.hp==5000-2*84 and not g.spells
