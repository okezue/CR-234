import math
import random

import pytest

from sim.cards import card
from sim.game import Game


DECK=['skeleton_king','knight','archers','fireball','giant','musketeer','bomber','arrows']


def setup(team='blue',x=9.5,y=10.5):
    random.seed(42)
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0})
    for tw in g.arena.towers:
        tw.rng=0
        if tw.troop:tw.troop.RNG=0
    p=g.players[team];p.deck.hand=DECK[:4];p.elixir=10
    assert g.play_card(team,'skeleton_king',x,y)==(True,'ok');g.run(2.2)
    king=next(t for t in p.troops if t.name=='Skeleton King');king.spd=0;p.elixir=10
    assert g.activate_ability(team,king)==(True,'ok')
    while not king.ability.active:g.tick()
    return g,king


def births(g,king,continued=False):
    ab=king.ability;tick=ab.tick;records=[]
    def observe(dt,tr,game):
        before=set(game.players[tr.team].troops);result=tick(dt,tr,game)
        for child in game.players[tr.team].troops:
            if child not in before:records.append({'time':game.t,'anchor':(tr.x,tr.y),'position':(child.x,child.y),'child':child})
        return result
    ab.tick=observe
    if continued:king.take_damage(king.hp);g._proc_deaths()
    g.run(3)
    return records


def t_current_soul_radius_is_three_point_five_tiles():
    cfg=card('skeleton_king')
    assert cfg['skills']['ability']['skills']['spawn']['radius']==3.5
    assert cfg['units']['skeleton']['radius']==3.5
    g,king=setup();assert king.ability.radius==3.5
    records=births(g,king)
    assert len(records)==6
    assert all(math.dist(r['anchor'],r['position'])<=3.5 for r in records)


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('x',(0.5,9.5,17.5))
@pytest.mark.parametrize('continued',(False,True))
def t_soul_births_stay_in_circle_and_arena_before_movement(team,x,continued):
    g,king=setup(team,x,10.5 if team=='blue' else 21.5);records=births(g,king,continued)
    assert len(records)==6
    for r in records:
        sx,sy=r['position'];child=r['child']
        assert 0<=sx<g.arena.W and 0<=sy<g.arena.H
        assert math.dist(r['anchor'],r['position'])<=3.5+1e-12
        assert child.max_hp==1 and child.is_clone and child.no_soul
    assert [round(records[i+1]['time']-records[i]['time'],8) for i in range(5)]==[.25]*5


@pytest.mark.parametrize('angle',(0,math.pi/2,math.pi,3*math.pi/2))
def t_disk_sampler_has_full_radius_without_square_corners(monkeypatch,angle):
    g,king=setup();values=iter([angle,1]*6)
    monkeypatch.setattr('sim.fx.random.uniform',lambda a,b:next(values))
    records=births(g,king)
    assert len(records)==6
    assert all(math.dist(r['anchor'],r['position'])==pytest.approx(3.5) for r in records)


@pytest.mark.parametrize('anchor,angle',(((0,0),5*math.pi/4),((18-1e-9,32-1e-9),math.pi/4)))
def t_corner_projection_stays_inside_arena_and_original_disk(monkeypatch,anchor,angle):
    g,king=setup();king.x,king.y=anchor
    values=iter([angle,1]*6);monkeypatch.setattr('sim.fx.random.uniform',lambda a,b:next(values))
    records=births(g,king,continued=True)
    assert len(records)==6
    for r in records:
        x,y=r['position'];assert 0<=x<18 and 0<=y<32
        assert math.dist(r['anchor'],r['position'])<=3.5


def t_geometry_patch_survives_card_rebuild_inputs():
    import json
    from pathlib import Path
    from data.build import set_path
    patches=json.loads((Path(__file__).resolve().parents[1]/'data/patches/2026-09-16.json').read_text())
    changes=[p for p in patches if p['card']=='skeleton_king'];assert len(changes)==1
    cfg={'skills':{'ability':{'skills':{'spawn':{'radius':4}}}},'src':{}}
    for p in changes:set_path(cfg,p['path'],p['value'],'patch:2026-09-16')
    assert cfg['skills']['ability']['skills']['spawn']['radius']==card('skeleton_king')['skills']['ability']['skills']['spawn']['radius']==3.5
    assert cfg['src']['skills.ability.skills.spawn.radius']=='patch:2026-09-16'


@pytest.mark.parametrize('x',(.5,17.5))
def t_edge_summons_keep_environment_observations_in_bounds(x):
    from sim.env import CREnv
    env=CREnv(blue_deck=DECK,decision_freq=1);env.reset(seed=4);g=env.game;p=g.players['blue']
    p.deck.hand=DECK[:4];p.elixir=10;p.drag_del=0
    for tw in g.arena.towers:
        tw.rng=0
        if tw.troop:tw.troop.RNG=0
    obs,*_=env.step({'card':0,'x':x,'y':10.5});assert env.observation_space.contains(obs)
    for _ in range(44):
        obs,*_=env.step(4);assert env.observation_space.contains(obs)
    king=next(t for t in p.troops if t.name=='Skeleton King');king.spd=0;p.elixir=10
    assert g.activate_ability('blue',king)==(True,'ok')
    for _ in range(100):
        obs,*_=env.step(4);assert env.observation_space.contains(obs)
        if king.alive and king.ability.active and king.ability.q==5:king.take_damage(king.hp)
    assert len(p.troops)==6 and king not in p.troops
    assert all(t.hp==1 and t.is_clone for t in p.troops)
    env.close()


def t_disk_sampler_zero_radius_stays_at_anchor(monkeypatch):
    g,king=setup();monkeypatch.setattr('sim.fx.random.uniform',lambda a,b:0)
    records=births(g,king)
    assert all(r['position']==r['anchor'] for r in records)
