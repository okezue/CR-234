import pytest

from sim.cards import create, load
from sim.game import Game
from sim.units import Status
from tests.util import quiet


DECK=['golden_knight','knight','archers','fireball','giant','musketeer','bomber','arrows']
# export GoldenKnight_Pending_Buff: speedMultiplier 200 with an open-ended spawn time; wiki: a speed boost until a unit is within 5.5 tiles
BOOST=2.0


def setup(team='blue',y=None):
    g=quiet(Game(p1={'deck':DECK,'drag_del':0,'ability_del':.05,'ability_std':0},p2={'deck':DECK,'drag_del':0,'ability_del':.05,'ability_std':0}))
    p=g.players[team];p.deck.hand=DECK[:4];p.elixir=10
    if y is None:y=6.0 if team=='blue' else 26.0
    assert g.play_card(team,'golden_knight',9.5,y)==(True,'ok');g.run(2.0)
    knight=next(t for t in p.troops if t.name=='Golden Knight')
    return g,knight


def target(g,card,team,x,y):
    t=create(card,11,team,x,y)
    if isinstance(t,list):t=t[0]
    t.spd=0;t.dmg=0;t.rng=0;g.deploy(team,t)
    return t


def move_rate(g,tr):
    return g._status_mods(tr)[2]


def travelled(g,tr,seconds):
    # path length over the interval, since the knight walks a bent lane path rather than straight up the arena
    total=0
    for _ in range(int(round(seconds/g.DT))):
        x,y=tr.x,tr.y;g.tick();total+=((tr.x-x)**2+(tr.y-y)**2)**.5
    return total


def t_sourced_boost_value():
    b=load()['cards']['golden_knight']['skills']['ability']['skills']['boost']
    assert b['speedMultiplier']==200 and b['duration'] is None


@pytest.mark.parametrize('team',('blue','red'))
def t_no_target_in_range_keeps_the_ability_pending_with_double_speed(team):
    g,knight=setup(team);p=g.players[team];p.elixir=10
    assert g.activate_ability(team,knight)==(True,'ok');g.run(1.1)
    assert knight.ability.active and knight.ability.dashes==0
    assert move_rate(g,knight)==pytest.approx(BOOST)
    assert g._status_mods(knight)[1]==pytest.approx(1.0)
    assert travelled(g,knight,1.0)==pytest.approx(knight.spd*BOOST,rel=.05)
    assert knight.ability.active and knight.ability.dashes==0
    # elixir regenerates during the wait; the single use is spent at activation
    assert knight.ability.uses==0 and p.elixir<10


def t_pending_dash_starts_when_a_ground_troop_enters_range():
    g,knight=setup();g.players['blue'].elixir=10
    assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.1)
    assert knight.ability.active and move_rate(g,knight)==pytest.approx(BOOST)
    victim=target(g,'giant','red',knight.x,knight.y+3);hp=victim.hp
    g.run(0.3)
    assert victim in knight.ability.hit and hp-victim.hp>=knight.ability.dd and knight.ability.dashes==1
    assert move_rate(g,knight)==pytest.approx(1.0)


def t_pending_ignores_air_only_enemies():
    g,knight=setup();g.players['blue'].elixir=10
    assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.1)
    air=target(g,'baby_dragon','red',knight.x,knight.y+2);hp=air.hp;g.run(0.5)
    assert air.hp==hp and knight.ability.active and knight.ability.dashes==0 and move_rate(g,knight)==pytest.approx(BOOST)


@pytest.mark.parametrize('kind',('freeze','stun'))
def t_pending_boost_survives_stun_and_freeze(kind):
    g,knight=setup();g.players['blue'].elixir=10
    assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.1)
    knight.statuses.append(Status(kind,0.6));y0=knight.y;g.run(0.5)
    assert knight.y==y0 and knight.ability.active
    g.run(0.6)
    assert knight.ability.active and knight.ability.dashes==0 and move_rate(g,knight)==pytest.approx(BOOST)
    assert knight.y!=y0


def t_target_in_range_at_activation_dashes_at_once_without_boost():
    g,knight=setup();knight.spd=0;victim=target(g,'giant','red',knight.x,knight.y+3);g.players['blue'].elixir=10
    assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.1)
    assert victim in knight.ability.hit and knight.ability.dashes==1
    assert move_rate(g,knight)==pytest.approx(1.0)


def t_boost_ends_with_the_dash_chain_and_the_single_use_is_spent():
    g,knight=setup();g.players['blue'].elixir=10
    assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.1)
    victim=target(g,'giant','red',knight.x,knight.y+3);g.run(0.5)
    assert not knight.ability.dashing and not knight.ability.active
    assert move_rate(g,knight)==pytest.approx(1.0) and not any(s.kind=='mboost' for s in knight.statuses)
    assert g.activate_ability('blue',knight)==(False,'ability not ready') and victim in knight.ability.hit


def t_environment_step_keeps_pending_knight_observation_valid():
    from sim.env import CREnv
    env=CREnv(blue_deck=DECK,decision_freq=1);env.reset(seed=4);g=quiet(env.game);p=g.players['blue']
    p.deck.hand=DECK[:4];p.elixir=10;p.drag_del=0;p.ability_del=.05;p.ability_std=0
    env.step({'card':0,'x':9.5,'y':6.0})
    for _ in range(40):env.step(4)
    knight=next(t for t in p.troops if t.name=='Golden Knight');p.elixir=10
    assert g.activate_ability('blue',knight)==(True,'ok')
    for _ in range(25):
        obs,*_=env.step(4);assert env.observation_space.contains(obs)
    assert knight.ability.active and move_rate(g,knight)==pytest.approx(BOOST)
    env.close()


def t_pending_dash_triggers_on_a_ground_building():
    g,knight=setup();g.players['blue'].elixir=10
    assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.1)
    cannon=target(g,'cannon','red',knight.x,knight.y+3);hp=cannon.hp;g.run(0.3)
    assert cannon in knight.ability.hit and hp-cannon.hp>=knight.ability.dd and move_rate(g,knight)==pytest.approx(1.0)


def t_pending_knight_reaching_a_tower_dashes_once_and_stops():
    g,knight=setup(y=14.0);g.players['blue'].elixir=10
    towers=[t for t in g.arena.towers if t.team=='red' and t.ttype=='princess']
    assert all(t.dist(knight.x,knight.y)>5.5 for t in towers)
    assert g.activate_ability('blue',knight)==(True,'ok');g.run(1.1)
    assert knight.ability.active and move_rate(g,knight)==pytest.approx(BOOST)
    hp={t:t.hp for t in towers};g.run(4.0)
    hit=[t for t in towers if t in knight.ability.hit]
    assert len(hit)==1 and hp[hit[0]]-hit[0].hp>=knight.ability.dd
    # a Crown Tower ends the chain; the boost is gone and the single use is spent
    assert not knight.ability.active and move_rate(g,knight)==pytest.approx(1.0) and knight.ability.uses==0
