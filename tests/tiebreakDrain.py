import pytest

from sim.game import Game
from sim.units import Status
from tests.util import Dummy


def prepare(hp):
    g=Game()
    for t,value in zip(g.arena.towers,hp):t.hp=value
    g.t=g.END-g.DT;g.phase='overtime'
    return g


@pytest.mark.parametrize('reverse',(False,True))
@pytest.mark.parametrize('swap',(False,True))
def t_tiebreak_drain_matches_broadcast_terminal_health(reverse,swap):
    blue=[4824,1316,3052];red=[4699,2702,191]
    if swap:blue,red=red,blue
    g=prepare(blue+red);towers=list(g.arena.towers)
    if reverse:g.arena.towers.reverse()
    for team in ('blue','red'):g.deploy(team,Dummy(team,9,10 if team=='blue' else 22,spd=0,dmg=0))
    g.tick()
    assert [t.hp for t in towers]==[v-191 for v in blue+red]
    assert g.winner==('red' if swap else 'blue') and g.ended and g.phase=='end'
    assert g.players[g.winner].crowns==1 and g.players[g._opp(g.winner)].crowns==0
    assert sum(not t.alive for t in towers)==1
    assert all(not p.troops for p in g.players.values())
    assert not g.arena.get_tower(g.winner,'king').active
    g.replay.snap(g)
    assert [t['hp'] for t in g.replay.snaps[-1]['towers']]==[t.hp for t in g.arena.towers]
    before=[t.hp for t in towers];g.tick();assert [t.hp for t in towers]==before


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('minimum',(1,191.5))
def t_tiebreak_king_drain_does_not_restore_cascade_victims(team,minimum):
    g=prepare([4824,3000,2900,4824,2800,2700]);loser=g.arena.get_tower(team,'king');loser.hp=minimum
    survivors=[t for t in g.arena.towers if t.team!=team];hp=[t.hp for t in survivors]
    g.tick()
    assert g.winner==g._opp(team) and g.players[g.winner].crowns==3
    assert all(t.hp==0 and not t.alive and t.down for t in g.arena.towers if t.team==team)
    assert [t.hp for t in survivors]==[v-minimum for v in hp]
    assert not g.arena.get_tower(g.winner,'king').active
    g._tower_down(loser);assert g.players[g.winner].crowns==3


def t_tiebreak_ignores_previously_destroyed_tower():
    g=prepare([4824,3000,2900,4824,2800,100]);dead=g.arena.get_tower('blue','princess','left')
    other=g.arena.get_tower('red','princess','left')
    for t in (dead,other):t.hp=0;t.alive=False;t.down=True
    for p in g.players.values():p.crowns=1
    g._tiebreaker()
    assert [t.hp for t in g.arena.towers]==[4724,0,2800,4724,0,0]
    assert g.winner=='blue' and g.players['blue'].crowns==2 and g.players['red'].crowns==1
    assert not dead.alive and dead.down


def t_tiebreak_survivor_drain_keeps_status_and_attack_state():
    g=prepare([4824,3000,2900,4824,2800,100]);king=g.arena.get_tower('blue','king')
    status=Status('freeze',3);king.statuses=[status];king.cd=1.25
    g._tiebreaker()
    assert king.hp==4724 and not king.active and king.cd==1.25 and king.statuses==[status]
    assert status.dur==3


@pytest.mark.parametrize('hp',([4824,100,2000,4824,100,2100],[100,3000,2900,100,2800,2700]))
def t_tiebreak_shared_minimum_draw_stays_unchanged(hp):
    g=prepare(hp);g._tiebreaker()
    assert g.ended and g.winner is None and [t.hp for t in g.arena.towers]==hp
    assert all(t.alive for t in g.arena.towers) and all(p.crowns==0 for p in g.players.values())


def t_tiebreak_same_team_minimum_tie_keeps_existing_resolution():
    g=prepare([4824,3000,2900,4824,100,100]);g._tiebreaker()
    assert g.ended and g.winner=='blue' and g.players['blue'].crowns==1
    assert [t.hp for t in g.arena.towers]==[4824,3000,2900,4824,0,100]


def t_tiebreak_empty_arena_draw():
    g=Game()
    for t in g.arena.towers:t.hp=0;t.alive=False
    g._tiebreaker()
    assert g.ended and g.winner is None and all(t.hp==0 for t in g.arena.towers)
