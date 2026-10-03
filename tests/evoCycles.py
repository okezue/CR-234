import pytest

from sim import replay as R
from sim.cards import card,create
from sim.game import Game
from tests.evolutionCadence import _game,_play,_row,_static_replay
from tests.util import Dummy,quiet

# Baby Dragon, Bomber and Lumberjack Evolutions need 2 cycles (wiki evolution infoboxes CycleCost 2; histories 6/10/2025, 14/5/2024,
# 9/4/2025: "increased the cycles required to 2 (from 1)"); the Knight, always 2, is the control
TWO=['baby_dragon','bomber','lumberjack']


@pytest.mark.parametrize('name',TWO)
def t_evolution_cycle_records_are_two(name):
    assert card(name)['evo']['cycles']==2


@pytest.mark.parametrize('name',TWO+['knight'])
def t_engine_evolves_every_third_play(name):
    g=_game(name,evolutions=[name])
    assert [_play(g,name).evolved for _ in range(6)]==[False,False,True]*2


@pytest.mark.parametrize('name',TWO+['knight'])
@pytest.mark.parametrize('suffix,kind',[('','evo'),('-ev1','normal')])
def t_replay_evolves_every_third_recorded_play(monkeypatch,name,suffix,kind):
    # the recorded rows mark the deck slot, not the play (card_type 'evo' on every play of the card), so the cycle count decides
    _static_replay(monkeypatch)
    plays=[_row(name.replace('_','-')+suffix,tm,n*20,kind) for n in range(6) for tm in ('blue','red')]
    g,_=R.replay_battle('evo-cycles',plays,{'result':'D','tc':0,'oc':0})
    for tm in ('blue','red'):assert [p.evolved for p in g.pending if p.team==tm]==[False,False,True]*2


def _archer_damage(gap,evolved):
    # one rooted Archer and a passive dummy gap tiles away edge to edge (both collision radii 0.5), towers silenced
    g=quiet(Game());a=create('archers',11,'blue',2.6,11,evolved=evolved)[0];a.spd=0;g.deploy('blue',a)
    d=Dummy('red',a.x+gap+1.0,a.y,hp=50000,dmg=0,spd=0);g.deploy('red',d)
    g.run(4)
    return 50000-d.hp


def t_evolved_archers_range_six_and_sight_six_point_six():
    # export Archer_EV1 range 6000, sightRange 6600; wiki Archers/Evolution range 6, "1 tile further than the originals"
    assert all((t.rng,t.sight_r)==(6.0,6.6) for t in create('archers',11,'blue',9,10,evolved=True))
    assert all((t.rng,t.sight_r)==(5.0,5.5) for t in create('archers',11,'blue',9,10))


def t_evolved_archers_power_shot_from_five_to_six_tiles():
    # 5.4 and 5.9 tiles are inside the evolved range and the 4 to 6 tile Power Shot band: every hit is 112 + 28 at level 11
    for gap in (5.4,5.9):
        dmg=_archer_damage(gap,True)
        assert dmg>=3*140 and dmg%140==0
    assert _archer_damage(5.4,False)==0
    assert _archer_damage(6.2,True)==0


@pytest.mark.parametrize('evolved',[False,True])
def t_dart_goblin_sight_seven_tiles(evolved):
    # Supercell March 2026 note "Dart Goblin, Sight Range: 7.5 tiles -> 7 tiles"; export BlowdartGoblin sightRange 7000
    dg=create('dart_goblin',11,'blue',1,10,evolved=evolved)
    assert dg.sight_r==7.0 and dg.rng==6.5
    g=Game();g.deploy('blue',dg)
    near=Dummy('red',1+6.9+1.0,10,spd=0);g.deploy('red',near)
    assert g._find_target(dg)[0] is near
    near.alive=False;dg.aggro_tgt=None
    far=Dummy('red',1+7.25+1.0,10,spd=0);g.deploy('red',far)
    assert hasattr(g._find_target(dg)[0],'ttype')
