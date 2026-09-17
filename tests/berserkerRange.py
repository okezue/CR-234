from sim.cards import create, load
from sim.game import Game
from tests.util import quiet


def t_berserker_range_is_the_wiki_short_melee():
    # wiki attributes: Melee: Short (0.8); the export omits the field and the engine minimum is 0.5
    assert load()['cards']['berserker']['range']==0.8
    t=create('berserker',11,'blue',9,9);assert t.rng==0.8 and t.hspd==0.6 and t.dmg==102


def t_berserker_attacks_from_short_melee_reach_without_closing():
    g=quiet(Game());b=create('berserker',11,'blue',9,8);b.spd=0;g.deploy('blue',b)
    knight=create('knight',11,'red',9,8+0.45+0.5+0.7);knight.spd=0;knight.dmg=0;g.deploy('red',knight);hp=knight.hp
    # centres 1.65 apart with collision radii 0.45 and 0.5: edge distance 0.7, inside 0.8 but outside the old 0.5 minimum
    assert 0.5<g._dist(b,knight)<=0.8
    # one second deploy, then the first hit lands 0.2 s later
    g.run(2.0)
    assert knight.hp<hp and b.y==8
