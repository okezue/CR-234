import pytest

from sim.cards import card,create
from sim.game import Game
from sim.units import has
from tests.util import Dummy

# Supercell December 2025 balance changes: "Electro Spirit Chain Speed: 0.2s -> 0.25"; wiki Electro Spirit history 1/12/2025: Shock Chain
# Period 0.25 s (from 0.2 s). Each target is stunned 0.5 s when hit (export ElectroSpiritProjectile buffTime 500). No source gives a chain
# period for the Electro Dragon, so its three hits (and the evolved dragon's first three) still land together.


def line(n,team='red',y=12.0,x0=2.0,gap=2.5):
    g=Game()
    for t in g.arena.towers:t.alive=False
    ds=[]
    for i in range(n):
        d=Dummy(team,x0+gap*i,y,hp=10**6,spd=0,dmg=0);d._settled=True;g.deploy(team,d);ds.append(d)
    return g,ds


def hits(g,ds,until=5.0):
    out={}
    while g.t<until:
        g.tick()
        for i,d in enumerate(ds):
            if d.hp<10**6 and i not in out:out[i]=(round(g.t,2),has(d,'stun'))
    return out


def t_spirit_period_data():
    assert card('electro_spirit')['skills']['pierce']['bounceDelay']==0.25
    assert create('electro_spirit',11,'blue',9,9).chain_period==0.25


def t_spirit_bounces_one_every_quarter_second_and_stuns_each():
    g,ds=line(4);s=create('electro_spirit',11,'blue',2,10);s._settled=True;g.deploy('blue',s)
    h=hits(g,ds)
    assert sorted(h)==[0,1,2,3] and all(st for _,st in h.values())
    t=[h[i][0] for i in range(4)]
    assert [round(b-a,2) for a,b in zip(t,t[1:])]==[0.25,0.25,0.25]


def t_spirit_ninth_target_is_hit_two_seconds_after_the_first():
    g,ds=line(10,gap=1.5,x0=1.5);s=create('electro_spirit',11,'blue',1.5,10);s._settled=True;g.deploy('blue',s)
    h=hits(g,ds)
    assert len(h)==9 and round(max(t for t,_ in h.values())-min(t for t,_ in h.values()),2)==2.0


def t_spirit_chain_stops_when_the_next_body_is_out_of_reach():
    g,ds=line(3);s=create('electro_spirit',11,'blue',2,10);s._settled=True;g.deploy('blue',s)
    first=None
    while g.t<5 and first is None:
        g.tick()
        if ds[0].hp<10**6:first=g.t
    ds[1].x=17.5;ds[2].x=17.8
    g.run(1.0)
    assert ds[1].hp==ds[2].hp==10**6 and not [sp for sp in g.spells if type(sp).__name__=='Chain']


@pytest.mark.parametrize('evolved',(False,True))
def t_electro_dragon_first_three_hits_land_together(evolved):
    g,ds=line(3,y=14.0,x0=7.0,gap=2.0)
    d=create('electro_dragon',11,'blue',9,11,evolved=evolved);d.spd=0;d._settled=True;g.deploy('blue',d)
    assert getattr(d,'chain_period',0)==0
    h=hits(g,ds,4.0)
    assert sorted(h)==[0,1,2] and len({t for t,_ in h.values()})==1
