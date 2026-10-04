import builtins
import gc
import weakref

from sim import spells
from sim.cards import card,create
from sim.game import Game
from tests.util import Dummy,quiet


# Lightning: the export's area (areaEffectObjectData hitSpeed 460, lifeDuration 1500) fires its bolt projectile every 0.46 s, three times;
# the wiki: "up to 3 lightning bolts", each on a different target. Recorded in 02JY9RR09299: Rin's 253 tower falls 0.52 s after the
# Lightning's area appears on the broadcast, with nothing struck before it.


def cast(g,x,y,team='blue'):
    lt=create('lightning',11,team,x,y);g._cast(team,lt,x,y);return lt,g.t


def hit_times(g,t0,bodies,until=2.0):
    out={}
    while g.t<t0+until:
        g.tick()
        for b in bodies:
            if id(b) not in out and b.hp<b.max_hp:out[id(b)]=round(g.t-t0,2)
    return [out.get(id(b)) for b in bodies]


def t_lightning_strike_timing_comes_from_the_export_and_the_recording():
    c=card('lightning');s=c['skills']['stun'];src=c['src']
    assert (s['delayBetweenStrikes'],s['strikes'],s['firstDelay'])==(0.46,3,0.46),s
    assert src['skills.stun.delayBetweenStrikes'].startswith('gd:') and src['skills.stun.strikes'].startswith('gd:'),src
    assert src['skills.stun.firstDelay']=='patch:2026-10-03lt',src


def t_lightning_bolts_fall_one_by_one_on_the_highest_hitpoints():
    # bolts at 0.46, 0.92 and 1.38 s land on the ticks nearest them, the largest body first; the fourth body is never struck
    g=quiet(Game())
    ds=[Dummy('red',x,y,hp=hp,spd=0,dmg=0) for x,y,hp in ((9,10,5000),(10,10,3000),(10,11,2000),(9,11,500))]
    for d in ds:g.deploy('red',d)
    lt,t0=cast(g,9.5,10.5)
    assert all(d.hp==d.max_hp for d in ds),"No bolt falls when the spell is cast"
    assert hit_times(g,t0,ds)==[0.45,0.9,1.4,None],hit_times
    assert [d.max_hp-d.hp for d in ds[:3]]==[1057]*3


def t_lightning_takes_a_low_tower_after_the_first_interval():
    # the 02JY9RR09299 ending: a 253 tower struck alone stands until the first bolt
    g=quiet(Game());tw=g.arena.get_tower('red','princess','left');tw.hp=253
    lt,t0=cast(g,tw.cx-1,tw.cy)
    g.run_to(t0+0.40)
    assert tw.alive,"The tower should stand until the first bolt"
    g.run_to(t0+0.45)
    assert not tw.alive and g.players['blue'].crowns==1,(tw.hp,g.players['blue'].crowns)


def t_lightning_strikes_a_body_that_enters_between_bolts():
    # wiki Lightning: it retargets onto the Barbarians a destroyed Battle Ram leaves; a body placed after the first bolt takes the second
    g=quiet(Game())
    a=Dummy('red',9,10,hp=5000,spd=0,dmg=0);b=Dummy('red',10,10,hp=1000,spd=0,dmg=0)
    for d in (a,b):g.deploy('red',d)
    lt,t0=cast(g,9.5,10.5)
    g.run_to(t0+0.6)
    n=Dummy('red',9.5,11,hp=3000,spd=0,dmg=0);g.deploy('red',n)
    g.run_to(t0+1.5)
    assert a.hp==5000-1057 and n.hp==3000-1057 and not b.alive,(a.hp,n.hp,b.hp)
    assert lt not in g.spells,"The spell should be gone after its third bolt"


def t_lightning_bolt_stuns_only_when_it_falls():
    g=quiet(Game());d=Dummy('red',9,10,hp=5000,spd=0,dmg=0);g.deploy('red',d)
    lt,t0=cast(g,9,10)
    g.run_to(t0+0.40)
    assert not any(s.kind=='stun' for s in d.statuses),"No stun before the bolt"
    g.run_to(t0+0.45)
    assert any(s.kind=='stun' for s in d.statuses),"The bolt stuns its target"


def t_lightning_strikes_a_new_troop_given_a_dead_struck_troops_id(monkeypatch):
    # a troop a bolt kills leaves the game and is freed, and CPython may give the next troop its id(); force that reuse for the newcomer
    g=quiet(Game());v=Dummy('red',9,10,hp=500,spd=0,dmg=0);g.deploy('red',v)
    lt,t0=cast(g,9.5,10.5)
    g.run_to(t0+0.6)
    assert not v.alive and v not in g.players['red'].troops,"The first bolt kills the troop and the game drops it"
    n=Dummy('red',9.5,11,hp=3000,spd=0,dmg=0);dead=id(v)
    monkeypatch.setattr(spells,'id',lambda o:dead if o is n else builtins.id(o),raising=False)
    g.deploy('red',n);g.run_to(t0+1.0)
    assert n.hp==3000-1057,"The newcomer holding the dead troop's id() takes the second bolt"


def t_lightning_holds_the_troops_it_struck_while_it_lasts():
    # held, a struck troop the game has dropped is not freed while the spell lasts, so no later troop can take its address
    g=quiet(Game());v=Dummy('red',9,10,hp=500,spd=0,dmg=0);g.deploy('red',v)
    lt,t0=cast(g,9.5,10.5)
    g.run_to(t0+0.6)
    gone=weakref.ref(v);del v;gc.collect()
    assert gone() is not None and not gone().alive,"The spell should still hold the troop its first bolt killed"
    n=Dummy('red',9.5,11,hp=3000,spd=0,dmg=0);g.deploy('red',n);g.run_to(t0+1.0)
    assert n.hp==3000-1057,"A troop placed after the kill takes the second bolt"
