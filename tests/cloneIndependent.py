import pytest

from sim import fx
from sim.cards import create
from sim.game import Game
from tests.util import Dummy

# wiki Clone: troops with charging, dashing, jumping or grabbing attacks (Dark Prince, Prince, Battle Ram, Ram Rider, Evolved Royal Recruits,
# Bandit, Boss Bandit, Mega Knight, Fisherman) "retain their respective abilities, and cloning them will not interrupt the original troop's
# actions. However, the cloned unit will not inherit the current status of the troop"; history 13/3/2017: "the clones spawn without being
# charged". So each copy owns its components and starts with no attack in progress.


def bare():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


def unit(g,name,x,y,team='blue',evolved=False):
    t=create(name,11,team,x,y,evolved=evolved);t=t[0] if isinstance(t,list) else t
    t._settled=True;g.deploy(team,t);return t


def target(g,x,y,team='red'):
    d=Dummy(team,x,y,hp=10**6,spd=0,dmg=0);d._settled=True;g.deploy(team,d);return d


def clone(g,t,away=False):
    # away: the copy is moved 4 tiles aside so that body collisions do not shove the original
    create('clone',11,t.team,t.x,t.y).apply(g)
    c=next(c for c in g.players[t.team].troops if getattr(c,'is_clone',False) and c.name==t.name)
    if away:c.x-=4
    return c


def comp(t,kind):return next(c for c in t.components if isinstance(c,kind))


@pytest.mark.parametrize('name,evolved',[('prince',False),('dark_prince',False),('battle_ram',False),('battle_ram',True),('ram_rider',False),
                                         ('bandit',False),('boss_bandit',False),('mega_knight',False),('mega_knight',True),('fisherman',False),
                                         ('inferno_dragon',False),('inferno_dragon',True),('mighty_miner',False),('royal_recruits',True),
                                         ('mother_witch',False),('monk',False),('witch',False)])
def t_clone_owns_every_component(name,evolved):
    g=bare();t=unit(g,name,9,10,evolved=evolved);c=clone(g,t)
    assert c.components and not [type(x).__name__ for x in c.components if any(x is y for y in t.components)]


def charge_time(cloned_at=None):
    # a Prince walks at a far target; returns the charge times of the original and, if cloned, of the clone
    g=bare();p=unit(g,'prince',9,4);target(g,9,9.4);cl=None;out={}
    while g.t<4.5:
        g.tick()
        if cloned_at is not None and cl is None and g.t>=cloned_at:cl=clone(g,p,True)
        for k,u in (('orig',p),('clone',cl)):
            if u is not None and k not in out and comp(u,fx.Charge).charged:out[k]=round(g.t,2)
    return g,p,cl,out


def t_cloned_walking_prince_is_not_charged_and_charges_after_its_own_run():
    g=bare();p=unit(g,'prince',9,4);target(g,9,9.4);g.run(1.0)
    before=comp(p,fx.Charge).moved;assert 0.9<before<2.0
    cl=clone(g,p,True);g.run(0.1)
    assert not comp(p,fx.Charge).charged and not comp(cl,fx.Charge).charged
    assert abs(comp(p,fx.Charge).moved-before-0.12)<0.02
    assert comp(cl,fx.Charge).moved<0.2 and cl.spd==p.spd==1.2


def t_cloning_does_not_change_when_the_original_charges():
    _,_,_,alone=charge_time(None);_,_,_,cloned=charge_time(1.0)
    assert alone['orig']==cloned['orig']
    # the copy needs its own 2.5-tile run-up: at 1.2 tiles/s that is about 2 s after it appears
    assert cloned['clone']-1.0>1.5


def t_charged_prince_clone_walks_uncharged_at_resting_speed():
    g=bare();p=unit(g,'prince',9,4);target(g,9,12.5)
    while not comp(p,fx.Charge).charged:g.tick()
    assert p.spd==2.4 and abs(p.fhspd-0.4)<1e-9
    cl=clone(g,p)
    assert cl.spd==1.2 and abs(cl.fhspd-0.5)<1e-9 and not comp(cl,fx.Charge).charged
    assert comp(p,fx.Charge).charged and p.spd==2.4


def t_cloned_inferno_dragon_starts_at_stage_one_and_the_original_keeps_its_stage():
    g=bare();d=unit(g,'inferno_dragon',9,8);d.spd=0;target(g,9,10.5);ramp=comp(d,fx.RampUp)
    g.run(2.5);assert d.dmg==ramp.stages[1]
    cl=clone(g,d);cl.spd=0;g.run(0.3)
    assert d.dmg==ramp.stages[1] and cl.dmg==ramp.stages[0]
    g.run(1.8);assert d.dmg==ramp.stages[2] and cl.dmg==ramp.stages[1]


def dash_land(name,kind,cloned_at=None):
    # the original winds up its dash or jump on a dummy 5 tiles ahead; returns when the dummy is first hit and the clone
    g=bare();u=unit(g,name,9,4);d=target(g,9,9);cl=None
    while g.t<3:
        g.tick()
        if cloned_at is not None and cl is None and g.t>=cloned_at:
            assert getattr(comp(u,kind),'charging');cl=clone(g,u,True);spd=cl.spd;busy=getattr(comp(cl,kind),'charging')
        if d.hp<10**6:return round(g.t,2),cl,spd if cl else None,busy if cl else None
    return None,cl,None,None


@pytest.mark.parametrize('name,kind,rest',[('bandit',fx.BanditDash,1.8),('boss_bandit',fx.BanditDash,1.8),('mega_knight',fx.MKJump,1.2)])
def t_clone_during_a_wind_up_walks_and_the_original_lands_on_time(name,kind,rest):
    alone,_,_,_=dash_land(name,kind)
    landed,cl,spd,busy=dash_land(name,kind,0.3)
    assert alone is not None and landed==alone
    assert spd==rest and busy is False


def t_cloned_fisherman_is_not_holding_a_hook():
    g=bare();f=unit(g,'fisherman',9,4);target(g,9,9)
    while not comp(f,fx.Hook).charging:g.tick()
    assert f.spd==0
    cl=clone(g,f)
    assert cl.spd==1.2 and not comp(cl,fx.Hook).charging and comp(f,fx.Hook).charging


def t_mother_witch_mark_made_before_the_clone_gives_one_hog():
    g=bare();m=unit(g,'mother_witch',9,8);m.spd=0;v=Dummy('red',9,11,hp=10**6,spd=0,dmg=0);v._settled=True;g.deploy('red',v)
    while not comp(m,fx.CurseOnHit).marks:g.tick()
    clone(g,m);v.hp=1;v.take_damage(1);g.run(0.5)
    assert len([t for t in g.players['blue'].troops if t.alive and 'Hog' in t.name])==1
