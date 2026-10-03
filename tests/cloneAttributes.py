import pytest

from sim.cards import create
from sim.game import Game
from tests.util import Dummy

# wiki Clone: "Cloned troops are fragile, but pack the same punch as the original!"; each copy has 1 hitpoint and a 1-point shield when the
# original has a shield. The copy carries the original's charge, jump, chain, stun and slow attributes.


def bare():
    g=Game()
    for t in g.arena.towers:t.alive=False
    return g


def copy_of(name,x=9,y=4,team='blue'):
    # the original is cloned after a tick, then parked in a far corner so only the copy fights
    g=bare();o=create(name,11,team,x,y);o=o[0] if isinstance(o,list) else o;o._settled=True;g.deploy(team,o);g.tick()
    create('clone',11,team,o.x,o.y).apply(g)
    c=next(t for t in g.players[team].troops if getattr(t,'is_clone',False) and t.name==o.name)
    o.x,o.y,o.spd=1.0,1.0,0
    return g,o,c


def dummy(g,x,y,team='red'):
    d=Dummy(team,x,y,hp=10**6,spd=0,dmg=0);d._settled=True;g.deploy(team,d);return d


def first_damage(g,ds,until=6.0):
    while g.t<until:
        g.tick()
        hit=[10**6-d.hp for d in ds]
        if any(hit):return hit
    return [0]*len(ds)


@pytest.mark.parametrize('name,attrs',[('prince',('charge_dmg',)),('dark_prince',('charge_dmg',)),('battle_ram',('charge_dmg',)),
                                       ('mega_knight',('jump_dmg','spawn_zap_dmg')),('electro_dragon',('chain_count','chain_range','chain_stun')),
                                       ('electro_spirit',('chain_count','chain_range','chain_stun')),('electro_wizard',('stun_dur',)),
                                       ('zappies',('stun_dur',)),('ice_wizard',('slow_dur','slow_val'))])
def t_clone_carries_the_attack_attributes_with_one_hitpoint(name,attrs):
    g,o,c=copy_of(name)
    assert all(getattr(c,a)==getattr(o,a) and getattr(o,a) for a in attrs)
    assert c.hp==c.max_hp==1 and c.shield_hp==c.max_shield_hp==int(o.shield_hp>0) and c.dmg==o.dmg


def t_cloned_prince_charged_hit_deals_the_charge_damage():
    # a longer sight lets the copy see the dummy from 6 tiles, so it walks past the 2.5-tile run-up
    g,o,c=copy_of('prince');c.x=5;c.sight_r=9;d=dummy(g,5,9.5)
    assert first_damage(g,[d])==[o.charge_dmg]


def t_cloned_mega_knight_jump_lands_with_the_jump_damage():
    g,o,c=copy_of('mega_knight');c.x=5;d=dummy(g,5,8.6)
    assert first_damage(g,[d])==[o.jump_dmg]


def t_cloned_electro_dragon_chains_three_targets():
    g,o,c=copy_of('electro_dragon',9,11);c.spd=0;ds=[dummy(g,x,13.6) for x in (7,9,11)]
    assert first_damage(g,ds)==[o.dmg]*3


@pytest.mark.parametrize('name,kind',[('electro_wizard','stun'),('zappies','stun'),('ice_wizard','slow')])
def t_cloned_stunner_and_slower_keep_their_effect(name,kind):
    g,o,c=copy_of(name,9,8);c.spd=0;d=dummy(g,c.x,c.y+3.5)
    first_damage(g,[d])
    assert any(st.kind==kind and st.dur>0.3 for st in d.statuses)
