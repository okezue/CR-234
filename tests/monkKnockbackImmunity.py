import math

import pytest

from sim.cards import create, load
from sim.game import Game
from tests.util import quiet


# Wiki Version History/2025, Balance Changes (12/12/2025): "Monk: immune to any knockback". The 24/11/2025 update had listed the pushes
# he was vulnerable to (Evolved Mega Knight uppercut, another Monk's combo, Fireball and Giant Snowball corner hits); 12/12 restored
# immunity to all of them. Wiki Monk: his own third strike still knocks back "even if the targeted troop is normally immune to knockback";
# wiki The Log: pushes back all ground troops, resetting the Prince's charge, so it also overrides ordinary immunity.


def freeze(*troops):
    for t in troops:t.spd=0


def disarm(*troops):
    for t in troops:t.dmg=0


def run(g,seconds):
    for _ in range(int(round(seconds/g.DT))):g.tick()


def t_monk_immunity_is_sourced_on_the_patch():
    c=load()['cards']['monk']
    assert c['skills']['immunity']=={'knockback':True,'forcedKnockback':True}
    assert c['src']['skills.immunity']=='patch:2026-09-17e'
    m=create('monk',11,'blue',9,8)
    assert m.kb_immune and m.kb_immune_all


def t_another_monks_third_strike_leaves_a_monk_on_its_tile():
    g=quiet(Game());striker=create('monk',11,'blue',9,8);g.deploy('blue',striker)
    victim=create('monk',11,'red',9,9.2);g.deploy('red',victim)
    freeze(striker,victim);disarm(victim);hp=victim.hp;pos=(victim.x,victim.y)
    # first hit 0.2 s, then 0.8 s each: the third strike lands at 1.8 s and the fourth would follow at 2.6 s
    run(g,2.4)
    assert hp-victim.hp==140+140+422 and (victim.x,victim.y)==pos


def t_the_same_third_strike_still_shoves_an_ordinary_immune_troop():
    # the Prince's ordinary immunity does not resist the combo (wiki Monk: "even if the targeted troop is normally immune to knockback")
    g=quiet(Game());striker=create('monk',11,'blue',9,8);g.deploy('blue',striker)
    prince=create('prince',11,'red',9,9.2);g.deploy('red',prince)
    freeze(striker,prince);disarm(prince);y0=prince.y
    assert prince.kb_immune and not getattr(prince,'kb_immune_all',False)
    run(g,2.6)
    assert prince.y>y0+1.0


@pytest.mark.parametrize('spell',('the_log','fireball','giant_snowball'))
def t_knockback_spells_move_a_knight_but_not_a_monk(spell):
    # the spells land half a tile short of each troop so the push has a direction; the Log rolls up from a tile behind
    g=quiet(Game());knight=create('knight',11,'red',6,20);monk=create('monk',11,'red',12,20)
    freeze(knight,monk);g.deploy('red',knight);g.deploy('red',monk)
    kp=(knight.x,knight.y);mp=(monk.x,monk.y)
    for x in (6,12):
        s=create(spell,11,'blue',x,19 if spell=='the_log' else 19.5);s.apply(g);g.spells.append(s)
    run(g,3.0)
    assert (knight.x,knight.y)!=kp and knight.hp<knight.max_hp
    assert (monk.x,monk.y)==mp and monk.hp<monk.max_hp


def t_evolved_royal_giant_recoil_damages_a_monk_without_moving_it():
    g=quiet(Game());rg=create('royal_giant',11,'red',9,20,evolved=True);rg.spd=0;g.deploy('red',rg)
    cannon=create('cannon',11,'blue',9,14.5);g.deploy('blue',cannon)
    monk=create('monk',11,'blue',9,21.6);knight=create('knight',11,'blue',10.3,21.6)
    freeze(monk,knight);disarm(monk,knight);g.deploy('blue',monk);g.deploy('blue',knight)
    hp=monk.hp;mp=(monk.x,monk.y);kp=(knight.x,knight.y)
    run(g,3.0)
    assert hp-monk.hp>=81 and (hp-monk.hp)%81==0
    # the shove is radial from the Giant, so the knight's displacement is measured as a distance
    assert (monk.x,monk.y)==mp and math.dist((knight.x,knight.y),kp)>0.9


def t_evolved_mega_knight_uppercut_leaves_a_monk_but_launches_a_knight():
    def duel(card):
        g=quiet(Game());mk=create('mega_knight',11,'blue',9,13.4,evolved=True);g.deploy('blue',mk)
        # the victim stands on the bank (row 15 is water) clear of the Mega Knight's body
        v=create(card,11,'red',9,14.8);freeze(mk,v);disarm(v);g.deploy('red',v);y0=v.y
        run(g,4.0)
        return v,y0
    knight,ky=duel('knight')
    assert knight.y>ky+2.0
    monk,my=duel('monk')
    assert monk.hp<monk.max_hp and monk.y==my


def t_fisherman_hook_still_drags_a_monk():
    # the hook is a pull, not a knockback: immunity does not cover it
    g=quiet(Game());fm=create('fisherman',11,'blue',9,10);fm.spd=0;g.deploy('blue',fm)
    monk=create('monk',11,'red',9,15);freeze(monk);disarm(monk);g.deploy('red',monk);y0=monk.y
    run(g,4.0)
    assert monk.y<y0-1.0


def t_evolved_executioner_axe_pushes_a_knight_but_not_a_monk_or_a_heavy_troop():
    # wiki Executioner/Evolution: he pushes back all troops affected by knockback; the Giant Skeleton cannot be pushed
    def smash(card):
        g=quiet(Game());ex=create('executioner',11,'blue',9,10,evolved=True);ex.spd=0;g.deploy('blue',ex)
        v=create(card,11,'red',9,12);freeze(v);disarm(v);g.deploy('red',v);y0=v.y;hp=v.hp
        run(g,3.0)
        return v,y0,hp
    knight,ky,khp=smash('knight')
    assert knight.y>ky+0.9 and knight.hp<khp
    monk,my,mhp=smash('monk')
    assert monk.y==my and monk.hp<mhp
    giant,gy,ghp=smash('giant_skeleton')
    assert giant.mass>=10 and giant.y==gy and giant.hp<ghp


def t_cloned_monk_keeps_the_immunity():
    g=quiet(Game());monk=create('monk',11,'red',9,20);g.deploy('red',monk)
    clone_spell=create('clone',11,'red',9,20);clone_spell.apply(g)
    clone=next(t for t in g.players['red'].troops if t is not monk)
    assert clone.is_clone and clone.hp==1 and clone.kb_immune and clone.kb_immune_all


def t_tornado_still_pulls_a_monk():
    # the Tornado is a pull, not a knockback
    g=quiet(Game());monk=create('monk',11,'red',11,10);freeze(monk);g.deploy('red',monk)
    spell=create('tornado',11,'blue',9,10);spell.apply(g);spell.tick(g.DT,g)
    assert monk.x<11 and monk.y==10
