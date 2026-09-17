import pytest

from sim.cards import create, load
from sim.game import Game
from tests.util import quiet


# game data Monk: damage 55, variableDamage2 55, variableDamage3 165, hitSpeed 800, isMeleePushbackAll3; level 11 -> 140 / 140 / 422
COMBO={11:(140,422),16:(224,674)}


def t_monk_combo_damage_is_sourced_on_the_pushback_skill():
    c=load()['cards']['monk'];pb=c['skills']['pushback']
    assert pb['cycle']==3 and pb['distance']==1.8
    assert pb['damage'][10]==422 and pb['damage'][15]==674
    assert c['stats']['damage'][10]==140


def hits_on(g,victim,seconds):
    last=victim.hp;out=[]
    for _ in range(int(round(seconds/g.DT))):
        g.tick()
        if victim.hp!=last:out.append((round(g.t,2),last-victim.hp));last=victim.hp
        if not victim.alive:break
    return out


@pytest.mark.parametrize('lvl',(11,16))
def t_every_third_hit_deals_the_combo_damage(lvl):
    # the monk keeps walking so it re-closes after its own knockback
    g=quiet(Game());monk=create('monk',lvl,'blue',9,8);g.deploy('blue',monk)
    victim=create('pekka',16,'red',9,9.2);victim.spd=0;victim.dmg=0;g.deploy('red',victim)
    hits=hits_on(g,victim,9.0)
    base,combo=COMBO[lvl]
    # the third hit lands as the base hit plus the combo remainder in the same tick
    grouped=[];i=0
    while i<len(hits):
        t,d=hits[i]
        if i+1<len(hits) and hits[i+1][0]==t:d+=hits[i+1][1];i+=1
        grouped.append(d);i+=1
    assert grouped[:6]==[base,base,combo,base,base,combo],grouped


def t_third_hit_knocks_back_a_knockback_immune_troop_but_not_a_building():
    g=quiet(Game());monk=create('monk',11,'blue',9,8);monk.spd=0;g.deploy('blue',monk)
    pekka=create('pekka',11,'red',9,9.2);pekka.spd=0;pekka.dmg=0;g.deploy('red',pekka)
    assert pekka.mass>=10
    y0=pekka.y;hits_on(g,pekka,2.6)
    assert pekka.y>y0+1.0
    g2=quiet(Game());monk2=create('monk',11,'blue',9,8);monk2.spd=0;g2.deploy('blue',monk2)
    cannon=create('cannon',11,'red',9,9.6);g2.deploy('red',cannon);cy=cannon.y
    # buildings also lose health to their lifetime decay every tick, so only the attack-sized drops are compared
    hits=[round(d-cannon.decay*g2.DT) for _,d in hits_on(g2,cannon,2.4) if d>=100]
    assert cannon.y==cy and hits[:3]==[140,140,422]


def t_combo_damage_reaches_crown_towers_in_full():
    g=quiet(Game());tw=next(t for t in g.arena.towers if t.team=='red' and t.ttype=='princess')
    monk=create('monk',11,'blue',tw.cx,tw.cy-2.2);monk.spd=0;g.deploy('blue',monk)
    last=tw.hp;hits=[]
    for _ in range(int(round(3.0/g.DT))):
        g.tick()
        if tw.hp!=last:hits.append(last-tw.hp);last=tw.hp
    # no crownTowerDamagePercent on the Monk: the export's full values land on towers as well
    assert hits[:3]==[140,140,422]


def t_kill_with_a_plain_hit_keeps_the_combo_count():
    g=quiet(Game());monk=create('monk',11,'blue',9,8);monk.spd=0;g.deploy('blue',monk)
    weak=create('skeletons',11,'red',9,9.2)[0];weak.spd=0;weak.dmg=0;g.deploy('red',weak)
    tank=create('pekka',16,'red',9.6,9.2);tank.spd=0;tank.dmg=0;g.deploy('red',tank)
    hp=tank.hp
    # three swings fit in two seconds (first hit 0.2 s, then 0.8 s each): the skeleton takes the first, the tank the second and the combo
    for _ in range(int(round(2.0/g.DT))):g.tick()
    combo=next(c for c in monk.components if type(c).__name__=='MonkCombo')
    assert not weak.alive and hp-tank.hp==140+422 and combo.cnt==0 and monk.dmg==140


def t_third_hit_is_one_blow_against_a_shield():
    # a shield that absorbs a whole hit discards the overflow: one 422 blow on a 40 point shield leaves the body untouched
    from sim.units import Troop
    g=quiet(Game());monk=create('monk',11,'blue',9,8);monk.spd=0;g.deploy('blue',monk)
    guard=create('guards',11,'red',9,9.2)[0];guard.spd=0;guard.dmg=0;g.deploy('red',guard)
    assert isinstance(guard,Troop) and guard.shield_hp>0
    guard.shield_hp=guard.max_shield_hp=320;body=guard.hp
    for _ in range(int(round(2.0/g.DT))):g.tick()
    assert guard.shield_hp==0 and guard.hp==body and monk.dmg==140


def t_monk_carries_no_ramp_component():
    monk=create('monk',11,'blue',9,8)
    assert [type(c).__name__ for c in monk.components]==['MonkCombo']


def t_ordinary_knockback_troop_unchanged():
    g=quiet(Game());bowler=create('bowler',11,'blue',9,12);bowler.spd=0;g.deploy('blue',bowler)
    assert not any(type(c).__name__=='MonkCombo' for c in bowler.components)
