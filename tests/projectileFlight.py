from sim.cards import card,create as mk_card
from sim.game import Game
from tests.util import Dummy,quiet


# Projectile speeds are the game data export's units per tick over 50, the convention verified for walking speeds and, in the LCQ
# recordings of 5 September 2026, for the rolling Log (200: 10.1 tiles in 2.6 to 2.7 s), the Bomber's bomb (400: about 3.9 tiles in
# 0.45 to 0.55 s) and the evolved Giant Snowball (800: 10.5 tiles in 0.55 to 0.65 s, 23.2 tiles in more than 1.2 s). The Tower Princess
# arrow and the Fireball, both 600 in the export, fly faster in every recorded flight: the arrow 6.6 to 7.3 tiles in 0.35 to 0.45 s
# (two towers, two games), the Fireball 17.6 tiles in 0.90 to 0.95 s and 10.1 tiles in about 0.6 s. Their speeds come from the
# recordings (patch 2026-09-28: 17 and 18 tiles/s); the convention and every other projectile keep the export value.


def release_and_hit(g,tr,until=3.0):
    # engine ticks of the tower's first release (a projectile appears) and of the standing dummy's first loss of health
    rel=hit=None;hp=tr.hp
    while g.t<until-1e-9:
        g.tick()
        if rel is None and g.projs:rel=round(g.t,2)
        if tr.hp<hp:hit=round(g.t,2);break
    return rel,hit


def t_tower_princess_arrow_record_carries_the_recorded_speed():
    c=card('tower_princess')
    assert c['projectile']['speed']==17.0 and c['src']['projectile.speed']=='patch:2026-09-28',(c['projectile'],c['src'].get('projectile.speed'))


def t_tower_princess_arrow_reaches_a_target_seven_tiles_out_in_two_fifths_of_a_second():
    # 6.8 tiles from the left tower's centre, in range from the first tick; the export's 600 over 50 needs 0.60 s of flight
    g=Game();tr=Dummy('red',3.0,13.3,hp=50000,spd=0);g.deploy('red',tr)
    rel,hit=release_and_hit(g,tr)
    assert rel is not None and hit is not None,(rel,hit)
    assert 0.35<=round(hit-rel,2)<=0.45,(rel,hit)


def t_fireball_record_carries_the_recorded_speed():
    c=card('fireball')
    assert c['projectile']['speed']==18.0 and c['src']['projectile.speed']=='patch:2026-09-28',(c['projectile'],c['src'].get('projectile.speed'))


def t_fireball_cast_eighteen_tiles_from_the_king_lands_after_one_second():
    # blue's King Tower stands at (9, 3); the export's 600 over 50 lands this cast at 1.5 s
    g=quiet(Game());tr=Dummy('red',9.0,21.0,hp=50000,spd=0);g.deploy('red',tr)
    fb=mk_card('fireball',11,'blue',9.0,21.0);g._cast('blue',fb,9.0,21.0);t0=g.t
    assert len(g.projs)==1 and tr.hp==50000
    while tr.hp==50000 and g.t<t0+2.0-1e-9:g.tick()
    assert tr.hp<50000 and 0.95<=round(g.t-t0,2)<=1.05,(g.t-t0,tr.hp)


def t_other_projectiles_keep_the_export_value_over_fifty():
    # the recorded controls of the convention and two unmeasured projectiles stay on the export
    assert card('the_log')['skills']['pierce']['speed']==4.0
    assert card('bomber')['projectile']['speed']==8.0
    assert card('giant_snowball')['projectile']['speed']==16.0
    assert card('arrows')['projectile']['speed']==22.0 and card('rocket')['projectile']['speed']==7.0
    assert card('king_tower')['projectile']['speed'] is None
