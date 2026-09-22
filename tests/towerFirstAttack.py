from sim.game import Game
from tests.util import Dummy


# Tower troops load the same way as troops: while idle the swing sits at the first attack period (hit speed less load time, the whole
# hit speed without a load time) and a target acquired from idle is shot after that period, then on the attack period. The wiki
# Tower Princess attribute table gives Attack Period 0.8 s and First Attack Period 0.8 s; the 1080p recording of LCQ game 09YP9UPGJGU8
# (first hog push, 35.5 to 36.6 s) shows the princess turning to the hogs and releasing her arrow about 0.55 s later. The engine had the
# Tower Princess fire on the tick she acquired.


def hits(tr,g,until,step=0.05,stop_at_first=False):
    # times at which the standing dummy's health drops, read at the engine's tick
    out=[];hp=tr.hp
    while g.t<until-1e-9:
        g.run(step)
        if tr.hp<hp:
            out.append(round(g.t,2));hp=tr.hp
            if stop_at_first:break
    return out


def t_tower_princess_fires_her_first_arrow_after_the_first_attack_period():
    g=Game();tr=Dummy('red',3.0,13.0,hp=50000,spd=0);g.deploy('red',tr)
    # 6.5 tiles from the tower centre, inside range from the first tick; the arrow flies 12 tiles/s, about 0.55 s
    h=hits(tr,g,3.0)
    assert h and 1.3<=h[0]<=1.45,h
    assert len(h)>=2 and 0.75<=h[1]-h[0]<=0.85,h


def t_an_idle_tower_reloads_to_the_first_attack_period_between_targets():
    g=Game();tr=Dummy('red',3.0,13.0,hp=50000,spd=0);g.deploy('red',tr)
    h=hits(tr,g,1.6);assert len(h)==1,h
    tr.alive=False;g.players['red'].troops.clear()
    g.run(3.0)
    tr2=Dummy('red',3.0,13.0,hp=50000,spd=0);g.deploy('red',tr2);t0=g.t
    h2=hits(tr2,g,t0+2.0)
    assert h2 and 1.3<=h2[0]-t0<=1.45,(h2,t0)


def t_a_target_that_follows_a_kill_is_shot_on_the_attack_period_without_reloading():
    g=Game();a=Dummy('red',3.0,13.0,hp=50000,spd=0);b=Dummy('red',3.0,13.4,hp=50000,spd=0);g.deploy('red',a);g.deploy('red',b)
    ha=hits(a,g,1.6,stop_at_first=True);assert len(ha)==1 and b.hp==50000,(ha,b.hp)
    # the tower holds its nearest target; it dies as the first arrow lands and the farther body is already in range, so the next
    # arrow leaves on the attack period and lands one period after the first (the two flights differ by a few hundredths)
    a.alive=False;a.hp=0;t_kill=g.t
    hb=hits(b,g,t_kill+1.6)
    assert hb and abs(hb[0]-ha[0]-0.8)<=0.15,(ha,hb)


def t_the_king_tower_loads_half_a_second_before_its_first_shot():
    g=Game();k=g.arena.get_tower('blue','king');k.activate()
    for t in g.arena.towers:
        if t.ttype=='princess':t.troop.dmg=0
    g.run(5.0)
    tr=Dummy('red',9.0,9.5,hp=50000,spd=0);g.deploy('red',tr);t0=g.t
    # the king's first attack period is its 1.0 s hit speed less the 0.5 s load; the cannonball has no flight in the engine
    h=hits(tr,g,t0+1.5)
    assert h and 0.45<=h[0]-t0<=0.6,(h,t0)
    assert len(h)>=2 and 0.95<=h[1]-h[0]<=1.05,h


def t_cannoneer_first_shot_timing_is_unchanged():
    g=Game(p1={'tt_name':'cannoneer','tt_lvl':11});tr=Dummy('red',3.0,13.0,hp=50000,spd=0);g.deploy('red',tr)
    h=hits(tr,g,3.6)
    assert h and 1.1<=h[0]<=1.25 and len(h)>=2 and 2.1<=h[1]-h[0]<=2.3,h
