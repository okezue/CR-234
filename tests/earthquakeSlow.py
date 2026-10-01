from sim.cards import card,create
from sim.game import Game
from sim.units import has
from tests.util import Dummy,quiet


# Earthquake slows enemy troops by 50% over its 3 s (wiki: Duration 3 sec, Slowdown -50%). The slow is the area's buff: the game data export's
# Earthquake areaEffectObjectData re-applies it every hitSpeed 100 ms for buffTime 1000 ms within lifeDuration 3000 (RoyaleAPI cr-api-data:
# hit_speed 100, buff_time 1000, cap_buff_time_to_area_effect_time true), so a troop is slowed while inside and for up to 1 s after leaving,
# never past the spell. The damage stays at its three 1 s ticks.


def quake(bodies=((9,25),)):
    # the spell is cast between ticks, so its first tick (slow and damage) comes with the next one
    g=quiet(Game())
    out=[]
    for bx,by in bodies:
        d=Dummy('red',bx,by,hp=50000,spd=0,dmg=0);g.deploy('red',d);out.append(d)
    g.run(0.5)
    eq=create('earthquake',11,'blue',9,25);eq.apply(g);g.spells.append(eq)
    return g,eq,out,g.t


def slowed(g,d):return has(d,'mslow') and g._status_mods(d)[2]==0.5


def t_earthquake_record_carries_the_export_buff():
    c=card('earthquake');sl=c['skills']['slow']
    assert (sl['duration'],sl['interval'],sl['speedMultiplier'])==(1.0,0.1,-50),sl
    assert c['src']['skills.slow.duration'].startswith('gd:') and c['src']['skills.slow.interval'].startswith('gd:'),c['src']
    eq=create('earthquake',11,'blue',9,25)
    assert (eq.slow_dur,eq.slow_every,eq.life,eq.slow_pct)==(1.0,0.1,3.0,0.5),(eq.slow_dur,eq.slow_every,eq.life,eq.slow_pct)


def t_troop_walking_in_is_slowed_within_a_tenth_of_a_second():
    g,eq,(d,),tc=quake(bodies=[(9,15)])
    g.run_to(tc+0.3)
    assert not slowed(g,d)
    d.y=25
    g.run_to(tc+0.4)
    assert slowed(g,d),f"A troop entering at 0.3 s should be slowed by 0.4 s, statuses {[(s.kind,s.dur) for s in d.statuses]}"


def t_slow_lingers_one_second_after_leaving():
    g,eq,(d,),tc=quake()
    g.run_to(tc+1.5)
    d.y=15
    g.run_to(tc+2.4)
    assert slowed(g,d),"A troop leaving at 1.5 s should stay slowed until 2.5 s"
    g.run_to(tc+2.55)
    assert not slowed(g,d),"The slow should be gone 1 s after leaving"


def t_slow_ends_with_the_spell_and_damage_keeps_its_three_ticks():
    g,eq,(d,),tc=quake()
    hits=[];last=d.hp
    while g.t<tc+3.2:
        g.tick()
        if d.hp!=last:hits.append(round(g.t-tc,2));last=d.hp
        if g.t<tc+2.9:assert slowed(g,d),f"A troop inside should be slowed throughout, t {g.t-tc:.2f}"
    assert not slowed(g,d),"The slow should end with the spell"
    assert len(hits)==3 and hits[0]<=0.05 and abs(hits[1]-hits[0]-1)<0.06 and abs(hits[2]-hits[0]-2)<0.06 and d.hp==50000-3*82,(hits,d.hp)
