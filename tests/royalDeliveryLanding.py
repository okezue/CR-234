from sim.cards import card,create
from sim.game import Game
from sim.units import has
from tests.util import Dummy,quiet


# The box lands with the area's single hit 2 s after the cast: game data export RoyalDeliveryArea lifeDuration 2000 and hitSpeed 2000
# (the export is current: its landing damage 150 at level 1 is the wiki's 4/8/2026 value), RoyaleAPI cr-api-data RoyalDelivery deploy_time
# 2000, ClashStrategic duration 2.0. The wiki's "3 sec" (Deploy Time; "3 second delay") dates from the page's creation on 2/3/2020 with
# no balance entry. The Recruit it drops deploys in 0.25 s (cr-api spawn_character_deploy_time 250; wiki Royal Recruit Deploy Time 0.25).
LAND=2.0


def cast(x=9,y=10,bodies=()):
    # quiet towers, still enemy bodies that do not attack; returns the game, the spell, the bodies and the cast time
    g=quiet(Game())
    out=[]
    for bx,by in bodies:
        d=Dummy('red',bx,by,hp=5000,spd=0,dmg=0);g.deploy('red',d);out.append(d)
    g.run(0.5)
    rd=create('royal_delivery',11,'blue',x,y);g._cast('blue',rd,x,y)
    return g,rd,out,g.t


def recruits(g):return [t for t in g.players['blue'].troops if t.alive]


def t_royal_delivery_record_carries_the_export_landing_and_the_recruit_deploy():
    c=card('royal_delivery')
    assert c['duration']==LAND and c['src']['duration'].startswith('gd:'),(c['duration'],c['src'].get('duration'))
    assert c['skills']['spawn']['deployTime']==0.25 and c['src']['skills.spawn.deployTime']=='patch:2026-10-01s',c['skills']['spawn']
    rd=create('royal_delivery',11,'blue',9,10)
    assert rd.delay==LAND and rd.tcfg['deploy']==0.25,(rd.delay,rd.tcfg['deploy'])


def t_box_lands_two_seconds_after_the_cast():
    g,rd,(d,),tc=cast(bodies=[(9,10)])
    assert d.hp==5000 and not recruits(g),"Nothing should happen in the cast tick"
    while d.hp==5000 and g.t<tc+4:
        assert not recruits(g),f"The Recruit should not be there before the landing, t {g.t-tc:.2f}"
        g.tick()
    assert round(g.t-tc,2)==LAND and d.hp==5000-384,(round(g.t-tc,2),d.hp)
    assert len(recruits(g))==1,"The Recruit should drop with the box"


def t_landing_hits_whoever_is_under_the_box_then():
    # one body leaves the area and another walks in during the fall
    g,rd,(gone,came),tc=cast(bodies=[(9,10),(3,4)])
    g.run_to(tc+1.0)
    gone.x,gone.y=9,20;came.x,came.y=9.5,10.5
    g.run_to(tc+LAND+0.05)
    assert gone.hp==5000 and came.hp==5000-384,f"Landing damage goes to the body there at {LAND} s: gone {gone.hp}, came {came.hp}"


def t_recruit_deploys_a_quarter_second_after_the_landing():
    # a body beside the drop point: the box hits it at the landing, the Recruit's first swing (first hit 0.5 s) follows its 0.25 s deploy
    g,rd,(d,),tc=cast(bodies=[(9,11.2)])
    g.run_to(tc+LAND)
    (r,)=recruits(g)
    assert has(r,'deploying') and r.shield_hp==240 and d.hp==5000-384,(r.statuses,r.shield_hp,d.hp)
    g.run_to(tc+LAND+0.25)
    assert not has(r,'deploying'),"The Recruit's deploy should be over 0.25 s after the landing"
    g.run_to(tc+LAND+0.65)
    assert d.hp==5000-384,f"No Recruit hit before its deploy and first hit, hp {d.hp}"
    g.run_to(tc+LAND+0.75)
    assert d.hp==5000-384-r.dmg,f"The Recruit's first hit should land about 0.7 s after the box, hp {d.hp}"
