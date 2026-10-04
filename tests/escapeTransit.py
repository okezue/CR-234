from sim import fx
from sim.cards import card,create
from sim.game import Game
from sim.units import has
from tests.util import Dummy,quiet


# Explosive Escape: "After a 1-second delay, the Mighty Miner changes his position to the horizontally mirrored position of himself. He is
# intangible while changing positions. He also drops a bomb in the position he originally was" (wiki Mighty Miner). The engine's cast is
# the generic 1.0 s. Recorded in 09PP9JRPYR9Y (ability played at 71.05): he vanishes and his bomb appears in the same frame 0.37 to 0.40 s
# after the play, he surfaces in the other lane 1.20 to 1.23 s after it and the bomb goes off 1.00 s after it appeared.


def played(x=14.5,y=12.0):
    # the replay's activation: one 0.05 s tick from the play to the cast; towers quiet, the miner rooted; returns the game, the miner and
    # the play time
    g=quiet(Game(p1={'ability_del':0,'ability_std':0}))
    mm=create('mighty_miner',11,'blue',x,y);mm.spd=0;g.deploy('blue',mm)
    g.run(0.5);g.players['blue'].elixir=10;mm.ability.cd=0
    t0=g.t
    assert g.activate_ability('blue',mm)[0]
    return g,mm,t0


def t_escape_dig_delay_is_the_recorded_patch():
    c=card('mighty_miner');b=c['skills']['ability']['skills']['burrow']
    assert b['delay']==0.4 and c['src']['skills.ability.skills.burrow.delay']=='patch:2026-10-03mm',(b,c['src'].get('skills.ability.skills.burrow.delay'))
    assert create('mighty_miner',11,'blue',9,10).ability.dig==0.4


def t_escape_miner_goes_under_with_his_bomb_four_tenths_after_the_play():
    g,mm,t0=played()
    while not has(mm,'burrowed') and g.t<t0+3:g.tick()
    assert has(mm,'burrowed') and round(g.t-t0,2)==0.4,f"The miner should go underground 0.4 s after the play, at {g.t-t0:.2f}"
    assert abs(mm.x-14.5)<1e-9,f"He goes under where he stands, x {mm.x}"
    bombs=[s for s in g.spells if isinstance(s,fx.Timer) and s.name==mm.name]
    assert len(bombs)==1 and (bombs[0].x,bombs[0].y)==(14.5,12.0),"The bomb should appear where he went under, in the same tick"


def t_escape_miner_travels_intangible_and_surfaces_mirrored_when_the_cast_ends():
    g,mm,t0=played()
    xs=[]
    while not has(mm,'burrowed') and g.t<t0+3:g.tick()
    while has(mm,'burrowed') and g.t<t0+3:
        hp=mm.hp;mm.take_damage(500);assert mm.hp==hp,"An underground miner takes no damage"
        xs.append(mm.x);g.tick()
    assert round(g.t-t0,2)==1.0 and abs(mm.x-3.5)<1e-9,f"He should surface at x 3.5 when the 1 s cast ends, at {g.t-t0:.2f} x {mm.x}"
    assert xs==sorted(xs,reverse=True) and 3.5<xs[len(xs)//2]<14.5,f"He should travel underground towards the other lane, {xs}"


def t_escape_ends_a_musketeer_lock_while_he_is_underground():
    # a red Musketeer 5 tiles away holds the miner; from the dig she can neither target nor hurt him, and he surfaces 12 tiles away
    g,mm,t0=played()
    m=create('musketeer',11,'red',14.5,17.0);m.spd=0;g.deploy('red',m)
    g.run(0.2)
    assert m.tgt is mm,f"The Musketeer should hold the miner before the dig, {m.tgt}"
    while not has(mm,'burrowed') and g.t<t0+3:g.tick()
    assert has(mm,'burrowed'),"The miner should go underground during the cast"
    hp=mm.hp
    g.tick()
    assert m.tgt is not mm,"The Musketeer should drop the underground miner"
    g.run_to(t0+1.0)
    assert mm.hp==hp,f"No shot should reach him underground, {hp} -> {mm.hp}"


def t_escape_bomb_goes_off_one_second_after_it_appears():
    # an air body over the miner's spot: he cannot hit it, so only the bomb can; recorded 1.40 to 1.43 s after the play
    g,mm,t0=played()
    a=Dummy('red',14.5,12.0,hp=5000,spd=0,dmg=0);a.targets=[];a.transport='Air';g.deploy('red',a)
    while a.hp==5000 and g.t<t0+3:g.tick()
    assert round(g.t-t0,2)==1.4 and a.hp==5000-332,(round(g.t-t0,2),a.hp)


def t_escape_called_directly_still_switches_at_once():
    # activate() outside a cast (the edge tests) keeps the immediate switch and leaves him tangible
    g=quiet(Game());mm=create('mighty_miner',11,'blue',14.5,12.0);g.deploy('blue',mm)
    mm.ability.activate(mm,g)
    assert abs(mm.x-3.5)<1e-9 and not has(mm,'burrowed'),(mm.x,mm.statuses)
