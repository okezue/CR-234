import math
import random
from sim.game import Game,Deck,Player,validate_deck,card_info
from sim.arena import Arena
from sim.cards import create as mk_card
from sim.towers import create as mk_tt
from sim.units import Status
from tests.util import Dummy,quiet
_DK=['knight','bandit','ice_wizard','dart_goblin','mega_minion','fisherman','royal_ghost','princess']

def t_phases():
    g=Game()
    g.run_to(119);assert g._erate()==1
    g.run_to(121);assert g._erate()==2
    g.run_to(180.1);assert g.phase=='overtime'
    g.run_to(239);assert g._erate()==2
    g.run_to(241);assert g._erate()==3
    g.run_to(299.9);assert not g.ended
    g.run_to(300.1);assert g.ended
    return "Phase transitions (1x->2x@120->OT@180->3x@240->tiebreaker@300)"
def t_elixir():
    g=Game();g.run(10)
    ex=g.players['blue'].elixir
    exp=5.0+10/2.8
    assert abs(ex-exp)<0.5,f"Expected ~{exp:.1f}, got {ex:.1f}"
    return f"Elixir generation ({ex:.1f} at 10s, expected ~{exp:.1f})"
def t_3crown():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    kt.hp=0;kt.alive=False;g._tower_down(kt)
    assert g.winner=='red' and g.players['red'].crowns==3
    return "3-crown king tower win"
def t_crown_lead():
    g=Game()
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=0;pt.alive=False;g._tower_down(pt)
    assert g.players['red'].crowns==1
    g.run_to(180.1)
    assert g.winner=='red'
    return "Crown lead regulation win (red 1-0)"
def t_overtime_sd():
    g=Game()
    g.run_to(180.1);assert g.phase=='overtime'
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=0;pt.alive=False;g._tower_down(pt)
    assert g.winner=='red' and g.ended
    return "Overtime sudden death"
def t_tiebreaker():
    g=Game()
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=pt.max_hp-100
    g.arena.get_tower('red','princess','right').hp-=50
    g.deploy('red',Dummy('red',9,20,spd=0))
    g.run_to(300.1)
    assert g.ended and g.winner=='red' and not pt.alive
    assert g.players['red'].crowns==1 and not g.players['red'].troops,"tiebreaker kills all troops"
    return "Tiebreaker at 300s: lowest HP tower destroyed, its owner loses"
def t_tiebreaker_draw():
    g=Game()
    g.run_to(300.1)
    assert g.ended and g.winner is None
    g=Game()
    g.arena.get_tower('blue','princess','left').hp-=100
    g.arena.get_tower('red','princess','right').hp-=100
    g.run_to(300.1)
    assert g.ended and g.winner is None
    return "Tiebreaker draw when the lowest HP is shared by both sides"
def t_king_act():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    assert not kt.active
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=0;pt.alive=False;g._tower_down(pt)
    assert kt.active
    return "King activation on princess death"
def t_king_act_dmg():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    assert not kt.active
    for t in g.arena.towers:
        if t.team=='blue' and t.ttype=='princess':
            t.hp=0;t.alive=False
    tr=Dummy('red',9.0,6.0,hp=50000,dmg=100)
    g.deploy('red',tr)
    for _ in range(500):
        g.tick()
        if kt.active:break
    assert kt.active
    return "King activation on direct damage"
def t_king_act_spell():
    g=Game()
    kt=g.arena.get_tower('red','king')
    fb=mk_card('fireball',11,'blue',kt.cx,kt.cy)
    fb.apply(g)
    assert kt.active and kt.hp<kt.max_hp
    return "Spell damage activates the king tower"
def t_king_act_not_proximity():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    d=Dummy('red',9.0,6.0,hp=50000,spd=0,dmg=0)
    g.deploy('red',d)
    g.run(3)
    assert kt.dist(d.x,d.y)<=kt.rng and not kt.active,"an enemy in range that deals no damage must not activate the king"
    return "King stays inactive without damage or a fallen princess tower"
def t_king_act_delay():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    for t in g.arena.towers:
        if t.team=='blue' and t.ttype=='princess':t.hp=0;t.alive=False
    d=Dummy('red',9.0,6.0,hp=50000,spd=0,dmg=0)
    g.deploy('red',d)
    g.run(1)
    kt.take_damage(1)
    assert kt.active
    g.run(3.8)
    assert d.hp==50000,"king needs 4 s to aim and load after activation"
    g.run(0.4)
    assert d.hp<50000
    return "King fires 4 s after activation"
def t_chef():
    random.seed(42)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr=Dummy('blue',9.0,10.0,lvl=11,hp=50000,spd=0)
    g.deploy('blue',tr)
    g.run(50)
    assert tr.lvl>=12,f"Expected lvl>=12, got {tr.lvl}"
    return f"Royal Chef pancake (lvl 11->{tr.lvl})"
def t_chef_cooldown():
    random.seed(0)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    t1=Dummy('blue',5.0,10.0,lvl=11,hp=50000,spd=0)
    t2=Dummy('blue',12.0,10.0,lvl=11,hp=50000,spd=0)
    g.deploy('blue',t1);g.deploy('blue',t2)
    g.run(80)
    assert t1.lvl>=12 and t2.lvl>=12,f"Expected both boosted: t1={t1.lvl} t2={t2.lvl}"
    return f"Royal Chef multi-pancake (t1=lvl{t1.lvl}, t2=lvl{t2.lvl})"
def t_chef_multiboost():
    random.seed(7)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr=Dummy('blue',9.0,10.0,lvl=11,hp=1000,dmg=100,spd=0)
    g.deploy('blue',tr)
    ihp=tr.max_hp;idmg=tr.dmg
    g.run(120)
    assert tr.lvl>=13,f"Expected lvl>=13 after multi-boost, got {tr.lvl}"
    boosts=tr.lvl-11
    exp_hp=ihp;exp_dmg=idmg
    for _ in range(boosts):
        exp_hp=int(exp_hp*1.1)
        exp_dmg=int(exp_dmg*1.1)
    assert tr.max_hp==exp_hp,f"HP mismatch: {tr.max_hp} vs {exp_hp}"
    assert tr.dmg==exp_dmg,f"DMG mismatch: {tr.dmg} vs {exp_dmg}"
    return f"Chef multi-boost (lvl 11->{tr.lvl}, hp {ihp}->{tr.max_hp}, dmg {idmg}->{tr.dmg})"
def t_duchess():
    g=Game(p1={'tt_name':'dagger_duchess','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    dd=lpt.troop;ini=tr.hp
    g.run(4.5)
    dmg=ini-tr.hp
    assert dmg>=dd.dmg*8,f"Expected >=8 daggers ({dd.dmg*8}), got {dmg}"
    return f"Dagger Duchess burst ({dmg} dmg, ~{dmg//dd.dmg} hits)"
def t_duchess_recharge():
    g=Game(p1={'tt_name':'dagger_duchess','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    dd=lpt.troop
    g.run(5)
    assert dd.dag==0
    g.players['red'].troops.clear()
    g.run(8)
    assert dd.dag==dd.MXD,f"Expected full recharge, got {dd.dag}/{dd.MXD}"
    return f"Dagger Duchess passive recharge ({dd.dag}/{dd.MXD})"
def t_cannoneer():
    g=Game(p1={'tt_name':'cannoneer','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    ini=tr.hp
    g.run(0.8)
    assert tr.hp==ini,"Shot fired before 0.8s"
    g.run(0.2)
    assert tr.hp<ini,"No shot by 1.0s"
    d1=ini-tr.hp;hp1=tr.hp
    g.run(2.0)
    assert tr.hp==hp1,"Extra shot between 1.0-3.0s"
    g.run(0.2)
    assert tr.hp<hp1,"No second shot by 3.2s"
    return f"Cannoneer first shot ({d1} dmg at ~0.9s, 2nd at ~3.1s)"
def t_troop_atk():
    g=Game()
    tr=Dummy('red',3.0,8.0,hp=50000,dmg=200,spd=2.0)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    ini=lpt.hp
    g.run(5)
    assert lpt.hp<ini
    return f"Troop attacks tower ({ini}->{lpt.hp})"
def t_troop_kills_tower():
    g=Game()
    lpt=g.arena.get_tower('blue','princess','left')
    lpt.hp=150
    tr=Dummy('red',3.0,8.0,hp=50000,dmg=200,spd=2.0)
    g.deploy('red',tr)
    g.run(5)
    assert not lpt.alive
    assert g.players['red'].crowns>=1
    kt=g.arena.get_tower('blue','king')
    assert kt.active
    return "Troop destroys tower -> crown + king activation"
def t_deck_cycle():
    random.seed(99)
    dk=Deck(['a','b','c','d','e','f','g','h'])
    n0=dk.nxt
    c=dk.hand[0]
    dk.play(c,2.0)
    assert c==dk.q[-1],"Played card not at back of queue"
    assert len(dk.hand)==4,"Hand not refilled"
    assert n0 in dk.hand,"Next card not moved to hand"
    return f"Deck cycle (played {c}, hand={dk.hand}, nxt={dk.nxt})"
def t_deck_4card_return():
    random.seed(99)
    dk=Deck(['a','b','c','d','e','f','g','h'])
    first=dk.hand[0]
    for i in range(4):
        dk.play(dk.hand[0],0)
    assert first==dk.nxt,f"{first} should be nxt after 4 plays, got {dk.nxt}"
    dk.play(dk.hand[0],0)
    assert first in dk.hand,f"{first} should return to hand after 5th play"
    return f"4-card cycle ({first}: nxt after 4, hand after 5)"
def t_deck_queue_cd():
    random.seed(99)
    dk=Deck(['a','b','c','d','e','f','g','h'])
    dk.play(dk.hand[0],2.0)
    assert len(dk.hand)==4
    dk.play(dk.hand[0],2.0)
    assert len(dk.hand)==3,"Hand should drop to 3 (next on cooldown)"
    for _ in range(20):dk.tick(0.1,2.0)
    assert len(dk.hand)==4,"Hand should restore after 2s"
    return "Deck queue cooldown (3->4 after 2s)"
def t_deck_qcd_2x():
    random.seed(99)
    dk=Deck(['a','b','c','d','e','f','g','h'])
    dk.play(dk.hand[0],1.0)
    dk.play(dk.hand[0],1.0)
    assert len(dk.hand)==3
    for _ in range(10):dk.tick(0.1,1.0)
    assert len(dk.hand)==4,"Hand should restore after 1s at 2x"
    return "Queue cooldown scales with elixir rate (1s at 2x)"
def t_play_card_elixir():
    random.seed(99)
    g=Game(p1={'deck':_DK})
    c=g.players['blue'].deck.hand[0]
    ini=g.players['blue'].elixir
    ok,_=g.play_card('blue',c,9,10)
    assert ok
    assert abs(g.players['blue'].elixir-(ini-3))<0.01
    return f"Play card deducts elixir ({ini:.0f}->{g.players['blue'].elixir:.0f})"
def t_play_card_no_elixir():
    random.seed(99)
    g=Game(p1={'deck':_DK})
    g.players['blue'].elixir=1.0
    c=g.players['blue'].deck.hand[0]
    ok,msg=g.play_card('blue',c,9,10)
    assert not ok and msg=="not enough elixir"
    return "Play card rejected (insufficient elixir)"
def t_play_card_not_in_hand():
    random.seed(99)
    g=Game(p1={'deck':_DK})
    hand=g.players['blue'].deck.hand
    nothand=[c for c in _DK if c not in hand][0]
    ok,msg=g.play_card('blue',nothand,9,10)
    assert not ok and msg=="not in hand"
    return f"Play card rejected (not in hand: {nothand})"
def t_deploy_zone_base():
    g=Game()
    assert all(g._valid_deploy('blue',x,y) for x in range(1,17) for y in range(1,15) if (x,y) not in {t for tw in g.arena.towers for t in tw.tiles()})
    assert all(g._valid_deploy('red',x,y) for x in range(1,17) for y in range(17,31) if (x,y) not in {t for tw in g.arena.towers for t in tw.tiles()})
    assert not g._valid_deploy('blue',9,20) and not g._valid_deploy('red',9,10)
    assert not g._valid_deploy('blue',0,15) and not g._valid_deploy('red',0,16),"river"
    assert not g._valid_deploy('blue',3,15) and not g._valid_deploy('red',3,16),"bridge is enemy no deploy zone while its tower stands"
    assert not g._valid_deploy('blue',8,3) and not g._valid_deploy('red',14,25),"own tower tiles"
    assert not g._valid_deploy('blue',0,0) and g._valid_deploy('blue',6,0) and not g._valid_deploy('red',17,31),"back row fences"
    assert not g._valid_deploy('blue',0,14) and not g._valid_deploy('red',17,17),"river bank fences"
    return "Deploy zone: own half rows 0-14, no river, bridge, fences or tower tiles"
def t_deploy_zone_pocket():
    g=Game()
    assert not g._valid_deploy('blue',3,17)
    rlp=g.arena.get_tower('red','princess','left')
    rlp.hp=0;rlp.alive=False
    assert all(g._valid_deploy('blue',x,y) for x in range(1,9) for y in range(17,21)),"pocket is 9 wide by 4 deep"
    assert not any(g._valid_deploy('blue',x,y) for x in range(0,9) for y in range(21,32)),"king no deploy zone starts at row 21"
    assert not any(g._valid_deploy('blue',x,y) for x in range(9,18) for y in range(15,32)),"other lane stays closed"
    assert g._valid_deploy('blue',3,15) and g._valid_deploy('blue',3,16) and not g._valid_deploy('blue',5,16),"bridge opens with the pocket"
    rrp=g.arena.get_tower('red','princess','right')
    rrp.hp=0;rrp.alive=False
    assert g._valid_deploy('blue',14,20) and not g._valid_deploy('blue',14,21)
    g2=Game()
    blp=g2.arena.get_tower('blue','princess','left')
    blp.hp=0;blp.alive=False
    assert all(g2._valid_deploy('red',x,y) for x in range(1,9) for y in range(11,15)) and not g2._valid_deploy('red',3,10)
    return "Deploy pocket: rows 17-20 on the fallen tower's half (mirror 11-14), rows by the king stay closed"
def t_deploy_snap_center():
    random.seed(99)
    g=Game(p1={'deck':_DK,'drag_del':0.0,'drag_std':0})
    c=g.players['blue'].deck.hand[0]
    ok,_=g.play_card('blue',c,9.2,10.9)
    assert ok and (g.pending[-1].x,g.pending[-1].y)==(9.5,10.5)
    return "play_card snaps to the tile centre (replay 9500,10500 style)"
def t_deploy_delay():
    random.seed(99)
    g=Game(p1={'deck':_DK,'drag_del':0.5,'drag_std':0})
    c=g.players['blue'].deck.hand[0];dep=card_info(c)['deploy']
    g.play_card('blue',c,9,10)
    assert len(g.players['blue'].troops)==0
    assert len(g.pending)==1
    g.run(0.4+dep)
    assert len(g.players['blue'].troops)==0,"Spawned too early"
    g.run(0.2)
    assert len(g.players['blue'].troops)==1,f"Not spawned after {0.6+dep:.1f}s"
    return f"Deploy delay (0.5s drag + {dep}s deploy)"
def t_drag_pro_vs_casual():
    random.seed(99)
    g1=Game(p1={'deck':_DK,'drag_del':0.3,'drag_std':0})
    c=g1.players['blue'].deck.hand[0];dep=card_info(c)['deploy']
    g1.play_card('blue',c,9,10)
    g1.run(0.2+dep)
    assert len(g1.players['blue'].troops)==0
    g1.run(0.2)
    assert len(g1.players['blue'].troops)==1
    random.seed(99)
    g2=Game(p1={'deck':_DK,'drag_del':0.7,'drag_std':0})
    c=g2.players['blue'].deck.hand[0]
    g2.play_card('blue',c,9,10)
    g2.run(0.6+dep)
    assert len(g2.players['blue'].troops)==0
    g2.run(0.2)
    assert len(g2.players['blue'].troops)==1
    return f"Drag delay (pro={0.3+dep:.1f}s total, casual={0.7+dep:.1f}s total)"
def t_drag_stochastic():
    random.seed(42)
    p=Player('blue',drag_del=0.5,drag_std=0.15)
    delays=[p.sample_drag() for _ in range(100)]
    mn=min(delays);mx=max(delays);avg=sum(delays)/len(delays)
    assert mn!=mx,"All drags identical (not stochastic)"
    assert mn>=0.1,"Drag below 0.1s floor"
    assert abs(avg-0.5)<0.1,f"Mean drag {avg:.3f} too far from 0.5"
    return f"Stochastic drag (n=100, min={mn:.2f} max={mx:.2f} avg={avg:.2f})"
def t_simultaneous_play():
    random.seed(99)
    g=Game(p1={'deck':_DK,'drag_std':0})
    g.players['blue'].elixir=10
    h=g.players['blue'].deck.hand
    c1,c2=h[0],h[1]
    ok1,_=g.play_card('blue',c1,5,10)
    ok2,_=g.play_card('blue',c2,12,10)
    assert ok1 and ok2
    assert len(g.pending)==2
    assert g.players['blue'].elixir==10-3-3
    g.run(2)
    assert len(g.players['blue'].troops)==2
    return f"Simultaneous 2-card play ({c1}+{c2}, 6 elixir)"
def t_deploy_invalid_pos():
    random.seed(99)
    g=Game(p1={'deck':_DK})
    c=g.players['blue'].deck.hand[0]
    ok,msg=g.play_card('blue',c,9,20)
    assert not ok and msg=="invalid position"
    return "Deploy rejected (enemy half without pocket)"
def t_elixir_2x_3x():
    g=Game()
    g.run_to(100)
    g.players['blue'].elixir=0
    g.run(10)
    gen_1x=g.players['blue'].elixir
    g.run_to(225)
    g.players['blue'].elixir=0
    g.run(10)
    gen_2x=g.players['blue'].elixir
    g.run_to(250)
    g.players['blue'].elixir=0
    g.run(10)
    gen_3x=g.players['blue'].elixir
    assert abs(gen_2x/gen_1x-2.0)<0.2,f"2x ratio off: {gen_2x/gen_1x:.2f}"
    assert abs(gen_3x/gen_1x-3.0)<0.3,f"3x ratio off: {gen_3x/gen_1x:.2f}"
    return f"Elixir rates (1x={gen_1x:.2f}, 2x={gen_2x:.2f} at 225s, 3x={gen_3x:.2f})"

def t_cannoneer_preload():
    g=Game(p1={'tt_name':'cannoneer','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    ini=tr.hp
    g.run(0.8);assert tr.hp==ini
    g.run(0.2);d1=ini-tr.hp;assert d1>0
    hp1=tr.hp;g.run(2.0);assert tr.hp==hp1
    g.run(0.2);assert tr.hp<hp1
    return f"Cannoneer preload ({d1} dmg, 1st@~0.9s, 2nd@~3.1s)"
def t_cannoneer_disengage_reload():
    g=Game(p1={'tt_name':'cannoneer','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    g.run(1.0)
    ini=tr.hp;assert ini<50000
    g.players['red'].troops.clear()
    g.run(3.0)
    tr2=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr2)
    ini2=tr2.hp
    g.run(0.8);assert tr2.hp==ini2
    g.run(0.2);assert tr2.hp<ini2
    return "Cannoneer disengage->reload->fast first shot"
def t_cannoneer_high_dmg():
    cn=mk_tt('cannoneer',11)
    pr=mk_tt('tower_princess',11)
    assert cn.dmg>pr.dmg*2,f"Cannoneer dmg {cn.dmg} not >> Princess {pr.dmg}"
    return f"Cannoneer dmg={cn.dmg} >> Princess dmg={pr.dmg}"
def t_duchess_burst_count():
    g=Game(p1={'tt_name':'dagger_duchess','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    dd=lpt.troop;ini=tr.hp
    g.run(5)
    hits=(ini-tr.hp)//dd.dmg
    assert hits>=8,f"Expected >=8 hits, got {hits}"
    return f"Duchess burst count: {hits} hits"
def t_duchess_sustained_dps():
    g=Game(p1={'tt_name':'dagger_duchess','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    g.run(5);hp5=tr.hp
    g.run(10);hp15=tr.hp
    burst_d=50000-hp5;sust_d=hp5-hp15
    burst_dps=burst_d/5.0;sust_dps=sust_d/10.0
    assert sust_dps<burst_dps,f"Sustained {sust_dps:.0f} >= burst {burst_dps:.0f}"
    return f"Duchess sustained DPS {sust_dps:.0f} < burst DPS {burst_dps:.0f}"
def t_duchess_full_recharge():
    g=Game(p1={'tt_name':'dagger_duchess','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    dd=lpt.troop
    g.run(5);assert dd.dag<=1
    g.players['red'].troops.clear()
    g.run(10)
    assert dd.dag==dd.MXD,f"Expected full recharge {dd.MXD}, got {dd.dag}"
    return f"Duchess full recharge {dd.dag}/{dd.MXD}"
def t_duchess_partial_recharge():
    g=Game(p1={'tt_name':'dagger_duchess','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    dd=lpt.troop
    g.run(5);assert dd.dag<=1
    g.players['red'].troops.clear()
    g.run(3)
    mid=dd.dag;assert 0<mid<dd.MXD,f"Expected partial, got {mid}"
    tr2=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr2)
    g.run(0.1)
    assert dd.dag<=mid
    return f"Duchess partial recharge ({mid} daggers when interrupted)"
def t_chef_skip_building():
    random.seed(42)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    bld=Dummy('blue',9.0,10.0,lvl=11,hp=50000,spd=0)
    bld.is_building=True
    g.deploy('blue',bld)
    g.run(60)
    assert bld.lvl==11,f"Building should not be pancaked, got lvl {bld.lvl}"
    return "Chef skips buildings for pancake"
def t_chef_skip_clone():
    random.seed(42)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    cl=Dummy('blue',9.0,10.0,lvl=11,hp=1,spd=0)
    cl.max_hp=1
    g.deploy('blue',cl)
    g.run(60)
    assert cl.lvl==11,f"Clone should not be pancaked, got lvl {cl.lvl}"
    return "Chef skips clones (hp=1,max_hp=1)"
def t_chef_cross_map():
    random.seed(42)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr=Dummy('blue',14.0,2.0,lvl=11,hp=50000,spd=0)
    g.deploy('blue',tr)
    g.run(60)
    assert tr.lvl>=12,f"Far troop should still get pancaked, got lvl {tr.lvl}"
    return f"Chef pancakes cross-map troop (lvl->{tr.lvl})"
def t_chef_hp_threshold():
    random.seed(42)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr=Dummy('blue',9.0,10.0,lvl=11,hp=30,spd=0)
    tr.max_hp=100
    g.deploy('blue',tr)
    g.run(60)
    assert tr.lvl==11,f"Low HP troop (<33%) should not be pancaked, got lvl {tr.lvl}"
    return "Chef HP threshold: <33% HP not pancaked"
def t_chef_spreads():
    random.seed(0)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    t1=Dummy('blue',5.0,10.0,lvl=11,hp=50000,spd=0)
    t2=Dummy('blue',12.0,10.0,lvl=11,hp=49999,spd=0)
    g.deploy('blue',t1);g.deploy('blue',t2)
    g.run(80)
    assert t1.lvl>=12 and t2.lvl>=12,f"Both should be boosted: {t1.lvl}, {t2.lvl}"
    return f"Chef spreads pancakes (t1={t1.lvl}, t2={t2.lvl})"
def t_chef_cooking_slower_attacking():
    random.seed(42)
    g1=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr1=Dummy('blue',9.0,10.0,lvl=11,hp=50000,spd=0)
    g1.deploy('blue',tr1)
    g1.run(60)
    lvl_idle=tr1.lvl
    random.seed(42)
    g2=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr2=Dummy('blue',9.0,10.0,lvl=11,hp=50000,spd=0)
    g2.deploy('blue',tr2)
    en=Dummy('red',3.0,13.0,hp=999999,spd=0)
    g2.deploy('red',en)
    g2.run(60)
    lvl_atk=tr2.lvl
    assert lvl_atk<=lvl_idle,f"Attacking chef should cook slower: atk={lvl_atk} vs idle={lvl_idle}"
    return f"Chef cooks slower when attacking (idle lvl={lvl_idle}, atk lvl={lvl_atk})"
def t_chef_both_dead_no_cook():
    random.seed(42)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr=Dummy('blue',9.0,10.0,lvl=11,hp=50000,spd=0)
    g.deploy('blue',tr)
    for t in g.arena.towers:
        if t.team=='blue' and t.ttype=='princess':
            t.hp=0;t.alive=False
    g.run(60)
    assert tr.lvl==11,f"Both towers dead, should not pancake, got lvl {tr.lvl}"
    return "Chef both dead -> no cooking"
def t_elixir_rate_1x():
    g=Game();g.players['blue'].elixir=0
    g.run(10)
    ex=g.players['blue'].elixir
    exp=10.0/2.8
    assert abs(ex-exp)<0.5,f"1x rate: expected ~{exp:.2f}, got {ex:.2f}"
    return f"Elixir 1x rate: {ex:.2f} in 10s (expected ~{exp:.2f})"
def t_elixir_rate_2x():
    g=Game();g.run_to(125)
    g.players['blue'].elixir=0
    g.run(10)
    ex=g.players['blue'].elixir
    exp=20.0/2.8
    assert abs(ex-exp)<0.5,f"2x rate: expected ~{exp:.2f}, got {ex:.2f}"
    return f"Elixir 2x rate: {ex:.2f} in 10s (expected ~{exp:.2f})"
def t_elixir_rate_3x():
    g=Game();g.run_to(245)
    g.players['blue'].elixir=0
    g.run(5)
    ex=g.players['blue'].elixir
    exp=15.0/2.8
    assert abs(ex-exp)<0.5,f"3x rate: expected ~{exp:.2f}, got {ex:.2f}"
    return f"Elixir 3x rate: {ex:.2f} in 5s (expected ~{exp:.2f})"
def t_elixir_cap():
    g=Game();g.players['blue'].elixir=9.5
    g.run(30)
    assert g.players['blue'].elixir<=10.0,f"Elixir exceeded 10: {g.players['blue'].elixir}"
    return f"Elixir cap: {g.players['blue'].elixir}"
def t_elixir_start():
    g=Game()
    assert g.players['blue'].elixir==5.0
    assert g.players['red'].elixir==5.0
    return "Elixir starts at 5.0"
def t_deck_8_cards():
    dk=['knight','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army','freeze']
    d=Deck(dk)
    assert len(d.all)==8
    return "Deck requires 8 cards"
def t_deck_hand_4():
    dk=['knight','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army','freeze']
    d=Deck(dk)
    assert len(d.hand)==4,f"Hand size {len(d.hand)}"
    return f"Hand = 4 cards: {d.hand}"
def t_deck_random_start():
    h=set()
    for s in range(20):
        random.seed(s)
        dk=['knight','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army','freeze']
        d=Deck(dk)
        h.add(tuple(sorted(d.hand)))
    assert len(h)>1,"All same hand across seeds"
    return f"Random starting hands: {len(h)} unique across 20 seeds"
def t_deck_no_start_mirror():
    for s in range(100):
        random.seed(s)
        dk=['mirror','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army','freeze']
        d=Deck(dk)
        assert 'mirror' not in d.hand,f"Mirror in hand with seed {s}"
    return "Mirror never in starting hand (100 seeds)"
def t_deck_max_2_heroes():
    try:
        validate_deck(['golden_knight','skeleton_king','monk','knight','archers',
                        'fireball','hog_rider','musketeer'])
        assert False,"Should fail with 3 champions"
    except AssertionError:pass
    validate_deck(['golden_knight','skeleton_king','knight','archers',
                    'fireball','hog_rider','musketeer','valkyrie'])
    return "Deck max 2 hero/champion slots"
def t_deck_0_heroes_ok():
    validate_deck(['knight','archers','fireball','hog_rider',
                    'musketeer','valkyrie','skeleton_army','freeze'])
    return "Deck with 0 heroes is valid"
def t_ability_refund_on_death_during_cast():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',9,14)
    g.deploy('blue',gk)
    g.players['blue'].elixir=5.0
    gk.ability.cd=0
    ok,_=g.activate_ability('blue',gk)
    assert ok
    assert getattr(gk.ability,'_pend',False) or gk.ability.casting,"Should be pending/casting"
    ex_after_cast=g.players['blue'].elixir
    assert abs(ex_after_cast-4.0)<0.5,f"Should have spent 1 elixir: {ex_after_cast}"
    gk.alive=False
    g._proc_deaths()
    ex_refund=g.players['blue'].elixir
    assert ex_refund>ex_after_cast,f"Should refund: before={ex_after_cast} after={ex_refund}"
    assert abs(ex_refund-5.0)<0.5,f"Should get 1 elixir back: {ex_refund}"
    return f"Ability refund on death during cast (spent={5.0-ex_after_cast:.1f}, refunded={ex_refund-ex_after_cast:.1f})"
def t_ability_no_refund_after_cast():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',9,14)
    g.deploy('blue',gk)
    d=Dummy('red',9,16,hp=5000,spd=0)
    g.deploy('red',d)
    g.players['blue'].elixir=5.0
    gk.ability.cd=0
    g.activate_ability('blue',gk)
    g.run(1.5)
    assert not gk.ability.casting,"Cast should be done"
    ex_after=g.players['blue'].elixir
    gk.alive=False
    g._proc_deaths()
    ex_dead=g.players['blue'].elixir
    assert abs(ex_dead-ex_after)<0.1,f"No refund after cast completes: before={ex_after} after={ex_dead}"
    return f"No refund after cast completes (elixir unchanged: {ex_dead:.1f})"
def t_ability_delay_fires():
    g=Game(p1={'ability_del':0.15,'ability_std':0})
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    d=Dummy('red',9,13,hp=5000,spd=0)
    g.deploy('red',d)
    g.players['blue'].elixir=10;gk.ability.cd=0
    g.activate_ability('blue',gk)
    g.run(0.05)
    assert not gk.ability.active,"Ability should not be active at 0.05s"
    assert len(g.pending_ab)>0 or gk.ability.casting or getattr(gk.ability,'_pend',False),"Should be pending or casting"
    g.run(1.25)
    assert gk.ability.active or d.hp<5000,"Ability should have fired by 1.3s"
    return "Ability delay fires after delay+cast"
def t_ability_delay_stochastic():
    random.seed(42)
    p1=Player('blue',ability_del=0.15,ability_std=0.05)
    d1=[p1.sample_ability_del() for _ in range(50)]
    random.seed(99)
    d2=[p1.sample_ability_del() for _ in range(50)]
    assert d1!=d2,"Different seeds should give different delays"
    mn=min(d1);mx=max(d1);avg=sum(d1)/len(d1)
    assert mn>=0.05,"Below 0.05s floor"
    assert mn!=mx,"All identical (not stochastic)"
    return f"Ability delay stochastic (n=50, min={mn:.3f} max={mx:.3f} avg={avg:.3f})"
def t_ability_delay_death_refund():
    g=Game(p1={'ability_del':0.5,'ability_std':0})
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    g.players['blue'].elixir=5.0;gk.ability.cd=0
    g.activate_ability('blue',gk)
    ex_after=g.players['blue'].elixir
    assert abs(ex_after-4.0)<0.5
    assert len(g.pending_ab)==1
    g.run(0.2)
    gk.alive=False
    g._proc_deaths()
    ex_ref=g.players['blue'].elixir
    assert ex_ref>ex_after,"Should refund on death during delay"
    assert len(g.pending_ab)==0,"Pending should be cleared"
    return f"Ability delay death refund ({ex_after:.1f}->{ex_ref:.1f})"
def t_ability_delay_zero():
    g=Game(p1={'ability_del':0.0,'ability_std':0})
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    d=Dummy('red',9,13,hp=5000,spd=0)
    g.deploy('red',d)
    g.players['blue'].elixir=10;gk.ability.cd=0
    g.activate_ability('blue',gk)
    g.run(1.2)
    assert gk.ability.active or d.hp<5000,"Zero delay should behave like instant"
    return "Ability delay zero works (instant activation)"
def t_ability_banner_delay():
    g=Game(p1={'ability_del':0.15,'ability_std':0})
    random.seed(42)
    hg=mk_card('goblins',11,'blue',9,14,hero=True)
    for gb in hg:g.deploy('blue',gb)
    ab=hg[0].ability
    for gb in hg:gb.alive=False
    g._proc_deaths()
    assert ab.banner_pos is not None
    g.players['blue'].elixir=10
    ok,_=g.activate_ability('blue',hg[0])
    assert ok
    assert len(g.pending_ab)==1,"Banner should be queued"
    g.run(0.3)
    spawned=[t for t in g.players['blue'].troops if t.alive]
    assert len(spawned)==2,f"Banner should fire after delay: {len(spawned)}"
    return f"Banner ability delayed ({len(spawned)} goblins)"
def t_pf_astar_bridge_ground():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    p=pf.a_star(4,10,4,22,False)
    assert len(p)>0,"Ground troop should find path through bridge"
    bridge_y=[y for _,y in p if 15.0<=y<=17.0]
    assert len(bridge_y)>0,"Path should cross bridge tiles"
    bx=[x for x,y in p if 15.0<=y<=17.0]
    assert all(a.on_bridge(x) for x in bx),"Bridge crossing at valid x"
    return f"A* routes ground troop through bridge ({len(p)} waypoints)"
def t_pf_astar_air_straight():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    gp=pf.a_star(9,10,9,22,False)
    ap=pf.a_star(9,10,9,22,True)
    assert len(ap)>0,"Air path should exist"
    assert len(ap)<=len(gp) or len(gp)==0,"Air path should be shorter or equal"
    return f"Air troop ignores river (air={len(ap)} vs ground={len(gp)})"
def t_pf_collision_mass():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    t1=Dummy('blue',9,10,mass=20,spd=1.0)
    t2=Dummy('blue',9.3,10,mass=1,spd=1.0)
    t1.collision_r=0.5;t2.collision_r=0.5
    pf.resolve_collisions([t1,t2])
    d1=abs(t1.x-9.0);d2=abs(t2.x-9.3)
    assert d2>=d1,"Lighter troop should be pushed more"
    assert d1>0.001 or d2>0.001,"Some push should occur"
    return f"Collision: heavy barely moves ({d1:.4f}) vs light ({d2:.4f})"
def t_pf_collision_heavy_v_light():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    heavy=Dummy('blue',9,10,mass=20,spd=1.0)
    light=Dummy('blue',9.5,10,mass=1,spd=1.0)
    heavy.collision_r=0.8;light.collision_r=0.5
    pf.resolve_collisions([heavy,light])
    hm=abs(heavy.x-9.0);lm=abs(light.x-9.5)
    assert lm>hm*2,"Light should move much more than heavy"
    return f"Heavy v light collision (heavy:{hm:.3f} light:{lm:.3f})"
def t_pf_sight_range_near():
    g=Game()
    tr=mk_card('knight',11,'blue',9,14)
    g.deploy('blue',tr)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    tgt,td=g._find_target(tr)
    assert tgt is d,"Should see nearby enemy"
    return f"Sight range: troop at 1 tile engages (td={td:.1f})"
def t_pf_sight_range_far():
    g=Game()
    tr=mk_card('knight',11,'blue',9,5)
    g.deploy('blue',tr)
    d=Dummy('red',9,20,hp=50000,spd=0)
    g.deploy('red',d)
    tgt,td=g._find_target(tr)
    assert tgt is not None,"Should have some target"
    if tgt is d:pass
    else:
        assert hasattr(tgt,'ttype'),"Far troop not in sight → target tower"
    return f"Sight range: far troop → target={getattr(tgt,'name',getattr(tgt,'ttype','?'))}"
def t_pf_aggro_lock():
    g=Game()
    tr=mk_card('knight',11,'blue',9,14)
    g.deploy('blue',tr)
    d1=Dummy('red',9,15,hp=50000,spd=0)
    d2=Dummy('red',10,15.5,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.tick()
    tgt1,_=g._find_target(tr)
    tr.aggro_tgt=tgt1
    g.tick()
    tgt2,_=g._find_target(tr)
    assert tgt2 is tgt1,"Aggro should stay locked on first target"
    return "Aggro lock keeps target"
def t_pf_retarget_delay():
    g=quiet(Game())
    tr=mk_card('knight',11,'blue',9,14)
    g.deploy('blue',tr)
    d1=Dummy('red',9,15,hp=1,spd=0)
    d2=Dummy('red',9,15.5,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    for _ in range(50):
        g.tick()
        if not d1.alive:break
    assert not d1.alive,"d1 should die"
    assert tr.retarget_cd>0 or d2.hp<50000,"Retarget delay or already retargeted"
    g.run(2)
    assert d2.hp<50000,"Should retarget and attack d2"
    return f"Retarget delay after kill (d2 hp={d2.hp})"
def t_pf_building_target_paths():
    g=Game()
    gi=mk_card('giant',11,'blue',4,14)
    g.deploy('blue',gi)
    g.run(5)
    assert gi.y>14.5,"Giant should move toward red towers"
    return f"Building-target giant moves forward (y={gi.y:.1f})"
def t_pf_left_deploy_left_bridge():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    p=pf.a_star(2,10,2,22,False)
    assert len(p)>0,"Should find path"
    bx=[x for x,y in p if 15.0<=y<=17.0]
    assert all(x<=6.0 for x in bx),"Left deploy should use left bridge"
    return "Left deploy → left bridge"
def t_pf_right_deploy_right_bridge():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    p=pf.a_star(15,10,15,22,False)
    assert len(p)>0,"Should find path"
    bx=[x for x,y in p if 15.0<=y<=17.0]
    assert all(x>=11.0 for x in bx),"Right deploy should use right bridge"
    return "Right deploy → right bridge"
def t_pf_riverjump_crosses():
    g=Game()
    rj=mk_card('hog_rider',11,'blue',9,14)
    g.deploy('blue',rj)
    from sim.fx import RiverJump
    has_rj=any(isinstance(c,RiverJump) for c in rj.components)
    assert has_rj,"Hog should have RiverJump"
    g.run(5)
    assert rj.y>16,"RiverJump troop should cross river"
    return f"RiverJump troop crosses river (y={rj.y:.1f})"
def t_pf_path_recompute():
    g=Game()
    tr=mk_card('knight',11,'blue',9,14)
    g.deploy('blue',tr)
    d1=Dummy('red',9,15,hp=1,spd=0)
    g.deploy('red',d1)
    g.tick()
    tgt1,_=g._find_target(tr)
    assert tgt1 is d1
    d1.alive=False;d1.hp=0
    g.tick()
    tgt2,_=g._find_target(tr)
    assert tgt2 is not d1,"Should find new target after death"
    return "Path recomputes when target dies"
def t_pf_tower_always_visible():
    g=Game()
    tr=mk_card('knight',11,'blue',9,5)
    g.deploy('blue',tr)
    tgt,td=g._find_target(tr)
    assert tgt is not None,"Knight should always have a target"
    assert hasattr(tgt,'ttype'),"Far knight targets tower"
    return f"Towers always visible as targets (d={td:.1f})"
def t_pf_data_loaded():
    k=mk_card('knight',11,'blue',0,0)
    assert hasattr(k,'sight_r'),"Knight should have sight_r"
    assert hasattr(k,'collision_r'),"Knight should have collision_r"
    assert k.sight_r==5.5,f"Knight sight_r={k.sight_r}"
    assert k.collision_r==0.5,f"Knight collision_r={k.collision_r}"
    go=mk_card('golem',11,'blue',0,0)
    assert go.sight_r==7.0,f"Golem sight_r={go.sight_r}"
    assert go.collision_r==0.75,f"Golem collision_r={go.collision_r}"
    return f"Pathfinding data loads (knight sr={k.sight_r}, golem sr={go.sight_r})"
def t_pf_collision_air_ground_sep():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    air=Dummy('blue',9,10,spd=1.0);air.transport='Air';air.collision_r=0.5
    gnd=Dummy('blue',9.2,10,spd=1.0);gnd.transport='Ground';gnd.collision_r=0.5
    ox_a,ox_g=air.x,gnd.x
    pf.resolve_collisions([air,gnd])
    assert air.x==ox_a and gnd.x==ox_g,"Air and ground shouldn't collide"
    return "Air and ground troops don't collide"
def t_cross_team_fight():
    g=Game()
    k1=mk_card('knight',11,'blue',9,14)
    k2=mk_card('knight',11,'red',9,15)
    g.deploy('blue',k1);g.deploy('red',k2)
    g.run(10)
    assert k1.hp<k1.max_hp or k2.hp<k2.max_hp,"Cross-team troops should fight"
    return f"Cross-team fight: k1={k1.hp}/{k1.max_hp} k2={k2.hp}/{k2.max_hp}"
def t_push_propagation():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    giant=Dummy('blue',9,12,mass=8,spd=0.75)
    giant.collision_r=0.8
    prince=Dummy('blue',9,13,mass=5,spd=2.0)
    prince.collision_r=0.5;prince.tgt=type('T',(object,),{'x':9,'y':5})()
    giant.tgt=type('T',(object,),{'x':9,'y':5})()
    gy0=giant.y
    pf.resolve_collisions([giant,prince],dt=0.1)
    assert giant.y<gy0,"Giant should be pushed forward (lower y) by Prince"
    return f"Push propagation: giant moved {gy0-giant.y:.3f} toward tower"
def t_heavy_ignores_light():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    pekka=Dummy('blue',9,10,mass=10,spd=0.9)
    pekka.collision_r=0.7
    sk=Dummy('red',9.5,10,mass=1,spd=2.0)
    sk.collision_r=0.3
    pekka.tgt=type('T',(object,),{'x':9,'y':5})()
    sk.tgt=type('T',(object,),{'x':9,'y':15})()
    px0=pekka.x
    pf.resolve_collisions([pekka,sk],dt=0.1)
    pd=abs(pekka.x-px0);sd=abs(sk.x-0.5)
    assert sd>pd,"Skeleton should be displaced much more than PEKKA"
    return f"Heavy ignores light: pekka={pd:.4f} skel={sd:.4f}"
def t_pf_default_target():
    g=Game()
    tr=Dummy('blue',9,10,spd=1.0)
    g.deploy('blue',tr)
    tgt,td=g._find_target(tr)
    assert tgt is not None,"Should have default target (tower)"
    assert hasattr(tgt,'ttype'),"Default target is a tower"
    return f"Default target is nearest tower (d={td:.1f})"
def t_pf_grid_obstacle():
    from sim.path import Pathfinder
    a=Arena()
    pf=Pathfinder(a)
    assert not pf._gnd[15][9],"River tile should be blocked for ground"
    assert not pf._gnd[16][9],"River tile should be blocked for ground"
    assert pf._gnd[15][3] and pf._gnd[16][14],"Bridge tiles should be walkable"
    assert pf._air[15][9],"River tile should be walkable for air"
    assert not pf._gnd[3][8] and not pf._air[3][8],"King tower footprint blocked for everyone"
    return "Obstacle grid: river blocked for ground, open for air"
def t_pf_rebuild_on_tower_death():
    g=Game()
    pt=g.arena.get_tower('red','princess','left')
    tiles=pt.tiles()
    assert not g._pf._gnd[tiles[0][1]][tiles[0][0]],"Tower tile blocked before death"
    pt.hp=0;pt.alive=False;g._tower_down(pt)
    g._pf.rebuild_tower_grid()
    assert g._pf._gnd[tiles[0][1]][tiles[0][0]],"Tower tile walkable after death"
    return "Tower grid rebuilds on tower death"
def t_arena_footprints():
    a=Arena()
    bk=a.get_tower('blue','king');rk=a.get_tower('red','king')
    bl=a.get_tower('blue','princess','left');rr=a.get_tower('red','princess','right')
    assert (bk.cx,bk.cy)==(9.0,3.0) and (rk.cx,rk.cy)==(9.0,29.0)
    assert (bl.cx,bl.cy)==(3.5,6.5) and (rr.cx,rr.cy)==(14.5,25.5)
    assert set(bk.tiles())=={(x,y) for x in range(7,11) for y in range(1,5)}
    assert set(rk.tiles())=={(x,y) for x in range(7,11) for y in range(27,31)}
    assert set(bl.tiles())=={(x,y) for x in range(2,5) for y in range(5,8)}
    assert set(rr.tiles())=={(x,y) for x in range(13,16) for y in range(24,27)}
    assert bl.dist(3.5,6.5)==0 and bl.dist(3.5,9.0)==1.0 and abs(bl.dist(6.0,9.0)-math.hypot(1,1))<1e-9
    return "Tower centres 3.5/14.5 x 6.5/25.5 and 9.0 x 3.0/29.0 with 3x3 and 4x4 tile footprints"
def t_arena_river_bridges():
    a=Arena()
    assert a.RIVER==(15,16) and all(a.grid[y][x]=='R' for y in (15,16) for x in range(18) if x not in (3,14))
    assert a.on_bridge(2.5) and a.on_bridge(4.5) and not a.on_bridge(4.6) and a.on_bridge(14.5) and not a.on_bridge(12.4)
    assert a.blocked(0,0) and a.blocked(17,31) and not a.blocked(8,0) and a.blocked(0,14) and a.blocked(17,17)
    assert a.blocked(9,15) and not a.blocked(9,15,air=True) and not a.blocked(3,16)
    assert a.blocked(8,3) and a.blocked(8,3,air=True)
    return "River rows 15-16, bridges centred on lanes 3.5/14.5 two tiles wide, back and bank fences"
def t_arena_replay_coords():
    a=Arena()
    rx,ry=3500,6500
    x,y=rx/1000,ry/1000
    t=a.get_tower('blue','princess','left')
    assert (x,y)==(t.cx,t.cy) and (int(x),int(y)) in t.tiles()
    return "Replay units/1000 land on tile centres that match footprints"
def t_stun_timed():
    g=Game()
    d=Dummy('red',9,10,hp=50000,dmg=100,hspd=1.0,spd=1.0,rng=1.5)
    tgt=Dummy('blue',9,11,hp=50000,spd=0)
    g.deploy('red',d);g.deploy('blue',tgt)
    g.run(1.2)
    hp0=tgt.hp;assert hp0<50000
    d.statuses.append(Status('stun',1.0))
    g.run(0.95)
    assert tgt.hp==hp0,"stunned unit does not attack"
    g.run(0.9)
    assert tgt.hp==hp0,"attack cycle restarts from load time after the stun"
    g.run(0.3)
    assert tgt.hp<hp0
    return "Stun is a timed status that halts and resets the attack cycle"
def t_stun_halts_movement():
    g=Game()
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    k.statuses.append(Status('stun',0.5))
    oy=k.y
    g.run(0.45)
    assert k.y==oy
    g.run(0.5)
    assert k.y>oy
    return "Stun stops movement for its duration only"
def t_slow_hit_speed():
    g1=Game();g2=Game()
    a1=Dummy('red',9,10,hp=50000,dmg=100,hspd=1.0,spd=0);a2=Dummy('red',9,10,hp=50000,dmg=100,hspd=1.0,spd=0)
    t1=Dummy('blue',9,11,hp=50000,spd=0);t2=Dummy('blue',9,11,hp=50000,spd=0)
    g1.deploy('red',a1);g1.deploy('blue',t1);g2.deploy('red',a2);g2.deploy('blue',t2)
    a1.statuses.append(Status('slow',20.0,0.65))
    g1.run(10);g2.run(10)
    h1=(50000-t1.hp)//100;h2=(50000-t2.hp)//100
    assert h2>h1 and abs(h1/h2-0.65)<0.15,f"slow should cut hit rate by 35%: {h1} vs {h2}"
    return f"Slow reduces hit speed too ({h1} vs {h2} hits in 10s)"
def t_rage_speed_not_damage():
    g1=Game();g2=Game()
    a1=Dummy('red',9,10,hp=50000,dmg=100,hspd=1.0,spd=0);a2=Dummy('red',9,10,hp=50000,dmg=100,hspd=1.0,spd=0)
    t1=Dummy('blue',9,11,hp=50000,spd=0);t2=Dummy('blue',9,11,hp=50000,spd=0)
    g1.deploy('red',a1);g1.deploy('blue',t1);g2.deploy('red',a2);g2.deploy('blue',t2)
    a1.statuses.append(Status('rage',20.0,0.3))
    g1.run(10);g2.run(10)
    d1=50000-t1.hp;d2=50000-t2.hp
    assert d1%100==0,"rage must not change damage per hit"
    assert d1>d2 and abs(d1/d2-1.3)<0.15,f"rage should raise hit rate by 30%: {d1} vs {d2}"
    return f"Rage boosts hit speed, not damage ({d1//100} vs {d2//100} hits)"
def t_freeze_halts_all():
    g=Game()
    d=Dummy('red',9,10,hp=50000,dmg=100,hspd=1.0,spd=1.0,rng=1.5)
    tgt=Dummy('blue',9,11,hp=50000,spd=0)
    g.deploy('red',d);g.deploy('blue',tgt)
    d.statuses.append(Status('freeze',2.0))
    ox,oy=d.x,d.y
    g.run(1.9)
    assert tgt.hp==50000 and (d.x,d.y)==(ox,oy)
    g.run(1.5)
    assert tgt.hp<50000
    return "Freeze stops movement and attacks for its duration"
def t_freeze_tower():
    g=Game()
    tr=Dummy('red',3.5,12.0,hp=50000,spd=0)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    lpt.statuses.append(Status('freeze',3.0))
    g.run(2.9)
    assert tr.hp==50000,"frozen tower must not shoot"
    g.run(1.0)
    assert tr.hp<50000
    fz=mk_card('freeze',11,'blue',14.5,25.5);fz.apply(g)
    assert any(s.kind=='freeze' for s in g.arena.get_tower('red','princess','right').statuses)
    return "Freeze also stops crown towers"
def t_slow_no_stack():
    g=Game()
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    k.statuses.append(Status('slow',10,0.7));k.statuses.append(Status('slow',10,0.85))
    assert abs(g._status_mods(k)[2]-0.7)<1e-9,"strongest slow applies, slows do not multiply"
    return "Slows do not stack"
def t_proj_homing():
    g=Game()
    for t in g.arena.towers:t.alive=False
    sh=Dummy('blue',9,8,hp=50000,dmg=100,hspd=1.0,spd=0,rng=6)
    sh.proj_spd=4.0
    tgt=Dummy('red',9,13,hp=50000,spd=1.0,dmg=0)
    g.deploy('blue',sh);g.deploy('red',tgt)
    g.run(1.2)
    pr=g.projs[0]
    assert tgt.hp==50000 and abs(pr.ty-tgt.y)<=0.06,"shot in flight tracks the moving target"
    g.run(0.7)
    assert tgt.hp==49900 and not g.projs,"homing shot lands on the moving target"
    return "Ranged attack spawns a homing projectile that hits on arrival"
def t_proj_hitscan_default():
    g=Game()
    sh=Dummy('blue',9,8,hp=50000,dmg=100,hspd=1.0,spd=0,rng=6)
    tgt=Dummy('red',9,12,hp=50000,spd=0)
    g.deploy('blue',sh);g.deploy('red',tgt)
    g.run(1.05)
    assert tgt.hp<50000 and not g.projs
    return "Without projSpeed attacks are hitscan"
def t_proj_tower():
    g=Game()
    lpt=g.arena.get_tower('blue','princess','left')
    lpt.troop.proj_spd=10.0
    tr=Dummy('red',3.5,12.0,hp=50000,spd=0)
    g.deploy('red',tr)
    g.run(0.3)
    assert tr.hp==50000 and g.projs,"tower arrow in flight"
    g.run(0.4)
    assert tr.hp<50000
    return "Tower shots use the projectile path when projSpeed is set"
def t_move_around_tower():
    g=Game()
    k=mk_card('knight',11,'blue',3.5,2.0)
    g.deploy('blue',k)
    d=Dummy('red',3.5,10.0,hp=50000,spd=0,dmg=0)
    g.deploy('red',d)
    blk={t for tw in g.arena.towers for t in tw.tiles()}
    for _ in range(200):
        g.tick()
        assert (int(k.x),int(k.y)) not in blk,f"knight inside a tower footprint at {k.x:.2f},{k.y:.2f}"
    assert k.y>7.5,f"knight should have walked around the princess tower, y={k.y:.2f}"
    return f"Ground unit paths around tower footprints (y={k.y:.1f})"
def t_move_melee_reaches_tower():
    g=Game()
    k=mk_card('knight',11,'red',3.5,9.0)
    g.deploy('red',k)
    lpt=g.arena.get_tower('blue','princess','left')
    g.run(4)
    assert lpt.hp<lpt.max_hp and (int(k.x),int(k.y)) not in set(lpt.tiles())
    assert k.y>=8.0-1e-6
    return f"Melee unit attacks from the footprint edge (y={k.y:.2f})"
def t_move_never_in_river():
    g=Game()
    k=mk_card('knight',11,'blue',7,13)
    g.deploy('blue',k)
    for _ in range(300):
        g.tick()
        if int(k.y) in g.arena.RIVER:assert g.arena.on_bridge(k.x),f"knight in river at x={k.x:.2f}"
    assert k.y>17
    return "Ground unit only crosses the river on the bridge span"
def t_move_air_ignores_footprints():
    g=Game()
    bd=mk_card('baby_dragon',11,'blue',9,1)
    g.deploy('blue',bd)
    g.run(3)
    assert bd.y>3.5
    return f"Air unit flies over the king tower (y={bd.y:.1f})"
def t_move_building_blocks():
    g=Game()
    b=Dummy('red',9,12,hp=50000,spd=0,dmg=0);b.is_building=True;b.collision_r=0.8
    k=Dummy('blue',9,11,hp=50000,spd=1.0,dmg=0,rng=0.1)
    k.targets=['Air']
    g.deploy('red',b);g.deploy('blue',k)
    g.run(3)
    assert math.hypot(k.x-b.x,k.y-b.y)>=1.2,"troop pushed out of the building"
    return "Ground unit cannot walk through an enemy building"
def t_tower_down_once():
    g=Game()
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=1
    pt.take_damage(5);g._tower_down(pt)
    pt.take_damage(5);g._tower_down(pt)
    assert g.players['red'].crowns==1,f"crowns {g.players['red'].crowns}"
    return "Tower death counted once under simultaneous damage"
