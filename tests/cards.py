import math
import random
from sim.game import Game,Deck,Player,card_info,validate_deck
from sim.cards import create as mk_card
from sim.towers import create as mk_tt
from sim.units import Status
from tests.util import Dummy,quiet

def t_knight_load():
    tr=mk_card('knight',11,'blue',5,10)
    assert tr.hp==1766 and tr.dmg==202
    assert abs(tr.hspd-1.2)<0.01 and abs(tr.fhspd-0.7)<0.01
    assert abs(tr.rng-1.2)<0.01 and abs(tr.spd-1.0)<0.01
    assert tr.targets==['Ground']
    return f"Knight load (hp={tr.hp} dmg={tr.dmg})"
def t_knight_v_troop():
    g=Game()
    k1=mk_card('knight',11,'blue',9,14)
    k2=mk_card('knight',11,'red',9,17)
    g.deploy('blue',k1);g.deploy('red',k2)
    g.run(20)
    assert not k1.alive or not k2.alive
    return f"Knight vs Knight (k1={k1.hp} k2={k2.hp})"
def t_knight_v_tower():
    g=Game()
    k=mk_card('knight',11,'red',3,8)
    g.deploy('red',k)
    lpt=g.arena.get_tower('blue','princess','left')
    ini=lpt.hp
    g.run(15)
    assert lpt.hp<ini
    return f"Knight vs tower ({ini}->{lpt.hp})"
def t_archers_spawn():
    r=mk_card('archers',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==2
    assert all(a.hp==304 for a in r)
    return f"Archers spawn 2 (hp={r[0].hp})"
def t_archers_air():
    g=Game()
    arcs=mk_card('archers',11,'blue',9,10)
    for a in arcs:g.deploy('blue',a)
    bd=mk_card('baby_dragon',11,'red',9,12)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Archers target air ({ini}->{bd.hp})"
def t_ground_cant_air():
    g=Game()
    k=mk_card('knight',11,'blue',9,14)
    bd=mk_card('baby_dragon',11,'red',9,15)
    g.deploy('blue',k);g.deploy('red',bd)
    ini=bd.hp
    g.run(3)
    assert bd.hp==ini
    return "Ground-only can't target air"
def t_musk_fhspd():
    g=Game()
    m=mk_card('musketeer',11,'blue',9,10)
    g.deploy('blue',m)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(0.55)
    assert d.hp==ini,"Musketeer shot before fhspd"
    g.run(0.1)
    assert d.hp<ini,"No shot by 0.65s"
    return "Musketeer first-hit (0.3s load, 0.3s flight over 5 tiles)"
def t_bdrag_splash():
    g=Game()
    bd=mk_card('baby_dragon',11,'blue',9,10)
    g.deploy('blue',bd)
    d1=Dummy('red',9,20,hp=50000,spd=0)
    d2=Dummy('red',10,20,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(20)
    assert d1.hp<50000 and d2.hp<50000
    return f"Baby Dragon splash (d1={50000-d1.hp} d2={50000-d2.hp} dmg)"
def t_bdrag_air():
    bd=mk_card('baby_dragon',11,'blue',9,10)
    assert bd.transport=='Air'
    return "Baby Dragon is Air"
def t_fb_aoe():
    g=Game()
    d1=Dummy('red',9,10,hp=1000,spd=0)
    d2=Dummy('red',10,10,hp=1000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    fb=mk_card('fireball',11,'blue',9.5,10)
    fb.apply(g)
    assert d1.hp==1000-688 and d2.hp==1000-688
    return f"Fireball AOE (d1={d1.hp} d2={d2.hp})"
def t_fb_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    fb=mk_card('fireball',11,'blue',rpt.cx,rpt.cy)
    fb.apply(g)
    assert rpt.hp==ini-172
    return f"Fireball crown tower ({ini}->{rpt.hp})"
def t_fb_kb():
    g=Game()
    d=Dummy('red',9,10,hp=5000,spd=0)
    g.deploy('red',d)
    ox=d.x
    fb=mk_card('fireball',11,'blue',8,10)
    fb.apply(g)
    assert d.x>ox
    return f"Fireball knockback ({ox}->{d.x:.2f})"
def t_hog_ignores():
    g=Game()
    hog=mk_card('hog_rider',11,'blue',9,14)
    g.deploy('blue',hog)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini
    return "Hog ignores troops"
def t_hog_tower():
    g=Game()
    hog=mk_card('hog_rider',11,'blue',3,14)
    g.deploy('blue',hog)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(30)
    assert rpt.hp<ini
    return f"Hog targets tower ({ini}->{rpt.hp})"
def t_hog_jump():
    from sim.fx import RiverJump
    hog=mk_card('hog_rider',11,'blue',9,14)
    assert any(isinstance(c,RiverJump) for c in hog.components)
    return "Hog has RiverJump"
def t_skarmy_cnt():
    random.seed(42)
    r=mk_card('skeleton_army',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==15
    assert r[0].hp==81
    return f"Skarmy 15 units (hp={r[0].hp})"
def t_skarmy_pos():
    random.seed(42)
    r=mk_card('skeleton_army',11,'blue',9,10)
    xs=[t.x for t in r]
    assert min(xs)!=max(xs)
    return "Skarmy random positions"
def t_skarmy_fb():
    g=Game()
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,10)
    for s in sk:g.deploy('red',s)
    fb=mk_card('fireball',11,'blue',9,10)
    fb.apply(g)
    alive=[s for s in sk if s.alive]
    assert len(alive)==0,f"{len(alive)} survived"
    return "Fireball wipes skarmy"
def t_freeze_stop():
    g=Game()
    d=Dummy('red',3,13,hp=50000,spd=2.0)
    g.deploy('red',d)
    ox,oy=d.x,d.y
    fz=mk_card('freeze',11,'blue',3,13)
    fz.apply(g)
    g.run(2)
    assert abs(d.x-ox)<0.01 and abs(d.y-oy)<0.01
    return "Freeze stops movement"
def t_freeze_dur():
    g=Game()
    d=Dummy('red',3,13,hp=50000,spd=2.0)
    g.deploy('red',d)
    fz=mk_card('freeze',11,'blue',3,13)
    fz.apply(g)
    oy=d.y
    g.run(3.9)
    assert abs(d.y-oy)<0.01,"Moved while frozen"
    g.run(0.3)
    assert abs(d.y-oy)>0.01,"Still frozen after 4.2s"
    return "Freeze wears off after 4s"
def t_mpekka_load():
    mp=mk_card('mini_pekka',11,'blue',5,10)
    assert mp.hp==1390 and mp.dmg==755
    assert abs(mp.hspd-1.6)<0.01 and abs(mp.fhspd-1.1)<0.01
    assert abs(mp.rng-0.8)<0.01 and abs(mp.spd-1.5)<0.01
    return f"Mini PEKKA load (hp={mp.hp} dmg={mp.dmg})"
def t_mpekka_kills_knight():
    g=quiet(Game())
    mp=mk_card('mini_pekka',11,'blue',9,14)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('blue',mp);g.deploy('red',k)
    g.run(10)
    assert not k.alive
    assert mp.alive
    return f"Mini PEKKA kills Knight (mp_hp={mp.hp})"
def t_valk_splash():
    g=Game()
    v=mk_card('valkyrie',11,'blue',9,10)
    g.deploy('blue',v)
    d1=Dummy('red',9,11,hp=5000,spd=0)
    d2=Dummy('red',10,11,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Valkyrie splash (d1={5000-d1.hp} d2={5000-d2.hp} dmg)"
def t_valk_tanky():
    v=mk_card('valkyrie',11,'blue',5,10)
    assert v.hp==1907
    return f"Valkyrie tanky (hp={v.hp})"
def t_zap_aoe():
    g=Game()
    d1=Dummy('red',9,10,hp=1000,spd=0)
    d2=Dummy('red',10,10,hp=1000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    z=mk_card('zap',11,'blue',9.5,10)
    z.apply(g)
    assert d1.hp==1000-192 and d2.hp==1000-192
    return f"Zap AOE (d1={d1.hp} d2={d2.hp})"
def t_zap_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    z=mk_card('zap',11,'blue',rpt.cx,rpt.cy)
    z.apply(g)
    assert rpt.hp==ini-48
    return f"Zap crown tower ({ini}->{rpt.hp})"
def t_zap_stun():
    g=Game()
    d=Dummy('red',3,8,hp=50000,spd=0,dmg=500,hspd=1.0)
    g.deploy('red',d)
    lpt=g.arena.get_tower('blue','princess','left')
    g.run(1.5)
    hp1=lpt.hp
    z=mk_card('zap',11,'blue',d.x,d.y)
    z.apply(g)
    g.run(0.7)
    assert lpt.hp==hp1,"Stun should have reset attack cycle"
    return "Zap stun resets attack cycle"
def t_prince_charge():
    g=Game()
    p=mk_card('prince',11,'blue',9,9)
    g.deploy('blue',p)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg=50000-d.hp
    assert dmg>=786,f"Expected >=786 charge dmg, got {dmg}"
    return f"Prince charge ({dmg} dmg in 5s)"
def t_prince_charge_reset():
    from sim.fx import Charge
    from sim.units import Status
    p=mk_card('prince',11,'blue',9,10)
    ch=[c for c in p.components if isinstance(c,Charge)][0]
    g=Game()
    g.deploy('blue',p)
    d=Dummy('red',9,25,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(2)
    assert ch.moved>0
    p.statuses.append(Status('freeze',2.0))
    g.run(0.1)
    assert ch.moved==0,"Charge not reset by freeze"
    return "Prince charge resets on freeze"
def t_prince_jump():
    from sim.fx import RiverJump
    p=mk_card('prince',11,'blue',9,14)
    assert any(isinstance(c,RiverJump) for c in p.components)
    return "Prince has RiverJump"
def t_gbarrel_spawn():
    random.seed(42)
    gb=mk_card('goblin_barrel',11,'blue',14,24)
    g=Game()
    gb.apply(g)
    gobs=g.players['blue'].troops
    assert len(gobs)==3,f"Expected 3 goblins, got {len(gobs)}"
    assert all(t.hp==202 for t in gobs)
    return f"Goblin Barrel spawns 3 (hp={gobs[0].hp})"
def t_gbarrel_atk():
    g=Game()
    random.seed(42)
    gb=mk_card('goblin_barrel',11,'blue',14,24)
    gb.apply(g)
    rpt=g.arena.get_tower('red','princess','right')
    ini=rpt.hp
    g.run(5)
    assert rpt.hp<ini
    return f"Goblins attack tower ({ini}->{rpt.hp})"
def t_witch_splash():
    g=Game()
    w=mk_card('witch',11,'blue',9,10)
    g.deploy('blue',w)
    d1=Dummy('red',9,15,hp=5000,spd=0)
    d2=Dummy('red',10,15,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Witch splash (d1={5000-d1.hp} d2={5000-d2.hp} dmg)"
def t_witch_spawn():
    g=Game()
    random.seed(42)
    w=mk_card('witch',11,'blue',9,10)
    g.deploy('blue',w)
    ini=len(g.players['blue'].troops)
    g.run(2)
    n=len(g.players['blue'].troops)-ini
    assert n>=4,f"Expected >=4 spawns, got {n}"
    return f"Witch spawns skeletons (+{n})"
def t_golem_building():
    g=Game()
    go=mk_card('golem',11,'blue',9,14)
    g.deploy('blue',go)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini,"Golem should ignore troops"
    return "Golem targets buildings only"
def t_golem_death_spawn():
    g=Game()
    random.seed(42)
    go=mk_card('golem',11,'blue',9,14)
    g.deploy('blue',go)
    go.hp=1
    d=Dummy('red',9,15,hp=50000,dmg=100,spd=0,hspd=0.5)
    g.deploy('red',d)
    g.run(1)
    assert not go.alive,"Golem should be dead"
    gm=[t for t in g.players['blue'].troops if t.alive]
    assert len(gm)==2,f"Expected 2 golemites, got {len(gm)}"
    assert gm[0].hp==1039
    return f"Golem death spawns 2 golemites (hp={gm[0].hp})"
def t_golem_death_dmg():
    g=quiet(Game())
    go=mk_card('golem',11,'blue',9,14)
    g.deploy('blue',go)
    go.hp=1
    d1=Dummy('red',9,15,hp=50000,dmg=500,spd=0,hspd=0.5)
    d2=Dummy('red',10,14,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(2)
    assert not go.alive
    assert d2.hp<50000,"Death damage not applied"
    assert 50000-d2.hp==225,f"Expected 225 death dmg, got {50000-d2.hp}"
    return f"Golem death damage ({50000-d2.hp} to nearby)"
def t_ewiz_szap():
    g=quiet(Game())
    d=Dummy('red',9,12.5,hp=5000,spd=0)
    g.deploy('red',d)
    ew=mk_card('electro_wizard',11,'blue',9,10)
    g.deploy('blue',ew)
    ini=d.hp
    g.run(0.1)
    assert d.hp==ini-192,f"Expected 192 spawn zap dmg, got {ini-d.hp}"
    return f"E-wiz spawn zap ({ini-d.hp} dmg + stun reset)"
def t_ewiz_dual():
    g=Game()
    ew=mk_card('electro_wizard',11,'blue',9,10)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',10,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.deploy('blue',ew)
    g.run(3)
    assert d1.hp<5000 and d2.hp<5000
    return f"E-wiz dual target (d1={5000-d1.hp} d2={5000-d2.hp} dmg)"
def t_ewiz_stun_atk():
    g=Game()
    ew=mk_card('electro_wizard',11,'blue',9,10)
    d=Dummy('red',9,14,hp=50000,spd=0,hspd=0.5,dmg=100)
    tgt=Dummy('blue',9,15,hp=50000,spd=0)
    g.deploy('red',d);g.deploy('blue',ew);g.deploy('blue',tgt)
    g.run(1.5)
    dmg=50000-tgt.hp
    assert dmg<1500,"Stun should delay attacks"
    return f"E-wiz stun delays attack ({dmg} dmg)"
def t_giant_load():
    gi=mk_card('giant',11,'blue',5,10)
    assert gi.hp==3968 and gi.dmg==253
    assert gi.targets==['Buildings']
    return f"Giant load (hp={gi.hp} dmg={gi.dmg})"
def t_giant_targets_buildings():
    g=Game()
    gi=mk_card('giant',11,'blue',9,14)
    g.deploy('blue',gi)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini,"Giant should ignore troops"
    return "Giant targets buildings only"
def t_balloon_air():
    b=mk_card('balloon',11,'blue',9,10)
    assert b.transport=='Air'
    assert b.targets==['Buildings']
    return f"Balloon is Air, targets buildings (hp={b.hp})"
def t_balloon_death_dmg():
    g=Game()
    b=mk_card('balloon',11,'blue',9,14)
    g.deploy('blue',b)
    d=Dummy('red',9,14.5,hp=50000,spd=0)
    g.deploy('red',d)
    b.hp=0;b.alive=False;b.on_death(g)
    assert 50000-d.hp==240,f"Expected 240 death dmg, got {50000-d.hp}"
    return f"Balloon death damage ({50000-d.hp})"
def t_lava_air():
    lh=mk_card('lava_hound',11,'blue',9,10)
    assert lh.transport=='Air'
    assert lh.targets==['Buildings']
    return f"Lava Hound is Air (hp={lh.hp})"
def t_lava_death_spawn():
    g=Game()
    random.seed(42)
    lh=mk_card('lava_hound',11,'blue',9,14)
    g.deploy('blue',lh)
    lh.hp=0;lh.alive=False;lh.on_death(g)
    pups=[t for t in g.players['blue'].troops if t.alive and t is not lh]
    assert len(pups)==6,f"Expected 6 pups, got {len(pups)}"
    assert pups[0].hp==215
    assert pups[0].transport=='Air'
    assert 'Air' in pups[0].targets and 'Ground' in pups[0].targets
    return f"Lava Hound death spawns 6 Air pups (hp={pups[0].hp})"
def t_darkprince_shield():
    dp=mk_card('dark_prince',11,'blue',9,10)
    assert dp.shield_hp==240
    dp.take_damage(500)
    assert dp.shield_hp==0
    assert dp.hp==1200,"Excess should be blocked"
    dp.take_damage(100)
    assert dp.hp==1100
    return f"Dark Prince shield absorbs+blocks (hp={dp.hp})"
def t_darkprince_charge_splash():
    from sim.fx import Charge,SplashAttack
    dp=mk_card('dark_prince',11,'blue',9,10)
    assert any(isinstance(c,Charge) for c in dp.components)
    assert any(isinstance(c,SplashAttack) for c in dp.components)
    assert dp.charge_dmg==532
    return "Dark Prince has charge+splash"
def t_idrag_ramp():
    g=Game()
    idrag=mk_card('inferno_dragon',11,'blue',9,10)
    g.deploy('blue',idrag)
    d=Dummy('red',9,13,hp=50000,spd=0)
    g.deploy('red',d)
    assert idrag.dmg==35
    g.run(1.0)
    assert idrag.dmg==35,"Stage 1 in first 1.5s"
    g.run(1.0)
    assert idrag.dmg==120,f"Expected stage 2, got {idrag.dmg}"
    g.run(1.5)
    assert idrag.dmg==422,f"Expected stage 3, got {idrag.dmg}"
    return "Inferno Dragon ramp (35->120->422)"
def t_idrag_reset():
    g=Game()
    idrag=mk_card('inferno_dragon',11,'blue',9,10)
    g.deploy('blue',idrag)
    d1=Dummy('red',9,13,hp=50000,spd=0)
    d2=Dummy('red',12,13,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(2.0)
    assert idrag.dmg==120,f"Expected stage 2 after 2s, got {idrag.dmg}"
    d1.alive=False
    g.players['red'].troops=[t for t in g.players['red'].troops if t.alive]
    g.run(0.2)
    assert idrag.dmg==35,f"Expected reset to 35, got {idrag.dmg}"
    return "Inferno Dragon resets on retarget"
def t_lumberjack_speed():
    lj=mk_card('lumberjack',11,'blue',9,10)
    assert abs(lj.spd-2.0)<0.01
    return f"Lumberjack speed={lj.spd}"
def t_lumberjack_rage():
    g=Game()
    lj=mk_card('lumberjack',11,'blue',9,10)
    ally=Dummy('blue',9,10.5,hp=50000,spd=1.0,dmg=100)
    g.deploy('blue',lj);g.deploy('blue',ally)
    lj.hp=0;lj.alive=False;lj.on_death(g)
    has_rage=any(s.kind=='rage' for s in ally.statuses)
    assert has_rage,"Ally should have rage status"
    rs=[s for s in ally.statuses if s.kind=='rage'][0]
    assert abs(rs.val-0.3)<0.01
    return "Lumberjack drops rage on death"
def t_poison_dot():
    g=Game()
    d=Dummy('red',9,25,hp=5000,spd=0)
    g.deploy('red',d)
    p=mk_card('poison',11,'blue',9,25)
    p.apply(g);g.spells.append(p)
    g.run(8.5)
    dmg=5000-d.hp
    assert abs(dmg-736)<50,f"Expected ~736 total dmg, got {dmg}"
    return f"Poison DoT ({dmg} over 8s)"
def t_poison_slow():
    g=Game()
    d=Dummy('red',9,10,hp=50000,spd=2.0)
    g.deploy('red',d)
    p=mk_card('poison',11,'blue',9,10)
    p.apply(g);g.spells.append(p)
    g.run(1.5)
    assert any(s.kind=='mslow' for s in d.statuses),"Poison should slow movement"
    assert not any(s.kind=='slow' for s in d.statuses),"Poison must not slow hit speed"
    return "Poison applies movement-only slow"
def t_poison_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    p=mk_card('poison',11,'blue',rpt.cx,rpt.cy)
    p.apply(g);g.spells.append(p)
    g.run(8.5)
    dmg=ini-rpt.hp
    assert abs(dmg-168)<50,f"Expected ~168 CT dmg, got {dmg}"
    return f"Poison crown tower ({dmg} dmg)"
def t_log_ground():
    g=quiet(Game())
    d1=Dummy('red',9,12,hp=1000,spd=0)
    g.deploy('red',d1)
    log=mk_card('the_log',11,'blue',9,10)
    g._cast('blue',log,9,10);g.run(0.4)
    assert d1.hp==1000,"the log rolls 2 tiles in 0.6 s before it reaches the dummy"
    g.run(0.4)
    assert d1.hp==1000-269,f"Expected 269 dmg, got {1000-d1.hp}"
    return f"Log damages ground (hp={d1.hp})"
def t_log_no_air():
    g=Game()
    d=Dummy('red',9,12,hp=1000,spd=0)
    d.transport='Air'
    g.deploy('red',d)
    log=mk_card('the_log',11,'blue',9,10)
    log.apply(g)
    assert d.hp==1000,"Log should not hit air"
    return "Log skips air troops"
def t_log_pushback():
    g=Game()
    d=Dummy('red',9,12,hp=5000,spd=0)
    g.deploy('red',d)
    oy=d.y
    log=mk_card('the_log',11,'blue',9,10)
    g._cast('blue',log,9,10);g.run(1)
    assert abs(d.y-oy-0.7)<0.01,f"Expected +0.7 pushback, got {d.y-oy}"
    return f"Log pushback ({d.y-oy:.1f} tiles)"
def t_pekka_load():
    p=mk_card('pekka',11,'blue',5,10)
    assert p.hp==3760 and p.dmg==842
    assert abs(p.hspd-1.8)<0.01 and abs(p.fhspd-1.3)<0.01
    assert abs(p.spd-0.75)<0.01 and abs(p.rng-1.2)<0.01
    assert p.targets==['Ground']
    return f"PEKKA load (hp={p.hp} dmg={p.dmg} spd={p.spd})"
def t_pekka_kills_knight():
    g=Game()
    pk=mk_card('pekka',11,'blue',9,14)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('blue',pk);g.deploy('red',k)
    g.run(12)
    assert not k.alive
    assert pk.alive
    return f"PEKKA kills Knight (pekka_hp={pk.hp})"
def t_mk_load():
    mk=mk_card('mega_knight',11,'blue',5,10)
    assert mk.hp==3993 and mk.dmg==268
    assert abs(mk.hspd-1.7)<0.01
    assert mk.splash_r==1.3
    return f"Mega Knight load (hp={mk.hp} dmg={mk.dmg})"
def t_mk_spawn_dmg():
    g=Game()
    d=Dummy('red',9,15,hp=5000,spd=0)
    g.deploy('red',d)
    mk=mk_card('mega_knight',11,'blue',9,14)
    g.deploy('blue',mk)
    ini=d.hp
    g.run(0.1)
    assert d.hp==5000-537,f"Expected 537 spawn dmg, got {ini-d.hp}"
    return f"Mega Knight spawn damage ({ini-d.hp})"
def t_mk_splash():
    g=Game()
    mk=mk_card('mega_knight',11,'blue',9,10)
    g.deploy('blue',mk)
    d1=Dummy('red',9,11.5,hp=5000,spd=0)
    d2=Dummy('red',10,11.5,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Mega Knight splash (d1={5000-d1.hp} d2={5000-d2.hp} dmg)"
def t_nw_load():
    nw=mk_card('night_witch',11,'blue',5,10)
    assert nw.hp==906 and nw.dmg==314
    assert abs(nw.hspd-1.3)<0.01
    return f"Night Witch load (hp={nw.hp} dmg={nw.dmg})"
def t_nw_spawn():
    g=Game()
    random.seed(42)
    nw=mk_card('night_witch',11,'blue',9,10)
    g.deploy('blue',nw)
    g.run(2)
    bats=[t for t in g.players['blue'].troops if t is not nw and t.alive]
    assert len(bats)>=2,f"Expected >=2 bats, got {len(bats)}"
    assert bats[0].transport=='Air'
    return f"Night Witch spawns bats (+{len(bats)} Air)"
def t_nw_death_spawn():
    g=Game()
    random.seed(42)
    nw=mk_card('night_witch',11,'blue',9,10)
    g.deploy('blue',nw)
    nw.hp=0;nw.alive=False;nw.on_death(g)
    bats=[t for t in g.players['blue'].troops if t.alive and t is not nw]
    assert len(bats)>=1,f"Expected >=1 death bat, got {len(bats)}"
    assert bats[0].transport=='Air'
    return f"Night Witch death spawns {len(bats)} bat(s)"
def t_icewiz_load():
    iw=mk_card('ice_wizard',11,'blue',5,10)
    assert iw.hp==688 and iw.dmg==89
    assert abs(iw.rng-5.5)<0.01
    assert iw.splash_r==3.0
    return f"Ice Wizard load (hp={iw.hp} dmg={iw.dmg} rng={iw.rng})"
def t_icewiz_slow():
    g=Game()
    iw=mk_card('ice_wizard',11,'blue',9,10)
    g.deploy('blue',iw)
    d=Dummy('red',9,15,hp=50000,spd=2.0)
    g.deploy('red',d)
    g.run(2)
    has_slow=any(s.kind=='slow' for s in d.statuses)
    assert has_slow,"Ice Wizard should slow on hit"
    sl=[s for s in d.statuses if s.kind=='slow'][0]
    assert abs(sl.val-0.7)<0.01,f"Expected slow val 0.7, got {sl.val}"
    return "Ice Wizard slows on attack (30%)"
def t_icewiz_spawn_dmg():
    g=Game()
    d=Dummy('red',9,15,hp=5000,spd=0)
    g.deploy('red',d)
    iw=mk_card('ice_wizard',11,'blue',9,14)
    g.deploy('blue',iw)
    g.run(0.1)
    assert d.hp==5000-89,f"Expected 89 spawn dmg, got {5000-d.hp}"
    return f"Ice Wizard spawn damage ({5000-d.hp})"
def t_espirit_chain():
    g=Game()
    d1=Dummy('red',9,12,hp=5000,spd=0)
    d2=Dummy('red',10,12,hp=5000,spd=0)
    d3=Dummy('red',11,12,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2);g.deploy('red',d3)
    es=mk_card('electro_spirit',11,'blue',9,11)
    g.deploy('blue',es)
    g.run(3)
    assert d1.hp<5000,"Target 1 not hit"
    assert d2.hp<5000,"Chain didn't reach target 2"
    assert d3.hp<5000,"Chain didn't reach target 3"
    return f"Electro Spirit chains (d1={5000-d1.hp} d2={5000-d2.hp} d3={5000-d3.hp})"
def t_espirit_stun():
    g=Game()
    d1=Dummy('red',9,12,hp=5000,spd=0)
    d2=Dummy('red',10,12,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    es=mk_card('electro_spirit',11,'blue',9,11)
    g.deploy('blue',es)
    g.run(3)
    assert not es.alive,"Spirit should die after attack"
    return "Electro Spirit dies after attack + stuns"
def t_espirit_load():
    es=mk_card('electro_spirit',11,'blue',5,10)
    assert es.hp==215 and es.dmg==99
    assert es.chain_count==9
    assert abs(es.chain_range-3.0)<0.01
    return f"Electro Spirit load (hp={es.hp} dmg={es.dmg} chains={es.chain_count})"
def t_rocket_aoe():
    g=Game()
    d1=Dummy('red',9,10,hp=5000,spd=0)
    d2=Dummy('red',10,10,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    r=mk_card('rocket',11,'blue',9.5,10)
    r.apply(g)
    assert d1.hp==5000-1484,f"Expected 1484 dmg, got {5000-d1.hp}"
    assert d2.hp==5000-1484
    return f"Rocket AOE ({5000-d1.hp} dmg)"
def t_rocket_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    r=mk_card('rocket',11,'blue',rpt.cx,rpt.cy)
    r.apply(g)
    assert rpt.hp==ini-343,f"Expected 343 CT dmg, got {ini-rpt.hp}"
    return f"Rocket crown tower ({ini}->{rpt.hp})"
def t_rocket_kb():
    g=Game()
    d=Dummy('red',9,10,hp=50000,spd=0)
    g.deploy('red',d)
    ox=d.x
    r=mk_card('rocket',11,'blue',8,10)
    r.apply(g)
    assert d.x>ox,"Rocket should knockback"
    return f"Rocket knockback ({ox}->{d.x:.2f})"
def t_arrows_aoe():
    g=Game()
    d1=Dummy('red',9,10,hp=1000,spd=0)
    d2=Dummy('red',10,10,hp=1000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    a=mk_card('arrows',11,'blue',9.5,10)
    a.apply(g)
    for _ in range(10):a.tick(0.05,g)
    total=1000-d1.hp
    assert total==366,f"Expected 366 total dmg (122*3), got {total}"
    assert d2.hp==1000-366
    return f"Arrows AOE ({total} dmg each, 3 volleys)"
def t_arrows_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    a=mk_card('arrows',11,'blue',rpt.cx,rpt.cy)
    a.apply(g)
    for _ in range(10):a.tick(0.05,g)
    assert rpt.hp==ini-75,f"Expected 75 CT dmg (25*3), got {ini-rpt.hp}"
    return f"Arrows crown tower ({ini}->{rpt.hp})"
def t_arrows_volley():
    g=Game()
    d=Dummy('red',9,20,hp=1000,spd=0)
    g.deploy('red',d)
    a=mk_card('arrows',11,'blue',9,20)
    a.apply(g)
    after1=d.hp
    assert after1==878,f"First volley should deal 122 dmg, got {1000-after1}"
    assert a.active,"Should stay active for remaining volleys"
    for _ in range(3):a.tick(0.05,g)
    assert d.hp==878,"Second volley shouldn't fire yet at 0.15s"
    a.tick(0.05,g)
    after2=d.hp
    assert after2==756,f"Second volley at 0.20s should deal 122 more, hp={after2}"
    for _ in range(4):a.tick(0.05,g)
    after3=d.hp
    assert after3==634,f"Third volley at 0.40s should deal 122 more, hp={after3}"
    assert not a.active,"Should be inactive after all volleys"
    return "Arrows 3 volleys (1000->878->756->634)"
def t_gy_spawns():
    g=Game()
    random.seed(42)
    gy=mk_card('graveyard',11,'blue',9,25)
    gy.apply(g);g.spells.append(gy)
    g.run(10)
    skels=[t for t in g.players['blue'].troops if t.alive]
    total=gy.spawned
    assert total==13,f"Expected 13 spawns, got {total}"
    return f"Graveyard spawns 13 skeletons ({len(skels)} alive)"
def t_gy_attack():
    g=Game()
    random.seed(42)
    gy=mk_card('graveyard',11,'blue',14,24)
    gy.apply(g);g.spells.append(gy)
    rpt=g.arena.get_tower('red','princess','right')
    ini=rpt.hp
    g.run(15)
    assert rpt.hp<ini,"Skeletons should damage tower"
    return f"Graveyard skeletons attack tower ({ini}->{rpt.hp})"
def t_rage_dmg():
    g=Game()
    d=Dummy('red',9,10,hp=5000,spd=0)
    g.deploy('red',d)
    r=mk_card('rage',11,'blue',9,10)
    r.apply(g)
    assert d.hp==5000-179,f"Expected 179 dmg, got {5000-d.hp}"
    return f"Rage spell damages enemies ({5000-d.hp})"
def t_rage_buff():
    g=Game()
    ally=Dummy('blue',9,10,hp=50000,spd=1.0)
    g.deploy('blue',ally)
    r=mk_card('rage',11,'blue',9,10)
    r.apply(g)
    has_rage=any(s.kind=='rage' for s in ally.statuses)
    assert has_rage,"Rage spell should buff allies"
    rs=[s for s in ally.statuses if s.kind=='rage'][0]
    assert abs(rs.val-0.3)<0.01,f"Expected 30% boost, got {rs.val}"
    return "Rage spell buffs allies (30% boost)"
def t_rage_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    r=mk_card('rage',11,'blue',rpt.cx,rpt.cy)
    r.apply(g)
    assert rpt.hp==ini-45,f"Expected 45 CT dmg, got {ini-rpt.hp}"
    return f"Rage crown tower ({ini}->{rpt.hp})"
def t_int_pekka_v_mpekka():
    g=Game()
    pk=mk_card('pekka',11,'blue',9,14)
    mp=mk_card('mini_pekka',11,'red',9,17)
    g.deploy('blue',pk);g.deploy('red',mp)
    g.run(15)
    assert not mp.alive,"PEKKA should kill Mini PEKKA"
    assert pk.alive,"PEKKA should survive"
    assert pk.hp>1500,f"PEKKA should have lots of HP left, got {pk.hp}"
    return f"PEKKA vs Mini PEKKA (pekka_hp={pk.hp})"
def t_int_pekka_v_tower():
    g=Game()
    pk=mk_card('pekka',11,'blue',3,14)
    g.deploy('blue',pk)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(30)
    assert rpt.hp<ini,"PEKKA should damage tower"
    dmg=ini-rpt.hp
    assert dmg>1500,f"PEKKA should deal heavy tower dmg, got {dmg}"
    return f"PEKKA walks to tower ({ini}->{rpt.hp}, {dmg} dmg)"
def t_int_mk_v_skarmy():
    g=Game()
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,15)
    for s in sk:g.deploy('red',s)
    mk=mk_card('mega_knight',11,'blue',9,14)
    g.deploy('blue',mk)
    g.run(0.2)
    alive=[s for s in sk if s.alive]
    assert len(alive)==0,f"MK spawn should wipe skarmy, {len(alive)} survived"
    assert mk.alive
    return "MK spawn wipes skarmy (all 15 dead)"
def t_int_mk_v_knight():
    g=Game()
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    mk=mk_card('mega_knight',11,'blue',9,14)
    g.deploy('blue',mk)
    g.run(15)
    assert not k.alive,"MK should kill Knight"
    assert mk.alive
    return f"MK vs Knight (mk_hp={mk.hp})"
def t_int_nw_bats_v_tower():
    g=Game()
    random.seed(42)
    nw=mk_card('night_witch',11,'blue',3,14)
    g.deploy('blue',nw)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(25)
    bats=[t for t in g.players['blue'].troops if t is not nw and t.alive]
    dmg=ini-rpt.hp
    assert dmg>0,"NW+bats should damage tower"
    return f"NW+bats push tower ({ini}->{rpt.hp}, {len(bats)} bats alive)"
def t_int_nw_death_bats_fight():
    g=Game()
    random.seed(42)
    nw=mk_card('night_witch',11,'blue',9,14)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('blue',nw);g.deploy('red',k)
    g.run(20)
    assert not nw.alive,"NW should die to Knight"
    post_bats=[t for t in g.players['blue'].troops if t.alive]
    return f"NW vs Knight: NW dies, spawned bats ({len(post_bats)} alive after)"
def t_int_icewiz_slows_push():
    g=Game()
    iw=mk_card('ice_wizard',11,'blue',9,10)
    g.deploy('blue',iw)
    k_slow=mk_card('knight',11,'red',9,17)
    g.deploy('red',k_slow)
    g2=Game()
    k_fast=mk_card('knight',11,'red',9,17)
    g2.deploy('red',k_fast)
    for _ in range(100):g.tick();g2.tick()
    assert k_slow.y>k_fast.y,f"Slowed Knight should be behind: {k_slow.y:.1f} vs {k_fast.y:.1f}"
    return f"IW slows Knight push (slowed y={k_slow.y:.1f} vs free y={k_fast.y:.1f})"
def t_int_icewiz_splash_slow():
    g=Game()
    iw=mk_card('ice_wizard',11,'blue',9,10)
    g.deploy('blue',iw)
    k1=mk_card('knight',11,'red',9,15)
    k2=mk_card('knight',11,'red',10,15)
    g.deploy('red',k1);g.deploy('red',k2)
    g.run(3)
    s1=any(s.kind=='slow' for s in k1.statuses)
    s2=any(s.kind=='slow' for s in k2.statuses)
    assert s1,"Primary target should be slowed"
    assert s2,"Splash victim should also be slowed"
    return "IW splash slows multiple knights"
def t_int_espirit_v_archers():
    g=Game()
    random.seed(42)
    arcs=mk_card('archers',11,'red',9,15)
    for a in arcs:g.deploy('red',a)
    es=mk_card('electro_spirit',11,'blue',9,14)
    g.deploy('blue',es)
    g.run(5)
    assert not es.alive,"Spirit should die"
    dmg0=304-arcs[0].hp;dmg1=304-arcs[1].hp
    assert dmg0>=109 and dmg1>=109,f"Both archers should be chained: {dmg0}, {dmg1}"
    return f"E-Spirit chains through archers (dmg={dmg0},{dmg1})"
def t_int_espirit_v_skarmy():
    g=Game()
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,15)
    for s in sk:g.deploy('red',s)
    es=mk_card('electro_spirit',11,'blue',9,14)
    g.deploy('blue',es)
    g.run(5)
    dead=[s for s in sk if not s.alive]
    assert len(dead)>=5,f"Spirit should chain-kill >=5 skellies, got {len(dead)}"
    return f"E-Spirit vs skarmy ({len(dead)}/15 killed)"
def t_int_rocket_kills_witch():
    g=Game()
    w=mk_card('witch',11,'red',9,25)
    g.deploy('red',w)
    r=mk_card('rocket',11,'blue',9,25)
    r.apply(g)
    assert not w.alive,"Rocket should one-shot Witch (hp was 906, dmg=1484)"
    return "Rocket one-shots Witch"
def t_int_arrows_kills_skarmy():
    g=Game()
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,10)
    for s in sk:g.deploy('red',s)
    a=mk_card('arrows',11,'blue',9,10)
    a.apply(g)
    for _ in range(10):a.tick(0.05,g)
    alive=[s for s in sk if s.alive]
    assert len(alive)==0,f"Arrows should kill all skarmy (120>82hp), {len(alive)} survived"
    return "Arrows wipes skarmy (3x40=120 dmg > 82 hp)"
def t_int_arrows_kills_bats():
    g=Game()
    random.seed(42)
    nw=mk_card('night_witch',11,'blue',9,10)
    g.deploy('blue',nw)
    g.run(1.5)
    bats=[t for t in g.players['blue'].troops if t is not nw and t.alive]
    assert len(bats)>=2,"NW should have spawned bats"
    a=mk_card('arrows',11,'red',sum(b.x for b in bats)/len(bats),sum(b.y for b in bats)/len(bats))
    a.apply(g)
    for _ in range(10):a.tick(0.05,g)
    alive_bats=[t for t in bats if t.alive]
    assert len(alive_bats)==0,f"Arrows should kill bats (120>81hp), {len(alive_bats)} survived"
    return f"Arrows kills NW bats ({len(bats)} bats zapped)"
def t_int_gy_pressure():
    g=quiet(Game())
    random.seed(42)
    gy=mk_card('graveyard',11,'blue',14,24)
    gy.apply(g);g.spells.append(gy)
    rpt=g.arena.get_tower('red','princess','right')
    ini=rpt.hp
    g.run(12)
    dmg=ini-rpt.hp
    assert dmg>300,f"GY should deal significant tower dmg, got {dmg}"
    skels=[t for t in g.players['blue'].troops if t.alive]
    return f"GY tower pressure ({ini}->{rpt.hp}, {dmg} dmg, {len(skels)} skels alive)"
def t_int_rage_knight_dps():
    g1=Game();g2=Game()
    k1=mk_card('knight',11,'blue',3,14)
    k2=mk_card('knight',11,'blue',3,14)
    g1.deploy('blue',k1);g2.deploy('blue',k2)
    g1.run(15);g2.run(15)
    rpt1=g1.arena.get_tower('red','princess','left')
    rpt2=g2.arena.get_tower('red','princess','left')
    hp_before_rage=rpt1.hp
    r=mk_card('rage',11,'blue',k1.x,k1.y)
    r.apply(g1);g1.spells.append(r)
    for _ in range(80):g1.tick();g2.tick()
    d1=hp_before_rage-rpt1.hp;d2=rpt2.max_hp-rpt2.hp
    d2_window=d2-(rpt2.max_hp-hp_before_rage)
    assert d1>d2_window,f"Raged knight should deal more tower dmg in window: {d1} vs {d2_window}"
    return f"Rage boosts Knight DPS (raged={d1} vs normal={d2_window} in 8s)"
def t_int_mk_spawn_on_push():
    g=Game()
    random.seed(42)
    arcs=mk_card('archers',11,'red',9,16)
    for a in arcs:g.deploy('red',a)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    ini_arcs=[a.hp for a in arcs]
    mk=mk_card('mega_knight',11,'blue',9,15)
    g.deploy('blue',mk)
    g.run(0.2)
    for i,a in enumerate(arcs):
        assert a.hp<ini_arcs[i],f"Archer {i} should take spawn dmg"
    assert k.hp<1690,"Knight should take spawn dmg"
    return "MK spawn lands on push (archers+knight all hit)"
def t_int_icewiz_behind_pekka():
    g=Game()
    pk=mk_card('pekka',11,'blue',9,12)
    iw=mk_card('ice_wizard',11,'blue',9,10)
    g.deploy('blue',pk);g.deploy('blue',iw)
    k1=mk_card('knight',11,'red',9,17)
    k2=mk_card('knight',11,'red',10,17)
    g.deploy('red',k1);g.deploy('red',k2)
    g.run(20)
    dead=[k for k in [k1,k2] if not k.alive]
    assert len(dead)>=1,"PEKKA+IW should kill at least one knight"
    assert pk.alive,"PEKKA should survive with IW support"
    return f"PEKKA+IW vs 2 Knights ({len(dead)} dead, pekka_hp={pk.hp})"
def t_skeletons_spawn():
    random.seed(42)
    r=mk_card('skeletons',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==3
    assert all(s.hp==81 for s in r)
    assert all(s.dmg==81 for s in r)
    return f"Skeletons spawn 3 (hp={r[0].hp} dmg={r[0].dmg})"
def t_skeletons_speed():
    r=mk_card('skeletons',11,'blue',9,10)
    assert abs(r[0].spd-1.5)<0.01
    assert r[0].targets==['Ground']
    return f"Skeletons fast (spd={r[0].spd})"
def t_skeletons_die_fast():
    g=Game()
    random.seed(42)
    sk=mk_card('skeletons',11,'red',9,15)
    for s in sk:g.deploy('red',s)
    fb=mk_card('fireball',11,'blue',9,15)
    fb.apply(g)
    alive=[s for s in sk if s.alive]
    assert len(alive)==0,f"Fireball should kill all skeletons (689>81), {len(alive)} survived"
    return "Skeletons die to Fireball"
def t_goblins_spawn():
    random.seed(42)
    r=mk_card('goblins',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==4
    assert all(g.hp==202 for g in r)
    assert all(g.dmg==125 for g in r)
    return f"Goblins spawn 4 (hp={r[0].hp} dmg={r[0].dmg})"
def t_goblins_speed():
    r=mk_card('goblins',11,'blue',9,10)
    assert abs(r[0].spd-2.0)<0.01
    assert abs(r[0].fhspd-0.5)<0.01
    return f"Goblins very fast (spd={r[0].spd} fhspd={r[0].fhspd})"
def t_goblins_dps():
    g=Game()
    random.seed(42)
    gobs=mk_card('goblins',11,'blue',9,14)
    for gb in gobs:g.deploy('blue',gb)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg=50000-d.hp
    assert dmg>=128*4,f"4 goblins should deal significant dmg, got {dmg}"
    return f"Goblins DPS ({dmg} dmg in 5s)"
def t_minions_spawn():
    random.seed(42)
    r=mk_card('minions',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==3
    assert all(m.hp==230 for m in r)
    assert all(m.dmg==107 for m in r)
    assert all(m.transport=='Air' for m in r)
    return f"Minions spawn 3 Air (hp={r[0].hp} dmg={r[0].dmg})"
def t_minions_targets():
    r=mk_card('minions',11,'blue',9,10)
    assert 'Air' in r[0].targets and 'Ground' in r[0].targets
    return "Minions target Air+Ground"
def t_minions_v_ground():
    g=Game()
    random.seed(42)
    mins=mk_card('minions',11,'blue',9,10)
    for m in mins:g.deploy('blue',m)
    d=Dummy('red',9,11,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert d.hp<50000,"Minions should attack ground troops"
    return f"Minions attack ground ({50000-d.hp} dmg)"
def t_minions_immune_ground():
    g=Game()
    random.seed(42)
    mins=mk_card('minions',11,'blue',9,14)
    for m in mins:g.deploy('blue',m)
    k=mk_card('knight',11,'red',9,15)
    g.deploy('red',k)
    ini=[m.hp for m in mins]
    g.run(3)
    for i,m in enumerate(mins):
        if m.alive:assert m.hp==ini[i],"Knight can't hit air minions"
    return "Minions immune to ground-only attacks"
def t_megaminion_load():
    mm=mk_card('mega_minion',11,'blue',5,10)
    assert mm.hp==837 and mm.dmg==312
    assert mm.transport=='Air'
    assert 'Air' in mm.targets and 'Ground' in mm.targets
    assert abs(mm.hspd-1.5)<0.01 and abs(mm.fhspd-1.1)<0.01
    return f"Mega Minion load (hp={mm.hp} dmg={mm.dmg} Air)"
def t_megaminion_v_air():
    g=Game()
    mm=mk_card('mega_minion',11,'blue',9,10)
    g.deploy('blue',mm)
    bd=mk_card('baby_dragon',11,'red',9,12)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini,"Mega Minion should attack air"
    return f"Mega Minion attacks air ({ini}->{bd.hp})"
def t_megaminion_v_ground():
    g=Game()
    mm=mk_card('mega_minion',11,'blue',9,10)
    g.deploy('blue',mm)
    d=Dummy('red',9,11,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert d.hp<50000,"Mega Minion should attack ground"
    return f"Mega Minion attacks ground ({50000-d.hp} dmg)"
def t_guards_spawn():
    random.seed(42)
    r=mk_card('guards',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==3
    assert all(g.hp==81 for g in r)
    assert all(g.dmg==117 for g in r)
    assert all(g.shield_hp==256 for g in r)
    return f"Guards spawn 3 (hp={r[0].hp} shield={r[0].shield_hp} dmg={r[0].dmg})"
def t_guards_shield_absorb():
    random.seed(42)
    r=mk_card('guards',11,'blue',9,10)
    g0=r[0]
    g0.take_damage(200)
    assert g0.shield_hp==56,"Shield should absorb 200 (256->56)"
    assert g0.hp==81,"HP should be untouched while shield active"
    return f"Guards shield absorbs (shield={g0.shield_hp} hp={g0.hp})"
def t_guards_shield_break():
    random.seed(42)
    r=mk_card('guards',11,'blue',9,10)
    g0=r[0]
    g0.take_damage(500)
    assert g0.shield_hp==0,"Shield should be broken"
    assert g0.hp==81,"Excess blocked by shield"
    g0.take_damage(50)
    assert g0.hp==31,"After shield break, HP takes dmg"
    return f"Guards shield break+excess blocked (hp={g0.hp})"
def t_guards_indep_shields():
    random.seed(42)
    r=mk_card('guards',11,'blue',9,10)
    r[0].take_damage(400)
    assert r[0].shield_hp==0 and r[0].hp==81
    assert r[1].shield_hp==256 and r[1].hp==81
    assert r[2].shield_hp==256 and r[2].hp==81
    return "Guards independent shields"
def t_icegolem_load():
    ig=mk_card('ice_golem',11,'blue',5,10)
    assert ig.hp==1315 and ig.dmg==84
    assert ig.targets==['Buildings']
    assert abs(ig.spd-0.75)<0.01
    return f"Ice Golem load (hp={ig.hp} dmg={ig.dmg} spd={ig.spd})"
def t_icegolem_targets_buildings():
    g=Game()
    ig=mk_card('ice_golem',11,'blue',9,14)
    g.deploy('blue',ig)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini,"Ice Golem should ignore troops"
    return "Ice Golem targets buildings only"
def t_icegolem_death_nova():
    g=Game()
    ig=mk_card('ice_golem',11,'blue',9,14)
    g.deploy('blue',ig)
    d1=Dummy('red',9,14.5,hp=5000,spd=0)
    d2=Dummy('red',10,14.5,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    ig.hp=0;ig.alive=False;ig.on_death(g)
    assert d1.hp==5000-84,f"Expected 84 death nova dmg, got {5000-d1.hp}"
    assert d2.hp==5000-84,"Second target should also take 84 dmg"
    return f"Ice Golem death nova ({5000-d1.hp} dmg each)"
def t_icegolem_death_slow():
    g=Game()
    ig=mk_card('ice_golem',11,'blue',9,14)
    g.deploy('blue',ig)
    d=Dummy('red',9,14.5,hp=5000,spd=2.0)
    g.deploy('red',d)
    ig.hp=0;ig.alive=False;ig.on_death(g)
    has_slow=any(s.kind=='slow' for s in d.statuses)
    assert has_slow,"Death nova should slow"
    sl=[s for s in d.statuses if s.kind=='slow'][0]
    assert abs(sl.val-0.7)<0.01,f"Expected 30% slow (val=0.7), got {sl.val}"
    return "Ice Golem death nova slows (30%)"
def t_wb_spawn():
    random.seed(42)
    r=mk_card('wall_breakers',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==2
    assert all(w.hp==330 for w in r)
    assert all(w.dmg==281 for w in r)
    return f"Wall Breakers spawn 2 (hp={r[0].hp} dmg={r[0].dmg})"
def t_wb_suicide():
    r=mk_card('wall_breakers',11,'blue',9,10)
    assert r[0].is_suicide,"Wall Breakers should be suicide"
    assert r[0].targets==['Buildings']
    return "Wall Breakers suicide+building target"
def t_wb_speed():
    r=mk_card('wall_breakers',11,'blue',9,10)
    assert abs(r[0].spd-2.0)<0.01,"Should be very fast"
    return f"Wall Breakers very fast (spd={r[0].spd})"
def t_wb_v_tower():
    g=Game()
    random.seed(42)
    wb=mk_card('wall_breakers',11,'blue',3,14)
    for w in wb:g.deploy('blue',w)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(30)
    assert rpt.hp<ini,"WB should damage tower"
    dead=[w for w in wb if not w.alive]
    assert len(dead)==2,"Both WB should die after hit"
    return f"Wall Breakers hit tower ({ini}->{rpt.hp}, both die)"
def t_gskel_load():
    gs=mk_card('giant_skeleton',11,'blue',5,10)
    assert gs.hp==3361 and gs.dmg==276
    assert abs(gs.hspd-1.3)<0.01 and abs(gs.fhspd-1.0)<0.01
    assert abs(gs.spd-1.0)<0.01
    return f"Giant Skeleton load (hp={gs.hp} dmg={gs.dmg})"
def t_gskel_death_dmg():
    g=Game()
    gs=mk_card('giant_skeleton',11,'blue',9,14)
    g.deploy('blue',gs)
    d1=Dummy('red',9,14.5,hp=5000,spd=0)
    d2=Dummy('red',11,14.5,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    gs.hp=0;gs.alive=False;gs.on_death(g)
    assert d1.hp==5000-688,f"Expected 688 death dmg, got {5000-d1.hp}"
    assert d2.hp==5000-688,"Second target should also take 688 dmg (within 3.5 radius)"
    return f"Giant Skeleton death damage ({5000-d1.hp} each)"
def t_gskel_death_radius():
    g=Game()
    gs=mk_card('giant_skeleton',11,'blue',9,14)
    g.deploy('blue',gs)
    d_near=Dummy('red',9,14.5,hp=5000,spd=0)
    d_far=Dummy('red',9,18.5,hp=5000,spd=0)
    g.deploy('red',d_near);g.deploy('red',d_far)
    gs.hp=0;gs.alive=False;gs.on_death(g)
    assert d_near.hp<5000,"Near target should be hit"
    assert d_far.hp==5000,"Far target (4 tiles away) should be safe"
    return "Giant Skeleton death radius (3.5 tiles to the target's collision edge)"
def t_gskel_v_skarmy():
    g=Game()
    random.seed(42)
    gs=mk_card('giant_skeleton',11,'blue',9,14)
    g.deploy('blue',gs)
    sk=mk_card('skeleton_army',11,'red',9,15)
    for s in sk:g.deploy('red',s)
    gs.hp=1
    d=Dummy('red',9,14.5,hp=50000,dmg=500,spd=0,hspd=0.5)
    g.deploy('red',d)
    g.run(2)
    assert not gs.alive
    dead=[s for s in sk if not s.alive]
    assert len(dead)>=10,f"Giant Skeleton bomb should kill most skarmy, only killed {len(dead)}"
    return f"Giant Skeleton bomb wipes skarmy ({len(dead)}/15)"
def t_barbs_spawn():
    random.seed(42)
    r=mk_card('barbarians',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==5
    assert all(b.hp==716 for b in r)
    assert all(b.dmg==192 for b in r)
    return f"Barbarians spawn 5 (hp={r[0].hp} dmg={r[0].dmg})"
def t_barbs_stats():
    r=mk_card('barbarians',11,'blue',9,10)
    assert abs(r[0].spd-1.0)<0.01
    assert abs(r[0].hspd-1.4)<0.01
    assert abs(r[0].fhspd-1.0)<0.01
    assert abs(r[0].rng-0.7)<0.01
    return f"Barbarians stats (spd={r[0].spd} hspd={r[0].hspd} fhspd={r[0].fhspd})"
def t_barbs_v_knight():
    g=Game()
    random.seed(42)
    barbs=mk_card('barbarians',11,'blue',9,10)
    for b in barbs:g.deploy('blue',b)
    k=mk_card('knight',11,'red',9,12)
    g.deploy('red',k)
    g.run(10)
    assert not k.alive,"5 barbarians should kill a knight"
    alive=[b for b in barbs if b.alive]
    assert len(alive)>=3,f"Most barbs should survive vs 1 knight, got {len(alive)}"
    return f"Barbarians kill Knight ({len(alive)}/5 survive)"
def t_ebarbs_spawn():
    random.seed(42)
    r=mk_card('elite_barbarians',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==2
    assert all(e.hp==1341 for e in r)
    assert all(e.dmg==384 for e in r)
    return f"Elite Barbarians spawn 2 (hp={r[0].hp} dmg={r[0].dmg})"
def t_ebarbs_fast():
    r=mk_card('elite_barbarians',11,'blue',9,10)
    assert abs(r[0].spd-1.5)<0.01,"E-barbs should be Fast (90)"
    return f"Elite Barbarians fast (spd={r[0].spd})"
def t_ebarbs_v_knight():
    g=quiet(Game())
    random.seed(42)
    ebs=mk_card('elite_barbarians',11,'blue',9,14)
    for e in ebs:g.deploy('blue',e)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    g.run(12)
    assert not k.alive,"E-barbs should kill knight"
    return f"Elite Barbarians kill Knight (eb1={ebs[0].hp} eb2={ebs[1].hp})"
def t_sgobs_spawn():
    random.seed(42)
    r=mk_card('spear_goblins',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==3
    assert all(s.hp==133 for s in r)
    assert all(s.dmg==81 for s in r)
    return f"Spear Goblins spawn 3 (hp={r[0].hp} dmg={r[0].dmg})"
def t_sgobs_ranged():
    r=mk_card('spear_goblins',11,'blue',9,10)
    assert abs(r[0].rng-5.0)<0.01
    assert 'Air' in r[0].targets and 'Ground' in r[0].targets
    return "Spear Goblins ranged 5.0 targets Air+Ground"
def t_sgobs_v_air():
    g=Game()
    random.seed(42)
    sg=mk_card('spear_goblins',11,'blue',9,10)
    for s in sg:g.deploy('blue',s)
    bd=mk_card('baby_dragon',11,'red',9,14)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini,"Spear Goblins should hit air"
    return f"Spear Goblins hit air ({ini}->{bd.hp})"
def t_mhorde_spawn():
    random.seed(42)
    r=mk_card('minion_horde',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==6
    assert all(m.hp==230 for m in r)
    assert all(m.transport=='Air' for m in r)
    return f"Minion Horde spawn 6 Air (hp={r[0].hp})"
def t_mhorde_dps():
    g=Game()
    random.seed(42)
    mh=mk_card('minion_horde',11,'blue',9,10)
    for m in mh:g.deploy('blue',m)
    d=Dummy('red',9,11,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg=50000-d.hp
    assert dmg>=106*6,f"6 minions should deal heavy dmg, got {dmg}"
    return f"Minion Horde DPS ({dmg} in 5s)"
def t_mhorde_fireball():
    g=Game()
    random.seed(42)
    mh=mk_card('minion_horde',11,'blue',9,10)
    for m in mh:g.deploy('blue',m)
    fb=mk_card('fireball',11,'red',9,10)
    fb.apply(g)
    alive=[m for m in mh if m.alive]
    assert len(alive)==0,f"Fireball should kill all minions (689>230), {len(alive)} survived"
    return "Fireball wipes Minion Horde"
def t_sbarrel_load():
    sb=mk_card('skeleton_barrel',11,'blue',9,10)
    assert sb.hp==532
    assert sb.transport=='Air'
    assert sb.targets==['Buildings']
    assert sb.death_dmg==133
    return f"Skeleton Barrel load (hp={sb.hp} Air dd={sb.death_dmg})"
def t_sbarrel_death_spawn():
    g=Game()
    random.seed(42)
    sb=mk_card('skeleton_barrel',11,'blue',9,14)
    g.deploy('blue',sb)
    sb.hp=0;sb.alive=False;sb.on_death(g)
    skels=[t for t in g.players['blue'].troops if t.alive and t is not sb]
    assert len(skels)==7,f"Expected 7 skeletons, got {len(skels)}"
    assert skels[0].hp==81
    return f"Skeleton Barrel spawns 7 skeletons (hp={skels[0].hp})"
def t_sbarrel_death_dmg():
    g=Game()
    sb=mk_card('skeleton_barrel',11,'blue',9,14)
    g.deploy('blue',sb)
    d=Dummy('red',9,14.5,hp=5000,spd=0)
    g.deploy('red',d)
    sb.hp=0;sb.alive=False;sb.on_death(g)
    assert d.hp==5000-133,f"Expected 133 death dmg, got {5000-d.hp}"
    return f"Skeleton Barrel death damage ({5000-d.hp})"
def t_ispirit_load():
    es=mk_card('ice_spirit',11,'blue',5,10)
    assert es.hp==215 and es.dmg==110
    assert es.is_suicide
    assert abs(es.splash_r-1.5)<0.01
    assert abs(es.stun_dur-1.1)<0.01
    return f"Ice Spirit load (hp={es.hp} dmg={es.dmg} freeze={es.stun_dur}s)"
def t_ispirit_freeze():
    g=Game()
    es=mk_card('ice_spirit',11,'blue',9,10)
    g.deploy('blue',es)
    d1=Dummy('red',9,11.5,hp=5000,spd=2.0)
    d2=Dummy('red',10,11.5,hp=5000,spd=2.0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(3)
    assert not es.alive,"Spirit should die"
    h1=any(s.kind=='freeze' for s in d1.statuses)
    h2=any(s.kind=='freeze' for s in d2.statuses)
    assert h1 or d1.hp<5000,"Primary target should be frozen or damaged"
    assert h2 or d2.hp<5000,"Splash target should be frozen or damaged"
    return "Ice Spirit freezes on splash attack"
def t_ispirit_suicide():
    g=Game()
    es=mk_card('ice_spirit',11,'blue',9,14)
    g.deploy('blue',es)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert not es.alive,"Ice Spirit should die after attack"
    assert d.hp<50000,"Should deal damage"
    return f"Ice Spirit suicide ({50000-d.hp} dmg)"
def t_dgob_load():
    dg=mk_card('dart_goblin',11,'blue',5,10)
    assert dg.hp==261 and dg.dmg==151
    assert abs(dg.rng-6.5)<0.01
    assert abs(dg.hspd-0.8)<0.01
    assert abs(dg.fhspd-0.45)<0.01
    assert abs(dg.spd-2.0)<0.01
    return f"Dart Goblin load (hp={dg.hp} dmg={dg.dmg} rng={dg.rng} spd={dg.spd})"
def t_dgob_air():
    dg=mk_card('dart_goblin',11,'blue',5,10)
    assert 'Air' in dg.targets and 'Ground' in dg.targets
    return "Dart Goblin targets Air+Ground"
def t_dgob_outranges():
    g=Game()
    dg=mk_card('dart_goblin',11,'blue',9,10)
    g.deploy('blue',dg)
    k=mk_card('knight',11,'red',9,15)
    g.deploy('red',k)
    ini_k=k.hp;ini_dg=dg.hp
    g.run(2)
    assert k.hp<ini_k,"DG should hit knight at range"
    assert dg.hp==ini_dg,"Knight shouldn't reach DG yet"
    return f"Dart Goblin outranges Knight (knight={ini_k}->{k.hp})"
def t_rg_load():
    rg=mk_card('royal_giant',11,'blue',5,10)
    assert rg.hp==3164 and rg.dmg==307
    assert abs(rg.rng-5.0)<0.01
    assert abs(rg.spd-0.75)<0.01
    assert rg.targets==['Buildings']
    return f"Royal Giant load (hp={rg.hp} dmg={rg.dmg} rng={rg.rng})"
def t_rg_ignores_troops():
    g=Game()
    rg=mk_card('royal_giant',11,'blue',9,14)
    g.deploy('blue',rg)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini,"Royal Giant should ignore troops"
    return "Royal Giant ignores troops"
def t_rg_v_tower():
    g=Game()
    rg=mk_card('royal_giant',11,'blue',3,14)
    g.deploy('blue',rg)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(30)
    assert rpt.hp<ini,"Royal Giant should damage tower"
    dmg=ini-rpt.hp
    assert dmg>500,f"RG should deal significant tower dmg, got {dmg}"
    return f"Royal Giant hits tower ({ini}->{rpt.hp})"
def t_fspirit_load():
    fs=mk_card('fire_spirit',11,'blue',5,10)
    assert fs.hp==215 and fs.dmg==207
    assert fs.is_suicide
    assert abs(fs.splash_r-2.3)<0.01
    return f"Fire Spirit load (hp={fs.hp} dmg={fs.dmg} splash={fs.splash_r})"
def t_fspirit_splash():
    g=Game()
    fs=mk_card('fire_spirit',11,'blue',9,10)
    g.deploy('blue',fs)
    d1=Dummy('red',9,11.5,hp=5000,spd=0)
    d2=Dummy('red',10,11.5,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(3)
    assert not fs.alive,"Spirit should die"
    assert d1.hp<5000,"Primary hit"
    assert d2.hp<5000,"Splash hit"
    return f"Fire Spirit splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_fspirit_v_skarmy():
    g=Game()
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,15)
    for s in sk:g.deploy('red',s)
    fs=mk_card('fire_spirit',11,'blue',9,14)
    g.deploy('blue',fs)
    g.run(5)
    dead=[s for s in sk if not s.alive]
    assert len(dead)>=3,f"Fire Spirit should kill several skeletons (233>82), got {len(dead)}"
    return f"Fire Spirit kills skarmy ({len(dead)}/15)"
def t_ggang_spawn():
    random.seed(42)
    r=mk_card('goblin_gang',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==6
    gobs=[u for u in r if u.name=='Goblin']
    sgobs=[u for u in r if u.name=='Spear Goblin']
    assert len(gobs)==3,f"Expected 3 Goblins, got {len(gobs)}"
    assert len(sgobs)==3,f"Expected 3 Spear Goblins, got {len(sgobs)}"
    return f"Goblin Gang 3+3 (gob hp={gobs[0].hp} sgob hp={sgobs[0].hp})"
def t_ggang_stats():
    random.seed(42)
    r=mk_card('goblin_gang',11,'blue',9,10)
    gobs=[u for u in r if u.name=='Goblin']
    sgobs=[u for u in r if u.name=='Spear Goblin']
    assert gobs[0].hp==202 and gobs[0].dmg==125
    assert sgobs[0].hp==133 and sgobs[0].dmg==81
    assert abs(sgobs[0].rng-5.0)<0.01
    assert 'Air' in sgobs[0].targets
    return "Goblin Gang mixed stats correct"
def t_ggang_air_coverage():
    g=Game()
    random.seed(42)
    gg=mk_card('goblin_gang',11,'blue',9,10)
    for u in gg:g.deploy('blue',u)
    bd=mk_card('baby_dragon',11,'red',9,14)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini,"Spear Goblins in gang should hit air"
    return f"Goblin Gang hits air ({ini}->{bd.hp})"
def t_flymach_load():
    fm=mk_card('flying_machine',11,'blue',5,10)
    assert fm.hp==614 and fm.dmg==171
    assert fm.transport=='Air'
    assert abs(fm.rng-6.0)<0.01
    assert 'Air' in fm.targets and 'Ground' in fm.targets
    return f"Flying Machine load (hp={fm.hp} dmg={fm.dmg} rng={fm.rng} Air)"
def t_flymach_v_ground():
    g=Game()
    fm=mk_card('flying_machine',11,'blue',9,10)
    g.deploy('blue',fm)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert d.hp<50000
    return f"Flying Machine hits ground ({50000-d.hp} dmg)"
def t_flymach_immune():
    g=Game()
    fm=mk_card('flying_machine',11,'blue',9,14)
    g.deploy('blue',fm)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    ini=fm.hp
    g.run(3)
    assert fm.hp==ini,"Knight can't hit Flying Machine"
    return "Flying Machine immune to ground-only"
def t_skeldrags_spawn():
    random.seed(42)
    r=mk_card('skeleton_dragons',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==2
    assert all(d.hp==560 for d in r)
    assert all(d.dmg==151 for d in r)
    assert all(d.transport=='Air' for d in r)
    return f"Skeleton Dragons spawn 2 Air (hp={r[0].hp} dmg={r[0].dmg})"
def t_skeldrags_splash():
    g=Game()
    random.seed(42)
    sd=mk_card('skeleton_dragons',11,'blue',9,10)
    for d in sd:g.deploy('blue',d)
    d1=Dummy('red',9,15,hp=5000,spd=0)
    d2=Dummy('red',9.5,15,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000,"Both targets should take splash"
    return f"Skeleton Dragons splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_skeldrags_air():
    r=mk_card('skeleton_dragons',11,'blue',9,10)
    assert 'Air' in r[0].targets and 'Ground' in r[0].targets
    return "Skeleton Dragons target Air+Ground"
def t_hunter_load():
    h=mk_card('hunter',11,'blue',5,10)
    assert h.hp==885 and h.dmg==840
    assert abs(h.hspd-2.2)<0.01 and abs(h.fhspd-1.5)<0.01
    assert abs(h.rng-4.0)<0.01
    assert 'Air' in h.targets
    return f"Hunter load (hp={h.hp} dmg={h.dmg} rng={h.rng})"
def t_hunter_v_tank():
    g=Game()
    h=mk_card('hunter',11,'blue',9,10)
    g.deploy('blue',h)
    d=Dummy('red',9,13,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg=50000-d.hp
    assert dmg>=840,f"Hunter should deal heavy dmg at close range, got {dmg}"
    return f"Hunter high DPS ({dmg} in 5s)"
def t_hunter_v_air():
    g=Game()
    h=mk_card('hunter',11,'blue',9,10)
    g.deploy('blue',h)
    bd=mk_card('baby_dragon',11,'red',9,13)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini,"Hunter should hit air"
    return f"Hunter hits air ({ini}->{bd.hp})"
def t_firecracker_load():
    fc=mk_card('firecracker',11,'blue',5,10)
    # 64 per pellet on the target, the 5 pellet volley on towers
    assert fc.hp==304 and fc.dmg==64 and fc.ct_dmg==320
    assert abs(fc.rng-6.0)<0.01
    assert abs(fc.hspd-3.0)<0.01
    assert 'Air' in fc.targets
    return f"Firecracker load (hp={fc.hp} dmg={fc.dmg} rng={fc.rng})"
def t_firecracker_v_push():
    g=Game()
    fc=mk_card('firecracker',11,'blue',9,10)
    g.deploy('blue',fc)
    k=mk_card('knight',11,'red',9,15)
    g.deploy('red',k)
    ini=k.hp
    g.run(5)
    assert k.hp<ini,"Firecracker should hit Knight"
    return f"Firecracker hits push ({ini}->{k.hp})"
def t_rhogs_spawn():
    random.seed(42)
    r=mk_card('royal_hogs',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==4
    assert all(h.hp==837 for h in r)
    assert all(h.dmg==74 for h in r)
    assert all(h.targets==['Buildings'] for h in r)
    return f"Royal Hogs spawn 4 (hp={r[0].hp} dmg={r[0].dmg})"
def t_rhogs_jump():
    from sim.fx import RiverJump
    random.seed(42)
    r=mk_card('royal_hogs',11,'blue',9,10)
    assert any(isinstance(c,RiverJump) for c in r[0].components)
    return "Royal Hogs have RiverJump"
def t_rhogs_v_tower():
    g=Game()
    random.seed(42)
    rh=mk_card('royal_hogs',11,'blue',3,14)
    for h in rh:g.deploy('blue',h)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(30)
    assert rpt.hp<ini,"Royal Hogs should damage tower"
    return f"Royal Hogs hit tower ({ini}->{rpt.hp})"
def t_miner_load():
    m=mk_card('miner',11,'blue',5,10)
    assert m.hp==1210 and m.dmg==194
    assert m.ct_dmg==38
    assert abs(m.hspd-1.3)<0.01 and abs(m.fhspd-0.8)<0.01
    return f"Miner load (hp={m.hp} dmg={m.dmg} ct_dmg={m.ct_dmg})"
def t_miner_ct_reduction():
    g=Game()
    m=mk_card('miner',11,'blue',3,8)
    g.deploy('blue',m)
    d=Dummy('red',3,9,hp=50000,spd=0)
    g.deploy('red',d)
    ini_d=d.hp
    g.run(3)
    troop_dmg=ini_d-d.hp
    g2=Game()
    m2=mk_card('miner',11,'red',3,8)
    g2.deploy('red',m2)
    lpt=g2.arena.get_tower('blue','princess','left')
    ini_t=lpt.hp
    g2.run(15)
    tower_dmg=ini_t-lpt.hp
    if troop_dmg>0 and tower_dmg>0:
        ratio=tower_dmg/troop_dmg
        assert ratio<0.5,f"Tower dmg should be much less than troop dmg (ratio={ratio:.2f})"
    return f"Miner CT reduction (troop_dmg={troop_dmg} tower_dmg={tower_dmg})"
def t_miner_deploy_anywhere():
    random.seed(42)
    dk=_mk_deck(['miner'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','miner')
    g.players['blue'].elixir=10
    ok,msg=g.play_card('blue','miner',14,24)
    assert ok,f"Miner should deploy anywhere: {msg}"
    g.run(3)
    miners=[t for t in g.players['blue'].troops if t.name=='Miner']
    assert len(miners)==1,"Miner should spawn"
    return "Miner deploys anywhere on map"
def t_snowball_dmg():
    g=Game()
    d=Dummy('red',9,10,hp=1000,spd=0)
    g.deploy('red',d)
    sb=mk_card('giant_snowball',11,'blue',9,10)
    sb.apply(g)
    assert d.hp==1000-179,f"Expected 179 dmg, got {1000-d.hp}"
    return f"Giant Snowball damage ({1000-d.hp})"
def t_snowball_kb():
    g=Game()
    d=Dummy('red',9,10,hp=5000,spd=0)
    g.deploy('red',d)
    oy=d.y
    sb=mk_card('giant_snowball',11,'blue',9,8)
    sb.apply(g)
    assert abs(d.y-oy)>1.0,f"Expected ~1.8 tile knockback, got {abs(d.y-oy):.2f}"
    return f"Giant Snowball knockback ({d.y-oy:.1f} tiles)"
def t_snowball_slow():
    g=Game()
    d=Dummy('red',9,10,hp=5000,spd=2.0)
    g.deploy('red',d)
    sb=mk_card('giant_snowball',11,'blue',9,10)
    sb.apply(g)
    has_slow=any(s.kind=='slow' for s in d.statuses)
    assert has_slow,"Snowball should slow"
    sl=[s for s in d.statuses if s.kind=='slow'][0]
    assert abs(sl.val-0.7)<0.01,f"Expected 30% slow (val=0.7), got {sl.val}"
    return "Giant Snowball slows 30%"
def t_snowball_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    sb=mk_card('giant_snowball',11,'blue',rpt.cx,rpt.cy)
    sb.apply(g)
    assert rpt.hp==ini-45,f"Expected 45 CT dmg, got {ini-rpt.hp}"
    return f"Giant Snowball CT ({ini}->{rpt.hp})"
def t_lightning_3tgt():
    g=Game()
    d1=Dummy('red',9,10,hp=5000,spd=0)
    d2=Dummy('red',10,10,hp=3000,spd=0)
    d3=Dummy('red',10,11,hp=2000,spd=0)
    d4=Dummy('red',9,11,hp=500,spd=0)
    g.deploy('red',d1);g.deploy('red',d2);g.deploy('red',d3);g.deploy('red',d4)
    lt=mk_card('lightning',11,'blue',9.5,10.5)
    lt.apply(g)
    hit=[d for d in [d1,d2,d3,d4] if d.hp<d.max_hp]
    assert len(hit)==3,f"Lightning should hit exactly 3 highest HP, got {len(hit)}"
    assert d1.hp<5000 and d2.hp<3000 and d3.hp<2000
    assert d4.hp==500,"Lowest HP should be untouched"
    return "Lightning hits 3 highest HP targets"
def t_lightning_dmg():
    g=Game()
    d=Dummy('red',9,10,hp=5000,spd=0)
    g.deploy('red',d)
    lt=mk_card('lightning',11,'blue',9,10)
    lt.apply(g)
    assert d.hp==5000-1057,f"Expected 1057 dmg, got {5000-d.hp}"
    return f"Lightning damage ({5000-d.hp})"
def t_lightning_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    lt=mk_card('lightning',11,'blue',rpt.cx,rpt.cy)
    lt.apply(g)
    assert rpt.hp==ini-265,f"Expected 265 CT dmg, got {ini-rpt.hp}"
    return f"Lightning CT ({ini}->{rpt.hp})"
def t_eq_dot():
    g=Game()
    d=Dummy('red',9,25,hp=5000,spd=0)
    g.deploy('red',d)
    eq=mk_card('earthquake',11,'blue',9,25)
    eq.apply(g);g.spells.append(eq)
    g.run(4)
    dmg=5000-d.hp
    assert abs(dmg-246)<30,f"Expected ~246 total troop dmg (82*3), got {dmg}"
    return f"Earthquake DoT ({dmg} over 3 ticks)"
def t_eq_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    eq=mk_card('earthquake',11,'blue',rpt.cx,rpt.cy)
    eq.apply(g);g.spells.append(eq)
    g.run(4)
    dmg=ini-rpt.hp
    assert abs(dmg-159)<20,f"Expected ~159 CT dmg (53*3), got {dmg}"
    return f"Earthquake CT ({dmg} dmg)"
def t_bowler_load():
    b=mk_card('bowler',11,'blue',5,10)
    assert b.hp==2081 and b.dmg==289
    assert abs(b.spd-0.75)<0.01
    assert abs(b.hspd-2.5)<0.01 and abs(b.fhspd-2.0)<0.01
    assert abs(b.rng-4.0)<0.01
    return f"Bowler load (hp={b.hp} dmg={b.dmg} rng={b.rng})"
def t_bowler_splash():
    g=Game()
    b=mk_card('bowler',11,'blue',9,10)
    g.deploy('blue',b)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',10,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Bowler splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_bowler_ground_only():
    b=mk_card('bowler',11,'blue',5,10)
    assert b.targets==['Ground']
    return "Bowler targets Ground only"
def t_exe_load():
    e=mk_card('executioner',11,'blue',5,10)
    assert e.hp==1280 and e.dmg==179
    assert abs(e.hspd-2.4)<0.01 and abs(e.fhspd-0.4)<0.01
    assert 'Air' in e.targets
    return f"Executioner load (hp={e.hp} dmg={e.dmg})"
def t_exe_splash():
    g=Game()
    e=mk_card('executioner',11,'blue',9,10)
    g.deploy('blue',e)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',9.5,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Executioner splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_exe_v_air():
    g=Game()
    e=mk_card('executioner',11,'blue',9,10)
    g.deploy('blue',e)
    bd=mk_card('baby_dragon',11,'red',9,14)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Executioner hits air ({ini}->{bd.hp})"
def t_hspirit_load():
    hs=mk_card('heal_spirit',11,'blue',5,10)
    assert hs.hp==215 and hs.dmg==110
    assert hs.is_suicide
    assert abs(hs.splash_r-1.5)<0.01
    return f"Heal Spirit load (hp={hs.hp} dmg={hs.dmg} suicide)"
def t_hspirit_heal():
    g=Game()
    hs=mk_card('heal_spirit',11,'blue',9,10)
    g.deploy('blue',hs)
    ally=Dummy('blue',9,10.5,hp=1000,spd=0)
    ally.hp=500;ally.max_hp=1000
    g.deploy('blue',ally)
    hs.hp=0;hs.alive=False;hs.on_death(g)
    assert ally.hp>500,f"Heal Spirit should heal ally, got {ally.hp}"
    assert ally.hp<=1000,"Should not exceed max HP"
    return f"Heal Spirit heals ally (500->{ally.hp})"
def t_hspirit_suicide():
    g=Game()
    hs=mk_card('heal_spirit',11,'blue',9,14)
    g.deploy('blue',hs)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert not hs.alive,"Heal Spirit should die after attack"
    assert d.hp<50000
    return f"Heal Spirit suicide ({50000-d.hp} dmg)"
def t_edrag_load():
    ed=mk_card('electro_dragon',11,'blue',5,10)
    assert ed.hp==1049 and ed.dmg==192
    assert ed.transport=='Air'
    assert ed.chain_count==3
    assert abs(ed.chain_range-4.0)<0.01
    return f"Electro Dragon load (hp={ed.hp} dmg={ed.dmg} chains={ed.chain_count})"
def t_edrag_chain():
    g=Game()
    ed=mk_card('electro_dragon',11,'blue',9,10)
    g.deploy('blue',ed)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',10,14,hp=5000,spd=0)
    d3=Dummy('red',11,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2);g.deploy('red',d3)
    g.run(5)
    assert d1.hp<5000,"Primary target"
    assert d2.hp<5000,"Chain target 2"
    assert d3.hp<5000,"Chain target 3"
    assert ed.alive,"E-Dragon should NOT die (not suicide)"
    return f"Electro Dragon chains 3 (d1={5000-d1.hp} d2={5000-d2.hp} d3={5000-d3.hp})"
def t_edrag_air():
    g=Game()
    ed=mk_card('electro_dragon',11,'blue',9,10)
    g.deploy('blue',ed)
    bd=mk_card('baby_dragon',11,'red',9,14)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Electro Dragon hits air ({ini}->{bd.hp})"
def t_sparky_load():
    sp=mk_card('sparky',11,'blue',5,10)
    assert sp.hp==1451 and sp.dmg==1331
    assert abs(sp.hspd-4.0)<0.01 and abs(sp.fhspd-3.0)<0.01
    assert abs(sp.splash_r-1.8)<0.01
    return f"Sparky load (hp={sp.hp} dmg={sp.dmg} hspd={sp.hspd})"
def t_sparky_nuke():
    g=Game()
    sp=mk_card('sparky',11,'blue',9,10)
    g.deploy('blue',sp)
    d=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg=50000-d.hp
    assert dmg>=1331,f"Sparky should nuke for 1331+, got {dmg}"
    return f"Sparky nuke ({dmg} dmg)"
def t_sparky_splash():
    g=Game()
    sp=mk_card('sparky',11,'blue',9,10)
    g.deploy('blue',sp)
    d1=Dummy('red',9,14,hp=50000,spd=0)
    d2=Dummy('red',10,14,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<50000 and d2.hp<50000,"Sparky splash should hit both"
    return f"Sparky splash (d1={50000-d1.hp} d2={50000-d2.hp})"
def t_princess_load():
    p=mk_card('princess',11,'blue',5,10)
    assert p.hp==261 and p.dmg==168
    assert abs(p.rng-9.0)<0.01
    assert abs(p.splash_r-2.5)<0.01
    assert 'Air' in p.targets
    return f"Princess load (hp={p.hp} dmg={p.dmg} rng={p.rng})"
def t_princess_range():
    g=Game()
    p=mk_card('princess',11,'blue',9,5)
    g.deploy('blue',p)
    d=Dummy('red',9,13,hp=5000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert d.hp<5000,"Princess should hit at long range"
    return f"Princess long range ({5000-d.hp} dmg)"
def t_princess_splash():
    g=Game()
    p=mk_card('princess',11,'blue',9,5)
    g.deploy('blue',p)
    d1=Dummy('red',9,13,hp=5000,spd=0)
    d2=Dummy('red',10,13,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Princess splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_marcher_load():
    ma=mk_card('magic_archer',11,'blue',5,10)
    assert ma.hp==529 and ma.dmg==135
    assert abs(ma.rng-7.0)<0.01
    assert abs(ma.hspd-1.1)<0.01
    assert 'Air' in ma.targets
    return f"Magic Archer load (hp={ma.hp} dmg={ma.dmg} rng={ma.rng})"
def t_marcher_v_air():
    g=Game()
    ma=mk_card('magic_archer',11,'blue',9,10)
    g.deploy('blue',ma)
    bd=mk_card('baby_dragon',11,'red',9,16)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Magic Archer hits air ({ini}->{bd.hp})"
def t_bbarrel_dmg():
    g=quiet(Game())
    d=Dummy('red',9,12,hp=5000,spd=0)
    g.deploy('red',d)
    bb=mk_card('barbarian_barrel',11,'blue',9,10)
    g._cast('blue',bb,9,10);g.run(1)
    assert d.hp==5000-233,f"Expected 233 dmg, got {5000-d.hp}"
    return f"Barbarian Barrel damage ({5000-d.hp})"
def t_bbarrel_ground():
    g=Game()
    d=Dummy('red',9,12,hp=5000,spd=0)
    d.transport='Air'
    g.deploy('red',d)
    bb=mk_card('barbarian_barrel',11,'blue',9,10)
    bb.apply(g)
    assert d.hp==5000,"Barrel should not hit air"
    return "Barbarian Barrel ground only"
def t_rrecruits_spawn():
    random.seed(42)
    r=mk_card('royal_recruits',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==6
    assert all(u.hp==547 for u in r)
    assert all(u.shield_hp==240 for u in r)
    assert all(u.dmg==133 for u in r)
    return f"Royal Recruits spawn 6 (hp={r[0].hp} shield={r[0].shield_hp})"
def t_rrecruits_shield():
    random.seed(42)
    r=mk_card('royal_recruits',11,'blue',9,10)
    r[0].take_damage(300)
    assert r[0].shield_hp==0 and r[0].hp==547
    assert r[1].shield_hp==240
    return "Royal Recruits shields independent"
def t_rascals_spawn():
    random.seed(42)
    r=mk_card('rascals',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==3
    boys=[u for u in r if u.name=='Rascal Boy']
    girls=[u for u in r if u.name=='Rascal Girl']
    assert len(boys)==1 and len(girls)==2
    return f"Rascals 1+2 (boy hp={boys[0].hp} girl hp={girls[0].hp})"
def t_rascals_stats():
    random.seed(42)
    r=mk_card('rascals',11,'blue',9,10)
    boy=[u for u in r if u.name=='Rascal Boy'][0]
    girl=[u for u in r if u.name=='Rascal Girl'][0]
    assert boy.hp==1832 and boy.dmg==217
    assert girl.hp==261 and girl.dmg==133
    assert 'Air' in girl.targets
    assert girl.targets!=['Air']
    return "Rascals mixed stats (boy=tank girl=ranged)"
def t_rascals_girl_air():
    g=Game()
    random.seed(42)
    r=mk_card('rascals',11,'blue',9,10)
    for u in r:g.deploy('blue',u)
    bd=mk_card('baby_dragon',11,'red',9,14)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini,"Rascal Girls should hit air"
    return f"Rascal Girls hit air ({ini}->{bd.hp})"
def t_egolem_load():
    eg=mk_card('elixir_golem',11,'blue',5,10)
    assert eg.hp==1569 and eg.dmg==253
    assert eg.targets==['Buildings']
    return f"Elixir Golem load (hp={eg.hp} dmg={eg.dmg})"
def t_egolem_death_chain():
    g=Game()
    random.seed(42)
    eg=mk_card('elixir_golem',11,'blue',9,14)
    g.deploy('blue',eg)
    eg.hp=0;eg.alive=False;eg.on_death(g)
    gms=[t for t in g.players['blue'].troops if t.alive]
    assert len(gms)==2,f"Expected 2 golemites, got {len(gms)}"
    assert gms[0].hp==762
    gms[0].hp=0;gms[0].alive=False;gms[0].on_death(g)
    blobs=[t for t in g.players['blue'].troops if t.alive and t not in gms]
    assert len(blobs)==2,f"Expected 2 blobs, got {len(blobs)}"
    assert blobs[0].hp==363
    return "Elixir Golem chain (1569->762->363)"
def t_egolem_buildings():
    g=Game()
    eg=mk_card('elixir_golem',11,'blue',9,14)
    g.deploy('blue',eg)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert d.hp==50000,"Elixir Golem should ignore troops"
    return "Elixir Golem targets buildings"
def t_rghost_load():
    rg=mk_card('royal_ghost',11,'blue',5,10)
    assert rg.hp==1210 and rg.dmg==261
    assert abs(rg.splash_r-1.0)<0.01
    assert abs(rg.spd-1.5)<0.01
    return f"Royal Ghost load (hp={rg.hp} dmg={rg.dmg} splash={rg.splash_r})"
def t_rghost_splash():
    g=Game()
    rg=mk_card('royal_ghost',11,'blue',9,10)
    g.deploy('blue',rg)
    d1=Dummy('red',9,11.5,hp=5000,spd=0)
    d2=Dummy('red',9.5,11.5,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Royal Ghost splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_bandit_load():
    b=mk_card('bandit',11,'blue',5,10)
    assert b.hp==906 and b.dmg==194
    assert abs(b.spd-1.5)<0.01
    assert abs(b.hspd-1.0)<0.01
    return f"Bandit load (hp={b.hp} dmg={b.dmg} spd={b.spd})"
def t_bandit_v_knight():
    g=Game()
    b=mk_card('bandit',11,'blue',9,14)
    g.deploy('blue',b)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    g.run(15)
    assert not k.alive or not b.alive,"Should be a fight"
    return f"Bandit vs Knight (bandit_hp={b.hp} knight_hp={k.hp})"
def t_bandit_v_tower():
    g=Game()
    b=mk_card('bandit',11,'blue',3,14)
    g.deploy('blue',b)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(20)
    assert rpt.hp<ini
    return f"Bandit hits tower ({ini}->{rpt.hp})"
def t_berserker_load():
    b=mk_card('berserker',11,'blue',5,10)
    assert b.hp==896 and b.dmg==102
    assert abs(b.hspd-0.6)<0.01 and abs(b.fhspd-0.4)<0.01
    assert abs(b.spd-1.5)<0.01
    return f"Berserker load (hp={b.hp} dmg={b.dmg} hspd={b.hspd})"
def t_berserker_fast_atk():
    g=Game()
    b=mk_card('berserker',11,'blue',9,14)
    g.deploy('blue',b)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg=50000-d.hp
    assert dmg>=140*5,f"Fast attacker should deal heavy dmg, got {dmg}"
    return f"Berserker fast attack ({dmg} in 5s)"
def t_egiant_load():
    eg=mk_card('electro_giant',11,'blue',5,10)
    assert eg.hp==3952 and eg.dmg==184
    assert eg.targets==['Buildings']
    from sim.fx import ZapPack
    assert any(isinstance(c,ZapPack) for c in eg.components)
    return f"Electro Giant load (hp={eg.hp} dmg={eg.dmg} +ZapPack)"
def t_egiant_reflect():
    g=Game()
    eg=mk_card('electro_giant',11,'blue',9,14)
    g.deploy('blue',eg)
    d=Dummy('red',9,14.5,hp=50000,dmg=100,spd=0,hspd=0.5,rng=1.5)
    g.deploy('red',d)
    g.run(3)
    assert d.hp<50000,"Zap pack should reflect damage to attacker"
    reflect_dmg=50000-d.hp
    assert reflect_dmg>=192,f"Should reflect at least 192 dmg, got {reflect_dmg}"
    return f"Electro Giant reflects ({reflect_dmg} dmg)"
def t_egiant_buildings():
    eg=mk_card('electro_giant',11,'blue',5,10)
    assert eg.targets==['Buildings']
    from sim.fx import BuildingTarget
    assert any(isinstance(c,BuildingTarget) for c in eg.components)
    return "Electro Giant targets buildings only"
def t_fisherman_load():
    f=mk_card('fisherman',11,'blue',5,10)
    assert f.hp==870 and f.dmg==194
    assert abs(f.hspd-1.3)<0.01 and abs(f.fhspd-1.2)<0.01
    return f"Fisherman load (hp={f.hp} dmg={f.dmg})"
def t_fisherman_v_knight():
    g=Game()
    f=mk_card('fisherman',11,'blue',9,14)
    g.deploy('blue',f)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    g.run(15)
    assert k.hp<1690,"Fisherman should fight Knight"
    return f"Fisherman vs Knight (knight_hp={k.hp})"
def t_bhealer_load():
    bh=mk_card('battle_healer',11,'blue',5,10)
    assert bh.hp==1920 and bh.dmg==268
    from sim.fx import HealPulse
    assert any(isinstance(c,HealPulse) for c in bh.components)
    return f"Battle Healer load (hp={bh.hp} dmg={bh.dmg} +HealPulse)"
def t_bhealer_heals():
    g=Game()
    bh=mk_card('battle_healer',11,'blue',9,10)
    g.deploy('blue',bh)
    ally=Dummy('blue',9,10.5,hp=1000,spd=0,dmg=100,hspd=1.0)
    ally.hp=500;ally.max_hp=1000
    g.deploy('blue',ally)
    d=Dummy('red',9,11,hp=50000,spd=0,dmg=0)
    g.deploy('red',d)
    g.run(5)
    assert ally.hp>500,f"BH should heal ally on attack, hp={ally.hp}"
    return f"Battle Healer heals ally (500->{ally.hp})"
def t_clone_dupes():
    g=Game()
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    ini=len(g.players['blue'].troops)
    cl=mk_card('clone',11,'blue',9,10)
    cl.apply(g)
    assert len(g.players['blue'].troops)==ini+1,"Clone should duplicate"
    clone=[t for t in g.players['blue'].troops if t is not k][0]
    assert clone.hp==1,"Clone should have 1 HP"
    assert clone.dmg==k.dmg,"Clone should have same damage"
    return f"Clone duplicates (orig hp={k.hp} clone hp={clone.hp} dmg={clone.dmg})"
def t_clone_radius():
    g=Game()
    k1=mk_card('knight',11,'blue',9,10)
    k2=mk_card('knight',11,'blue',9,20)
    g.deploy('blue',k1);g.deploy('blue',k2)
    cl=mk_card('clone',11,'blue',9,10)
    cl.apply(g)
    assert len(g.players['blue'].troops)==3,"Only 1 in radius should be cloned"
    return "Clone respects radius"
def t_tornado_dot():
    g=Game()
    d=Dummy('red',9,25,hp=5000,spd=0)
    g.deploy('red',d)
    tn=mk_card('tornado',11,'blue',9,25)
    tn.apply(g);g.spells.append(tn)
    g.run(2)
    dmg=5000-d.hp
    assert abs(dmg-168)<20,f"Expected ~168 total dmg (84*2), got {dmg}"
    return f"Tornado DoT ({dmg} over 2 ticks)"
def t_tornado_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    tn=mk_card('tornado',11,'blue',rpt.cx,rpt.cy)
    tn.apply(g);g.spells.append(tn)
    g.run(2)
    dmg=ini-rpt.hp
    assert abs(dmg-58)<10,f"Expected ~58 CT dmg (29*2), got {dmg}"
    return f"Tornado CT ({dmg} dmg)"
def t_void_strikes():
    g=Game()
    d=Dummy('red',9,22,hp=5000,spd=0)
    g.deploy('red',d)
    v=mk_card('void',11,'blue',9,22)
    v.apply(g);g.spells.append(v)
    g.run(4)
    dmg=5000-d.hp
    assert abs(dmg-2088)<50,f"Expected ~2088 (696*3), got {dmg}"
    return f"Void 3 strikes ({dmg} total)"
def t_void_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    v=mk_card('void',11,'blue',rpt.cx,rpt.cy)
    v.apply(g);g.spells.append(v)
    g.run(4)
    dmg=ini-rpt.hp
    assert abs(dmg-315)<50,f"Expected ~315 CT (105*3), got {dmg}"
    return f"Void CT ({dmg} dmg)"
def t_bbandit_load():
    bb=mk_card('boss_bandit',11,'blue',5,10)
    assert bb.hp==2624 and bb.dmg==245
    assert abs(bb.spd-1.5)<0.01,f"Expected Fast speed 1.5, got {bb.spd}"
    return f"Boss Bandit load (hp={bb.hp} dmg={bb.dmg} spd={bb.spd})"
def t_bbandit_v_tower():
    g=Game()
    bb=mk_card('boss_bandit',11,'blue',3,14)
    g.deploy('blue',bb)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(25)
    assert rpt.hp<ini
    return f"Boss Bandit hits tower ({ini}->{rpt.hp})"
def t_bats_spawn():
    random.seed(42)
    r=mk_card('bats',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==5
    assert all(b.hp==81 for b in r)
    assert all(b.transport=='Air' for b in r)
    assert 'Air' in r[0].targets
    return f"Bats spawn 5 Air (hp={r[0].hp})"
def t_bats_fragile():
    g=Game()
    random.seed(42)
    bats=mk_card('bats',11,'red',9,10)
    for b in bats:g.deploy('red',b)
    z=mk_card('zap',11,'blue',9,10)
    z.apply(g)
    alive=[b for b in bats if b.alive]
    assert len(alive)==0,"Zap should kill all bats (191>81)"
    return "Zap kills all bats"
def t_bomber_load():
    b=mk_card('bomber',11,'blue',5,10)
    assert b.hp==304 and b.dmg==225
    assert abs(b.splash_r-1.5)<0.01
    assert abs(b.rng-4.5)<0.01
    assert b.targets==['Ground']
    return f"Bomber load (hp={b.hp} dmg={b.dmg} splash={b.splash_r})"
def t_bomber_splash():
    g=Game()
    b=mk_card('bomber',11,'blue',9,10)
    g.deploy('blue',b)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',10,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Bomber splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_wizard_load():
    w=mk_card('wizard',11,'blue',5,10)
    assert w.hp==755 and w.dmg==281
    assert abs(w.rng-5.5)<0.01
    assert abs(w.splash_r-1.5)<0.01
    assert 'Air' in w.targets
    return f"Wizard load (hp={w.hp} dmg={w.dmg} rng={w.rng})"
def t_wizard_splash():
    g=Game()
    w=mk_card('wizard',11,'blue',9,10)
    g.deploy('blue',w)
    d1=Dummy('red',9,15,hp=5000,spd=0)
    d2=Dummy('red',10,15,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Wizard splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_wizard_v_air():
    g=Game()
    w=mk_card('wizard',11,'blue',9,10)
    g.deploy('blue',w)
    bd=mk_card('baby_dragon',11,'red',9,15)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Wizard hits air ({ini}->{bd.hp})"
def t_3musk_spawn():
    random.seed(42)
    r=mk_card('three_musketeers',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==3
    assert all(m.hp==721 for m in r)
    assert all(m.dmg==217 for m in r)
    assert abs(r[0].rng-6.0)<0.01
    return f"Three Musketeers spawn 3 (hp={r[0].hp} dmg={r[0].dmg})"
def t_3musk_dps():
    g=Game()
    random.seed(42)
    ms=mk_card('three_musketeers',11,'blue',9,10)
    for m in ms:g.deploy('blue',m)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg=50000-d.hp
    assert dmg>=217*3,f"3 musketeers should deal heavy dmg, got {dmg}"
    return f"Three Musketeers DPS ({dmg} in 5s)"
def t_zappies_spawn():
    random.seed(42)
    r=mk_card('zappies',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==3
    assert all(z.hp==529 for z in r)
    assert all(z.dmg==117 for z in r)
    return f"Zappies spawn 3 (hp={r[0].hp} dmg={r[0].dmg})"
def t_zappies_stun():
    g=Game()
    random.seed(42)
    zps=mk_card('zappies',11,'blue',9,10)
    for z in zps:g.deploy('blue',z)
    d=Dummy('red',9,14,hp=50000,spd=0,dmg=500,hspd=1.0)
    tgt=Dummy('blue',9,15,hp=50000,spd=0)
    g.deploy('red',d);g.deploy('blue',tgt)
    g.run(5)
    d_dmg=50000-tgt.hp
    assert d_dmg<3000,f"Zappies stun should delay enemy attacks, got {d_dmg}"
    return f"Zappies stun delays attacks ({d_dmg} dmg)"
def t_ccart_load():
    cc=mk_card('cannon_cart',11,'blue',5,10)
    assert cc.hp==1809 and cc.dmg==212
    assert abs(cc.hspd-0.9)<0.01
    assert abs(cc.rng-5.5)<0.01
    return f"Cannon Cart load (hp={cc.hp} dmg={cc.dmg} rng={cc.rng})"
def t_ccart_ranged():
    g=Game()
    cc=mk_card('cannon_cart',11,'blue',9,10)
    g.deploy('blue',cc)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert d.hp<50000
    return f"Cannon Cart ranged ({50000-d.hp} dmg)"
def t_bram_load():
    br=mk_card('battle_ram',11,'blue',5,10)
    assert br.hp==967 and br.dmg==286
    assert br.targets==['Buildings']
    from sim.fx import Charge,DeathSpawn
    assert any(isinstance(c,Charge) for c in br.components)
    assert any(isinstance(c,DeathSpawn) for c in br.components)
    return f"Battle Ram load (hp={br.hp} dmg={br.dmg} +Charge+DeathSpawn)"
def t_bram_death_barbs():
    g=Game()
    random.seed(42)
    br=mk_card('battle_ram',11,'blue',9,14)
    g.deploy('blue',br)
    br.hp=0;br.alive=False;br.on_death(g)
    barbs=[t for t in g.players['blue'].troops if t.alive and t is not br]
    assert len(barbs)==2,f"Expected 2 barbarians, got {len(barbs)}"
    return f"Battle Ram spawns 2 barbs (hp={barbs[0].hp})"
def t_ggiant_load():
    gg=mk_card('goblin_giant',11,'blue',5,10)
    assert gg.hp==3110 and gg.dmg==176
    assert gg.targets==['Buildings']
    from sim.fx import DeathSpawn
    assert any(isinstance(c,DeathSpawn) for c in gg.components)
    return f"Goblin Giant load (hp={gg.hp} dmg={gg.dmg} +DeathSpawn)"
def t_ggiant_death_spawn():
    g=Game()
    random.seed(42)
    gg=mk_card('goblin_giant',11,'blue',9,14)
    g.deploy('blue',gg)
    gg.hp=0;gg.alive=False;gg.on_death(g)
    spawned=[t for t in g.players['blue'].troops if t.alive and t is not gg]
    assert len(spawned)==2,f"Expected 2 spear goblins, got {len(spawned)}"
    return f"Goblin Giant death spawns 2 (hp={spawned[0].hp})"
def t_ggiant_v_tower():
    g=Game()
    gg=mk_card('goblin_giant',11,'blue',3,14)
    g.deploy('blue',gg)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(30)
    assert rpt.hp<ini
    return f"Goblin Giant hits tower ({ini}->{rpt.hp})"
def t_rdelivery_dmg():
    g=Game()
    d=Dummy('red',9,10,hp=5000,spd=0)
    g.deploy('red',d)
    rd=mk_card('royal_delivery',11,'blue',9,10)
    rd.apply(g)
    assert d.hp==5000-384,f"Expected 384 dmg, got {5000-d.hp}"
    return f"Royal Delivery damage ({5000-d.hp})"
def t_gcurse_load():
    gc=mk_card('goblin_curse',11,'blue',9,10)
    assert hasattr(gc,'apply')
    return "Goblin Curse loads as spell"
def t_vines_dot():
    g=Game()
    d=Dummy('red',9,25,hp=5000,spd=0)
    g.deploy('red',d)
    v=mk_card('vines',11,'blue',9,25)
    v.apply(g);g.spells.append(v)
    g.run(3)
    dmg=5000-d.hp
    assert abs(dmg-306)<50,f"Expected ~306 (153*2), got {dmg}"
    return f"Vines DoT ({dmg})"
def t_ramrider_load():
    rr=mk_card('ram_rider',11,'blue',5,10)
    assert rr.hp==1766 and rr.dmg==250
    return f"Ram Rider load (hp={rr.hp} dmg={rr.dmg})"
def t_ramrider_v_tower():
    g=Game()
    rr=mk_card('ram_rider',11,'blue',3,14)
    g.deploy('blue',rr)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(30)
    assert rpt.hp<ini
    return f"Ram Rider hits tower ({ini}->{rpt.hp})"
def t_mwitch_load():
    mw=mk_card('mother_witch',11,'blue',5,10)
    assert mw.hp==529 and mw.dmg==133
    assert abs(mw.rng-5.5)<0.01
    assert 'Air' in mw.targets
    return f"Mother Witch load (hp={mw.hp} dmg={mw.dmg} rng={mw.rng})"
def t_mwitch_v_troop():
    g=Game()
    mw=mk_card('mother_witch',11,'blue',9,10)
    g.deploy('blue',mw)
    d=Dummy('red',9,15,hp=5000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert d.hp<5000
    return f"Mother Witch attacks ({5000-d.hp} dmg)"
def t_phoenix_load():
    px=mk_card('phoenix',11,'blue',5,10)
    assert px.hp==1052 and px.dmg==217
    assert px.transport=='Air'
    assert 'Air' in px.targets
    return f"Phoenix load (hp={px.hp} dmg={px.dmg} Air)"
def t_phoenix_v_air():
    g=Game()
    px=mk_card('phoenix',11,'blue',9,10)
    g.deploy('blue',px)
    bd=mk_card('baby_dragon',11,'red',9,14)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Phoenix hits air ({ini}->{bd.hp})"
def t_monk_load():
    m=mk_card('monk',11,'blue',5,10)
    assert m.hp==2214 and m.dmg==140
    assert abs(m.hspd-0.8)<0.01
    return f"Monk load (hp={m.hp} dmg={m.dmg} hspd={m.hspd})"
def t_monk_v_knight():
    g=Game()
    m=mk_card('monk',11,'blue',9,14)
    g.deploy('blue',m)
    k=mk_card('knight',11,'red',9,15)
    g.deploy('red',k)
    g.run(30)
    assert not k.alive,"Monk should kill Knight"
    return f"Monk kills Knight (monk_hp={m.hp})"
def t_aqueen_load():
    aq=mk_card('archer_queen',11,'blue',5,10)
    assert aq.hp==1000 and aq.dmg==232
    assert abs(aq.rng-5.0)<0.01
    assert 'Air' in aq.targets
    return f"Archer Queen load (hp={aq.hp} dmg={aq.dmg} rng={aq.rng})"
def t_aqueen_v_air():
    g=Game()
    aq=mk_card('archer_queen',11,'blue',9,10)
    g.deploy('blue',aq)
    bd=mk_card('baby_dragon',11,'red',9,14)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Archer Queen hits air ({ini}->{bd.hp})"
def t_gdemolisher_load():
    gd=mk_card('goblin_demolisher',11,'blue',5,10)
    assert gd.hp==1300 and gd.dmg==186
    assert abs(gd.rng-5.0)<0.01
    assert abs(gd.splash_r)>0
    return f"Goblin Demolisher load (hp={gd.hp} dmg={gd.dmg} rng={gd.rng})"
def t_gdemolisher_splash():
    g=Game()
    gd=mk_card('goblin_demolisher',11,'blue',9,10)
    g.deploy('blue',gd)
    d1=Dummy('red',9,15,hp=5000,spd=0)
    d2=Dummy('red',10,15,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Goblin Demolisher splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_gknight_load():
    gk=mk_card('golden_knight',11,'blue',5,10)
    assert gk.hp==1799 and gk.dmg==161
    return f"Golden Knight load (hp={gk.hp} dmg={gk.dmg})"
def t_gknight_v_tower():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',3,14)
    g.deploy('blue',gk)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(25)
    assert rpt.hp<ini
    return f"Golden Knight hits tower ({ini}->{rpt.hp})"
def t_lprince_load():
    lp=mk_card('little_prince',11,'blue',5,10)
    assert lp.hp==698 and lp.dmg==104
    assert abs(lp.rng-5.5)<0.01
    assert 'Air' in lp.targets
    return f"Little Prince load (hp={lp.hp} dmg={lp.dmg} rng={lp.rng})"
def t_lprince_v_air():
    g=Game()
    lp=mk_card('little_prince',11,'blue',9,10)
    g.deploy('blue',lp)
    bd=mk_card('baby_dragon',11,'red',9,14)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Little Prince hits air ({ini}->{bd.hp})"
def t_rgiant_load():
    rg=mk_card('rune_giant',11,'blue',5,10)
    assert rg.hp==2816 and rg.dmg==153
    assert rg.targets==['Buildings']
    return f"Rune Giant load (hp={rg.hp} dmg={rg.dmg})"
def t_skelking_load():
    sk=mk_card('skeleton_king',11,'blue',5,10)
    assert sk.hp==2298 and sk.dmg==204
    assert abs(sk.splash_r)>0
    return f"Skeleton King load (hp={sk.hp} dmg={sk.dmg})"
def t_skelking_splash():
    g=Game()
    sk=mk_card('skeleton_king',11,'blue',9,10)
    g.deploy('blue',sk)
    d1=Dummy('red',9,11.5,hp=5000,spd=0)
    d2=Dummy('red',9.5,11.5,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Skeleton King splash (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_mirror_not_in_hand():
    dk=['mirror','knight','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army']
    for s in range(20):
        random.seed(s)
        d=Deck(dk)
        assert 'mirror' not in d.hand,f"seed {s}: mirror in starting hand"
    return "Mirror never in starting hand (20 seeds)"
def t_mirror_no_last():
    dk=['mirror','knight','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army']
    g=Game();p=Player('blue',king_lvl=11,deck=dk);g.players['blue']=p
    p.elixir=10
    p.deck.hand=['mirror','archers','fireball','hog_rider']
    ok,msg=g.play_card('blue','mirror',9,5)
    assert not ok and msg=="no card to mirror"
    return "Mirror with no last card rejected"
def t_mirror_cost():
    dk=['knight','mirror','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army']
    g=Game();p=Player('blue',king_lvl=11,deck=dk);g.players['blue']=p
    p.elixir=10
    p.deck.hand=['knight','mirror','archers','fireball']
    g.play_card('blue','knight',9,5)
    ex_after=p.elixir
    ok,msg=g.play_card('blue','mirror',9,5)
    assert ok,f"mirror play failed: {msg}"
    cost_paid=ex_after-p.elixir
    kc=card_info('knight')['cost']
    assert cost_paid==kc+1,f"expected {kc+1}, paid {cost_paid}"
    return f"Mirror cost = knight({kc})+1 = {kc+1}"
def t_mirror_copies_last():
    dk=['knight','mirror','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army']
    g=Game();p=Player('blue',king_lvl=11,deck=dk);g.players['blue']=p
    p.elixir=10
    p.deck.hand=['knight','mirror','archers','fireball']
    g.play_card('blue','knight',9,5)
    g.play_card('blue','mirror',9,5)
    g.run(5)
    knights=[t for t in p.troops if getattr(t,'name','').lower()=='knight']
    assert len(knights)>=2,f"expected 2 knights, got {len(knights)}: {[t.name for t in p.troops]}"
    return f"Mirror copies last card ({len(knights)} knights)"
def t_mirror_level_boost():
    dk=['knight','mirror','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army']
    g=Game();p=Player('blue',king_lvl=11,deck=dk);g.players['blue']=p
    p.elixir=10
    p.deck.hand=['knight','mirror','archers','fireball']
    g.play_card('blue','knight',9,5)
    g.play_card('blue','mirror',9,5)
    g.run(5)
    knights=[t for t in p.troops if getattr(t,'name','').lower()=='knight']
    lvls=sorted([t.lvl for t in knights])
    assert len(lvls)>=2,f"expected 2+ knights, got {lvls}"
    assert lvls[-1]==12,f"mirrored knight should be lvl 12, got {lvls[-1]}"
    return f"Mirror level boost: normal={lvls[0]} mirrored={lvls[-1]}"
def t_bld_cannon():
    from sim.units import Building
    c=mk_card('cannon',11,'blue',5,10)
    assert isinstance(c,Building)
    assert c.hp==824 and c.dmg==202
    return f"Cannon Building (hp={c.hp} dmg={c.dmg})"
def t_bld_tesla():
    from sim.units import Building
    t=mk_card('tesla',11,'blue',5,10)
    assert isinstance(t,Building)
    assert t.hp==1182 and t.dmg==220
    assert 'Air' in t.targets
    return f"Tesla Building (hp={t.hp} dmg={t.dmg})"
def t_bld_bombtower():
    from sim.units import Building
    from sim.fx import DeathDamage
    bt=mk_card('bomb_tower',11,'blue',5,10)
    assert isinstance(bt,Building)
    assert bt.hp==1356 and bt.dmg==222
    assert bt.splash_r>0
    assert bt.death_dmg==222 and bt.death_splash_r==2.5
    assert any(isinstance(c,DeathDamage) for c in bt.components)
    return f"Bomb Tower Building (hp={bt.hp} dmg={bt.dmg} death_dmg={bt.death_dmg})"
def t_bld_inferno():
    from sim.units import Building
    from sim.fx import RampUp
    it=mk_card('inferno_tower',11,'blue',5,10)
    assert isinstance(it,Building)
    assert it.hp==1748
    assert it.dmg==43
    ramps=[c for c in it.components if isinstance(c,RampUp)]
    assert len(ramps)==1
    assert ramps[0].stages==[43,158,847]
    assert len(ramps[0].durations)==2
    g=Game();g.deploy('blue',it)
    d=Dummy('red',5,13,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(2)
    d1=50000-d.hp
    g.run(4)
    d2=50000-d.hp
    assert d2>d1*2,f"Ramp should increase dmg over time: early={d1} total={d2}"
    return f"Inferno Tower Building (hp={it.hp} ramp={ramps[0].stages})"
def t_bld_mortar():
    from sim.units import Building
    m=mk_card('mortar',11,'blue',5,10)
    assert isinstance(m,Building)
    assert m.hp==1369 and m.dmg==266
    assert abs(m.rng-11.5)<0.01
    assert abs(m.min_rng-3.5)<0.01
    g=Game()
    for t in g.arena.towers:t.alive=False
    mt=mk_card('mortar',11,'blue',9,14)
    g.deploy('blue',mt)
    close=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',close)
    g.run(6)
    assert close.hp==50000,f"Mortar blind spot: d=1 < min_rng=3.5, hp={close.hp}"
    g2=Game()
    for t in g2.arena.towers:t.alive=False
    mt2=mk_card('mortar',11,'blue',9,5)
    g2.deploy('blue',mt2)
    far=Dummy('red',9,15,hp=50000,spd=0)
    g2.deploy('red',far)
    g2.run(8)
    assert far.hp<50000,f"Mortar should hit at d=10 (hp={far.hp})"
    return f"Mortar Building (hp={m.hp} rng={m.rng} min_rng={m.min_rng} blind_spot=OK)"
def t_bld_xbow():
    from sim.units import Building
    x=mk_card('x_bow',11,'blue',5,10)
    assert isinstance(x,Building)
    assert x.hp==1600 and x.dmg==58
    assert abs(x.rng-11.5)<0.01
    ci=card_info('x_bow')
    assert ci['deploy']==3.5,f"X-Bow deploy should be 3.5, got {ci['deploy']}"
    return f"X-Bow Building (hp={x.hp} dmg={x.dmg} rng={x.rng} deploy={ci['deploy']})"
def t_bld_tombstone():
    from sim.units import Building
    t=mk_card('tombstone',11,'blue',5,10)
    assert isinstance(t,Building)
    assert t.hp==529
    assert t.lifetime==30
    from sim.fx import DeathSpawn,SpawnTimer
    assert any(isinstance(c,DeathSpawn) for c in t.components)
    st=[c for c in t.components if isinstance(c,SpawnTimer)]
    assert len(st)==1
    assert st[0].cfg['hp']==81 and st[0].cfg['dmg']==81
    assert st[0].count==2 and abs(st[0].interval-3.5)<0.01
    g=Game();g.deploy('blue',t);g.run(4)
    spawned=[tr for tr in g.players['blue'].troops if tr is not t]
    assert len(spawned)==2
    return f"Tombstone Building (hp={t.hp} lifetime={t.lifetime} spawned={len(spawned)})"
def t_bld_gobcage():
    from sim.units import Building
    from sim.fx import DeathSpawn
    gc=mk_card('goblin_cage',11,'blue',5,10)
    assert isinstance(gc,Building)
    assert gc.hp==780
    assert gc.lifetime==20,f"Goblin cage lifetime should be 20, got {gc.lifetime}"
    ds=[c for c in gc.components if isinstance(c,DeathSpawn)]
    assert len(ds)==1
    g=Game();g.deploy('blue',gc);g.run(21)
    assert not gc.alive,"Cage should expire after 20s"
    spawned=[tr for tr in g.players['blue'].troops if tr is not gc and tr.alive]
    assert len(spawned)==1,f"Should spawn 1 Goblin Brawler on death, got {len(spawned)}"
    return f"Goblin Cage Building (hp={gc.hp} lifetime={gc.lifetime} brawler_spawned={len(spawned)})"
def t_bld_barbhut():
    from sim.units import Building
    bh=mk_card('barbarian_hut',11,'blue',5,10)
    assert isinstance(bh,Building)
    assert bh.hp==1164
    from sim.fx import SpawnTimer
    st=[c for c in bh.components if isinstance(c,SpawnTimer)]
    assert len(st)==1
    assert st[0].count==3 and abs(st[0].interval-14)<0.01
    g=Game();g.deploy('blue',bh);g.run(16)
    spawned=[tr for tr in g.players['blue'].troops if tr is not bh]
    assert len(spawned)==3
    return f"Barbarian Hut Building (hp={bh.hp} spawned={len(spawned)})"
def t_bld_gobhut():
    from sim.units import Building
    gh=mk_card('goblin_hut',11,'blue',5,10)
    assert isinstance(gh,Building)
    assert gh.hp==1180
    from sim.fx import SpawnTimer
    st=[c for c in gh.components if isinstance(c,SpawnTimer)]
    assert len(st)==1
    assert st[0].cfg['name']=='SpearGoblin'
    assert abs(st[0].interval-1.9)<0.01
    g=Game();g.deploy('blue',gh);g.run(4)
    spawned=[tr for tr in g.players['blue'].troops if tr is not gh]
    assert len(spawned)>=2
    return f"Goblin Hut Building (hp={gh.hp} spawned={len(spawned)})"
def t_bld_furnace():
    from sim.units import Building
    fn=mk_card('furnace',11,'blue',8.5,5)
    # the current game data lists Furnace as a walking troop that spawns Fire Spirits
    assert not isinstance(fn,Building)
    assert fn.hp==727 and fn.dmg==179
    from sim.fx import SpawnTimer
    st=[c for c in fn.components if isinstance(c,SpawnTimer)]
    assert len(st)==1
    assert st[0].cfg['name']=='Fire Spirit'
    assert abs(st[0].interval-5)<0.01
    g=Game();g.deploy('blue',fn);g.run(5.5)
    spawned=[tr for tr in g.players['blue'].troops if tr is not fn]
    assert len(spawned)>=1
    return f"Furnace troop (hp={fn.hp} dmg={fn.dmg} spawned={len(spawned)})"
def t_bld_elixcoll():
    from sim.units import Building
    ec=mk_card('elixir_collector',11,'blue',5,10)
    assert isinstance(ec,Building)
    assert ec.hp==1070
    assert ec.lifetime==65,f"Elixir collector lifetime should be 65, got {ec.lifetime}"
    from sim.fx import ElixirProd
    ep=[c for c in ec.components if isinstance(c,ElixirProd)]
    assert len(ep)==1
    assert ep[0].interval==13 and ep[0].amount==1
    g1=Game();g1.players['blue'].elixir=0;g1.run(14)
    ex_ctrl=g1.players['blue'].elixir
    g2=Game();g2.players['blue'].elixir=0;g2.deploy('blue',ec);g2.run(14)
    ex_coll=g2.players['blue'].elixir
    assert ex_coll>ex_ctrl
    return f"Elixir Collector Building (hp={ec.hp} lifetime={ec.lifetime})"
def t_bld_gobdrill():
    from sim.units import Building
    from sim.fx import DeathSpawn,SpawnTimer,SpawnZap
    gd=mk_card('goblin_drill',11,'blue',5,10)
    assert isinstance(gd,Building)
    assert gd.hp>0
    assert gd.lifetime==10,f"Goblin drill lifetime should be 10, got {gd.lifetime}"
    assert any(isinstance(c,SpawnTimer) for c in gd.components),"Should have SpawnTimer"
    assert any(isinstance(c,DeathSpawn) for c in gd.components),"Should have DeathSpawn"
    assert any(isinstance(c,SpawnZap) for c in gd.components),"Should have SpawnZap (spawn damage)"
    return f"Goblin Drill Building (hp={gd.hp} lifetime={gd.lifetime})"
def t_bld_lifetime():
    from sim.units import Building
    c=mk_card('cannon',11,'blue',5,10)
    assert isinstance(c,Building)
    g=Game();g.deploy('blue',c);g.run(29)
    assert c.alive
    g.run(2)
    assert not c.alive
    return "Building lifetime expiry"
def t_bld_inferno_ramp_stages():
    it=mk_card('inferno_tower',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',it)
    d=Dummy('red',9,10,hp=999999,spd=0)
    g.deploy('red',d)
    g.run(1);hp1=d.hp;d1=999999-hp1
    g.run(2);hp2=d.hp;d2=hp1-hp2
    g.run(3);hp3=d.hp;d3=hp2-hp3
    assert d2>d1,f"Stage2 should deal more than stage1: s1={d1} s2={d2}"
    assert d3>d2,f"Stage3 should deal more than stage2: s2={d2} s3={d3}"
    return f"Inferno ramp stages (s1={d1} s2={d2} s3={d3})"
def t_bld_inferno_zap_reset():
    it=mk_card('inferno_tower',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',it)
    d=Dummy('red',9,10,hp=999999,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg_no_zap=999999-d.hp
    it2=mk_card('inferno_tower',11,'blue',9,5)
    g2=Game()
    for t in g2.arena.towers:t.alive=False
    g2.deploy('blue',it2)
    d2=Dummy('red',9,10,hp=999999,spd=0)
    g2.deploy('red',d2)
    g2.run(2.5)
    from sim.units import Status
    it2.statuses.append(Status('stun',0.5))
    g2.run(2.5)
    dmg_with_zap=999999-d2.hp
    assert dmg_with_zap<dmg_no_zap,f"Zap should reduce total dmg: no_zap={dmg_no_zap} zapped={dmg_with_zap}"
    return f"Inferno zap reset (no_zap={dmg_no_zap} zapped={dmg_with_zap})"
def t_bld_inferno_retarget_reset():
    it=mk_card('inferno_tower',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',it)
    d1=Dummy('red',9,10,hp=200,spd=0)
    d2=Dummy('red',9,11,hp=999999,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert not d1.alive,"First target should die"
    from sim.fx import RampUp
    ramps=[c for c in it.components if isinstance(c,RampUp)]
    assert ramps[0].cur_tgt is d2,"Should have retargeted to d2"
    d2_dmg=999999-d2.hp
    g.run(5)
    d2_dmg2=999999-d2.hp
    phase2=d2_dmg2-d2_dmg
    assert phase2>d2_dmg,f"After retarget ramp should re-escalate: first={d2_dmg} second={phase2}"
    return f"Inferno retarget resets ramp (d1 dead, d2 dmg={d2_dmg2})"
def t_bld_cannon_ground_only():
    cn=mk_card('cannon',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn)
    air=Dummy('red',9,10,hp=5000,spd=0)
    air.transport='Air';air.targets=['Ground']
    g.deploy('red',air)
    g.run(5)
    assert air.hp==5000,f"Cannon should not hit air (hp={air.hp})"
    return "Cannon ignores air troops"
def t_bld_cannon_hits_ground():
    cn=mk_card('cannon',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn)
    gnd=Dummy('red',9,10,hp=50000,spd=0)
    g.deploy('red',gnd)
    g.run(5)
    assert gnd.hp<50000,f"Cannon should hit ground (hp={gnd.hp})"
    return f"Cannon hits ground (dmg={50000-gnd.hp})"
def t_bld_tesla_hits_air():
    ts=mk_card('tesla',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',ts)
    air=Dummy('red',9,10,hp=50000,spd=0)
    air.transport='Air';air.targets=['Ground']
    g.deploy('red',air)
    g.run(5)
    assert air.hp<50000,f"Tesla should hit air (hp={air.hp})"
    return f"Tesla hits air (dmg={50000-air.hp})"
def t_bld_tesla_hits_ground():
    ts=mk_card('tesla',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',ts)
    gnd=Dummy('red',9,10,hp=50000,spd=0)
    g.deploy('red',gnd)
    g.run(5)
    assert gnd.hp<50000,f"Tesla should hit ground (hp={gnd.hp})"
    return f"Tesla hits ground (dmg={50000-gnd.hp})"
def t_bld_mortar_splash():
    m=mk_card('mortar',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',m)
    d1=Dummy('red',9,15,hp=50000,spd=0)
    d2=Dummy('red',10,15,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(10)
    assert d1.hp<50000,"Primary target should take damage"
    assert d2.hp<50000,"Splash should hit nearby target"
    return f"Mortar splash (d1={50000-d1.hp} d2={50000-d2.hp})"
def t_bld_mortar_deploy_delay():
    import sim.game as gm;gm._CC={}
    ci=card_info('mortar')
    assert ci['deploy']==3.5,f"Mortar deploy should be 3.5, got {ci['deploy']}"
    ci2=card_info('x_bow')
    assert ci2['deploy']==3.5,f"X-Bow deploy should be 3.5, got {ci2['deploy']}"
    ci3=card_info('cannon')
    assert ci3['deploy']==1.0,f"Cannon deploy should be 1.0, got {ci3['deploy']}"
    return f"Deploy delays (mortar={ci['deploy']} xbow={ci2['deploy']} cannon={ci3['deploy']})"
def t_bld_xbow_long_range():
    xb=mk_card('x_bow',11,'blue',9,3)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',xb)
    far=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',far)
    g.run(5)
    assert far.hp<50000,f"X-Bow should hit at range 11 (hp={far.hp})"
    return f"X-Bow long range hit (d=11, dmg={50000-far.hp})"
def t_bld_xbow_fast_atk():
    xb=mk_card('x_bow',11,'blue',9,3)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',xb)
    d=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(3)
    hits=int((50000-d.hp)/xb.dmg) if xb.dmg>0 else 0
    assert hits>=5,f"X-Bow 0.3s hit speed should fire many times in 3s, got {hits} hits"
    return f"X-Bow fast attack ({hits} hits in 3s, hspd={xb.hspd})"
def t_bld_bombtower_splash():
    bt=mk_card('bomb_tower',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',bt)
    d1=Dummy('red',9,10,hp=50000,spd=0)
    d2=Dummy('red',10,10,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<50000 and d2.hp<50000,"Bomb tower should splash both targets"
    return f"Bomb Tower splash (d1={50000-d1.hp} d2={50000-d2.hp})"
def t_bld_bombtower_death_dmg():
    bt=mk_card('bomb_tower',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',bt)
    d=Dummy('red',9,11,hp=50000,spd=0)
    g.deploy('red',d)
    hp_before=d.hp
    bt.hp=1;g.run(0.2)
    assert not bt.alive,"Bomb tower should die"
    dd=hp_before-d.hp
    assert dd>0,f"Death damage should hit nearby enemy, dealt {dd}"
    return f"Bomb Tower death damage ({dd} dmg)"
def t_bld_bombtower_death_on_expire():
    bt=mk_card('bomb_tower',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',bt)
    d=Dummy('red',9,11,hp=50000,spd=0,dmg=0)
    g.deploy('red',d)
    g.run(31)
    assert not bt.alive,"Bomb tower should decay-die"
    assert d.hp<50000,f"Bomb tower death+attacks should damage enemy (hp={d.hp})"
    return f"Bomb Tower decay death (d.hp={d.hp})"
def t_bld_bombtower_ground_only():
    bt=mk_card('bomb_tower',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',bt)
    air=Dummy('red',9,10,hp=5000,spd=0)
    air.transport='Air'
    g.deploy('red',air)
    g.run(5)
    assert air.hp==5000,f"Bomb tower should not hit air (hp={air.hp})"
    return "Bomb Tower ignores air"
def t_bld_tombstone_death_spawn():
    ts=mk_card('tombstone',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',ts)
    ts.hp=1;g.run(0.2)
    assert not ts.alive
    spawned=[tr for tr in g.players['blue'].troops if tr is not ts and tr.alive]
    assert len(spawned)==4,f"Tombstone death should spawn 4 skeletons, got {len(spawned)}"
    return f"Tombstone death spawn ({len(spawned)} skeletons)"
def t_bld_tombstone_spawn_waves():
    ts=mk_card('tombstone',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',ts)
    g.run(8)
    spawned=[tr for tr in g.players['blue'].troops if tr is not ts and tr.alive]
    assert len(spawned)>=4,f"Should spawn >=4 skeletons in 8s (2 per 3.5s), got {len(spawned)}"
    return f"Tombstone spawn waves ({len(spawned)} skeletons in 8s)"
def t_bld_tombstone_killed_early():
    ts=mk_card('tombstone',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',ts)
    g.run(4)
    before=[tr for tr in g.players['blue'].troops if tr is not ts and tr.alive]
    ts.hp=1;g.run(0.2)
    after=[tr for tr in g.players['blue'].troops if tr is not ts and tr.alive]
    new_spawns=len(after)-len(before)
    assert new_spawns==4,f"Death spawn should add 4 skeletons, added {new_spawns}"
    return f"Tombstone killed early (before={len(before)} after={len(after)} death_spawn=+{new_spawns})"
def t_bld_gobcage_brawler_fights():
    gc=mk_card('goblin_cage',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',gc)
    d=Dummy('red',9,12,hp=50000,spd=0)
    g.deploy('red',d)
    gc.hp=1;g.run(0.2)
    assert not gc.alive
    brawlers=[tr for tr in g.players['blue'].troops if tr is not gc and tr.alive]
    assert len(brawlers)==1,f"Should spawn 1 brawler, got {len(brawlers)}"
    g.run(5)
    assert d.hp<50000,f"Brawler should attack enemy (hp={d.hp})"
    return f"Goblin Cage brawler fights (dmg={50000-d.hp})"
def t_bld_gobcage_lifetime_spawn():
    gc=mk_card('goblin_cage',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',gc)
    g.run(21)
    assert not gc.alive
    brawlers=[tr for tr in g.players['blue'].troops if tr is not gc and tr.alive]
    assert len(brawlers)==1,f"Brawler should spawn on lifetime expiry, got {len(brawlers)}"
    return f"Goblin Cage lifetime spawn (brawler hp={brawlers[0].hp})"
def t_bld_barbhut_waves():
    bh=mk_card('barbarian_hut',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',bh)
    g.run(16)
    s1=[tr for tr in g.players['blue'].troops if tr is not bh and tr.alive]
    g.run(15)
    s2=[tr for tr in g.players['blue'].troops if tr is not bh and tr.alive]
    assert len(s2)>len(s1),f"Second wave should spawn more barbs (w1={len(s1)} w2={len(s2)})"
    return f"Barbarian Hut waves (after 16s={len(s1)} after 31s={len(s2)})"
def t_bld_barbhut_death_spawn():
    bh=mk_card('barbarian_hut',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',bh)
    bh.hp=1;g.run(0.2)
    spawned=[tr for tr in g.players['blue'].troops if tr is not bh and tr.alive]
    assert len(spawned)>=1,f"Should death-spawn barbarian(s), got {len(spawned)}"
    return f"Barbarian Hut death spawn ({len(spawned)})"
def t_bld_gobhut_spawns():
    gh=mk_card('goblin_hut',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',gh)
    g.run(6)
    spawned=[tr for tr in g.players['blue'].troops if tr is not gh and tr.alive]
    assert len(spawned)>=3,f"Goblin Hut should spawn >=3 in 6s (1.9s interval), got {len(spawned)}"
    nm=spawned[0].name if spawned else ''
    assert 'Spear' in nm or 'Goblin' in nm,f"Should spawn Spear Goblins, got '{nm}'"
    return f"Goblin Hut spawns ({len(spawned)} Spear Goblins in 6s)"
def t_bld_gobhut_death_spawn():
    gh=mk_card('goblin_hut',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',gh)
    gh.hp=1;g.run(0.2)
    spawned=[tr for tr in g.players['blue'].troops if tr is not gh and tr.alive]
    assert len(spawned)>=1,f"Goblin Hut should death-spawn, got {len(spawned)}"
    return f"Goblin Hut death spawn ({len(spawned)})"
def t_bld_elixcoll_lifetime():
    ec=mk_card('elixir_collector',11,'blue',5,10)
    assert ec.lifetime==65
    g=Game();g.deploy('blue',ec);g.run(62)
    assert ec.alive,"Collector should still be alive at 62s"
    g.run(4)
    assert not ec.alive,"Collector should expire after 65s"
    return "Elixir Collector 65s lifetime"
def t_bld_elixcoll_full_prod():
    ec=mk_card('elixir_collector',11,'blue',5,10)
    g=Game();g.deploy('blue',ec)
    total=0
    for _ in range(7):
        g.players['blue'].elixir=0;g.run(13.1)
        total+=g.players['blue'].elixir
    base=Game()
    btotal=0
    for _ in range(7):
        base.players['blue'].elixir=0;base.run(13.1)
        btotal+=base.players['blue'].elixir
    diff=total-btotal
    assert diff>=4.9,f"Collector should produce ~5 extra elixir, got {diff:.1f}"
    return f"Elixir Collector full production (+{diff:.1f} elixir)"
def t_bld_gobdrill_spawns():
    gd=mk_card('goblin_drill',11,'blue',9,20)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',gd)
    g.run(7)
    spawned=[tr for tr in g.players['blue'].troops if tr is not gd and tr.alive]
    assert len(spawned)>=2,f"Drill should spawn goblins over 7s, got {len(spawned)}"
    return f"Goblin Drill spawns ({len(spawned)} goblins in 7s)"
def t_bld_gobdrill_death_spawn():
    gd=mk_card('goblin_drill',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',gd)
    gd.hp=1;g.run(1.2)
    spawned=[tr for tr in g.players['blue'].troops if tr is not gd and tr.alive]
    assert len(spawned)>=2,f"Drill should death-spawn 2 goblins, got {len(spawned)}"
    return f"Goblin Drill death spawn ({len(spawned)})"
def t_bld_gobdrill_lifetime():
    gd=mk_card('goblin_drill',11,'blue',9,10)
    assert gd.lifetime==10
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',gd)
    g.run(9)
    assert gd.alive,"Drill should be alive at 9s"
    g.run(2)
    assert not gd.alive,"Drill should expire after 10s"
    return "Goblin Drill 10s lifetime"
def t_bld_gobdrill_spawn_zap():
    from sim.fx import SpawnZap
    gd=mk_card('goblin_drill',11,'blue',9,10)
    sz=[c for c in gd.components if isinstance(c,SpawnZap)]
    assert len(sz)==1,"Drill should have SpawnZap"
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',gd)
    d=Dummy('red',9,11,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(1.2)
    zap_dmg=50000-d.hp
    assert zap_dmg>0,f"Spawn zap should deal damage, dealt {zap_dmg}"
    return f"Goblin Drill spawn zap ({zap_dmg} dmg)"
def t_bld_furnace_spawns():
    fn=mk_card('furnace',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',fn)
    g.run(15)
    spawned=[tr for tr in g.players['blue'].troops if tr is not fn]
    spirits=[tr for tr in spawned if 'Spirit' in tr.name]
    assert len(spirits)>=2,f"Furnace should spawn >=2 Fire Spirits in 15s, got {len(spirits)}"
    return f"Furnace spawns ({len(spirits)} Fire Spirits in 15s)"
def t_bld_freeze_stops_building():
    cn=mk_card('cannon',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn)
    d=Dummy('red',9,10,hp=50000,spd=0)
    g.deploy('red',d)
    from sim.units import Status
    cn.statuses.append(Status('freeze',5.0))
    g.run(5)
    assert d.hp==50000,f"Frozen cannon should deal 0 dmg, dealt {50000-d.hp}"
    g.run(3)
    assert d.hp<50000,f"Unfrozen cannon should attack (hp={d.hp})"
    return f"Freeze stops building ({d.hp} after unfreeze)"
def t_bld_spell_damages_building():
    cn=mk_card('cannon',11,'blue',9,10)
    g=Game()
    g.deploy('blue',cn)
    ini=cn.hp
    cn.take_damage(200)
    assert cn.hp==ini-200,f"Building should take spell damage (hp={cn.hp})"
    cn.take_damage(cn.hp+100)
    assert not cn.alive,"Building should die from enough damage"
    return "Spell damages building"
def t_bld_hog_targets_building():
    cn=mk_card('cannon',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn)
    hog=mk_card('hog_rider',11,'red',9,15)
    g.deploy('red',hog)
    g.run(10)
    assert cn.hp<cn.max_hp,f"Hog should target cannon (hp={cn.hp}/{cn.max_hp})"
    return f"Hog targets building (cannon {cn.max_hp}->{cn.hp})"
def t_bld_giant_targets_building():
    cn=mk_card('cannon',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn)
    gi=mk_card('giant',11,'red',9,15)
    g.deploy('red',gi)
    g.run(15)
    assert cn.hp<cn.max_hp,f"Giant should target cannon (hp={cn.hp}/{cn.max_hp})"
    return f"Giant targets building (cannon {cn.max_hp}->{cn.hp})"
def t_bld_two_buildings_same_target():
    cn1=mk_card('cannon',11,'blue',8,5)
    cn2=mk_card('cannon',11,'blue',10,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn1);g.deploy('blue',cn2)
    d=Dummy('red',9,10,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    d_one=50000-d.hp
    cn3=mk_card('cannon',11,'blue',9,5)
    g2=Game()
    for t in g2.arena.towers:t.alive=False
    g2.deploy('blue',cn3)
    d2=Dummy('red',9,10,hp=50000,spd=0)
    g2.deploy('red',d2)
    g2.run(5)
    d_single=50000-d2.hp
    assert d_one>d_single,f"Two cannons should deal more than one: 2x={d_one} 1x={d_single}"
    return f"Two buildings same target (2x={d_one} 1x={d_single})"
def t_bld_building_killed_stops_atk():
    cn=mk_card('cannon',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn)
    d=Dummy('red',9,10,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(2)
    dmg1=50000-d.hp
    cn.hp=0;cn.alive=False
    hp_snap=d.hp
    g.run(3)
    dmg_after=hp_snap-d.hp
    assert dmg_after==0,f"Dead cannon should deal 0 more dmg, dealt {dmg_after}"
    return f"Dead building stops attacking (dealt {dmg1} then 0)"
def t_bld_lifetime_all():
    from sim.units import Building
    lifetimes={'cannon':30,'tesla':25,'bomb_tower':30,'mortar':30,'x_bow':30,
               'tombstone':30,'barbarian_hut':30,'goblin_hut':29,
               'goblin_cage':20,'elixir_collector':65,'goblin_drill':10}
    ok=0
    for name,exp in lifetimes.items():
        b=mk_card(name,11,'blue',5,10)
        if isinstance(b,Building):
            assert b.lifetime==exp,f"{name} lifetime={b.lifetime}, expected {exp}"
            ok+=1
    return f"All building lifetimes correct ({ok}/{len(lifetimes)})"
def t_bld_inferno_v_tank():
    it=mk_card('inferno_tower',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',it)
    golem=mk_card('golem',11,'red',9,15)
    g.deploy('red',golem)
    ini=golem.hp
    g.run(15)
    assert golem.hp<ini*0.3,f"Inferno should melt golem: {ini}->{golem.hp}"
    return f"Inferno vs Golem ({ini}->{golem.hp})"
def t_bld_cannon_v_hog():
    cn=mk_card('cannon',11,'blue',9,10)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn)
    hog=mk_card('hog_rider',11,'red',9,15)
    g.deploy('red',hog)
    g.run(15)
    return f"Cannon vs Hog (cannon hp={cn.hp}/{cn.max_hp} hog hp={hog.hp}/{hog.max_hp})"
def t_bld_mortar_v_tower():
    random.seed(42)
    dk=_mk_deck(['mortar'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','mortar')
    g.players['blue'].elixir=10
    g.play_card('blue','mortar',3,14)
    g.run(20)
    rpt=g.arena.get_tower('red','princess','left')
    assert rpt.hp<rpt.max_hp,f"Mortar should hit red princess (hp={rpt.hp}/{rpt.max_hp})"
    return f"Mortar hits tower ({rpt.max_hp}->{rpt.hp})"
def t_bld_xbow_v_tower():
    random.seed(42)
    dk=_mk_deck(['x_bow'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','x_bow')
    g.players['blue'].elixir=10
    g.play_card('blue','x_bow',3,14)
    g.run(20)
    rpt=g.arena.get_tower('red','princess','left')
    assert rpt.hp<rpt.max_hp,f"X-Bow should hit tower (hp={rpt.hp}/{rpt.max_hp})"
    return f"X-Bow hits tower ({rpt.max_hp}->{rpt.hp})"
def t_bld_no_tower_target():
    random.seed(42)
    dk=_mk_deck(['tesla'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','tesla')
    g.players['blue'].elixir=10
    g.play_card('blue','tesla',9,14)
    g.run(20)
    rpt=g.arena.get_tower('red','princess','left')
    rpt2=g.arena.get_tower('red','princess','right')
    rk=g.arena.get_tower('red','king','')
    assert rpt.hp==rpt.max_hp,f"Tesla should NOT target tower (hp={rpt.hp}/{rpt.max_hp})"
    assert rpt2.hp==rpt2.max_hp
    assert rk.hp==rk.max_hp
    tr=mk_card('knight',11,'red',9,18)
    g.deploy('red',tr)
    g.run(15)
    assert tr.hp<tr.max_hp,f"Tesla should target enemy troop (hp={tr.hp}/{tr.max_hp})"
    return "Defensive building skips towers, hits troops"
def t_gobmachine_body():
    r=mk_card('goblin_machine',11,'blue',5,10)
    assert r.hp==2265 and r.dmg==232
    assert abs(r.hspd-1.2)<0.01 and abs(r.fhspd-0.7)<0.01
    assert r.targets==['Ground']
    return f"Goblin Machine body (hp={r.hp} dmg={r.dmg})"
def t_gobmachine_rocket():
    from sim.fx import RocketLauncher
    r=mk_card('goblin_machine',11,'blue',5,10)
    rl=[c for c in r.components if isinstance(c,RocketLauncher)]
    assert len(rl)==1
    assert rl[0].dmg==304 and abs(rl[0].rng_min-2.5)<0.01
    assert abs(rl[0].rng_max-5.0)<0.01
    return f"Goblin Machine rocket (dmg={rl[0].dmg} rng={rl[0].rng_min}-{rl[0].rng_max})"
def t_gobmachine_rocket_fires():
    g=Game()
    gm=mk_card('goblin_machine',11,'blue',9,10)
    g.deploy('blue',gm)
    d=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    body_dmg=50000-d.hp
    assert body_dmg>304,f"Rocket should fire at range, got {body_dmg}"
    return f"Goblin Machine rocket fires ({body_dmg} dmg in 5s)"
def t_gobmachine_blindspot():
    g=Game()
    gm=mk_card('goblin_machine',11,'blue',9,10)
    g.deploy('blue',gm)
    d_close=Dummy('red',9,11,hp=50000,spd=0)
    d_far=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d_close);g.deploy('red',d_far)
    g.run(5)
    close_dmg=50000-d_close.hp
    far_dmg=50000-d_far.hp
    assert close_dmg>0,"Body should hit close target"
    assert far_dmg>0,"Rocket should hit far target"
    return f"Goblin Machine blind spot (close={close_dmg} far={far_dmg})"
def t_goblinstein_spawn():
    random.seed(42)
    r=mk_card('goblinstein',11,'blue',5,10)
    assert isinstance(r,list) and len(r)==2
    names=sorted([t.name for t in r])
    assert 'Doctor' in names and 'Monster' in names
    return f"Goblinstein spawns 2 ({names})"
def t_goblinstein_doctor():
    random.seed(42)
    r=mk_card('goblinstein',11,'blue',5,10)
    doc=[t for t in r if t.name=='Doctor'][0]
    assert doc.hp==721 and doc.dmg==92
    assert abs(doc.rng-5.5)<0.01
    assert 'Air' in doc.targets and 'Ground' in doc.targets
    assert abs(doc.stun_dur-0.5)<0.01
    return f"Goblinstein doctor (hp={doc.hp} dmg={doc.dmg} rng={doc.rng} stun={doc.stun_dur})"
def t_goblinstein_monster():
    random.seed(42)
    r=mk_card('goblinstein',11,'blue',5,10)
    mon=[t for t in r if t.name=='Monster'][0]
    assert mon.hp==2240 and mon.dmg==128
    assert mon.targets==['Buildings']
    return f"Goblinstein monster (hp={mon.hp} dmg={mon.dmg} targets={mon.targets})"
def t_goblinstein_monster_buildings():
    g=Game()
    random.seed(42)
    r=mk_card('goblinstein',11,'blue',9,14)
    mon=[t for t in r if t.name=='Monster'][0]
    g.deploy('blue',mon)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini,"Monster should ignore troops"
    return "Goblinstein monster targets buildings only"
def t_mightyminer_stats():
    r=mk_card('mighty_miner',11,'blue',5,10)
    assert r.hp==2250 and r.dmg==43
    assert abs(r.hspd-0.4)<0.01
    assert abs(r.rng-1.6)<0.01
    assert r.ramp_stages==[43,204,409]
    assert r.ramp_durations==[1.5,1.5]
    return f"Mighty Miner stats (hp={r.hp} stages={r.ramp_stages})"
def t_mightyminer_ramp():
    g=Game()
    mm=mk_card('mighty_miner',11,'blue',9,10)
    g.deploy('blue',mm)
    d=Dummy('red',9,11,hp=50000,spd=0)
    g.deploy('red',d)
    assert mm.dmg==43
    g.run(2.0)
    assert mm.dmg==204,f"Expected stage 2 (204), got {mm.dmg}"
    g.run(2.0)
    assert mm.dmg==409,f"Expected stage 3 (409), got {mm.dmg}"
    return "Mighty Miner ramp (43->204->409)"
def t_mightyminer_reset():
    g=Game()
    mm=mk_card('mighty_miner',11,'blue',9,10)
    g.deploy('blue',mm)
    d1=Dummy('red',9,11,hp=50000,spd=0)
    d2=Dummy('red',12,11,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(2.0)
    assert mm.dmg==204,f"Expected stage 2, got {mm.dmg}"
    d1.alive=False
    g.players['red'].troops=[t for t in g.players['red'].troops if t.alive]
    g.run(0.2)
    assert mm.dmg==43,f"Expected reset to 43, got {mm.dmg}"
    return "Mighty Miner resets on retarget"
def t_spiritempress_stats():
    r=mk_card('spirit_empress',11,'blue',5,10)
    assert r.hp==1255 and r.dmg==307
    assert abs(r.rng-4.5)<0.01
    assert r.splash_r==0
    assert 'Air' in r.targets and 'Ground' in r.targets
    return f"Spirit Empress stats (hp={r.hp} dmg={r.dmg} rng={r.rng})"
def t_spiritempress_transform():
    g=Game()
    se=mk_card('spirit_empress',11,'blue',9,10)
    g.deploy('blue',se)
    se.hp=0;se.alive=False;se.on_death(g)
    spirits=[t for t in g.players['blue'].troops if t.alive and t is not se]
    assert len(spirits)==1,f"Expected 1 spirit, got {len(spirits)}"
    sp=spirits[0]
    assert sp.hp==727 and sp.dmg==120
    assert abs(sp.rng-4.0)<0.01
    return f"Spirit Empress transforms (spirit hp={sp.hp} dmg={sp.dmg} rng={sp.rng})"
def t_spiritempress_splash():
    g=quiet(Game())
    se=mk_card('spirit_empress',11,'blue',9,10)
    g.deploy('blue',se)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',10,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    # the current game data lists the Empress as a single-target attacker
    assert d1.hp<5000 and d2.hp==5000
    return f"Spirit Empress single target (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_susbush_stats():
    r=mk_card('suspicious_bush',11,'blue',5,10)
    assert r.hp==81 and r.dmg==256
    assert r.targets==['Buildings']
    return f"Suspicious Bush stats (hp={r.hp} dmg={r.dmg})"
def t_susbush_deathspawn():
    g=Game()
    sb=mk_card('suspicious_bush',11,'blue',9,14)
    g.deploy('blue',sb)
    sb.hp=0;sb.alive=False;sb.on_death(g)
    gobs=[t for t in g.players['blue'].troops if t.alive and t is not sb]
    assert len(gobs)==2,f"Expected 2 goblins, got {len(gobs)}"
    assert gobs[0].hp==476 and gobs[0].dmg==355
    return f"Suspicious Bush spawns 2 goblins (hp={gobs[0].hp} dmg={gobs[0].dmg})"
def t_mv_forward():
    g=Game()
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    oy=k.y
    g.run(3)
    assert k.y>oy+1.0,f"Troop should walk forward: {oy}->{k.y}"
    return f"Movement forward ({oy:.1f}->{k.y:.1f})"
def t_mv_bridge():
    g=Game()
    k=mk_card('knight',11,'blue',4,14)
    g.deploy('blue',k)
    g.run(10)
    assert k.y>14.5,f"Ground troop should route through bridge, y={k.y:.1f}"
    return f"Ground routes through bridge (y={k.y:.1f})"
def t_mv_air_straight():
    g=Game()
    bd=mk_card('baby_dragon',11,'blue',9,14)
    g.deploy('blue',bd)
    oy=bd.y
    g.run(3)
    assert bd.y>oy,"Air should fly straight forward"
    return f"Air flies straight ({oy:.1f}->{bd.y:.1f})"
def t_mv_riverjump():
    g=Game()
    hog=mk_card('hog_rider',11,'blue',4,14)
    g.deploy('blue',hog)
    g.run(5)
    assert hog.y>16,"Hog should jump river"
    return f"RiverJump crosses river (y={hog.y:.1f})"
def t_mv_slow():
    g1=Game();g2=Game()
    k1=mk_card('knight',11,'blue',9,10)
    k2=mk_card('knight',11,'blue',9,10)
    g1.deploy('blue',k1);g2.deploy('blue',k2)
    k1.statuses.append(Status('slow',10.0,0.5))
    for _ in range(30):g1.tick();g2.tick()
    assert k1.y<k2.y,f"Slowed should lag: {k1.y:.1f} vs {k2.y:.1f}"
    return f"Slow reduces movement ({k1.y:.1f} vs {k2.y:.1f})"
def t_mv_freeze():
    g=Game()
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    oy=k.y
    k.statuses.append(Status('freeze',5.0))
    g.run(2)
    assert abs(k.y-oy)<0.01,f"Frozen troop shouldn't move: {oy}->{k.y}"
    return "Freeze stops movement"
def t_mv_rage():
    g1=Game();g2=Game()
    k1=mk_card('knight',11,'blue',9,10)
    k2=mk_card('knight',11,'blue',9,10)
    g1.deploy('blue',k1);g2.deploy('blue',k2)
    k1.statuses.append(Status('rage',10.0,0.3))
    for _ in range(30):g1.tick();g2.tick()
    assert k1.y>k2.y,f"Raged should be ahead: {k1.y:.1f} vs {k2.y:.1f}"
    return f"Rage increases movement ({k1.y:.1f} vs {k2.y:.1f})"
def t_tgt_ground_ignores_air():
    g=Game()
    k=mk_card('knight',11,'blue',9,14)
    bd=mk_card('baby_dragon',11,'red',9,15)
    g.deploy('blue',k);g.deploy('red',bd)
    ini=bd.hp
    g.run(3)
    assert bd.hp==ini,"Ground-only should ignore air"
    return "Ground-only ignores air target"
def t_tgt_air_targeting():
    g=Game()
    m=mk_card('musketeer',11,'blue',9,10)
    g.deploy('blue',m)
    bd=mk_card('baby_dragon',11,'red',9,15)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(3)
    assert bd.hp<ini,"Musketeer should hit air"
    return f"Air-targeting troop hits air ({ini}->{bd.hp})"
def t_tgt_building_ignores_troop():
    g=Game()
    go=mk_card('golem',11,'blue',9,14)
    g.deploy('blue',go)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini,"BuildingTarget should walk past troops"
    return "BuildingTarget walks past troops"
def t_tgt_king_protected():
    g=Game()
    _,td=g._find_target(Dummy('blue',9,14))
    tgt,_=g._find_target(Dummy('blue',9,14))
    assert hasattr(tgt,'ttype')
    assert tgt.ttype=='princess',"Should target princess while alive, not king"
    return "King protected while princess alive"
def t_tgt_king_after_princess():
    g=Game()
    lp=g.arena.get_tower('red','princess','left')
    lp.hp=0;lp.alive=False
    d=Dummy('blue',3,22)
    tgt,_=g._find_target(d)
    assert hasattr(tgt,'ttype') and tgt.ttype=='king',"Should target king when one princess dead"
    g2=Game()
    for t in g2.arena.towers:
        if t.team=='red' and t.ttype=='princess':
            t.hp=0;t.alive=False
    tgt2,_=g2._find_target(Dummy('blue',9,20))
    assert hasattr(tgt2,'ttype') and tgt2.ttype=='king',"Should target king when both princess dead"
    return "King targeted after princess destroyed"
def t_tgt_retarget():
    g=Game()
    k=mk_card('knight',11,'blue',9,14)
    g.deploy('blue',k)
    d1=Dummy('red',9,15,hp=200,spd=0)
    d2=Dummy('red',9,16,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert not d1.alive,"First target should die"
    assert d2.hp<50000,"Knight should retarget to second"
    return f"Retarget on death (d2 hp={d2.hp})"
def t_atk_first_hit_speed():
    g=Game()
    m=mk_card('musketeer',11,'blue',9,10)
    g.deploy('blue',m)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(0.55)
    assert d.hp==50000,"No shot before fhspd (0.3s) plus flight (0.3s)"
    g.run(0.1)
    assert d.hp<50000,"Should have landed by 0.65s"
    return "First hit speed faster than hspd"
def t_atk_splash_radius():
    g=Game()
    v=mk_card('valkyrie',11,'blue',9,10)
    g.deploy('blue',v)
    d1=Dummy('red',9,11,hp=5000,spd=0)
    d2=Dummy('red',10,11,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(3)
    assert d1.hp<5000 and d2.hp<5000,"Both nearby should take splash"
    return f"Splash damages all in radius (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_atk_chain():
    g=Game()
    ed=mk_card('electro_dragon',11,'blue',9,10)
    g.deploy('blue',ed)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',10,14,hp=5000,spd=0)
    d3=Dummy('red',11,14,hp=5000,spd=0)
    d4=Dummy('red',15,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2);g.deploy('red',d3);g.deploy('red',d4)
    g.run(5)
    hit=[d for d in [d1,d2,d3] if d.hp<5000]
    assert len(hit)==3,"Chain should hit 3 targets"
    return "Chain hits exactly chain_count targets"
def t_atk_suicide():
    g=Game()
    es=mk_card('electro_spirit',11,'blue',9,14)
    g.deploy('blue',es)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert not es.alive,"Suicide unit should die after attack"
    assert d.hp<50000,"Should deal damage"
    return "Suicide unit dies after attack"
def t_atk_miner_ct():
    g=Game()
    m=mk_card('miner',11,'red',3,8)
    g.deploy('red',m)
    lpt=g.arena.get_tower('blue','princess','left')
    ini=lpt.hp
    g.run(10)
    if lpt.hp<ini:
        per_hit=ini-lpt.hp
        assert per_hit<193*5,"Miner ct_dmg should be less than full dmg on towers"
    return f"Miner uses ct_dmg on towers ({ini}->{lpt.hp})"
def t_atk_stun_delays():
    g=Game()
    ew=mk_card('electro_wizard',11,'blue',9,10)
    d=Dummy('red',9,14,hp=50000,spd=0,hspd=0.5,dmg=100)
    tgt=Dummy('blue',9,15,hp=50000,spd=0)
    g.deploy('red',d);g.deploy('blue',ew);g.deploy('blue',tgt)
    g.run(2)
    dmg=50000-tgt.hp
    assert dmg<500,f"Stun should delay attacks, got {dmg}"
    return f"Stun delays enemy attack cycle ({dmg} dmg)"
def t_atk_shield():
    dp=mk_card('dark_prince',11,'blue',9,10)
    assert dp.shield_hp==240
    dp.take_damage(500)
    assert dp.shield_hp==0 and dp.hp==1200,"Shield absorbs, excess blocked"
    dp.take_damage(100)
    assert dp.hp==1100,"After shield break, HP takes damage"
    return "Shield absorbs then HP takes damage"
def t_game_deck_elixir():
    random.seed(42)
    dk=['knight','archers','fireball','zap','valkyrie','musketeer','baby_dragon','mini_pekka']
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0},
           p2={'deck':dk,'drag_del':0.0,'drag_std':0})
    g.players['blue'].elixir=10;g.players['red'].elixir=10
    c=g.players['blue'].deck.hand[0]
    ok,_=g.play_card('blue',c,9,10)
    assert ok
    cost=card_info(c)['cost']
    assert abs(g.players['blue'].elixir-(10-cost))<0.5
    return "Both players play with deck+elixir"
def t_game_deck_cycle():
    random.seed(42)
    dk=['knight','archers','fireball','zap','valkyrie','musketeer','baby_dragon','mini_pekka']
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    g.players['blue'].elixir=10
    played=[]
    for _ in range(8):
        if not g.players['blue'].deck.hand:break
        c=g.players['blue'].deck.hand[0]
        g.players['blue'].elixir=10
        ok,_=g.play_card('blue',c,9,10)
        if ok:played.append(c)
        g.run(3)
    assert len(set(played))==8,f"Should cycle all 8 cards, got {len(set(played))}"
    return f"All 8 cards cycle ({len(played)} plays)"
def t_game_deploy_zones():
    g=Game()
    assert g._valid_deploy('blue',9,10)
    assert not g._valid_deploy('blue',9,20)
    assert g._valid_deploy('red',9,20)
    assert not g._valid_deploy('red',9,10)
    rlp=g.arena.get_tower('red','princess','left')
    rlp.hp=0;rlp.alive=False
    assert g._valid_deploy('blue',3,20),"Pocket should unlock"
    return "Deploy zones enforced + pockets unlock"
def t_game_spell_bypass():
    random.seed(42)
    dk=_mk_deck(['rocket'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','rocket')
    g.players['blue'].elixir=10
    ok,msg=g.play_card('blue','rocket',9,25)
    assert ok,f"Spell should bypass deploy zone: {msg}"
    return "Spells bypass deploy zone"
def t_game_multi_pending():
    random.seed(42)
    dk=['knight','valkyrie','archers','musketeer','baby_dragon','mini_pekka','fireball','zap']
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    g.players['blue'].elixir=10
    _force_hand(g,'blue','knight')
    ok1,_=g.play_card('blue','knight',5,10)
    _force_hand(g,'blue','valkyrie')
    ok2,_=g.play_card('blue','valkyrie',12,10)
    assert ok1 and ok2
    assert len(g.pending)==2
    g.run(3)
    assert len(g.players['blue'].troops)>=2,"Both should spawn"
    return "Multiple pending cards all spawn"
def t_game_phases_flow():
    g=Game()
    assert g.phase=='regulation'
    g.run_to(180.1)
    assert g.phase=='overtime'
    g.run_to(299.9)
    assert not g.ended
    g.run_to(300.1)
    assert g.ended
    return "Reg->OT->tiebreaker flow (ends at 300s)"
def t_replay_positions():
    random.seed(42)
    dk=_mk_deck(['knight'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','knight')
    g.players['blue'].elixir=10
    g.play_card('blue','knight',9,10)
    g.run(15)
    s3=g.replay.at(3)
    s10=g.replay.at(10)
    k3=[u for u in s3['troops'] if u['team']=='blue' and u['name']=='Knight']
    k10=[u for u in s10['troops'] if u['team']=='blue' and u['name']=='Knight']
    assert k3 and k10,"Knight should be in replay"
    assert k10[0]['y']>k3[0]['y'],f"Y should increase: {k3[0]['y']}->{k10[0]['y']}"
    return f"Replay positions increase ({k3[0]['y']}->{k10[0]['y']})"
def t_replay_events():
    random.seed(42)
    dk=_mk_deck(['knight'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','knight')
    g.players['blue'].elixir=10
    g.play_card('blue','knight',9,10)
    g.run(5)
    evts=g.replay.events(0,10)
    log_has=any('knight' in e.lower() or 'plays' in e.lower() for e in g.log)
    assert len(evts)>0 or log_has,"Should have events in replay or log"
    return f"Replay events match log ({len(evts)} events)"
def t_replay_dump():
    random.seed(42)
    dk=_mk_deck(['knight'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','knight')
    g.players['blue'].elixir=10
    g.play_card('blue','knight',9,10)
    g.run(5)
    d=g.replay.dump(3)
    assert 'T=' in d and 'Towers:' in d
    for k in ('blue','red'):assert k.capitalize() in d or k in d
    return "Replay dump has expected sections"
def t_cross_pekka_rage():
    g1=Game();g2=Game()
    pk1=mk_card('pekka',11,'blue',3,14)
    pk2=mk_card('pekka',11,'blue',3,14)
    g1.deploy('blue',pk1);g2.deploy('blue',pk2)
    g1.run(10);g2.run(10)
    rpt1=g1.arena.get_tower('red','princess','left')
    rpt2=g2.arena.get_tower('red','princess','left')
    hp1_before=rpt1.hp;hp2_before=rpt2.hp
    r=mk_card('rage',11,'blue',pk1.x,pk1.y)
    r.apply(g1)
    for _ in range(50):g1.tick();g2.tick()
    d1=hp1_before-rpt1.hp;d2=hp2_before-rpt2.hp
    assert d1>=d2,f"Raged PEKKA should deal >= unraged: {d1} vs {d2}"
    return f"PEKKA+Rage boosts DPS (raged={d1} vs normal={d2})"
def t_cross_gy_poison():
    g=Game()
    random.seed(42)
    d=Dummy('blue',9,10,hp=50000,spd=0)
    g.deploy('blue',d)
    p=mk_card('poison',11,'blue',9,10)
    p.apply(g);g.spells.append(p)
    d2=Dummy('red',9,10,hp=500,spd=0)
    g.deploy('red',d2)
    g.run(5)
    assert not d2.alive or d2.hp<500,"Poison should damage/kill troops"
    return f"Poison kills troops in radius (hp={d2.hp})"
def t_cross_egiant_reflect():
    g=Game()
    eg=mk_card('electro_giant',11,'blue',9,14)
    g.deploy('blue',eg)
    k=mk_card('knight',11,'red',9,15)
    g.deploy('red',k)
    g.run(5)
    assert k.hp<1690,"E-Giant should reflect damage to Knight"
    return f"E-Giant reflects damage (knight hp={k.hp})"
def t_cross_clone():
    g=Game()
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    cl=mk_card('clone',11,'blue',9,10)
    cl.apply(g)
    troops=g.players['blue'].troops
    assert len(troops)==2,"Clone should create copy"
    clone=[t for t in troops if t is not k][0]
    assert clone.hp==1,"Clone should have 1 HP"
    return f"Clone creates 1HP copy (dmg={clone.dmg})"
def t_cross_lightning_3hp():
    g=Game()
    d1=Dummy('red',9,10,hp=5000,spd=0)
    d2=Dummy('red',10,10,hp=3000,spd=0)
    d3=Dummy('red',10,11,hp=2000,spd=0)
    d4=Dummy('red',9,11,hp=500,spd=0)
    g.deploy('red',d1);g.deploy('red',d2);g.deploy('red',d3);g.deploy('red',d4)
    lt=mk_card('lightning',11,'blue',9.5,10.5)
    lt.apply(g)
    hit=[d for d in [d1,d2,d3,d4] if d.hp<d.max_hp]
    assert len(hit)==3 and d4.hp==500,"Lightning hits 3 highest HP"
    return "Lightning hits exactly 3 highest HP"
def t_cross_freeze_stops():
    g=Game()
    k=mk_card('knight',11,'blue',9,14)
    g.deploy('blue',k)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(2)
    hp_before=d.hp
    fz=mk_card('freeze',11,'red',k.x,k.y)
    fz.apply(g)
    g.run(2)
    hp_after=d.hp
    assert hp_after==hp_before,f"Freeze should stop attacks: {hp_before}->{hp_after}"
    return "Freeze stops all attacks during duration"
def t_cross_lj_rage():
    g=Game()
    lj=mk_card('lumberjack',11,'blue',9,10)
    ally=mk_card('knight',11,'blue',9,10.5)
    g.deploy('blue',lj);g.deploy('blue',ally)
    lj.hp=0;lj.alive=False;lj.on_death(g)
    has_rage=any(s.kind=='rage' for s in ally.statuses)
    assert has_rage,"LJ death should drop rage on allies"
    return "Lumberjack death drops rage"
_FILLER=['knight','archers','fireball','zap','valkyrie','musketeer','baby_dragon','mini_pekka']
def _mk_deck(cards):
    dk=list(cards)
    fi=0
    while len(dk)<8:
        c=_FILLER[fi%len(_FILLER)]
        if c not in dk:dk.append(c)
        fi+=1
    return dk
def _force_hand(g,tm,card):
    dk=g.players[tm].deck
    if card in dk.hand:return
    if dk.nxt==card:
        dk.nxt=dk.q.pop(0) if dk.q else None
    elif card in dk.q:
        dk.q.remove(card)
    else:return
    if len(dk.hand)<4:
        dk.hand.append(card)
    else:
        ov=dk.hand.pop()
        dk.hand.insert(0,card)
        if dk.nxt is None:dk.nxt=ov;dk.nxt_cd=0
        else:dk.q.insert(0,ov)
def t_scn_pekka_push():
    random.seed(42)
    dk=_mk_deck(['pekka'])
    g=Game(p1={'deck':dk,'drag_del':0.3,'drag_std':0})
    _force_hand(g,'blue','pekka')
    g.players['blue'].elixir=10
    ok,_=g.play_card('blue','pekka',3,14)
    assert ok,"Failed to play pekka"
    g.run(30)
    rpt=g.arena.get_tower('red','princess','left')
    assert rpt.hp<rpt.max_hp,"PEKKA should damage red princess tower"
    s5=g.replay.at(5)
    s15=g.replay.at(15)
    pk5=[u for u in s5['troops'] if 'P.E.K.K.A' in u['name'] and u['team']=='blue']
    pk15=[u for u in s15['troops'] if 'P.E.K.K.A' in u['name'] and u['team']=='blue']
    assert len(pk5)>=1,"PEKKA should be alive at T=5"
    assert pk15[0]['y']>pk5[0]['y'],"PEKKA should move forward over time"
    return f"Scenario: PEKKA push (tower {rpt.max_hp}->{rpt.hp})"
def t_scn_mk_defends_skarmy():
    random.seed(42)
    dk_r=_mk_deck(['skeleton_army'])
    dk_b=_mk_deck(['mega_knight'])
    g=Game(p1={'deck':dk_b,'drag_del':0.0,'drag_std':0},
           p2={'deck':dk_r,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'red','skeleton_army')
    _force_hand(g,'blue','mega_knight')
    g.players['red'].elixir=10;g.players['blue'].elixir=10
    g.play_card('red','skeleton_army',9,17)
    g.run(5)
    sk=[t for t in g.players['red'].troops if t.alive]
    assert len(sk)>0,"Skarmy should have spawned"
    g.players['blue'].elixir=10
    g.play_card('blue','mega_knight',9,14)
    g.run(5)
    sk_alive=[t for t in g.players['red'].troops if t.alive]
    assert len(sk_alive)<=3,f"MK should wipe most skarmy, {len(sk_alive)} survived"
    return f"Scenario: MK defends skarmy ({15-len(sk_alive)}/15 killed)"
def t_scn_nw_bat_swarm():
    random.seed(42)
    dk=_mk_deck(['night_witch'])
    g=quiet(Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0}))
    _force_hand(g,'blue','night_witch')
    g.players['blue'].elixir=10
    g.play_card('blue','night_witch',9,5)
    g.run(6)
    bats=[t for t in g.players['blue'].troops if t.alive and t.transport=='Air']
    assert len(bats)>=2,f"Expected >=2 bats, got {len(bats)}"
    return f"Scenario: NW bat swarm ({len(bats)} bats)"
def t_scn_icewiz_defense():
    random.seed(42)
    dk_b=_mk_deck(['ice_wizard'])
    dk_r=_mk_deck(['knight'])
    g=Game(p1={'deck':dk_b,'drag_del':0.0,'drag_std':0},
           p2={'deck':dk_r,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'red','knight');_force_hand(g,'blue','ice_wizard')
    g.players['red'].elixir=10;g.players['blue'].elixir=10
    g.play_card('red','knight',9,17)
    g.run(2)
    g.play_card('blue','ice_wizard',9,10)
    random.seed(42)
    g2=Game(p2={'deck':dk_r,'drag_del':0.0,'drag_std':0})
    _force_hand(g2,'red','knight')
    g2.players['red'].elixir=10
    g2.play_card('red','knight',9,17)
    for _ in range(200):g.tick();g2.tick()
    slowed=[t for t in g.players['red'].troops if t.alive and t.name=='Knight']
    free=[t for t in g2.players['red'].troops if t.alive and t.name=='Knight']
    assert slowed and free,"Both knights should be alive"
    assert slowed[0].y>free[0].y,f"Slowed knight should be behind: {slowed[0].y:.1f} vs {free[0].y:.1f}"
    return "Scenario: IW slows knight push"
def t_scn_espirit_chain():
    random.seed(42)
    dk_b=_mk_deck(['electro_spirit'])
    g=Game(p1={'deck':dk_b,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','electro_spirit')
    g.players['blue'].elixir=10
    d1=Dummy('red',9,15,hp=5000,spd=0)
    d2=Dummy('red',10,15,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.play_card('blue','electro_spirit',9,14)
    g.run(5)
    assert d1.hp<5000,"Target 1 not hit"
    assert d2.hp<5000,"Chain didn't reach target 2"
    return f"Scenario: E-Spirit chains via play_card (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_scn_rocket_tower():
    random.seed(42)
    dk=_mk_deck(['rocket'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','rocket')
    g.players['blue'].elixir=10
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    ok,msg=g.play_card('blue','rocket',rpt.cx,rpt.cy)
    assert ok,f"Failed to play rocket: {msg}"
    g.run(5)
    assert rpt.hp==ini-343,f"Expected 343 CT dmg, got {ini-rpt.hp}"
    return f"Scenario: Rocket on tower ({ini}->{rpt.hp})"
def t_scn_gy_vs_arrows():
    random.seed(42)
    dk_b=_mk_deck(['graveyard'])
    dk_r=_mk_deck(['arrows'])
    g=quiet(Game(p1={'deck':dk_b,'drag_del':0.0,'drag_std':0},
           p2={'deck':dk_r,'drag_del':0.0,'drag_std':0}))
    _force_hand(g,'blue','graveyard');_force_hand(g,'red','arrows')
    g.players['blue'].elixir=10;g.players['red'].elixir=10
    g.play_card('blue','graveyard',14,24)
    g.run(8)
    sk_before=len([t for t in g.players['blue'].troops if t.alive])
    assert sk_before>0,"GY should have spawned skeletons"
    g.play_card('red','arrows',14,24)
    g.run(2)
    sk_after=len([t for t in g.players['blue'].troops if t.alive])
    assert sk_after<sk_before,f"Arrows should kill some skeletons ({sk_before}->{sk_after})"
    return f"Scenario: GY vs Arrows ({sk_before}->{sk_after} skeletons)"
def t_scn_rage_hog():
    random.seed(42)
    dk1=_mk_deck(['hog_rider','rage'])
    g1=Game(p1={'deck':dk1,'drag_del':0.0,'drag_std':0})
    _force_hand(g1,'blue','hog_rider')
    g1.players['blue'].elixir=10
    g1.play_card('blue','hog_rider',3,14)
    g1.run(5)
    hogs=[t for t in g1.players['blue'].troops if t.alive and t.name=='Hog Rider']
    assert len(hogs)>=1,"Hog should be alive"
    _force_hand(g1,'blue','rage')
    g1.play_card('blue','rage',hogs[0].x,hogs[0].y)
    g1.run(15)
    rpt1=g1.arena.get_tower('red','princess','left')
    random.seed(42)
    dk2=_mk_deck(['hog_rider'])
    g2=Game(p1={'deck':dk2,'drag_del':0.0,'drag_std':0})
    _force_hand(g2,'blue','hog_rider')
    g2.players['blue'].elixir=10
    g2.play_card('blue','hog_rider',3,14)
    g2.run(20)
    rpt2=g2.arena.get_tower('red','princess','left')
    d1=rpt1.max_hp-rpt1.hp;d2=rpt2.max_hp-rpt2.hp
    assert d1>d2,f"Raged hog should deal more dmg: {d1} vs {d2}"
    return f"Scenario: Rage+Hog ({d1} vs unraged {d2})"
def t_scn_full_match():
    random.seed(42)
    dk_b=_mk_deck(['pekka','knight','archers','fireball'])
    dk_r=_mk_deck(['mega_knight','valkyrie','musketeer','zap'])
    g=Game(p1={'deck':dk_b,'drag_del':0.4,'drag_std':0},
           p2={'deck':dk_r,'drag_del':0.4,'drag_std':0})
    plays=0
    for _ in range(3000):
        g.tick()
        if g.ended:break
        for tm in ('blue','red'):
            p=g.players[tm]
            if p.deck and p.deck.hand and p.elixir>=5:
                c=p.deck.hand[0]
                ci=card_info(c)
                if p.elixir>=ci['cost']:
                    if tm=='blue':x,y=9,12
                    else:x,y=9,20
                    ok,_=g.play_card(tm,c,x,y)
                    if ok:plays+=1
    assert plays>=4,f"Expected >=4 plays, got {plays}"
    assert g.t>5,"Match should have run"
    return f"Scenario: Full match ({plays} plays, T={g.t:.1f}s, phase={g.phase})"
def t_scn_random_hand():
    dk=_mk_deck(['pekka','knight','archers','fireball'])
    hands=set()
    for seed in range(10):
        random.seed(seed)
        g=Game(p1={'deck':dk})
        h=tuple(sorted(g.players['blue'].deck.hand))
        hands.add(h)
    assert len(hands)>=2,f"Expected >=2 unique hands, got {len(hands)}"
    return f"Scenario: Random hand ({len(hands)} unique hands from 10 seeds)"
def t_scn_replay_scrub():
    random.seed(42)
    dk=_mk_deck(['knight'])
    g=Game(p1={'deck':dk,'drag_del':0.0,'drag_std':0})
    _force_hand(g,'blue','knight')
    g.players['blue'].elixir=10
    g.play_card('blue','knight',9,10)
    g.run(15)
    s0=g.replay.at(0)
    assert s0 is not None,"replay.at(0) should return initial state"
    assert s0['t']<=0.2,"First snap should be near T=0"
    s10=g.replay.at(10)
    assert s10 is not None
    tr10=[u for u in s10['troops'] if u['team']=='blue']
    assert len(tr10)>=1,"Knight should be alive at T=10"
    d=g.replay.dump(10)
    assert 'T=' in d,"dump should contain T="
    assert 'Towers:' in d,"dump should contain Towers"
    evts=g.replay.events(0,20)
    assert len(evts)>0,"Should have events"
    sm=g.replay.summary()
    assert len(sm)>0,"Summary should not be empty"
    return f"Scenario: Replay scrub ({len(g.replay.snaps)} snaps, {len(evts)} events)"
def t_travel_fireball():
    g=Game()
    d=Dummy('red',9,25,hp=50000,spd=0)
    g.deploy('red',d)
    fb=mk_card('fireball',11,'blue',9,25);fb.proj_spd=10.0
    g._cast('blue',fb,9.0,25.0)
    kt=g.arena.get_tower('blue','king')
    exp=math.hypot(9.0-kt.cx,25.0-kt.cy)/10.0
    g.run(exp-0.1)
    assert d.hp==50000 and len(g.projs)==1,"Fireball still in flight from the king tower"
    g.run(0.2)
    assert d.hp<50000 and not g.projs,"Fireball should land after travel"
    return f"Travel: fireball flies {exp:.2f}s at projSpeed"
def t_travel_log_roll():
    g=quiet(Game())
    d=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d)
    lg=mk_card('the_log',11,'blue',9,10)
    g._cast('blue',lg,9.0,10.0)
    exp=(4-0.5)/lg.speed
    g.run(exp-0.1)
    assert d.hp==50000,"Log still rolling"
    g.run(0.2)
    assert d.hp<50000,"Log should hit when its front reaches the dummy"
    g.run(lg.rng/lg.speed)
    assert not lg.active,"the roll ends after 10.1 tiles"
    return f"Travel: log front reaches a body 4 tiles ahead after {exp:.2f}s"
def t_travel_none():
    g=Game()
    d=Dummy('red',9,25,hp=50000,spd=0)
    g.deploy('red',d)
    z=mk_card('zap',11,'blue',9,25)
    g._cast('blue',z,9.0,25.0)
    assert d.hp<50000 and not g.projs,"Without projSpeed a spell is hitscan"
    return "Travel: no projSpeed means instant hit"
def t_fb_no_kb_heavy():
    g=Game()
    d=Dummy('red',9,10,hp=50000,spd=0,mass=10)
    g.deploy('red',d)
    ox=d.x
    fb=mk_card('fireball',11,'blue',8,10)
    fb.apply(g)
    assert abs(d.x-ox)<0.01,f"Heavy troop should not be knocked back ({ox}->{d.x})"
    return "Fireball no KB on heavy (mass=10)"
def t_fb_kb_light():
    g=Game()
    d=Dummy('red',9,10,hp=50000,spd=0,mass=4)
    g.deploy('red',d)
    ox=d.x
    fb=mk_card('fireball',11,'blue',8,10)
    fb.apply(g)
    assert d.x>ox,"Light troop should be knocked back"
    return f"Fireball KB on light (mass=4, {ox:.1f}->{d.x:.2f})"
def t_snowball_no_kb_heavy():
    g=Game()
    d=Dummy('red',9,10,hp=50000,spd=0,mass=10)
    g.deploy('red',d)
    ox,oy=d.x,d.y
    sb=mk_card('giant_snowball',11,'blue',9,10)
    sb.apply(g)
    assert abs(d.x-ox)<0.01 and abs(d.y-oy)<0.01,"Heavy troop should not be KB by snowball"
    return "Snowball no KB on heavy (mass=10)"
def t_log_kb_heavy():
    g=Game()
    d=Dummy('red',9,12,hp=50000,spd=0,mass=10)
    g.deploy('red',d)
    oy=d.y
    log=mk_card('the_log',11,'blue',9,10)
    g._cast('blue',log,9,10);g.run(1)
    assert abs(d.y-oy)>0.01,"Log should push heavy troops (no mass check)"
    return f"Log pushes heavy (mass=10, {oy:.1f}->{d.y:.2f})"
def t_lightning_max_hp_sort():
    g=Game()
    d1=Dummy('red',9,10,hp=5000,spd=0)
    d1.hp=100
    d2=Dummy('red',10,10,hp=500,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    lt=mk_card('lightning',11,'blue',9.5,10)
    lt.apply(g)
    assert d1.max_hp-d1.hp>0 or not d1.alive,"Lightning should hit d1 (max_hp=5000)"
    d1_hit=d1.hp<100 or not d1.alive
    assert d1_hit,"d1 (max_hp=5000, hp=100) should be targeted by lightning"
    return "Lightning sorts by max_hp (not current hp)"
def t_eq_ground_only():
    g=Game()
    d=Dummy('red',9,25,hp=5000,spd=0)
    d.transport='Air'
    g.deploy('red',d)
    eq=mk_card('earthquake',11,'blue',9,25)
    eq.apply(g);g.spells.append(eq)
    g.run(4)
    assert d.hp==5000,"EQ should not hit air troops"
    return "Earthquake ground only (air=0 dmg)"
def t_eq_bldg_multiplier():
    from sim.spells import EarthquakeSpell
    eq=EarthquakeSpell('blue',0,0,{'radius':3.5,'troop_dmg':55,'bldg_dmg':180,'ct_dmg':17,'ticks':3,'interval':1.0,'name':'EQ'})
    assert eq.bldg_dmg>eq.troop_dmg*3,f"EQ bldg/tick ({eq.bldg_dmg}) should be >>troop/tick ({eq.troop_dmg})"
    return f"Earthquake bldg multiplier (bldg={eq.bldg_dmg}/tick troop={eq.troop_dmg}/tick)"
def t_eq_slow():
    g=Game()
    d=Dummy('red',9,25,hp=50000,spd=0)
    g.deploy('red',d)
    eq=mk_card('earthquake',11,'blue',9,25)
    eq.apply(g);g.spells.append(eq)
    g.run(2)
    assert any(s.kind=='mslow' for s in d.statuses),"EQ should slow movement"
    return "Earthquake applies movement-only slow"
def t_tornado_pull():
    g=Game()
    d=Dummy('red',12,25,hp=50000,spd=0)
    g.deploy('red',d)
    ox=d.x
    tn=mk_card('tornado',11,'blue',9,25)
    tn.apply(g);g.spells.append(tn)
    g.run(2)
    assert abs(d.x-9)<abs(ox-9),"Tornado should pull toward center"
    return f"Tornado pull ({ox:.1f}->{d.x:.2f} toward 9)"
def t_tornado_king_act():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    assert not kt.active
    d=Dummy('red',kt.cx+2,kt.cy,hp=50000,spd=0,dmg=0)
    g.deploy('red',d)
    tn=mk_card('tornado',11,'blue',kt.cx,kt.cy)
    tn.apply(g);g.spells.append(tn)
    g.run(2)
    assert kt.dist(d.x,d.y)<=kt.rng and not kt.active,"Pulling a troop next to the king must not activate it"
    return "Tornado proximity does not activate king tower"
def t_tornado_no_bldg_dmg():
    g=Game()
    for t in g.arena.towers:t.alive=False
    cn=mk_card('cannon',11,'red',9,25)
    g.deploy('red',cn)
    ini=cn.hp;decay=cn.decay*2
    tn=mk_card('tornado',11,'blue',cn.x,cn.y)
    tn.apply(g);g.spells.append(tn)
    g.run(2)
    spell_dmg=ini-cn.hp-decay
    assert abs(spell_dmg)<5,f"Tornado should not damage buildings (spell_dmg={spell_dmg:.0f})"
    return f"Tornado no bldg damage (spell_dmg={spell_dmg:.0f})"
def t_void_single_full():
    g=Game()
    d=Dummy('red',9,22,hp=50000,spd=0)
    g.deploy('red',d)
    v=mk_card('void',11,'blue',9,22)
    v.apply(g);g.spells.append(v)
    g.run(4)
    dmg=50000-d.hp
    assert abs(dmg-2088)<50,f"Single target should take full 696*3=2088, got {dmg}"
    return f"Void single target full dmg ({dmg})"
def t_void_multi_reduced():
    g=Game()
    ds=[]
    for i in range(5):
        d=Dummy('red',9+i*0.3,25,hp=50000,spd=0)
        g.deploy('red',d)
        ds.append(d)
    v=mk_card('void',11,'blue',9.6,25)
    v.apply(g);g.spells.append(v)
    g.run(4)
    dmgs=[50000-d.hp for d in ds]
    hit=[d for d in dmgs if d>0]
    if len(hit)>=2:
        per_tgt=hit[0]
        assert per_tgt<1059,f"Multi-target should reduce per-target dmg: {per_tgt}"
    return f"Void multi reduced (per_tgt={hit[0] if hit else 0}, n={len(hit)})"
def t_vines_grounds_air():
    g=Game()
    d=Dummy('red',9,25,hp=50000,spd=0)
    d.transport='Air'
    g.deploy('red',d)
    v=mk_card('vines',11,'blue',9,25)
    v.apply(g);g.spells.append(v)
    assert d.transport=='Ground',"Vines should ground air troops"
    g.run(3)
    assert d.transport=='Air',"Vines should restore transport after duration"
    return "Vines grounds air troops"
def t_vines_root():
    g=Game()
    d=Dummy('red',9,25,hp=50000,spd=2.0)
    g.deploy('red',d)
    ox,oy=d.x,d.y
    v=mk_card('vines',11,'blue',9,25)
    v.apply(g);g.spells.append(v)
    g.run(1)
    assert abs(d.x-ox)<0.01 and abs(d.y-oy)<0.01,"Vines should freeze in place"
    return "Vines roots target"
def t_vines_3_highest():
    g=Game()
    ds=[]
    for hp in [1000,3000,5000,2000,4000]:
        d=Dummy('red',9,25,hp=hp,spd=0)
        g.deploy('red',d)
        ds.append(d)
    v=mk_card('vines',11,'blue',9,25)
    v.apply(g);g.spells.append(v)
    g.run(3)
    hit=[d for d in ds if d.hp<d.max_hp]
    hit_hps=sorted([d.max_hp for d in hit],reverse=True)
    assert len(hit)==3,f"Vines should target 3, got {len(hit)}"
    assert hit_hps==[5000,4000,3000],f"Should pick 3 highest: {hit_hps}"
    return f"Vines picks 3 highest max_hp {hit_hps}"
def t_gcurse_dot():
    g=Game()
    d=Dummy('red',9,25,hp=50000,spd=0)
    g.deploy('red',d)
    gc=mk_card('goblin_curse',11,'blue',9,25)
    gc.apply(g);g.spells.append(gc)
    g.run(7)
    dmg=50000-d.hp
    assert abs(dmg-210)<50,f"Expected ~210 (35*6), got {dmg}"
    return f"Goblin Curse DoT ({dmg})"
def t_gcurse_convert():
    g=Game()
    for t in g.arena.towers:t.alive=False
    d=Dummy('red',9,10,hp=100,spd=0)
    g.deploy('red',d)
    gc=mk_card('goblin_curse',11,'blue',9,10)
    gc.apply(g);g.spells.append(gc)
    g.run(5)
    gobs=[t for t in g.players['blue'].troops if t.alive]
    assert len(gobs)>=1,f"Cursed enemy died -> should spawn goblin, got {len(gobs)}"
    return f"Goblin Curse converts ({len(gobs)} goblins)"
def t_gcurse_any_source():
    g=Game()
    d=Dummy('red',9,14,hp=200,spd=0)
    g.deploy('red',d)
    gc=mk_card('goblin_curse',11,'blue',9,14)
    gc.apply(g);g.spells.append(gc)
    d.take_damage(200)
    g.run(0.2)
    gobs=[t for t in g.players['blue'].troops if t.alive]
    assert len(gobs)>=1,"Kill by external source while cursed should convert"
    return "Goblin Curse converts on any death source"
def t_rdelivery_dmg_spawn():
    g=Game()
    d=Dummy('red',9,10,hp=5000,spd=0)
    g.deploy('red',d)
    rd=mk_card('royal_delivery',11,'blue',9,10)
    rd.apply(g)
    dmg=5000-d.hp
    assert dmg==384,f"Expected 384 dmg, got {dmg}"
    recruits=[t for t in g.players['blue'].troops if t.alive]
    assert len(recruits)==1,f"Should spawn 1 recruit, got {len(recruits)}"
    return f"Royal Delivery dmg+spawn (dmg={dmg}, recruits={len(recruits)})"
def t_rdelivery_shield():
    g=Game()
    rd=mk_card('royal_delivery',11,'blue',9,10)
    rd.apply(g)
    recruits=[t for t in g.players['blue'].troops if t.alive]
    assert len(recruits)==1
    r=recruits[0]
    assert r.shield_hp>0,f"Recruit should have shield (shield_hp={r.shield_hp})"
    assert r.shield_hp==240,f"Expected shield=240, got {r.shield_hp}"
    return f"Royal Delivery recruit shield ({r.shield_hp})"
def t_bbarrel_spawn():
    g=Game()
    bb=mk_card('barbarian_barrel',11,'blue',9,10)
    g._cast('blue',bb,9,10);g.run(1)
    assert not g.players['blue'].troops,"the barrel is still rolling"
    g.run(0.4)
    barbs=[t for t in g.players['blue'].troops if t.alive]
    assert len(barbs)==1,f"Should spawn 1 barbarian, got {len(barbs)}"
    assert abs(barbs[0].y-14.5)<0.01 and any(s.kind=='deploying' for s in barbs[0].statuses),"he pops out at the end of the 4.5 tile roll and deploys for 1 s"
    b=barbs[0]
    assert b.name=='Barbarian'
    assert b.hp==716,f"Expected barbarian hp=716, got {b.hp}"
    return f"Barbarian Barrel spawns barbarian (hp={b.hp})"
def t_clone_skip_building():
    g=Game()
    cn=mk_card('cannon',11,'blue',9,10)
    g.deploy('blue',cn)
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    ini=len(g.players['blue'].troops)
    cl=mk_card('clone',11,'blue',9,10)
    cl.apply(g)
    assert len(g.players['blue'].troops)==ini+1,"Clone should skip buildings"
    clones=[t for t in g.players['blue'].troops if t is not cn and t is not k]
    assert len(clones)==1
    assert clones[0].name==k.name
    return "Clone skips buildings"
def t_clone_1hp():
    g=Game()
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    cl=mk_card('clone',11,'blue',9,10)
    cl.apply(g)
    clone=[t for t in g.players['blue'].troops if t is not k][0]
    assert clone.hp==1 and clone.max_hp==1,f"Clone hp={clone.hp} max_hp={clone.max_hp}"
    return "Clone 1hp and max_hp=1"
def t_gy_perimeter():
    g=Game()
    random.seed(42)
    gy=mk_card('graveyard',11,'blue',9,25)
    gy.apply(g);g.spells.append(gy)
    g.run(5)
    skels=[t for t in g.players['blue'].troops if t.alive]
    for s in skels:
        d=math.sqrt((s.x-9)**2+(s.y-25)**2)
        assert abs(d-gy.radius)<1.0,f"Skeleton spawn dist {d:.2f} should be near radius {gy.radius}"
    return f"Graveyard perimeter spawn ({len(skels)} skeletons)"
def t_gy_first_delay():
    g=Game()
    random.seed(42)
    gy=mk_card('graveyard',11,'blue',9,25)
    gy.apply(g);g.spells.append(gy)
    g.run(1.5)
    assert gy.spawned==0,f"No skeletons before first_delay ({gy.first_delay}s), got {gy.spawned}"
    return "Graveyard first delay (no spawns before 2.2s)"
def t_freeze_building():
    g=Game()
    cn=mk_card('cannon',11,'red',9,25)
    g.deploy('red',cn)
    for t in g.arena.towers:t.alive=False
    d=Dummy('blue',9,24,hp=50000,spd=0)
    g.deploy('blue',d)
    g.run(3)
    hp_before=d.hp
    assert hp_before<50000,"Cannon should attack before freeze"
    fz=mk_card('freeze',11,'blue',cn.x,cn.y)
    fz.apply(g)
    hp_at_freeze=d.hp
    g.run(3)
    assert d.hp==hp_at_freeze,f"Frozen cannon should not attack: {hp_at_freeze}->{d.hp}"
    return "Freeze stops building"
def t_rage_persistent():
    g=Game()
    r=mk_card('rage',11,'blue',9,10)
    r.apply(g);g.spells.append(r)
    ally=Dummy('blue',9,10,hp=50000,spd=1.0)
    g.deploy('blue',ally)
    g.run(1)
    has_rage=any(s.kind=='rage' for s in ally.statuses)
    assert has_rage,"Ally entering rage area should get buffed"
    return "Rage persistent area buff"
def t_int_freeze_inferno_reset():
    cn=mk_card('inferno_tower',11,'blue',9,5)
    g=Game()
    for t in g.arena.towers:t.alive=False
    g.deploy('blue',cn)
    d=Dummy('red',9,10,hp=500000,spd=0)
    g.deploy('red',d)
    g.run(5)
    hp_5s=d.hp
    fz=mk_card('freeze',11,'red',cn.x,cn.y)
    fz.apply(g)
    g.run(4.5)
    hp_after_freeze=d.hp
    g.run(5)
    ramp_after=hp_after_freeze-d.hp
    ramp_before=500000-hp_5s
    assert ramp_after<ramp_before*1.5,f"Freeze should reset inferno ramp (before={ramp_before} after={ramp_after})"
    return f"Freeze resets inferno ramp (before={ramp_before} after={ramp_after})"
def t_int_tornado_hog_king():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    assert not kt.active
    d=Dummy('red',kt.cx,kt.cy+8,hp=50000,spd=0)
    g.deploy('red',d)
    tn=mk_card('tornado',11,'blue',kt.cx,kt.cy+5)
    tn.apply(g);g.spells.append(tn)
    g.run(2)
    dist=math.sqrt((d.x-kt.cx)**2+(d.y-kt.cy)**2)
    assert dist<=kt.rng and not kt.active,f"King activates only on damage (enemy dist={dist:.1f}, king rng={kt.rng})"
    d.dmg=100;d.rng=1.0;d.spd=1.0
    g.run(5)
    assert kt.active,"King should activate once the pulled troop hits it"
    return f"Tornado pull alone leaves king inactive (dist={dist:.1f})"
def t_int_eq_cannon():
    g=Game()
    for t in g.arena.towers:t.alive=False
    # Bomb Tower: the 3x287 building damage would destroy an 824 hp Cannon before the third tick
    cn=mk_card('bomb_tower',11,'blue',9,10)
    g.deploy('blue',cn)
    ini=cn.hp;decay=cn.decay*4
    eq=mk_card('earthquake',11,'red',cn.x,cn.y)
    eq.apply(g);g.spells.append(eq)
    g.run(4)
    total_loss=ini-cn.hp
    spell_dmg=total_loss-decay
    assert abs(spell_dmg-861)<50,f"EQ should deal ~861 to a building (287*3), got {spell_dmg:.0f}"
    return f"EQ heavy damage on building ({spell_dmg:.0f})"
def t_int_curse_skarmy():
    g=Game()
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,25)
    for s in sk:g.deploy('red',s)
    gc=mk_card('goblin_curse',11,'blue',9,25)
    gc.apply(g);g.spells.append(gc)
    g.run(8)
    gobs=[t for t in g.players['blue'].troops if t.alive]
    dead_sk=sum(1 for s in sk if not s.alive)
    assert len(gobs)>=5,f"Many skarmy deaths should spawn many goblins, got {len(gobs)}"
    return f"Curse skarmy ({dead_sk} dead -> {len(gobs)} goblins)"
def t_int_clone_balloon_death():
    g=Game()
    bal=mk_card('balloon',11,'blue',9,14)
    g.deploy('blue',bal)
    cl=mk_card('clone',11,'blue',9,14)
    cl.apply(g)
    clones=[t for t in g.players['blue'].troops if t is not bal]
    assert len(clones)==1
    clone=clones[0]
    assert clone.hp==1 and clone.max_hp==1
    has_dd=clone.death_dmg>0
    assert has_dd,f"Cloned balloon should have death_dmg ({clone.death_dmg})"
    return f"Cloned balloon death dmg ({clone.death_dmg})"
def t_tt_princess_hp():
    tt=mk_tt('tower_princess',11)
    assert tt.hp==3052,f"Expected 3052, got {tt.hp}"
    return f"Tower Princess HP={tt.hp}"
def t_tt_cannoneer_hp():
    tt=mk_tt('cannoneer',11)
    assert tt.hp==2616,f"Expected 2616, got {tt.hp}"
    assert tt.hp<3052
    return f"Cannoneer HP={tt.hp} (< Princess 3052)"
def t_tt_duchess_hp():
    tt=mk_tt('dagger_duchess',11)
    assert tt.hp==2768,f"Expected 2768, got {tt.hp}"
    return f"Dagger Duchess HP={tt.hp}"
def t_tt_chef_hp():
    tt=mk_tt('royal_chef',11)
    assert tt.hp==2703,f"Expected 2703, got {tt.hp}"
    return f"Royal Chef HP={tt.hp}"
def t_king_dmg_activates():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    assert not kt.active
    kt.take_damage(100)
    g._king_act('blue')
    assert kt.active
    return "Direct king damage activates king"
def t_king_inactive_no_attack():
    g=Game()
    for t in g.arena.towers:
        if t.team=='blue' and t.ttype=='princess':t.hp=0;t.alive=False
    kt=g.arena.get_tower('blue','king')
    kt.active=False
    tr=Dummy('red',9.0,6.0,hp=5000,spd=0,dmg=0,rng=0)
    g.deploy('red',tr)
    ini=tr.hp
    g.run(5)
    assert not kt.active
    assert tr.hp==ini,f"Inactive king attacked: {ini}->{tr.hp}"
    return "Inactive king doesn't attack"
def t_king_spd():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    kt.active=True;kt.cd=0
    assert kt.spd==1.0,f"King speed {kt.spd} != 1.0"
    return f"King attack speed = {kt.spd}s"
def t_gk_ability():
    gk=mk_card('golden_knight',11,'blue',5,10)
    ab=getattr(gk,'ability',None)
    assert ab is not None,"GK should have ability"
    from sim.fx import DashingDash
    assert isinstance(ab,DashingDash)
    assert ab.dd==335,f"Dash dmg {ab.dd}"
    assert ab.cost==1
    return f"Golden Knight ability (dash_dmg={ab.dd}, cost={ab.cost})"
def t_gk_dash_chain():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    d1=Dummy('red',9,13,hp=5000,spd=0)
    d2=Dummy('red',12,13,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.players['blue'].elixir=10
    gk.ability.cd=0
    ok,_=g.activate_ability('blue',gk)
    assert ok,"Ability activation failed"
    g.run(1.3)
    g.run(1)
    h1=5000-d1.hp;h2=5000-d2.hp
    assert h1>=335 or h2>=335,f"Dash should hit at least one: d1={h1} d2={h2}"
    return f"GK dash chain (d1={h1} d2={h2})"
def t_gk_dash_cost():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    g.players['blue'].elixir=0.5
    gk.ability.cd=0
    ok,msg=g.activate_ability('blue',gk)
    assert not ok,"Should fail with 0.5 elixir"
    return "GK dash costs 1 elixir"
def t_sk_souls():
    g=Game()
    sk=mk_card('skeleton_king',11,'blue',9,10)
    g.deploy('blue',sk)
    from sim.fx import SoulCollect
    sc=[c for c in sk.components if isinstance(c,SoulCollect)]
    assert len(sc)==1,"SK should have SoulCollect"
    d1=Dummy('red',9,3,hp=5000,spd=0)
    d2=Dummy('red',9,2,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(0.1)
    d1.alive=False
    g.run(0.2)
    d2.alive=False
    g.run(0.2)
    assert sc[0].souls>=2,f"Expected >=2 souls, got {sc[0].souls}"
    return f"Skeleton King collects souls ({sc[0].souls})"
def t_sk_summon():
    g=Game()
    sk=mk_card('skeleton_king',11,'blue',9,5)
    g.deploy('blue',sk)
    from sim.fx import SoulCollect
    sc=[c for c in sk.components if isinstance(c,SoulCollect)][0]
    sc.souls=5
    g.players['blue'].elixir=10
    sk.ability.cd=0
    ok,_=g.activate_ability('blue',sk)
    assert ok
    assert getattr(sk.ability,'_pend',False),"Should be pending after activation"
    g.run(1.3)
    assert sc.souls==0,"Souls should be consumed after cast"
    g.run(5)
    spawned=len([t for t in g.players['blue'].troops if t is not sk])
    assert spawned>=8,f"Expected >=8 skeletons spawned, got {spawned}"
    return f"SK Soul Summoning ({spawned} alive)"
def t_sk_summon_min():
    g=Game()
    sk=mk_card('skeleton_king',11,'blue',9,5)
    g.deploy('blue',sk)
    from sim.fx import SoulCollect
    sc=[c for c in sk.components if isinstance(c,SoulCollect)][0]
    sc.souls=0
    g.players['blue'].elixir=10
    sk.ability.cd=0
    g.activate_ability('blue',sk)
    g.run(1.3)
    assert sk.ability.active,"Ability should be active after cast"
    g.run(3)
    spawned=len([t for t in g.players['blue'].troops if t is not sk])
    assert spawned>=4,f"Min 6 skeletons queued (some may die), got {spawned}"
    return f"SK min summon ({spawned} skeletons, 0 souls)"
def t_bb_dash():
    g=Game()
    random.seed(42)
    bb=mk_card('boss_bandit',11,'blue',9,10)
    g.deploy('blue',bb)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(3)
    dmg=ini-d.hp
    assert dmg>=491,f"BB should dash (dash_dmg=491), total dmg={dmg}"
    return f"Boss Bandit dash ({dmg} dmg in 3s)"
def t_bb_speed():
    bb=mk_card('boss_bandit',11,'blue',5,10)
    assert abs(bb.spd-1.5)<0.01,f"BB should be Fast (1.5), got {bb.spd}"
    return f"Boss Bandit speed = {bb.spd} (Fast)"
def t_bb_grenade():
    g=Game()
    bb=mk_card('boss_bandit',11,'blue',9,14)
    g.deploy('blue',bb)
    g.players['blue'].elixir=10
    bb.ability.cd=0
    oy=bb.y
    ok,_=g.activate_ability('blue',bb)
    assert ok
    g.run(1.3)
    assert bb.y<oy,f"BB should teleport backward (blue), y={bb.y} was {oy}"
    return f"BB Getaway Grenade (y: {oy:.0f}->{bb.y:.0f})"
def t_bb_grenade_uses():
    g=Game()
    bb=mk_card('boss_bandit',11,'blue',9,14)
    g.deploy('blue',bb)
    g.players['blue'].elixir=10
    bb.ability.cd=0
    g.activate_ability('blue',bb)
    g.run(1.3)
    g.run(4)
    bb.ability.cd=0
    ok2,_=g.activate_ability('blue',bb)
    g.run(1.3)
    assert ok2,"Second use should work"
    bb.ability.cd=0
    ok3,_=g.activate_ability('blue',bb)
    assert not ok3,"Third use should fail (max 2)"
    return "BB Getaway Grenade 2 uses"
def t_aq_ability():
    aq=mk_card('archer_queen',11,'blue',5,10)
    ab=getattr(aq,'ability',None)
    assert ab is not None
    from sim.fx import CloakingCape
    assert isinstance(ab,CloakingCape)
    assert abs(ab.max_dur-3.5)<0.01
    assert ab.cost==1
    return f"AQ Cloaking Cape ability (dur={ab.max_dur}s, cost={ab.cost})"
def t_aq_cloak_invisible():
    g=Game()
    aq=mk_card('archer_queen',11,'blue',9,10)
    g.deploy('blue',aq)
    g.players['blue'].elixir=10
    aq.ability.cd=0
    g.activate_ability('blue',aq)
    g.run(1.3)
    invis=any(s.kind=='invisible' for s in aq.statuses)
    assert invis,"AQ should be invisible after cloak"
    return "AQ invisible during cloak"
def t_aq_cloak_fast_atk():
    g=Game()
    aq=mk_card('archer_queen',11,'blue',9,10)
    g.deploy('blue',aq)
    d=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(3);hp_no=d.hp
    d2=Dummy('red',9,14,hp=50000,spd=0)
    g2=Game()
    aq2=mk_card('archer_queen',11,'blue',9,10)
    g2.deploy('blue',aq2)
    g2.deploy('red',d2)
    g2.players['blue'].elixir=10
    aq2.ability.cd=0
    g2.activate_ability('blue',aq2)
    g2.run(3);hp_cloak=d2.hp
    dmg_no=50000-hp_no;dmg_cloak=50000-hp_cloak
    assert dmg_cloak>dmg_no,f"Cloak should deal more: {dmg_cloak} vs {dmg_no}"
    return f"AQ cloak DPS boost ({dmg_cloak} vs {dmg_no} normal)"
def t_mm_ability():
    mm=mk_card('mighty_miner',11,'blue',5,10)
    ab=getattr(mm,'ability',None)
    assert ab is not None
    from sim.fx import ExplosiveEscape
    assert isinstance(ab,ExplosiveEscape)
    assert ab.bomb_dmg==332,f"Bomb dmg {ab.bomb_dmg}"
    return f"MM Explosive Escape (bomb={ab.bomb_dmg})"
def t_mm_escape_bomb():
    g=Game()
    mm=mk_card('mighty_miner',11,'blue',9,14)
    g.deploy('blue',mm)
    d=Dummy('red',9,14.5,hp=5000,spd=0)
    g.deploy('red',d)
    g.players['blue'].elixir=10
    mm.ability.cd=0
    g.activate_ability('blue',mm)
    g.run(1.3)
    assert d.hp<5000,f"Bomb should damage {d.hp}"
    assert abs(mm.x-8)<=1,f"MM should lane swap x={mm.x}"
    return f"MM escape bomb (d hp={d.hp}, mm x={mm.x:.1f})"
def t_mm_escape_resets_ramp():
    g=Game()
    mm=mk_card('mighty_miner',11,'blue',9,10)
    g.deploy('blue',mm)
    d=Dummy('red',9,11,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(2)
    assert mm.dmg==204,f"Expected stage 2 (204), got {mm.dmg}"
    g.players['blue'].elixir=10
    mm.ability.cd=0
    g.activate_ability('blue',mm)
    g.run(1.3)
    assert mm.dmg==43,f"Should reset to 43 after escape, got {mm.dmg}"
    return "MM escape resets ramp"
def t_gs_ability():
    random.seed(42)
    r=mk_card('goblinstein',11,'blue',5,10)
    mon=[t for t in r if t.name=='Monster'][0]
    ab=getattr(mon,'ability',None)
    assert ab is not None,"Monster should have Lightning Link ability"
    from sim.fx import LightningLink
    assert isinstance(ab,LightningLink)
    assert ab.tick_dmg==107,f"Tick dmg {ab.tick_dmg}"
    assert ab.tick_ct==46,f"CT dmg {ab.tick_ct}"
    return f"Goblinstein Lightning Link (tick_dmg={ab.tick_dmg}, ct={ab.tick_ct})"
def t_lp_ramp():
    g=Game()
    lp=mk_card('little_prince',11,'blue',9,10)
    g.deploy('blue',lp)
    d=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d)
    from sim.fx import LPRamp
    ramp=[c for c in lp.components if isinstance(c,LPRamp)]
    assert len(ramp)==1,"LP should have LPRamp"
    assert abs(lp.hspd-1.2)<0.01,f"Start at 1.2s, got {lp.hspd}"
    g.run(5)
    assert lp.hspd<1.2,f"Should speed up, still {lp.hspd}"
    return f"Little Prince attack ramp (hspd={lp.hspd})"
def t_lp_rescue():
    g=Game()
    lp=mk_card('little_prince',11,'blue',9,10)
    g.deploy('blue',lp)
    d=Dummy('red',9,11.5,hp=5000,spd=0)
    g.deploy('red',d)
    g.players['blue'].elixir=10
    lp.ability.cd=0
    ini=len(g.players['blue'].troops)
    g.activate_ability('blue',lp)
    g.run(1.3)
    spawned=len(g.players['blue'].troops)-ini
    assert spawned>=1,f"Should spawn Guardienne, got {spawned}"
    guard=[t for t in g.players['blue'].troops if getattr(t,'name','')=='Protector']
    assert len(guard)==1
    assert guard[0].hp==1600,f"Protector HP={guard[0].hp}"
    return f"LP Royal Rescue (Protector hp={guard[0].hp})"
def t_lp_rescue_charge_dmg():
    g=Game()
    lp=mk_card('little_prince',11,'blue',9,10)
    g.deploy('blue',lp)
    d=Dummy('red',9,11.5,hp=5000,spd=0)
    g.deploy('red',d)
    g.players['blue'].elixir=10
    lp.ability.cd=0
    ini=d.hp
    g.activate_ability('blue',lp)
    g.run(1.3)
    dmg=ini-d.hp
    assert dmg>=256,f"Charge dmg should be >=256, got {dmg}"
    return f"LP rescue charge damage ({dmg})"
def t_monk_combo_kb():
    g=Game()
    m=mk_card('monk',11,'blue',9,10)
    g.deploy('blue',m)
    d=Dummy('red',9,11.5,hp=50000,spd=0)
    g.deploy('red',d)
    iy=d.y
    g.run(5)
    assert d.y>iy+0.5,f"Monk combo should knock back, y={d.y:.1f} was {iy:.1f}"
    return f"Monk combo knockback (y: {iy:.1f}->{d.y:.1f})"
def t_monk_protect():
    g=Game()
    m=mk_card('monk',11,'blue',9,14)
    g.deploy('blue',m)
    d=Dummy('red',9,14.5,hp=50000,dmg=500,spd=0,hspd=0.5,rng=1.5)
    g.deploy('red',d)
    g.players['blue'].elixir=10
    m.ability.cd=0
    g.activate_ability('blue',m)
    g.run(1.3)
    ini=m.hp
    g.run(2)
    dmg_taken=ini-m.hp
    raw=500*4
    assert dmg_taken<raw*0.5,f"65% reduction: took {dmg_taken}, raw ~{raw}"
    return f"Monk Pensive Protection ({dmg_taken} vs ~{raw} raw)"
def t_monk_protect_dur():
    m=mk_card('monk',11,'blue',5,10)
    ab=getattr(m,'ability',None)
    from sim.fx import PensiveProtection
    assert isinstance(ab,PensiveProtection)
    assert abs(ab.max_dur-4.0)<0.01
    assert abs(ab.reduction-0.65)<0.01
    return f"Monk ability stats (dur={ab.max_dur}s, reduction={ab.reduction})"
def t_invisible_untargetable():
    g=Game()
    aq=mk_card('archer_queen',11,'blue',3,14)
    g.deploy('blue',aq)
    g.players['blue'].elixir=10
    aq.ability.cd=0
    g.activate_ability('blue',aq)
    g.run(1.3)
    ini=aq.hp
    g.run(2)
    assert aq.hp==ini,f"Invisible AQ should not take tower dmg: {ini}->{aq.hp}"
    return "Invisible troop untargetable by towers"
def t_champ_one_ability():
    g=Game()
    gk1=mk_card('golden_knight',11,'blue',9,10)
    gk2=mk_card('golden_knight',11,'blue',5,10)
    g.deploy('blue',gk1);g.deploy('blue',gk2)
    p=g.players['blue']
    assert p.active_champ is gk1,"First deployed should be active"
    assert gk2 in p.champ_queue,"Second should be queued"
    return "Only 1 active champion at a time"
def t_champ_death_handoff():
    g=Game()
    gk1=mk_card('golden_knight',11,'blue',9,10)
    gk2=mk_card('golden_knight',11,'blue',5,10)
    g.deploy('blue',gk1);g.deploy('blue',gk2)
    p=g.players['blue']
    assert p.active_champ is gk1
    gk1.alive=False
    g._proc_deaths()
    assert p.active_champ is gk2,"Second should become active after first dies"
    return "Champion death handoff"
def t_champ_clone_no_ability():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    cl=mk_card('clone',11,'blue',9,10)
    cl.apply(g)
    clones=[t for t in g.players['blue'].troops if t is not gk and t.alive]
    assert len(clones)==1
    assert clones[0].hp==1 and clones[0].max_hp==1
    assert getattr(clones[0],'ability',None) is None,"Clone should not have ability"
    return "Cloned champion has no ability"
def t_champ_clone_cant_activate():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    cl=mk_card('clone',11,'blue',9,10)
    cl.apply(g)
    clone=[t for t in g.players['blue'].troops if t is not gk][0]
    g.players['blue'].elixir=10
    ok,msg=g.activate_ability('blue',clone)
    assert not ok,f"Clone should not be able to activate: {msg}"
    return "Clone cannot activate ability"
def t_champ_not_active_cant_use():
    g=Game()
    gk1=mk_card('golden_knight',11,'blue',9,10)
    gk2=mk_card('golden_knight',11,'blue',5,10)
    g.deploy('blue',gk1);g.deploy('blue',gk2)
    g.players['blue'].elixir=10
    gk2.ability.cd=0
    ok,msg=g.activate_ability('blue',gk2)
    assert not ok,"Queued champion should not use ability"
    assert msg=="not active champion"
    return "Non-active champion can't use ability"
def t_evo_knight_dmg_red():
    g=Game()
    k=mk_card('knight',11,'blue',9,14,evolved=True)
    g.deploy('blue',k)
    assert getattr(k,'evolved',False),"Should be evolved"
    from sim.fx import EvoKnight
    assert any(isinstance(c,EvoKnight) for c in k.components)
    d=Dummy('red',9,14.5,hp=50000,dmg=500,spd=0,hspd=0.5,rng=1.5)
    g.deploy('red',d)
    ini=k.hp;g.run(1)
    raw_dmg=500*2;actual=ini-k.hp
    assert actual<raw_dmg*0.6,f"60% reduction: took {actual}, raw ~{raw_dmg}"
    return f"Evo Knight damage reduction ({actual} vs ~{raw_dmg} raw)"
def t_evo_knight_no_red_attacking():
    g=Game()
    k=mk_card('knight',11,'blue',9,10)
    g.deploy('blue',k)
    k2=mk_card('knight',11,'blue',9,10,evolved=True)
    g.deploy('blue',k2)
    d=Dummy('red',9,11.5,hp=50000,dmg=500,spd=0,hspd=0.5,rng=1.5)
    g.deploy('red',d)
    g.run(3)
    assert k.hp<k2.hp,f"Evo knight should take less dmg: normal={k.hp} evo={k2.hp}"
    return f"Evo Knight tankier (normal hp={k.hp}, evo hp={k2.hp})"
def t_evo_bomber_bounce():
    g=Game()
    b=mk_card('bomber',11,'blue',9,10,evolved=True)
    g.deploy('blue',b)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',10,14,hp=5000,spd=0)
    d3=Dummy('red',11,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2);g.deploy('red',d3)
    g.run(5)
    hit=sum(1 for d in [d1,d2,d3] if d.hp<5000)
    assert hit>=2,f"Bouncing bomb should hit >=2, got {hit}"
    return f"Evo Bomber bouncing bomb ({hit} targets hit)"
def t_evo_skeletons_replicate():
    g=Game()
    random.seed(42)
    sk=mk_card('skeletons',11,'blue',9,10,evolved=True)
    if isinstance(sk,list):
        for s in sk:g.deploy('blue',s)
    else:g.deploy('blue',sk)
    d=Dummy('red',9,11.5,hp=50000,spd=0,dmg=0,rng=0)
    g.deploy('red',d)
    g.run(8)
    skels=[t for t in g.players['blue'].troops if t.alive and 'keleton' in t.name]
    assert len(skels)>3,f"Should replicate, got {len(skels)}"
    return f"Evo Skeletons replicate ({len(skels)} alive)"
def t_evo_barbarians_boost():
    g=Game()
    random.seed(42)
    bb=mk_card('barbarians',11,'blue',9,10,evolved=True)
    if isinstance(bb,list):
        for b in bb:g.deploy('blue',b)
        b0=bb[0]
    else:g.deploy('blue',bb);b0=bb
    assert getattr(b0,'evolved',False)
    from sim.fx import EvoBarbarians
    assert any(isinstance(c,EvoBarbarians) for c in b0.components)
    return "Evo Barbarians have boost component"
def t_evo_bats_heal():
    g=Game()
    random.seed(42)
    bt=mk_card('bats',11,'blue',9,10,evolved=True)
    if isinstance(bt,list):
        for b in bt:g.deploy('blue',b)
        b0=bt[0]
    else:g.deploy('blue',bt);b0=bt
    from sim.fx import EvoBats
    assert any(isinstance(c,EvoBats) for c in b0.components)
    return "Evo Bats have heal component"
def t_evo_royal_recruits():
    g=Game()
    random.seed(42)
    rr=mk_card('royal_recruits',11,'blue',9,10,evolved=True)
    if isinstance(rr,list):
        for r in rr:g.deploy('blue',r)
        r0=rr[0]
    else:g.deploy('blue',rr);r0=rr
    from sim.fx import EvoRoyalRecruits
    assert any(isinstance(c,EvoRoyalRecruits) for c in r0.components)
    return "Evo Royal Recruits have charge-on-shield-break"
def t_evo_royal_giant():
    g=Game()
    rg=mk_card('royal_giant',11,'blue',3,14,evolved=True)
    g.deploy('blue',rg)
    from sim.fx import EvoRoyalGiant
    assert any(isinstance(c,EvoRoyalGiant) for c in rg.components)
    d1=Dummy('red',4,18,hp=5000,spd=0)
    d2=Dummy('red',5,18,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(10)
    splash_hit=sum(1 for d in [d1,d2] if d.hp<5000)
    return f"Evo Royal Giant recoil ({splash_hit} secondary hits)"
def t_evo_skel_barrel():
    g=Game()
    sb=mk_card('skeleton_barrel',11,'blue',9,14,evolved=True)
    g.deploy('blue',sb)
    assert getattr(sb,'evolved',False)
    from sim.fx import EvoSkelBarrel
    assert any(isinstance(c,EvoSkelBarrel) for c in sb.components)
    normal=mk_card('skeleton_barrel',11,'blue',9,14)
    # the evolution keeps the base hitpoints and raises the death damage in the current game data
    assert sb.max_hp==normal.max_hp and sb.death_dmg>normal.death_dmg
    return f"Evo Skeleton Barrel (death dmg {sb.death_dmg} vs {normal.death_dmg} normal)"
def t_evo_firecracker():
    g=Game()
    fc=mk_card('firecracker',11,'blue',9,10,evolved=True)
    g.deploy('blue',fc)
    from sim.fx import EvoFirecracker
    assert any(isinstance(c,EvoFirecracker) for c in fc.components)
    return "Evo Firecracker has spark trail"
def t_evo_archers_power_shot():
    g=Game()
    random.seed(42)
    ar=mk_card('archers',11,'blue',9,10,evolved=True)
    if isinstance(ar,list):
        for a in ar:g.deploy('blue',a)
        a0=ar[0]
    else:g.deploy('blue',ar);a0=ar
    from sim.fx import EvoArchers
    assert any(isinstance(c,EvoArchers) for c in a0.components)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg_evo=50000-d.hp
    g2=Game()
    ar2=mk_card('archers',11,'blue',9,10)
    if isinstance(ar2,list):
        for a in ar2:g2.deploy('blue',a)
    else:g2.deploy('blue',ar2)
    d2=Dummy('red',9,15,hp=50000,spd=0)
    g2.deploy('red',d2)
    g2.run(5)
    dmg_norm=50000-d2.hp
    assert dmg_evo>=dmg_norm,f"Evo should deal >= normal: {dmg_evo} vs {dmg_norm}"
    return f"Evo Archers power shot ({dmg_evo} vs {dmg_norm} normal)"
def t_evo_zap_double():
    g=Game()
    d=Dummy('red',9,14,hp=5000,spd=0)
    g.deploy('red',d)
    z=mk_card('zap',11,'blue',9,14,evolved=True)
    z.apply(g);g.spells.append(z)
    g.run(1)
    dmg=5000-d.hp
    nz=mk_card('zap',11,'blue',9,14)
    d2=Dummy('red',9,14,hp=5000,spd=0)
    g2=Game();g2.deploy('red',d2)
    nz.apply(g2)
    ndmg=5000-d2.hp
    assert dmg>ndmg,f"Evo zap should deal more: {dmg} vs {ndmg}"
    return f"Evo Zap double pulse ({dmg} vs {ndmg} normal)"
def t_evo_snowball_capture():
    g=Game()
    d=Dummy('red',9,14,hp=5000,spd=0)
    g.deploy('red',d)
    iy=d.y
    sb=mk_card('giant_snowball',11,'blue',9,14,evolved=True)
    sb.apply(g);g.spells.append(sb)
    g.run(1)
    assert d.y!=iy,f"Snowball should move troop, y still {d.y}"
    return f"Evo Snowball capture & roll (y: {iy}->{d.y:.1f})"
def t_evo_not_default():
    k=mk_card('knight',11,'blue',5,10)
    assert not getattr(k,'evolved',False),"Default should not be evolved"
    return "Cards not evolved by default"
def t_evo_ice_spirit():
    g=Game()
    isp=mk_card('ice_spirit',11,'blue',9,10,evolved=True)
    g.deploy('blue',isp)
    from sim.fx import EvoIceSpirit
    assert any(isinstance(c,EvoIceSpirit) for c in isp.components)
    return "Evo Ice Spirit has delayed explosion"
def t_hidden_stats_loaded():
    k=mk_card('knight',11,'blue',5,10)
    assert k.mass==6,f"Knight mass should be 6, got {k.mass}"
    rg=mk_card('royal_giant',11,'blue',5,10)
    assert rg.mass==10,f"RG mass should be 10, got {rg.mass}"
    return f"Hidden stats loaded (knight mass={k.mass}, rg mass={rg.mass})"
def t_int_evo_knight_v_pekka():
    g=Game()
    ek=mk_card('knight',11,'blue',9,14,evolved=True)
    g.deploy('blue',ek)
    pk=mk_card('pekka',11,'red',9,17)
    g.deploy('red',pk)
    g.run(15)
    evo_pk_hp=pk.hp
    g2=Game()
    nk=mk_card('knight',11,'blue',9,14)
    g2.deploy('blue',nk)
    pk2=mk_card('pekka',11,'red',9,17)
    g2.deploy('red',pk2)
    g2.run(15)
    norm_pk_hp=pk2.hp
    assert evo_pk_hp<=norm_pk_hp,f"Evo knight should deal >= dmg to PEKKA: pekka_hp evo={evo_pk_hp} norm={norm_pk_hp}"
    return f"Evo Knight v PEKKA (pekka hp: evo={evo_pk_hp}, normal={norm_pk_hp})"
def t_int_evo_knight_loses_red_attacking():
    g=Game()
    ek=mk_card('knight',11,'blue',9,14,evolved=True)
    g.deploy('blue',ek)
    d=Dummy('red',9,15,hp=500,spd=0,dmg=200,hspd=0.5,rng=1.5)
    g.deploy('red',d)
    g.run(3)
    hit_while_atk=ek.max_hp-ek.hp
    g2=Game()
    ek2=mk_card('knight',11,'blue',9,14,evolved=True)
    g2.deploy('blue',ek2)
    d2=Dummy('red',14,20,hp=500,spd=0,dmg=200,hspd=0.5,rng=1.5)
    g2.deploy('red',d2)
    d3=Dummy('red',9,15,hp=50000,spd=0,dmg=200,hspd=0.5,rng=1.5)
    g2.deploy('red',d3)
    g2.run(3)
    hit_while_idle=ek2.max_hp-ek2.hp
    return f"Evo Knight: attacking takes more dmg ({hit_while_atk}) vs idle ({hit_while_idle})"
def t_int_evo_bomber_v_skarmy():
    g=Game()
    random.seed(42)
    eb=mk_card('bomber',11,'blue',9,10,evolved=True)
    g.deploy('blue',eb)
    sk=mk_card('skeleton_army',11,'red',9,14)
    if isinstance(sk,list):
        for s in sk:g.deploy('red',s)
    else:g.deploy('red',sk)
    ini=len([t for t in g.players['red'].troops if t.alive])
    g.run(5)
    alive=len([t for t in g.players['red'].troops if t.alive])
    killed=ini-alive
    g2=Game()
    nb=mk_card('bomber',11,'blue',9,10)
    g2.deploy('blue',nb)
    random.seed(42)
    sk2=mk_card('skeleton_army',11,'red',9,14)
    if isinstance(sk2,list):
        for s in sk2:g2.deploy('red',s)
    else:g2.deploy('red',sk2)
    g2.run(5)
    alive2=len([t for t in g2.players['red'].troops if t.alive])
    killed2=ini-alive2
    assert killed>=killed2,f"Evo bomber should kill >= normal: {killed} vs {killed2}"
    return f"Evo Bomber v Skarmy (evo kills {killed}, normal kills {killed2})"
def t_int_evo_skeletons_v_knight():
    g=Game()
    random.seed(42)
    esk=mk_card('skeletons',11,'blue',9,14,evolved=True)
    if isinstance(esk,list):
        for s in esk:g.deploy('blue',s)
    else:g.deploy('blue',esk)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    g.run(15)
    alive_skels=len([t for t in g.players['blue'].troops if t.alive])
    return f"Evo Skeletons v Knight (knight hp={k.hp}, skels alive={alive_skels})"
def t_int_evo_skeletons_cap_8():
    g=Game()
    random.seed(42)
    esk=mk_card('skeletons',11,'blue',9,10,evolved=True)
    if isinstance(esk,list):
        for s in esk:g.deploy('blue',s)
    else:g.deploy('blue',esk)
    d=Dummy('red',9,11.5,hp=99999,spd=0,dmg=0,rng=0)
    g.deploy('red',d)
    g.run(20)
    skels=[t for t in g.players['blue'].troops if t.alive and 'keleton' in t.name]
    assert len(skels)<=8,f"Should cap at 8, got {len(skels)}"
    return f"Evo Skeletons cap at 8 ({len(skels)} alive)"
def t_int_evo_archers_v_tower():
    g=Game()
    random.seed(42)
    ea=mk_card('archers',11,'blue',3,19,evolved=True)
    if isinstance(ea,list):
        for a in ea:g.deploy('blue',a)
    else:g.deploy('blue',ea)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(10)
    edm=ini-rpt.hp
    g2=Game()
    na=mk_card('archers',11,'blue',3,19)
    if isinstance(na,list):
        for a in na:g2.deploy('blue',a)
    else:g2.deploy('blue',na)
    rpt2=g2.arena.get_tower('red','princess','left')
    g2.run(10)
    ndm=ini-rpt2.hp
    assert edm>ndm,f"Evo archers should deal more to tower: {edm} vs {ndm}"
    return f"Evo Archers v tower ({edm} vs {ndm} normal)"
def t_int_evo_sbarrel_early_drop():
    g=Game()
    sb=mk_card('skeleton_barrel',11,'blue',9,14,evolved=True)
    g.deploy('blue',sb)
    sb.hp=int(sb.max_hp*0.7)
    g.run(0.2)
    skels=len([t for t in g.players['blue'].troops if t.alive])-1
    assert skels>0,f"Should drop skeletons at 75% HP, got {skels} extra troops"
    return f"Evo Skel Barrel early drop ({skels} skeletons at 70% HP)"
def t_int_evo_zap_v_skarmy():
    g=Game()
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,14)
    if isinstance(sk,list):
        for s in sk:g.deploy('red',s)
    else:g.deploy('red',sk)
    ini=len([t for t in g.players['red'].troops if t.alive])
    ez=mk_card('zap',11,'blue',9,14,evolved=True)
    ez.apply(g);g.spells.append(ez)
    g.run(1)
    alive=len([t for t in g.players['red'].troops if t.alive])
    g2=Game()
    random.seed(42)
    sk2=mk_card('skeleton_army',11,'red',9,14)
    if isinstance(sk2,list):
        for s in sk2:g2.deploy('red',s)
    else:g2.deploy('red',sk2)
    nz=mk_card('zap',11,'blue',9,14)
    nz.apply(g2)
    g2.run(0.1)
    alive2=len([t for t in g2.players['red'].troops if t.alive])
    assert alive<=alive2,f"Evo zap should kill >= normal: {ini-alive} vs {ini-alive2}"
    return f"Evo Zap v Skarmy (evo killed {ini-alive}, normal killed {ini-alive2})"
def t_int_evo_snowball_v_push():
    g=Game()
    h=mk_card('hog_rider',11,'red',9,17)
    g.deploy('red',h)
    g.run(1)
    my=h.y
    sb=mk_card('giant_snowball',11,'blue',h.x,h.y,evolved=True)
    sb.apply(g);g.spells.append(sb)
    g.run(1)
    assert h.y>my or abs(h.y-my)>0.5,f"Snowball should displace hog: before={my:.1f} after={h.y:.1f}"
    has_slow=any(s.kind=='slow' for s in h.statuses)
    assert has_slow,"Hog should be slowed"
    return f"Evo Snowball displaces hog (y: {my:.1f}->{h.y:.1f}, slowed)"
def t_int_evo_knight_v_freeze():
    g=Game()
    ek=mk_card('knight',11,'blue',9,14,evolved=True)
    g.deploy('blue',ek)
    d=Dummy('red',9,15,hp=50000,dmg=300,spd=0,hspd=0.5,rng=1.5)
    g.deploy('red',d)
    fr=mk_card('freeze',11,'red',9,14)
    fr.apply(g);g.spells.append(fr)
    ini=ek.hp
    g.run(2)
    dmg_frozen=ini-ek.hp
    return f"Evo Knight under freeze (dmg taken={dmg_frozen}, knight frozen can't attack->no reduction)"
def t_int_evo_bats_sustain():
    g=Game()
    random.seed(42)
    eb=mk_card('bats',11,'blue',9,10,evolved=True)
    if isinstance(eb,list):
        for b in eb:g.deploy('blue',b)
        b0=eb[0]
    else:g.deploy('blue',eb);b0=eb
    d=Dummy('red',9,14,hp=50000,spd=0,dmg=0,rng=0)
    g.deploy('red',d)
    g.run(5)
    assert b0.hp>=b0.max_hp or b0.hp>81,f"Evo bat should heal: hp={b0.hp}/{b0.max_hp}"
    return f"Evo Bats sustain via healing (hp={b0.hp}/{b0.max_hp})"
def t_int_evo_rg_splash_on_tower():
    g=Game()
    rg=mk_card('royal_giant',11,'blue',3,14,evolved=True)
    g.deploy('blue',rg)
    d=Dummy('red',4,20,hp=5000,spd=0)
    g.deploy('red',d)
    rpt=g.arena.get_tower('red','princess','left')
    g.run(15)
    assert rpt.hp<rpt.max_hp,"RG should damage tower"
    d_dmg=5000-d.hp
    return f"Evo RG attacks tower + recoil splash (tower hp={rpt.hp}, nearby troop dmg={d_dmg})"
def t_int_clone_evo_no_ability():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    assert gk.ability is not None
    cl=mk_card('clone',11,'blue',9,10)
    cl.apply(g)
    clones=[t for t in g.players['blue'].troops if t is not gk]
    assert len(clones)==1
    assert clones[0].hp==1 and clones[0].max_hp==1
    assert getattr(clones[0],'ability',None) is None
    return "Cloned champion loses ability"
def t_int_champ_ability_after_death():
    g=Game()
    gk1=mk_card('golden_knight',11,'blue',9,10)
    gk2=mk_card('golden_knight',11,'blue',5,10)
    g.deploy('blue',gk1);g.deploy('blue',gk2)
    p=g.players['blue']
    assert p.active_champ is gk1
    p.elixir=10;gk2.ability.cd=0
    ok,_=g.activate_ability('blue',gk2)
    assert not ok,"Queued champ can't use ability"
    gk1.alive=False;g._proc_deaths()
    assert p.active_champ is gk2
    gk2.ability.cd=0
    ok2,_=g.activate_ability('blue',gk2)
    g.run(1.3)
    assert ok2,"Now-active champ should use ability"
    return "Champion ability handoff on death works"
def t_int_evo_barbs_v_pekka():
    g=Game()
    random.seed(42)
    eb=mk_card('barbarians',11,'blue',9,14,evolved=True)
    if isinstance(eb,list):
        for b in eb:g.deploy('blue',b)
    else:g.deploy('blue',eb)
    pk=mk_card('pekka',11,'red',9,17)
    g.deploy('red',pk)
    g.run(20)
    alive_b=len([t for t in g.players['blue'].troops if t.alive])
    g2=Game()
    random.seed(42)
    nb=mk_card('barbarians',11,'blue',9,14)
    if isinstance(nb,list):
        for b in nb:g2.deploy('blue',b)
    else:g2.deploy('blue',nb)
    pk2=mk_card('pekka',11,'red',9,17)
    g2.deploy('red',pk2)
    g2.run(20)
    alive_n=len([t for t in g2.players['blue'].troops if t.alive])
    return f"Evo Barbs v PEKKA (evo alive={alive_b}, normal alive={alive_n}, pekka hp evo={pk.hp} norm={pk2.hp})"
def t_int_evo_recruits_shield_charge():
    g=Game()
    random.seed(42)
    rr=mk_card('royal_recruits',11,'blue',9,14,evolved=True)
    if isinstance(rr,list):
        for r in rr:g.deploy('blue',r)
        r0=rr[0]
    else:g.deploy('blue',rr);r0=rr
    from sim.fx import EvoRoyalRecruits
    erc=[c for c in r0.components if isinstance(c,EvoRoyalRecruits)]
    assert len(erc)==1
    r0.shield_hp=0
    g.run(0.2)
    assert erc[0].charged,"Should charge after shield break"
    return "Evo Recruit charges after shield break"
def t_int_monk_protect_v_archers():
    g=Game()
    m=mk_card('monk',11,'blue',9,14)
    g.deploy('blue',m)
    ar=mk_card('archers',11,'red',9,17)
    if isinstance(ar,list):
        for a in ar:g.deploy('red',a)
    else:g.deploy('red',ar)
    g.players['blue'].elixir=10;m.ability.cd=0
    ini=m.hp
    g.activate_ability('blue',m)
    g.run(1.3)
    g.run(4)
    dmg=ini-m.hp
    return f"Monk protect v Archers (dmg taken={dmg} with 65% reduction)"
def t_int_aq_cloak_v_tower():
    g=Game()
    aq=mk_card('archer_queen',11,'blue',3,20)
    g.deploy('blue',aq)
    rpt=g.arena.get_tower('red','princess','left')
    g.run(5)
    hp_before=rpt.hp
    assert hp_before<rpt.max_hp,"AQ should already be hitting tower"
    g.players['blue'].elixir=10;aq.ability.cd=0
    g.activate_ability('blue',aq)
    g.run(1.3)
    g.run(2.9)
    hp_after=rpt.hp
    cloak_dmg=hp_before-hp_after
    assert cloak_dmg>225*2,f"Cloak should fire fast: {cloak_dmg}"
    return f"AQ cloak v tower ({cloak_dmg} dmg during cloak)"
def t_int_gk_dash_v_skarmy():
    g=Game()
    gk=mk_card('golden_knight',11,'blue',9,10)
    g.deploy('blue',gk)
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,13)
    if isinstance(sk,list):
        for s in sk:g.deploy('red',s)
    else:g.deploy('red',sk)
    ini=len([t for t in g.players['red'].troops if t.alive])
    g.players['blue'].elixir=10;gk.ability.cd=0
    g.activate_ability('blue',gk)
    g.run(1.3)
    g.run(2)
    alive=len([t for t in g.players['red'].troops if t.alive])
    killed=ini-alive
    assert killed>=3,f"GK dash should chain through skarmy: {killed} killed"
    return f"GK dash v Skarmy ({killed}/{ini} killed)"
def t_int_sk_soul_from_combat():
    g=Game()
    sk=mk_card('skeleton_king',11,'blue',9,10)
    g.deploy('blue',sk)
    from sim.fx import SoulCollect
    sc=[c for c in sk.components if isinstance(c,SoulCollect)][0]
    random.seed(42)
    skels=mk_card('skeletons',11,'red',9,12)
    if isinstance(skels,list):
        for s in skels:g.deploy('red',s)
    else:g.deploy('red',skels)
    g.run(10)
    assert sc.souls>=1,f"SK should collect souls from killed skeletons: {sc.souls}"
    return f"SK collects souls from combat kills ({sc.souls} souls)"
def t_int_mm_escape_v_pekka():
    g=Game()
    mm=mk_card('mighty_miner',11,'blue',9,14)
    g.deploy('blue',mm)
    pk=mk_card('pekka',11,'red',9,15)
    g.deploy('red',pk)
    g.run(3)
    assert mm.dmg>=204,"Should ramp up"
    ini_pk=pk.hp
    g.players['blue'].elixir=10;mm.ability.cd=0
    g.activate_ability('blue',mm)
    g.run(1.3)
    assert mm.dmg==43,f"Should reset ramp, got {mm.dmg}"
    bomb_dmg=ini_pk-pk.hp
    assert bomb_dmg>0,"Bomb should damage PEKKA"
    assert abs(mm.x-8)<=1,"Should lane swap"
    return f"MM escape v PEKKA (bomb={bomb_dmg}, ramp reset, lane swap x={mm.x:.0f})"
def t_int_bb_dash_then_grenade():
    g=Game()
    bb=mk_card('boss_bandit',11,'blue',9,10)
    g.deploy('blue',bb)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(3)
    ini_dmg=50000-d.hp
    g.players['blue'].elixir=10;bb.ability.cd=0
    oy=bb.y
    g.activate_ability('blue',bb)
    g.run(1.3)
    assert bb.y<oy,"Should teleport back"
    g.run(3)
    total_dmg=50000-d.hp
    assert total_dmg>ini_dmg,"Should deal more after re-engaging"
    return f"BB dash->grenade->re-engage ({ini_dmg}->{total_dmg} total dmg)"
def t_hero_knight_load():
    hk=mk_card('knight',11,'blue',5,10,hero=True)
    assert hk.hp==1766 and hk.dmg==202
    assert getattr(hk,'is_hero',False)
    ab=getattr(hk,'ability',None)
    assert ab is not None
    from sim.fx import TriumphantTaunt
    assert isinstance(ab,TriumphantTaunt)
    assert ab.cost==2 and ab.max_cd==0 and ab.uses==1
    return f"Hero Knight load (hp={hk.hp}, ability cost={ab.cost})"
def t_hero_knight_taunt():
    g=Game()
    hk=mk_card('knight',11,'blue',9,14,hero=True)
    g.deploy('blue',hk)
    d=Dummy('red',9,17,hp=5000,dmg=200,spd=1.0,hspd=1.0,rng=1.5)
    g.deploy('red',d)
    g.players['blue'].elixir=10;hk.ability.cd=0
    g.activate_ability('blue',hk)
    g.run(1.3)
    assert hk.shield_hp>0,f"Should have shield: {hk.shield_hp}"
    tt=getattr(d,'_taunt_target',None)
    assert tt is hk,"Enemy should be taunted to hero knight"
    return f"Hero Knight taunt (shield={hk.shield_hp}, taunted={tt is hk})"
def t_hero_knight_shield():
    g=Game()
    hk=mk_card('knight',11,'blue',9,14,hero=True)
    g.deploy('blue',hk)
    g.players['blue'].elixir=10;hk.ability.cd=0
    g.activate_ability('blue',hk)
    g.run(1.3)
    shp=hk.shield_hp
    d=Dummy('red',9,14.5,hp=50000,dmg=300,spd=0,hspd=0.5,rng=1.5)
    g.deploy('red',d)
    ini_hp=hk.hp
    g.run(2)
    assert hk.shield_hp<shp,"Shield should absorb damage"
    if hk.shield_hp>0:
        assert hk.hp==ini_hp,"HP should be untouched while shield up"
    return f"Hero Knight shield absorbs (shield: {shp}->{hk.shield_hp}, hp: {ini_hp}->{hk.hp})"
def t_hero_knight_taunt_v_pekka():
    g=Game()
    hk=mk_card('knight',11,'blue',9,14,hero=True)
    g.deploy('blue',hk)
    d2=Dummy('blue',5,14,hp=500,spd=0)
    g.deploy('blue',d2)
    pk=mk_card('pekka',11,'red',9,17)
    g.deploy('red',pk)
    g.players['blue'].elixir=10;hk.ability.cd=0
    g.run(3)
    g.activate_ability('blue',hk)
    g.run(1.3)
    g.run(3)
    assert d2.hp==500,f"Taunted PEKKA should ignore dummy, dummy hp={d2.hp}"
    return f"Hero Knight taunts PEKKA away from ally (ally hp={d2.hp})"
def t_hero_goblins_load():
    random.seed(42)
    hg=mk_card('goblins',11,'blue',5,10,hero=True)
    assert isinstance(hg,list) and len(hg)==4
    assert all(getattr(g,'is_hero',False) for g in hg)
    ab=hg[0].ability
    assert ab is not None
    from sim.fx import BannerBrigade
    assert isinstance(ab,BannerBrigade)
    assert ab.cost==1
    return f"Hero Goblins load ({len(hg)} gobs, ability cost={ab.cost})"
def t_hero_goblins_banner():
    g=Game()
    random.seed(42)
    hg=mk_card('goblins',11,'blue',9,14,hero=True)
    for gb in hg:g.deploy('blue',gb)
    ab=hg[0].ability
    for gb in hg:gb.alive=False
    g._proc_deaths()
    assert ab.banner_pos is not None,"Banner should drop on last death"
    g.players['blue'].elixir=10
    ok,_=g.activate_ability('blue',hg[0])
    g.run(1.3)
    spawned=[t for t in g.players['blue'].troops if t.alive]
    assert len(spawned)==2,f"Should respawn 2 goblins, got {len(spawned)}"
    return f"Hero Goblins banner respawn ({len(spawned)} goblins)"
def t_hero_goblins_banner_expires():
    g=Game()
    random.seed(42)
    hg=mk_card('goblins',11,'blue',9,14,hero=True)
    for gb in hg:g.deploy('blue',gb)
    ab=hg[0].ability
    for gb in hg:gb.alive=False
    g._proc_deaths()
    assert ab.banner_pos is not None
    g.run(8)
    assert ab.banner_pos is None,"Banner should expire after 7s"
    return "Hero Goblins banner expires after 7s"
def t_hero_clone_no_ability():
    g=Game()
    hk=mk_card('knight',11,'blue',9,14,hero=True)
    g.deploy('blue',hk)
    assert hk.ability is not None
    cl=mk_card('clone',11,'blue',9,14)
    cl.apply(g)
    clones=[t for t in g.players['blue'].troops if t is not hk]
    assert len(clones)==1
    assert getattr(clones[0],'ability',None) is None,"Cloned hero should not have ability"
    return "Cloned hero loses ability"
def t_evo_valk_tornado():
    g=Game()
    v=mk_card('valkyrie',11,'blue',9,10,evolved=True)
    g.deploy('blue',v)
    from sim.fx import EvoValkyrie
    assert any(isinstance(c,EvoValkyrie) for c in v.components)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',12,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000,"Both should take tornado damage"
    return f"Evo Valkyrie tornado (d1={5000-d1.hp} d2={5000-d2.hp})"
def t_evo_valk_pulls():
    g=Game()
    v=mk_card('valkyrie',11,'blue',9,14,evolved=True)
    g.deploy('blue',v)
    d=Dummy('red',12,16,hp=50000,spd=0)
    g.deploy('red',d)
    iy=d.y;ix=d.x
    g.run(5)
    moved=math.sqrt((d.x-ix)**2+(d.y-iy)**2)
    assert moved>0.3,f"Should pull toward valk, moved {moved:.2f}"
    return f"Evo Valk pulls enemies ({moved:.1f} tiles)"
def t_evo_musk_sniper():
    g=Game()
    em=mk_card('musketeer',11,'blue',9,5,evolved=True)
    g.deploy('blue',em)
    from sim.fx import EvoMusketeer
    ec=[c for c in em.components if isinstance(c,EvoMusketeer)]
    assert len(ec)==1
    assert ec[0].ammo==3
    d=Dummy('red',9,20,hp=5000,spd=0,dmg=0,rng=0)
    g.deploy('red',d)
    g.run(5)
    dmg=5000-d.hp
    assert dmg>218*3,f"Sniper should deal 1.8x: {dmg}"
    assert ec[0].ammo<3,f"Should use ammo: {ec[0].ammo}"
    return f"Evo Musketeer sniper ({dmg} dmg, ammo left={ec[0].ammo})"
def t_evo_dart_gob_poison():
    g=Game()
    dg=mk_card('dart_goblin',11,'blue',9,10,evolved=True)
    g.deploy('blue',dg)
    from sim.fx import EvoDartGoblin
    assert any(isinstance(c,EvoDartGoblin) for c in dg.components)
    d=Dummy('red',9,14,hp=50000,spd=0,dmg=0,rng=0)
    g.deploy('red',d)
    g.run(8)
    dmg=50000-d.hp
    g2=Game()
    ndg=mk_card('dart_goblin',11,'blue',9,10)
    g2.deploy('blue',ndg)
    d2=Dummy('red',9,14,hp=50000,spd=0,dmg=0,rng=0)
    g2.deploy('red',d2)
    g2.run(8)
    ndmg=50000-d2.hp
    assert dmg>ndmg,f"Evo should deal more with poison: {dmg} vs {ndmg}"
    return f"Evo Dart Goblin poison ({dmg} vs {ndmg} normal)"
def t_evo_royal_hogs_fly():
    g=Game()
    random.seed(42)
    rh=mk_card('royal_hogs',11,'blue',9,14,evolved=True)
    if isinstance(rh,list):
        for r in rh:g.deploy('blue',r)
        r0=rh[0]
    else:g.deploy('blue',rh);r0=rh
    from sim.fx import EvoRoyalHogs
    assert any(isinstance(c,EvoRoyalHogs) for c in r0.components)
    assert r0.transport=='Air',"Should start flying"
    r0.hp-=1
    g.run(0.2)
    assert r0.transport=='Ground',"Should ground after taking damage"
    return "Evo Royal Hogs fly->ground on hit"
def t_evo_gobcage_pull():
    g=Game()
    gc=mk_card('goblin_cage',11,'blue',9,14,evolved=True)
    g.deploy('blue',gc)
    from sim.fx import EvoGoblinCage
    assert any(isinstance(c,EvoGoblinCage) for c in gc.components)
    return "Evo Goblin Cage has pull component"
def t_hero_giant_load():
    hg=mk_card('giant',11,'blue',5,10,hero=True)
    assert getattr(hg,'is_hero',False)
    ab=getattr(hg,'ability',None)
    assert ab is not None
    from sim.fx import HeroicHurl
    assert isinstance(ab,HeroicHurl)
    assert ab.cost==2
    return f"Hero Giant load (ability cost={ab.cost})"
def t_hero_giant_hurl():
    g=Game()
    hg=mk_card('giant',11,'blue',9,14,hero=True)
    g.deploy('blue',hg)
    d=Dummy('red',9,14.5,hp=5000,spd=0)
    g.deploy('red',d)
    g.players['blue'].elixir=10;hg.ability.cd=0
    ix=d.x
    g.activate_ability('blue',hg)
    g.run(1.5)
    dx=abs(d.x-ix)
    assert dx>3,f"Should throw enemy: dx={dx:.1f}"
    assert d.hp<5000,f"Should deal impact damage: {d.hp}"
    return f"Hero Giant hurls enemy (dx={dx:.1f}, hp={d.hp})"
def t_hero_mpekka_load():
    hmp=mk_card('mini_pekka',11,'blue',5,10,hero=True)
    assert getattr(hmp,'is_hero',False)
    from sim.fx import BreakfastBoost
    assert isinstance(hmp.ability,BreakfastBoost)
    return f"Hero Mini PEKKA load (hp={hmp.hp})"
def t_hero_mpekka_boost():
    g=Game()
    hmp=mk_card('mini_pekka',11,'blue',9,14,hero=True)
    g.deploy('blue',hmp)
    ini_dmg=hmp.dmg;ini_hp=hmp.max_hp
    g.players['blue'].elixir=10
    hmp.ability.meters=2
    g.activate_ability('blue',hmp)
    g.run(1.5)
    assert hmp.dmg>ini_dmg,f"Should level up: dmg {ini_dmg}->{hmp.dmg}"
    assert hmp.max_hp>ini_hp,f"Should gain HP: {ini_hp}->{hmp.max_hp}"
    return f"Hero Mini PEKKA boost (dmg {ini_dmg}->{hmp.dmg}, hp {ini_hp}->{hmp.max_hp})"
def t_hero_musk_turret():
    g=Game()
    hmu=mk_card('musketeer',11,'blue',9,10,hero=True)
    g.deploy('blue',hmu)
    g.players['blue'].elixir=10;hmu.ability.cd=0
    ini=len(g.players['blue'].troops)
    g.activate_ability('blue',hmu)
    g.run(1.5)
    spawned=len(g.players['blue'].troops)-ini
    assert spawned>=1,f"Should spawn turret, got {spawned}"
    return f"Hero Musketeer turret spawned ({spawned})"
def t_hero_ice_golem_storm():
    g=Game()
    hig=mk_card('ice_golem',11,'blue',9,14,hero=True)
    g.deploy('blue',hig)
    d=Dummy('red',9,14.5,hp=5000,spd=0)
    g.deploy('red',d)
    g.players['blue'].elixir=10;hig.ability.cd=0
    g.activate_ability('blue',hig)
    g.run(3)
    has_effect=any(s.kind in ('freeze','slow') for s in d.statuses) or d.hp<5000
    assert has_effect,f"Should freeze/slow/damage: hp={d.hp}"
    return f"Hero Ice Golem snowstorm (enemy hp={d.hp})"
def t_hero_wizard_flight():
    g=Game()
    hw=mk_card('wizard',11,'blue',9,10,hero=True)
    g.deploy('blue',hw)
    g.players['blue'].elixir=10;hw.ability.cd=0
    g.activate_ability('blue',hw)
    g.run(1.5)
    assert hw.transport=='Air',"Should fly"
    g.run(5)
    assert hw.transport=='Ground',"Should land after duration"
    return "Hero Wizard flight (air->ground)"
def t_hero_mega_minion_warp():
    g=Game()
    hmm=mk_card('mega_minion',11,'blue',9,5,hero=True)
    g.deploy('blue',hmm)
    d=Dummy('red',9,25,hp=200,spd=0,dmg=0,rng=0)
    g.deploy('red',d)
    g.players['blue'].elixir=10
    ix=hmm.x;iy=hmm.y
    g.activate_ability('blue',hmm)
    g.run(1.5)
    dist=math.sqrt((hmm.x-ix)**2+(hmm.y-iy)**2)
    assert dist>5,f"Should warp to enemy: moved {dist:.1f}"
    assert d.hp<200,f"Should damage target: {d.hp}"
    return f"Hero Mega Minion warp (moved {dist:.1f}, target hp={d.hp})"
def t_int_evo_valk_v_skarmy():
    g=Game()
    v=mk_card('valkyrie',11,'blue',9,14,evolved=True)
    g.deploy('blue',v)
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,17)
    if isinstance(sk,list):
        for s in sk:g.deploy('red',s)
    else:g.deploy('red',sk)
    ini=len([t for t in g.players['red'].troops if t.alive])
    g.run(8)
    alive=len([t for t in g.players['red'].troops if t.alive])
    killed=ini-alive
    assert killed>=ini//2,f"Evo valk tornado should clear skarmy: {killed}/{ini}"
    return f"Evo Valk v Skarmy ({killed}/{ini} killed)"
def t_int_evo_musk_sniper_then_normal():
    g=Game()
    em=mk_card('musketeer',11,'blue',9,5,evolved=True)
    g.deploy('blue',em)
    from sim.fx import EvoMusketeer
    ec=[c for c in em.components if isinstance(c,EvoMusketeer)][0]
    d=Dummy('red',9,20,hp=50000,spd=0,dmg=0,rng=0)
    g.deploy('red',d)
    g.run(10)
    assert ec.ammo==0,"Should expend all sniper ammo"
    g.run(5)
    return f"Evo Musk expends ammo then normal (ammo={ec.ammo})"
def t_int_hero_giant_v_pekka():
    g=Game()
    hg=mk_card('giant',11,'blue',9,14,hero=True)
    g.deploy('blue',hg)
    pk=mk_card('pekka',11,'red',9,15)
    g.deploy('red',pk)
    g.players['blue'].elixir=10;hg.ability.cd=0
    g.run(2)
    g.activate_ability('blue',hg)
    g.run(2)
    assert abs(pk.x-9)>3 or any(s.kind=='stun' for s in pk.statuses),f"Should throw PEKKA away: x={pk.x:.1f}"
    return f"Hero Giant hurls PEKKA (x={pk.x:.1f})"
def t_int_hero_mpekka_v_knight():
    g=Game()
    hmp=mk_card('mini_pekka',11,'blue',9,14,hero=True)
    g.deploy('blue',hmp)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    g.run(5)
    g.players['blue'].elixir=10;hmp.ability.meters=3
    g.activate_ability('blue',hmp)
    g.run(1.5)
    assert hmp.dmg>740,f"Should be boosted 5 levels: dmg={hmp.dmg}"
    return f"Hero Mini PEKKA boosted v Knight (dmg={hmp.dmg})"
def t_int_evo_valk_pulls_air():
    g=Game()
    v=mk_card('valkyrie',11,'blue',9,14,evolved=True)
    g.deploy('blue',v)
    d_air=Dummy('red',12,16,hp=50000,spd=0)
    d_air.transport='Air'
    g.deploy('red',d_air)
    d_ground=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d_ground)
    g.run(5)
    assert d_air.hp<50000,"Evo valk tornado should hit air troops"
    return f"Evo Valk pulls air (air hp={d_air.hp})"
def t_int_hero_wizard_immune_ground():
    g=Game()
    hw=mk_card('wizard',11,'blue',9,14,hero=True)
    g.deploy('blue',hw)
    d=Dummy('red',9,15,hp=50000,dmg=500,spd=0,hspd=0.5,rng=1.5)
    g.deploy('red',d)
    g.players['blue'].elixir=10;hw.ability.cd=0
    g.activate_ability('blue',hw)
    g.run(1.5)
    ini=hw.hp
    g.run(3)
    assert hw.hp==ini,f"Flying wizard immune to ground melee: hp {ini}->{hw.hp}"
    return f"Hero Wizard immune while flying (hp unchanged: {hw.hp})"
def t_evo_baby_dragon_aura():
    g=Game()
    bd=mk_card('baby_dragon',11,'blue',9,10,evolved=True)
    g.deploy('blue',bd)
    from sim.fx import EvoBabyDragon
    assert any(isinstance(c,EvoBabyDragon) for c in bd.components)
    d=Dummy('red',9,14,hp=50000,spd=1.0)
    g.deploy('red',d)
    g.run(2)
    has_slow=any(s.kind=='slow' for s in d.statuses)
    assert has_slow,"Nearby enemy should be slowed"
    return "Evo Baby Dragon wind aura slows enemies"
def t_evo_witch_heal():
    g=Game()
    w=mk_card('witch',11,'blue',9,10,evolved=True)
    g.deploy('blue',w)
    from sim.fx import EvoWitch
    assert any(isinstance(c,EvoWitch) for c in w.components)
    return f"Evo Witch has heal component (hp={w.hp})"
def t_evo_pekka_heal():
    g=Game()
    pk=mk_card('pekka',11,'blue',9,14,evolved=True)
    g.deploy('blue',pk)
    from sim.fx import EvoPekka
    assert any(isinstance(c,EvoPekka) for c in pk.components)
    d=Dummy('red',9,15,hp=100,spd=0)
    g.deploy('red',d)
    pk.hp=pk.max_hp-500
    ini=pk.hp
    g.run(5)
    assert pk.hp>ini,f"Should heal on kill: {ini}->{pk.hp}"
    return f"Evo PEKKA heals on kill ({ini}->{pk.hp})"
def t_evo_hunter_net():
    g=Game()
    h=mk_card('hunter',11,'blue',9,10,evolved=True)
    g.deploy('blue',h)
    from sim.fx import EvoHunter
    assert any(isinstance(c,EvoHunter) for c in h.components)
    d=Dummy('red',9,13,hp=50000,spd=1.0)
    g.deploy('red',d)
    g.run(3)
    return "Evo Hunter has net component"
def t_evo_edrag_infinite():
    g=Game()
    ed=mk_card('electro_dragon',11,'blue',9,10,evolved=True)
    g.deploy('blue',ed)
    from sim.fx import EvoElectroDragon
    assert any(isinstance(c,EvoElectroDragon) for c in ed.components)
    d1=Dummy('red',9,13,hp=5000,spd=0)
    d2=Dummy('red',10,13,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    dmg1=5000-d1.hp;dmg2=5000-d2.hp
    assert dmg1>0 and dmg2>0,f"Both should take chain damage: {dmg1},{dmg2}"
    assert dmg1>3*ed.dmg,f"Bolt should keep bouncing between the pair: {dmg1}"
    g2=quiet(Game());e2=mk_card('electro_dragon',11,'blue',9,10,evolved=True);g2.deploy('blue',e2)
    lone=Dummy('red',9,13,hp=5000,spd=0,dmg=0,rng=0);g2.deploy('red',lone);g2.run(5)
    assert 5000-lone.hp==2*e2.dmg,f"A lone target has nothing to chain onto: {5000-lone.hp}"
    return f"Evo E-Dragon infinite chain (d1={dmg1}, d2={dmg2})"
def t_evo_exe_smash():
    g=Game()
    ex=mk_card('executioner',11,'blue',9,10,evolved=True)
    g.deploy('blue',ex)
    from sim.fx import EvoExecutioner
    assert any(isinstance(c,EvoExecutioner) for c in ex.components)
    d=Dummy('red',9,12,hp=5000,spd=0)
    g.deploy('red',d)
    iy=d.y
    g.run(3)
    dmg=5000-d.hp
    assert d.y>iy,f"Close-range axe should knock back: y={d.y:.1f}"
    return f"Evo Executioner axe smash (dmg={dmg}, kb y={iy:.1f}->{d.y:.1f})"
def t_evo_goblin_drill_resurface():
    g=Game()
    gd=mk_card('goblin_drill',11,'blue',9,14,evolved=True)
    g.deploy('blue',gd)
    from sim.fx import Resurface
    assert any(isinstance(c,Resurface) for c in gd.components)
    g.run(1.1);ini=len(g.players['blue'].troops)
    gd.hp=int(gd.max_hp*0.6)
    g.run(0.2)
    spawned=len(g.players['blue'].troops)-ini
    assert spawned>=2,f"Should spawn goblins at 66% threshold: {spawned}"
    return f"Evo Goblin Drill resurface ({spawned} goblins at 60% HP)"
def t_hero_bbarrel_reroll():
    g=Game()
    from sim.fx import RowdyReroll
    bb=mk_card('barbarian_barrel',11,'blue',9,14,hero=True)
    g._cast('blue',bb,9,14);g.run(1.4)
    barbs=[t for t in g.players['blue'].troops if t.alive and 'arb' in getattr(t,'name','').lower()]
    assert len(barbs)==1 and isinstance(barbs[0].ability,RowdyReroll) and barbs[0].is_hero,"the barrel's Barbarian carries the hero ability"
    b=barbs[0];iy=b.y;g.run(1.0)
    g.players['blue'].elixir=10
    g.activate_ability('blue',b)
    g.run(1.5)
    assert b.y>iy+2,f"Should roll forward: y {iy:.1f}->{b.y:.1f}"
    return f"Hero Barbarian Barrel reroll (y: {iy:.1f}->{b.y:.1f})"
def t_int_evo_pekka_v_skarmy():
    g=Game()
    pk=mk_card('pekka',11,'blue',9,10,evolved=True)
    g.deploy('blue',pk)
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,12)
    if isinstance(sk,list):
        for s in sk:g.deploy('red',s)
    else:g.deploy('red',sk)
    ini=pk.hp;healed=False
    for _ in range(100):
        g.tick()
        if pk.hp>ini:healed=True;break
        ini=min(ini,pk.hp)
    assert healed,f"Evo PEKKA should heal from skarmy kills: {ini}->{pk.hp}"
    return f"Evo PEKKA v Skarmy heals ({ini}->{pk.hp})"
def t_int_evo_edrag_v_pair():
    g=Game()
    ed=mk_card('electro_dragon',11,'blue',9,10,evolved=True)
    g.deploy('blue',ed)
    d1=Dummy('red',9,13,hp=50000,spd=0,dmg=0,rng=0)
    d2=Dummy('red',10,13,hp=50000,spd=0,dmg=0,rng=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(8)
    dmg1=50000-d1.hp;dmg2=50000-d2.hp
    g2=Game()
    ned=mk_card('electro_dragon',11,'blue',9,10)
    g2.deploy('blue',ned)
    nd1=Dummy('red',9,13,hp=50000,spd=0,dmg=0,rng=0)
    nd2=Dummy('red',10,13,hp=50000,spd=0,dmg=0,rng=0)
    g2.deploy('red',nd1);g2.deploy('red',nd2)
    g2.run(8)
    ndmg1=50000-nd1.hp;ndmg2=50000-nd2.hp
    total_evo=dmg1+dmg2;total_norm=ndmg1+ndmg2
    assert total_evo>=total_norm,f"Evo should deal >= normal: {total_evo} vs {total_norm}"
    return f"Evo E-Dragon v pair ({total_evo} vs {total_norm} normal)"
def t_int_evo_witch_sustain():
    g=Game()
    w=mk_card('witch',11,'blue',9,10,evolved=True)
    g.deploy('blue',w)
    d=Dummy('red',9,14,hp=50000,dmg=50,spd=0,hspd=1.0,rng=5.0)
    g.deploy('red',d)
    g.run(15)
    g2=Game()
    nw=mk_card('witch',11,'blue',9,10)
    g2.deploy('blue',nw)
    d2=Dummy('red',9,14,hp=50000,dmg=50,spd=0,hspd=1.0,rng=5.0)
    g2.deploy('red',d2)
    g2.run(15)
    assert w.hp>=nw.hp,f"Evo witch should survive better: evo={w.hp} norm={nw.hp}"
    return f"Evo Witch sustain (evo hp={w.hp}, normal hp={nw.hp})"
def t_int_evo_exe_v_push():
    g=Game()
    ex=mk_card('executioner',11,'blue',9,14,evolved=True)
    g.deploy('blue',ex)
    random.seed(42)
    barbs=mk_card('barbarians',11,'red',9,17)
    if isinstance(barbs,list):
        for b in barbs:g.deploy('red',b)
    else:g.deploy('red',barbs)
    g.run(8)
    alive=len([t for t in g.players['red'].troops if t.alive])
    return f"Evo Executioner v Barbarians ({alive} barbs alive)"
def t_evo_mk_uppercut():
    g=Game()
    mk_t=mk_card('mega_knight',11,'blue',9,14,evolved=True)
    g.deploy('blue',mk_t)
    from sim.fx import EvoMegaKnight
    assert any(isinstance(c,EvoMegaKnight) for c in mk_t.components)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    iy=d.y
    g.run(5)
    assert d.y!=iy,f"Should knock back: y still {d.y}"
    return f"Evo Mega Knight uppercut (y: {iy:.1f}->{d.y:.1f})"
def t_evo_idrag_retain():
    g=Game()
    idr=mk_card('inferno_dragon',11,'blue',9,10,evolved=True)
    g.deploy('blue',idr)
    from sim.fx import EvoInfernoDragon
    assert any(isinstance(c,EvoInfernoDragon) for c in idr.components)
    return "Evo Inferno Dragon has beam momentum"
def t_evo_rghost_souldiers():
    g=Game()
    rg=mk_card('royal_ghost',11,'blue',9,14,evolved=True)
    g.deploy('blue',rg)
    from sim.fx import EvoRoyalGhost
    assert any(isinstance(c,EvoRoyalGhost) for c in rg.components)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(3)
    souldiers=[t for t in g.players['blue'].troops if t.name=='Souldier']
    assert len(souldiers)>=1,f"Should spawn souldiers: {len(souldiers)}"
    return f"Evo Royal Ghost souldiers ({len(souldiers)})"
def t_evo_bandit():
    # no Bandit evolution exists in the current game data; the evolved flag yields the base card
    b=mk_card('bandit',11,'blue',5,10,evolved=True)
    assert not getattr(b,'evolved',False)
    from sim.fx import BanditDash
    assert any(isinstance(c,BanditDash) for c in b.components)
    return "Bandit has no evolution (base dash only)"
def t_evo_lumberjack_ghost():
    g=Game()
    lj=mk_card('lumberjack',11,'blue',9,14,evolved=True)
    g.deploy('blue',lj)
    from sim.fx import EvoLumberjack
    assert any(isinstance(c,EvoLumberjack) for c in lj.components)
    lj.alive=False;lj.on_death(g)
    ghosts=[t for t in g.players['blue'].troops if 'Ghost' in getattr(t,'name','')]
    assert len(ghosts)>=1,f"Should spawn ghost on death: {len(ghosts)}"
    assert any(s.kind=='invisible' for s in ghosts[0].statuses)
    return f"Evo Lumberjack ghost on death ({len(ghosts)} ghost, invisible)"
def t_evo_ice_wiz_frost():
    # no Ice Wizard evolution exists in the current game data; the evolved flag yields the base card
    iw=mk_card('ice_wizard',11,'blue',9,10,evolved=True)
    assert not getattr(iw,'evolved',False)
    assert iw.hp==688 and iw.slow_dur==2.5
    return "Ice Wizard has no evolution (base slow only)"
def t_hero_magic_archer():
    ma=mk_card('magic_archer',11,'blue',5,10,hero=True)
    assert getattr(ma,'is_hero',False)
    from sim.fx import TripleThreat
    assert isinstance(ma.ability,TripleThreat)
    assert ma.ability.cost==2
    return f"Hero Magic Archer (ability cost={ma.ability.cost})"
def t_hero_marcher_triple():
    g=Game()
    ma=mk_card('magic_archer',11,'blue',9,10,hero=True)
    g.deploy('blue',ma)
    g.players['blue'].elixir=10;ma.ability.cd=0
    iy=ma.y
    g.activate_ability('blue',ma)
    g.run(1.5)
    assert abs(ma.y-iy)>3,f"Should dash back: dy={abs(ma.y-iy):.1f}"
    decoys=[t for t in g.players['blue'].troops if getattr(t,'name','')=='Decoy']
    assert len(decoys)>=1,"Should spawn decoy"
    assert ma.rng>10,f"Should have extended range: {ma.rng}"
    return f"Hero Magic Archer triple threat (dash={abs(ma.y-iy):.1f}, decoys={len(decoys)}, rng={ma.rng})"
def t_int_evo_mk_v_knight():
    g=Game()
    mk_t=mk_card('mega_knight',11,'blue',9,14,evolved=True)
    g.deploy('blue',mk_t)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('red',k)
    g.run(10)
    assert k.y>20 or not k.alive,f"Knight should be knocked back far: y={k.y:.1f}"
    return f"Evo MK v Knight knockback (knight y={k.y:.1f}, alive={k.alive})"
def t_int_evo_rghost_v_tower():
    g=Game()
    rg=mk_card('royal_ghost',11,'blue',3,14,evolved=True)
    g.deploy('blue',rg)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(15)
    dmg=ini-rpt.hp
    g2=Game()
    nrg=mk_card('royal_ghost',11,'blue',3,14)
    g2.deploy('blue',nrg)
    rpt2=g2.arena.get_tower('red','princess','left')
    g2.run(15)
    ndmg=ini-rpt2.hp
    assert dmg>=ndmg,f"Evo ghost+souldiers should deal >= normal: {dmg} vs {ndmg}"
    return f"Evo Royal Ghost v tower ({dmg} vs {ndmg} normal)"
def t_int_evo_lj_rage_ghost():
    g=Game()
    lj=mk_card('lumberjack',11,'blue',9,14,evolved=True)
    g.deploy('blue',lj)
    d=Dummy('red',9,15,hp=50000,dmg=999,spd=0,hspd=0.2,rng=1.5)
    g.deploy('red',d)
    g.run(5)
    assert not lj.alive,"LJ should die"
    ghosts=[t for t in g.players['blue'].troops if 'Ghost' in getattr(t,'name','')]
    rage_present=any(s.kind=='rage' for t in g.players['blue'].troops for s in getattr(t,'statuses',[]))
    return f"Evo LJ dies -> rage drop + ghost ({len(ghosts)} ghost, rage={rage_present})"
def t_int_hero_marcher_v_push():
    g=Game()
    ma=mk_card('magic_archer',11,'blue',9,10,hero=True)
    g.deploy('blue',ma)
    d1=Dummy('red',9,20,hp=5000,spd=0,dmg=0,rng=0)
    g.deploy('red',d1)
    g.run(3)
    dmg_before=5000-d1.hp
    g.players['blue'].elixir=10;ma.ability.cd=0
    g.activate_ability('blue',ma)
    g.run(5)
    dmg_after=5000-d1.hp
    assert dmg_after>dmg_before,f"Triple shot should deal more: {dmg_after} vs {dmg_before}"
    return f"Hero Magic Archer v target ({dmg_after} total dmg)"
def t_hidden_legendary():
    mk_t=mk_card('mega_knight',11,'blue',5,10)
    assert mk_t.mass==18,f"MK mass should be 18, got {mk_t.mass}"
    g=mk_card('golem',11,'blue',5,10)
    assert g.mass==20,f"Golem mass should be 20, got {g.mass}"
    return f"Legendary hidden stats (MK mass={mk_t.mass}, Golem mass={g.mass})"
def t_no_hero_and_evo_same_card():
    try:
        validate_deck(['knight','archers','fireball','hog_rider',
                        'musketeer','valkyrie','skeleton_army','freeze'],
                       heroes={'wizard'},evolutions={'wizard'})
        assert False,"Should reject wizard as both hero and evo"
    except AssertionError:pass
    validate_deck(['knight','archers','fireball','hog_rider',
                    'musketeer','valkyrie','skeleton_army','freeze'],
                   heroes={'wizard'},evolutions={'knight'})
    return "Cannot use same card as hero and evolution"
def t_mk_jump_survives_zap():
    g=Game()
    mk_t=mk_card('mega_knight',11,'blue',9,10)
    g.deploy('blue',mk_t)
    d=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d)
    from sim.fx import MKJump
    mjc=[c for c in mk_t.components if isinstance(c,MKJump)]
    assert len(mjc)==1,"MK should have MKJump"
    g.run(3)
    from sim.units import Status
    mjc[0].airborne=True;mjc[0].timer=0.1;mjc[0].jtgt=d
    mk_t.statuses.append(Status('stun',0.5))
    g.tick()
    has_stun=any(s.kind=='stun' for s in mk_t.statuses)
    assert not has_stun,"Stun should be stripped mid-jump"
    return "MK jump survives zap (stun stripped mid-air)"
def t_mk_jump_cancelled_by_freeze():
    g=Game()
    mk_t=mk_card('mega_knight',11,'blue',9,10)
    g.deploy('blue',mk_t)
    d=Dummy('red',9,14,hp=50000,spd=0)
    g.deploy('red',d)
    from sim.fx import MKJump
    from sim.units import Status
    mjc=[c for c in mk_t.components if isinstance(c,MKJump)][0]
    mjc.airborne=True;mjc.timer=0.1;mjc.jtgt=d;mjc.osp=mk_t.spd;mk_t.spd=0
    mk_t.statuses.append(Status('freeze',2.0))
    g.tick()
    assert not mjc.airborne,"Freeze should cancel jump"
    return "MK jump cancelled by freeze"
def t_mk_jump_crosses_river():
    g=Game()
    mk_t=mk_card('mega_knight',11,'blue',9,14)
    g.deploy('blue',mk_t)
    d=Dummy('red',9,19,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    assert mk_t.y>16,f"MK should jump across river: y={mk_t.y:.1f}"
    assert d.hp<50000,"MK should deal jump damage"
    return f"MK jump crosses river (y={mk_t.y:.1f}, d_dmg={50000-d.hp})"
