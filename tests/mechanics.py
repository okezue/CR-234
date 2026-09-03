import random
from sim.game import Game
from sim.cards import create as mk_card,card
from sim.units import has
from sim import fx
from tests.util import Dummy,quiet

def _game(towers=False):
    g=Game()
    if not towers:
        for t in g.arena.towers:t.alive=False
    return g
def _dummies(g,*xy,hp=50000):
    ds=[Dummy('red',x,y,hp=hp,spd=0,dmg=0) for x,y in xy]
    for d in ds:g.deploy('red',d)
    return ds
def _act(g,tr):
    g.players[tr.team].elixir=10;tr.ability.cd=0
    ok,msg=g.activate_ability(tr.team,tr);assert ok,msg
def _named(g,team,name):return [t for t in g.players[team].troops if t.alive and t.name==name]
def _comps(t):return [type(c).__name__ for c in t.components]

def t_magic_archer_pierces_line():
    g=_game();ma=mk_card('magic_archer',11,'blue',9,10);g.deploy('blue',ma)
    a,b,c=_dummies(g,(9,14),(9,18),(13,14))
    g.run(2)
    assert 50000-a.hp==50000-b.hp==2*ma.dmg and c.hp==50000,f"{a.hp} {b.hp} {c.hp}"
    assert 'LineAttack' in _comps(ma) and 'SplashAttack' not in _comps(ma)
    return f"Magic Archer arrow hits both units on its 11 tile line ({50000-a.hp} each), none beside it"
def t_executioner_axe_hits_out_and_back():
    g=_game();ex=mk_card('executioner',11,'blue',9,10);g.deploy('blue',ex)
    a,b=_dummies(g,(9,13),(9,15.5))
    g.run(1.0);out=(50000-a.hp,50000-b.hp)
    g.run(1.7);back=(50000-a.hp,50000-b.hp)
    assert out==(179,179) and back==(358,358),f"{out} {back}"
    return "Executioner axe strikes both units on the way out (179) and back (358 total) before the next throw"
def t_bowler_boulder_rolls_through_with_knockback():
    g=_game();bw=mk_card('bowler',11,'blue',9,10);g.deploy('blue',bw)
    a,b=_dummies(g,(9,13),(9,16))
    g.run(4)
    assert 50000-a.hp==50000-b.hp==bw.dmg and abs(a.y-14)<0.01 and abs(b.y-17)<0.01,f"{a.hp} {b.hp} {a.y} {b.y}"
    return f"Bowler boulder damages the target and the unit 3 tiles behind, pushing both 1 tile ({a.y:.1f},{b.y:.1f})"
def t_firecracker_sparks_and_tower_volley():
    g=_game();fc=mk_card('firecracker',11,'blue',9,10);g.deploy('blue',fc)
    a,b=_dummies(g,(9,14),(9,16))
    g.run(3.5)
    assert 50000-a.hp==64 and 50000-b.hp>=64 and fc.dmg==64 and fc.ct_dmg==320,f"{50000-a.hp} {50000-b.hp}"
    g=quiet(Game());fc=mk_card('firecracker',11,'blue',9,20);g.deploy('blue',fc);tw=g.arena.get_tower('red','princess','left');ini=tw.hp
    g.run(3.5)
    assert ini-tw.hp==320
    return f"Firecracker: one pellet on the target, {50000-b.hp} from sparks behind it, the full 5 pellet volley (320) on a tower"
def t_the_log_shares_the_strip():
    g=Game();random.seed(1);sk=mk_card('skeleton_army',11,'red',9,22)
    for s in sk:g.deploy('red',s)
    b=mk_card('cannon',11,'red',9,21);g.deploy('red',b);by=b.y
    lg=mk_card('the_log',11,'blue',9,18);lg.apply(g);g.spells.append(lg)
    assert not any(s.alive for s in sk) and b.hp<b.max_hp and b.y==by
    return "The Log rolls through the strip: 15 skeletons dead, the cannon damaged but not pushed"
def t_little_prince_charge_is_the_guardienne_dash():
    g=_game();lp=mk_card('little_prince',11,'blue',9,10);g.deploy('blue',lp)
    assert not any(isinstance(c,fx.Charge) for c in lp.components) and isinstance(lp.ability,fx.RoyalRescue)
    d,far=_dummies(g,(9,13),(9,16));_act(g,lp);g.run(1.3)
    gd=_named(g,'blue','Protector')
    # the prince keeps shooting the same dummy for 104 a bolt
    assert (50000-d.hp-320)%104==0 and far.hp==50000 and d.y>13 and len(gd)==1 and not any(isinstance(c,fx.Charge) for c in gd[0].components)
    assert lp.ability.uses==0 and lp.ability.rng==4.0
    return f"Royal Rescue: Guardienne dashes up to 4 tiles for 320 charge damage and a {d.y-13:.1f} tile pushback, single use"
def t_elite_barbarians_no_unsourced_charge():
    eb=mk_card('elite_barbarians',11,'blue',9,10)
    assert not any(isinstance(c,fx.Charge) for c in eb[0].components) and card('elite_barbarians')['skills']['charge']['range'] is None
    return "Base Elite Barbarians keep no charge (no wiki or RoyaleAPI source for the cs2.4.0 charge damage)"
def t_evo_elite_barbarians_rage_spears():
    g=_game();eb=mk_card('elite_barbarians',11,'blue',9,10,evolved=True)
    for t in eb:g.deploy('blue',t)
    d,=_dummies(g,(9,14.5));g.run(0.3)
    assert 50000-d.hp in (220,440) and len([z for z in g.spells if getattr(z,'name','')=='Rage'])>=2,f"{50000-d.hp} {len(g.spells)}"
    k=mk_card('knight',11,'blue',9,12.3);g.deploy('blue',k);g.run(0.2)
    assert has(k,'rage')
    return "Evo Elite Barbarians throw 220 spears at 3.5 to 5 tiles and leave 1 tile rage circles that rage allies"
def t_goblin_demolisher_rocket_ride():
    g=_game();gd=mk_card('goblin_demolisher',11,'blue',9,17);g.deploy('blue',gd)
    b=mk_card('cannon',11,'red',9,20);g.deploy('red',b);d,=_dummies(g,(10.5,19),hp=5000)
    gd.hp=int(gd.max_hp*0.4);g.run(0.1)
    assert gd.targets==['Buildings'] and abs(gd.spd-2.0)<0.01 and gd.is_suicide
    g.run(6)
    assert not gd.alive and not b.alive and 5000-d.hp==847 and d.x>10.5,f"{b.hp} {d.hp} {d.x}"
    return f"Goblin Demolisher below 50% rides the rocket to the cannon and explodes for 847 in 2.5 tiles with knockback (dummy x {d.x:.1f})"
def t_phoenix_egg_hatches():
    g=_game();ph=mk_card('phoenix',11,'blue',9,14);g.deploy('blue',ph);d,=_dummies(g,(9.5,14),hp=5000)
    ph.alive=False;g.run(0.1)
    egg=_named(g,'blue','Phoenix Egg');assert len(egg)==1 and egg[0].hp==240 and egg[0].targets==[] and 5000-d.hp==163 and d.x>9.5
    g.run(4.1);assert egg[0].alive and not _named(g,'blue','PhoenixNoRespawn')
    g.run(0.2);rb=_named(g,'blue','PhoenixNoRespawn')
    assert not egg[0].alive and len(rb)==1 and rb[0].hp==ph.max_hp and rb[0].death_dmg==0 and not rb[0].components
    return "Phoenix death: 163 area damage with knockback, a 240 hp egg that hatches after 4.3 s into a Phoenix without egg or death damage"
def t_phoenix_egg_can_be_destroyed():
    g=_game();ph=mk_card('phoenix',11,'blue',9,14);g.deploy('blue',ph);ph.alive=False;g.run(0.1)
    egg=_named(g,'blue','Phoenix Egg')[0];egg.take_damage(240);g.run(5)
    assert not _named(g,'blue','PhoenixNoRespawn')
    return "A destroyed egg does not hatch"
def t_mega_knight_jump_time_and_knockback():
    g=_game();mk=mk_card('mega_knight',11,'blue',9,10);g.deploy('blue',mk);d,=_dummies(g,(9,15.5))
    t0=None
    while g.t<3 and t0 is None:
        g.tick()
        if d.hp<50000:t0=g.t
    assert t0 is not None and abs(t0-0.9)<=0.1 and abs(d.y-16.5)<0.01,f"{t0} {d.y}"
    return f"Mega Knight jump lands after {t0:.2f} s (Jump Time 0.9) for 537 and 1 tile knockback"
def t_miner_burrows_untargetable():
    g=Game();mn=mk_card('miner',11,'blue',9,25);g.deploy('blue',mn)
    assert has(mn,'burrowed') and (mn.x,mn.y)==(9.0,3.0)
    ts=None
    while g.t<4 and ts is None:
        g.tick()
        if not has(mn,'burrowed'):ts=g.t
    exp=max(1.0,22.0/(650/60))
    assert abs(ts-exp)<=0.1 and mn.hp==mn.max_hp and abs(mn.x-9)<0.1 and abs(mn.y-25)<0.1,f"{ts} {mn.hp} {mn.x},{mn.y}"
    g.run(1);assert mn.hp<mn.max_hp
    return f"Miner travels underground from the King Tower at 650 and surfaces at {ts:.2f} s (expected {exp:.2f}) untouched by the towers"
def t_miner_close_placement_waits_the_deploy_time():
    g=Game();mn=mk_card('miner',11,'blue',9,6);g.deploy('blue',mn);ts=None
    while g.t<3 and ts is None:
        g.tick()
        if not has(mn,'burrowed'):ts=g.t
    assert abs(ts-1.0)<0.01
    return "A Miner placed near his King Tower surfaces after the 1 s deploy time"
def t_goblin_drill_surfaces_and_spawns():
    g=_game();gd=mk_card('goblin_drill',11,'blue',9,20);g.deploy('blue',gd);d,=_dummies(g,(9,21))
    g.run(0.9);assert has(gd,'burrowed') and d.hp==50000 and gd.hp==gd.max_hp
    g.run(0.2);assert not has(gd,'burrowed') and 50000-d.hp==66
    g.run(1.0);assert len(_named(g,'blue','Goblin'))==1
    g.run(3.0);assert len(_named(g,'blue','Goblin'))==2
    g=Game();gd=mk_card('goblin_drill',11,'blue',14.5,23.5);g.deploy('blue',gd);tw=g.arena.get_tower('red','princess','right');ini=tw.hp;g.run(1.2)
    assert tw.hp==ini
    return "Goblin Drill surfaces after 1 s with 66 spawn damage (none to towers), a Goblin 1 s later then every 3 s"
def t_evo_goblin_drill_resurfaces_twice():
    g=_game();gd=mk_card('goblin_drill',11,'blue',9,20,evolved=True);g.deploy('blue',gd);d,=_dummies(g,(9,21));g.run(1.1)
    n0=len(_named(g,'blue','Goblin'));h0=50000-d.hp
    gd.hp=int(gd.max_hp*0.6);g.run(0.1);n1=len(_named(g,'blue','Goblin'))
    gd.hp=int(gd.max_hp*0.3);g.run(0.1);n2=len(_named(g,'blue','Goblin'))
    assert (n1-n0,n2-n1)==(2,1) and 50000-d.hp==h0+2*66
    return "Evo Goblin Drill resurfaces at 66% and 33% leaving 2 then 1 Goblins and repeating its spawn damage"
def t_mother_witch_cursed_hog():
    g=_game();mw=mk_card('mother_witch',11,'blue',9,10);g.deploy('blue',mw);d,=_dummies(g,(9,13),hp=150)
    g.run(3);hogs=_named(g,'blue','Cursed Hog')
    assert not d.alive and len(hogs)==1 and hogs[0].targets==['Buildings'] and hogs[0].hp==529 and hogs[0].dmg==53
    g=_game();mw=mk_card('mother_witch',11,'blue',9,10);g.deploy('blue',mw);d,=_dummies(g,(9,13),hp=5000)
    g.run(1.2);mw.alive=False;g.run(0.1);g.run(5.5);d.alive=False;g.run(0.2)
    assert not _named(g,'blue','Cursed Hog')
    return "A troop killed within 5 s of a Mother Witch hit becomes a 529 hp Cursed Hog; the curse expires after 5 s"
def t_suspicious_bush_invisible_until_the_building():
    g=quiet(Game());bu=mk_card('suspicious_bush',11,'blue',9,20);g.deploy('blue',bu);tw=g.arena.get_tower('red','princess','left');ini=tw.hp
    g.run(0.2);assert has(bu,'invisible') and bu.hp==bu.max_hp and bu.targets==['Buildings']
    g.run(4);assert bu.hp==bu.max_hp or not bu.alive
    g.run(4);assert not bu.alive and ini-tw.hp>=256 and len(_named(g,'blue','Bush Goblin'))==2
    return "Suspicious Bush walks invisible to the tower, bursts for 256 and leaves two Bush Goblins"
def t_ronin_parries_melee_only():
    g=_game();rn=mk_card('ronin',11,'blue',9,14);g.deploy('blue',rn);rn.dmg=0;pk=mk_card('pekka',11,'red',9,15.2);g.deploy('red',pk);ini=pk.hp
    for _ in range(60):
        g.tick()
        if pk.hp<ini:break
    assert rn.hp==rn.max_hp and ini-pk.hp==2*pk.dmg,f"ronin {rn.hp} pekka lost {ini-pk.hp}"
    assert abs(rn.fhspd-0.4)<0.01
    g=_game();rn=mk_card('ronin',11,'blue',9,14);g.deploy('blue',rn);m=mk_card('musketeer',11,'red',9,18);g.deploy('red',m);g.run(2)
    assert rn.max_hp-rn.hp==2*m.dmg and m.hp==m.max_hp
    return "Ronin parries the P.E.K.K.A's swing and returns 200% of it; Musketeer shots are not parried"
def t_ronin_parry_cooldown():
    g=_game();rn=mk_card('ronin',11,'blue',9,14);g.deploy('blue',rn);k=mk_card('knight',11,'red',9,15);g.deploy('red',k);k.dmg=100
    g.run(4)
    lost=rn.max_hp-rn.hp
    assert 0<lost<300 and lost%100==0,lost
    return f"Ronin takes {lost} from a 100 damage Knight over 4 s: one swing parried, the parry recharges in 3.5 s"
def t_rune_giant_enchants_two_nearest():
    g=_game();rg=mk_card('rune_giant',11,'blue',9,10);g.deploy('blue',rg)
    assert rg.ability is None and any(isinstance(c,fx.Enchant) for c in rg.components)
    ks=[mk_card('knight',11,'blue',x,12) for x in (8,9,10)];b=mk_card('cannon',11,'blue',9,8);wb=mk_card('wall_breakers',11,'blue',7,9)
    for t in ks+[b]+wb:g.deploy('blue',t)
    d,=_dummies(g,(9,13));g.run(0.1)
    en=[t for t in ks if 'Enchanted' in _comps(t)]
    assert len(en)==2 and 'Enchanted' not in _comps(b) and not any('Enchanted' in _comps(w) for w in wb)
    d0=d.hp;g.run(1.2)
    n=(d0-d.hp)//202
    g.run(5);hits=[c for c in en[0].components if isinstance(c,fx.Enchanted)][0].n
    assert d0-d.hp>=hits*202+ (hits//3)*220-2*202,f"{d0-d.hp} {hits}"
    return f"Rune Giant enchants the 2 nearest troops (not the cannon or Wall Breakers); every 3rd hit adds 220 ({n} hits in the first 1.2 s)"
def t_rune_giant_enchantment_outlives_her_5s():
    g=_game();rg=mk_card('rune_giant',11,'blue',9,10);g.deploy('blue',rg);k=mk_card('knight',11,'blue',9,12);g.deploy('blue',k);g.run(0.1)
    assert 'Enchanted' in _comps(k)
    rg.alive=False;g.run(4.5);assert 'Enchanted' in _comps(k)
    g.run(1);assert 'Enchanted' not in _comps(k)
    return "The enchantment stays 5 s after the Rune Giant dies"
def t_evo_battle_ram_keeps_charging():
    random.seed(3);g=Game();br=mk_card('battle_ram',11,'blue',3.5,20,evolved=True);g.deploy('blue',br)
    assert not br.is_suicide and br.dmg==286
    tw=g.arena.get_tower('red','princess','left');ini=tw.hp;hits=[]
    while g.t<12 and br.alive:
        g.tick()
        if ini-tw.hp!=(hits[-1] if hits else 0):hits.append(ini-tw.hp)
    deltas=[b-a for a,b in zip([0]+hits,hits)]
    assert deltas[0]==286 and 573 in deltas[1:],deltas
    return f"Evo Battle Ram connects for 286, recoils 2 tiles and charges straight back for 573: {deltas}"
def t_evo_battle_ram_bulldozes_and_rages_barbarians():
    g=_game();br=mk_card('battle_ram',11,'blue',3.5,17,evolved=True);g.deploy('blue',br);b=mk_card('cannon',11,'red',3.5,24);g.deploy('red',b)
    d,=_dummies(g,(3.5,21.5));g.run(6)
    assert 50000-d.hp==212 and d.y>21.5,f"{50000-d.hp} {d.y}"
    br.alive=False;g.run(0.1);bs=_named(g,'blue','Barbarian')
    assert len(bs)==2 and all(has(x,'rage') for x in bs)
    return f"The charging evo ram hits troops in its path for 212 and pushes them ({d.y-21.5:.1f} tiles); its Barbarians drop raged"
def t_evo_cannon_barrage():
    g=_game();cn=mk_card('cannon',11,'blue',9,14,evolved=True);g.deploy('blue',cn)
    front=_dummies(g,(7,16.5),(11,16.5),(9,18.5));behind,=_dummies(g,(9,11));g.run(0.1)
    assert all(d.hp<50000 for d in front) and behind.hp==50000
    g=Game();cn=mk_card('cannon',11,'blue',3.5,22,evolved=True);g.deploy('blue',cn);tw=g.arena.get_tower('red','princess','left');ini=tw.hp;g.run(0.1)
    assert (ini-tw.hp)%89==0 and ini>tw.hp
    return f"Evo Cannon barrage lands 9 balls in two rows ahead (tower takes {ini-tw.hp} at 89 per ball)"
def t_evo_furnace_hot_spawns():
    g=_game();fu=mk_card('furnace',11,'blue',9,10,evolved=True);g.deploy('blue',fu);st=[c for c in fu.components if isinstance(c,fx.SpawnTimer)][0]
    g.run(0.2);cold=st.interval;_dummies(g,(9,13));g.run(0.2);hot=st.interval
    assert cold==5 and abs(hot-5/2.91)<0.01,f"{cold} {hot}"
    return f"Evo Furnace spawns every {hot:.2f} s while attacking instead of every 5 s"
def t_evo_goblin_barrel_decoys():
    g=_game();gb=mk_card('goblin_barrel',11,'blue',3.5,25,evolved=True);gb.apply(g)
    real=_named(g,'blue','Goblin');dec=_named(g,'blue','Decoy Goblin')
    assert len(real)==3 and len(dec)==3 and all(t.hp==202 and t.dmg==125 for t in real) and all(t.hp==81 and t.dmg==66 for t in dec)
    assert all(abs(t.x-14.5)<=1 for t in dec) and all(abs(t.x-3.5)<=1 for t in real)
    return "Evo Goblin Barrel drops 3 Goblins and a decoy barrel of 3 Decoy Goblins (81 hp, 66 damage) on the mirrored tile"
def t_evo_minion_horde_veil():
    g=_game();mh=mk_card('minion_horde',11,'blue',9,10,evolved=True)
    for t in mh:g.deploy('blue',t)
    m=mh[0];m.take_damage(10);g.run(0.1)
    assert has(m,'invisible') and has(m,'invincible') and m.hp==220
    m.take_damage(1000);assert m.hp==220
    _,rate,mrate=g._status_mods(m);assert abs(rate-0.67)<0.01 and abs(mrate-0.67)<0.01
    g.run(3.1);assert not has(m,'invisible');m.take_damage(10);assert m.hp==210
    return "Evo Minion: the first hit veils it for 3 s (untargetable, immune, hit and move speed x0.67)"
def t_evo_princess_icy_arrows_and_frost_zone():
    g=_game();pr=mk_card('princess',11,'blue',9,10,evolved=True);g.deploy('blue',pr);d,=_dummies(g,(9,16))
    g.run(3.5);s=[x for x in d.statuses if x.kind=='slow']
    assert s and s[0].val==0.7 and s[0].dur>5.0
    d.statuses=[];g.run(3);assert not has(d,'slow')
    g.run(3);assert has(d,'slow')
    n,=_dummies(g,(9,12));pr.alive=False;g.run(0.2)
    assert 50000-n.hp==168 and any(getattr(z,'name','')=='Frost' for z in g.spells) and has(n,'slow')
    return "Evo Princess slows 30% for 5.5 s on the first and every other shot; her death deals 168 and leaves a frost zone"
def t_evo_skeleton_army_general_gerry():
    g=_game();sa=mk_card('skeleton_army',11,'blue',9,14,evolved=True)
    for t in sa:g.deploy('blue',t)
    gerry=[t for t in sa if t.name=='General Gerry'];sk=[t for t in sa if t.name!='General Gerry']
    assert len(gerry)==1 and len(sk)==15
    gy=gerry[0];assert gy.hp==81 and gy.shield_hp==81 and gy.dmg==81 and gy.hspd==1.0 and gy.rng==1.6 and abs(gy.spd-1.5)<0.01
    sk[0].alive=False;g.run(0.1);sh=_named(g,'blue','Shadow Skeleton')
    assert len(sh)==1 and has(sh[0],'invisible') and has(sh[0],'invincible') and abs(sh[0].spd-1.0)<0.01
    sh[0].take_damage(1000);assert sh[0].alive
    gy.alive=False;g.run(0.1);assert not sh[0].alive
    return "Evo Skeleton Army: 15 Skeletons plus General Gerry (81/81 shield/81); fallen skeletons rise as indestructible shadows that fall with him"
def t_evo_tesla_pulse_kills_skeletons():
    g=_game();random.seed(1);sks=mk_card('skeletons',11,'red',9,16)
    for s in sks:g.deploy('red',s)
    te=mk_card('tesla',11,'blue',9,14,evolved=True);g.deploy('blue',te);g.run(0.1)
    assert not any(s.alive for s in sks)
    d,=_dummies(g,(9,19));te.alive=False;g.run(0.1)
    assert 50000-d.hp==174 and has(d,'stun')
    return "Evo Tesla pulses 174 in 6 tiles on deploy (kills Skeletons) and again on destruction"
def t_hero_balloon_coffin_cadets():
    g=_game();bl=mk_card('balloon',11,'blue',9,10,hero=True);g.deploy('blue',bl);d,=_dummies(g,(9,14),hp=5000)
    assert isinstance(bl.ability,fx.CoffinCadets) and bl.ability.uses==1
    _act(g,bl);g.run(1.5);tr=_named(g,'blue','Skeletrooper')
    assert len(tr)==1 and 5000-d.hp==263 and tr[0].hp==473 and tr[0].dmg==204 and tr[0].ct_dmg==20 and tr[0].targets==['Ground']
    ok,_=g.activate_ability('blue',bl);assert not ok
    return "Coffin Cadets drops a 473 hp Skeletrooper on the nearest ground enemy for 263 landing damage, once"
def t_hero_berserker_savage_survival():
    g=_game();be=mk_card('berserker',11,'blue',9,10,hero=True);g.deploy('blue',be);d,=_dummies(g,(9,11))
    _act(g,be);g.run(1.3)
    assert be.dmg==167 and be.ct_dmg==42 and abs(be.hspd-0.2)<0.01 and abs(be.spd-2.25)<0.01 and be.hp_floor==1
    be.take_damage(5000);assert be.alive and be.hp==1
    g.run(3.5);assert be.dmg==102 and abs(be.hspd-0.6)<0.01 and be.hp_floor==0
    return "Savage Survival: 167 damage bear swings every 0.2 s at Ultra Fast speed, hp floored at 1 for 3.5 s"
def t_hero_bowler_stone_swish():
    g=_game();bw=mk_card('bowler',11,'blue',9,10,hero=True);g.deploy('blue',bw);d,=_dummies(g,(9,20))
    assert isinstance(bw.ability,fx.StoneSwish) and bw.ability.CAST_TIME==2.5
    _act(g,bw);g.run(3.0)
    assert bw.spd==0 and bw.rng==11.5 and abs(bw.hspd-2.5/1.3)<0.01 and bw.dmg==578 and bw.ct_dmg==289 and 'SplashAttack' in _comps(bw)
    g.run(6);assert d.hp<50000
    g.run(3);assert bw.rng==4.0 and bw.dmg==289 and 'LineAttack' in _comps(bw) and bw.spd>0
    return "Stone Swish: after a 2.5 s cast the planted Bowler lobs 578 boulders to 11.5 tiles for 7.3 s"
def t_hero_dark_prince_destructive_dismount():
    g=_game();dp=mk_card('dark_prince',11,'blue',9,10,hero=True);g.deploy('blue',dp);d,=_dummies(g,(9.5,10.7),hp=5000)
    _act(g,dp);g.run(1.2);rh=_named(g,'blue','Rhino')
    assert 5000-d.hp>=307 and len(rh)==1 and rh[0].hp==1356 and rh[0].dmg==179 and rh[0].charge_dmg==358 and rh[0].targets==['Buildings']
    assert any(isinstance(c,fx.Charge) for c in rh[0].components) and not any(isinstance(c,fx.Charge) for c in dp.components)
    return "Destructive Dismount: 307 impact damage, the Rhino (1356 hp) charges buildings for 358, the prince fights on without his charge"
def t_hero_ice_golem_three_blasts_kill_goblins():
    g=_game();ig=mk_card('ice_golem',11,'blue',9,10,hero=True);g.deploy('blue',ig);d,=_dummies(g,(9,13))
    assert isinstance(ig.ability,fx.Snowstorm) and ig.ability.uses==1
    _act(g,ig);g.run(1.3);s=[x for x in d.statuses if x.kind=='slow']
    assert 50000-d.hp==69 and s and s[0].val==0.7 and not has(d,'freeze')
    g.run(4.5);assert 50000-d.hp==207
    g=_game();random.seed(2);gs=mk_card('goblins',11,'red',9,12)
    for x in gs:g.deploy('red',x)
    ig2=mk_card('ice_golem',11,'blue',9,10,hero=True);g.deploy('blue',ig2);_act(g,ig2);g.run(6)
    assert not any(x.alive for x in gs)
    return "Snowstorm: three 69 damage blasts in 4 tiles, each slowing 30% (no freeze); all three take out Goblins"
def t_hero_ice_wizard_frosty_fella():
    g=_game();iw=mk_card('ice_wizard',11,'blue',9,10,hero=True);g.deploy('blue',iw);d,=_dummies(g,(9,14),hp=5000)
    assert isinstance(iw.ability,fx.FrostyFella)
    g.run(1.0);_act(g,iw);g.run(1.3);sm=_named(g,'blue','Snowman')
    assert len(sm)==1 and sm[0].is_building and abs(sm[0].y-15)<0.01 and sm[0].targets==[] and has(d,'freeze')
    g.run(8);assert not sm[0].alive and not has(d,'freeze')
    return "Frosty Fella raises a Snowman one tile behind the target that freezes everything within 2.5 tiles for its 7 s life"
def t_hero_tombstone_regal_revive():
    g=_game();ts=mk_card('tombstone',11,'blue',9,10,hero=True);g.deploy('blue',ts)
    assert isinstance(ts.ability,fx.RegalRevive) and ts.ability.cost==5 and ts.hp==529
    _act(g,ts);g.run(1.3);q=_named(g,'blue','Tomb Queen')
    assert not ts.alive and len(q)==1 and q[0].hp==4224 and q[0].dmg==422 and q[0].hspd==2.1 and q[0].targets==['Buildings'] and q[0].sight_r==7.0
    assert not _named(g,'blue','Skeleton')
    return "Regal Revive (5 elixir) replaces the tombstone with the 4224 hp Tomb Queen: 422 damage every 2.1 s at buildings, sight 7"
def t_hero_valkyrie_wild_whirlwind():
    g=_game();vk=mk_card('valkyrie',11,'blue',9,10,hero=True);g.deploy('blue',vk);d,=_dummies(g,(9,14))
    assert isinstance(vk.ability,fx.WildWhirlwind)
    _act(g,vk);g.run(1.3)
    assert vk.hspd==0.25 and vk.dmg==97 and vk.ct_dmg==48 and vk.splash_r==2.5 and abs(vk.spd-2.0)<0.01 and vk._dmg_reduction==0.15 and vk.y>11
    g.run(3.5);assert vk.hspd==1.5 and vk.dmg==266 and vk._dmg_reduction==0 and d.hp<50000
    return "Wild Whirlwind: dash to the nearest troop, 97 damage spins every 0.25 s in 2.5 tiles, double speed and 15% damage reduction for 3.5 s"
def t_abilities_single_use_except_boss_bandit():
    g=_game();hk=mk_card('knight',11,'blue',9,10,hero=True);g.deploy('blue',hk)
    _act(g,hk);g.run(1.5);ok,msg=g.activate_ability('blue',hk);assert not ok and hk.ability.uses==0
    bb=mk_card('boss_bandit',11,'blue',9,10);assert bb.ability.uses is None and bb.ability.max_cd==3
    for k in ('golden_knight','archer_queen','skeleton_king','mighty_miner','monk','goblinstein'):
        assert card(k)['skills']['ability']['cooldown'] is None
    return "Every hero and champion ability is single use since August 4; Boss Bandit keeps her 3 s cooldown"
def t_graveyard_first_skeleton_2_2s():
    g=_game();gy=mk_card('graveyard',11,'blue',9,20);gy.apply(g);g.spells.append(gy)
    g.run(2.1);assert not _named(g,'blue','Skeleton')
    g.run(0.2);assert len(_named(g,'blue','Skeleton'))==1
    return "Graveyard's first Skeleton rises 2.2 s after the cast"
def t_royal_chef_cook_window_from_data():
    from sim.towers import RoyalChef
    rc=RoyalChef(11);assert rc.ckdel==7.0 and rc.ckmin==23.0 and rc.ckmax==38.0
    return "Royal Chef waits 7 s then cooks for 23 to 38 s, all from cards.json"
def t_evo_inferno_dragon_retain_7s():
    idr=mk_card('inferno_dragon',11,'blue',9,10,evolved=True);c=[x for x in idr.components if isinstance(x,fx.EvoInfernoDragon)][0]
    assert c.retain==7.0 and c.s4_time==20.0 and c.s4_dmg==844
    return "Evo Inferno Dragon keeps its stage for 7 s and reaches the 844 damage stage after 20 s"
def t_minion_giant_flying_building_tank():
    g=quiet(Game());mg=mk_card('minion_giant',11,'blue',9,14);g.deploy('blue',mg);d,=_dummies(g,(9,15))
    assert mg.hp==1817 and mg.dmg==189 and mg.transport=='Air' and mg.targets==['Buildings'] and mg.rng==4.0 and mg.hspd==1.5 and mg.mass==15
    tw=g.arena.get_tower('red','princess','left');ini=tw.hp;g.run(15)
    assert d.hp==50000 and ini-tw.hp==5*189
    return "Minion Giant flies past troops and hits the tower for 189 every 1.5 s from 4 tiles"
