import json,os,random,sys,copy
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from troop import Troop
from spell import Spell,SpawnSpell
from spell import LogSpell,GraveyardSpell,RageSpell,LightningSpell,CloneSpell
from spell import EarthquakeSpell,TornadoSpell,VoidSpell,VinesSpell
from spell import GoblinCurseSpell,RoyalDeliverySpell,BarbarianBarrelSpell
from spell import EvoZapSpell,EvoSnowballSpell
from components import (SplashAttack,BuildingTarget,RiverJump,Recoil,RiderAttack,
    Charge,SpawnTimer,DeathDamage,DeathNova,DeathSpawn,SpawnZap,DualTarget,
    RampUp,RageDrop,SuicideChain,ChainAttack,HealBurst,ZapPack,HealPulse,
    RocketLauncher,FormTransform,ElixirProd,
    BanditDash,SoulCollect,MonkCombo,LPRamp,
    Ability,DashingDash,SoulSummoning,GetawayGrenade,CloakingCape,
    ExplosiveEscape,LightningLink,RoyalRescue,PensiveProtection,
    EvoKnight,EvoBomber,EvoSkeletons,EvoBarbarians,EvoBats,
    EvoRoyalRecruits,EvoRoyalGiant,EvoIceSpirit,EvoSkelBarrel,
    EvoFirecracker,EvoArchers,
    EvoValkyrie,EvoMusketeer,EvoDartGoblin,EvoRoyalHogs,EvoGoblinCage,
    EvoBabyDragon,EvoWitch,EvoPekka,EvoGoblinGiant,EvoHunter,
    EvoElectroDragon,EvoWallBreakers,EvoExecutioner,EvoGoblinBarrel,EvoGoblinDrill,
    MKJump,
    EvoMegaKnight,EvoInfernoDragon,EvoRoyalGhost,EvoBandit,EvoFisherman,
    EvoLumberjack,EvoIceWizard,
    TriumphantTaunt,BannerBrigade,
    HeroicHurl,BreakfastBoost,TrustyTurret,Snowstorm,FieryFlight,WoundingWarp,
    RowdyReroll,TripleThreat)
from building import Building
_CD=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','game_data','cards')
_JC={}
def _ld(name):
    if name not in _JC:
        p=os.path.join(_CD,name+'.json')
        with open(p) as f:_JC[name]=json.load(f)
    return _JC[name]
def _transpose_sbl(sbl,lvl):
    first_v=next(iter(sbl.values()),None)
    if isinstance(first_v,dict) and any(k.isdigit() for k in first_v):
        r={}
        for sn,levels in sbl.items():
            v=levels.get(str(lvl))
            if v is not None:r[sn]=v
        return r
    return sbl.get(str(lvl),{})
def _parse_old(d,lvl):
    lv=d['levels']
    if 'stats' in lv:
        stats=None
        for s in lv['stats']:
            if s['level']==lvl:stats=s;break
        if not stats:stats=lv['stats'][-1]
    else:
        stats=lv.get(str(lvl))
        if not stats:
            ks=sorted(lv.keys(),key=lambda k:int(k) if k.isdigit() else 0)
            stats=lv[ks[-1]] if ks else {}
    hp=stats.get('hp',stats.get('hitpoints',0))
    dmg=stats.get('damage',0)
    hs_raw=d.get('hit_speed','1.0s')
    hspd=float(str(hs_raw).rstrip('s'))
    fhspd=float(str(d.get('first_hit',hs_raw)).rstrip('s'))
    sv=int(d.get('speed','Medium (60)').split('(')[1].rstrip(')'))
    spd=sv/60.0
    rng_raw=d.get('range','1.2')
    rng_str=str(rng_raw)
    if '(' in rng_str:rng=float(rng_str.split('(')[1].rstrip(')'))
    elif ':' in rng_str:rng=float(rng_str.split(':')[1].strip())
    else:rng=float(rng_str)
    tgt_str=str(d.get('targets','Ground'))
    if 'Air' in tgt_str and 'Ground' in tgt_str:tgts=['Air','Ground']
    elif 'Air' in tgt_str:tgts=['Air']
    elif 'Buildings' in tgt_str:tgts=['Buildings']
    else:tgts=['Ground']
    comps=[]
    sp_r=d.get('splash',d.get('area_splash',0))
    if sp_r:comps.append(SplashAttack())
    cl=d.get('chain_lightning')
    cfg={'hp':hp,'dmg':dmg,'hspd':hspd,'fhspd':fhspd,'spd':spd,
         'rng':rng,'targets':tgts,'transport':d.get('transport','Ground'),
         'atk_type':'area' if sp_r else 'single_target','splash_r':sp_r,'ct_dmg':0,
         'lvl':lvl,'name':d.get('name',''),'components':comps,
         'mass':d.get('hidden_stats',{}).get('mass') or 4,
         'sight_r':d.get('hidden_stats',{}).get('sight_range_tiles') or 5.5,
         'collision_r':d.get('hidden_stats',{}).get('collision_radius_tiles') or 0.5}
    if d.get('splash_360'):cfg['splash_360']=True
    fd_raw=d.get('freeze_duration','')
    if fd_raw:cfg['freeze_dur']=float(str(fd_raw).rstrip('s'))
    if d.get('attack_type')=='Suicide' or cl:
        cfg['is_suicide']=True
        stn_raw=d.get('stun_duration','0.5s')
        cfg['chain_stun']=float(str(stn_raw).rstrip('s')) if stn_raw else 0.5
    if cl:
        cfg['chain_count']=cl.get('max_targets',9)
        cr=cl.get('chain_range','4.0 tiles')
        cfg['chain_range']=float(str(cr).split()[0])
        comps.append(SuicideChain())
    return cfg
def _parse_new(d,lvl,sub=None):
    if sub:
        sd=sub
        sbl=sd.get('stats_by_level',{})
        stats=sbl.get(str(lvl))
        if not stats:
            ks=sorted(sbl.keys(),key=int)
            stats=sbl[ks[-1]] if ks else {}
        mv=sd.get('movement',{});atk=sd.get('attack',{})
    else:
        sbl=d.get('stats_by_level',{})
        stats=_transpose_sbl(sbl,lvl)
        mv=d.get('movement',d.get('movement_speed',{}));atk=d.get('attack',{})
    hp=stats.get('hitpoints',stats.get('hp',0))
    dph=stats.get('damage_per_hit')
    if isinstance(dph,dict):
        dmg=dph.get('stage_1',0)
    else:
        dmg=stats.get('damage',stats.get('area_damage',stats.get('damage_per_bolt',
            stats.get('total_damage',
            stats.get('damage_per_shrapnel',stats.get('damage_per_pellet',
            stats.get('ram_damage',stats.get('rider_damage',0))))))))
    if not hp and not dmg:
        pfx=d.get('name','').lower().replace(' ','_')+'_'
        hp=stats.get(pfx+'hitpoints',0)
        dmg=stats.get(pfx+'damage',stats.get(pfx+'area_damage',0))
    ct_dmg=stats.get('crown_tower_damage',0)
    if isinstance(mv,dict):
        sp=mv.get('speed')
        if isinstance(sp,dict):sv=sp.get('value',60)
        elif sp is not None:sv=int(sp)
        else:sv=mv.get('speed_value',mv.get('value',60))
        transport=mv.get('transport',d.get('transport','Ground'))
    else:
        sv=60;transport=d.get('transport','Ground')
    spd=sv/60.0
    hspd=atk.get('hit_speed_sec',d.get('hit_speed_sec',1.0))
    fhspd=atk.get('first_hit_speed_sec',d.get('first_hit_speed_sec',hspd))
    rng_d=atk.get('range',d.get('range',{}))
    if isinstance(rng_d,dict):
        rng=rng_d.get('tiles',rng_d.get('tiles_max',rng_d.get('max',1.2)))
    else:
        rng=float(rng_d)
    min_rng=0
    if isinstance(rng_d,dict):
        min_rng=rng_d.get('tiles_min',rng_d.get('min',0))
    tgts_raw=atk.get('targets',d.get('targets','Ground'))
    if isinstance(tgts_raw,list):tgts=list(tgts_raw)
    elif 'Air' in str(tgts_raw) and 'Ground' in str(tgts_raw):tgts=['Air','Ground']
    elif 'Buildings' in str(tgts_raw):tgts=['Buildings']
    else:tgts=[str(tgts_raw)]
    at=atk.get('damage_type','single_target')
    sr=atk.get('splash_radius_tiles',0)
    comps=[]
    if at in ('area','splash','area_splash','area_damage'):
        if sr<=0:sr=1.2
        comps.append(SplashAttack())
    if at=='chain_lightning':
        ct=atk.get('chain_targets',3)
        cr=atk.get('chain_range_tiles',4.0)
        cs=atk.get('stun_duration_sec',0.5)
        comps.append(ChainAttack())
    if tgts==['Buildings']:comps.append(BuildingTarget())
    if d.get('mechanics',{}).get('river_jump',{}).get('enabled') or d.get('movement',{}).get('can_jump_river'):comps.append(RiverJump())
    rc=atk.get('recoil_tiles',0)
    if rc>0:comps.append(Recoil(rc))
    sui=atk.get('dies_on_attack',False)
    if hspd is None:hspd=1.0
    if fhspd is None:fhspd=hspd
    cfg={'hp':hp,'dmg':dmg,'hspd':hspd,'fhspd':fhspd,'spd':spd,
         'rng':rng,'min_rng':min_rng,'targets':tgts,'transport':transport,
         'atk_type':at,'splash_r':sr,'ct_dmg':ct_dmg,
         'components':comps,'lvl':lvl,'name':d.get('name','')}
    if atk.get('splash_360'):cfg['splash_360']=True
    if sui:cfg['is_suicide']=True
    if at=='chain_lightning':
        cfg['chain_count']=atk.get('chain_targets',3)
        cfg['chain_range']=atk.get('chain_range_tiles',4.0)
        cfg['chain_stun']=atk.get('stun_duration_sec',0.5)
    cfg['mass']=d.get('hidden_stats',{}).get('mass') or 4
    cfg['sight_r']=d.get('hidden_stats',{}).get('sight_range_tiles') or 5.5
    cfg['collision_r']=d.get('hidden_stats',{}).get('collision_radius_tiles') or 0.5
    return cfg
def _parse_spell(d,lvl):
    if 'levels' in d:
        stats=None
        for s in d['levels']['stats']:
            if s['level']==lvl:stats=s;break
        if not stats:stats=d['levels']['stats'][-1]
        dmg=stats.get('damage',0)
        ct_dmg=stats.get('crown_tower_damage',0)
        radius=float(d.get('radius',2.5))
        sd=d.get('stun_duration','')
        dur=float(sd.rstrip('s')) if sd else 0
        sk='stun' if dur>0 else None;sv=1.0
        kb_raw=d.get('knockback','')
        kb=float(str(kb_raw).split()[0]) if kb_raw else 0
        sl_raw=d.get('slow_effect','')
        if sl_raw:
            sl_dur_raw=d.get('slow_duration','0s')
            dur=float(str(sl_dur_raw).rstrip('s'))
            sk='slow';sv=1.0-abs(int(str(sl_raw).rstrip('%')))/100.0
        ti=d.get('tick_interval','')
        if not ti and stats.get('damage_per_tick'):
            ti=d.get('hit_speed','')
        if ti:
            ti_v=float(str(ti).rstrip('s'))
            td_raw=stats.get('troop_damage_per_tick',stats.get('damage_per_tick',0))
            tt_raw=stats.get('total_troop_damage',stats.get('total_damage',0))
            tks=int(d.get('ticks',max(1,tt_raw//td_raw if td_raw else 3)))
            td=stats.get('troop_damage_per_tick',stats.get('damage_per_tick',0))
            tcd=stats.get('crown_tower_damage_per_tick',0)
            sp_raw=d.get('slow_effect','')
            sp=abs(int(str(sp_raw).rstrip('%')))/100.0 if sp_raw else 0
            dr=float(str(d.get('duration','0')).rstrip('s'))
            return {'dmg':0,'ct_dmg':0,'radius':radius,'kb':0,'dur':dr,
                    'status_kind':None,'name':d.get('name',''),
                    'tick_dmg':td,'tick_ct_dmg':tcd,'tick_interval':ti_v,
                    'ticks_left':tks,'slow_pct':sp}
        return {'dmg':dmg,'ct_dmg':ct_dmg,'radius':radius,
                'kb':kb,'dur':dur,'status_kind':sk,'status_val':sv,'name':d.get('name','')}
    sbl=d.get('stats_by_level',{})
    stats=sbl.get(str(lvl))
    if not stats:
        ks=sorted(sbl.keys(),key=int)
        stats=sbl[ks[-1]] if ks else {}
    sa=d.get('spell_attributes',{})
    dmg=stats.get('area_damage',stats.get('damage',stats.get('damage_single_target',0)))
    ct_dmg=stats.get('crown_tower_damage',0)
    dur=sa.get('duration_sec',0)
    strikes=sa.get('strikes',0)
    ti=sa.get('tick_interval_sec',sa.get('strike_interval_sec',0))
    if strikes and ti:dur=strikes*ti
    if ti>0:
        td=stats.get('damage_per_tick',dmg if strikes else 0)
        tcd=stats.get('crown_tower_damage_per_tick',ct_dmg if strikes else 0)
        sp=sa.get('slow_percent',0)/100.0
        tks=sa.get('ticks',int(dur/ti))
        return {'dmg':0,'ct_dmg':0,'radius':sa.get('radius_tiles',2.5),
                'kb':0,'dur':dur,'status_kind':None,'name':d.get('name',''),
                'tick_dmg':td,'tick_ct_dmg':tcd,'tick_interval':ti,
                'ticks_left':tks,'slow_pct':sp}
    sk='freeze' if dur>0 else None
    kbv=sa.get('knockback_tiles',0)
    if not kbv:
        kb_raw=sa.get('knockback',False)
        kbv=0.5 if kb_raw else 0
    volleys=sa.get('volleys',1)
    if volleys>1:
        per_dmg=dmg//volleys;per_ct=ct_dmg//volleys
    else:
        per_dmg=dmg;per_ct=ct_dmg
    return {'dmg':per_dmg,'ct_dmg':per_ct,'radius':sa.get('radius_tiles',2.5),
            'kb':kbv,'dur':dur,
            'status_kind':sk,'name':d.get('name',''),
            'volleys':volleys,'volley_interval':0.15,
            'total_dmg':dmg,'total_ct':ct_dmg}
def _build_sub_cfg(su_data,lvl,parent_su=None):
    sbl=su_data.get('stats_by_level',{})
    stats=sbl.get(str(lvl))
    if not stats:
        ks=sorted(sbl.keys(),key=int)
        stats=sbl[ks[-1]] if ks else {}
    mv=su_data.get('movement',{})
    atk=su_data.get('attack',{})
    hp=stats.get('hitpoints',0)
    dmg=stats.get('damage',stats.get('area_damage',0))
    sv=mv.get('speed',{}).get('value',60)
    spd=sv/60.0
    hspd=atk.get('hit_speed_sec',1.0)
    fhspd=atk.get('first_hit_speed_sec',hspd)
    rng_d=atk.get('range',{})
    rng=rng_d.get('tiles',1.2) if isinstance(rng_d,dict) else float(rng_d)
    tgts_raw=atk.get('targets','Ground')
    if isinstance(tgts_raw,list):tgts=list(tgts_raw)
    elif 'Buildings' in str(tgts_raw):tgts=['Buildings']
    else:tgts=[str(tgts_raw)]
    comps=[]
    if tgts==['Buildings']:comps.append(BuildingTarget())
    sub_mech=su_data.get('mechanics',{})
    dd=sub_mech.get('death_damage',{})
    death_dmg=0;death_splash_r=0
    if dd.get('enabled'):
        death_dmg=stats.get('death_damage',0)
        death_splash_r=dd.get('splash_radius_tiles',2.0)
        comps.append(DeathDamage())
    ds=sub_mech.get('death_spawn')
    if ds and parent_su is not None:
        ds_name=ds['unit'];ds_cnt=ds['count']
        ds_data=parent_su.get(ds_name,{})
        if ds_data:
            blob_cfg=_build_sub_cfg(ds_data,lvl)
            comps.append(DeathSpawn(blob_cfg,ds_cnt))
    return {'hp':hp,'dmg':dmg,'hspd':hspd,'fhspd':fhspd,'spd':spd,
            'rng':rng,'targets':tgts,'transport':mv.get('transport','Ground'),
            'atk_type':'single_target','splash_r':0,'ct_dmg':0,
            'components':comps,'lvl':lvl,'name':'',
            'death_dmg':death_dmg,'death_splash_r':death_splash_r}
def _add_mechanics(d,cfg,lvl):
    mech=d.get('mechanics',{})
    sbl=d.get('stats_by_level',{})
    stats=_transpose_sbl(sbl,lvl)
    atk=d.get('attack',{})
    ch=mech.get('charge')
    if ch:
        dist=ch.get('charge_distance_tiles',2.5)
        cfg['components'].append(Charge(dist))
        cfg['charge_dmg']=stats.get('charge_damage',cfg['dmg']*2)
    jmp=mech.get('jump')
    if jmp:
        td=jmp.get('trigger_distance_tiles',{})
        mn=td.get('min',3.5);mx=td.get('max',5.0)
        sr=jmp.get('splash_radius_tiles',3.5)
        js=jmp.get('jump_speed',750)/60.0
        cfg['components'].append(MKJump(mn,mx,sr,js))
        jdk=jmp.get('damage_key','spawn_jump_damage')
        cfg['jump_dmg']=stats.get(jdk,stats.get('spawn_jump_damage',cfg['dmg']*2))
    szap=d.get('spawn_zap')
    if szap:
        cfg['components'].append(SpawnZap())
        cfg['spawn_zap_dmg']=stats.get('spawn_zap_damage',0)
        cfg['spawn_zap_r']=szap.get('radius_tiles',3.0)
    if atk.get('damage_type')=='dual_zap_split':
        cfg['components'].append(DualTarget())
        cfg['stun_dur']=atk.get('stun_duration_sec',0.5)
    elif atk.get('stun_duration_sec') and not cfg.get('stun_dur'):
        cfg['stun_dur']=atk['stun_duration_sec']
    dd=mech.get('death_damage',{}) or mech.get('death_bomb',{})
    if dd.get('enabled'):
        cfg['components'].append(DeathDamage())
        dk=dd.get('damage_key','death_damage')
        cfg['death_dmg']=stats.get(dk,stats.get('death_damage',0))
        cfg['death_splash_r']=dd.get('splash_radius_tiles',2.0)
    dn=mech.get('death_nova')
    if dn:
        sp=dn.get('slow_percent',0)
        sd=dn.get('slow_duration_sec',0)
        cfg['components'].append(DeathNova(abs(sp),sd))
        cfg['death_dmg']=stats.get('death_damage',0)
        cfg['death_splash_r']=dn.get('radius_tiles',2.0)
    if atk.get('self_destruct_on_hit'):
        cfg['is_suicide']=True
    ddrop=atk.get('death_drop')
    if ddrop:
        cfg['death_dmg']=stats.get('death_damage',0)
        cfg['death_splash_r']=ddrop.get('splash_radius_tiles',2.0)
        cfg['components'].append(DeathDamage())
        sc=ddrop.get('spawns_skeletons',0)
        if sc>0:
            su=d.get('sub_units',{})
            sn=list(su.keys())[0] if su else 'Skeleton'
            scfg=_build_sub_cfg(su.get(sn,{}),lvl)
            scfg['hp']=stats.get('skeleton_hitpoints',scfg['hp'])
            scfg['dmg']=stats.get('skeleton_damage',scfg['dmg'])
            cfg['components'].append(DeathSpawn(scfg,sc))
    sh=mech.get('shield',{})
    sh_on=sh.get('enabled') if isinstance(sh,dict) else bool(sh)
    if sh_on:
        sk=sh.get('shield_hp_key','shield_hitpoints') if isinstance(sh,dict) else 'shield_hitpoints'
        sv=stats.get(sk,0)
        cfg['shield_hp']=sv;cfg['max_shield_hp']=sv
    ru=atk.get('mechanics',{}).get('ramp_up') or atk.get('damage_ramp')
    if ru:
        dph=stats.get('damage_per_hit',{})
        stgs=[];durs=[]
        ns=ru.get('stages',3);ns=len(ns) if isinstance(ns,list) else int(ns)
        for i in range(1,ns+1):
            sv=dph.get(f'stage_{i}',stats.get(f'damage_stage{i}',cfg['dmg']))
            stgs.append(sv)
            dk=f'stage_{i}_duration_sec'
            if dk in ru:durs.append(ru[dk])
        if not durs and ru.get('stage_change_time_sec'):
            sct=ru['stage_change_time_sec']
            durs=[sct]*(ns-1)
        if not durs:durs=[2.0]*(ns-1)
        if stgs:cfg['dmg']=stgs[0]
        cfg['ramp_stages']=stgs;cfg['ramp_durations']=durs
        cfg['components'].append(RampUp(stgs,durs))
    sd=mech.get('spawn_damage')
    if sd and sd.get('enabled'):
        cfg['components'].append(SpawnZap())
        dk=sd.get('damage_key','spawn_damage')
        cfg['spawn_zap_dmg']=stats.get(dk,0)
        cfg['spawn_zap_r']=sd.get('radius_tiles',3.0)
        if not cfg.get('stun_dur'):cfg['stun_dur']=0
    sl=mech.get('slow_effect')
    if sl:
        cfg['slow_dur']=sl.get('duration_sec',2.0)
        cfg['slow_val']=1.0-sl.get('speed_reduction_percent',35)/100.0
    dm=d.get('death_mechanic',{})
    rg=dm.get('rage')
    if rg:
        cfg['components'].append(RageDrop(rg['radius_tiles'],rg['duration_sec'],rg['boost_percent']/100.0))
    ds=mech.get('death_spawn')
    if ds:
        su=d.get('sub_units',{})
        su_name=ds['unit'];su_cnt=ds['count']
        sub_cfg=_build_sub_cfg(su.get(su_name,{}),lvl,parent_su=su)
        cfg['components'].append(DeathSpawn(sub_cfg,su_cnt))
    sp=d.get('spawns')
    if sp and sp.get('unit'):
        su=d.get('sub_units',{})
        su_name=sp['unit']
        sub_cfg=_build_sub_cfg(su.get(su_name,{}),lvl)
        cfg['components'].append(SpawnTimer(sub_cfg,sp['spawn_interval_sec'],
            sp.get('spawn_count_per_interval',1),sp.get('spawn_first_delay_sec',1.0),
            sp.get('spawn_pattern','')))
        dsc=sp.get('death_spawn_count',0)
        if dsc>0:
            cfg['components'].append(DeathSpawn(sub_cfg,dsc))
    ms=mech.get('spawn')
    if ms and ms.get('unit'):
        su=d.get('sub_units',{})
        su_name=ms['unit']
        sub_cfg=_build_sub_cfg(su.get(su_name,{}),lvl)
        pfx=su_name.lower().replace(' ','_')+'_'
        if sub_cfg['hp']==0:
            sub_cfg['hp']=stats.get(pfx+'hitpoints',0)
            sub_cfg['dmg']=stats.get(pfx+'damage',sub_cfg['dmg'])
        sub_cfg['name']=su_name
        cnt=ms.get('count_per_wave',1)
        intv=ms['spawn_interval_sec']
        cfg['components'].append(SpawnTimer(sub_cfg,intv,cnt,intv))
    rs=mech.get('reactive_spawn')
    if rs:
        su=d.get('sub_units',{})
        su_name=rs.get('spawned_unit','')
        sub_cfg=_build_sub_cfg(su.get(su_name,{}),lvl)
        sub_cfg['name']=su_name
        cnt=rs.get('count_per_spawn',1)
        intv=rs.get('spawn_speed_sec',2.0)
        cfg['components'].append(SpawnTimer(sub_cfg,intv,cnt,intv))
    pr=d.get('production')
    if pr:
        cfg['components'].append(ElixirProd(pr['interval_sec'],pr.get('elixir_per_tick',1)))
    hl=mech.get('healing')
    if hl:
        hps=stats.get('healing_per_second',0)
        hr=hl.get('radius_tiles',2.5)
        cfg['components'].append(HealBurst(hps,hr))
    zp=mech.get('zap_pack')
    if zp:
        zd=stats.get('zap_damage',0)
        zr=zp.get('reflect_range_tiles',2.5)
        zs=zp.get('stun_duration_sec',0.5)
        cfg['components'].append(ZapPack(zd,zr,zs))
    hoa=mech.get('heal_on_attack')
    if hoa:
        hpp=stats.get('heal_per_pulse',0)
        hr=hoa.get('heal_radius_tiles',4.0)
        pc=hoa.get('pulse_count',4)
        cfg['components'].append(HealPulse(hpp,hr,pc))
    spwn=d.get('spawning')
    if spwn and spwn.get('spawns_unit'):
        su=d.get('sub_units',{})
        su_name=spwn['spawns_unit']
        sub_cfg=_build_sub_cfg(su.get(su_name,{}),lvl)
        pfx='spawned_'+su_name.lower().replace(' ','_')+'_'
        if sub_cfg['hp']==0:
            sub_cfg['hp']=stats.get(pfx+'hitpoints',0)
            sub_cfg['dmg']=stats.get(pfx+'area_damage',stats.get(pfx+'damage',sub_cfg['dmg']))
        sub_cfg['name']=su_name
        cnt=spwn.get('spawn_count_per_spawn',1)
        intv=spwn.get('spawn_speed_sec',7.0)
        cfg['components'].append(SpawnTimer(sub_cfg,intv,cnt,intv))
        sod=spwn.get('spawn_on_death',{})
        if sod.get('enabled') and sod.get('count',0)>0:
            cfg['components'].append(DeathSpawn(sub_cfg,sod['count']))
def _build_spawn_spell(d,lvl,team,x,y):
    sa=d.get('spell_attributes',{})
    su_name=sa['spawns_unit']
    su_data=d.get('sub_units',{}).get(su_name,{})
    sbl=d.get('stats_by_level',{})
    stats=sbl.get(str(lvl))
    if not stats:
        ks=sorted(sbl.keys(),key=int)
        stats=sbl[ks[-1]] if ks else {}
    pf=su_name.lower()+'_'
    hp=stats.get(pf+'hitpoints',stats.get('hitpoints',0))
    dmg=stats.get(pf+'damage',stats.get('damage',0))
    mv=su_data.get('movement',{})
    atk=su_data.get('attack',{})
    sv=mv.get('speed',{}).get('value',60)
    spd=sv/60.0
    hspd=atk.get('hit_speed_sec',1.0)
    fhspd=atk.get('first_hit_speed_sec',hspd)
    rng_d=atk.get('range',{})
    rng=rng_d.get('tiles',0.5) if isinstance(rng_d,dict) else float(rng_d)
    tgts_raw=atk.get('targets','Ground')
    if isinstance(tgts_raw,list):tgts=list(tgts_raw)
    else:tgts=[str(tgts_raw)]
    tcfg={'hp':hp,'dmg':dmg,'hspd':hspd,'fhspd':fhspd,'spd':spd,
          'rng':rng,'targets':tgts,'transport':'Ground',
          'atk_type':'single_target','splash_r':0,'ct_dmg':0,
          'components':[],'lvl':lvl,'name':su_name}
    cnt=sa.get('spawn_count',3)
    return SpawnSpell(team,x,y,{'troop_cfg':tcfg,'count':cnt,'name':d.get('name','')})
def create(name,lvl,team,x,y,evolved=False,hero=False):
    d=_ld(name)
    if d.get('type')=='Spell':
        sh=d.get('shape')
        if sh:
            sbl=d.get('stats_by_level',{})
            stats=sbl.get(str(lvl))
            if not stats:
                ks=sorted(sbl.keys(),key=int)
                stats=sbl[ks[-1]] if ks else {}
            mech=d.get('mechanics',{})
            cfg={'dmg':stats.get('area_damage',0),'ct_dmg':stats.get('crown_tower_damage',0),
                 'range':sh['range_tiles'],'width':sh['width_tiles'],
                 'pushback':mech.get('pushback_distance_tiles',0),'name':d.get('name','')}
            return LogSpell(team,x,y,cfg)
        spwn=d.get('spawning')
        if spwn:
            su=d.get('sub_units',{})
            su_name=spwn.get('spawned_unit','Skeleton')
            sub_cfg=_build_sub_cfg(su.get(su_name,{}),lvl)
            sp_d=d.get('spell',{})
            cfg={'troop_cfg':sub_cfg,'total':spwn['total_count'],
                 'interval':spwn['spawn_interval_sec'],
                 'radius':sp_d.get('radius_tiles',3.0),
                 'dur':sp_d.get('duration_sec',9.0),
                 'name':d.get('name','')}
            return GraveyardSpell(team,x,y,cfg)
        sa=d.get('spell_attributes',{})
        if sa.get('clone_hp') is not None:
            return CloneSpell(team,x,y,{'radius':sa.get('radius_tiles',3.0),'name':d.get('name','')})
        if sa.get('target_selection','').startswith('3_highest'):
            sbl=d.get('stats_by_level',{})
            stats=sbl.get(str(lvl))
            if not stats:
                ks=sorted(sbl.keys(),key=int)
                stats=sbl[ks[-1]] if ks else {}
            cfg={'dmg':stats.get('damage',0),'ct_dmg':stats.get('crown_tower_damage',0),
                 'radius':sa.get('radius_tiles',3.5),'max_targets':sa.get('max_targets',3),
                 'stun_dur':sa.get('stun_duration_sec',0.5),'name':d.get('name','')}
            return LightningSpell(team,x,y,cfg)
        _sp=sa.get('spawns',{})
        if isinstance(_sp,dict) and _sp.get('unit'):
            sbl=d.get('stats_by_level',{})
            stats=sbl.get(str(lvl))
            if not stats:
                ks=sorted(sbl.keys(),key=int)
                stats=sbl[ks[-1]] if ks else {}
            su=d.get('sub_units',{});su_name=_sp['unit']
            su_data=su.get(su_name,{})
            pf=su_name.lower().replace(' ','_')+'_'
            shrt=su_name.lower().split()[-1]+'_' if ' ' in su_name else pf
            mv=su_data.get('movement',{});atk=su_data.get('attack',{})
            _hspd=atk.get('hit_speed_sec',1.3);_fhspd=atk.get('first_hit_speed_sec',_hspd)
            rng_d=atk.get('range',{})
            _rng=rng_d.get('tiles',1.6) if isinstance(rng_d,dict) else float(rng_d)
            _sv=mv.get('speed',{}).get('value',60);_spd=_sv/60.0
            tgts_raw=atk.get('targets','Ground')
            _tgts=[str(tgts_raw)] if not isinstance(tgts_raw,list) else list(tgts_raw)
            _hp=stats.get(pf+'hitpoints',stats.get(shrt+'hitpoints',0))
            _shp=stats.get(pf+'shield_hitpoints',stats.get(shrt+'shield_hitpoints',0))
            _dmgr=stats.get(pf+'damage',stats.get(shrt+'damage',0))
            _sh=su_data.get('mechanics',{}).get('shield',False)
            tcfg={'hp':_hp,'dmg':_dmgr,'hspd':_hspd,'fhspd':_fhspd,'spd':_spd,
                  'rng':_rng,'targets':_tgts,'transport':'Ground',
                  'atk_type':'single_target','splash_r':0,'ct_dmg':0,
                  'components':[],'lvl':lvl,'name':su_name}
            if _sh:tcfg['shield_hp']=_shp;tcfg['max_shield_hp']=_shp
            cfg={'dmg':stats.get('area_damage',0),'ct_dmg':stats.get('crown_tower_damage',0),
                 'radius':sa.get('radius_tiles',3.0),'troop_cfg':tcfg,
                 'name':d.get('name','')}
            return RoyalDeliverySpell(team,x,y,cfg)
        if sa.get('spawns_unit'):
            return _build_spawn_spell(d,lvl,team,x,y)
        if sa.get('speed_boost_percent',0)>0:
            sbl=d.get('stats_by_level',{})
            stats=sbl.get(str(lvl))
            if not stats:
                ks=sorted(sbl.keys(),key=int)
                stats=sbl[ks[-1]] if ks else {}
            cfg={'dmg':stats.get('area_damage',0),'ct_dmg':stats.get('crown_tower_damage',0),
                 'radius':sa.get('radius_tiles',3.0),
                 'rage_boost':sa['speed_boost_percent']/100.0,
                 'rage_dur':sa.get('duration_sec',4.5),
                 'name':d.get('name','')}
            return RageSpell(team,x,y,cfg)
        su_spell=d.get('spawned_unit')
        if su_spell and sa.get('range_tiles'):
            sbl=d.get('stats_by_level',{})
            stats=sbl.get(str(lvl))
            if not stats:
                ks=sorted(sbl.keys(),key=int)
                stats=sbl[ks[-1]] if ks else {}
            cfg={'dmg':stats.get('barrel_area_damage',stats.get('area_damage',0)),
                 'ct_dmg':0,'range':sa['range_tiles'],'width':sa.get('width_tiles',2.6),
                 'pushback':0,'name':d.get('name','')}
            if su_spell.get('name')=='Barbarian':
                pf='barbarian_'
                _hspd=su_spell.get('hit_speed_sec',1.3)
                _fhspd=su_spell.get('first_hit_speed_sec',0.4)
                _sv=su_spell.get('movement_speed',{}).get('value',60);_spd=_sv/60.0
                _rng=su_spell.get('range_tiles',0.5)
                tcfg={'hp':stats.get(pf+'hitpoints',0),'dmg':stats.get(pf+'damage',0),
                      'hspd':_hspd,'fhspd':_fhspd,'spd':_spd,'rng':_rng,
                      'targets':['Ground'],'transport':'Ground',
                      'atk_type':'single_target','splash_r':0,'ct_dmg':0,
                      'components':[],'lvl':lvl,'name':'Barbarian'}
                cfg['troop_cfg']=tcfg
                return BarbarianBarrelSpell(team,x,y,cfg)
            return LogSpell(team,x,y,cfg)
        if sa.get('damage_scaling'):
            sbl=d.get('stats_by_level',{})
            stats=sbl.get(str(lvl))
            if not stats:
                ks=sorted(sbl.keys(),key=int)
                stats=sbl[ks[-1]] if ks else {}
            cfg={'single_dmg':stats.get('damage_single_target',0),
                 'ct_dmg':stats.get('crown_tower_damage',0),
                 'radius':sa.get('radius_tiles',2.5),
                 'strikes':sa.get('strikes',3),
                 'interval':sa.get('strike_interval_sec',1.0),
                 'name':d.get('name','')}
            return VoidSpell(team,x,y,cfg)
        if sa.get('grounds_air'):
            sbl=d.get('stats_by_level',{})
            stats=sbl.get(str(lvl))
            if not stats:
                ks=sorted(sbl.keys(),key=int)
                stats=sbl[ks[-1]] if ks else {}
            cfg={'radius':sa.get('radius_tiles',2.5),
                 'max_targets':sa.get('max_targets',3),
                 'dur':sa.get('duration_sec',2.0),
                 'tick_dmg':stats.get('damage_per_tick',80),
                 'tick_interval':sa.get('tick_interval_sec',0.5),
                 'name':d.get('name','')}
            return VinesSpell(team,x,y,cfg)
        if sa.get('conversion'):
            sbl=d.get('stats_by_level',{})
            stats=sbl.get(str(lvl))
            if not stats:
                ks=sorted(sbl.keys(),key=int)
                stats=sbl[ks[-1]] if ks else {}
            su=d.get('sub_units',{})
            su_name=list(su.keys())[0] if su else 'Converted Goblin'
            gcfg=_build_sub_cfg(su.get(su_name,{}),lvl)
            gcfg['name']=su_name
            cfg={'radius':sa.get('radius_tiles',3.0),
                 'tick_dmg':stats.get('damage_per_tick',35),
                 'ct_dmg':stats.get('crown_tower_damage_per_tick',0),
                 'ticks':sa.get('total_ticks',6),
                 'interval':sa.get('tick_interval_sec',1.0),
                 'goblin_cfg':gcfg,
                 'name':d.get('name','')}
            return GoblinCurseSpell(team,x,y,cfg)
        if d.get('pull_strength'):
            lv=d['levels']
            stats=None
            for s in lv['stats']:
                if s['level']==lvl:stats=s;break
            if not stats:stats=lv['stats'][-1]
            hi=float(str(d.get('hit_speed','0.55s')).rstrip('s'))
            ps_raw=d.get('pull_strength','360%')
            ps=float(str(ps_raw).rstrip('%'))/100.0
            dur=float(str(d.get('duration','1.05s')).rstrip('s'))
            ticks=int(round(dur/hi)) if hi>0 else 2
            cfg={'radius':float(d.get('radius',5.5)),
                 'tick_dmg':stats.get('damage_per_tick',45),
                 'ct_dmg':stats.get('crown_tower_damage_per_tick',14),
                 'ticks':ticks,'interval':hi,
                 'pull_str':ps,'dur':dur,
                 'name':d.get('name','')}
            return TornadoSpell(team,x,y,cfg)
        if d.get('ticks') and d.get('targets')=='Ground':
            lv=d['levels']
            stats=None
            for s in lv['stats']:
                if s['level']==lvl:stats=s;break
            if not stats:stats=lv['stats'][-1]
            ti=float(str(d.get('tick_interval','1s')).rstrip('s'))
            se_raw=d.get('slow_effect','')
            sp=abs(int(str(se_raw).rstrip('%')))/100.0 if se_raw else 0.5
            cfg={'radius':float(d.get('radius',3.5)),
                 'troop_dmg':stats.get('troop_damage_per_tick',55),
                 'bldg_dmg':stats.get('building_damage_per_tick',180),
                 'ct_dmg':stats.get('crown_tower_damage_per_tick',17),
                 'ticks':int(d['ticks']),'interval':ti,
                 'slow_pct':sp,
                 'name':d.get('name','')}
            return EarthquakeSpell(team,x,y,cfg)
        scfg=_parse_spell(d,lvl)
        if evolved:
            evo=d.get('evolution',{})
            mech=evo.get('mechanic','')
            if mech=='Double Pulse' or 'zap' in d.get('name','').lower():
                scfg['pulse_2_radius']=evo.get('pulse_2_radius',3.0)
                return EvoZapSpell(team,x,y,scfg)
            if mech=='Capture & Roll' or 'snowball' in d.get('name','').lower():
                scfg['roll_distance']=float(evo.get('roll_distance','4.5').split()[0]) if isinstance(evo.get('roll_distance'),str) else evo.get('roll_distance',4.5)
                scfg['roll_duration']=float(evo.get('roll_duration','0.75').split('s')[0]) if isinstance(evo.get('roll_duration'),str) else evo.get('roll_duration',0.75)
                scfg['slow_duration']=float(evo.get('slow_duration','4').split('s')[0]) if isinstance(evo.get('slow_duration'),str) else evo.get('slow_duration',4.0)
                return EvoSnowballSpell(team,x,y,scfg)
        return Spell(team,x,y,scfg)
    comps_d=d.get('components')
    if comps_d and 'body' in comps_d and 'rocket_launcher' in comps_d:
        body=comps_d['body'];rl=comps_d['rocket_launcher']
        bsbl=body.get('stats_by_level',{})
        bst=bsbl.get(str(lvl))
        if not bst:
            bks=sorted(bsbl.keys(),key=int)
            bst=bsbl[bks[-1]] if bks else {}
        rsbl=rl.get('stats_by_level',{})
        rst=rsbl.get(str(lvl))
        if not rst:
            rks=sorted(rsbl.keys(),key=int)
            rst=rsbl[rks[-1]] if rks else {}
        batk=body.get('attack',{})
        bhspd=batk.get('hit_speed_sec',1.2)
        bfhspd=batk.get('first_hit_speed_sec',bhspd)
        brng_d=batk.get('range',{})
        brng=brng_d.get('tiles',1.2) if isinstance(brng_d,dict) else float(brng_d)
        mv=d.get('movement',{})
        sv=mv.get('speed',{}).get('value',60)
        spd=sv/60.0
        tgts_raw=body.get('targets',['Ground'])
        tgts=list(tgts_raw) if isinstance(tgts_raw,list) else [str(tgts_raw)]
        ratk=rl.get('attack',{})
        rhspd=ratk.get('hit_speed_sec',3.5)
        rfhspd=ratk.get('first_hit_speed_sec',rhspd)
        rrng=ratk.get('range',{})
        rmin=rrng.get('min_tiles',2.5)
        rmax=rrng.get('max_tiles',5.0)
        rsr=ratk.get('area_damage_radius_tiles',1.5)
        rdmg=rst.get('damage',0)
        bcomps=[RocketLauncher(rdmg,rhspd,rfhspd,rmin,rmax,rsr)]
        cfg={'hp':bst.get('hitpoints',0),'dmg':bst.get('damage',0),
             'hspd':bhspd,'fhspd':bfhspd,'spd':spd,'rng':brng,
             'targets':tgts,'transport':mv.get('transport','Ground'),
             'atk_type':'single_target','splash_r':0,'ct_dmg':0,
             'components':bcomps,'lvl':lvl,'name':d.get('name','')}
        return Troop(team,x,y,cfg)
    ents=d.get('entities')
    if ents:
        sbl=d.get('stats_by_level',{})
        stats=sbl.get(str(lvl))
        if not stats:
            ks=sorted(sbl.keys(),key=int)
            stats=sbl[ks[-1]] if ks else {}
        mv=d.get('movement',{})
        sv=mv.get('speed_value',mv.get('speed',{}).get('value',60))
        spd=sv/60.0
        troops=[]
        for ename,edata in ents.items():
            es=stats.get(ename,{})
            eatk=edata.get('attack',{})
            ehspd=eatk.get('hit_speed_sec',1.0)
            efhspd=eatk.get('first_hit_speed_sec',ehspd)
            erng_d=eatk.get('range',{})
            erng=erng_d.get('tiles',1.2) if isinstance(erng_d,dict) else float(erng_d)
            etgts_raw=eatk.get('targets','Ground')
            etgts=list(etgts_raw) if isinstance(etgts_raw,list) else [str(etgts_raw)]
            ecomps=[]
            if etgts==['Buildings']:ecomps.append(BuildingTarget())
            estn=eatk.get('stun_duration_sec',0)
            ecfg={'hp':es.get('hitpoints',0),'dmg':es.get('damage',0),
                  'hspd':ehspd,'fhspd':efhspd,'spd':spd,'rng':erng,
                  'targets':etgts,'transport':edata.get('transport','Ground'),
                  'atk_type':'single_target','splash_r':0,'ct_dmg':0,
                  'components':ecomps,'lvl':lvl,'name':ename}
            if estn>0:ecfg['stun_dur']=estn
            hs=d.get('hidden_stats',{}).get(ename,{})
            if isinstance(hs,dict):ecfg['mass']=hs.get('mass',4)
            ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
            tr=Troop(team,x+ox,y+oy,ecfg)
            if ename=='monster' and d.get('ability'):
                _add_champion(d,tr,lvl)
            troops.append(tr)
        return troops
    forms=d.get('forms')
    if forms:
        sbl=d.get('stats_by_level',{})
        stats=sbl.get(str(lvl))
        if not stats:
            ks=sorted(sbl.keys(),key=int)
            stats=sbl[ks[-1]] if ks else {}
        mv=d.get('movement',{})
        sv=mv.get('speed',{}).get('value',60)
        spd=sv/60.0
        ef=forms.get('empress',{})
        sf=forms.get('spirit',{})
        eatk=ef.get('attack',{})
        ehspd=eatk.get('hit_speed_sec',1.5)
        efhspd=eatk.get('first_hit_speed_sec',ehspd)
        erng_d=eatk.get('range',{})
        erng=erng_d.get('tiles',5.0) if isinstance(erng_d,dict) else float(erng_d)
        etgts_raw=eatk.get('targets',['Air','Ground'])
        etgts=list(etgts_raw) if isinstance(etgts_raw,list) else [str(etgts_raw)]
        esr=eatk.get('splash_radius_tiles',0)
        edt=eatk.get('damage_type','single_target')
        satk=sf.get('attack',{})
        shspd=satk.get('hit_speed_sec',1.0)
        sfhspd=satk.get('first_hit_speed_sec',shspd)
        srng_d=satk.get('range',{})
        srng=srng_d.get('tiles',4.0) if isinstance(srng_d,dict) else float(srng_d)
        stgts_raw=satk.get('targets',['Air','Ground'])
        stgts=list(stgts_raw) if isinstance(stgts_raw,list) else [str(stgts_raw)]
        scfg={'hp':stats.get('spirit_hitpoints',0),'dmg':stats.get('spirit_damage',0),
              'hspd':shspd,'fhspd':sfhspd,'spd':spd,'rng':srng,
              'targets':stgts,'transport':'Ground',
              'atk_type':'single_target','splash_r':0,'ct_dmg':0,
              'components':[],'lvl':lvl,'name':'Spirit'}
        ecomps=[]
        if edt in ('splash','area') and esr>0:ecomps.append(SplashAttack())
        ecomps.append(FormTransform(scfg))
        ecfg={'hp':stats.get('empress_hitpoints',0),'dmg':stats.get('empress_damage',0),
              'hspd':ehspd,'fhspd':efhspd,'spd':spd,'rng':erng,
              'targets':etgts,'transport':'Ground',
              'atk_type':edt,'splash_r':esr,'ct_dmg':0,
              'components':ecomps,'lvl':lvl,'name':d.get('name','')}
        return Troop(team,x,y,ecfg)
    sbl_check=d.get('stats_by_level',{})
    stcheck=_transpose_sbl(sbl_check,lvl)
    if stcheck.get('bush_hitpoints') is not None:
        mv=d.get('movement',{})
        sv=mv.get('speed',{}).get('value',60)
        spd=sv/60.0
        su=d.get('sub_units',{})
        su_name=list(su.keys())[0] if su else 'Bush Goblin'
        su_data=su.get(su_name,{})
        su_atk=su_data.get('attack',{})
        su_mv=su_data.get('movement',{})
        su_sv=su_mv.get('speed',{}).get('value',60)
        su_spd=su_sv/60.0
        su_hspd=su_atk.get('hit_speed_sec',1.4)
        su_fhspd=su_atk.get('first_hit_speed_sec',su_hspd)
        su_rng_d=su_atk.get('range',{})
        su_rng=su_rng_d.get('tiles',0.8) if isinstance(su_rng_d,dict) else float(su_rng_d)
        su_tgts_raw=su_atk.get('targets','Ground')
        su_tgts=list(su_tgts_raw) if isinstance(su_tgts_raw,list) else [str(su_tgts_raw)]
        pfx=su_name.lower().replace(' ','_')+'_'
        gcfg={'hp':stcheck.get(pfx+'hitpoints',0),'dmg':stcheck.get(pfx+'damage',0),
              'hspd':su_hspd,'fhspd':su_fhspd,'spd':su_spd,'rng':su_rng,
              'targets':su_tgts,'transport':'Ground',
              'atk_type':'single_target','splash_r':0,'ct_dmg':0,
              'components':[],'lvl':lvl,'name':su_name}
        spawn_cnt=d.get('mechanics',{}).get('spawns_on_arrival_or_death',{}).get('count',2)
        bcomps=[BuildingTarget(),DeathSpawn(gcfg,spawn_cnt)]
        bcfg={'hp':stcheck.get('bush_hitpoints',0),'dmg':0,
              'hspd':1.0,'fhspd':1.0,'spd':spd,'rng':1.2,
              'targets':['Buildings'],'transport':'Ground',
              'atk_type':'single_target','splash_r':0,'ct_dmg':0,
              'components':bcomps,'lvl':lvl,'name':d.get('name','')}
        return Troop(team,x,y,bcfg)
    catk=d.get('composite_attack')
    if catk:
        sbl=d.get('stats_by_level',{})
        stats=sbl.get(str(lvl))
        if not stats:
            ks=sorted(sbl.keys(),key=int)
            stats=sbl[ks[-1]] if ks else {}
        ram=catk.get('ram',{})
        rider=catk.get('rider',{})
        mv=d.get('movement',{})
        sv=mv.get('speed',{}).get('value',60)
        spd=sv/60.0
        hp=stats.get('hitpoints',0)
        dmg=stats.get('ram_damage',stats.get('damage',0))
        rng_d=ram.get('range',{})
        rng=rng_d.get('tiles',0.8) if isinstance(rng_d,dict) else float(rng_d)
        hspd=ram.get('hit_speed_sec',1.8)
        tgts=ram.get('targets',['Buildings'])
        if not isinstance(tgts,list):tgts=[tgts]
        comps=[]
        if tgts==['Buildings']:comps.append(BuildingTarget())
        ch=ram.get('charge',{})
        if ch:
            cr=ch.get('charge_range_tiles',2.5)
            comps.append(Charge(cr))
        if mv.get('can_jump_river'):comps.append(RiverJump())
        cfg={'hp':hp,'dmg':dmg,'hspd':hspd,'fhspd':hspd,'spd':spd,
             'rng':rng,'targets':tgts,'transport':mv.get('transport','Ground'),
             'atk_type':'single_target','splash_r':0,'ct_dmg':0,
             'components':comps,'lvl':lvl,'name':d.get('name',''),
             'charge_dmg':stats.get('ram_charge_damage',dmg*2),
             'mass':d.get('hidden_stats',{}).get('mass') or 4,
             'sight_r':d.get('hidden_stats',{}).get('sight_range_tiles') or 5.5,
             'collision_r':d.get('hidden_stats',{}).get('collision_radius_tiles') or 0.5}
        return Troop(team,x,y,cfg)
    su=d.get('sub_units');cnt=d.get('count',1)
    comp=d.get('composition')
    if comp and su:
        sbl=d.get('stats_by_level',{})
        stats=sbl.get(str(lvl))
        if not stats:
            ks=sorted(sbl.keys(),key=int)
            stats=sbl[ks[-1]] if ks else {}
        troops=[]
        for _,info in comp.items():
            sn=info['type'];sc=info['count']
            scfg=_parse_new(d,lvl,sub=su.get(sn,{}))
            pfx=sn.lower().replace(' ','_')+'_'
            shrt=sn.lower().split()[-1]+'_' if ' ' in sn else pfx
            hp_k=pfx+'hitpoints' if pfx+'hitpoints' in stats else shrt+'hitpoints'
            dm_k=pfx+'damage' if pfx+'damage' in stats else shrt+'damage'
            scfg['hp']=stats.get(hp_k,scfg['hp'])
            scfg['dmg']=stats.get(dm_k,scfg['dmg'])
            scfg['name']=sn
            _add_mechanics(d,scfg,lvl)
            for _ in range(sc):
                ox=random.uniform(-1.5,1.5);oy=random.uniform(-1.5,1.5)
                troops.append(Troop(team,x+ox,y+oy,copy.deepcopy(scfg)))
        return troops
    if su and cnt>1:
        sk=list(su.keys())[0]
        cfg=_parse_new(d,lvl,sub=su[sk])
    elif 'levels' in d:
        lv=d['levels']
        if isinstance(lv,dict) and 'stats' not in lv and any(k.isdigit() for k in lv):
            d2=dict(d,stats_by_level=lv)
            cfg=_parse_new(d2,lvl)
        else:
            cfg=_parse_old(d,lvl)
    else:
        cfg=_parse_new(d,lvl)
    _add_mechanics(d,cfg,lvl)
    if cnt<=1:
        if d.get('type')=='Building':
            cfg['lifetime']=d.get('building_attributes',{}).get('lifetime_sec',
                d.get('building_stats',{}).get('lifetime_sec',
                d.get('lifetime_sec',30.0)))
            bld=Building(team,x,y,cfg)
            if evolved:_add_evolution(d,bld,lvl)
            return bld
        tr=Troop(team,x,y,cfg)
        _add_champion(d,tr,lvl)
        if evolved:_add_evolution(d,tr,lvl)
        if hero:_add_hero(d,tr,lvl)
        return tr
    troops=[]
    for i in range(cnt):
        ox=random.uniform(-1.5,1.5);oy=random.uniform(-1.5,1.5)
        c=copy.deepcopy(cfg)
        tr=Troop(team,x+ox,y+oy,c)
        if evolved:_add_evolution(d,tr,lvl)
        troops.append(tr)
    if hero and troops:_add_hero(d,troops[0],lvl,troops)
    return troops
def _add_champion(d,tr,lvl):
    ab=d.get('ability')
    if not ab:return
    nm=d.get('name','')
    sbl=d.get('stats_by_level',{})
    stats=sbl.get(str(lvl))
    if not stats:
        ks=sorted(sbl.keys(),key=int)
        stats=sbl[ks[-1]] if ks else {}
    aname=ab.get('name','')
    if aname=='Dashing Dash':
        da=ab.get('dash',{})
        asbl=d.get('ability_stats_by_level',{}).get('dashing_dash',{})
        ast=asbl.get(str(lvl),{})
        dd=ast.get('dash_damage',335)
        mxd=da.get('max_dashes',10)
        sr=da.get('chain_search_radius_tiles',5.5)
        tr.ability=DashingDash(dd,mxd,sr,ab.get('elixir_cost',1),ab.get('cooldown_sec',8.0))
    elif aname=='Soul Summoning':
        tr.components.append(SoulCollect(ab.get('soul_capacity',10)))
        su=d.get('sub_units',{})
        sn=ab.get('summoned_unit','Soul Skeleton (cloned)')
        scfg=_build_sub_cfg(su.get(sn,{}),lvl)
        scfg['name']=sn
        tr.ability=SoulSummoning(scfg,ab.get('summon_radius_tiles',3.5),ab.get('elixir_cost',2),ab.get('cooldown_sec',20.0))
    elif aname=='Getaway Grenade':
        mech=d.get('mechanics',{})
        da=mech.get('dash',{})
        if da:
            band=da.get('trigger_target_band_tiles',{})
            mn=band.get('min',3.5);mx=band.get('max',6.0)
            ct=da.get('dash_charge_time_sec',0.8)
            tr.components.append(BanditDash(mn,mx,ct))
            tr.dash_dmg=stats.get('dash_damage',tr.dmg*2)
        ef=ab.get('effects',{})
        tp=ef.get('teleport',{})
        iv=ef.get('invisibility',{})
        tr.ability=GetawayGrenade(tp.get('distance_tiles',6.0),iv.get('duration_sec',1.0),
            ab.get('elixir_cost',1),ab.get('cooldown_sec',3.0),ab.get('uses_per_deploy',2))
    elif aname=='Cloaking Cape':
        ef=ab.get('effects',{})
        mv=ef.get('movement_speed_override',{})
        asb=ef.get('attack_speed_boost',{})
        tr.ability=CloakingCape(ab.get('duration_sec',3.5),mv.get('speed_value',45),
            asb.get('boost_percent',180)/100.0,ab.get('elixir_cost',1),ab.get('cooldown_sec',17.0))
    elif aname=='Explosive Escape':
        ef=ab.get('effects',{})
        bm=ef.get('bomb',{})
        bdl=bm.get('area_damage_by_level',{})
        bd=bdl.get(str(lvl),328)
        tr.ability=ExplosiveEscape(bd,3.0,bm.get('knockback_distance_tiles',1.8),
            ab.get('elixir_cost',1),ab.get('cooldown_sec',13.0))
    elif aname=='Lightning Link':
        ef=ab.get('effects',{})
        lk=ef.get('link',{})
        tdl=ef.get('tick_damage_by_level',{}).get(str(lvl),{})
        td=tdl.get('damage',107);tc=tdl.get('crown_tower_damage',46)
        tr.ability=LightningLink(td,tc,lk.get('radius_tiles',2.0),
            ab.get('duration_sec',4.0),lk.get('tick_interval_sec',0.5),
            ab.get('elixir_cost',2),ab.get('cooldown_sec',17.0))
    elif aname=='Royal Rescue':
        su=d.get('sub_units',{})
        gdata=su.get('Guardienne',{})
        gcfg=_build_sub_cfg(gdata,lvl)
        gcfg['name']='Guardienne'
        asbl=d.get('ability_stats_by_level',{}).get('royal_rescue_charge_damage',{})
        ast=asbl.get(str(lvl),{})
        cdmg=ast.get('charge_damage',256)
        atk=d.get('attack',{})
        ru=atk.get('ramp_up',{})
        if ru:
            hs=atk.get('hit_speed_sec_by_stage',{})
            stgs=[hs.get('stage_1',1.2),hs.get('stage_2',0.6),hs.get('stage_3',0.4)]
            per=ru.get('attacks_per_stage_increase',3)
            tr.components.append(LPRamp(stgs,per))
        tr.ability=RoyalRescue(gcfg,cdmg,2.5,ab.get('elixir_cost',3),ab.get('cooldown_sec',30.0))
    elif aname=='Pensive Protection':
        ef=ab.get('effects',{})
        dr=ef.get('damage_reduction',{})
        pct=dr.get('all_damage_types_percent',65)/100.0
        co=ab.get('attack',{}).get('mechanics',{}).get('combo',{})
        if not co:co=d.get('attack',{}).get('mechanics',{}).get('combo',{})
        if co:
            cy=co.get('hits_per_cycle',3)
            kb=co.get('combo_knockback',{}).get('knockback_distance_tiles',1.8)
            tr.components.append(MonkCombo(cy,kb))
        tr.ability=PensiveProtection(pct,ab.get('duration_sec',4.0),
            ab.get('elixir_cost',1),ab.get('cooldown_sec',17.0))
def _add_evolution(d,tr,lvl):
    evo=d.get('evolution',{})
    if not evo:return
    nm=d.get('name','').lower().replace(' ','_')
    ch=evo.get('changes',{})
    mech=evo.get('mechanic','')
    if ch.get('power_shot') or mech=='Power Shot':
        tr.components.append(EvoArchers(ch.get('power_shot_min_range_tiles',4.0),
            ch.get('power_shot_max_range_tiles',6.0),ch.get('power_shot_damage_multiplier',1.5)))
    elif mech=='Damage Reduction' or (nm=='knight'):
        tr.components.append(EvoKnight(0.6))
    elif mech=='Bouncing Bomb' or 'bomber' in nm:
        tr.components.append(EvoBomber(2,2.5))
    elif mech=='Replicate on Attack' or 'skeleton' in nm and 'barrel' not in nm and 'dragon' not in nm:
        tr.components.append(EvoSkeletons(8))
    elif ch.get('attack_speed_boost_percent') or 'barbarian' in nm and 'elite' not in nm:
        tr.components.append(EvoBarbarians(
            ch.get('hitpoints_multiplier',1.1),
            ch.get('attack_speed_boost_percent',30),
            ch.get('move_speed_boost_percent',30),
            ch.get('boost_duration_sec',3.0)))
        if ch.get('hitpoints_multiplier'):
            m=ch['hitpoints_multiplier']
            tr.max_hp=int(tr.max_hp*m);tr.hp=tr.max_hp
    elif mech=='Heal on Attack' or 'bat' in nm:
        tr.components.append(EvoBats(2,2.0))
    elif ch.get('charge_on_shield_break') or 'recruit' in nm:
        tr.components.append(EvoRoyalRecruits(
            ch.get('charge_damage_multiplier',2.0),ch.get('charge_activation_tiles',2.5)))
    elif ch.get('recoil_area_damage') or 'royal_giant' in nm:
        tr.components.append(EvoRoyalGiant(
            ch.get('recoil_radius_tiles',2.5),ch.get('recoil_pushback_tiles',1.0)))
    elif mech=='Ice Blast' or 'ice_spirit' in nm:
        tr.components.append(EvoIceSpirit(3.0))
    elif ch.get('drops_at_75_percent_hp') or 'skeleton_barrel' in nm:
        tr.components.append(EvoSkelBarrel(0.75,ch.get('death_damage_multiplier',1.64)))
        if ch.get('hitpoints_multiplier'):
            m=ch['hitpoints_multiplier']
            tr.max_hp=int(tr.max_hp*m);tr.hp=tr.max_hp
    elif ch.get('big_spark_trail') or 'firecracker' in nm:
        tr.components.append(EvoFirecracker(ch.get('small_sparks',5),abs(ch.get('spark_slow_percent',15))))
    elif ch.get('electric_pulse_on_surface') or 'tesla' in nm:
        from components import SpawnZap as _SZ
        tr.spawn_zap_dmg=tr.dmg;tr.spawn_zap_r=ch.get('pulse_radius_tiles',6.0)
        tr.stun_dur=ch.get('pulse_stun_duration_sec',0.5)
    elif ch.get('deploy_barrage') or 'cannon' in nm:
        pass
    elif ch.get('spawns_goblin_each_attack') or 'mortar' in nm:
        if ch.get('hit_speed_sec'):tr.hspd=ch['hit_speed_sec']
    elif ch.get('tornado_on_attack') or 'valkyrie' in nm:
        tr.components.append(EvoValkyrie(ch.get('tornado_radius_tiles',5.0),ch.get('tornado_damage',84),ch.get('tornado_duration_sec',0.5)))
    elif ch.get('sniper_mode') or 'musketeer' in nm:
        tr.components.append(EvoMusketeer(ch.get('sniper_ammo',3),ch.get('sniper_range_tiles',30.0),ch.get('sniper_damage_multiplier',1.8),ch.get('sniper_min_range_tiles',6.0)))
    elif ch.get('poison_darts') or 'dart_goblin' in nm:
        tiers=tuple([ch.get('tier_1_dps',51),ch.get('tier_2_dps',115),ch.get('tier_3_dps',307)])
        esc=tuple(ch.get('tier_escalation_darts',[1,4,7]))
        tr.components.append(EvoDartGoblin(ch.get('poison_radius_tiles',1.5),ch.get('poison_duration_sec',1.0),tiers,esc))
    elif ch.get('flying_start') or 'royal_hogs' in nm:
        tr.components.append(EvoRoyalHogs(ch.get('landing_damage',84),ch.get('landing_radius_tiles',2.0)))
        tr.transport='Air'
    elif ch.get('cage_pull') or 'goblin_cage' in nm:
        tr.components.append(EvoGoblinCage(ch.get('pull_radius_tiles',3.0)))
    elif mech=='Wind Aura' or 'wind_aura' in str(ch) or 'baby_dragon' in nm:
        tr.components.append(EvoBabyDragon())
    elif mech=='Heal on Skeleton Death' or 'heal_on_skeleton' in str(ch) or ('witch' in nm and 'witch' not in 'mother'):
        esl=evo.get('stats_by_level',{})
        est=esl.get(str(lvl),{}) if isinstance(esl,dict) else {}
        hv=est.get('skeleton_death_heal',ch.get('heal_per_skeleton',109))
        oc=ch.get('heal_on_skeleton_death',{}).get('overheal_cap_multiplier',ch.get('overheal_cap',1.24))
        tr.components.append(EvoWitch(hv,oc))
    elif mech=='Heal on Kill' or 'heal_on_kill' in str(ch) or 'on_kill' in str(evo) or ('p.e.k.k.a' in nm):
        esl=evo.get('stats_by_level',[])
        est=None
        for s in (esl if isinstance(esl,list) else []):
            if s.get('level')==lvl:est=s;break
        if not est and esl:est=esl[-1] if isinstance(esl,list) else {}
        if est:
            tr.components.append(EvoPekka(est.get('heal_per_kill_stage_1',160),est.get('heal_per_kill_stage_2',304),est.get('heal_per_kill_stage_3',577)))
        else:
            tr.components.append(EvoPekka())
    elif mech=='Goblin Spawner' or 'goblin_spawner' in str(ch) or 'goblin_giant' in nm:
        ec=EvoGoblinGiant(ch.get('hp_threshold',0.5),ch.get('spawn_interval_sec',2.2))
        tr.components.append(ec)
    elif mech=='Net Ability' or 'net_ability' in str(ch) or 'hunter' in nm:
        tr.components.append(EvoHunter(ch.get('net_duration_sec',3.0),ch.get('net_cooldown_sec',5.0)))
    elif mech=='Infinite Bounce' or 'infinite_bounce' in str(ch) or 'electro_dragon' in nm:
        tr.components.append(EvoElectroDragon(ch.get('damage_reduction',0.33)))
    elif mech=='Death Spawn Runners' or 'wall_breaker' in nm:
        rcfg={'hp':int(tr.max_hp*0.5),'dmg':int(tr.dmg*0.5),'hspd':1.0,'fhspd':0.5,
              'spd':2.0,'rng':0.5,'targets':['Buildings'],'transport':'Ground',
              'atk_type':'single_target','splash_r':0,'ct_dmg':0,
              'components':[],'lvl':tr.lvl,'name':'Runner','is_suicide':True}
        tr.components.append(EvoWallBreakers(rcfg,2))
    elif mech=='Axe Smash' or 'axe_smash' in str(ch) or 'executioner' in nm:
        tr.components.append(EvoExecutioner(ch.get('close_range_tiles',3.5),ch.get('damage_multiplier',2.0),ch.get('knockback_tiles',1.0)))
    elif mech=='Decoy Barrel' or 'decoy' in str(ch) or 'goblin_barrel' in nm:
        tr.components.append(EvoGoblinBarrel())
    elif mech=='Resurface' or 'resurface' in str(ch) or 'goblin_drill' in nm:
        tr.components.append(EvoGoblinDrill())
    elif 'uppercut' in str(ch) or 'mega_knight' in nm:
        tr.components.append(EvoMegaKnight(ch.get('uppercut',{}).get('knockback_tiles',4.0)))
    elif 'stage_4' in str(ch) or 'inferno_dragon' in nm:
        tr.components.append(EvoInfernoDragon(9.0,20.0,844))
    elif 'souldier' in str(ch).lower() or 'royal_ghost' in nm:
        sd=ch.get('souldiers',{})
        ssl=sd.get('souldier_stats_by_level',{}).get(str(lvl),{})
        tr.components.append(EvoRoyalGhost(sd.get('count',2),ssl.get('hitpoints',81),ssl.get('damage',261)))
    elif 'dash_cycle' in str(ch) or ('bandit' in nm and 'boss' not in nm):
        tr.components.append(EvoBandit())
    elif 'double_hook' in str(ch) or 'fisherman' in nm:
        tr.components.append(EvoFisherman())
    elif 'ghost_on_death' in str(evo) or 'lumberjack' in nm:
        gd=evo.get('ghost_on_death',{})
        tr.components.append(EvoLumberjack(gd.get('lifetime_sec',5.5)))
    elif 'enhanced_slow' in str(ch) or 'ice_wizard' in nm:
        tr.components.append(EvoIceWizard())
    tr.evolved=True
def _add_hero(d,tr,lvl,all_troops=None):
    hd=d.get('hero',{})
    if not hd:return
    ab=hd.get('ability',{})
    aname=ab.get('name','')
    tr.is_hero=True
    if aname=='Triumphant Taunt':
        shp_map=ab.get('shield_hp_pct_by_level',{})
        shp=shp_map.get(str(lvl),769)
        tr.ability=TriumphantTaunt(
            ab.get('effects',{}).get('taunt',{}).get('radius_tiles',ab.get('taunt_radius_tiles',6.5)),
            shp,ab.get('duration_sec',5.0),ab.get('elixir_cost',2),ab.get('cooldown_sec',25.0))
    elif aname=='Banner Brigade':
        bb=BannerBrigade(ab.get('spawns_count',4),ab.get('banner_duration_sec',7.0),ab.get('elixir_cost',1))
        sbl=d.get('levels',{})
        if isinstance(sbl,dict) and 'stats' in sbl:
            stats=None
            for s in sbl['stats']:
                if s['level']==lvl:stats=s;break
            if not stats:stats=sbl['stats'][-1]
        else:stats={}
        gcfg={'hp':stats.get('hp',204),'dmg':stats.get('damage',128),
              'hspd':tr.hspd,'fhspd':tr.fhspd,'spd':tr.spd,'rng':tr.rng,
              'targets':tr.targets,'transport':tr.transport,
              'atk_type':'single_target','splash_r':0,'ct_dmg':0,
              'components':[],'lvl':lvl,'name':'Goblin'}
        bb.set_base_cfg(gcfg)
        tr.ability=bb
        if all_troops:
            for t in all_troops:t.ability=bb;t.is_hero=True
    elif aname=='Heroic Hurl':
        sbl=d.get('stats_by_level',{})
        st=sbl.get(str(lvl),{})
        idmg=st.get('impact_damage',ab.get('impact_damage',64))
        tr.ability=HeroicHurl(ab.get('throw_range_tiles',9.0),ab.get('stun_duration_sec',2.0),
            idmg,ab.get('elixir_cost',2),ab.get('cooldown_sec',14.0))
    elif aname=='Breakfast Boost':
        tr.ability=BreakfastBoost(ab.get('healing_percent',30)/100.0,ab.get('elixir_cost',1))
    elif aname=='Trusty Turret':
        tcfg=ab.get('turret',{})
        sbl=d.get('stats_by_level',{})
        st=sbl.get(str(lvl),{})
        cfg={'hp':tcfg.get('hp',st.get('turret_hitpoints',726)),'dmg':tcfg.get('damage',st.get('turret_damage',66)),
             'hspd':tcfg.get('hit_speed_sec',0.5),'fhspd':tcfg.get('hit_speed_sec',0.5),
             'spd':0,'rng':tcfg.get('range_tiles',4.0),'targets':['Air','Ground'],
             'transport':'Ground','atk_type':'single_target','splash_r':0,'ct_dmg':0,
             'components':[],'lvl':lvl,'name':'Turret','is_building':True}
        tr.ability=TrustyTurret(cfg,ab.get('elixir_cost',3),ab.get('cooldown_sec',22.0))
    elif aname=='Snowstorm':
        tr.ability=Snowstorm(ab.get('blizzard_radius_tiles',4.0),ab.get('number_of_blasts',3),
            ab.get('freeze_duration_sec',1.5),ab.get('elixir_cost',2),ab.get('cooldown_sec',17.0))
    elif aname=='Fiery Flight':
        tr.ability=FieryFlight(ab.get('flight_duration_sec',5.0),0.5,
            ab.get('tornado_radius_tiles',4.0),ab.get('elixir_cost',1),ab.get('cooldown_sec',20.0))
    elif aname=='Wounding Warp':
        tr.ability=WoundingWarp(ab.get('bonus_damage_percent',50)/100.0,ab.get('elixir_cost',2))
    elif aname=='Rowdy Reroll':
        tr.ability=RowdyReroll(ab.get('second_roll_distance_tiles',4.0),
            ab.get('heal_percent',50)/100.0,0,ab.get('elixir_cost',1))
    elif aname=='Triple Threat':
        sbl=d.get('stats_by_level',{})
        st=sbl.get(str(lvl),{})
        tr.ability=TripleThreat(ab.get('dash_distance_tiles',5.0),
            st.get('decoy_hitpoints',ab.get('decoy_hp',518)),
            ab.get('triple_shot_range_tiles',15.5),ab.get('empowered_window_sec',7.0),
            ab.get('elixir_cost',1),ab.get('cooldown_sec',25.0))
