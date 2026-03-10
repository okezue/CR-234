import json,os,random,sys
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from troop import Troop
from spell import Spell,SpawnSpell
from spell import LogSpell
from components import (SplashAttack,BuildingTarget,RiverJump,
    Charge,SpawnTimer,DeathDamage,DeathSpawn,SpawnZap,DualTarget,
    RampUp,RageDrop)
_CD=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','game_data','cards')
_JC={}
def _ld(name):
    if name not in _JC:
        p=os.path.join(_CD,name+'.json')
        with open(p) as f:_JC[name]=json.load(f)
    return _JC[name]
def _parse_old(d,lvl):
    stats=None
    for s in d['levels']['stats']:
        if s['level']==lvl:stats=s;break
    if not stats:stats=d['levels']['stats'][-1]
    hp=stats.get('hp',stats.get('hitpoints',0))
    dmg=stats.get('damage',0)
    hspd=float(d['hit_speed'].rstrip('s'))
    fhspd=float(d.get('first_hit',d['hit_speed']).rstrip('s'))
    sv=int(d.get('speed','Medium (60)').split('(')[1].rstrip(')'))
    spd=sv/60.0
    rng_str=d.get('range','1.2')
    rng=float(rng_str.split('(')[1].rstrip(')')) if '(' in rng_str else float(rng_str)
    tgt_str=d.get('targets','Ground')
    if 'Air' in tgt_str and 'Ground' in tgt_str:tgts=['Air','Ground']
    elif 'Air' in tgt_str:tgts=['Air']
    elif 'Buildings' in tgt_str:tgts=['Buildings']
    else:tgts=['Ground']
    return {'hp':hp,'dmg':dmg,'hspd':hspd,'fhspd':fhspd,'spd':spd,
            'rng':rng,'targets':tgts,'transport':'Ground',
            'atk_type':'single_target','splash_r':0,'ct_dmg':0,
            'lvl':lvl,'name':d.get('name',''),'components':[]}
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
        stats=sbl.get(str(lvl))
        if not stats:
            ks=sorted(sbl.keys(),key=int)
            stats=sbl[ks[-1]] if ks else {}
        mv=d.get('movement',{});atk=d.get('attack',{})
    hp=stats.get('hitpoints',stats.get('hp',0))
    dph=stats.get('damage_per_hit')
    if isinstance(dph,dict):
        dmg=dph.get('stage_1',0)
    else:
        dmg=stats.get('damage',stats.get('area_damage',stats.get('damage_per_bolt',0)))
    ct_dmg=stats.get('crown_tower_damage',0)
    sv=mv.get('speed',{}).get('value',60)
    spd=sv/60.0
    transport=mv.get('transport','Ground')
    hspd=atk.get('hit_speed_sec',1.0)
    fhspd=atk.get('first_hit_speed_sec',hspd)
    rng_d=atk.get('range',{})
    rng=rng_d.get('tiles',1.2) if isinstance(rng_d,dict) else float(rng_d)
    tgts_raw=atk.get('targets','Ground')
    if isinstance(tgts_raw,list):tgts=list(tgts_raw)
    elif 'Air' in str(tgts_raw) and 'Ground' in str(tgts_raw):tgts=['Air','Ground']
    elif 'Buildings' in str(tgts_raw):tgts=['Buildings']
    else:tgts=[str(tgts_raw)]
    at=atk.get('damage_type','single_target')
    sr=atk.get('splash_radius_tiles',0)
    comps=[]
    if at=='area':
        if sr<=0:sr=1.2
        comps.append(SplashAttack())
    if tgts==['Buildings']:comps.append(BuildingTarget())
    if d.get('mechanics',{}).get('river_jump',{}).get('enabled'):comps.append(RiverJump())
    return {'hp':hp,'dmg':dmg,'hspd':hspd,'fhspd':fhspd,'spd':spd,
            'rng':rng,'targets':tgts,'transport':transport,
            'atk_type':at,'splash_r':sr,'ct_dmg':ct_dmg,
            'components':comps,'lvl':lvl,'name':d.get('name','')}
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
        sk='stun' if dur>0 else None
        return {'dmg':dmg,'ct_dmg':ct_dmg,'radius':radius,
                'kb':0,'dur':dur,'status_kind':sk,'name':d.get('name','')}
    sbl=d.get('stats_by_level',{})
    stats=sbl.get(str(lvl))
    if not stats:
        ks=sorted(sbl.keys(),key=int)
        stats=sbl[ks[-1]] if ks else {}
    sa=d.get('spell_attributes',{})
    dmg=stats.get('area_damage',stats.get('damage',0))
    ct_dmg=stats.get('crown_tower_damage',0)
    dur=sa.get('duration_sec',0)
    ti=sa.get('tick_interval_sec',0)
    if ti>0:
        td=stats.get('damage_per_tick',0)
        tcd=stats.get('crown_tower_damage_per_tick',0)
        sp=sa.get('slow_percent',0)/100.0
        tks=sa.get('ticks',int(dur/ti))
        return {'dmg':0,'ct_dmg':0,'radius':sa.get('radius_tiles',2.5),
                'kb':0,'dur':dur,'status_kind':None,'name':d.get('name',''),
                'tick_dmg':td,'tick_ct_dmg':tcd,'tick_interval':ti,
                'ticks_left':tks,'slow_pct':sp}
    sk='freeze' if dur>0 else None
    return {'dmg':dmg,'ct_dmg':ct_dmg,'radius':sa.get('radius_tiles',2.5),
            'kb':sa.get('knockback_tiles',0),'dur':dur,
            'status_kind':sk,'name':d.get('name','')}
def _build_sub_cfg(su_data,lvl):
    sbl=su_data.get('stats_by_level',{})
    stats=sbl.get(str(lvl))
    if not stats:
        ks=sorted(sbl.keys(),key=int)
        stats=sbl[ks[-1]] if ks else {}
    mv=su_data.get('movement',{})
    atk=su_data.get('attack',{})
    hp=stats.get('hitpoints',0)
    dmg=stats.get('damage',0)
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
    return {'hp':hp,'dmg':dmg,'hspd':hspd,'fhspd':fhspd,'spd':spd,
            'rng':rng,'targets':tgts,'transport':mv.get('transport','Ground'),
            'atk_type':'single_target','splash_r':0,'ct_dmg':0,
            'components':comps,'lvl':lvl,'name':'',
            'death_dmg':death_dmg,'death_splash_r':death_splash_r}
def _add_mechanics(d,cfg,lvl):
    mech=d.get('mechanics',{})
    sbl=d.get('stats_by_level',{})
    stats=sbl.get(str(lvl))
    if not stats:
        ks=sorted(sbl.keys(),key=int)
        stats=sbl[ks[-1]] if ks else {}
    atk=d.get('attack',{})
    ch=mech.get('charge')
    if ch:
        dist=ch.get('charge_distance_tiles',2.5)
        cfg['components'].append(Charge(dist))
        cfg['charge_dmg']=stats.get('charge_damage',cfg['dmg']*2)
    szap=d.get('spawn_zap')
    if szap:
        cfg['components'].append(SpawnZap())
        cfg['spawn_zap_dmg']=stats.get('spawn_zap_damage',0)
        cfg['spawn_zap_r']=szap.get('radius_tiles',3.0)
    if atk.get('damage_type')=='dual_zap_split':
        cfg['components'].append(DualTarget())
        cfg['stun_dur']=atk.get('stun_duration_sec',0.5)
    dd=mech.get('death_damage',{})
    if dd.get('enabled'):
        cfg['components'].append(DeathDamage())
        cfg['death_dmg']=stats.get('death_damage',0)
        cfg['death_splash_r']=dd.get('splash_radius_tiles',2.0)
    sh=mech.get('shield',{})
    if sh.get('enabled'):
        sk=sh.get('shield_hp_key','shield_hitpoints')
        sv=stats.get(sk,0)
        cfg['shield_hp']=sv;cfg['max_shield_hp']=sv
    ru=atk.get('mechanics',{}).get('ramp_up')
    if ru:
        dph=stats.get('damage_per_hit',{})
        stgs=[];durs=[]
        for i in range(1,ru.get('stages',3)+1):
            stgs.append(dph.get(f'stage_{i}',cfg['dmg']))
            dk=f'stage_{i}_duration_sec'
            if dk in ru:durs.append(ru[dk])
        cfg['ramp_stages']=stgs;cfg['ramp_durations']=durs
        cfg['components'].append(RampUp(stgs,durs))
    dm=d.get('death_mechanic',{})
    rg=dm.get('rage')
    if rg:
        cfg['components'].append(RageDrop(rg['radius_tiles'],rg['duration_sec'],rg['boost_percent']/100.0))
    ds=mech.get('death_spawn')
    if ds:
        su=d.get('sub_units',{})
        su_name=ds['unit'];su_cnt=ds['count']
        sub_cfg=_build_sub_cfg(su.get(su_name,{}),lvl)
        cfg['components'].append(DeathSpawn(sub_cfg,su_cnt))
    sp=d.get('spawns')
    if sp:
        su=d.get('sub_units',{})
        su_name=sp['unit']
        sub_cfg=_build_sub_cfg(su.get(su_name,{}),lvl)
        cfg['components'].append(SpawnTimer(sub_cfg,sp['spawn_interval_sec'],
            sp['spawn_count_per_interval'],sp.get('spawn_first_delay_sec',1.0)))
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
def create(name,lvl,team,x,y):
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
        sa=d.get('spell_attributes',{})
        if sa.get('spawns_unit'):
            return _build_spawn_spell(d,lvl,team,x,y)
        return Spell(team,x,y,_parse_spell(d,lvl))
    su=d.get('sub_units');cnt=d.get('count',1)
    if su and cnt>1:
        sk=list(su.keys())[0]
        cfg=_parse_new(d,lvl,sub=su[sk])
    elif 'levels' in d:
        cfg=_parse_old(d,lvl)
    else:
        cfg=_parse_new(d,lvl)
    if cnt<=1:
        _add_mechanics(d,cfg,lvl)
        return Troop(team,x,y,cfg)
    troops=[]
    for i in range(cnt):
        ox=random.uniform(-1.5,1.5);oy=random.uniform(-1.5,1.5)
        troops.append(Troop(team,x+ox,y+oy,cfg))
    return troops
