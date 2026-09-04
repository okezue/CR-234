import json
import math
import re
from pathlib import Path
from sim.units import Troop,Building
from sim import fx
from sim.spells import (Spell,SpawnSpell,GraveyardSpell,RageSpell,LightningSpell,CloneSpell,LogSpell,EarthquakeSpell,TornadoSpell,VoidSpell,
    VinesSpell,GoblinCurseSpell,RoyalDeliverySpell,BarbarianBarrelSpell,EvoZapSpell,EvoSnowballSpell,DecoyBarrelSpell)
from sim.knobs import K

DATA=Path(__file__).resolve().parents[1]/'data'
TGT={'air':'Air','ground':'Ground','buildings':'Buildings'}
_DB={}

# the game's speed value is in units per tick (20 ticks/s, 1000 units per tile), not tiles per minute: medium 60 walks 1.2 tiles/s,
# measured on four recordings (vid/), so tiles per second = speed / 50
SPD=50.0

def load():
    if not _DB:
        _DB.update(json.loads((DATA/'cards.json').read_text()))
        _DB['aliases']=json.loads((DATA/'aliases.json').read_text())
    return _DB
def key(name):
    db=load();k=db['aliases']['api'].get(name,name)
    return k if k in db['cards'] or k in db['towers'] else None
def card(name):
    db=load();k=key(name)
    if k is None:raise KeyError(name)
    return db['cards'].get(k) or db['towers'][k]
def snake(s):return re.sub(r'[^a-z0-9]+','_',re.sub(r'(?<=[a-z0-9])(?=[A-Z])','_',s).lower()).strip('_')
def at(v,lvl):
    # 16-arrays are indexed by absolute level; below the rarity minimum the lowest defined level is used
    if isinstance(v,list) and len(v)==16:
        return v[lvl-1] if v[lvl-1] is not None else next((x for x in v if x is not None),None)
    return v
def mult(v):
    # ClashStrategic percents: 130 and 30 both mean x1.3, -30 means x0.7, -100 stops
    return v/100 if v>100 else 1+v/100
def empty(v):return v is None or v==[] or v=={} or (isinstance(v,list) and all(x is None for x in v))
def pick(chain,*keys):
    for o in chain:
        for k in keys:
            v=o.get(k)
            if v is None and isinstance(o.get('stats'),dict):v=o['stats'].get(k)
            if not empty(v):return v
    return None
def targets(t):
    t=t or []
    if 'buildings' in t and 'ground' not in t:return ['Buildings']
    return sorted(TGT[x] for x in t if x!='buildings')
def count(v):return v.get('base',1) if isinstance(v,dict) else (v or 1)
def first(hs,lt):
    # the load time is the part of the swing a unit carries in while walking, so a loaded unit hits after hit speed less load time
    # (the wiki's First Hit Speed); a load time at or above the hit speed (Inferno Tower) leaves the whole hit speed
    return hs-K['load_carry']*lt if lt and lt<hs else hs
def merge(a,b):
    # evo skills override the base skill parameter by parameter
    return {**a,**{k:{**a.get(k,{}),**{p:v for p,v in b[k].items() if v is not None}} for k in b}}

def base(chain,lvl,name,parent=None):
    # chain: skill dict, unit record, referenced card, parent card; first non-null wins, kamikaze never inherits from the parent
    p=lambda *k:pick(chain,*k)
    hs=p('hitSpeed') or 1.0;pj=p('projectile') or {}
    splash=p('hitType')=='splash' and (p('radius') or 0)>0
    # an explicit [] on the spawn or unit record (Decoy, Snowman, Phoenix Egg) is a passive unit; only a missing entry inherits the parent's targets
    own=[o.get('targets') for o in chain[:-1] if o.get('targets') is not None]
    tg=targets(next((t for t in own if t),None) if any(own) else ([] if own else p('targets')))
    # spawner buildings carry a 10 s dummy hit speed and no range in the game data: they do not attack
    rng=0 if p('kind')=='building' and (hs>=10 or not tg) else p('range') or 0 if p('kind')=='building' else max(p('range') or 0,0.5)
    # a spread of small projectiles (Hunter, Firecracker) lists the damage per pellet
    shots=pj.get('count') or 1 if p('kind')!='spell' and (p('radius') or 0)<=0.5 else 1
    return {'hp':at(p('hitpoints'),lvl) or 0,'dmg':(at(p('damage'),lvl) or 0)*shots,'hspd':hs,'fhspd':first(hs,p('loadTime')),
            'spd':(p('speed') or 0)/SPD,'rng':rng,'min_rng':p('minRange') or 0,'targets':tg,
            'transport':'Air' if p('flying') else 'Ground','atk_type':'area' if splash else 'single_target','splash_r':p('radius') if splash else 0,
            'ct_dmg':(at(p('towerDamage'),lvl) or 0)*shots,'components':[],'lvl':lvl,'name':name,'mass':p('mass') or 4,'sight_r':p('sightRange') or 5.5,
            'collision_r':p('collisionRadius') or 0.5,'projSpeed':pj.get('speed') or 0,'deploy':p('deployTime') or 0,
            'is_suicide':bool(pick([o for o in chain if o is not parent],'kamikaze'))}

def unit(c,sk,lvl):
    # a spawned character: its own skill params, then the units record, then the card it is a copy of, then the parent
    name=sk.get('character');k=snake(name);u=c['units'].get(k,{})
    ref=load()['cards'].get(u.get('card')) if u.get('card') else None
    chain=[sk,u]+([ref] if ref else [])+[c]
    cfg=base(chain,lvl,u.get('name') or name,parent=c)
    skills={**(ref['skills'] if ref else {}),**u.get('skills',{})}
    if 'spawnOnDeath' in sk and not empty(sk.get('damage')):skills={**skills,'areaDamageOnDeath':{'damage':sk['damage'],'radius':sk.get('radius')}}
    elif not empty(u.get('deathDamage')):skills={**skills,'areaDamageOnDeath':{**skills.get('areaDamageOnDeath',{}),'damage':u['deathDamage']}}
    attach(cfg,c,skills,lvl,chain)
    return cfg

def attach(cfg,c,sk,lvl,chain=None):
    # the generic skills -> components mapping; every number comes from the record at this level
    cs=cfg['components'];d=cfg['dmg'];pb=sk.get('pushback',{});kb=pb.get('distance') or 0
    if cfg['atk_type']=='area':cs.append(fx.SplashAttack())
    if cfg['targets']==['Buildings']:cs.append(fx.BuildingTarget())
    if 'jump' in sk and 'dash' not in sk:cs.append(fx.RiverJump())
    ch=sk.get('charge',{})
    # a charge on a champion whose ability spawns a character is the spawn's (Little Prince's Guardienne)
    if ch.get('range') and not sk.get('ability',{}).get('skills',{}).get('spawn'):
        cs.append(fx.Charge(ch['range']));cfg['charge_dmg']=at(ch.get('damage'),lvl) or d
    da=sk.get('dash',{})
    if da.get('minRange') is not None and 'jump' in sk:
        cs.append(fx.MKJump(da['minRange'],da['maxRange'],da.get('radius') or 0,da.get('speed') or 1,sk['jump'].get('duration') or 0,kb))
        cfg['jump_dmg']=at(da.get('damage'),lvl) or d
    elif da.get('minRange') is not None:
        cs.append(fx.BanditDash(da['minRange'],da['maxRange'],da.get('chargeTime') or 0,da.get('speed') or 8.333));cfg['dash_dmg']=at(da.get('damage'),lvl) or d
    sd=sk.get('spawnOnDeath',{});ps=sk.get('periodicSpawn',{})
    egg=ps.get('character') and ps.get('pauseTime') and snake(ps['character']).startswith(snake(c['name']))
    if egg:
        # a respawn of the card itself: the death spawn is the egg, which hatches into the respawn after pauseTime (its lifetime, counted from
        # the moment it appears, so the egg deploys instantly)
        reborn=unit(c,ps,lvl)
        if sd.get('character'):
            e=unit(c,sd,lvl);e['deploy']=0;e['components'].append(fx.Hatch(reborn,ps['pauseTime']));cs.append(fx.DeathSpawn(e,1))
        else:cs.append(fx.DeathSpawn(reborn,count(ps.get('count'))))
    elif sd.get('character') and count(sd.get('count')):cs.append(fx.DeathSpawn(unit(c,sd,lvl),count(sd.get('count'))))
    dd=sk.get('areaDamageOnDeath',{})
    if not empty(dd.get('damage')):
        cfg['death_dmg']=at(dd['damage'],lvl);cfg['death_splash_r']=dd.get('radius') or 0
        sl=sk.get('slow',{})
        if 'boost' in sk:
            b=sk['boost'];cs.append(fx.RageDrop(b.get('radius') or dd.get('radius') or 0,b.get('duration') or 0,mult(b.get('speedMultiplier') or 0)-1))
        elif sl.get('duration') and cfg['atk_type']!='area':cs.append(fx.DeathNova(-(sl.get('speedMultiplier') or 0),sl['duration']))
        else:cs.append(fx.DeathDamage(kb,dd.get('fuse') or 0))
    elif sk.get('slow',{}).get('duration') and 'secondaryAttack' not in sk:
        sl=sk['slow'];cfg['slow_dur']=sl['duration'];cfg['slow_val']=mult(sl.get('speedMultiplier') or 0)
    if ps.get('character') and ps.get('pauseTime') and not ps.get('hpPercent') and not egg:
        fd=ps.get('firstDelay');cs.append(fx.SpawnTimer(unit(c,ps,lvl),ps['pauseTime'],count(ps.get('count')),ps['pauseTime'] if fd is None else fd,
                                                            ps.get('spawnInterval') or 0,ps.get('range') or 0))
    az=sk.get('areaDamageOnSpawn',{})
    if not empty(az.get('damage')):
        cs.append(fx.SpawnZap(kb if 'dash' in sk else 0));cfg['spawn_zap_dmg']=at(az['damage'],lvl);cfg['spawn_zap_r']=az.get('radius') or 0
        cfg['spawn_zap_ct']=at(az.get('towerDamage'),lvl)
    bu=sk.get('burrow')
    if bu is not None:cs.append(fx.Burrow((bu.get('speed') or 0)/SPD,c['deployTime'] or 1.0))
    tf=sk.get('transform',{})
    if tf.get('hpPercent') and tf.get('building'):cs.append(fx.Breakdown(tf['hpPercent']/100,tf.get('lifetime') or 0))
    elif tf.get('hpPercent'):cs.append(fx.RocketRide(tf['hpPercent']/100,(tf.get('speed') or 0)/SPD,tf.get('range') or 0.5,tf.get('lifetime') or 0))
    sok=sk.get('spawnOnKill',{})
    if sok.get('character'):cs.append(fx.CurseOnHit(unit(c,sok,lvl),sok.get('markDuration') or 0))
    st=sk.get('stun',{});pi=sk.get('pierce',{})
    if pi.get('range') and not pi.get('bounces') and cfg['rng']>0:
        # a line shot: the spread cards list damage per pellet, the target takes one pellet and towers the whole volley
        n=(c['projectile'] or {}).get('count') or 1
        if n>1:cfg['dmg']//=n
        cfg['atk_type']='single_target';cfg['splash_r']=0;cfg['components']=[x for x in cs if not isinstance(x,fx.SplashAttack)];cs=cfg['components']
        ret=pi.get('returnTime') or 0
        cs.append(fx.LineAttack(pi['range'],pi.get('radius') or cfg['collision_r'],kb,2 if ret else 1,ret,n if n>1 else 0))
    if st.get('duration') and 'reflect' not in sk:
        if pi.get('bounces'):
            cs.append(fx.SuicideChain() if cfg['is_suicide'] else fx.ChainAttack())
            cfg['chain_count']=pi['bounces']+1;cfg['chain_range']=pi.get('bounceDistance') or 0;cfg['chain_stun']=st['duration']
        elif st.get('targets')==2:
            cfg['atk_type']='single_target';cfg['splash_r']=0;cfg['components']=[x for x in cs if not isinstance(x,fx.SplashAttack)];cs=cfg['components']
            cs.append(fx.DualTarget());cfg['stun_dur']=st['duration']
        else:cfg['stun_dur']=st['duration']
    rf=sk.get('reflect',{})
    if not empty(rf.get('damage')):cs.append(fx.ZapPack(at(rf['damage'],lvl),rf.get('radius') or 0,st.get('duration') or 0))
    elif rf.get('damageMultiplier'):cs.append(fx.Parry(mult(rf['damageMultiplier']),rf.get('cooldown') or 0))
    rd=sk.get('rampingDamage',{})
    if rd.get('damageTiers'):
        tiers=[at(t,lvl) for t in rd['damageTiers']];cfg['dmg']=tiers[0];cfg['ramp_stages']=tiers;cfg['ramp_durations']=[rd['rampInterval']]*(len(tiers)-1)
        cs.append(fx.RampUp(tiers,cfg['ramp_durations']))
    sh=sk.get('shield',{})
    if not empty(sh.get('hitpoints')):cfg['shield_hp']=cfg['max_shield_hp']=at(sh['hitpoints'],lvl)
    hl=sk.get('heal',{})
    if not empty(hl.get('perAttack')):
        h=at(hl['perAttack'],lvl);r=hl.get('radius') or 0
        cs.append(fx.HealBurst(h,r) if cfg['is_suicide'] else fx.HealPulse(h,r,1))
    sa=sk.get('secondaryAttack',{})
    if sa.get('character'):
        u=c['units'].get(snake(sa['character']),{});p=lambda *k:pick([sa,u],*k)
        if chain and 'buildings' in (pick(chain,'targets') or []) and cfg['targets']!=['Buildings']:
            # the card lists the union of both attackers; the mount itself only hits buildings
            cfg['targets']=['Buildings'];cs.append(fx.BuildingTarget())
        if p('minRange') is not None:
            cs.append(fx.RocketLauncher(at(p('damage'),lvl),p('hitSpeed'),first(p('hitSpeed'),p('loadTime')),p('minRange'),p('range'),p('radius') or 0))
        else:
            sl=sk.get('slow',{})
            cs.append(fx.RiderAttack(at(p('damage'),lvl),p('hitSpeed'),p('range'),1-mult(sl.get('speedMultiplier') or 0) if sl.get('duration') else 0,
                sl.get('duration') or 0,first(p('hitSpeed'),p('loadTime')),count(p('count'))))
    pe=sk.get('produceElixir',{})
    if pe.get('interval'):cs.append(fx.ElixirProd(pe['interval'],pe.get('amount') or 1))
    if kb and 'areaDamageOnDeath' not in sk and 'dash' not in sk and 'pierce' not in sk and cfg['atk_type']!='area':
        cs.append(fx.MonkCombo(pb['cycle'],kb) if pb.get('cycle') else fx.Knockback(kb))
    if sk.get('recoil',{}).get('distance'):cs.append(fx.Recoil(sk['recoil']['distance']))
    ms=sk.get('meleeSwitch',{})
    if not empty(ms.get('damage')):cs.append(fx.MeleeSwitch(at(ms['damage'],lvl),ms.get('range') or 1.6))
    rh=sk.get('rampingHitSpeed',{})
    if rh.get('hitSpeedTiers'):cs.append(fx.LPRamp(rh['hitSpeedTiers'],rh.get('hitsPerTier') or 1))
    iv=sk.get('invisibility',{})
    # a null idle time means invisible from deployment until the first attack (Suspicious Bush)
    if 'whenNotAttackingTime' in iv:cs.append(fx.Stealth(iv['whenNotAttackingTime'] or 0))
    if sk.get('multiply',{}).get('maxUnits'):cs.append(fx.EvoSkeletons(sk['multiply']['maxUnits']))
    return cfg

def uses(ab,a):
    # cooldown null encodes a single-use ability (all but Boss Bandit since the August 4 update)
    if ab is not None and a['cooldown'] is None:ab.uses=a.get('uses') or 1
    return ab

def ability(c,a,lvl,tr):
    # champion abilities keyed by their game-data name; the parameters are the ability's skills at this level
    s=a['skills'];cost=a['cost'];cd=a['cooldown'] or 0;n=a['name']
    da=s.get('dash',{});iv=s.get('invisibility',{});b=s.get('boost',{});po=s.get('poison',{});rf=s.get('reflect',{});sp=s.get('spawn',{})
    if n=='Giantbuffer_ability':
        # passive: the enchantment is a component, not a cast
        if b.get('radius'):
            tr.components.append(fx.Enchant(b['radius'],b.get('count') or 1,b.get('hitsPerBonus') or 1,at(b.get('damage'),lvl) or 0,b.get('duration') or 0))
        return None
    if n=='GoldenKnightChain':return fx.DashingDash(at(da['damage'],lvl),da['count'],da['maxRange'],cost,cd)
    if n=='SkeletonKing':
        tr.components.append(fx.SoulCollect(sp['count']['maxStacks']))
        return fx.SoulSummoning(unit(c,sp,lvl),sp['radius'],cost,cd,sp['count']['base'],sp['interval'])
    if n=='BossBandit_ability':return fx.GetawayGrenade(s['teleport']['distance'],iv['duration'],cost,cd,a.get('uses') or 1)
    if n=='ArcherQueenRapid':
        return fx.CloakingCape(iv.get('duration') or b['duration'],tr.spd*mult(b.get('speedMultiplier') or 0),mult(b['hitSpeedMultiplier'])-1,cost,cd)
    if n=='MightyMinerLaneSwitch':
        # the drill's ramp is filed under the ability in the game data but is the base attack
        rd=s['rampingDamage'];tiers=[at(t,lvl) for t in rd['damageTiers']]
        tr.dmg=tiers[0];tr.ramp_stages=tiers;tr.ramp_durations=[rd['rampInterval']]*(len(tiers)-1)
        tr.components.append(fx.RampUp(tiers,tr.ramp_durations))
        dd=s['areaDamageOnDeath'];return fx.ExplosiveEscape(at(dd['damage'],lvl),dd['radius'],s.get('pushback',{}).get('distance') or 0,cost,cd)
    if n=='Goblinstein_ability':return fx.LightningLink(at(po['damage'],lvl),at(po['towerDamage'],lvl),po['radius'],po['duration'],po['tickInterval'],cost,cd)
    if n=='ChampGuardianAbility':
        # the card's charge is the Guardienne's Royal Rescue dash: damage and range on the card, pushback on the ability
        ch=c['skills']['charge'];kb=s.get('pushback',{}).get('strength') or 0
        return fx.RoyalRescue(unit(c,sp,lvl),at(ch['damage'],lvl),kb,ch.get('range') or 0,cost,cd)
    if n=='Deflect':return fx.PensiveProtection(rf['damageReductionPercent']/100,rf['duration'],cost,cd)
    return None

def hero(c,a,lvl,tr):
    s=a['skills'];cost=a['cost'];cd=a['cooldown'] or 0;n=a['name']
    sp=s.get('spawn',{});st=s.get('stun',{});sl=s.get('slow',{});b=s.get('boost',{});pb=s.get('pushback',{});da=s.get('dash',{});sn=s.get('snipe',{})
    iv=s.get('invincible',{});az=s.get('areaDamageOnSpawn',{});sh=s.get('shield',{})
    if n=='Knight_hero':return fx.TriumphantTaunt(s['taunt']['range'],at(s['shield']['hitpoints'],lvl),s['taunt']['duration'],cost,cd)
    if n=='Balloon_hero':return fx.CoffinCadets(unit(c,sp,lvl),at(da.get('damage'),lvl) or 0,at(da.get('towerDamage'),lvl) or 0,da.get('maxRange') or 0,cost,cd)
    if n=='Berserker_hero':
        hs=b.get('hitSpeed') or tr.hspd/mult(b.get('hitSpeedMultiplier') or 0);sp_=mult(b.get('speedMultiplier') or 0)
        return fx.SavageSurvival(b['duration'],at(b.get('damage'),lvl),at(b.get('towerDamage'),lvl),hs,sp_,iv.get('minHitpoints') or 0,cost,cd)
    if n=='Bowler_hero':
        ctm=(sn.get('towerDamagePercent') or 100)/100;cast=a.get('castTime') or fx.Ability.CAST_TIME;hs=mult(sn.get('hitSpeedMultiplier') or 0)
        return fx.StoneSwish(sn['range'],hs,mult(sn.get('damageMultiplier') or 0),ctm,sn.get('aoeRadius') or 0,sn['rootDuration'],cast,cost,cd)
    if n=='DarkPrince_hero':
        r=unit(c,sp,lvl);r['charge_dmg']=at(s['charge'].get('damage'),lvl) or r['dmg'];r['components'].append(fx.Charge(s['charge']['range']))
        return fx.DestructiveDismount(r,at(az.get('damage'),lvl) or 0,az.get('radius') or 0,cost,cd)
    if n=='IceGolemite_hero':
        val=mult(sl.get('speedMultiplier') or 0)
        return fx.Snowstorm(sl.get('strikes') or 1,sl.get('radius') or 0,at(sl.get('damage'),lvl) or 0,val,sl['duration'],cost,cd)
    if n=='Frosty Fella':
        t=unit(c,sp,lvl);t['is_building']=True;t['lifetime']=sp.get('lifetime') or 0
        return fx.FrostyFella(t,at(sp.get('damage'),lvl) or 0,at(sp.get('towerDamage'),lvl) or 0,sp.get('radius') or st.get('radius') or 0,cost,cd)
    if n=='Tombstone_hero' and sp.get('character'):return fx.RegalRevive(unit(c,sp,lvl),cost,cd)
    if n=='Valkyrie_hero':
        ctm=(b.get('towerDamagePercent') or 100)/100;red=(sh.get('damageReductionPercent') or 0)/100
        return fx.WildWhirlwind(da.get('maxRange') or 0,b['duration'],b.get('hitSpeed') or 0,at(b.get('damage'),lvl) or 0,ctm,b.get('radius') or tr.splash_r,
                                mult(b.get('speedMultiplier') or 0),red,cost,cd)
    if n=='Goblins_hero':
        ps=s['periodicSpawn'];bb=fx.BannerBrigade(count(ps.get('count')),ps['lifetime'],cost);bb.set_base_cfg(unit(c,ps,lvl));return bb
    if n=='Giant_hero':return fx.HeroicHurl(pb['distance'],st['duration'],at(pb['damage'],lvl),cost,cd)
    if n=='MiniPekka_hero':
        k=s['stack'];return fx.BreakfastBoost(k['healPercent']/100,k['hpPerStack']/100,k['damagePerStack']/100,k['maxStacks'],k['interval'],cost)
    if n=='Musketeer_hero':
        t=unit(c,sp,lvl);t['is_building']=True;t['lifetime']=sp['lifetime'];return fx.TrustyTurret(t,cost,cd)
    if n=='Wizard_hero':return fx.FieryFlight(b['duration'],mult(b['speedMultiplier'])-1,s['pull']['radius'],bool(b.get('flying')),cost,cd)
    if n=='MegaMinion_hero':return fx.WoundingWarp(s['warp']['bonusDamagePercent']/100,cost)
    if n=='EliteArcher_hero':return fx.TripleThreat(s['teleport']['distance'],at(sp['hitpoints'],lvl),s['pierce']['range'],sp['lifetime'],cost,cd)
    if n=='BarbLog_hero':
        r=s['redeploy'];return fx.RowdyReroll(r['range'],r['healPercent']/100,at(r['damage'],lvl),cost)
    if not (sp or st or sl or b):return None
    spawns=[]
    if sp.get('character'):
        t=unit(c,sp,lvl);t['lifetime']=sp.get('lifetime') or 0
        if not t['targets']:t['is_building']=True
        spawns.append((t,count(sp.get('count'))))
    radius=st.get('radius') or sl.get('radius') or b.get('radius') or c['skills'].get('areaDamageOnDeath',{}).get('radius') or c.get('radius') or 0
    slow=(sl['duration'],mult(sl.get('speedMultiplier') or 0)) if sl.get('duration') else None
    boost=(b['duration'],mult(b.get('speedMultiplier') or b.get('hitSpeedMultiplier') or 0)-1) if b.get('duration') else None
    return fx.SkillAbility(spawns,st.get('duration') or 0,slow,boost,radius,cost,cd)

def evolve(c,k,s,lvl,tr):
    # evolution mechanics keyed by card; s is the base skills overlaid with evo.skills at this level
    sn=s.get('snipe',{});po=s.get('poison',{});pu=s.get('pull',{});hl=s.get('heal',{});ps=s.get('periodicSpawn',{});st=s.get('stun',{})
    pi=s.get('pierce',{});sd=s.get('spawnOnDeath',{});b=s.get('boost',{});sl=s.get('slow',{});pb=s.get('pushback',{});sp=s.get('spawn',{})
    iv=s.get('invincible',{});vo=s.get('volley',{});bu=s.get('burrow',{});rd=s.get('rampingDamage',{})
    E={'knight':lambda:fx.EvoKnight(s['shield']['damageReductionPercent']/100),
       'battle_ram':lambda:fx.EvoBattleRam(pb.get('strength') or 0,at(pb.get('damage'),lvl) or 0,mult(b.get('speedMultiplier') or 0)-1,b.get('duration') or 0),
       'cannon':lambda:fx.EvoCannon(vo['projectileCount'],vo.get('radius') or 0,at(vo['damage'],lvl),at(vo.get('towerDamage'),lvl) or 0,
                                    vo.get('knockback') or 0),
       'elite_barbarians':lambda:fx.EvoEliteBarbarians(at(sn['damage'],lvl),sn.get('minRange') or 0,sn['range'],sn.get('cooldown') or 0,b.get('radius') or 0,
                                                       b['duration'],mult(b.get('speedMultiplier') or 0)-1),
       'furnace':lambda:fx.EvoFurnace(mult(b['spawnSpeedMultiplier'])),
       'minion_horde':lambda:fx.EvoMinionHorde(iv['duration'],mult(iv.get('moveSpeedPenalty') or 0)),
       'princess':lambda:fx.EvoPrincess(sl.get('everyHits') or 1,sl.get('radius') or s['areaDamageOnDeath'].get('radius') or 0,mult(sl['speedMultiplier']),
                                        sl['duration']),
       'skeleton_army':lambda:fx.EvoSkelArmy(iv.get('radius') or 0,tr.spd*mult(iv.get('moveSpeedPenalty') or 0)),
       'tesla':lambda:fx.EvoTesla(at(st.get('damage'),lvl) or 0,st.get('radius') or 0,st['duration']),
       'bomber':lambda:fx.EvoBomber(pi['bounces'],pi['bounceDistance']),
       'barbarians':lambda:fx.EvoBarbarians(mult(b['hitSpeedMultiplier'])-1,mult(b['speedMultiplier'])-1,b['duration']),
       'bats':lambda:fx.EvoBats(at(hl['perAttack'],lvl),at(hl['overHeal'],lvl)),
       'royal_recruits':lambda:fx.EvoRoyalRecruits(at(s['charge']['damage'],lvl),s['charge']['range']),
       'royal_giant':lambda:fx.EvoRoyalGiant(pb['radius'],pb['distance'],at(pb['damage'],lvl)),
       'ice_spirit':lambda:fx.EvoIceSpirit(po['tickInterval'],po['radius'],po['tickStunDuration'],at(po['damage'],lvl)),
       'skeleton_barrel':lambda:fx.EvoSkelBarrel(sd['hpPercent']/100),
       'firecracker':lambda:fx.EvoFirecracker(sl['count'],-sl['speedMultiplier'],sl['duration']),
       'archers':lambda:fx.EvoArchers(sn['range'],sn['maxRange'],mult(sn['damageMultiplier'])),
       'valkyrie':lambda:fx.EvoValkyrie(pu['radius'],at(pu['damage'],lvl),pu['duration']),
       'musketeer':lambda:fx.EvoMusketeer(sn['ammo'],sn['range'],mult(sn['damageMultiplier']),sn['minRange']),
       'dart_goblin':lambda:fx.EvoDartGoblin(po['radius'],po['duration'],[at(po['damage'],lvl)*(i+1) for i in range(po['maxStacks'])],po['stackHits']),
       'royal_hogs':lambda:fx.EvoRoyalHogs(at(s['jump']['damage'],lvl),s['jump']['radius']),
       'goblin_cage':lambda:fx.EvoGoblinCage(pu['radius']),
       'baby_dragon':lambda:fx.EvoBabyDragon(b['radius'],mult(b['speedMultiplier'])-1,1-mult(sl['speedMultiplier'])),
       'witch':lambda:fx.EvoWitch(at(hl['onSpawn'],lvl),at(hl['overHeal'],lvl)),
       'pekka':lambda:fx.EvoPekka(*[at(t,lvl) for t in hl['perKillTiers']],at(hl['overHeal'],lvl)),
       'goblin_giant':lambda:fx.EvoGoblinGiant(ps['hpPercent']/100,ps['pauseTime'],unit(c,ps,lvl)),
       'hunter':lambda:fx.EvoHunter(st['duration'],st['delayBetweenStrikes']),
       'electro_dragon':lambda:fx.EvoElectroDragon(pi['bounceDamagePercent']/100,pi['bounceDistance'],pi['bounceDelay']/mult(pi['speedMultiplier'])),
       'wall_breakers':lambda:fx.EvoWallBreakers(unit(c,sd,lvl),count(sd.get('count'))),
       'executioner':lambda:fx.EvoExecutioner(sn['range'],mult(sn['damageMultiplier']),sn['pushbackDistance']),
       'goblin_drill':lambda:fx.Resurface([p/100 for p in bu['resurfacePercent']],bu.get('resurfaceCount') or [count(sd.get('count'))],unit(c,sd,lvl)),
       'mega_knight':lambda:fx.EvoMegaKnight(pb['strength']),
       'inferno_dragon':lambda:fx.EvoInfernoDragon(at(rd['damageTiers'][-1],lvl),rd.get('retainTime') or 0,rd.get('finalStageTime') or 0),
       'royal_ghost':lambda:fx.EvoRoyalGhost(count(sp.get('count')),unit(c,sp,lvl)),
       'lumberjack':lambda:fx.EvoLumberjack(s['invisibility']['duration'])}
    if k in E:tr.components.append(E[k]())
    # the evolved bolt stuns and deals full damage on the base card's three targets; the bounces after them are the evolution
    if k=='electro_dragon':tr.chain_count=c['skills']['pierce']['bounces']+1
    if k=='skeleton_barrel':tr.death_dmg=at(sd['damage'],lvl)
    if k=='royal_hogs':tr.transport='Air'
    if k=='battle_ram':tr.is_suicide=False
    tr.evolved=True

def troop(c,k,lvl,team,x,y,evolved,is_hero,ev,chain,sk,name):
    cfg=attach(base(chain,lvl,name),c,sk,lvl,chain)
    if c['kind']=='building':
        cfg['lifetime']=c['lifetime'] or 0
        tr=Building(team,x,y,cfg)
    else:
        tr=Troop(team,x,y,cfg)
        if 'ability' in sk:tr.ability=uses(ability(c,sk['ability'],lvl,tr),sk['ability'])
        # LoadFirstHit (Sparky): she comes in unloaded and must charge the whole hit speed, and a stun empties the charge
        if sk.get('charging',{}).get('loadFirstHit'):tr.load_first=True;tr.cd=tr.hspd
        if sk.get('immunity',{}).get('knockback'):tr.kb_immune=True
    if ev:evolve(c,k,sk,lvl,tr)
    if is_hero and c['hero']:tr.is_hero=True;tr.ability=uses(hero(c,c['hero']['ability'],lvl,tr),c['hero']['ability'])
    for x in tr.components:
        if isinstance(x,fx.Burrow):x.start(tr)
    return tr

def formation(n,r,x,y,team):
    # summons stand evenly on a circle of the summon radius, the first at the front; a pair stands side by side, a single unit on the point
    if n<=1:return [(x,y)]*n
    if n==2:return [(x-r,y),(x+r,y)]
    f=1 if team=='blue' else -1
    return [(x+r*math.sin(2*math.pi*i/n),y+f*r*math.cos(2*math.pi*i/n)) for i in range(n)]

def create(name,lvl,team,x,y,evolved=False,hero=False):
    c=card(name);k=key(name)
    if c['kind']=='spell':return spell(c,lvl,team,x,y,evolved,hero)
    ev=c['evo'] if evolved and c['evo'] else None
    sk=merge(c['skills'],ev['skills']) if ev else c['skills']
    chain=[ev['stats'] if ev else {},c]
    n=(ev or {}).get('count') or c['count'] or 1;out=[]
    grp=sk.get('group',{})
    r=(ev or {}).get('summonRadius') or c['summonRadius'];delay=c['summonDeployDelay'] or 0;ln=sk.get('line',{}).get('spacing')
    pts=[(x+(i-(n-1)/2)*ln,y) for i in range(n)] if ln else formation(n,c['collisionRadius'] or 0.5 if r is None else r,x,y,team)
    extra=sum(c['units'][snake(ch)]['count'] or 1 for ch in grp.get('characters',[]))
    # a lone leader (Rascal Boy, Goblinstein) stands on the point with the companions side by side the summon radius behind
    if n-extra==1:pts=[(x,y)]+[(px,y-(1 if team=='blue' else -1)*(r or 0)) for px,_ in formation(extra,c['collisionRadius'] or 0.5,x,y,team)]
    for ch in grp.get('characters',[]):
        u=c['units'][snake(ch)];n-=u['count'] or 1
        ref=load()['cards'].get(u.get('card')) if u.get('card') else None
        for _ in range(u['count'] or 1):
            px,py=pts.pop()
            out.append(troop(c,k,lvl,team,px,py,False,False,None,[u]+([ref] if ref else [])+[c],merge(ref['skills'] if ref else {},u.get('skills',{})),
                             u.get('name') or ch))
    name=c['name']
    if grp.get('leader'):
        u=c['units'][snake(grp['leader'])];chain=[u]+chain;sk=merge(sk,u.get('skills',{}));name=u.get('name') or grp['leader']
    if n<=1 and not out:return troop(c,k,lvl,team,x,y,evolved,hero,ev,chain,sk,name)
    out=[troop(c,k,lvl,team,px,py,evolved,hero,ev,chain,sk,name) for px,py in pts[:max(n,0)]]+out
    for i,t in enumerate(out):t.deploy_at=i*delay
    if hero and c['hero'] and len(out)>1:
        ab=next((t.ability for t in out if getattr(t,'ability',None)),None)
        for t in out:t.ability=ab;t.is_hero=True
    return out

def spell(c,lvl,team,x,y,evolved,is_hero=False):
    ev=c['evo'] if evolved and c['evo'] else None
    sk=merge(c['skills'],ev['skills']) if ev else c['skills'];st=ev['stats'] if ev else c['stats']
    dmg=at(st['damage'],lvl) or 0;ct=at(st['towerDamage'],lvl) or 0;r=c['radius'] or 0;dur=c['duration'] or 0;hs=c['hitSpeed'] or 0
    pj=c['projectile'] or {};name=c['name']
    cfg={'dmg':dmg,'ct_dmg':ct,'radius':r,'name':name,'projSpeed':pj.get('speed') or 0,'dur':dur}
    stn=sk.get('stun',{});sl=sk.get('slow',{});pb=sk.get('pushback',{});pu=sk.get('pull',{});sp=sk.get('spawn',{});pi=sk.get('pierce',{})
    # ticking spells: damage is per hit, tick.count hits over the duration (hitSpeed is the interval when given)
    ticks=(c.get('tick') or {}).get('count') or (round(dur/hs) if hs and dur else 0)
    hs=dur/ticks if ticks and c.get('tick') else hs
    roll={**cfg,'range':pi.get('range'),'width':r*2,'pushback':pb.get('distance') or 0,'speed':pi.get('speed') or 0}
    if sp.get('character'):
        # the evo spawn (decoy goblins) rides along with the base spawn, so the real troops come from the base skill
        bs=c['skills']['spawn'] if ev and 'spawn' in ev['skills'] else sp
        tc=unit(c,bs,lvl);n=count(bs.get('count')) if bs.get('count') else (c['count'] or 1)
        if not empty(sk.get('shield',{}).get('hitpoints')):tc['shield_hp']=tc['max_shield_hp']=at(sk['shield']['hitpoints'],lvl)
        # a hero spell's ability belongs to the troop it spawns (Barbarian Barrel's Rowdy Reroll)
        if is_hero and c['hero']:tc['hero']=lambda tr:uses(hero(c,c['hero']['ability'],lvl,tr),c['hero']['ability'])
        if pi.get('range'):return BarbarianBarrelSpell(team,x,y,{**roll,'troop_cfg':tc})
        if sp.get('interval'):
            gy={'troop_cfg':tc,'total':n,'interval':sp['interval'],'radius':r,'min_radius':sp.get('minRadius') or r,'dur':dur,'name':name,
                'first_delay':sp.get('firstDelay') or 0}
            return GraveyardSpell(team,x,y,gy)
        if c['hitType']=='splash':return RoyalDeliverySpell(team,x,y,{**cfg,'troop_cfg':tc})
        if bs is not sp:
            dc=unit(c,ev['skills']['spawn'],lvl);dc['name']='Decoy '+dc['name']
            return DecoyBarrelSpell(team,x,y,{'troop_cfg':tc,'count':n,'name':name,'projSpeed':cfg['projSpeed'],'decoy_cfg':dc,
                                              'decoy_count':count(sp.get('count'))})
        return SpawnSpell(team,x,y,{'troop_cfg':tc,'count':n,'name':name,'projSpeed':cfg['projSpeed']})
    if 'multiply' in sk:return CloneSpell(team,x,y,{'radius':r,'name':name})
    if pi.get('range'):return LogSpell(team,x,y,roll)
    if 'multiTarget' in sk:
        m=sk['multiTarget'];iv=c['hitSpeed'];strikes=len([t for t in range(ticks) if m['firstDelay']+t*iv<dur])
        return VoidSpell(team,x,y,{'radius':r,'tiers':[at(t,lvl) for t in m['damageTiers']],'tower_tiers':[at(t,lvl) for t in m['towerDamageTiers']],
                                   'max_units':m['maxUnits'],'strikes':strikes,'interval':iv,'first_delay':m['firstDelay'],'name':name})
    if 'spawnOnKill' in sk:
        return GoblinCurseSpell(team,x,y,{'radius':r,'tick_dmg':dmg,'ct_dmg':ct,'ticks':ticks,'interval':hs,'name':name,
                                          'goblin_cfg':unit(c,sk['spawnOnKill'],lvl)})
    if stn.get('targets'):
        if ticks:return VinesSpell(team,x,y,{'radius':r,'max_targets':stn['targets'],'dur':dur,'tick_dmg':dmg,'tick_interval':hs,'ticks':ticks,'name':name})
        return LightningSpell(team,x,y,{**cfg,'max_targets':stn['targets'],'stun_dur':stn['duration']})
    if 'boost' in sk:
        b=sk['boost'];return RageSpell(team,x,y,{**cfg,'rage_boost':mult(b['speedMultiplier'])-1,'rage_dur':b['duration']})
    tick={'radius':r,'ticks':ticks,'interval':hs,'name':name,'ct_dmg':ct}
    if pu.get('strength'):return TornadoSpell(team,x,y,{**tick,'tick_dmg':dmg,'pull_str':pu['strength']/100,'dur':dur})
    if not empty(st.get('buildingDamage')):
        return EarthquakeSpell(team,x,y,{**tick,'troop_dmg':dmg,'bldg_dmg':at(st['buildingDamage'],lvl),'slow_pct':1-mult(sl.get('speedMultiplier') or 0)})
    if ticks and c['hitSpeed']:
        return Spell(team,x,y,{**cfg,'dmg':0,'ct_dmg':0,'tick_dmg':dmg,'tick_ct_dmg':ct,'tick_interval':hs,'ticks_left':ticks,
                               'slow_pct':1-mult(sl['speedMultiplier']) if sl.get('speedMultiplier') else 0,'status_kind':None})
    waves=pj.get('waves') or 1
    if stn.get('duration'):kind,val,sdur='freeze' if stn['duration']>1 else 'stun',1.0,stn['duration']
    elif sl.get('duration'):kind,val,sdur='slow',mult(sl['speedMultiplier']),sl['duration']
    else:kind,val,sdur=None,1.0,0
    cfg.update({'dmg':dmg//waves,'ct_dmg':ct//waves,'kb':pb.get('distance') or 0,'dur':sdur,'status_kind':kind,'status_val':val,
                'volleys':waves,'volley_interval':pj.get('waveInterval') or 0})
    if ev and stn.get('strikes'):return EvoZapSpell(team,x,y,{**cfg,'pulse_2_radius':r+(stn.get('radiusGrowth') or 0),'pulse_2_delay':stn['duration']})
    if ev and pu.get('distance'):return EvoSnowballSpell(team,x,y,{**cfg,'roll_distance':pu['distance'],'roll_duration':pu['duration'],'slow_duration':sdur})
    return Spell(team,x,y,cfg)
