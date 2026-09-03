import math
import random
from sim.units import Status,Troop
from sim.fx import strip,hurt,push,tdist
class Spell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.radius=cfg['radius']
        self.kb=cfg.get('kb',0)
        self.dur=cfg.get('dur',0)
        self.dur_left=self.dur
        self.status_kind=cfg.get('status_kind',None)
        self.status_val=cfg.get('status_val',1.0)
        self.active=True;self.applied=False
        self.name=cfg.get('name','')
        self.tick_dmg=cfg.get('tick_dmg',0)
        self.tick_ct_dmg=cfg.get('tick_ct_dmg',0)
        self.tick_interval=cfg.get('tick_interval',0)
        self.ticks_left=cfg.get('ticks_left',0)
        self.slow_pct=cfg.get('slow_pct',0)
        self.tick_cd=0
        self.volleys=cfg.get('volleys',1)
        self.volley_interval=cfg.get('volley_interval',0)
        self.volleys_left=self.volleys-1
        self.volley_cd=self.volley_interval
        self.proj_spd=cfg.get('projSpeed',0)
    def _hit_volley(self,game):
        opp=game._opp(self.team)
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=tdist(e,self.x,self.y)
            if d<=self.radius:
                e.take_damage(self.dmg)
                push(e,self.x,self.y,self.kb)
                if self.status_kind and hasattr(e,'statuses'):
                    e.statuses.append(Status(self.status_kind,self.dur,self.status_val))
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=tw.dist(self.x,self.y)
            if d<=self.radius:
                dm=self.ct_dmg if self.ct_dmg else self.dmg
                tw.take_damage(dm)
                if self.status_kind:tw.statuses.append(Status(self.status_kind,self.dur,self.status_val))
                if not tw.alive:game._tower_down(tw)
    def apply(self,game):
        if self.applied:return
        self.applied=True
        self._hit_volley(game)
        if self.dur<=0 and self.volleys_left<=0:self.active=False
        elif self.volleys_left>0:self.active=True
    def tick(self,dt,game=None):
        if self.volleys_left>0 and game:
            self.volley_cd-=dt
            if self.volley_cd<=1e-9:
                self.volley_cd=self.volley_interval
                self.volleys_left-=1
                self._hit_volley(game)
                if self.volleys_left<=0 and self.dur<=0:self.active=False
                return
        if self.tick_dmg>0 and self.ticks_left>0 and game:
            self.tick_cd-=dt
            if self.tick_cd<=0:
                self.tick_cd=self.tick_interval
                self.ticks_left-=1
                opp=game._opp(self.team)
                for e in game.players[opp].troops:
                    if not e.alive:continue
                    d=tdist(e,self.x,self.y)
                    if d<=self.radius:
                        e.take_damage(self.tick_dmg)
                        if self.slow_pct>0 and hasattr(e,'statuses'):
                            e.statuses.append(Status('mslow',self.tick_interval,1.0-self.slow_pct))
                for tw in game.arena.towers:
                    if tw.team!=opp or not tw.alive:continue
                    d=tw.dist(self.x,self.y)
                    if d<=self.radius:
                        tw.take_damage(self.tick_ct_dmg)
                        if not tw.alive:game._tower_down(tw)
                if self.ticks_left<=0:self.active=False
                return
        if self.dur>0:
            self.dur_left-=dt
            if self.dur_left<=0:self.active=False
class SpawnSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.tcfg=cfg['troop_cfg']
        self.count=cfg['count']
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
        self.proj_spd=cfg.get('projSpeed',0)
    def apply(self,game):
        if self.applied:return
        self.applied=True
        for i in range(self.count):
            ox=random.uniform(-1.0,1.0);oy=random.uniform(-1.0,1.0)
            spawn(game,self.team,self.x+ox,self.y+oy,self.tcfg)
        self.active=False
    def tick(self,dt,game=None):pass
def spawn(game,team,x,y,cfg):
    # a character a spell drops stands through its own deploy time like a played card
    t=Troop(team,x,y,dict(cfg,components=list(cfg.get('components',[]))))
    if 'hero' in cfg:t.ability=cfg['hero'](t);t.is_hero=True
    game._place(team,t,cfg.get('deploy',0))
    return t
class DecoyBarrelSpell(SpawnSpell):
    # the evo barrel: a decoy barrel lands on the mirrored tile of the other lane with its own goblins
    def __init__(self,team,x,y,cfg):
        super().__init__(team,x,y,cfg);self.dcfg=cfg['decoy_cfg'];self.dcount=cfg['decoy_count']
    def apply(self,game):
        if self.applied:return
        super().apply(game);mx=game.arena.W-self.x
        for i in range(self.dcount):spawn(game,self.team,mx+random.uniform(-1,1),self.y+random.uniform(-1,1),self.dcfg)
class GraveyardSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.tcfg=cfg['troop_cfg']
        self.total=cfg['total'];self.interval=cfg['interval']
        self.radius=cfg['radius'];self.min_radius=cfg.get('min_radius',cfg['radius']);self.dur=cfg['dur']
        self.active=False;self.applied=False
        self.spawned=0;self.timer=0;self.dur_left=self.dur
        self.name=cfg.get('name','')
        self.first_delay=cfg.get('first_delay',0)
    def apply(self,game):
        if self.applied:return
        self.applied=True;self.active=True;self.timer=self.first_delay
    def tick(self,dt,game=None):
        if not self.active or not game:return
        self.dur_left-=dt;self.timer-=dt
        if self.timer<0.001 and self.spawned<self.total:
            ang=random.uniform(0,2*math.pi);rr=random.uniform(self.min_radius,self.radius)
            ox=rr*math.cos(ang);oy=rr*math.sin(ang)
            t=Troop(self.team,min(max(self.x+ox,0.3),17.7),min(max(self.y+oy,0.3),31.7),dict(self.tcfg,components=list(self.tcfg.get('components',[]))))
            game.players[self.team].troops.append(t)
            self.spawned+=1;self.timer=self.interval
        if self.spawned>=self.total or self.dur_left<=0:self.active=False
class RageSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.radius=cfg['radius']
        self.rage_boost=cfg['rage_boost']
        self.rage_dur=cfg['rage_dur']
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
        self.dur_left=self.rage_dur;self.tick_cd=0
    def apply(self,game):
        if self.applied:return
        self.applied=True;self.active=True
        opp=game._opp(self.team)
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=tdist(e,self.x,self.y)
            if d<=self.radius:e.take_damage(self.dmg)
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=tw.dist(self.x,self.y)
            if d<=self.radius:
                dm=self.ct_dmg if self.ct_dmg else self.dmg
                tw.take_damage(dm)
                if not tw.alive:game._tower_down(tw)
        for ally in game.players[self.team].troops:
            if not ally.alive:continue
            d=math.sqrt((ally.x-self.x)**2+(ally.y-self.y)**2)
            if d<=self.radius:
                ally.statuses.append(Status('rage',self.rage_dur,self.rage_boost))
    def tick(self,dt,game=None):
        if not self.active or not game:return
        self.dur_left-=dt
        if self.dur_left<=0:self.active=False;return
        self.tick_cd-=dt
        if self.tick_cd<=0:
            self.tick_cd=0.5
            for ally in game.players[self.team].troops:
                if not ally.alive:continue
                d=math.sqrt((ally.x-self.x)**2+(ally.y-self.y)**2)
                if d<=self.radius:
                    has=any(s.kind=='rage' for s in ally.statuses)
                    if not has:
                        ally.statuses.append(Status('rage',min(1.0,self.dur_left),self.rage_boost))
class LightningSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.radius=cfg['radius']
        self.max_tgt=cfg['max_targets']
        self.stun_dur=cfg['stun_dur']
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
    def apply(self,game):
        if self.applied:return
        self.applied=True
        opp=game._opp(self.team)
        cands=[]
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=tdist(e,self.x,self.y)
            if d<=self.radius:cands.append((-getattr(e,'max_hp',e.hp),e,'troop'))
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=tw.dist(self.x,self.y)
            if d<=self.radius:cands.append((-getattr(tw,'max_hp',tw.hp),tw,'tower'))
        cands.sort(key=lambda x:x[0])
        for _,tgt,kind in cands[:self.max_tgt]:
            dm=self.ct_dmg if kind=='tower' and self.ct_dmg else self.dmg
            tgt.take_damage(dm)
            if kind=='tower' and not tgt.alive:game._tower_down(tgt)
            if self.stun_dur>0:tgt.statuses.append(Status('stun',self.stun_dur))
        self.active=False
    def tick(self,dt,game=None):pass
class CloneSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
    def apply(self,game):
        if self.applied:return
        self.applied=True
        clones=[]
        for t in game.players[self.team].troops:
            if not t.alive:continue
            if getattr(t,'is_building',False):continue
            if t.hp==1 and t.max_hp==1:continue
            d=math.sqrt((t.x-self.x)**2+(t.y-self.y)**2)
            if d<=self.radius:
                oy=-0.5 if self.team=='blue' else 0.5
                cfg={'hp':1,'max_hp':1,'dmg':t.dmg,'hspd':t.hspd,'fhspd':t.fhspd,
                     'spd':t.spd,'rng':t.rng,'targets':t.targets,
                     'transport':t.transport,'atk_type':t.atk_type,
                     'splash_r':t.splash_r,'ct_dmg':t.ct_dmg,
                     'components':list(t.components),'lvl':t.lvl,'name':t.name,
                     'death_dmg':getattr(t,'death_dmg',0),
                     'death_splash_r':getattr(t,'death_splash_r',0)}
                cl=Troop(self.team,t.x,t.y+oy,cfg)
                cl.ability=None
                clones.append(cl)
        for c in clones:
            c.max_hp=1;c.hp=1
            game.players[self.team].troops.append(c)
        self.active=False
    def tick(self,dt,game=None):pass
class LogSpell:
    # dropped at the cast point, it rolls forward at the pierce speed and hits each body once as the front reaches it
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.rng=cfg['range'];self.width=cfg['width'];self.speed=cfg.get('speed',0)
        self.pushback=cfg.get('pushback',0)
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
        self.front=0.0;self.hit=[]
    def _dir(self):return 1 if self.team=='blue' else -1
    def _sweep(self,game,a,b):
        d=self._dir()
        for e in strip(game,self.team,self.x,self.y+d*a,self.x,self.y+d*b,self.width/2.0,air=False,skip=self.hit):
            self.hit.append(e);hurt(e,self.ct_dmg if hasattr(e,'ttype') and self.ct_dmg else self.dmg,game)
            if not hasattr(e,'ttype') and not getattr(e,'is_building',False):e.y+=d*self.pushback
    def apply(self,game):
        if self.applied:return
        self.applied=True
        if self.speed>0:self.active=True;return
        self._sweep(game,0,self.rng);self.done(game)
    def tick(self,dt,game=None):
        if not self.active or not game:return
        f=min(self.rng,self.front+self.speed*dt);self._sweep(game,self.front,f);self.front=f
        if self.front>=self.rng:self.active=False;self.done(game)
    def done(self,game):pass
class EarthquakeSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.troop_dmg=cfg['troop_dmg']
        self.bldg_dmg=cfg['bldg_dmg']
        self.ct_dmg=cfg['ct_dmg']
        self.ticks=cfg['ticks'];self.interval=cfg['interval']
        self.slow_pct=cfg.get('slow_pct',0)
        self.active=True;self.applied=False
        self.ticks_left=self.ticks;self.tick_cd=0
        self.name=cfg.get('name','')
    def apply(self,game):
        if self.applied:return
        self.applied=True
    def tick(self,dt,game=None):
        if not game or self.ticks_left<=0:self.active=False;return
        self.tick_cd-=dt
        if self.tick_cd<=0:
            self.tick_cd=self.interval;self.ticks_left-=1
            opp=game._opp(self.team)
            for e in game.players[opp].troops:
                if not e.alive:continue
                if getattr(e,'transport','Ground')=='Air':continue
                d=tdist(e,self.x,self.y)
                if d<=self.radius:
                    dm=self.bldg_dmg if getattr(e,'is_building',False) else self.troop_dmg
                    e.take_damage(dm)
                    if self.slow_pct>0 and hasattr(e,'statuses'):
                        e.statuses.append(Status('mslow',self.interval,1.0-self.slow_pct))
            for tw in game.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=tw.dist(self.x,self.y)
                if d<=self.radius:
                    tw.take_damage(self.ct_dmg)
                    if not tw.alive:game._tower_down(tw)
            if self.ticks_left<=0:self.active=False
class TornadoSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.tick_dmg=cfg['tick_dmg']
        self.ct_dmg=cfg.get('ct_dmg',0)
        self.ticks=cfg['ticks'];self.interval=cfg['interval']
        self.pull_str=cfg['pull_str']
        self.dur=cfg['dur']
        self.active=True;self.applied=False
        self.ticks_left=self.ticks;self.tick_cd=0
        self.dur_left=self.dur;self.pull_cd=0
        self.name=cfg.get('name','')
    def apply(self,game):
        if self.applied:return
        self.applied=True
    def tick(self,dt,game=None):
        if not game:return
        self.dur_left-=dt
        opp=game._opp(self.team)
        self.pull_cd-=dt
        if self.pull_cd<=0:
            self.pull_cd=dt
            for e in game.players[opp].troops:
                if not e.alive:continue
                dx=self.x-e.x;dy=self.y-e.y
                d=math.sqrt(dx*dx+dy*dy)
                if d<=self.radius and d>0.1:
                    mv=self.pull_str*dt
                    e.x+=dx/d*mv;e.y+=dy/d*mv
        if self.ticks_left>0:
            self.tick_cd-=dt
            if self.tick_cd<=0:
                self.tick_cd=self.interval;self.ticks_left-=1
                for e in game.players[opp].troops:
                    if not e.alive:continue
                    if getattr(e,'is_building',False):continue
                    d=tdist(e,self.x,self.y)
                    if d<=self.radius:e.take_damage(self.tick_dmg)
                for tw in game.arena.towers:
                    if tw.team!=opp or not tw.alive:continue
                    d=tw.dist(self.x,self.y)
                    if d<=self.radius:
                        tw.take_damage(self.ct_dmg)
                        if not tw.alive:game._tower_down(tw)
        if self.dur_left<=0 and self.ticks_left<=0:self.active=False
class VoidSpell:
    # damage per strike drops by target count: tiers[i] applies while count<=max_units[i], the last tier beyond
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.tiers=cfg['tiers'];self.tower_tiers=cfg['tower_tiers'];self.max_units=cfg['max_units']
        self.strikes=cfg['strikes'];self.interval=cfg['interval']
        self.active=True;self.applied=False
        self.strikes_left=self.strikes;self.tick_cd=cfg['first_delay']
        self.name=cfg.get('name','')
    def _tier(self,n):
        return next((i for i,m in enumerate(self.max_units) if n<=m),len(self.tiers)-1)
    def apply(self,game):
        if self.applied:return
        self.applied=True
    def tick(self,dt,game=None):
        if not game or self.strikes_left<=0:self.active=False;return
        self.tick_cd-=dt
        if self.tick_cd<=0:
            self.tick_cd=self.interval;self.strikes_left-=1
            opp=game._opp(self.team)
            tgts=[]
            for e in game.players[opp].troops:
                if not e.alive:continue
                d=tdist(e,self.x,self.y)
                if d<=self.radius:tgts.append(('troop',e))
            for tw in game.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=tw.dist(self.x,self.y)
                if d<=self.radius:tgts.append(('tower',tw))
            i=self._tier(len(tgts))
            for kind,t in tgts:
                if kind=='tower':
                    t.take_damage(self.tower_tiers[i])
                    if not t.alive:game._tower_down(t)
                else:
                    t.take_damage(self.tiers[i])
            if self.strikes_left<=0:self.active=False
class VinesSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.max_tgt=cfg['max_targets']
        self.dur=cfg['dur']
        self.tick_dmg=cfg['tick_dmg']
        self.tick_interval=cfg['tick_interval'];self.ticks_left=cfg['ticks']
        self.active=True;self.applied=False
        self.dur_left=self.dur;self.tick_cd=self.tick_interval
        self.rooted=[];self.orig_transport={}
        self.name=cfg.get('name','')
    def apply(self,game):
        if self.applied:return
        self.applied=True
        opp=game._opp(self.team)
        cands=[]
        for e in game.players[opp].troops:
            if not e.alive:continue
            if getattr(e,'is_building',False):continue
            d=tdist(e,self.x,self.y)
            if d<=self.radius:cands.append((-getattr(e,'max_hp',e.hp),e))
        cands.sort(key=lambda x:x[0])
        for _,e in cands[:self.max_tgt]:
            self.orig_transport[id(e)]=e.transport
            e.transport='Ground'
            e.statuses.append(Status('freeze',self.dur))
            self.rooted.append(e)
    def tick(self,dt,game=None):
        if not game:return
        self.dur_left-=dt;self.tick_cd-=dt
        if self.tick_cd<=0.001 and self.ticks_left>0:
            self.tick_cd=self.tick_interval;self.ticks_left-=1
            for e in self.rooted:
                if e.alive:e.take_damage(self.tick_dmg)
        if self.dur_left<=0:
            for e in self.rooted:
                oid=id(e)
                if oid in self.orig_transport:
                    e.transport=self.orig_transport[oid]
            self.active=False
class GoblinCurseSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.tick_dmg=cfg['tick_dmg']
        self.ct_dmg=cfg.get('ct_dmg',0)
        self.ticks=cfg['ticks']
        self.interval=cfg['interval']
        self.gcfg=cfg.get('goblin_cfg',{})
        self.active=True;self.applied=False
        self.ticks_left=self.ticks;self.tick_cd=0
        self.cursed=[]
        self.name=cfg.get('name','')
    def apply(self,game):
        if self.applied:return
        self.applied=True
        opp=game._opp(self.team)
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=tdist(e,self.x,self.y)
            if d<=self.radius:self.cursed.append(e)
    def tick(self,dt,game=None):
        if not game:return
        if self.ticks_left>0:
            self.tick_cd-=dt
            if self.tick_cd<=0:
                self.tick_cd=self.interval;self.ticks_left-=1
                opp=game._opp(self.team)
                for e in self.cursed:
                    if e.alive:e.take_damage(self.tick_dmg)
                for tw in game.arena.towers:
                    if tw.team!=opp or not tw.alive:continue
                    d=tw.dist(self.x,self.y)
                    if d<=self.radius:
                        tw.take_damage(self.ct_dmg)
                        if not tw.alive:game._tower_down(tw)
        dead_cursed=[e for e in self.cursed if not e.alive]
        for e in dead_cursed:
            self.cursed.remove(e)
            if self.gcfg:
                t=Troop(self.team,e.x,e.y,dict(self.gcfg,components=list(self.gcfg.get('components',[]))))
                game.players[self.team].troops.append(t)
        if self.ticks_left<=0 and not self.cursed:self.active=False
class RoyalDeliverySpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.radius=cfg['radius']
        self.tcfg=cfg.get('troop_cfg',{})
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
        self.proj_spd=cfg.get('projSpeed',0)
    def apply(self,game):
        if self.applied:return
        self.applied=True
        opp=game._opp(self.team)
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=tdist(e,self.x,self.y)
            if d<=self.radius:e.take_damage(self.dmg)
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=tw.dist(self.x,self.y)
            if d<=self.radius:
                dm=self.ct_dmg if self.ct_dmg else self.dmg
                tw.take_damage(dm)
                if not tw.alive:game._tower_down(tw)
        if self.tcfg:
            t=Troop(self.team,self.x,self.y,dict(self.tcfg,components=list(self.tcfg.get('components',[]))))
            game.players[self.team].troops.append(t)
        self.active=False
    def tick(self,dt,game=None):pass
class BarbarianBarrelSpell(LogSpell):
    def __init__(self,team,x,y,cfg):
        super().__init__(team,x,y,cfg)
        self.tcfg=cfg.get('troop_cfg',{})
    def done(self,game):
        if self.tcfg:spawn(game,self.team,self.x,self.y+self._dir()*self.rng,self.tcfg)
class EvoZapSpell(Spell):
    def __init__(self,team,x,y,cfg):
        super().__init__(team,x,y,cfg)
        self.r2=cfg['pulse_2_radius'];self.p2_delay=cfg['pulse_2_delay'];self.p2_done=False
    def apply(self,game):
        super().apply(game);self.active=True;self.p2_timer=self.p2_delay
    def tick(self,dt,game):
        if self.p2_done:self.active=False;return
        self.p2_timer-=dt
        if self.p2_timer<=0:
            opp='red' if self.team=='blue' else 'blue'
            for e in game.players[opp].troops:
                if not e.alive:continue
                d=tdist(e,self.x,self.y)
                if d<=self.r2:
                    e.take_damage(self.dmg)
                    if hasattr(e,'statuses') and self.dur>0:
                        e.statuses.append(Status('stun',self.dur))
            for tw in game.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=tw.dist(self.x,self.y)
                if d<=self.r2:
                    td=self.ct_dmg if self.ct_dmg else self.dmg
                    tw.take_damage(td)
                    if self.dur>0:tw.statuses.append(Status('stun',self.dur))
                    if not tw.alive:game._tower_down(tw)
            self.p2_done=True;self.active=False
class EvoSnowballSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.radius=cfg['radius'];self.kb=cfg.get('kb',0)
        self.roll_dist=cfg['roll_distance'];self.roll_dur=cfg['roll_duration']
        self.slow_dur=cfg['slow_duration'];self.slow_val=cfg['status_val']
        self.active=False;self.name=cfg.get('name','')
        self.proj_spd=cfg.get('projSpeed',0)
        self.captured=[];self.rolling=False;self.roll_t=0
        self.rx=self.x;self.ry=self.y;self.dir_y=0
    def apply(self,game):
        opp='red' if self.team=='blue' else 'blue'
        self.dir_y=1 if self.team=='blue' else -1
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=tdist(e,self.x,self.y)
            if d<=self.radius:
                e.take_damage(self.dmg)
                if hasattr(e,'statuses'):e.statuses.append(Status('slow',self.slow_dur,self.slow_val))
                self.captured.append(e)
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=tw.dist(self.x,self.y)
            if d<=self.radius:
                td=self.ct_dmg if self.ct_dmg else self.dmg
                tw.take_damage(td)
                if not tw.alive:game._tower_down(tw)
        self.rolling=True;self.roll_t=0;self.active=True
    def tick(self,dt,game):
        if not self.rolling:self.active=False;return
        self.roll_t+=dt
        spd=self.roll_dist/self.roll_dur
        self.ry+=self.dir_y*spd*dt
        for e in self.captured:
            if e.alive:e.x=self.rx;e.y=self.ry
        if self.roll_t>=self.roll_dur:
            self.rolling=False;self.active=False
