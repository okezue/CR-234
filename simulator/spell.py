import math,sys,os,random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from status import Status
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
        self.volley_interval=cfg.get('volley_interval',0.15)
        self.volleys_left=self.volleys-1
        self.volley_cd=self.volley_interval
    def _hit_volley(self,game):
        opp=game._opp(self.team)
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
            if d<=self.radius:
                e.take_damage(self.dmg)
                if self.kb>0 and getattr(e,'mass',4)<6:
                    dx=e.x-self.x;dy=e.y-self.y
                    ds=math.sqrt(dx*dx+dy*dy)
                    if ds>0:e.x+=dx/ds*self.kb;e.y+=dy/ds*self.kb
                if self.status_kind and hasattr(e,'statuses'):
                    e.statuses.append(Status(self.status_kind,self.dur,self.status_val))
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
            if d<=self.radius:
                dm=self.ct_dmg if self.ct_dmg else self.dmg
                tw.take_damage(dm)
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
            if self.volley_cd<=0:
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
                    d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
                    if d<=self.radius:
                        e.take_damage(self.tick_dmg)
                        if self.slow_pct>0 and hasattr(e,'statuses'):
                            e.statuses.append(Status('slow',self.tick_interval,1.0-self.slow_pct))
                for tw in game.arena.towers:
                    if tw.team!=opp or not tw.alive:continue
                    d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
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
    def apply(self,game):
        if self.applied:return
        self.applied=True
        from troop import Troop
        for i in range(self.count):
            ox=random.uniform(-1.0,1.0);oy=random.uniform(-1.0,1.0)
            t=Troop(self.team,self.x+ox,self.y+oy,dict(self.tcfg,components=list(self.tcfg.get('components',[]))))
            game.players[self.team].troops.append(t)
        self.active=False
    def tick(self,dt,game=None):pass
class GraveyardSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.tcfg=cfg['troop_cfg']
        self.total=cfg['total'];self.interval=cfg['interval']
        self.radius=cfg['radius'];self.dur=cfg['dur']
        self.active=False;self.applied=False
        self.spawned=0;self.timer=0;self.dur_left=self.dur
        self.name=cfg.get('name','')
        self.first_delay=cfg.get('first_delay',2.2)
    def apply(self,game):
        if self.applied:return
        self.applied=True;self.active=True;self.timer=self.first_delay
    def tick(self,dt,game=None):
        if not self.active or not game:return
        self.dur_left-=dt;self.timer-=dt
        if self.timer<0.001 and self.spawned<self.total:
            from troop import Troop
            ang=random.uniform(0,2*math.pi)
            ox=self.radius*math.cos(ang)
            oy=self.radius*math.sin(ang)
            t=Troop(self.team,self.x+ox,self.y+oy,dict(self.tcfg,components=list(self.tcfg.get('components',[]))))
            game.players[self.team].troops.append(t)
            self.spawned+=1;self.timer=self.interval
        if self.spawned>=self.total or self.dur_left<=0:self.active=False
class RageSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.radius=cfg['radius']
        self.rage_boost=cfg.get('rage_boost',0.3)
        self.rage_dur=cfg.get('rage_dur',4.5)
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
        self.dur_left=self.rage_dur;self.tick_cd=0
    def apply(self,game):
        if self.applied:return
        self.applied=True;self.active=True
        opp=game._opp(self.team)
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
            if d<=self.radius:e.take_damage(self.dmg)
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
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
        self.max_tgt=cfg.get('max_targets',3)
        self.stun_dur=cfg.get('stun_dur',0.5)
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
    def apply(self,game):
        if self.applied:return
        self.applied=True
        opp=game._opp(self.team)
        cands=[]
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
            if d<=self.radius:cands.append((-getattr(e,'max_hp',e.hp),e,'troop'))
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
            if d<=self.radius:cands.append((-getattr(tw,'max_hp',tw.hp),tw,'tower'))
        cands.sort(key=lambda x:x[0])
        for _,tgt,kind in cands[:self.max_tgt]:
            dm=self.ct_dmg if kind=='tower' and self.ct_dmg else self.dmg
            tgt.take_damage(dm)
            if kind=='tower' and not tgt.alive:game._tower_down(tgt)
            if kind=='troop' and hasattr(tgt,'statuses') and self.stun_dur>0:
                tgt.statuses.append(Status('stun',self.stun_dur))
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
        from troop import Troop
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
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.rng=cfg['range'];self.width=cfg['width']
        self.pushback=cfg.get('pushback',0)
        self.active=False;self.applied=False
        self.name=cfg.get('name','')
    def apply(self,game):
        if self.applied:return
        self.applied=True
        opp=game._opp(self.team)
        hw=self.width/2.0
        if self.team=='blue':y0,y1=self.y,self.y+self.rng
        else:y0,y1=self.y-self.rng,self.y
        for e in game.players[opp].troops:
            if not e.alive:continue
            if getattr(e,'transport','Ground')=='Air':continue
            if y0<=e.y<=y1 and self.x-hw<=e.x<=self.x+hw:
                e.take_damage(self.dmg)
                if self.pushback>0:
                    if self.team=='blue':e.y+=self.pushback
                    else:e.y-=self.pushback
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            if y0<=tw.cy<=y1 and self.x-hw<=tw.cx<=self.x+hw:
                tw.take_damage(self.ct_dmg if self.ct_dmg else self.dmg)
                if not tw.alive:game._tower_down(tw)
        self.active=False
    def tick(self,dt,game=None):pass
class EarthquakeSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.troop_dmg=cfg['troop_dmg']
        self.bldg_dmg=cfg['bldg_dmg']
        self.ct_dmg=cfg['ct_dmg']
        self.ticks=cfg['ticks'];self.interval=cfg['interval']
        self.slow_pct=cfg.get('slow_pct',0.5)
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
                d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
                if d<=self.radius:
                    dm=self.bldg_dmg if getattr(e,'is_building',False) else self.troop_dmg
                    e.take_damage(dm)
                    if self.slow_pct>0 and hasattr(e,'statuses'):
                        e.statuses.append(Status('slow',self.interval,1.0-self.slow_pct))
            for tw in game.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
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
        self.pull_str=cfg.get('pull_str',3.6)
        self.dur=cfg.get('dur',1.05)
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
                    kt=game.arena.get_tower(self.team,'king')
                    if kt and not getattr(kt,'active',True):
                        kd=math.sqrt((e.x-kt.cx)**2+(e.y-kt.cy)**2)
                        if kd<=kt.rng:game._king_act(self.team)
        if self.ticks_left>0:
            self.tick_cd-=dt
            if self.tick_cd<=0:
                self.tick_cd=self.interval;self.ticks_left-=1
                for e in game.players[opp].troops:
                    if not e.alive:continue
                    if getattr(e,'is_building',False):continue
                    d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
                    if d<=self.radius:e.take_damage(self.tick_dmg)
                for tw in game.arena.towers:
                    if tw.team!=opp or not tw.alive:continue
                    d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
                    if d<=self.radius:
                        tw.take_damage(self.ct_dmg)
                        if not tw.alive:game._tower_down(tw)
        if self.dur_left<=0 and self.ticks_left<=0:self.active=False
class VoidSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.single_dmg=cfg['single_dmg']
        self.ct_dmg=cfg.get('ct_dmg',0)
        self.strikes=cfg.get('strikes',3)
        self.interval=cfg.get('interval',1.0)
        self.active=True;self.applied=False
        self.strikes_left=self.strikes;self.tick_cd=0
        self.name=cfg.get('name','')
    def _scale(self,n):
        if n<=1:return 1.0
        if n<=4:return 0.47
        return 0.22
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
                d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
                if d<=self.radius:tgts.append(('troop',e))
            for tw in game.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
                if d<=self.radius:tgts.append(('tower',tw))
            n=len(tgts);sc=self._scale(n)
            for kind,t in tgts:
                if kind=='tower':
                    t.take_damage(int(self.ct_dmg*sc))
                    if not t.alive:game._tower_down(t)
                else:
                    t.take_damage(int(self.single_dmg*sc))
            if self.strikes_left<=0:self.active=False
class VinesSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.radius=cfg['radius']
        self.max_tgt=cfg.get('max_targets',3)
        self.dur=cfg.get('dur',2.0)
        self.tick_dmg=cfg.get('tick_dmg',80)
        self.tick_interval=cfg.get('tick_interval',0.5)
        self.active=True;self.applied=False
        self.dur_left=self.dur;self.tick_cd=0
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
            d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
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
        if self.tick_cd<=0:
            self.tick_cd=self.tick_interval
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
        self.tick_dmg=cfg.get('tick_dmg',35)
        self.ct_dmg=cfg.get('ct_dmg',0)
        self.ticks=cfg.get('ticks',6)
        self.interval=cfg.get('interval',1.0)
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
            d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
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
                    d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
                    if d<=self.radius:
                        tw.take_damage(self.ct_dmg)
                        if not tw.alive:game._tower_down(tw)
        dead_cursed=[e for e in self.cursed if not e.alive]
        for e in dead_cursed:
            self.cursed.remove(e)
            if self.gcfg:
                from troop import Troop
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
    def apply(self,game):
        if self.applied:return
        self.applied=True
        opp=game._opp(self.team)
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
            if d<=self.radius:e.take_damage(self.dmg)
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
            if d<=self.radius:
                dm=self.ct_dmg if self.ct_dmg else self.dmg
                tw.take_damage(dm)
                if not tw.alive:game._tower_down(tw)
        if self.tcfg:
            from troop import Troop
            t=Troop(self.team,self.x,self.y,dict(self.tcfg,components=list(self.tcfg.get('components',[]))))
            game.players[self.team].troops.append(t)
        self.active=False
    def tick(self,dt,game=None):pass
class BarbarianBarrelSpell(LogSpell):
    def __init__(self,team,x,y,cfg):
        super().__init__(team,x,y,cfg)
        self.tcfg=cfg.get('troop_cfg',{})
    def apply(self,game):
        super().apply(game)
        if self.tcfg:
            from troop import Troop
            if self.team=='blue':sy=self.y+self.rng
            else:sy=self.y-self.rng
            t=Troop(self.team,self.x,sy,dict(self.tcfg,components=list(self.tcfg.get('components',[]))))
            game.players[self.team].troops.append(t)
class EvoZapSpell(Spell):
    def __init__(self,team,x,y,cfg):
        super().__init__(team,x,y,cfg)
        self.r2=cfg.get('pulse_2_radius',3.0);self.p2_delay=0.5;self.p2_done=False
    def apply(self,game):
        super().apply(game);self.active=True;self.p2_timer=self.p2_delay
    def tick(self,dt,game):
        if self.p2_done:self.active=False;return
        self.p2_timer-=dt
        if self.p2_timer<=0:
            opp='red' if self.team=='blue' else 'blue'
            for e in game.players[opp].troops:
                if not e.alive:continue
                d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
                if d<=self.r2:
                    e.take_damage(self.dmg)
                    if hasattr(e,'statuses') and self.dur>0:
                        e.statuses.append(Status('stun',self.dur))
            for tw in game.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
                if d<=self.r2:
                    td=self.ct_dmg if self.ct_dmg else self.dmg
                    tw.take_damage(td)
                    if not tw.alive:game._tower_down(tw)
            self.p2_done=True;self.active=False
class EvoSnowballSpell:
    def __init__(self,team,x,y,cfg):
        self.team=team;self.x=float(x);self.y=float(y)
        self.dmg=cfg['dmg'];self.ct_dmg=cfg.get('ct_dmg',0)
        self.radius=cfg['radius'];self.kb=cfg.get('kb',0)
        self.roll_dist=cfg.get('roll_distance',4.5);self.roll_dur=cfg.get('roll_duration',0.75)
        self.slow_dur=cfg.get('slow_duration',4.0)
        self.active=False;self.name=cfg.get('name','')
        self.captured=[];self.rolling=False;self.roll_t=0
        self.rx=self.x;self.ry=self.y;self.dir_y=0
    def apply(self,game):
        opp='red' if self.team=='blue' else 'blue'
        self.dir_y=1 if self.team=='blue' else -1
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
            if d<=self.radius:
                e.take_damage(self.dmg)
                if hasattr(e,'statuses'):e.statuses.append(Status('slow',self.slow_dur,0.65))
                self.captured.append(e)
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
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
