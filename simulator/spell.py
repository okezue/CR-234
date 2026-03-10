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
        self.active=True;self.applied=False
        self.name=cfg.get('name','')
        self.tick_dmg=cfg.get('tick_dmg',0)
        self.tick_ct_dmg=cfg.get('tick_ct_dmg',0)
        self.tick_interval=cfg.get('tick_interval',0)
        self.ticks_left=cfg.get('ticks_left',0)
        self.slow_pct=cfg.get('slow_pct',0)
        self.tick_cd=0
    def apply(self,game):
        if self.applied:return
        self.applied=True
        opp=game._opp(self.team)
        for e in game.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-self.x)**2+(e.y-self.y)**2)
            if d<=self.radius:
                e.take_damage(self.dmg)
                if self.kb>0:
                    dx=e.x-self.x;dy=e.y-self.y
                    ds=math.sqrt(dx*dx+dy*dy)
                    if ds>0:e.x+=dx/ds*self.kb;e.y+=dy/ds*self.kb
                if self.status_kind and hasattr(e,'statuses'):
                    e.statuses.append(Status(self.status_kind,self.dur))
        for tw in game.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-self.x)**2+(tw.cy-self.y)**2)
            if d<=self.radius:
                dm=self.ct_dmg if self.ct_dmg else self.dmg
                tw.take_damage(dm)
                if not tw.alive:game._tower_down(tw)
        if self.dur<=0:self.active=False
    def tick(self,dt,game=None):
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
