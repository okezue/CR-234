import math,random
import sys,os
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from status import Status
class Component:
    def on_tick(self,tr,g):pass
    def on_attack(self,tr,tgt,g):pass
    def on_take_damage(self,tr,d,g):pass
    def on_death(self,tr,g):pass
    def modify_target(self,tr,c,g):return c
class SplashAttack(Component):
    def on_attack(self,tr,tgt,g):
        opp=g._opp(tr.team)
        if hasattr(tgt,'cx'):tx,ty=tgt.cx,tgt.cy
        else:tx,ty=tgt.x,tgt.y
        for e in g.players[opp].troops:
            if not e.alive or e is tgt:continue
            d=math.sqrt((e.x-tx)**2+(e.y-ty)**2)
            if d<=tr.splash_r:e.take_damage(tr.dmg)
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive or tw is tgt:continue
            d=math.sqrt((tw.cx-tx)**2+(tw.cy-ty)**2)
            if d<=tr.splash_r:
                tw.take_damage(tr.dmg)
                if not tw.alive:g._tower_down(tw)
class BuildingTarget(Component):
    def modify_target(self,tr,c,g):
        return [(d,t) for d,t in c if hasattr(t,'ttype')]
class RiverJump(Component):
    def on_tick(self,tr,g):
        if 14.0<tr.y<17.0:
            tr.y=17.0 if tr.team=='blue' else 14.0
class Charge(Component):
    def __init__(self,dist=2.5):
        self.dist=dist;self.moved=0;self.charged=False
        self.px=None;self.py=None;self.orig_spd=None
    def on_tick(self,tr,g):
        if self.px is not None:
            dx=tr.x-self.px;dy=tr.y-self.py
            self.moved+=math.sqrt(dx*dx+dy*dy)
        self.px=tr.x;self.py=tr.y
        if any(s.kind in ('stun','freeze') for s in getattr(tr,'statuses',[])):
            if self.charged and self.orig_spd is not None:
                tr.spd=self.orig_spd
            self.charged=False;self.moved=0;return
        if not self.charged and self.moved>=self.dist:
            self.charged=True;self.orig_spd=tr.spd;tr.spd*=2
    def on_attack(self,tr,tgt,g):
        if not self.charged:return
        extra=getattr(tr,'charge_dmg',tr.dmg*2)-tr.dmg
        if extra>0:
            tgt.take_damage(extra)
            if hasattr(tgt,'ttype') and not tgt.alive:g._tower_down(tgt)
        if self.orig_spd is not None:tr.spd=self.orig_spd
        self.charged=False;self.moved=0
class SpawnTimer(Component):
    def __init__(self,cfg,interval,count,first_delay):
        self.cfg=cfg;self.interval=interval;self.count=count
        self.timer=first_delay
    def on_tick(self,tr,g):
        self.timer-=g.DT
        if self.timer<=0:
            from troop import Troop
            for i in range(self.count):
                ox=random.uniform(-1.5,1.5);oy=random.uniform(-1.5,1.5)
                t=Troop(tr.team,tr.x+ox,tr.y+oy,dict(self.cfg,components=list(self.cfg.get('components',[]))))
                g.players[tr.team].troops.append(t)
            self.timer=self.interval
class DeathDamage(Component):
    def on_death(self,tr,g):
        dd=getattr(tr,'death_dmg',0)
        dr=getattr(tr,'death_splash_r',2.0)
        if dd<=0:return
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=dr:e.take_damage(dd)
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-tr.x)**2+(tw.cy-tr.y)**2)
            if d<=dr:
                tw.take_damage(dd)
                if not tw.alive:g._tower_down(tw)
class DeathSpawn(Component):
    def __init__(self,cfg,count):
        self.cfg=cfg;self.count=count
    def on_death(self,tr,g):
        from troop import Troop
        for i in range(self.count):
            ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
            t=Troop(tr.team,tr.x+ox,tr.y+oy,dict(self.cfg,components=list(self.cfg.get('components',[]))))
            g.players[tr.team].troops.append(t)
class SpawnZap(Component):
    def __init__(self):
        self.fired=False
    def on_tick(self,tr,g):
        if self.fired:return
        self.fired=True
        opp=g._opp(tr.team)
        r=getattr(tr,'spawn_zap_r',3.0)
        dmg=getattr(tr,'spawn_zap_dmg',0)
        sd=getattr(tr,'stun_dur',0.5)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=r:
                e.take_damage(dmg)
                if hasattr(e,'statuses'):e.statuses.append(Status('stun',sd))
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-tr.x)**2+(tw.cy-tr.y)**2)
            if d<=r:
                tw.take_damage(dmg)
                if not tw.alive:g._tower_down(tw)
class RampUp(Component):
    def __init__(self,stages,durations):
        self.stages=stages;self.durations=durations
        self.cur_tgt=None;self.elapsed=0
    def on_tick(self,tr,g):
        tgt=getattr(tr,'tgt',None)
        if tgt is not self.cur_tgt:
            self.cur_tgt=tgt;self.elapsed=0;tr.dmg=self.stages[0]
            return
        self.elapsed+=g.DT
        t=0
        for i,d in enumerate(self.durations):
            t+=d
            if self.elapsed<t:tr.dmg=self.stages[i];return
        tr.dmg=self.stages[-1]
class RageDrop(Component):
    def __init__(self,radius,dur,boost):
        self.radius=radius;self.dur=dur;self.boost=boost
    def on_death(self,tr,g):
        for ally in g.players[tr.team].troops:
            if not ally.alive or ally is tr:continue
            d=math.sqrt((ally.x-tr.x)**2+(ally.y-tr.y)**2)
            if d<=self.radius:
                ally.statuses.append(Status('rage',self.dur,self.boost))
class DualTarget(Component):
    def on_attack(self,tr,tgt,g):
        sd=getattr(tr,'stun_dur',0.5)
        if hasattr(tgt,'statuses'):tgt.statuses.append(Status('stun',sd))
        opp=g._opp(tr.team)
        cands=[]
        for e in g.players[opp].troops:
            if not e.alive or e is tgt:continue
            d=math.sqrt((tr.x-e.x)**2+(tr.y-e.y)**2)
            if d<=tr.rng:cands.append((d,e))
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive or tw is tgt:continue
            d=math.sqrt((tr.x-tw.cx)**2+(tr.y-tw.cy)**2)
            if d<=tr.rng:cands.append((d,tw))
        if cands:
            cands.sort(key=lambda x:x[0])
            t2=cands[0][1]
            t2.take_damage(tr.dmg)
            if hasattr(t2,'statuses'):t2.statuses.append(Status('stun',sd))
            if hasattr(t2,'ttype') and not t2.alive:g._tower_down(t2)
        else:
            tgt.take_damage(tr.dmg)
            if hasattr(tgt,'ttype') and not tgt.alive:g._tower_down(tgt)
