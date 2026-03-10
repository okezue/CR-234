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
        sd=getattr(tr,'slow_dur',0)
        sv=getattr(tr,'slow_val',1.0)
        fd=getattr(tr,'freeze_dur',0)
        atgts=getattr(tr,'targets',['Ground'])
        if sd>0 and hasattr(tgt,'statuses'):
            tgt.statuses.append(Status('slow',sd,sv))
        if fd>0 and hasattr(tgt,'statuses'):
            tgt.statuses.append(Status('freeze',fd))
        for e in g.players[opp].troops:
            if not e.alive or e is tgt:continue
            et=getattr(e,'transport','Ground')
            if et=='Air' and 'Air' not in atgts:continue
            d=math.sqrt((e.x-tx)**2+(e.y-ty)**2)
            if d<=tr.splash_r:
                e.take_damage(tr.dmg)
                if sd>0 and hasattr(e,'statuses'):
                    e.statuses.append(Status('slow',sd,sv))
                if fd>0 and hasattr(e,'statuses'):
                    e.statuses.append(Status('freeze',fd))
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive or tw is tgt:continue
            d=math.sqrt((tw.cx-tx)**2+(tw.cy-ty)**2)
            if d<=tr.splash_r:
                tw.take_damage(tr.dmg)
                if not tw.alive:g._tower_down(tw)
class BuildingTarget(Component):
    def modify_target(self,tr,c,g):
        return [(d,t) for d,t in c if hasattr(t,'ttype') or getattr(t,'is_building',False)]
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
class DeathNova(Component):
    def __init__(self,slow_pct,slow_dur):
        self.slow_pct=slow_pct;self.slow_dur=slow_dur
    def on_death(self,tr,g):
        dd=getattr(tr,'death_dmg',0)
        dr=getattr(tr,'death_splash_r',2.0)
        if dd<=0:return
        opp=g._opp(tr.team)
        sv=1.0-self.slow_pct/100.0
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=dr:
                e.take_damage(dd)
                if hasattr(e,'statuses'):e.statuses.append(Status('slow',self.slow_dur,sv))
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
        sld=getattr(tr,'slow_dur',0)
        slv=getattr(tr,'slow_val',1.0)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=r:
                e.take_damage(dmg)
                if sd>0 and hasattr(e,'statuses'):e.statuses.append(Status('stun',sd))
                if sld>0 and hasattr(e,'statuses'):e.statuses.append(Status('slow',sld,slv))
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
    def _reset(self,tr):
        self.cur_tgt=None;self.elapsed=0;tr.dmg=self.stages[0]
    def on_tick(self,tr,g):
        stn=any(s.kind=='stun' for s in getattr(tr,'statuses',[]))
        frz=any(s.kind=='freeze' for s in getattr(tr,'statuses',[]))
        if stn or frz:self._reset(tr);return
        tgt=getattr(tr,'tgt',None)
        if tgt is not self.cur_tgt or (self.cur_tgt and not getattr(self.cur_tgt,'alive',True)):
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
class SuicideChain(Component):
    def on_attack(self,tr,tgt,g):
        cc=getattr(tr,'chain_count',9)
        cr=getattr(tr,'chain_range',4.0)
        cs=getattr(tr,'chain_stun',0.5)
        opp=g._opp(tr.team)
        if cs>0 and hasattr(tgt,'statuses'):tgt.statuses.append(Status('stun',cs))
        hit=[tgt];prev=tgt
        for _ in range(cc-1):
            px=prev.cx if hasattr(prev,'cx') else prev.x
            py=prev.cy if hasattr(prev,'cy') else prev.y
            best=None;bd=999
            for e in g.players[opp].troops:
                if not e.alive or e in hit:continue
                d=math.sqrt((e.x-px)**2+(e.y-py)**2)
                if d<=cr and d<bd:bd=d;best=e
            for tw in g.arena.towers:
                if tw.team!=opp or not tw.alive or tw in hit:continue
                d=math.sqrt((tw.cx-px)**2+(tw.cy-py)**2)
                if d<=cr and d<bd:bd=d;best=tw
            if not best:break
            best.take_damage(tr.dmg)
            if cs>0 and hasattr(best,'statuses'):best.statuses.append(Status('stun',cs))
            if hasattr(best,'ttype') and not best.alive:g._tower_down(best)
            hit.append(best);prev=best
        tr.is_suicide=True
class ChainAttack(Component):
    def on_attack(self,tr,tgt,g):
        cc=getattr(tr,'chain_count',3)
        cr=getattr(tr,'chain_range',4.0)
        cs=getattr(tr,'chain_stun',0.5)
        opp=g._opp(tr.team)
        if cs>0 and hasattr(tgt,'statuses'):tgt.statuses.append(Status('stun',cs))
        hit=[tgt];prev=tgt
        for _ in range(cc-1):
            px=prev.cx if hasattr(prev,'cx') else prev.x
            py=prev.cy if hasattr(prev,'cy') else prev.y
            best=None;bd=999
            for e in g.players[opp].troops:
                if not e.alive or e in hit:continue
                d=math.sqrt((e.x-px)**2+(e.y-py)**2)
                if d<=cr and d<bd:bd=d;best=e
            for tw in g.arena.towers:
                if tw.team!=opp or not tw.alive or tw in hit:continue
                d=math.sqrt((tw.cx-px)**2+(tw.cy-py)**2)
                if d<=cr and d<bd:bd=d;best=tw
            if not best:break
            best.take_damage(tr.dmg)
            if cs>0 and hasattr(best,'statuses'):best.statuses.append(Status('stun',cs))
            if hasattr(best,'ttype') and not best.alive:g._tower_down(best)
            hit.append(best);prev=best
class HealBurst(Component):
    def __init__(self,heal,radius):
        self.heal=heal;self.radius=radius
    def on_attack(self,tr,g_unused,g2=None):pass
    def on_death(self,tr,g):
        for ally in g.players[tr.team].troops:
            if not ally.alive or ally is tr:continue
            d=math.sqrt((ally.x-tr.x)**2+(ally.y-tr.y)**2)
            if d<=self.radius:
                ally.hp=min(ally.max_hp,ally.hp+self.heal)
class ZapPack(Component):
    def __init__(self,dmg,rng,stun):
        self.dmg=dmg;self.rng=rng;self.stun=stun
    def on_take_damage(self,tr,attacker,g):
        if not tr.alive or not hasattr(attacker,'alive'):return
        if not attacker.alive:return
        dist=math.sqrt((attacker.x-tr.x)**2+(attacker.y-tr.y)**2)
        if dist<=self.rng:
            attacker.take_damage(self.dmg)
            if self.stun>0 and hasattr(attacker,'statuses'):
                attacker.statuses.append(Status('stun',self.stun))
class HealPulse(Component):
    def __init__(self,heal,radius,pulses):
        self.heal=heal;self.radius=radius;self.pulses=pulses
    def on_attack(self,tr,tgt,g):
        for _ in range(self.pulses):
            for ally in g.players[tr.team].troops:
                if not ally.alive:continue
                d=math.sqrt((ally.x-tr.x)**2+(ally.y-tr.y)**2)
                if d<=self.radius:
                    ally.hp=min(ally.max_hp,ally.hp+self.heal)
class RocketLauncher(Component):
    def __init__(self,dmg,hspd,fhspd,rng_min,rng_max,splash_r):
        self.dmg=dmg;self.hspd=hspd;self.fhspd=fhspd
        self.rng_min=rng_min;self.rng_max=rng_max;self.splash_r=splash_r
        self.cd=fhspd;self.first=True
    def on_tick(self,tr,g):
        frz=any(s.kind=='freeze' for s in getattr(tr,'statuses',[]))
        stn=any(s.kind=='stun' for s in getattr(tr,'statuses',[]))
        if frz or stn:return
        self.cd-=g.DT
        if self.cd>0:return
        opp=g._opp(tr.team)
        best=None;bd=999
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((tr.x-e.x)**2+(tr.y-e.y)**2)
            if self.rng_min<=d<=self.rng_max and d<bd:bd=d;best=e
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tr.x-tw.cx)**2+(tr.y-tw.cy)**2)
            if self.rng_min<=d<=self.rng_max and d<bd:bd=d;best=tw
        if not best:self.cd=0.1;return
        tx,ty=(best.cx,best.cy) if hasattr(best,'cx') else (best.x,best.y)
        best.take_damage(self.dmg)
        if hasattr(best,'ttype') and not best.alive:g._tower_down(best)
        for e in g.players[opp].troops:
            if not e.alive or e is best:continue
            d=math.sqrt((e.x-tx)**2+(e.y-ty)**2)
            if d<=self.splash_r:e.take_damage(self.dmg)
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive or tw is best:continue
            d=math.sqrt((tw.cx-tx)**2+(tw.cy-ty)**2)
            if d<=self.splash_r:
                tw.take_damage(self.dmg)
                if not tw.alive:g._tower_down(tw)
        self.cd=self.hspd
class FormTransform(Component):
    def __init__(self,spirit_cfg):
        self.spirit_cfg=spirit_cfg
    def on_death(self,tr,g):
        from troop import Troop
        t=Troop(tr.team,tr.x,tr.y,dict(self.spirit_cfg,components=list(self.spirit_cfg.get('components',[]))))
        g.players[tr.team].troops.append(t)
class ElixirProd(Component):
    def __init__(self,interval,amount):
        self.interval=interval;self.amount=amount;self.timer=interval
    def on_tick(self,tr,g):
        self.timer-=g.DT
        if self.timer<=0:
            p=g.players[tr.team]
            p.elixir=min(p.max_ex,p.elixir+self.amount)
            self.timer=self.interval
class BanditDash(Component):
    def __init__(self,mn=3.5,mx=6.0,ct=0.8):
        self.mn=mn;self.mx=mx;self.ct=ct
        self.charging=False;self.timer=0;self.osp=None;self.dtgt=None
    def on_tick(self,tr,g):
        tgt=getattr(tr,'tgt',None)
        if not tgt:
            if self.charging and self.osp is not None:tr.spd=self.osp;self.osp=None
            self.charging=False;return
        tx=tgt.cx if hasattr(tgt,'cx') else tgt.x
        ty=tgt.cy if hasattr(tgt,'cy') else tgt.y
        d=math.sqrt((tr.x-tx)**2+(tr.y-ty)**2)
        if self.mn<=d<=self.mx and not self.charging:
            self.charging=True;self.timer=self.ct;self.osp=tr.spd;tr.spd=0;self.dtgt=tgt
        if self.charging:
            self.timer-=g.DT
            if self.timer<=0:
                self.charging=False
                if self.osp is not None:tr.spd=self.osp;self.osp=None
                if self.dtgt and getattr(self.dtgt,'alive',True):
                    dd=getattr(tr,'dash_dmg',tr.dmg*2)
                    self.dtgt.take_damage(dd)
                    if hasattr(self.dtgt,'ttype') and not self.dtgt.alive:g._tower_down(self.dtgt)
                    if hasattr(self.dtgt,'cx'):tr.x=self.dtgt.cx;tr.y=self.dtgt.cy
                    else:tr.x=self.dtgt.x;tr.y=self.dtgt.y
                self.dtgt=None
class SoulCollect(Component):
    def __init__(self,cap=10):
        self.cap=cap;self.souls=0;self._prev=set()
    def on_tick(self,tr,g):
        opp=g._opp(tr.team)
        alive=set(id(e) for e in g.players[opp].troops if e.alive)
        died=self._prev-alive
        self.souls=min(self.cap,self.souls+len(died))
        self._prev=alive
class MonkCombo(Component):
    def __init__(self,cycle=3,kb=1.8):
        self.cycle=cycle;self.kb=kb;self.cnt=0
    def on_attack(self,tr,tgt,g):
        self.cnt+=1
        if self.cnt>=self.cycle:
            self.cnt=0
            if hasattr(tgt,'x') and hasattr(tgt,'y'):
                dx=tgt.x-tr.x;dy=tgt.y-tr.y
                d=math.sqrt(dx*dx+dy*dy)
                if d>0:tgt.x+=dx/d*self.kb;tgt.y+=dy/d*self.kb
class LPRamp(Component):
    def __init__(self,stages,per=3):
        self.stages=stages;self.per=per;self.hits=0;self.si=0
        self.px=None;self.py=None
    def on_tick(self,tr,g):
        if self.px is not None:
            dx=tr.x-self.px;dy=tr.y-self.py
            if dx*dx+dy*dy>0.01:
                self.hits=0;self.si=0;tr.hspd=self.stages[0]
        self.px=tr.x;self.py=tr.y
    def on_attack(self,tr,tgt,g):
        self.hits+=1
        if self.hits>=self.per and self.si<len(self.stages)-1:
            self.si+=1;self.hits=0;tr.hspd=self.stages[self.si]
class Ability:
    CAST_TIME=1.0
    def __init__(self,cost,cd,delay=1.0):
        self.cost=cost;self.max_cd=cd;self.cd=delay;self.active=False;self.dur=0
        self.casting=False;self.cast_timer=0
    def can_use(self):return self.cd<=0 and not self.active and not self.casting and not getattr(self,'_pend',False)
    def begin_cast(self,tr,g):
        self.casting=True;self.cast_timer=self.CAST_TIME;self._cast_tr=tr
    def activate(self,tr,g):pass
    def tick(self,dt,tr,g):
        if self.casting:
            self.cast_timer-=dt
            if self.cast_timer<=0:
                self.casting=False;self.activate(self._cast_tr,g)
            return
        if not self.active:self.cd=max(0,self.cd-dt)
class DashingDash(Ability):
    def __init__(self,dd,mxd=10,sr=5.5,cost=1,cd=8.0):
        super().__init__(cost,cd);self.dd=dd;self.mxd=mxd;self.sr=sr
        self.dashing=False;self.dashes=0;self.hit=set()
    def activate(self,tr,g):
        self.active=True;self.dashing=True;self.dashes=0;self.hit=set()
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        if not self.dashing:self.active=False;self.cd=self.max_cd;return
        opp=g._opp(tr.team)
        best=None;bd=999
        for e in g.players[opp].troops:
            if not e.alive or id(e) in self.hit:continue
            d=math.sqrt((tr.x-e.x)**2+(tr.y-e.y)**2)
            if d<=self.sr and d<bd:bd=d;best=e
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive or id(tw) in self.hit:continue
            d=math.sqrt((tr.x-tw.cx)**2+(tr.y-tw.cy)**2)
            if d<=self.sr and d<bd:bd=d;best=tw
        if not best:self.dashing=False;return
        best.take_damage(self.dd)
        self.hit.add(id(best))
        if hasattr(best,'cx'):tr.x=best.cx;tr.y=best.cy
        else:tr.x=best.x;tr.y=best.y
        if hasattr(best,'ttype'):
            if not best.alive:g._tower_down(best)
            self.dashing=False;return
        self.dashes+=1
        if self.dashes>=self.mxd:self.dashing=False
class SoulSummoning(Ability):
    def __init__(self,scfg,radius=3.5,cost=2,cd=20.0):
        super().__init__(cost,cd);self.scfg=scfg;self.radius=radius
        self.q=0;self.si=0.25;self.timer=0
    def activate(self,tr,g):
        sc=None
        for c in tr.components:
            if isinstance(c,SoulCollect):sc=c;break
        souls=sc.souls if sc else 0
        self.q=6+souls
        if sc:sc.souls=0
        self.active=True;self.timer=0
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        if self.q<=0:self.active=False;self.cd=self.max_cd;return
        self.timer-=dt
        if self.timer<=0:
            from troop import Troop
            ox=random.uniform(-self.radius,self.radius)
            oy=random.uniform(-self.radius,self.radius)
            t=Troop(tr.team,tr.x+ox,tr.y+oy,dict(self.scfg,components=[]))
            g.players[tr.team].troops.append(t)
            self.q-=1;self.timer=self.si
class GetawayGrenade(Ability):
    def __init__(self,dist=6.0,invis_dur=1.0,cost=1,cd=3.0,uses=2):
        super().__init__(cost,cd);self.dist=dist;self.invis_dur=invis_dur
        self.max_uses=uses;self.uses_left=uses
    def can_use(self):return self.cd<=0 and not self.active and self.uses_left>0 and not getattr(self,'_pend',False)
    def activate(self,tr,g):
        self.active=True;self.dur=self.invis_dur;self.uses_left-=1
        tr.statuses.append(Status('invisible',self.invis_dur))
        if tr.team=='blue':tr.y=max(0,tr.y-self.dist)
        else:tr.y=min(31,tr.y+self.dist)
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:self.active=False;self.cd=self.max_cd
class CloakingCape(Ability):
    def __init__(self,dur=3.5,spd_val=45,atk_boost=1.8,cost=1,cd=17.0):
        super().__init__(cost,cd);self.max_dur=dur;self.spd_val=spd_val
        self.atk_boost=atk_boost;self.orig_hspd=None;self.orig_spd=None
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur
        self.orig_hspd=tr.hspd;self.orig_spd=tr.spd
        tr.hspd=tr.hspd/(1+self.atk_boost)
        tr.spd=self.spd_val/60.0
        tr.statuses.append(Status('invisible',self.max_dur))
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:
            self.active=False;self.cd=self.max_cd
            if self.orig_hspd is not None:tr.hspd=self.orig_hspd;self.orig_hspd=None
            if self.orig_spd is not None:tr.spd=self.orig_spd;self.orig_spd=None
class ExplosiveEscape(Ability):
    def __init__(self,bomb_dmg,bomb_r=3.0,kb=1.8,cost=1,cd=13.0):
        super().__init__(cost,cd);self.bomb_dmg=bomb_dmg;self.bomb_r=bomb_r;self.kb=kb
    def activate(self,tr,g):
        ox,oy=tr.x,tr.y
        tr.x=g.arena.W-1-tr.x
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-ox)**2+(e.y-oy)**2)
            if d<=self.bomb_r:e.take_damage(self.bomb_dmg)
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=math.sqrt((tw.cx-ox)**2+(tw.cy-oy)**2)
            if d<=self.bomb_r:
                tw.take_damage(self.bomb_dmg)
                if not tw.alive:g._tower_down(tw)
        for c in tr.components:
            if hasattr(c,'_reset'):c._reset(tr)
        self.cd=self.max_cd
class LightningLink(Ability):
    def __init__(self,tick_dmg,tick_ct=0,radius=2.0,dur=4.0,ti=0.5,cost=2,cd=17.0):
        super().__init__(cost,cd);self.tick_dmg=tick_dmg;self.tick_ct=tick_ct
        self.radius=radius;self.max_dur=dur;self.ti=ti;self.timer=0
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur;self.timer=0
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt;self.timer-=dt
        if self.timer<=0:
            opp=g._opp(tr.team)
            for e in g.players[opp].troops:
                if not e.alive:continue
                d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
                if d<=self.radius:e.take_damage(self.tick_dmg)
            for tw in g.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=math.sqrt((tw.cx-tr.x)**2+(tw.cy-tr.y)**2)
                if d<=self.radius:
                    td=self.tick_ct if self.tick_ct>0 else self.tick_dmg
                    tw.take_damage(td)
                    if not tw.alive:g._tower_down(tw)
            self.timer=self.ti
        if self.dur<=0:self.active=False;self.cd=self.max_cd
class RoyalRescue(Ability):
    def __init__(self,gcfg,cdmg,kb=2.5,cost=3,cd=30.0):
        super().__init__(cost,cd);self.gcfg=gcfg;self.cdmg=cdmg;self.kb=kb
    def activate(self,tr,g):
        from troop import Troop
        gt=Troop(tr.team,tr.x,tr.y,dict(self.gcfg,components=list(self.gcfg.get('components',[]))))
        g.players[tr.team].troops.append(gt)
        opp=g._opp(tr.team)
        best=None;bd=999
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<bd:bd=d;best=e
        if best:
            best.take_damage(self.cdmg)
            dx=best.x-tr.x;dy=best.y-tr.y
            d=math.sqrt(dx*dx+dy*dy)
            if d>0:best.x+=dx/d*self.kb;best.y+=dy/d*self.kb
        self.cd=self.max_cd
class PensiveProtection(Ability):
    def __init__(self,reduction=0.65,dur=4.0,cost=1,cd=17.0):
        super().__init__(cost,cd);self.reduction=reduction;self.max_dur=dur
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur;tr._dmg_reduction=self.reduction
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:
            self.active=False;self.cd=self.max_cd;tr._dmg_reduction=0
class TriumphantTaunt(Ability):
    def __init__(self,radius=6.5,shp=769,dur=5.0,cost=2,cd=25.0):
        super().__init__(cost,cd);self.radius=radius;self.shp=shp;self.max_dur=dur
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur
        tr.shield_hp=self.shp;tr.max_shield_hp=self.shp
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=self.radius:e._taunt_target=tr
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            if getattr(tw,'troop',None):tw.troop._taunt_override=tr
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:
            self.active=False;self.cd=self.max_cd
            opp=g._opp(tr.team)
            for e in g.players[opp].troops:
                if getattr(e,'_taunt_target',None) is tr:e._taunt_target=None
            for tw in g.arena.towers:
                if getattr(tw,'troop',None) and getattr(tw.troop,'_taunt_override',None) is tr:
                    tw.troop._taunt_override=None
class BannerBrigade(Ability):
    def __init__(self,spawn_cnt=4,banner_dur=7.0,cost=1):
        super().__init__(cost,0,delay=999);self.spawn_cnt=spawn_cnt
        self.banner_dur=banner_dur;self.banner_pos=None;self.banner_timer=0
        self.all_dead=False;self.base_cfg=None;self.uses=1
    def can_use(self):return self.banner_pos is not None and self.banner_timer>0 and self.uses>0 and not getattr(self,'_pend',False)
    def set_base_cfg(self,cfg):self.base_cfg=cfg
    def on_last_death(self,tr,g):
        self.banner_pos=(tr.x,tr.y);self.banner_timer=self.banner_dur
        self.cd=0;self._team=tr.team
    def activate(self,tr,g):
        if not self.banner_pos or not self.base_cfg:return
        from troop import Troop
        tm=getattr(tr,'team',None) or self._team
        bx,by=self.banner_pos
        for _ in range(self.spawn_cnt):
            ox=random.uniform(-1.0,1.0);oy=random.uniform(-1.0,1.0)
            t=Troop(tm,bx+ox,by+oy,dict(self.base_cfg,components=[]))
            g.players[tm].troops.append(t)
        self.banner_pos=None;self.uses-=1
    def tick(self,dt,tr,g):
        if self.banner_pos:
            self.banner_timer-=dt
            if self.banner_timer<=0:self.banner_pos=None
class EvoKnight(Component):
    def __init__(self,red=0.6):self.red=red;self.attacking=False
    def on_tick(self,tr,g):
        self.attacking=getattr(tr,'tgt',None) is not None and tr.cd<=0.01
        tr._dmg_reduction=0 if self.attacking else self.red
    def on_attack(self,tr,tgt,g):tr._dmg_reduction=0
class EvoBomber(Component):
    def __init__(self,bounces=2,br=2.5):self.bounces=bounces;self.br=br
    def on_attack(self,tr,tgt,g):
        opp=g._opp(tr.team)
        prev=tgt;hit={id(tgt)}
        for _ in range(self.bounces):
            px=prev.cx if hasattr(prev,'cx') else prev.x
            py=prev.cy if hasattr(prev,'cy') else prev.y
            best=None;bd=999
            for e in g.players[opp].troops:
                if not e.alive or id(e) in hit:continue
                d=math.sqrt((e.x-px)**2+(e.y-py)**2)
                if d<=self.br and d<bd:bd=d;best=e
            for tw in g.arena.towers:
                if tw.team!=opp or not tw.alive or id(tw) in hit:continue
                d=math.sqrt((tw.cx-px)**2+(tw.cy-py)**2)
                if d<=self.br and d<bd:bd=d;best=tw
            if not best:break
            best.take_damage(tr.dmg)
            if hasattr(best,'ttype') and not best.alive:g._tower_down(best)
            hit.add(id(best));prev=best
class EvoSkeletons(Component):
    def __init__(self,mx=8):self.mx=mx
    def on_attack(self,tr,tgt,g):
        cnt=sum(1 for t in g.players[tr.team].troops if t.alive and t.name==tr.name)
        if cnt>=self.mx:return
        from troop import Troop
        cfg={'hp':tr.max_hp,'dmg':tr.dmg,'hspd':tr.hspd,'fhspd':tr.fhspd,
             'spd':tr.spd,'rng':tr.rng,'targets':tr.targets,'transport':tr.transport,
             'atk_type':tr.atk_type,'splash_r':tr.splash_r,'ct_dmg':tr.ct_dmg,
             'components':[],'lvl':tr.lvl,'name':tr.name}
        ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
        g.players[tr.team].troops.append(Troop(tr.team,tr.x+ox,tr.y+oy,cfg))
class EvoBarbarians(Component):
    def __init__(self,hp_m=1.1,aspd=30,mspd=30,dur=3.0):
        self.hp_m=hp_m;self.aspd=aspd/100.0;self.mspd=mspd/100.0;self.dur=dur
    def on_attack(self,tr,tgt,g):
        if not any(s.kind=='evo_boost' for s in tr.statuses):
            tr.statuses.append(Status('evo_boost',self.dur,self.aspd))
class EvoBats(Component):
    def __init__(self,pulses=2,mx_mult=2.0):self.pulses=pulses;self.mx=mx_mult
    def on_attack(self,tr,tgt,g):
        cap=int(tr.max_hp*self.mx)
        heal=tr.dmg//2
        for _ in range(self.pulses):
            tr.hp=min(cap,tr.hp+heal)
        if tr.max_hp<cap:tr.max_hp=cap
class EvoRoyalRecruits(Component):
    def __init__(self,cdmg_m=2.0,dist=2.5):
        self.cdmg_m=cdmg_m;self.dist=dist;self.charged=False;self.moved=0
        self.px=None;self.py=None;self.osp=None
    def on_tick(self,tr,g):
        if tr.shield_hp<=0 and not self.charged and getattr(tr,'max_shield_hp',0)>0:
            self.charged=True;self.moved=0;self.osp=tr.spd;tr.spd*=2
        if self.charged:
            if self.px is not None:
                dx=tr.x-self.px;dy=tr.y-self.py
                self.moved+=math.sqrt(dx*dx+dy*dy)
            self.px=tr.x;self.py=tr.y
    def on_attack(self,tr,tgt,g):
        if not self.charged:return
        extra=int(tr.dmg*(self.cdmg_m-1))
        if extra>0:tgt.take_damage(extra)
        if self.osp is not None:tr.spd=self.osp
        self.charged=False
class EvoRoyalGiant(Component):
    def __init__(self,radius=2.5,kb=1.0):self.radius=radius;self.kb=kb
    def on_attack(self,tr,tgt,g):
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive or e is tgt:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=self.radius:
                e.take_damage(tr.dmg//3)
                dx=e.x-tr.x;dy=e.y-tr.y
                dd=math.sqrt(dx*dx+dy*dy)
                if dd>0:e.x+=dx/dd*self.kb;e.y+=dy/dd*self.kb
class EvoIceSpirit(Component):
    def __init__(self,delay=3.0):self.delay=delay;self.boom_pos=None;self.timer=0
    def on_death(self,tr,g):
        self.boom_pos=(tr.x,tr.y);self.timer=self.delay
        g._evo_ice_pending=getattr(g,'_evo_ice_pending',[])
        g._evo_ice_pending.append(self)
    def tick_pending(self,dt,g,team):
        if not self.boom_pos:return True
        self.timer-=dt
        if self.timer<=0:
            x,y=self.boom_pos
            opp='red' if team=='blue' else 'blue'
            for e in g.players[opp].troops:
                if not e.alive:continue
                d=math.sqrt((e.x-x)**2+(e.y-y)**2)
                if d<=2.5:
                    e.statuses.append(Status('freeze',1.0))
            return True
        return False
class EvoSkelBarrel(Component):
    def __init__(self,drop_pct=0.75,dd_mult=1.64):
        self.drop_pct=drop_pct;self.dd_mult=dd_mult;self.dropped=False
    def on_tick(self,tr,g):
        if self.dropped:return
        if tr.hp<=tr.max_hp*self.drop_pct:
            self.dropped=True
            for c in tr.components:
                if isinstance(c,DeathSpawn):
                    from troop import Troop
                    for i in range(c.count):
                        ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
                        t=Troop(tr.team,tr.x+ox,tr.y+oy,dict(c.cfg,components=list(c.cfg.get('components',[]))))
                        g.players[tr.team].troops.append(t)
                    break
class EvoFirecracker(Component):
    def __init__(self,sparks=5,slow_pct=15):self.sparks=sparks;self.slow_pct=slow_pct
    def on_attack(self,tr,tgt,g):
        opp=g._opp(tr.team)
        tx=tgt.cx if hasattr(tgt,'cx') else tgt.x
        ty=tgt.cy if hasattr(tgt,'cy') else tgt.y
        dx=tx-tr.x;dy=ty-tr.y
        d=math.sqrt(dx*dx+dy*dy)
        if d<=0:return
        nx,ny=dx/d,dy/d
        sv=1.0-self.slow_pct/100.0
        for i in range(1,self.sparks+1):
            sx=tr.x+nx*i*1.0;sy=tr.y+ny*i*1.0
            for e in g.players[opp].troops:
                if not e.alive:continue
                dd=math.sqrt((e.x-sx)**2+(e.y-sy)**2)
                if dd<=1.0:e.statuses.append(Status('slow',2.0,sv))
class EvoArchers(Component):
    def __init__(self,mn_rng=4.0,mx_rng=6.0,dmg_m=1.5):
        self.mn=mn_rng;self.mx=mx_rng;self.dmg_m=dmg_m
    def on_attack(self,tr,tgt,g):
        tx=tgt.cx if hasattr(tgt,'cx') else tgt.x
        ty=tgt.cy if hasattr(tgt,'cy') else tgt.y
        d=math.sqrt((tr.x-tx)**2+(tr.y-ty)**2)
        if self.mn<=d<=self.mx:
            extra=int(tr.dmg*(self.dmg_m-1))
            tgt.take_damage(extra)
            if hasattr(tgt,'ttype') and not tgt.alive:g._tower_down(tgt)
class EvoValkyrie(Component):
    def __init__(self,radius=5.0,dmg=84,dur=0.5):
        self.radius=radius;self.dmg=dmg;self.dur=dur
    def on_attack(self,tr,tgt,g):
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=self.radius and d>0:
                e.take_damage(self.dmg)
                dx=tr.x-e.x;dy=tr.y-e.y
                dd=math.sqrt(dx*dx+dy*dy)
                if dd>0:
                    pull=min(1.5,dd)*0.5
                    e.x+=dx/dd*pull;e.y+=dy/dd*pull
class EvoMusketeer(Component):
    def __init__(self,ammo=3,rng=30.0,dmg_m=1.8,min_rng=6.0):
        self.ammo=ammo;self.rng=rng;self.dmg_m=dmg_m;self.min_rng=min_rng
        self.orig_rng=None
    def on_tick(self,tr,g):
        if self.ammo>0 and self.orig_rng is None:
            self.orig_rng=tr.rng;tr.rng=self.rng
        elif self.ammo<=0 and self.orig_rng is not None:
            tr.rng=self.orig_rng;self.orig_rng=None
    def on_attack(self,tr,tgt,g):
        if self.ammo<=0:return
        extra=int(tr.dmg*(self.dmg_m-1))
        tgt.take_damage(extra)
        if hasattr(tgt,'ttype') and not tgt.alive:g._tower_down(tgt)
        self.ammo-=1
class EvoDartGoblin(Component):
    def __init__(self,radius=1.5,dur=1.0,tiers=(51,115,307),esc=(1,4,7)):
        self.radius=radius;self.dur=dur;self.tiers=tiers;self.esc=esc
        self.hits=0;self.tier=0
    def on_attack(self,tr,tgt,g):
        self.hits+=1
        for i,th in enumerate(self.esc):
            if self.hits>=th:self.tier=i
        if self.tier<len(self.tiers):
            dps=self.tiers[self.tier]
            dmg=int(dps*self.dur)
            opp=g._opp(tr.team)
            tx=tgt.cx if hasattr(tgt,'cx') else tgt.x
            ty=tgt.cy if hasattr(tgt,'cy') else tgt.y
            for e in g.players[opp].troops:
                if not e.alive:continue
                d=math.sqrt((e.x-tx)**2+(e.y-ty)**2)
                if d<=self.radius:e.take_damage(dmg)
class EvoRoyalHogs(Component):
    def __init__(self,ldmg=84,lr=2.0):
        self.ldmg=ldmg;self.lr=lr;self.flying=True
    def on_tick(self,tr,g):
        if not self.flying:return
        if tr.hp<tr.max_hp:
            self.flying=False;tr.transport='Ground'
            opp=g._opp(tr.team)
            for e in g.players[opp].troops:
                if not e.alive:continue
                d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
                if d<=self.lr:e.take_damage(self.ldmg)
class EvoGoblinCage(Component):
    def __init__(self,pr=3.0):
        self.pr=pr;self.trapped=None;self.trap_timer=0
    def on_tick(self,tr,g):
        if self.trapped:
            if not self.trapped.alive:self.trapped=None;return
            self.trapped.x=tr.x if hasattr(tr,'x') else tr.cx
            self.trapped.y=tr.y if hasattr(tr,'y') else tr.cy
            self.trapped.statuses.append(Status('stun',g.DT+0.01))
            return
        opp=g._opp(tr.team)
        cx=tr.x if hasattr(tr,'x') else tr.cx
        cy=tr.y if hasattr(tr,'y') else tr.cy
        for e in g.players[opp].troops:
            if not e.alive or getattr(e,'transport','Ground')!='Ground':continue
            d=math.sqrt((e.x-cx)**2+(e.y-cy)**2)
            if d<=self.pr:
                self.trapped=e;break
class HeroicHurl(Ability):
    def __init__(self,throw_rng=9.0,stun_dur=2.0,impact_dmg=64,cost=2,cd=14.0):
        super().__init__(cost,cd);self.throw_rng=throw_rng;self.stun_dur=stun_dur
        self.impact_dmg=impact_dmg
    def activate(self,tr,g):
        opp=g._opp(tr.team)
        best=None;bhp=0
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=2.0 and e.max_hp>bhp:bhp=e.max_hp;best=e
        if not best:self.cd=0;return
        if tr.x<9:best.x=min(17,best.x+self.throw_rng)
        else:best.x=max(0,best.x-self.throw_rng)
        best.take_damage(self.impact_dmg)
        best.statuses.append(Status('stun',self.stun_dur))
        self.active=False;self.cd=self.max_cd
class BreakfastBoost(Ability):
    def __init__(self,heal_pct=0.3,cost=1):
        super().__init__(cost,0,delay=999);self.heal_pct=heal_pct
        self.meters=0;self.meter_progress=0;self.cook_time=22.0
        self.cook_timer=0;self.uses=1
    def can_use(self):return self.uses>0 and not self.casting and not getattr(self,'_pend',False)
    def activate(self,tr,g):
        if self.uses<=0:return
        lvls={0:1,1:2,2:3,3:5}.get(min(self.meters,3),1)
        for _ in range(lvls):tr.level_up()
        heal=int(tr.max_hp*self.heal_pct)
        tr.hp=min(tr.max_hp,tr.hp+heal)
        self.uses-=1
    def tick(self,dt,tr,g):
        if self.casting:
            self.cast_timer-=dt
            if self.cast_timer<=0:self.casting=False;self.activate(self._cast_tr,g)
            return
        if tr is None:return
        self.cook_timer+=dt
        if self.cook_timer>=self.cook_time:
            self.cook_timer-=self.cook_time
            if self.meters<3:self.meters+=1
    def on_attack_progress(self):
        self.meter_progress+=0.5
        while self.meter_progress>=1.0 and self.meters<3:
            self.meters+=1;self.meter_progress-=1.0
class TrustyTurret(Ability):
    def __init__(self,turret_cfg=None,cost=3,cd=22.0):
        super().__init__(cost,cd);self.turret_cfg=turret_cfg or {}
    def activate(self,tr,g):
        from building import Building
        dy=3.0 if tr.team=='blue' else -3.0
        cfg=dict(self.turret_cfg)
        cfg['lifetime']=10.0
        bld=Building(tr.team,tr.x,tr.y+dy,cfg)
        g.players[tr.team].troops.append(bld)
        self.active=False;self.cd=self.max_cd
class Snowstorm(Ability):
    def __init__(self,radius=4.0,pulses=3,freeze_dur=1.5,cost=2,cd=17.0):
        super().__init__(cost,cd);self.radius=radius;self.pulses=pulses
        self.freeze_dur=freeze_dur;self.pulse_i=0;self.pulse_timer=0
    def activate(self,tr,g):
        self.active=True;self.pulse_i=0;self.pulse_timer=0
    def tick(self,dt,tr,g):
        if self.casting:
            self.cast_timer-=dt
            if self.cast_timer<=0:self.casting=False;self.activate(self._cast_tr,g)
            return
        if not self.active:super().tick(dt,tr,g);return
        self.pulse_timer-=dt
        if self.pulse_timer>0:return
        if tr is None:self.active=False;self.cd=self.max_cd;return
        opp=g._opp(tr.team)
        is_freeze=(self.pulse_i==self.pulses-1)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=self.radius:
                e.take_damage(tr.dmg)
                if is_freeze:e.statuses.append(Status('freeze',self.freeze_dur))
                else:e.statuses.append(Status('slow',2.0,0.7))
        self.pulse_i+=1;self.pulse_timer=0.5
        if self.pulse_i>=self.pulses:self.active=False;self.cd=self.max_cd
class FieryFlight(Ability):
    def __init__(self,dur=5.0,spd_boost=0.5,tornado_r=4.0,cost=1,cd=20.0):
        super().__init__(cost,cd);self.max_dur=dur;self.spd_boost=spd_boost
        self.tornado_r=tornado_r;self.orig_spd=None;self.orig_transport=None
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur
        self.orig_spd=tr.spd;self.orig_transport=tr.transport
        tr.spd*=(1+self.spd_boost);tr.transport='Air'
    def tick(self,dt,tr,g):
        if self.casting:
            self.cast_timer-=dt
            if self.cast_timer<=0:self.casting=False;self.activate(self._cast_tr,g)
            return
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:
            self.active=False;self.cd=self.max_cd
            if tr and self.orig_spd:tr.spd=self.orig_spd
            if tr and self.orig_transport:tr.transport=self.orig_transport
class WoundingWarp(Ability):
    def __init__(self,bonus_dmg_pct=0.5,cost=2):
        super().__init__(cost,0,delay=999);self.bonus_pct=bonus_dmg_pct;self.uses=1
    def can_use(self):return self.uses>0 and not self.casting and not getattr(self,'_pend',False)
    def activate(self,tr,g):
        if self.uses<=0:return
        opp=g._opp(tr.team)
        best=None;bmhp=999999
        for e in g.players[opp].troops:
            if not e.alive:continue
            if e.max_hp<bmhp:bmhp=e.max_hp;best=e
        if not best:return
        tr.x=best.x;tr.y=best.y
        bonus=int(tr.dmg*self.bonus_pct)
        best.take_damage(tr.dmg+bonus)
        self.uses-=1
class EvoBabyDragon(Component):
    def __init__(self,radius=1.5,ally_boost=0.3,enemy_slow=0.3):
        self.radius=radius;self.ab=ally_boost;self.es=enemy_slow
    def on_tick(self,tr,g):
        for a in g.players[tr.team].troops:
            if a is tr or not a.alive:continue
            d=math.sqrt((a.x-tr.x)**2+(a.y-tr.y)**2)
            if d<=5.0:
                if not any(s.kind=='evo_speed' for s in getattr(a,'statuses',[])):
                    a.statuses.append(Status('evo_speed',0.2,self.ab))
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=5.0:
                if not any(s.kind=='slow' for s in e.statuses):
                    e.statuses.append(Status('slow',0.2,1.0-self.es))
class EvoSkelArmy(Component):
    def __init__(self):self.gerry=None;self.shadows=[]
    def set_gerry(self,g):self.gerry=g
class EvoWitch(Component):
    def __init__(self,heal=109,overheal=1.24):self.heal=heal;self.overheal=overheal
    def on_tick(self,tr,g):
        cap=int(tr.max_hp*self.overheal)
        dead=[t for t in g.players[tr.team].troops if not t.alive and 'keleton' in getattr(t,'name','')]
        for d in dead:
            if tr.hp<cap:tr.hp=min(cap,tr.hp+self.heal)
class EvoPekka(Component):
    def __init__(self,small=160,med=304,large=577,overheal=1.5):
        self.tiers={'s':small,'m':med,'l':large};self.overheal=overheal
    def on_attack(self,tr,tgt,g):
        if getattr(tgt,'alive',True):return
        cap=int(tr.max_hp*self.overheal)
        mhp=getattr(tgt,'max_hp',0)
        if mhp<=500:h=self.tiers['s']
        elif mhp<=1500:h=self.tiers['m']
        else:h=self.tiers['l']
        if tr.hp<cap:tr.hp=min(cap,tr.hp+h)
class EvoGoblinGiant(Component):
    def __init__(self,threshold=0.5,interval=2.2):
        self.threshold=threshold;self.interval=interval;self.timer=0;self.gcfg=None
    def set_gcfg(self,cfg):self.gcfg=cfg
    def on_tick(self,tr,g):
        if tr.hp>tr.max_hp*self.threshold or not self.gcfg:return
        self.timer-=g.DT
        if self.timer<=0:
            from troop import Troop
            ox=random.uniform(-1.0,1.0)
            t=Troop(tr.team,tr.x+ox,tr.y,dict(self.gcfg,components=[]))
            g.players[tr.team].troops.append(t)
            self.timer=self.interval
class EvoHunter(Component):
    def __init__(self,net_dur=3.0,net_cd=5.0):
        self.net_dur=net_dur;self.net_cd=net_cd;self.cd=0;self.first=True
    def on_tick(self,tr,g):
        if self.cd>0:self.cd-=g.DT
        if not self.first and self.cd>0:return
        if self.first or self.cd<=0:
            tgt=getattr(tr,'tgt',None)
            if tgt and hasattr(tgt,'statuses'):
                tgt.statuses.append(Status('stun',self.net_dur))
                self.cd=self.net_cd;self.first=False
class EvoElectroDragon(Component):
    def __init__(self,dmg_reduction=0.33,bounce_r=3.5):
        self.dr=dmg_reduction;self.br=bounce_r
    def on_attack(self,tr,tgt,g):
        opp=g._opp(tr.team)
        hit={id(tgt)};prev=tgt;rd=int(tr.dmg*(1-self.dr))
        for _ in range(20):
            px=prev.cx if hasattr(prev,'cx') else prev.x
            py=prev.cy if hasattr(prev,'cy') else prev.y
            best=None;bd=999
            for e in g.players[opp].troops:
                if not e.alive:continue
                d=math.sqrt((e.x-px)**2+(e.y-py)**2)
                if d<=self.br and d<bd:bd=d;best=e
            for tw in g.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=math.sqrt((tw.cx-px)**2+(tw.cy-py)**2)
                if d<=self.br and d<bd:bd=d;best=tw
            if not best or (id(best) in hit and len(hit)>1):break
            best.take_damage(rd)
            if hasattr(best,'ttype') and not best.alive:g._tower_down(best)
            hit.add(id(best));prev=best
class EvoWallBreakers(Component):
    def __init__(self,runner_cfg=None,cnt=2):
        self.runner_cfg=runner_cfg;self.cnt=cnt
    def on_death(self,tr,g):
        if not self.runner_cfg:return
        from troop import Troop
        for _ in range(self.cnt):
            ox=random.uniform(-0.5,0.5)
            t=Troop(tr.team,tr.x+ox,tr.y,dict(self.runner_cfg,components=[]))
            g.players[tr.team].troops.append(t)
class EvoExecutioner(Component):
    def __init__(self,close_rng=3.5,dmg_m=2.0,kb=1.0):
        self.close_rng=close_rng;self.dmg_m=dmg_m;self.kb=kb
    def on_attack(self,tr,tgt,g):
        tx=tgt.cx if hasattr(tgt,'cx') else tgt.x
        ty=tgt.cy if hasattr(tgt,'cy') else tgt.y
        d=math.sqrt((tr.x-tx)**2+(tr.y-ty)**2)
        if d<=self.close_rng:
            extra=int(tr.dmg*(self.dmg_m-1))
            tgt.take_damage(extra)
            if hasattr(tgt,'x'):
                dx=tgt.x-tr.x;dy=tgt.y-tr.y
                dd=math.sqrt(dx*dx+dy*dy)
                if dd>0:tgt.x+=dx/dd*self.kb;tgt.y+=dy/dd*self.kb
            if hasattr(tgt,'ttype') and not tgt.alive:g._tower_down(tgt)
class EvoGoblinBarrel(Component):
    def __init__(self,decoy_hp_pct=0.4,decoy_dmg_pct=0.73):
        self.dhp=decoy_hp_pct;self.ddmg=decoy_dmg_pct
class EvoGoblinDrill(Component):
    def __init__(self,thresholds=(0.66,0.33),gobs_per=2):
        self.thresholds=list(thresholds);self.gobs=gobs_per;self.triggered=set()
    def on_tick(self,tr,g):
        for i,th in enumerate(self.thresholds):
            if i in self.triggered:continue
            if tr.hp<=tr.max_hp*th:
                self.triggered.add(i)
                from troop import Troop
                for _ in range(self.gobs):
                    ox=random.uniform(-1.0,1.0);oy=random.uniform(-1.0,1.0)
                    cfg={'hp':204,'dmg':128,'hspd':1.1,'fhspd':0.6,'spd':2.0,
                         'rng':0.5,'targets':['Ground'],'transport':'Ground',
                         'atk_type':'single_target','splash_r':0,'ct_dmg':0,
                         'components':[],'lvl':tr.lvl,'name':'Goblin'}
                    t=Troop(tr.team,tr.x+ox,tr.y+oy,cfg)
                    g.players[tr.team].troops.append(t)
class RowdyReroll(Ability):
    def __init__(self,roll_dist=4.0,heal_pct=0.5,roll_dmg=0,cost=1):
        super().__init__(cost,0,delay=999);self.roll_dist=roll_dist
        self.heal_pct=heal_pct;self.roll_dmg=roll_dmg;self.uses=1
    def can_use(self):return self.uses>0 and not self.casting and not getattr(self,'_pend',False)
    def activate(self,tr,g):
        if self.uses<=0:return
        dy=self.roll_dist if tr.team=='blue' else -self.roll_dist
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            ex=e.x;ey=e.y
            if abs(ex-tr.x)<=1.3 and min(tr.y,tr.y+dy)<=ey<=max(tr.y,tr.y+dy):
                e.take_damage(self.roll_dmg if self.roll_dmg else tr.dmg)
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            if abs(tw.cx-tr.x)<=1.3 and min(tr.y,tr.y+dy)<=tw.cy<=max(tr.y,tr.y+dy):
                tw.take_damage(self.roll_dmg if self.roll_dmg else tr.dmg)
                if not tw.alive:g._tower_down(tw)
        tr.y+=dy
        lost=tr.max_hp-tr.hp
        tr.hp=min(tr.max_hp,tr.hp+int(lost*self.heal_pct))
        self.uses-=1
class MKJump(Component):
    def __init__(self,mn=3.5,mx=5.0,splash_r=3.5):
        self.mn=mn;self.mx=mx;self.sr=splash_r
        self.charging=False;self.airborne=False;self.timer=0
        self.osp=None;self.jtgt=None
    def on_tick(self,tr,g):
        if self.airborne:
            frz=any(s.kind=='freeze' for s in getattr(tr,'statuses',[]))
            if frz:
                self.airborne=False
                if self.osp is not None:tr.spd=self.osp;self.osp=None
                self.jtgt=None;return
            tr.statuses=[s for s in tr.statuses if s.kind!='stun']
            self.timer-=g.DT
            if self.timer<=0:
                self.airborne=False
                if self.osp is not None:tr.spd=self.osp;self.osp=None
                if self.jtgt and getattr(self.jtgt,'alive',True):
                    jd=getattr(tr,'jump_dmg',getattr(tr,'spawn_zap_dmg',tr.dmg*2))
                    if hasattr(self.jtgt,'cx'):tr.x=self.jtgt.cx;tr.y=self.jtgt.cy
                    else:tr.x=self.jtgt.x;tr.y=self.jtgt.y
                    opp=g._opp(tr.team)
                    atgts=getattr(tr,'targets',['Ground'])
                    for e in g.players[opp].troops:
                        if not e.alive:continue
                        et=getattr(e,'transport','Ground')
                        if et=='Air' and 'Air' not in atgts:continue
                        dd=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
                        if dd<=self.sr:e.take_damage(jd)
                    for tw in g.arena.towers:
                        if tw.team!=opp or not tw.alive:continue
                        dd=math.sqrt((tw.cx-tr.x)**2+(tw.cy-tr.y)**2)
                        if dd<=self.sr:
                            tw.take_damage(jd)
                            if not tw.alive:g._tower_down(tw)
                self.jtgt=None
            return
        tgt=getattr(tr,'tgt',None)
        if not tgt:
            if self.charging and self.osp is not None:tr.spd=self.osp;self.osp=None
            self.charging=False;return
        tx=tgt.cx if hasattr(tgt,'cx') else tgt.x
        ty=tgt.cy if hasattr(tgt,'cy') else tgt.y
        d=math.sqrt((tr.x-tx)**2+(tr.y-ty)**2)
        if self.mn<=d<=self.mx and not self.charging:
            self.charging=True;self.timer=0.1;self.osp=tr.spd;tr.spd=0;self.jtgt=tgt
        if self.charging:
            self.timer-=g.DT
            if self.timer<=0:
                self.charging=False;self.airborne=True;self.timer=0.1
class EvoMegaKnight(Component):
    def __init__(self,kb=4.0):self.kb=kb
    def on_attack(self,tr,tgt,g):
        if not hasattr(tgt,'x') or not hasattr(tgt,'y'):return
        team=getattr(tgt,'team','red')
        twy=2.5 if team=='blue' else 28.5
        dy=twy-tgt.y
        if abs(dy)>0.1:tgt.y+=dy/abs(dy)*min(self.kb,abs(dy))
class EvoInfernoDragon(Component):
    def __init__(self,retain_sec=9.0,s4_time=20.0,s4_dmg=844):
        self.retain=retain_sec;self.s4_time=s4_time;self.s4_dmg=s4_dmg
        self.idle_timer=0;self.total_beam=0;self.last_tgt=None;self.s4_active=False
    def on_tick(self,tr,g):
        tgt=getattr(tr,'tgt',None)
        if tgt:
            self.idle_timer=0
            if tgt is not self.last_tgt:self.last_tgt=tgt
            self.total_beam+=g.DT
            if self.total_beam>=self.s4_time and not self.s4_active:
                self.s4_active=True;tr.dmg=self.s4_dmg
        else:
            self.idle_timer+=g.DT
            if self.idle_timer>self.retain:
                for c in tr.components:
                    if isinstance(c,RampUp):c._reset(tr)
                self.total_beam=0;self.s4_active=False
    def on_attack(self,tr,tgt,g):
        pass
class EvoRoyalGhost(Component):
    def __init__(self,soul_cnt=2,soul_hp=81,soul_dmg=261,soul_r=1.0):
        self.cnt=soul_cnt;self.shp=soul_hp;self.sdmg=soul_dmg;self.sr=soul_r
        self.was_invis=True
    def on_tick(self,tr,g):
        is_invis=any(s.kind=='invisible' for s in getattr(tr,'statuses',[]))
        if self.was_invis and not is_invis:
            from troop import Troop
            opp=g._opp(tr.team)
            for e in g.players[opp].troops:
                if not e.alive:continue
                d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
                if d<=self.sr:e.take_damage(self.sdmg)
            for _ in range(self.cnt):
                ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
                cfg={'hp':self.shp,'dmg':self.sdmg,'hspd':1.5,'fhspd':0.5,
                     'spd':1.0,'rng':1.2,'targets':['Ground'],'transport':'Ground',
                     'atk_type':'single_target','splash_r':0,'ct_dmg':0,
                     'components':[],'lvl':tr.lvl,'name':'Souldier'}
                g.players[tr.team].troops.append(Troop(tr.team,tr.x+ox,tr.y+oy,cfg))
        self.was_invis=is_invis
class EvoBandit(Component):
    def __init__(self,cd_reduction=0.3):self.cd_red=cd_reduction
    def on_attack(self,tr,tgt,g):
        if not getattr(tgt,'alive',True):
            for c in tr.components:
                if isinstance(c,BanditDash):c.ct*=self.cd_red
class EvoFisherman(Component):
    def __init__(self):self.double=True
class EvoLumberjack(Component):
    def __init__(self,ghost_dur=5.5):
        self.ghost_dur=ghost_dur
    def on_death(self,tr,g):
        from troop import Troop
        cfg={'hp':1,'dmg':tr.dmg,'hspd':tr.hspd,'fhspd':tr.fhspd,
             'spd':tr.spd,'rng':tr.rng,'targets':tr.targets,'transport':'Ground',
             'atk_type':'single_target','splash_r':0,'ct_dmg':0,
             'components':[],'lvl':tr.lvl,'name':'Lumberjack Ghost'}
        ghost=Troop(tr.team,tr.x,tr.y,cfg)
        ghost.max_hp=1;ghost.hp=1
        ghost.statuses.append(Status('invisible',self.ghost_dur))
        g.players[tr.team].troops.append(ghost)
class EvoIceWizard(Component):
    def __init__(self,slow_pct=40,freeze_dur=1.0):
        self.slow_pct=slow_pct;self.freeze_dur=freeze_dur
    def on_death(self,tr,g):
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=3.0:e.statuses.append(Status('freeze',self.freeze_dur))
class TripleThreat(Ability):
    def __init__(self,dash_dist=5.0,decoy_hp=518,triple_rng=15.5,dur=7.0,cost=1,cd=25.0):
        super().__init__(cost,cd);self.dash_dist=dash_dist;self.decoy_hp=decoy_hp
        self.triple_rng=triple_rng;self.max_dur=dur;self.empowered=False
    def activate(self,tr,g):
        dy=-self.dash_dist if tr.team=='blue' else self.dash_dist
        tr.y=max(0,min(31,tr.y+dy))
        from troop import Troop
        dcfg={'hp':self.decoy_hp,'dmg':0,'hspd':99,'fhspd':99,'spd':0,
              'rng':0,'targets':['Ground'],'transport':'Ground',
              'atk_type':'single_target','splash_r':0,'ct_dmg':0,
              'components':[],'lvl':tr.lvl,'name':'Decoy'}
        g.players[tr.team].troops.append(Troop(tr.team,tr.x-dy,tr.y-dy,dcfg))
        self.empowered=True;self.active=True;self.dur=self.max_dur
        self._orig_rng=tr.rng;tr.rng=self.triple_rng
    def tick(self,dt,tr,g):
        if self.casting:
            self.cast_timer-=dt
            if self.cast_timer<=0:self.casting=False;self.activate(self._cast_tr,g)
            return
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:
            self.active=False;self.empowered=False;self.cd=self.max_cd
            if tr and hasattr(self,'_orig_rng'):tr.rng=self._orig_rng
