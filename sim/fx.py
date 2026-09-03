import math
import random
from sim.units import Status,Troop,Building,has
from sim.arena import Arena
class Component:
    def on_tick(self,tr,g):pass
    def on_attack(self,tr,tgt,g):pass
    def on_take_damage(self,tr,d,g):pass
    def on_death(self,tr,g):pass
    def modify_target(self,tr,c,g):return c
def pos(u):return (u.cx,u.cy) if hasattr(u,'cx') else (u.x,u.y)
def enemies(g,team,air=True,towers=True):
    opp=g._opp(team)
    for e in g.players[opp].troops:
        if e.alive and (air or getattr(e,'transport','Ground')!='Air'):yield e
    if towers:
        for tw in g.arena.towers:
            if tw.team==opp and tw.alive:yield tw
def tdist(u,x,y):
    # area effects reach a body when they touch it: the tower footprint or the troop's collision circle, not only the centre
    return u.dist(x,y) if hasattr(u,'ttype') else max(0.0,math.hypot(u.x-x,u.y-y)-getattr(u,'collision_r',0))
def near(g,team,x,y,r,air=True,towers=True):return [e for e in enemies(g,team,air,towers) if tdist(e,x,y)<=r]
def hurt(u,dmg,g):
    u.take_damage(dmg)
    if hasattr(u,'ttype') and not u.alive:g._tower_down(u)
_A=Arena()
def push(u,ox,oy,dist):
    # knockback away from (ox,oy): towers, buildings, mass 10+ (P.E.K.K.A, Goblin Machine) and immune cards ignore it; a hit troop's swing and charge restart
    if dist<=0 or hasattr(u,'ttype') or getattr(u,'is_building',False) or getattr(u,'kb_immune',False) or getattr(u,'mass',4)>=10:return
    dx=u.x-ox;dy=u.y-oy;d=math.hypot(dx,dy)
    if d<=0:return
    gnd=getattr(u,'transport','Ground')!='Air';n=max(1,int(dist*4));x0,y0=u.x,u.y
    for i in range(1,n+1):
        nx=min(max(x0+dx/d*dist*i/n,0.3),_A.W-0.3);ny=min(max(y0+dy/d*dist*i/n,0.3),_A.H-0.3)
        # a ground body stops at the river bank or a tower footprint instead of being thrown onto it
        if gnd and (_A.blocked(int(nx),int(ny),True) or (int(ny) in _A.RIVER and not _A.on_bridge(nx))):break
        u.x,u.y=nx,ny
    if hasattr(u,'statuses'):u.statuses.append(Status('knockback',0.05))
def strip(g,team,x0,y0,x1,y1,hw,air=True,skip=()):
    # enemies whose body overlaps the segment: perpendicular distance within hw plus their own radius
    dx=x1-x0;dy=y1-y0;L=math.hypot(dx,dy)
    if L<=0:return []
    out=[]
    for e in enemies(g,team,air):
        if e in skip:continue
        ex,ey=pos(e);t=((ex-x0)*dx+(ey-y0)*dy)/L
        r=max(e.w,e.h)/2 if hasattr(e,'ttype') else getattr(e,'collision_r',0.5)
        if -r<=t<=L+r and abs((ex-x0)*dy-(ey-y0)*dx)/L<=hw+r:out.append(e)
    return out
def refresh(u,kind,dur,val=0):
    s=next((s for s in u.statuses if s.kind==kind),None)
    if s:s.dur=max(s.dur,dur)
    else:u.statuses.append(Status(kind,dur,val))
class Timer:
    # a delayed callback living in g.spells
    def __init__(self,delay,fn,x=0,y=0,team=''):
        self.t=delay;self.fn=fn;self.x=x;self.y=y;self.team=team;self.active=True;self.name='';self.radius=0
    def tick(self,dt,g):
        self.t-=dt
        if self.t<=0:self.active=False;self.fn(g)
class Zone:
    # a lingering circle that keeps a status on the troops inside it (ally=True for rage trails)
    def __init__(self,team,x,y,r,dur,kind,val,ally=False,name=''):
        self.team=team;self.x=x;self.y=y;self.radius=r;self.dur=dur;self.kind=kind;self.val=val;self.ally=ally;self.name=name;self.active=True
    def tick(self,dt,g):
        self.dur-=dt
        if self.dur<=0:self.active=False;return
        for u in g.players[self.team if self.ally else g._opp(self.team)].troops:
            if u.alive and math.hypot(u.x-self.x,u.y-self.y)<=self.radius:refresh(u,self.kind,2*dt,self.val)
class SplashAttack(Component):
    def on_attack(self,tr,tgt,g):
        opp=g._opp(tr.team)
        if hasattr(tgt,'cx'):tx,ty=tgt.cx,tgt.cy
        else:tx,ty=tgt.x,tgt.y
        sd=getattr(tr,'slow_dur',0)
        sv=getattr(tr,'slow_val',1.0)
        st=getattr(tr,'stun_dur',0)
        atgts=getattr(tr,'targets',['Ground'])
        if sd>0 and hasattr(tgt,'statuses'):
            tgt.statuses.append(Status('slow',sd,sv))
        for e in g.players[opp].troops:
            if not e.alive or e is tgt:continue
            et=getattr(e,'transport','Ground')
            if et=='Air' and 'Air' not in atgts:continue
            if tdist(e,tx,ty)<=tr.splash_r:
                e.take_damage(tr.dmg)
                if sd>0 and hasattr(e,'statuses'):
                    e.statuses.append(Status('slow',sd,sv))
                if st>0 and hasattr(e,'statuses'):
                    e.statuses.append(Status('stun',st))
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive or tw is tgt:continue
            d=tw.dist(tx,ty)
            if d<=tr.splash_r:
                tw.take_damage(tr.dmg)
                if not tw.alive:g._tower_down(tw)
class BuildingTarget(Component):
    def modify_target(self,tr,c,g):
        return [(d,t) for d,t in c if hasattr(t,'ttype') or getattr(t,'is_building',False)]
class RiderAttack(Component):
    def __init__(self,dmg,hspd,rng,slow_pct=0,slow_dur=0,fhspd=None,count=1):
        self.dmg=dmg;self.hspd=hspd;self.rng=rng;self.count=count
        self.slow_pct=slow_pct;self.slow_dur=slow_dur;self.cd=fhspd if fhspd is not None else hspd
    def on_tick(self,tr,g):
        self.cd=max(0,self.cd-g.DT)
        if self.cd>0:return
        opp=g._opp(tr.team)
        best=None;bd=999
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((tr.x-e.x)**2+(tr.y-e.y)**2)
            if d<=self.rng and d<bd:bd=d;best=e
        if best:
            best.take_damage(self.dmg*self.count)
            if self.slow_pct>0 and hasattr(best,'statuses'):
                best.statuses.append(Status('slow',self.slow_dur,1.0-self.slow_pct))
            self.cd=self.hspd
class Recoil(Component):
    def __init__(self,dist):self.dist=dist
    def on_attack(self,tr,tgt,g):
        tx=tgt.cx if hasattr(tgt,'cx') else tgt.x
        ty=tgt.cy if hasattr(tgt,'cy') else tgt.y
        dx=tr.x-tx;dy=tr.y-ty
        d=math.sqrt(dx*dx+dy*dy)
        if d<0.01:return
        tr.x+=dx/d*self.dist;tr.y+=dy/d*self.dist
        tr.x=max(0.3,min(17.7,tr.x));tr.y=max(0.3,min(31.7,tr.y))
class RiverJump(Component):
    # marker: movement treats river tiles as walkable at normal speed instead of detouring to a bridge
    pass
class Charge(Component):
    # the charged swing's load time is level independent and not in cards.json
    def __init__(self,dist,fhspd=0.4):
        self.dist=dist;self.fhspd=fhspd;self.moved=0;self.charged=False
        self.px=None;self.py=None;self.orig_spd=None
    def on_tick(self,tr,g):
        if self.px is not None:
            dx=tr.x-self.px;dy=tr.y-self.py
            self.moved+=math.sqrt(dx*dx+dy*dy)
        self.px=tr.x;self.py=tr.y
        if any(s.kind in ('stun','freeze','knockback') for s in getattr(tr,'statuses',[])):
            if self.charged and self.orig_spd is not None:
                tr.spd=self.orig_spd
            if self.charged and hasattr(self,'_ofh'):tr.fhspd=self._ofh
            self.charged=False;self.moved=0;return
        if not self.charged and self.moved>=self.dist:
            self.charged=True;self.orig_spd=tr.spd;tr.spd*=2
            self._ofh=getattr(tr,'fhspd',tr.hspd);tr.fhspd=self.fhspd
    def on_attack(self,tr,tgt,g):
        if not self.charged:return
        extra=getattr(tr,'charge_dmg',tr.dmg*2)-tr.dmg
        if extra>0:
            tgt.take_damage(extra)
            if hasattr(tgt,'ttype') and not tgt.alive:g._tower_down(tgt)
        if self.orig_spd is not None:tr.spd=self.orig_spd
        if hasattr(self,'_ofh'):tr.fhspd=self._ofh
        self.charged=False;self.moved=0
class SpawnTimer(Component):
    def __init__(self,cfg,interval,count,first_delay,pattern=''):
        self.cfg=cfg;self.interval=interval;self.count=count
        self.timer=first_delay;self.pattern=pattern
    def on_tick(self,tr,g):
        if has(tr,'burrowed'):return
        self.timer-=g.DT
        if self.timer<=0:
            tgt=getattr(tr,'tgt',None)
            dx,dy=0,0
            if tgt:
                tx=tgt.cx if hasattr(tgt,'cx') else tgt.x
                ty=tgt.cy if hasattr(tgt,'cy') else tgt.y
                dx=tx-tr.x;dy=ty-tr.y
                ds=math.sqrt(dx*dx+dy*dy)
                if ds>0:dx/=ds;dy/=ds
            for i in range(self.count):
                ox=random.uniform(-1.0,1.0);oy=random.uniform(-1.0,1.0)
                ox+=dx*2.0;oy+=dy*2.0
                t=Troop(tr.team,tr.x+ox,tr.y+oy,dict(self.cfg,components=list(self.cfg.get('components',[]))))
                t._spawner_id=id(tr)
                g.players[tr.team].troops.append(t)
            self.timer=self.interval
class DeathDamage(Component):
    def __init__(self,kb=0):self.kb=kb
    def on_death(self,tr,g):
        dd=getattr(tr,'death_dmg',0)
        if dd<=0:return
        for e in near(g,tr.team,tr.x,tr.y,getattr(tr,'death_splash_r',0)):hurt(e,dd,g);push(e,tr.x,tr.y,self.kb)
class DeathNova(Component):
    def __init__(self,slow_pct,slow_dur):
        self.slow_pct=slow_pct;self.slow_dur=slow_dur
    def on_death(self,tr,g):
        dd=getattr(tr,'death_dmg',0)
        dr=getattr(tr,'death_splash_r',0)
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
            d=tw.dist(tr.x,tr.y)
            if d<=dr:
                tw.take_damage(dd)
                if not tw.alive:g._tower_down(tw)
class DeathSpawn(Component):
    def __init__(self,cfg,count):
        self.cfg=cfg;self.count=count
    def on_death(self,tr,g):
        for i in range(self.count):
            ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
            t=Troop(tr.team,tr.x+ox,tr.y+oy,dict(self.cfg,components=list(self.cfg.get('components',[]))))
            g.players[tr.team].troops.append(t)
class SpawnZap(Component):
    def __init__(self,kb=0):
        self.fired=False;self.kb=kb
    def on_tick(self,tr,g):
        if self.fired or has(tr,'burrowed'):return
        self.fired=True;self.fire(tr,g)
    def fire(self,tr,g):
        # spawn_zap_ct None means towers take the troop damage, 0 means they are not hit at all
        dmg=getattr(tr,'spawn_zap_dmg',0);ct=getattr(tr,'spawn_zap_ct',None);sd=getattr(tr,'stun_dur',0);sld=getattr(tr,'slow_dur',0)
        for e in near(g,tr.team,tr.x,tr.y,getattr(tr,'spawn_zap_r',0),towers=ct!=0):
            hurt(e,ct if hasattr(e,'ttype') and ct is not None else dmg,g);push(e,tr.x,tr.y,self.kb)
            if hasattr(e,'ttype'):continue
            if sd>0:e.statuses.append(Status('stun',sd))
            if sld>0:e.statuses.append(Status('slow',sld,getattr(tr,'slow_val',1.0)))
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
        sd=getattr(tr,'stun_dur',0)
        if hasattr(tgt,'statuses'):tgt.statuses.append(Status('stun',sd))
        opp=g._opp(tr.team)
        cands=[]
        for e in g.players[opp].troops:
            if not e.alive or e is tgt:continue
            d=math.sqrt((tr.x-e.x)**2+(tr.y-e.y)**2)
            if d<=tr.rng:cands.append((d,e))
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive or tw is tgt:continue
            d=tw.dist(tr.x,tr.y)
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
def bounce(g,opp,prev,r,skip):
    # the bolt jumps to the nearest body within r of the last one hit, measured between centres (a crown tower does not chain into the king tower)
    px,py=pos(prev);best=None;bd=r
    for e in g.players[opp].troops:
        if not e.alive or e in skip:continue
        d=math.hypot(e.x-px,e.y-py)
        if d<=bd:bd=d;best=e
    for tw in g.arena.towers:
        if tw.team!=opp or not tw.alive or tw in skip:continue
        d=math.hypot(tw.cx-px,tw.cy-py)
        if d<=bd:bd=d;best=tw
    return best
def chain(tr,tgt,g):
    cc=getattr(tr,'chain_count',1);cr=getattr(tr,'chain_range',0);cs=getattr(tr,'chain_stun',0)
    opp=g._opp(tr.team)
    if cs>0 and hasattr(tgt,'statuses'):tgt.statuses.append(Status('stun',cs))
    hit=[tgt]
    for _ in range(cc-1):
        best=bounce(g,opp,hit[-1],cr,hit)
        if not best:break
        best.take_damage(tr.dmg)
        if cs>0 and hasattr(best,'statuses'):best.statuses.append(Status('stun',cs))
        if hasattr(best,'ttype') and not best.alive:g._tower_down(best)
        hit.append(best)
    tr.chain_hit=hit
class SuicideChain(Component):
    def on_attack(self,tr,tgt,g):
        chain(tr,tgt,g);tr.is_suicide=True
class ChainAttack(Component):
    def on_attack(self,tr,tgt,g):chain(tr,tgt,g)
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
            d=tw.dist(tr.x,tr.y)
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
            d=tw.dist(tx,ty)
            if d<=self.splash_r:
                tw.take_damage(self.dmg)
                if not tw.alive:g._tower_down(tw)
        self.cd=self.hspd
class FormTransform(Component):
    def __init__(self,spirit_cfg):
        self.spirit_cfg=spirit_cfg
    def on_death(self,tr,g):
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
    def __init__(self,mn,mx,ct):
        self.mn=mn;self.mx=mx;self.ct=ct
        self.charging=False;self.timer=0;self.osp=None;self.dtgt=None
    def on_tick(self,tr,g):
        tgt=getattr(tr,'tgt',None)
        if not tgt:
            if self.charging and self.osp is not None:tr.spd=self.osp;self.osp=None
            self.charging=False;return
        if self.mn<=g._dist(tr,tgt)<=self.mx and not self.charging:
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
    def __init__(self,cap):
        self.cap=cap;self.souls=0;self._prev=set()
    def on_tick(self,tr,g):
        opp=g._opp(tr.team)
        alive=set(id(e) for e in g.players[opp].troops if e.alive)
        died=self._prev-alive
        self.souls=min(self.cap,self.souls+len(died))
        self._prev=alive
class MonkCombo(Component):
    def __init__(self,cycle,kb):
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
    def __init__(self,stages,per):
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
class Stealth(Component):
    def __init__(self,after):self.after=after;self.idle=0
    def on_tick(self,tr,g):
        self.idle+=g.DT
        if self.idle>=self.after and not any(s.kind=='invisible' for s in tr.statuses):tr.statuses.append(Status('invisible',g.DT*2))
        elif self.idle>=self.after:
            for s in tr.statuses:
                if s.kind=='invisible':s.dur=g.DT*2
    def on_attack(self,tr,tgt,g):
        self.idle=0;tr.statuses=[s for s in tr.statuses if s.kind!='invisible']
class Knockback(Component):
    def __init__(self,dist):self.dist=dist
    def on_attack(self,tr,tgt,g):
        if hasattr(tgt,'x'):push(tgt,tr.x,tr.y,self.dist)
class Ability:
    CAST_TIME=1.0
    def __init__(self,cost,cd,delay=1.0):
        self.cost=cost;self.max_cd=cd;self.cd=delay;self.active=False;self.dur=0
        self.casting=False;self.cast_timer=0;self.uses=None
    def can_use(self):return self.cd<=0 and not self.active and not self.casting and not getattr(self,'_pend',False) and (self.uses is None or self.uses>0)
    def begin_cast(self,tr,g):
        self.casting=True;self.cast_timer=self.CAST_TIME;self._cast_tr=tr
        if self.uses is not None:self.uses-=1
    def activate(self,tr,g):pass
    def tick(self,dt,tr,g):
        if self.casting:
            self.cast_timer-=dt
            if self.cast_timer<=0:
                self.casting=False;self.activate(self._cast_tr,g)
            return
        if not self.active:self.cd=max(0,self.cd-dt)
class DashingDash(Ability):
    def __init__(self,dd,mxd,sr,cost,cd):
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
            d=tw.dist(tr.x,tr.y)
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
    def __init__(self,scfg,radius,cost,cd,base,interval):
        super().__init__(cost,cd);self.scfg=scfg;self.radius=radius
        self.base=base;self.q=0;self.si=interval;self.timer=0
    def activate(self,tr,g):
        sc=None
        for c in tr.components:
            if isinstance(c,SoulCollect):sc=c;break
        souls=sc.souls if sc else 0
        self.q=self.base+souls
        if sc:sc.souls=0
        self.active=True;self.timer=0
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        if self.q<=0:self.active=False;self.cd=self.max_cd;return
        self.timer-=dt
        if self.timer<=0:
            ox=random.uniform(-self.radius,self.radius)
            oy=random.uniform(-self.radius,self.radius)
            t=Troop(tr.team,tr.x+ox,tr.y+oy,dict(self.scfg,components=[]))
            g.players[tr.team].troops.append(t)
            self.q-=1;self.timer=self.si
class GetawayGrenade(Ability):
    def __init__(self,dist,invis_dur,cost,cd,uses):
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
    def __init__(self,dur,spd,atk_boost,cost,cd):
        super().__init__(cost,cd);self.max_dur=dur;self.spd=spd
        self.atk_boost=atk_boost;self.orig_hspd=None;self.orig_spd=None
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur
        self.orig_hspd=tr.hspd;self.orig_spd=tr.spd
        tr.hspd=tr.hspd/(1+self.atk_boost)
        tr.spd=self.spd
        tr.statuses.append(Status('invisible',self.max_dur))
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:
            self.active=False;self.cd=self.max_cd
            if self.orig_hspd is not None:tr.hspd=self.orig_hspd;self.orig_hspd=None
            if self.orig_spd is not None:tr.spd=self.orig_spd;self.orig_spd=None
class ExplosiveEscape(Ability):
    def __init__(self,bomb_dmg,bomb_r,kb,cost,cd):
        super().__init__(cost,cd);self.bomb_dmg=bomb_dmg;self.bomb_r=bomb_r;self.kb=kb
    def activate(self,tr,g):
        ox,oy=tr.x,tr.y
        tr.x=g.arena.W-tr.x
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-ox)**2+(e.y-oy)**2)
            if d<=self.bomb_r:e.take_damage(self.bomb_dmg)
        for tw in g.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            d=tw.dist(ox,oy)
            if d<=self.bomb_r:
                tw.take_damage(self.bomb_dmg)
                if not tw.alive:g._tower_down(tw)
        for c in tr.components:
            if hasattr(c,'_reset'):c._reset(tr)
        self.cd=self.max_cd
class LightningLink(Ability):
    def __init__(self,tick_dmg,tick_ct,radius,dur,ti,cost,cd):
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
                d=tw.dist(tr.x,tr.y)
                if d<=self.radius:
                    td=self.tick_ct if self.tick_ct>0 else self.tick_dmg
                    tw.take_damage(td)
                    if not tw.alive:g._tower_down(tw)
            self.timer=self.ti
        if self.dur<=0:self.active=False;self.cd=self.max_cd
class RoyalRescue(Ability):
    # the Guardienne appears at the prince and dashes rng tiles to the nearest ground troop, dealing the charge damage and knocking it back
    def __init__(self,gcfg,cdmg,kb,rng,cost,cd):
        super().__init__(cost,cd);self.gcfg=gcfg;self.cdmg=cdmg;self.kb=kb;self.rng=rng
    def activate(self,tr,g):
        gt=Troop(tr.team,tr.x,tr.y,dict(self.gcfg,components=list(self.gcfg.get('components',[]))))
        g.players[tr.team].troops.append(gt)
        c=[(math.hypot(e.x-tr.x,e.y-tr.y),id(e),e) for e in enemies(g,tr.team,air=False,towers=False)]
        c=[x for x in c if x[0]<=self.rng]
        if c:
            best=min(c)[2];gt.x,gt.y=best.x,best.y;best.take_damage(self.cdmg);push(best,tr.x,tr.y,self.kb)
        self.cd=self.max_cd
class PensiveProtection(Ability):
    def __init__(self,reduction,dur,cost,cd):
        super().__init__(cost,cd);self.reduction=reduction;self.max_dur=dur
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur;tr._dmg_reduction=self.reduction
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:
            self.active=False;self.cd=self.max_cd;tr._dmg_reduction=0
class TriumphantTaunt(Ability):
    def __init__(self,radius,shp,dur,cost,cd):
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
    def __init__(self,spawn_cnt,banner_dur,cost):
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
    def __init__(self,red):self.red=red;self.attacking=False
    def on_tick(self,tr,g):
        self.attacking=getattr(tr,'tgt',None) is not None and tr.cd<=0.01
        tr._dmg_reduction=0 if self.attacking else self.red
    def on_attack(self,tr,tgt,g):tr._dmg_reduction=0
class EvoBomber(Component):
    def __init__(self,bounces,br):self.bounces=bounces;self.br=br
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
                d=tw.dist(px,py)
                if d<=self.br and d<bd:bd=d;best=tw
            if not best:break
            best.take_damage(tr.dmg)
            if hasattr(best,'ttype') and not best.alive:g._tower_down(best)
            hit.add(id(best));prev=best
class EvoSkeletons(Component):
    def __init__(self,mx):self.mx=mx
    def on_attack(self,tr,tgt,g):
        cnt=sum(1 for t in g.players[tr.team].troops if t.alive and t.name==tr.name)
        if cnt>=self.mx:return
        cfg={'hp':tr.max_hp,'dmg':tr.dmg,'hspd':tr.hspd,'fhspd':tr.fhspd,
             'spd':tr.spd,'rng':tr.rng,'targets':tr.targets,'transport':tr.transport,
             'atk_type':tr.atk_type,'splash_r':tr.splash_r,'ct_dmg':tr.ct_dmg,
             'components':[],'lvl':tr.lvl,'name':tr.name}
        ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
        g.players[tr.team].troops.append(Troop(tr.team,tr.x+ox,tr.y+oy,cfg))
class EvoBarbarians(Component):
    def __init__(self,aspd,mspd,dur):
        self.aspd=aspd;self.mspd=mspd;self.dur=dur
    def on_attack(self,tr,tgt,g):
        if not any(s.kind=='evo_boost' for s in tr.statuses):
            tr.statuses.append(Status('evo_boost',self.dur,self.aspd))
class EvoBats(Component):
    def __init__(self,heal,cap):self.heal=heal;self.cap=cap
    def on_attack(self,tr,tgt,g):
        tr.hp=min(self.cap,tr.hp+self.heal)
        if tr.max_hp<self.cap:tr.max_hp=self.cap
class EvoRoyalRecruits(Component):
    def __init__(self,cdmg,dist):
        self.cdmg=cdmg;self.dist=dist;self.charged=False;self.moved=0
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
        extra=self.cdmg-tr.dmg
        if extra>0:tgt.take_damage(extra)
        if self.osp is not None:tr.spd=self.osp
        self.charged=False
class EvoRoyalGiant(Component):
    def __init__(self,radius,kb,dmg):self.radius=radius;self.kb=kb;self.dmg=dmg
    def on_attack(self,tr,tgt,g):
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive or e is tgt:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=self.radius:
                e.take_damage(self.dmg)
                dx=e.x-tr.x;dy=e.y-tr.y
                dd=math.sqrt(dx*dx+dy*dy)
                if dd>0:e.x+=dx/dd*self.kb;e.y+=dy/dd*self.kb
class EvoIceSpirit(Component):
    def __init__(self,delay,radius,freeze,dmg):
        self.delay=delay;self.radius=radius;self.freeze=freeze;self.dmg=dmg;self.boom_pos=None;self.timer=0
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
                if d<=self.radius:
                    e.take_damage(self.dmg)
                    e.statuses.append(Status('freeze',self.freeze))
            return True
        return False
class EvoSkelBarrel(Component):
    def __init__(self,drop_pct):
        self.drop_pct=drop_pct;self.dropped=False
    def on_tick(self,tr,g):
        if self.dropped:return
        if tr.hp<=tr.max_hp*self.drop_pct:
            self.dropped=True
            for c in tr.components:
                if isinstance(c,DeathSpawn):
                    for i in range(c.count):
                        ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
                        t=Troop(tr.team,tr.x+ox,tr.y+oy,dict(c.cfg,components=list(c.cfg.get('components',[]))))
                        g.players[tr.team].troops.append(t)
                    break
class EvoFirecracker(Component):
    def __init__(self,sparks,slow_pct,dur):self.sparks=sparks;self.slow_pct=slow_pct;self.dur=dur
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
                if dd<=1.0:e.statuses.append(Status('slow',self.dur,sv))
class EvoArchers(Component):
    def __init__(self,mn_rng,mx_rng,dmg_m):
        self.mn=mn_rng;self.mx=mx_rng;self.dmg_m=dmg_m
    def on_attack(self,tr,tgt,g):
        if self.mn<=g._dist(tr,tgt)<=self.mx:
            extra=int(tr.dmg*(self.dmg_m-1))
            tgt.take_damage(extra)
            if hasattr(tgt,'ttype') and not tgt.alive:g._tower_down(tgt)
class EvoValkyrie(Component):
    def __init__(self,radius,dmg,dur):
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
    def __init__(self,ammo,rng,dmg_m,min_rng):
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
    def __init__(self,radius,dur,tiers,esc):
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
    def __init__(self,ldmg,lr):
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
    def __init__(self,pr):
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
    def __init__(self,throw_rng,stun_dur,impact_dmg,cost,cd):
        super().__init__(cost,cd);self.throw_rng=throw_rng;self.stun_dur=stun_dur
        self.impact_dmg=impact_dmg
    def activate(self,tr,g):
        opp=g._opp(tr.team)
        best=None;bhp=0
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=g._dist(tr,e)
            if d<=2.0 and e.max_hp>bhp:bhp=e.max_hp;best=e
        if not best:self.cd=0;return
        if tr.x<9:best.x=min(17,best.x+self.throw_rng)
        else:best.x=max(0,best.x-self.throw_rng)
        best.take_damage(self.impact_dmg)
        best.statuses.append(Status('stun',self.stun_dur))
        self.active=False;self.cd=self.max_cd
class BreakfastBoost(Ability):
    # stacks cook over time; each stack adds a percent of hp and damage on the single use
    def __init__(self,heal_pct,hp_pct,dmg_pct,max_stacks,cook_time,cost):
        super().__init__(cost,0,delay=999);self.heal_pct=heal_pct;self.hp_pct=hp_pct;self.dmg_pct=dmg_pct
        self.max_stacks=max_stacks;self.cook_time=cook_time;self.meters=0;self.cook_timer=0;self.uses=1
    def can_use(self):return self.uses>0 and not self.casting and not getattr(self,'_pend',False)
    def activate(self,tr,g):
        n=1+min(self.meters,self.max_stacks-1)
        oh=tr.max_hp
        tr.max_hp=int(tr.max_hp*(1+n*self.hp_pct));tr.hp+=tr.max_hp-oh
        tr.dmg=int(tr.dmg*(1+n*self.dmg_pct))
        tr.hp=min(tr.max_hp,tr.hp+int(tr.max_hp*self.heal_pct))
    def tick(self,dt,tr,g):
        if self.casting:
            self.cast_timer-=dt
            if self.cast_timer<=0:self.casting=False;self.activate(self._cast_tr,g)
            return
        if tr is None:return
        self.cook_timer+=dt
        if self.cook_timer>=self.cook_time:
            self.cook_timer-=self.cook_time
            if self.meters<self.max_stacks-1:self.meters+=1
class TrustyTurret(Ability):
    def __init__(self,turret_cfg,cost,cd):
        super().__init__(cost,cd);self.turret_cfg=turret_cfg
    def activate(self,tr,g):
        dy=3.0 if tr.team=='blue' else -3.0
        bld=Building(tr.team,tr.x,tr.y+dy,dict(self.turret_cfg))
        g.players[tr.team].troops.append(bld)
        self.active=False;self.cd=self.max_cd
class SkillAbility(Ability):
    # generic hero cast: spawn listed units, stun or slow enemies in radius, boost allies in radius
    def __init__(self,spawns,stun,slow,boost,radius,cost,cd):
        super().__init__(cost,cd);self.spawns=spawns;self.stun=stun;self.slow=slow;self.boost=boost;self.radius=radius
    def activate(self,tr,g):
        for cfg,n in self.spawns:
            for _ in range(n):
                ox=random.uniform(-1.0,1.0);oy=random.uniform(-1.0,1.0)
                mk=Building if cfg.get('is_building') else Troop
                g.players[tr.team].troops.append(mk(tr.team,tr.x+ox,tr.y+oy,dict(cfg,components=list(cfg['components']))))
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive or math.hypot(e.x-tr.x,e.y-tr.y)>self.radius:continue
            if self.stun:e.statuses.append(Status('stun',self.stun))
            if self.slow:e.statuses.append(Status('slow',self.slow[0],self.slow[1]))
        if self.boost:
            for a in g.players[tr.team].troops:
                if a.alive and math.hypot(a.x-tr.x,a.y-tr.y)<=self.radius:a.statuses.append(Status('rage',self.boost[0],self.boost[1]))
        self.active=False;self.cd=self.max_cd
class FieryFlight(Ability):
    def __init__(self,dur,spd_boost,tornado_r,flying,cost,cd):
        super().__init__(cost,cd);self.max_dur=dur;self.spd_boost=spd_boost;self.flying=flying
        self.tornado_r=tornado_r;self.orig_spd=None;self.orig_transport=None
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur
        self.orig_spd=tr.spd;self.orig_transport=tr.transport
        tr.spd*=(1+self.spd_boost)
        if self.flying:tr.transport='Air'
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
    def __init__(self,bonus_dmg_pct,cost):
        super().__init__(cost,0,delay=999);self.bonus_pct=bonus_dmg_pct;self.uses=1
    def can_use(self):return self.uses>0 and not self.casting and not getattr(self,'_pend',False)
    def activate(self,tr,g):
        opp=g._opp(tr.team)
        best=None;bmhp=999999
        for e in g.players[opp].troops:
            if not e.alive:continue
            if e.max_hp<bmhp:bmhp=e.max_hp;best=e
        if not best:return
        tr.x=best.x;tr.y=best.y
        bonus=int(tr.dmg*self.bonus_pct)
        best.take_damage(tr.dmg+bonus)
class EvoBabyDragon(Component):
    def __init__(self,radius,ally_boost,enemy_slow):
        self.radius=radius;self.ab=ally_boost;self.es=enemy_slow
    def on_tick(self,tr,g):
        for a in g.players[tr.team].troops:
            if a is tr or not a.alive:continue
            d=math.sqrt((a.x-tr.x)**2+(a.y-tr.y)**2)
            if d<=self.radius:
                if not any(s.kind=='evo_speed' for s in getattr(a,'statuses',[])):
                    a.statuses.append(Status('evo_speed',0.2,self.ab))
        opp=g._opp(tr.team)
        for e in g.players[opp].troops:
            if not e.alive:continue
            d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
            if d<=self.radius:
                if not any(s.kind=='slow' for s in e.statuses):
                    e.statuses.append(Status('slow',0.2,1.0-self.es))
class EvoWitch(Component):
    def __init__(self,heal,cap):self.heal=heal;self.cap=cap
    def on_tick(self,tr,g):
        cap=self.cap
        wid=id(tr)
        dead=[t for t in g.players[tr.team].troops if not t.alive and 'keleton' in getattr(t,'name','') and getattr(t,'_spawner_id',None)==wid]
        for d in dead:
            if tr.hp<cap:tr.hp=min(cap,tr.hp+self.heal)
class EvoPekka(Component):
    def __init__(self,small,med,large,cap):
        self.tiers={'s':small,'m':med,'l':large};self.cap=cap
    def on_attack(self,tr,tgt,g):
        if getattr(tgt,'alive',True):return
        cap=self.cap
        mhp=getattr(tgt,'max_hp',0)
        if mhp<=500:h=self.tiers['s']
        elif mhp<=1500:h=self.tiers['m']
        else:h=self.tiers['l']
        if tr.hp<cap:tr.hp=min(cap,tr.hp+h)
class EvoGoblinGiant(Component):
    def __init__(self,threshold,interval,gcfg):
        self.threshold=threshold;self.interval=interval;self.timer=0;self.gcfg=gcfg
    def on_tick(self,tr,g):
        if tr.hp>tr.max_hp*self.threshold or not self.gcfg:return
        self.timer-=g.DT
        if self.timer<=0:
            ox=random.uniform(-1.0,1.0)
            t=Troop(tr.team,tr.x+ox,tr.y,dict(self.gcfg,components=[]))
            g.players[tr.team].troops.append(t)
            self.timer=self.interval
class EvoHunter(Component):
    def __init__(self,net_dur,net_cd):
        self.net_dur=net_dur;self.net_cd=net_cd;self.cd=0;self.first=True
    def on_tick(self,tr,g):
        if self.cd>0:self.cd-=g.DT
        if not self.first and self.cd>0:return
        if self.first or self.cd<=0:
            tgt=getattr(tr,'tgt',None)
            if tgt and hasattr(tgt,'statuses'):
                tgt.statuses.append(Status('stun',self.net_dur))
                self.cd=self.net_cd;self.first=False
class Bolt:
    # the Evolved Electro Dragon's bolt keeps jumping between the enemies within reach of the last one hit until only one is left
    def __init__(self,team,prev,dmg,r,period):
        self.team=team;self.prev=prev;self.dmg=dmg;self.radius=r;self.period=period;self.t=period;self.active=True;self.name='';self.x,self.y=pos(prev)
    def tick(self,dt,g):
        self.t-=dt
        if self.t>0:return
        best=bounce(g,g._opp(self.team),self.prev,self.radius,(self.prev,))
        if best is None:self.active=False;return
        hurt(best,self.dmg,g);self.prev=best;self.x,self.y=pos(best);self.t=self.period
class EvoElectroDragon(Component):
    def __init__(self,dmg_pct,bounce_r,period):
        self.pct=dmg_pct;self.br=bounce_r;self.period=period
    def on_attack(self,tr,tgt,g):
        hit=getattr(tr,'chain_hit',[tgt])
        if len(hit)>1:g.spells.append(Bolt(tr.team,hit[-1],int(tr.dmg*self.pct),self.br,self.period))
class EvoWallBreakers(Component):
    def __init__(self,runner_cfg,cnt):
        self.runner_cfg=runner_cfg;self.cnt=cnt
    def on_death(self,tr,g):
        if not self.runner_cfg:return
        for _ in range(self.cnt):
            ox=random.uniform(-0.5,0.5)
            t=Troop(tr.team,tr.x+ox,tr.y,dict(self.runner_cfg,components=[]))
            g.players[tr.team].troops.append(t)
class EvoExecutioner(Component):
    def __init__(self,close_rng,dmg_m,kb):
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
class RowdyReroll(Ability):
    def __init__(self,roll_dist,heal_pct,roll_dmg,cost):
        super().__init__(cost,0,delay=999);self.roll_dist=roll_dist
        self.heal_pct=heal_pct;self.roll_dmg=roll_dmg;self.uses=1
    def can_use(self):return self.uses>0 and not self.casting and not getattr(self,'_pend',False)
    def activate(self,tr,g):
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
class MKJump(Component):
    # dur is the whole jump (wind-up plus flight); the wind-up is what remains after the flight time
    def __init__(self,mn,mx,splash_r,jspd,dur=0.9,kb=0):
        self.mn=mn;self.mx=mx;self.sr=splash_r;self.jspd=jspd;self.dur=dur;self.kb=kb
        self.charging=False;self.airborne=False;self.timer=0
        self.osp=None;self.jtgt=None;self.jdist=0
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
                    opp=g._opp(tr.team)
                    best=self.jtgt;bd=999
                    for e in g.players[opp].troops:
                        if not e.alive:continue
                        dd=math.sqrt((tr.x-e.x)**2+(tr.y-e.y)**2)
                        if dd<bd:bd=dd;best=e
                    self.jtgt=best
                    ax,ay=tr.x,tr.y;tr.x,tr.y=pos(self.jtgt);d=math.hypot(tr.x-ax,tr.y-ay) or 1
                    # the knockback origin sits a hair behind the landing so a troop under him is thrown forward
                    for e in near(g,tr.team,tr.x,tr.y,self.sr,air='Air' in getattr(tr,'targets',['Ground'])):
                        hurt(e,jd,g);push(e,tr.x-(tr.x-ax)/d*0.01,tr.y-(tr.y-ay)/d*0.01,self.kb)
                self.jtgt=None
            return
        tgt=getattr(tr,'tgt',None)
        if not tgt:
            if self.charging and self.osp is not None:tr.spd=self.osp;self.osp=None
            self.charging=False;return
        tx,ty=pos(tgt);d=math.hypot(tr.x-tx,tr.y-ty);rd=g._dist(tr,tgt)
        if self.mn<=rd<=self.mx and not self.charging:
            self.charging=True;self.timer=max(0.0,self.dur-d/self.jspd);self.osp=tr.spd;tr.spd=0;self.jtgt=tgt;self.jdist=d
        if self.charging:
            if tgt and tgt is not self.jtgt and rd<self.mn:
                self.jtgt=tgt;self.jdist=d
            self.timer-=g.DT
            if self.timer<=0:
                self.charging=False;self.airborne=True;self.timer=min(self.dur,self.jdist/self.jspd)
class EvoMegaKnight(Component):
    def __init__(self,kb):self.kb=kb
    def on_attack(self,tr,tgt,g):
        if not hasattr(tgt,'x') or not hasattr(tgt,'y'):return
        twy=g.arena.get_tower(getattr(tgt,'team','red'),'king').cy
        dy=twy-tgt.y
        if abs(dy)>0.1:tgt.y+=dy/abs(dy)*min(self.kb,abs(dy))
class EvoInfernoDragon(Component):
    def __init__(self,s4_dmg,retain_sec,s4_time):
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
    def __init__(self,soul_cnt,scfg):
        self.cnt=soul_cnt;self.scfg=scfg;self.sr=scfg['rng']
        self.was_invis=True
    def on_tick(self,tr,g):
        is_invis=any(s.kind=='invisible' for s in getattr(tr,'statuses',[]))
        if self.was_invis and not is_invis:
            opp=g._opp(tr.team)
            for e in g.players[opp].troops:
                if not e.alive:continue
                d=math.sqrt((e.x-tr.x)**2+(e.y-tr.y)**2)
                if d<=self.sr:e.take_damage(self.scfg['dmg'])
            for _ in range(self.cnt):
                ox=random.uniform(-0.5,0.5);oy=random.uniform(-0.5,0.5)
                g.players[tr.team].troops.append(Troop(tr.team,tr.x+ox,tr.y+oy,dict(self.scfg,components=[])))
        self.was_invis=is_invis
class EvoLumberjack(Component):
    def __init__(self,ghost_dur):
        self.ghost_dur=ghost_dur
    def on_death(self,tr,g):
        cfg={'hp':1,'dmg':tr.dmg,'hspd':tr.hspd,'fhspd':tr.fhspd,
             'spd':tr.spd,'rng':tr.rng,'targets':tr.targets,'transport':'Ground',
             'atk_type':'single_target','splash_r':0,'ct_dmg':0,
             'components':[],'lvl':tr.lvl,'name':'Lumberjack Ghost'}
        ghost=Troop(tr.team,tr.x,tr.y,cfg)
        ghost.max_hp=1;ghost.hp=1
        ghost.statuses.append(Status('invisible',self.ghost_dur))
        g.players[tr.team].troops.append(ghost)
class TripleThreat(Ability):
    def __init__(self,dash_dist,decoy_hp,triple_rng,dur,cost,cd):
        super().__init__(cost,cd);self.dash_dist=dash_dist;self.decoy_hp=decoy_hp
        self.triple_rng=triple_rng;self.max_dur=dur;self.empowered=False
    def activate(self,tr,g):
        dy=-self.dash_dist if tr.team=='blue' else self.dash_dist
        tr.y=max(0,min(31,tr.y+dy))
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
class LineAttack(Component):
    # the shot runs rng tiles from the shooter through the target, hw wide; passes>1 is a boomerang (second sweep after ret), sparks fan out behind the target
    def __init__(self,rng,hw,kb=0,passes=1,ret=0,sparks=0):
        self.rng=rng;self.hw=hw;self.kb=kb;self.passes=passes;self.ret=ret;self.sparks=sparks
    def on_attack(self,tr,tgt,g):
        tx,ty=pos(tgt);dx=tx-tr.x;dy=ty-tr.y;d=math.hypot(dx,dy)
        if d<=0:return
        air='Air' in getattr(tr,'targets',['Ground'])
        if self.sparks:
            a0=math.atan2(dy,dx)
            for i in range(self.sparks):
                a=a0+math.radians(-30+60*i/(self.sparks-1)) if self.sparks>1 else a0
                for e in strip(g,tr.team,tx,ty,tx+math.cos(a)*self.rng,ty+math.sin(a)*self.rng,self.hw,air,skip=(tgt,)):hurt(e,tr.dmg,g)
            return
        x1=tr.x+dx/d*self.rng;y1=tr.y+dy/d*self.rng;x0,y0=tr.x,tr.y
        def sweep(g,skip):
            for e in strip(g,tr.team,x0,y0,x1,y1,self.hw,air,skip):
                hurt(e,tr.ct_dmg if hasattr(e,'ttype') and tr.ct_dmg else tr.dmg,g);push(e,x0,y0,self.kb)
        sweep(g,(tgt,));push(tgt,x0,y0,self.kb)
        for i in range(1,self.passes):g.spells.append(Timer(self.ret*i,lambda g:sweep(g,()),x1,y1,tr.team))
class Burrow(Component):
    # underground from the own king tower to the deploy spot at spd tiles/s; surfaces after the deploy time or the travel, whichever is longer
    def __init__(self,spd,deploy):self.spd=spd;self.deploy=deploy;self.t=None
    def start(self,tr):
        self.tx,self.ty=tr.x,tr.y;self.sx,self.sy=Arena.W/2,Arena.KING_Y[0 if tr.team=='blue' else 1]
        self.T=max(self.deploy,math.hypot(self.tx-self.sx,self.ty-self.sy)/self.spd if self.spd>0 else 0);self.t=0
        tr.statuses.append(Status('burrowed',self.T));tr.x,tr.y=self.sx,self.sy
    def on_tick(self,tr,g):
        if self.t is None:self.start(tr)
        if self.t>=self.T:return
        self.t+=g.DT;f=min(1,self.t/self.T) if self.T>0 else 1
        tr.x=self.sx+(self.tx-self.sx)*f;tr.y=self.sy+(self.ty-self.sy)*f
        if f>=1:tr.statuses=[s for s in tr.statuses if s.kind!='burrowed']
class Resurface(Component):
    # evo drill: at each hp threshold it submerges, pops back up with its spawn damage and leaves goblins behind
    def __init__(self,thresholds,counts,cfg):self.th=list(thresholds);self.counts=list(counts);self.cfg=cfg;self.done=set()
    def on_tick(self,tr,g):
        for i,th in enumerate(self.th):
            if i in self.done or tr.hp>tr.max_hp*th:continue
            self.done.add(i)
            for _ in range(self.counts[i] if i<len(self.counts) else self.counts[-1]):
                g.players[tr.team].troops.append(Troop(tr.team,tr.x+random.uniform(-1,1),tr.y+random.uniform(-1,1),dict(self.cfg,components=list(self.cfg['components']))))
            z=next((c for c in tr.components if isinstance(c,SpawnZap)),None)
            if z:z.fire(tr,g)
class RocketRide(Component):
    # below pct of hp he mounts the rocket: very fast, buildings only, the explosion is his death damage on contact or when the fuse runs out
    def __init__(self,pct,spd,rng,life):self.pct=pct;self.spd=spd;self.rng=rng;self.life=life;self.on=False
    def on_tick(self,tr,g):
        if not self.on:
            if tr.hp>tr.max_hp*self.pct:return
            self.on=True;self.t=self.life;tr.spd=self.spd;tr.rng=self.rng;tr.targets=['Buildings'];tr.is_suicide=True;tr.proj_spd=0
            tr.dmg=tr.ct_dmg=0;tr.atk_type='single_target';tr.splash_r=0;tr.aggro_tgt=None
            tr.components=[c for c in tr.components if not isinstance(c,SplashAttack)]+[BuildingTarget()]
            return
        self.t-=g.DT
        if self.t<=0:tr.alive=False
class Breakdown(Component):
    # below pct of hp the cart is a rooted, knockback immune building whose remaining hitpoints drain over the lifetime
    def __init__(self,pct,life):self.pct=pct;self.life=life;self.on=False
    def on_tick(self,tr,g):
        if self.on or tr.hp>tr.max_hp*self.pct:return
        self.on=True;tr.is_building=True;tr.spd=0;tr.kb_immune=True;tr.decay=tr.hp/self.life if self.life>0 else 0;tr.aggro_tgt=None
class Hatch(Component):
    # the egg is removed without death effects and the reborn unit takes its place
    def __init__(self,cfg,delay):self.cfg=cfg;self.t=delay
    def on_tick(self,tr,g):
        self.t-=g.DT
        if self.t>0:return
        tr.alive=False;tr.components=[]
        g.players[tr.team].troops.append(Troop(tr.team,tr.x,tr.y,dict(self.cfg,components=list(self.cfg['components']))))
class CurseOnHit(Component):
    # every hit marks the troop for dur seconds; a marked troop that dies leaves a hog for the witch's side
    def __init__(self,cfg,dur):self.cfg=cfg;self.dur=dur;self.marks={}
    def on_attack(self,tr,tgt,g):
        if not hasattr(tgt,'ttype') and not getattr(tgt,'is_building',False):self.marks[id(tgt)]=(tgt,g.t+self.dur)
    def on_tick(self,tr,g):
        for k,(e,until) in list(self.marks.items()):
            if e.alive and g.t<=until:continue
            del self.marks[k]
            # troops that break into sub-troops (Golem, Lava Hound) do not turn into hogs
            if e.alive or any(isinstance(c,DeathSpawn) for c in e.components):continue
            g.players[tr.team].troops.append(Troop(tr.team,e.x,e.y,dict(self.cfg,components=list(self.cfg['components']))))
class Parry(Component):
    # blocks one melee hit every cd seconds and returns mult times the blocked damage to the attacker
    def __init__(self,mult,cd):self.mult=mult;self.cd=cd;self.ready=0
    def on_tick(self,tr,g):self.ready=max(0,self.ready-g.DT)
    def pre_damage(self,tr,att,dmg,g):
        melee=not hasattr(att,'ttype') and getattr(att,'proj_spd',0)<=0 and att.rng<2 and getattr(att,'transport','Ground')=='Ground'
        if self.ready>0 or not melee or dmg<=0:return dmg
        self.ready=self.cd;att.take_damage(int(dmg*self.mult));return 0
class Enchanted(Component):
    def __init__(self,bonus,every,src,linger):self.bonus=bonus;self.every=every;self.src=src;self.linger=linger;self.n=0;self.until=None
    def on_tick(self,tr,g):
        if self.until is None and not self.src.alive:self.until=g.t+self.linger
        if self.until is not None and g.t>self.until:tr.components=[c for c in tr.components if c is not self]
    def on_attack(self,tr,tgt,g):
        self.n+=1
        if self.n%self.every==0:hurt(tgt,self.bonus,g)
class Enchant(Component):
    # keeps the limit nearest allied troops in rng enchanted; buildings and self-destructing troops are skipped
    def __init__(self,rng,limit,every,bonus,linger):self.rng=rng;self.limit=limit;self.every=every;self.bonus=bonus;self.linger=linger
    def on_tick(self,tr,g):
        al=[a for a in g.players[tr.team].troops if a.alive and a is not tr]
        n=sum(1 for a in al if any(isinstance(c,Enchanted) and c.src is tr for c in a.components))
        if n>=self.limit:return
        ok=lambda a:not getattr(a,'is_building',False) and not a.is_suicide and not any(isinstance(c,Enchanted) for c in a.components)
        cands=sorted((math.hypot(a.x-tr.x,a.y-tr.y),id(a),a) for a in al if ok(a) and math.hypot(a.x-tr.x,a.y-tr.y)<=self.rng)
        for _,_,a in cands[:self.limit-n]:a.components.append(Enchanted(self.bonus,self.every,tr,self.linger))
class EvoBattleRam(Component):
    # survives the connection (the recoil re-arms the charge), bulldozes troops on its charge path and rages the barbarians it drops
    def __init__(self,kb,dmg,boost,dur):self.kb=kb;self.dmg=dmg;self.boost=boost;self.dur=dur;self.hit=set()
    def on_tick(self,tr,g):
        ch=next((c for c in tr.components if isinstance(c,Charge)),None)
        if not ch or not ch.charged:self.hit.clear();return
        for e in enemies(g,tr.team,air=False,towers=False):
            if id(e) in self.hit or getattr(e,'is_building',False) or math.hypot(e.x-tr.x,e.y-tr.y)>tr.collision_r+e.collision_r+0.2:continue
            self.hit.add(id(e));e.take_damage(self.dmg);push(e,tr.x,tr.y,self.kb)
    def on_attack(self,tr,tgt,g):
        # it won't stop charging: the recoil re-arms the charge without another run-up
        for c in tr.components:
            if isinstance(c,Charge):c.moved=c.dist
    def on_death(self,tr,g):
        for a in g.players[tr.team].troops:
            if a.alive and a.name=='Barbarian' and math.hypot(a.x-tr.x,a.y-tr.y)<=1.5:a.statuses.append(Status('rage',self.dur,self.boost))
class EvoCannon(Component):
    # the deploy barrage: two rows of cannonballs ahead of the cannon; the landing pattern is not published, the rows are 2 tiles apart
    def __init__(self,n,r,dmg,ct,kb):self.n=n;self.r=r;self.dmg=dmg;self.ct=ct;self.kb=kb;self.done=False
    def on_tick(self,tr,g):
        if self.done:return
        self.done=True;dy=1 if tr.team=='blue' else -1;top=(self.n+1)//2
        pts=[(tr.x+(i-(top-1)/2)*2.0,tr.y+dy*2.5) for i in range(top)]+[(tr.x+(i-(self.n-top-1)/2)*2.0,tr.y+dy*4.5) for i in range(self.n-top)]
        for x,y in pts:
            for e in near(g,tr.team,x,y,self.r,air=False):hurt(e,self.ct if hasattr(e,'ttype') else self.dmg,g);push(e,x,y,self.kb)
class EvoEliteBarbarians(Component):
    # a rage-tipped spear at a ground target between mn and mx tiles every cd seconds; rage circles on the target and along the path
    def __init__(self,dmg,mn,mx,cd,rr,rdur,boost):self.dmg=dmg;self.mn=mn;self.mx=mx;self.cd=cd;self.rr=rr;self.rdur=rdur;self.boost=boost;self.t=0
    def on_tick(self,tr,g):
        self.t=max(0,self.t-g.DT);tgt=getattr(tr,'tgt',None)
        if self.t>0 or not tgt or not getattr(tgt,'alive',False) or getattr(tgt,'transport','Ground')=='Air':return
        tx,ty=pos(tgt)
        if not self.mn<=g._dist(tr,tgt)<=self.mx:return
        self.t=self.cd;hurt(tgt,self.dmg,g)
        for f in (0.5,1.0):g.spells.append(Zone(tr.team,tr.x+(tx-tr.x)*f,tr.y+(ty-tr.y)*f,self.rr,self.rdur,'rage',self.boost,ally=True,name='Rage'))
class EvoFurnace(Component):
    # spawns mult times faster while it has a target in range
    def __init__(self,mult):self.mult=mult;self.hot=False
    def on_tick(self,tr,g):
        st=next((c for c in tr.components if isinstance(c,SpawnTimer)),None)
        if not st:return
        base=getattr(st,'base',st.interval);st.base=base
        tgt=getattr(tr,'tgt',None);hot=tgt is not None and getattr(tgt,'alive',False) and g._dist(tr,tgt)<=tr.rng
        st.interval=base/self.mult if hot else base
        if hot and not self.hot:st.timer=min(st.timer,st.interval)
        self.hot=hot
class EvoMinionHorde(Component):
    # the first hit taken veils the minion: untargetable and immune, slower to move and attack
    def __init__(self,dur,pen):self.dur=dur;self.pen=pen;self.done=False
    def on_tick(self,tr,g):
        if self.done or tr.hp>=tr.max_hp:return
        self.done=True
        for k,v in (('invincible',0),('invisible',0),('slow',self.pen)):tr.statuses.append(Status(k,self.dur,v))
class EvoPrincess(Component):
    # the first shot and every nth after it slow around the target; her death leaves a frost zone
    def __init__(self,every,r,val,dur):self.every=every;self.r=r;self.val=val;self.dur=dur;self.n=0
    def on_attack(self,tr,tgt,g):
        self.n+=1
        if (self.n-1)%self.every:return
        tx,ty=pos(tgt)
        for e in near(g,tr.team,tx,ty,self.r):e.statuses.append(Status('slow',self.dur,self.val))
    def on_death(self,tr,g):g.spells.append(Zone(tr.team,tr.x,tr.y,self.r,self.dur,'slow',self.val,name='Frost'))
class Shadow(Component):
    def __init__(self,gen):self.gen=gen
    def on_tick(self,tr,g):
        if not self.gen.alive:tr.alive=False
class EvoSkelArmy(Component):
    # a skeleton falling within the General's aura rises as an untargetable, indestructible shadow that fades with him
    def __init__(self,r,spd):self.r=r;self.spd=spd
    def on_death(self,tr,g):
        gen=next((a for a in g.players[tr.team].troops if a.alive and a.name=='General Gerry' and math.hypot(a.x-tr.x,a.y-tr.y)<=self.r),None)
        if not gen:return
        cfg={'hp':tr.max_hp,'dmg':tr.dmg,'hspd':tr.hspd,'fhspd':tr.fhspd,'spd':self.spd,'rng':tr.rng,'targets':tr.targets,'transport':'Ground',
             'atk_type':'single_target','splash_r':0,'ct_dmg':tr.ct_dmg,'components':[Shadow(gen)],'lvl':tr.lvl,'name':'Shadow Skeleton'}
        sh=Troop(tr.team,tr.x,tr.y,cfg);sh.statuses+=[Status('invincible',1e9),Status('invisible',1e9)]
        g.players[tr.team].troops.append(sh)
class EvoTesla(Component):
    # electro pulse on deploy and on destruction; troops only
    def __init__(self,dmg,r,stun):self.dmg=dmg;self.r=r;self.stun=stun;self.done=False
    def pulse(self,tr,g):
        for e in near(g,tr.team,tr.x,tr.y,self.r,towers=False):hurt(e,self.dmg,g);e.statuses.append(Status('stun',self.stun))
    def on_tick(self,tr,g):
        if not self.done:self.done=True;self.pulse(tr,g)
    def on_death(self,tr,g):self.pulse(tr,g)
class CoffinCadets(Ability):
    # a Skeletrooper drops on the nearest ground enemy within rng, dealing landing damage (ct to towers), then fights on foot
    def __init__(self,cfg,dmg,ct,rng,cost,cd):
        super().__init__(cost,cd);self.cfg=cfg;self.dmg=dmg;self.ct=ct;self.rng=rng
    def activate(self,tr,g):
        c=[(math.hypot(pos(e)[0]-tr.x,pos(e)[1]-tr.y),id(e),e) for e in enemies(g,tr.team,air=False)]
        c=[x for x in c if x[0]<=self.rng];best=min(c)[2] if c else None
        x,y=pos(best) if best else (tr.x,tr.y)
        if best:hurt(best,self.ct if hasattr(best,'ttype') else self.dmg,g)
        g.players[tr.team].troops.append(Troop(tr.team,x,y,dict(self.cfg,components=list(self.cfg['components']))))
        self.cd=self.max_cd
class SavageSurvival(Ability):
    # bear spirit: dmg per swing with reduced tower damage, hs hit speed, sp times faster feet, hp cannot drop below floor
    def __init__(self,dur,dmg,ct,hs,sp,floor,cost,cd):
        super().__init__(cost,cd);self.max_dur=dur;self.dmg=dmg;self.ct=ct;self.hs=hs;self.sp=sp;self.floor=floor
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur;self.o=(tr.dmg,tr.ct_dmg,tr.hspd,tr.spd)
        tr.dmg=self.dmg or tr.dmg;tr.ct_dmg=self.ct or tr.ct_dmg;tr.hspd=self.hs;tr.spd=tr.spd*self.sp;tr.hp_floor=self.floor;tr.cd=min(tr.cd,tr.hspd)
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:self.active=False;self.cd=self.max_cd;tr.dmg,tr.ct_dmg,tr.hspd,tr.spd=self.o;tr.hp_floor=0
class StoneSwish(Ability):
    # planted mortar stance: longer range, slower and heavier boulders landing as an area hit instead of rolling through
    def __init__(self,rng,hs,dm,ctm,aoe,dur,cast,cost,cd):
        super().__init__(cost,cd);self.rng=rng;self.hs=hs;self.dm=dm;self.ctm=ctm;self.aoe=aoe;self.max_dur=dur;self.CAST_TIME=cast
    def activate(self,tr,g):
        self.active=True;self.dur=self.max_dur;self.o=(tr.rng,tr.hspd,tr.dmg,tr.ct_dmg,tr.spd,tr.atk_type,tr.splash_r,tr.components)
        tr.rng=self.rng;tr.hspd=tr.hspd/self.hs;tr.dmg=int(tr.dmg*self.dm);tr.ct_dmg=int(tr.dmg*self.ctm);tr.spd=0;tr.atk_type='area';tr.splash_r=self.aoe
        tr.components=[c for c in tr.components if not isinstance(c,LineAttack)]+[SplashAttack()];tr.aggro_tgt=None
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:self.active=False;self.cd=self.max_cd;tr.rng,tr.hspd,tr.dmg,tr.ct_dmg,tr.spd,tr.atk_type,tr.splash_r,tr.components=self.o
class DestructiveDismount(Ability):
    # the prince jumps off with area damage and fights on foot without his charge while the Rhino charges buildings
    def __init__(self,rcfg,dmg,r,cost,cd):
        super().__init__(cost,cd);self.rcfg=rcfg;self.dmg=dmg;self.r=r
    def activate(self,tr,g):
        for e in near(g,tr.team,tr.x,tr.y,self.r,air=False,towers=False):e.take_damage(self.dmg)
        g.players[tr.team].troops.append(Troop(tr.team,tr.x,tr.y,dict(self.rcfg,components=list(self.rcfg['components']))))
        tr.components=[c for c in tr.components if not isinstance(c,Charge)];tr.charge_dmg=0;self.cd=self.max_cd
class Snowstorm(Ability):
    # n blasts in radius r, each damaging and slowing; blasts are spaced by the slow duration (the interval itself is not published)
    def __init__(self,n,r,dmg,val,sdur,cost,cd):
        super().__init__(cost,cd);self.n=n;self.r=r;self.dmg=dmg;self.val=val;self.sdur=sdur
    def activate(self,tr,g):self.active=True;self.left=self.n;self.t=0
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.t-=dt
        if self.t>0:return
        self.t=self.sdur;self.left-=1
        for e in near(g,tr.team,tr.x,tr.y,self.r,towers=False):e.take_damage(self.dmg);e.statuses.append(Status('slow',self.sdur,self.val))
        if self.left<=0:self.active=False;self.cd=self.max_cd
class FreezeAura(Component):
    # freezes everything within r while alive; self-destructs idle seconds after nothing is left to freeze
    def __init__(self,r,idle=2.0):self.r=r;self.idle=idle;self.t=0
    def on_tick(self,tr,g):
        hit=near(g,tr.team,tr.x,tr.y,self.r)
        for e in hit:refresh(e,'freeze',2*g.DT)
        self.t=0 if hit else self.t+g.DT
        if self.t>=self.idle:tr.alive=False
class FrostyFella(Ability):
    # the Snowman rises one tile behind the current target (or ahead of the hero) with spawn damage and a freeze aura for its lifetime
    def __init__(self,cfg,dmg,ct,r,cost,cd):
        super().__init__(cost,cd);self.cfg=cfg;self.dmg=dmg;self.ct=ct;self.r=r
    def activate(self,tr,g):
        tgt=getattr(tr,'tgt',None)
        if tgt and getattr(tgt,'alive',False):
            tx,ty=pos(tgt);d=math.hypot(tx-tr.x,ty-tr.y);x,y=(tx+(tx-tr.x)/d,ty+(ty-tr.y)/d) if d>0 else (tx,ty)
        else:x,y=tr.x,tr.y+(1 if tr.team=='blue' else -1)
        for e in near(g,tr.team,x,y,self.r):hurt(e,self.ct if hasattr(e,'ttype') else self.dmg,g)
        g.players[tr.team].troops.append(Building(tr.team,x,y,dict(self.cfg,components=[FreezeAura(self.r)])));self.cd=self.max_cd
class RegalRevive(Ability):
    # the tombstone crumbles without its death skeletons and the Tomb Queen rises in its place
    def __init__(self,cfg,cost,cd):super().__init__(cost,cd);self.cfg=cfg
    def activate(self,tr,g):
        tr.alive=False;tr.components=[]
        g.players[tr.team].troops.append(Troop(tr.team,tr.x,tr.y,dict(self.cfg,components=list(self.cfg['components']))));self.cd=self.max_cd
class WildWhirlwind(Ability):
    # dashes to the nearest ground troop then spins: rapid area hits at spin damage, reduced tower damage, faster feet, damage reduction
    def __init__(self,dash,dur,hs,dmg,ctm,r,sp,red,cost,cd):
        super().__init__(cost,cd);self.dash=dash;self.max_dur=dur;self.hs=hs;self.dmg=dmg;self.ctm=ctm;self.r=r;self.sp=sp;self.red=red
    def activate(self,tr,g):
        c=[(math.hypot(e.x-tr.x,e.y-tr.y),id(e),e) for e in enemies(g,tr.team,air=False,towers=False)]
        c=[x for x in c if x[0]<=self.dash]
        if c:
            e=min(c)[2];d=min(c)[0]
            if d>1:tr.x+=(e.x-tr.x)*(d-1)/d;tr.y+=(e.y-tr.y)*(d-1)/d
        self.active=True;self.dur=self.max_dur;self.o=(tr.hspd,tr.dmg,tr.ct_dmg,tr.splash_r,tr.spd,getattr(tr,'_dmg_reduction',0))
        tr.hspd=self.hs or tr.hspd;tr.dmg=self.dmg or tr.dmg;tr.ct_dmg=int(tr.dmg*self.ctm);tr.splash_r=self.r;tr.spd=tr.spd*self.sp
        tr._dmg_reduction=self.red;tr.cd=min(tr.cd,tr.hspd)
    def tick(self,dt,tr,g):
        if not self.active:super().tick(dt,tr,g);return
        self.dur-=dt
        if self.dur<=0:self.active=False;self.cd=self.max_cd;tr.hspd,tr.dmg,tr.ct_dmg,tr.splash_r,tr.spd,tr._dmg_reduction=self.o
