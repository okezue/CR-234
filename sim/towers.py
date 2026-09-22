import random
from sim.cards import card,at,first
from sim.units import hidden

def king(lvl):
    k=card('king_tower')
    return {'hp':at(k['stats']['hitpoints'],lvl),'dmg':at(k['stats']['damage'],lvl),'spd':k['hitSpeed'],'rng':k['range'],
            'fspd':first(k['hitSpeed'],k['loadTime']),'projSpeed':(k['projectile'] or {}).get('speed') or 0}

def lock(o,tw,en,rng):
    l=getattr(o,'lock',None)
    if l is not None and l.alive and l in en and not hidden(l) and tw.dist(l.x,l.y)-getattr(l,'collision_r',0)<=rng:return l
    b=None;bd=999
    for e in en:
        if not e.alive or hidden(e):continue
        d=tw.dist(e.x,e.y)-getattr(e,'collision_r',0)
        if d<=rng and d<bd:bd=d;b=e
    o.lock=b
    return b

def load_idle(cd,fspd,dt):
    # idle, the swing loads down to the first attack period and no further, the troop rule (wiki Tower Princess: Attack Period 0.8 s,
    # First Attack Period 0.8 s; the recording of game 09YP9UPGJGU8 shows her turning to the hogs and releasing about 0.55 s later)
    return max(fspd,cd-dt)

class TT:
    def __init__(self,jn,lvl):
        self.lvl=lvl;self.name=jn;self.lock=None
        d=card(jn);s=d['stats']
        self.hp=at(s['hitpoints'],lvl);self.dmg=at(s['damage'],lvl)
        self.spd=d['hitSpeed'];self.fspd=first(d['hitSpeed'],d['loadTime']);self.RNG=d['range'];self.cd=self.fspd
        self.proj_spd=(d['projectile'] or {}).get('speed') or 0
    def _tgt(self,tw,en):
        # the tower holds its target until it dies, hides or leaves range (a stun resets it); a tank keeps the fire off what follows
        return lock(self,tw,en,self.RNG)
    def tick(self,dt,tw,en,al,**kw):
        r=[];t=self._tgt(tw,en)
        if t:
            self.cd=max(0,self.cd-dt)
            if self.cd<=0:r.append(('atk',t,self.dmg));self.cd=self.spd
        else:self.cd=load_idle(self.cd,self.fspd,dt)
        return r

class TPrincess(TT):
    def __init__(self,lvl):super().__init__('tower_princess',lvl)

class Cannoneer(TT):
    # first shot 0.8 s after acquiring (2.2 hit speed less the 1.4 load), the shared tower troop rule
    def __init__(self,lvl):super().__init__('cannoneer',lvl)

class DaggerDuchess(TT):
    def __init__(self,lvl):
        super().__init__('dagger_duchess',lvl)
        v=card('dagger_duchess')['skills']['volley']
        self.bspd=self.spd;self.cspd=v['reloadTime'];self.MXD=v['projectileCount']
        self.dag=self.MXD
    def tick(self,dt,tw,en,al,**kw):
        r=[];self.cd=max(0,self.cd-dt)
        if self.cd>0:return r
        t=self._tgt(tw,en)
        if t and self.dag>0:
            r.append(('atk',t,self.dmg));self.dag-=1
            self.cd=self.bspd if self.dag>0 else self.cspd
        elif t and self.dag==0:
            self.dag=1;r.append(('atk',t,self.dmg));self.dag=0
            self.cd=self.cspd
        elif not t and self.dag<self.MXD:
            self.dag+=1
            if self.dag<self.MXD:self.cd=self.cspd
        return r

class RoyalChef(TT):
    def __init__(self,lvl):
        super().__init__('royal_chef',lvl)
        k=card('royal_chef')['skills']['stack']
        self.ckdel=k['firstDelay'];self.ckt=0;self.cking=False
        self.prdy=False;self.bst=set()
        self.ckmin=k['interval'];self.ckmax=k['maxInterval']
    def tick(self,dt,tw,en,al,pt_dead=0,**kw):
        r=super().tick(dt,tw,en,al)
        attacking=self.lock is not None
        if self.ckdel>0:
            self.ckdel-=dt;return r
        if not self.prdy:
            if not self.cking:
                self.ckt=random.uniform(self.ckmin,self.ckmax)
                self.cking=True
            if pt_dead>=2:
                rate=0
            elif attacking:
                rate=0.6
            else:
                rate=1.0
            self.ckt-=dt*rate
            if self.ckt<=0:
                self.prdy=True;self.cking=False
        if self.prdy and al:
            b=None;bh=0
            for a in al:
                if not a.alive or a.max_hp<1 or a.hp/a.max_hp<=0.33:continue
                if getattr(a,'is_building',False):continue
                if a.hp==1 and a.max_hp==1:continue
                if a in self.bst:continue
                if a.hp>bh:bh=a.hp;b=a
            if not b:
                self.bst.clear();bh=0
                for a in al:
                    if not a.alive or a.max_hp<1 or a.hp/a.max_hp<=0.33:continue
                    if getattr(a,'is_building',False):continue
                    if a.hp==1 and a.max_hp==1:continue
                    if a.hp>bh:bh=a.hp;b=a
            if b:
                r.append(('pancake',b,1))
                self.prdy=False;self.bst.add(b)
        return r

def create(name,lvl):
    return {'tower_princess':TPrincess,'cannoneer':Cannoneer,
            'dagger_duchess':DaggerDuchess,'royal_chef':RoyalChef}[name](lvl)
