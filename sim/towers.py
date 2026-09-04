import random
from sim.cards import card,at,first
from sim.units import hidden
from sim.knobs import K

def king(lvl):
    k=card('king_tower')
    return {'hp':at(k['stats']['hitpoints'],lvl),'dmg':at(k['stats']['damage'],lvl),'spd':k['hitSpeed'],'rng':k['range'],
            'projSpeed':(k['projectile'] or {}).get('speed') or 0}

def lock(o,tw,en,rng):
    l=getattr(o,'lock',None)
    if l is not None and l.alive and l in en and not hidden(l) and tw.dist(l.x,l.y)-getattr(l,'collision_r',0)<=rng:return l
    b=None;bd=999
    for e in en:
        if not e.alive or hidden(e):continue
        d=tw.dist(e.x,e.y)-getattr(e,'collision_r',0)
        if d<=rng and d<bd:bd=d;b=e
    if b is not None and b is not l:o.cd=max(o.cd,K['tower_acq']+(K['tower_first'] if l is None else 0))
    o.lock=b
    return b

class TT:
    def __init__(self,jn,lvl):
        self.lvl=lvl;self.name=jn;self.cd=0;self.lock=None
        d=card(jn);s=d['stats']
        self.hp=at(s['hitpoints'],lvl);self.dmg=at(s['damage'],lvl)
        self.spd=d['hitSpeed'];self.fspd=first(d['hitSpeed'],d['loadTime']);self.RNG=d['range']
        self.proj_spd=(d['projectile'] or {}).get('speed') or 0
    def _tgt(self,tw,en):
        # the tower holds its target until it dies, hides or leaves range (a stun resets it); a tank keeps the fire off what follows
        return lock(self,tw,en,self.RNG)
    def tick(self,dt,tw,en,al,**kw):return []

class TPrincess(TT):
    def __init__(self,lvl):super().__init__('tower_princess',lvl)
    def tick(self,dt,tw,en,al,**kw):
        r=[];self.cd=max(0,self.cd-dt)
        t=self._tgt(tw,en)
        if t and self.cd<=0:
            r.append(('atk',t,self.dmg));self.cd=self.spd
        return r

class Cannoneer(TT):
    def __init__(self,lvl):
        super().__init__('cannoneer',lvl);self.eng=False
    def tick(self,dt,tw,en,al,**kw):
        r=[];self.cd=max(0,self.cd-dt)
        t=self._tgt(tw,en)
        if t:
            if not self.eng:self.cd=max(self.cd,self.fspd);self.eng=True
            if self.cd<=0:
                r.append(('atk',t,self.dmg));self.cd=self.spd
        else:
            self.eng=False;self.cd=0
        return r

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
        r=[];self.cd=max(0,self.cd-dt)
        t=self._tgt(tw,en)
        attacking=t is not None
        if t and self.cd<=0:
            r.append(('atk',t,self.dmg));self.cd=self.spd
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
