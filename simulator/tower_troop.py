import json,math,os,random

_D=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','game_data','cards')

def _ld(n):
    with open(os.path.join(_D,n+'.json')) as f : return json.load(f)

KING_STATS={
    1:(2400,50),2:(2568,54),3:(2736,58),4:(2904,62),5:(3096,67),
    6:(3312,72),7:(3528,78),8:(3768,84),9:(4008,90),10:(4392,99),
    11:(4824,109),12:(5304,119),13:(5832,131),14:(6408,144),15:(7032,158)
}

class TT:
    RNG=7.5
    def __init__(self,jn,lvl):
        self.lvl=lvl;self.name=jn;self.cd=0
        d=_ld(jn);s=d['stats_by_level'].get(str(lvl),{})
        self.hp=s.get('tower_hitpoints',3052)
        self.dmg=s.get('damage',109)
    def _tgt(self,tw,en):
        b=None;bd=999
        for e in en:
            if not e.alive:continue
            d=math.sqrt((e.x-tw.cx)**2+(e.y-tw.cy)**2)
            if d<=self.RNG and d<bd:bd=d;b=e
        return b
    def tick(self,dt,tw,en,al,**kw):return []

class TPrincess(TT):
    def __init__(self,lvl):
        super().__init__('tower_princess',lvl);self.spd=0.8
    def tick(self,dt,tw,en,al,**kw):
        r=[];self.cd=max(0,self.cd-dt)
        t=self._tgt(tw,en)
        if t and self.cd<=0:
            r.append(('atk',t,self.dmg));self.cd=self.spd
        return r

class Cannoneer(TT):
    def __init__(self,lvl):
        super().__init__('cannoneer',lvl)
        self.spd=2.2;self.fspd=0.8;self.eng=False
    def tick(self,dt,tw,en,al,**kw):
        r=[];self.cd=max(0,self.cd-dt)
        t=self._tgt(tw,en)
        if t:
            if not self.eng:self.cd=self.fspd;self.eng=True
            if self.cd<=0:
                r.append(('atk',t,self.dmg));self.cd=self.spd
        else:
            self.eng=False;self.cd=0
        return r

class DaggerDuchess(TT):
    MXD=8
    def __init__(self,lvl):
        super().__init__('dagger_duchess',lvl)
        self.bspd=0.5;self.cspd=0.9;self.dag=self.MXD
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
        super().__init__('royal_chef',lvl);self.spd=1.0
        self.ckdel=7.0;self.ckt=0;self.cking=False
        self.prdy=False;self.bst=set()
        self.ckmin=23.0;self.ckmax=38.0
    def tick(self,dt,tw,en,al,pt_dead=0,**kw):
        r=[];self.cd=max(0,self.cd-dt)
        t=self._tgt(tw,en)
        if t and self.cd<=0:
            r.append(('atk',t,self.dmg));self.cd=self.spd
        if self.ckdel>0:
            self.ckdel-=dt;return r
        if not self.prdy:
            if not self.cking:
                self.ckt=random.uniform(self.ckmin,self.ckmax)
                self.cking=True
            rate={0:1.0,1:0.7}.get(pt_dead,0)
            self.ckt-=dt*rate
            if self.ckt<=0:
                self.prdy=True;self.cking=False
        if self.prdy and al:
            b=None;bh=0
            for a in al:
                if not a.alive or a.hp/a.max_hp<=0.33:continue
                if id(a) in self.bst:continue
                if a.hp>bh:bh=a.hp;b=a
            if not b:
                self.bst.clear();bh=0
                for a in al:
                    if not a.alive or a.hp/a.max_hp<=0.33:continue
                    if a.hp>bh:bh=a.hp;b=a
            if b:
                r.append(('pancake',b,1))
                self.prdy=False;self.bst.add(id(b))
        return r

def create(name,lvl):
    return {'tower_princess':TPrincess,'cannoneer':Cannoneer,
            'dagger_duchess':DaggerDuchess,'royal_chef':RoyalChef}[name](lvl)
