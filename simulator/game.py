import math,random,sys,os,json
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from arena import Arena
from tower_troop import create as mk_tt,KING_STATS
try:
    from factory import create as mk_card
except ImportError:
    mk_card=None

_CD=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','game_data','cards')
_CC={}
def card_info(name):
    if name not in _CC:
        p=os.path.join(_CD,name+'.json')
        if os.path.exists(p):
            with open(p) as f:d=json.load(f)
            _CC[name]={'name':name,'cost':d.get('elixir_cost',d.get('elixir',3)),
                       'deploy':d.get('deploy_time_sec',1.0)}
        else:
            _CC[name]={'name':name,'cost':3,'deploy':1.0}
    return _CC[name]

class Deck:
    def __init__(self,cards):
        assert len(cards)==8
        self.all=list(cards)
        idx=list(range(8))
        random.shuffle(idx)
        self.hand=[cards[i] for i in idx[:4]]
        self.nxt=cards[idx[4]]
        self.nxt_cd=0
        self.q=[cards[i] for i in idx[5:]]
    def tick(self,dt,qcd):
        if self.nxt_cd>0:
            self.nxt_cd=max(0,self.nxt_cd-dt)
        if self.nxt_cd<0.001 and len(self.hand)<4 and self.nxt:
            self.hand.append(self.nxt)
            if self.q:
                self.nxt=self.q.pop(0)
                self.nxt_cd=qcd
            else:
                self.nxt=None
    def play(self,card,qcd):
        if card not in self.hand:return False
        self.hand.remove(card)
        self.q.append(card)
        if self.nxt and self.nxt_cd<=0:
            self.hand.append(self.nxt)
            if self.q:
                self.nxt=self.q.pop(0)
                self.nxt_cd=qcd
            else:
                self.nxt=None
        return True
    def can_play(self,card):return card in self.hand
    def info(self):
        return f"hand={self.hand} nxt={self.nxt}({self.nxt_cd:.1f}s) q={self.q}"

class Pending:
    def __init__(self,team,card,x,y,rem):
        self.team=team;self.card=card
        self.x=x;self.y=y;self.rem=rem

class Dummy:
    _n=0
    def __init__(self,team,x,y,lvl=11,hp=500,dmg=100,spd=2.0,hspd=1.0,rng=1.5):
        Dummy._n+=1;self.id=Dummy._n
        self.team=team;self.x=float(x);self.y=float(y)
        self.hp=hp;self.max_hp=hp;self.dmg=dmg
        self.spd=spd;self.hspd=hspd;self.rng=rng
        self.alive=True;self.lvl=lvl;self.cd=0
        self.transport='Ground';self.targets=['Ground']
        self.components=[];self.statuses=[]
        self.atk_type='single_target';self.splash_r=0
        self.fhspd=hspd;self.first_atk=True;self.tgt=None
        self.name='dummy';self.ct_dmg=0
    def level_up(self):
        self.lvl+=1;oh=self.max_hp
        self.max_hp=int(self.max_hp*1.1)
        self.hp+=self.max_hp-oh
        self.dmg=int(self.dmg*1.1)
    def take_damage(self,a):
        if not self.alive:return
        self.hp-=a
        if self.hp<=0:self.hp=0;self.alive=False

class Player:
    def __init__(self,team,king_lvl=11,tt_name='tower_princess',tt_lvl=11,
                 deck=None,drag_del=0.5,drag_std=None):
        self.team=team;self.king_lvl=king_lvl
        self.tt_name=tt_name;self.tt_lvl=tt_lvl
        self.elixir=5.0;self.max_ex=10.0
        self.crowns=0;self.troops=[]
        self.drag_del=drag_del
        self.drag_std=drag_std if drag_std is not None else drag_del*0.2
        self.deck=Deck(deck) if deck else None
    def sample_drag(self):
        return max(0.1,random.gauss(self.drag_del,self.drag_std))

class Game:
    REG=180.0;OT=300.0;EBASE=2.8;DT=0.1
    def __init__(self,p1=None,p2=None):
        self.arena=Arena();self.t=0
        self.phase='regulation';self.winner=None
        self.ended=False;self.log=[];self.pending=[];self.spells=[]
        self.players={
            'blue':Player('blue',**(p1 or {})),
            'red':Player('red',**(p2 or {}))
        }
        self._setup()
    def _setup(self):
        for tm in ('blue','red'):
            p=self.players[tm]
            for t in self.arena.towers:
                if t.team!=tm:continue
                if t.ttype=='princess':
                    tt=mk_tt(p.tt_name,p.tt_lvl)
                    t.hp=tt.hp;t.max_hp=tt.hp;t.troop=tt
                elif t.ttype=='king':
                    hp,dmg=KING_STATS.get(p.king_lvl,(4824,109))
                    t.hp=hp;t.max_hp=hp;t.dmg=dmg
                    t.active=False;t.cd=0;t.troop=None
    def _erate(self):
        if self.t<120:return 1
        if self.t<240:return 2
        return 3
    def _qcd(self):return 2.0/self._erate()
    def _opp(self,t):return 'red' if t=='blue' else 'blue'
    def _gen_ex(self):
        a=self.DT*self._erate()/self.EBASE
        for p in self.players.values():
            p.elixir=min(p.max_ex,p.elixir+a)
    def _king_act(self,team):
        kt=self.arena.get_tower(team,'king')
        if kt and not getattr(kt,'active',True):
            kt.active=True;kt.cd=4.0
            self.log.append(f"[{self.t:.1f}] {team} king activated!")
    def _tower_down(self,tower):
        opp=self._opp(tower.team)
        self.players[opp].crowns+=1
        if tower.ttype=='king':
            self.players[opp].crowns=3
            self.winner=opp;self.phase='end';self.ended=True
            self.log.append(f"[{self.t:.1f}] {opp} 3-crown win!")
        else:
            self._king_act(tower.team)
            self.log.append(f"[{self.t:.1f}] {tower.team} princess down! {opp}:{self.players[opp].crowns}cr")
            if self.phase=='overtime':
                self.winner=opp;self.phase='end';self.ended=True
                self.log.append(f"[{self.t:.1f}] {opp} wins sudden death!")
    def _tiebreaker(self):
        ts=[t for t in self.arena.towers if t.alive]
        if not ts:
            self.winner=None
            self.log.append(f"[{self.t:.1f}] Draw!")
        else:
            mn=min(t.hp for t in ts)
            lows=[t for t in ts if t.hp==mn]
            tms=set(t.team for t in lows)
            if len(tms)==2:
                self.winner=None
                self.log.append(f"[{self.t:.1f}] Draw! (equal HP)")
            else:
                l=lows[0];l.hp=0;l.alive=False
                opp=self._opp(l.team)
                self.players[opp].crowns+=1
                if l.ttype=='princess':self._king_act(l.team)
                elif l.ttype=='king':self.players[opp].crowns=3
                self.winner=opp
                self.log.append(f"[{self.t:.1f}] Tiebreaker: {l.team} {l.ttype} destroyed")
        self.phase='end';self.ended=True
    def _check_phase(self):
        if self.ended:return
        if self.t>=self.OT and self.phase=='overtime':
            self._tiebreaker();return
        if self.t>=self.REG and self.phase=='regulation':
            bc=self.players['blue'].crowns
            rc=self.players['red'].crowns
            if bc!=rc:
                self.winner='blue' if bc>rc else 'red'
                self.phase='end';self.ended=True
                self.log.append(f"[{self.t:.1f}] {self.winner} wins {bc}-{rc}")
            else:
                self.phase='overtime'
                self.log.append(f"[{self.t:.1f}] Overtime! {bc}-{rc}")
    def _valid_deploy(self,team,x,y):
        if not(0<=x<self.arena.W and 0<=y<self.arena.H):return False
        c=self.arena.grid[y][x]
        if c=='R':return False
        if c and len(c)==2 and c[1] in ('K','P','T'):
            for t in self.arena.towers:
                if t.alive and (x,y) in t.tiles():return False
        opp=self._opp(team)
        if team=='blue':
            if y<=15:return True
            rlp=self.arena.get_tower('red','princess','left')
            if rlp and not rlp.alive and x<=8 and 17<=y<=31:return True
            rrp=self.arena.get_tower('red','princess','right')
            if rrp and not rrp.alive and x>=9 and 17<=y<=31:return True
        else:
            if y>=16:return True
            blp=self.arena.get_tower('blue','princess','left')
            if blp and not blp.alive and x<=8 and 0<=y<=14:return True
            brp=self.arena.get_tower('blue','princess','right')
            if brp and not brp.alive and x>=9 and 0<=y<=14:return True
        return False
    def play_card(self,team,card,x,y):
        p=self.players[team]
        if not p.deck:return False,"no deck"
        ci=card_info(card)
        if not p.deck.can_play(card):return False,"not in hand"
        if p.elixir<ci['cost']:return False,"not enough elixir"
        ix,iy=int(x),int(y)
        if not self._valid_deploy(team,ix,iy):return False,"invalid position"
        p.elixir-=ci['cost']
        qcd=self._qcd()
        p.deck.play(card,qcd)
        drag=p.sample_drag()
        delay=drag+ci['deploy']
        self.pending.append(Pending(team,card,float(x),float(y),delay))
        self.log.append(f"[{self.t:.1f}] {team} plays {card} at ({x},{y}) drag={drag:.2f}s delay={delay:.2f}s")
        return True,"ok"
    def _spawn(self,team,card,x,y):
        p=self.players[team]
        if mk_card:
            try:return mk_card(card,p.king_lvl,team,x,y)
            except:pass
        return Dummy(team,x,y,lvl=p.king_lvl)
    def _proc_pending(self):
        done=[]
        for pd in self.pending:
            pd.rem-=self.DT
            if pd.rem<=0:
                r=self._spawn(pd.team,pd.card,pd.x,pd.y)
                if isinstance(r,list):
                    for tr in r:self.players[pd.team].troops.append(tr)
                elif hasattr(r,'apply'):
                    r.apply(self);self.spells.append(r)
                else:
                    self.players[pd.team].troops.append(r)
                self.log.append(f"[{self.t:.1f}] {pd.card} spawned at ({pd.x:.0f},{pd.y:.0f})")
                done.append(pd)
        for d in done:self.pending.remove(d)
    def _proc_towers(self):
        for t in self.arena.towers:
            if not t.alive:continue
            opp=self._opp(t.team)
            en=self.players[opp].troops
            al=self.players[t.team].troops
            if t.ttype=='princess' and getattr(t,'troop',None):
                pd=sum(1 for x in self.arena.towers
                       if x.team==t.team and x.ttype=='princess' and not x.alive)
                evts=t.troop.tick(self.DT,t,en,al,pt_dead=pd)
                for ev in evts:
                    if ev[0]=='atk':
                        ev[1].take_damage(ev[2])
                    elif ev[0]=='pancake':
                        for _ in range(ev[2]):ev[1].level_up()
                        self.log.append(f"[{self.t:.1f}] Chef boost -> lvl {ev[1].lvl}")
            elif t.ttype=='king' and getattr(t,'active',False):
                t.cd=max(0,t.cd-self.DT)
                if t.cd<=0:
                    b=None;bd=999
                    for e in en:
                        if not e.alive:continue
                        d=math.sqrt((e.x-t.cx)**2+(e.y-t.cy)**2)
                        if d<=t.rng and d<bd:bd=d;b=e
                    if b:
                        b.take_damage(t.dmg);t.cd=t.spd
    def _waypoint(self,tr,tx,ty):
        tp=getattr(tr,'transport','Ground')
        if tp=='Air':return tx,ty
        from components import RiverJump
        has_rj=any(isinstance(c,RiverJump) for c in getattr(tr,'components',[]))
        if has_rj:return tx,ty
        across=(tr.y<15.0 and ty>16.0) or (tr.y>16.0 and ty<15.0)
        if not across:return tx,ty
        in_rv=15.0<=tr.y<=17.0
        on_br=((3<=tr.x<=5) or (12<=tr.x<=14)) and in_rv
        if on_br:
            return tr.x,(17.1 if tr.team=='blue' else 14.9)
        lc,rc=4.0,13.0
        ld=abs(tr.x-lc)+abs(tx-lc)
        rd=abs(tr.x-rc)+abs(tx-rc)
        bx=lc if ld<=rd else rc
        by=15.0 if tr.y<15.0 else 17.0
        return bx,by
    def _find_target(self,tr):
        opp=self._opp(tr.team);cands=[]
        tgts=getattr(tr,'targets',['Ground'])
        if tgts!=['Buildings']:
            for e in self.players[opp].troops:
                if not e.alive:continue
                et=getattr(e,'transport','Ground')
                if et=='Air' and 'Air' not in tgts:continue
                d=math.sqrt((tr.x-e.x)**2+(tr.y-e.y)**2)
                cands.append((d,e))
        for tw in self.arena.towers:
            if tw.team!=opp or not tw.alive:continue
            if tw.ttype=='king':
                if any(x.team==opp and x.ttype=='princess' and x.alive
                       for x in self.arena.towers):continue
            d=math.sqrt((tr.x-tw.cx)**2+(tr.y-tw.cy)**2)
            cands.append((d,tw))
        for c in getattr(tr,'components',[]):
            cands=c.modify_target(tr,cands,self)
        if not cands:return None,999
        cands.sort(key=lambda x:x[0])
        return cands[0][1],cands[0][0]
    def _do_attack(self,tr,tgt):
        d=int(tr.dmg*getattr(tr,'_dmg_mult',1.0))
        tgt.take_damage(d)
        if hasattr(tgt,'ttype'):
            if tgt.ttype=='king' and not getattr(tgt,'active',False):
                self._king_act(tgt.team)
            if not tgt.alive:self._tower_down(tgt)
        for c in getattr(tr,'components',[]):c.on_attack(tr,tgt,self)
    def _proc_troops(self):
        for tm in ('blue','red'):
            p=self.players[tm]
            for tr in p.troops:
                if not tr.alive:continue
                for c in getattr(tr,'components',[]):c.on_tick(tr,self)
                frz=any(s.kind=='freeze' for s in getattr(tr,'statuses',[]))
                stn=any(s.kind=='stun' for s in getattr(tr,'statuses',[]))
                if stn:
                    tr.cd=tr.hspd
                    tr.statuses=[s for s in tr.statuses if s.kind!='stun']
                spd=0 if frz else tr.spd
                dmg_mult=1.0
                for s in getattr(tr,'statuses',[]):
                    if s.kind=='slow':spd*=s.val
                    if s.kind=='rage':spd*=1+s.val;dmg_mult*=1+s.val
                can_atk=not frz
                tr._dmg_mult=dmg_mult
                tgt,td=self._find_target(tr)
                tr.tgt=tgt
                if not tgt:continue
                if td<=tr.rng:
                    if can_atk:
                        fa=getattr(tr,'first_atk',False)
                        if fa:tr.cd=getattr(tr,'fhspd',tr.hspd);tr.first_atk=False
                        tr.cd=max(0,tr.cd-self.DT)
                        if tr.cd<=0:
                            self._do_attack(tr,tgt);tr.cd=tr.hspd
                else:
                    if spd>0:
                        tx,ty=(tgt.cx,tgt.cy) if hasattr(tgt,'cx') else (tgt.x,tgt.y)
                        wx,wy=self._waypoint(tr,tx,ty)
                        dx=wx-tr.x;dy=wy-tr.y
                        ds=math.sqrt(dx*dx+dy*dy)
                        if ds>0:
                            oy=tr.y
                            tr.x+=dx/ds*spd*self.DT
                            tr.y+=dy/ds*spd*self.DT
                            if 15.0<=tr.y<=17.0 and getattr(tr,'transport','Ground')!='Air':
                                from components import RiverJump
                                hrj=any(isinstance(c,RiverJump) for c in getattr(tr,'components',[]))
                                if not hrj:
                                    ob=(3<=tr.x<=5) or (12<=tr.x<=14)
                                    if not ob:tr.y=oy
    def _proc_statuses(self):
        for tm in ('blue','red'):
            for tr in self.players[tm].troops:
                sl=getattr(tr,'statuses',None)
                if not sl:continue
                for s in sl:s.tick(self.DT)
                tr.statuses=[s for s in sl if not s.expired]
    def _proc_spells(self):
        for sp in self.spells:sp.tick(self.DT,self)
        self.spells=[sp for sp in self.spells if sp.active]
    def _proc_deaths(self):
        for tm in ('blue','red'):
            p=self.players[tm]
            dead=[tr for tr in p.troops if not tr.alive]
            for tr in dead:
                if hasattr(tr,'on_death'):tr.on_death(self)
            p.troops=[tr for tr in p.troops if tr.alive]
    def deploy(self,team,troop):
        self.players[team].troops.append(troop)
    def tick(self):
        if self.ended:return
        self.t+=self.DT
        self._gen_ex()
        qcd=self._qcd()
        for p in self.players.values():
            if p.deck:p.deck.tick(self.DT,qcd)
        self._proc_pending()
        self._proc_towers()
        self._proc_statuses()
        self._proc_spells()
        self._proc_troops()
        self._proc_deaths()
        self._check_phase()
    def run(self,dur):
        e=self.t+dur
        while self.t<e and not self.ended:self.tick()
    def run_to(self,t):
        while self.t<t and not self.ended:self.tick()
    def status(self):
        s=f"T={self.t:.1f}s Phase={self.phase}\n"
        for tm in ('blue','red'):
            p=self.players[tm]
            s+=f"  {tm}: {p.crowns}cr {p.elixir:.1f}ex"
            if p.deck:s+=f" hand={p.deck.hand} nxt={p.deck.nxt}"
            s+="\n"
        for t in self.arena.towers:
            act='*' if getattr(t,'active',True) else ' '
            s+=f"  {t.team} {t.ttype}: {t.hp}/{t.max_hp} {act}\n"
        return s

# ==================== TESTS ====================
def t_phases():
    g=Game()
    g.run_to(119);assert g._erate()==1
    g.run_to(121);assert g._erate()==2
    g.run_to(180.1);assert g.phase=='overtime'
    g.run_to(241);assert g._erate()==3
    g.run_to(300.1);assert g.ended
    return "Phase transitions (1x->2x->OT->3x->tiebreaker)"
def t_elixir():
    g=Game();g.run(10)
    ex=g.players['blue'].elixir
    exp=5.0+10/2.8
    assert abs(ex-exp)<0.5,f"Expected ~{exp:.1f}, got {ex:.1f}"
    return f"Elixir generation ({ex:.1f} at 10s, expected ~{exp:.1f})"
def t_3crown():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    kt.hp=0;kt.alive=False;g._tower_down(kt)
    assert g.winner=='red' and g.players['red'].crowns==3
    return "3-crown king tower win"
def t_crown_lead():
    g=Game()
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=0;pt.alive=False;g._tower_down(pt)
    assert g.players['red'].crowns==1
    g.run_to(180.1)
    assert g.winner=='red'
    return "Crown lead regulation win (red 1-0)"
def t_overtime_sd():
    g=Game()
    g.run_to(180.1);assert g.phase=='overtime'
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=0;pt.alive=False;g._tower_down(pt)
    assert g.winner=='red' and g.ended
    return "Overtime sudden death"
def t_tiebreaker():
    g=Game()
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=pt.max_hp-100
    g.run_to(300.1)
    assert g.ended and g.winner=='red'
    return "Tiebreaker (damaged tower destroyed)"
def t_tiebreaker_draw():
    g=Game()
    g.run_to(300.1)
    assert g.ended and g.winner is None
    return "Tiebreaker draw (all towers equal HP)"
def t_king_act():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    assert not kt.active
    pt=g.arena.get_tower('blue','princess','left')
    pt.hp=0;pt.alive=False;g._tower_down(pt)
    assert kt.active
    return "King activation on princess death"
def t_king_act_dmg():
    g=Game()
    kt=g.arena.get_tower('blue','king')
    assert not kt.active
    for t in g.arena.towers:
        if t.team=='blue' and t.ttype=='princess':
            t.hp=0;t.alive=False
    tr=Dummy('red',8.5,4.0,hp=50000,dmg=100)
    g.deploy('red',tr)
    for _ in range(500):
        g.tick()
        if kt.active:break
    assert kt.active
    return "King activation on direct damage"
def t_chef():
    random.seed(42)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr=Dummy('blue',9.0,10.0,lvl=11,hp=50000,spd=0)
    g.deploy('blue',tr)
    g.run(50)
    assert tr.lvl>=12,f"Expected lvl>=12, got {tr.lvl}"
    return f"Royal Chef pancake (lvl 11->{tr.lvl})"
def t_chef_cooldown():
    random.seed(0)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    t1=Dummy('blue',5.0,10.0,lvl=11,hp=50000,spd=0)
    t2=Dummy('blue',12.0,10.0,lvl=11,hp=50000,spd=0)
    g.deploy('blue',t1);g.deploy('blue',t2)
    g.run(80)
    assert t1.lvl>=12 and t2.lvl>=12,f"Expected both boosted: t1={t1.lvl} t2={t2.lvl}"
    return f"Royal Chef multi-pancake (t1=lvl{t1.lvl}, t2=lvl{t2.lvl})"
def t_chef_multiboost():
    random.seed(7)
    g=Game(p1={'tt_name':'royal_chef','tt_lvl':11})
    tr=Dummy('blue',9.0,10.0,lvl=11,hp=1000,dmg=100,spd=0)
    g.deploy('blue',tr)
    ihp=tr.max_hp;idmg=tr.dmg
    g.run(120)
    assert tr.lvl>=13,f"Expected lvl>=13 after multi-boost, got {tr.lvl}"
    boosts=tr.lvl-11
    exp_hp=ihp;exp_dmg=idmg
    for _ in range(boosts):
        exp_hp=int(exp_hp*1.1)
        exp_dmg=int(exp_dmg*1.1)
    assert tr.max_hp==exp_hp,f"HP mismatch: {tr.max_hp} vs {exp_hp}"
    assert tr.dmg==exp_dmg,f"DMG mismatch: {tr.dmg} vs {exp_dmg}"
    return f"Chef multi-boost (lvl 11->{tr.lvl}, hp {ihp}->{tr.max_hp}, dmg {idmg}->{tr.dmg})"
def t_duchess():
    g=Game(p1={'tt_name':'dagger_duchess','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    dd=lpt.troop;ini=tr.hp
    g.run(4.5)
    dmg=ini-tr.hp
    assert dmg>=dd.dmg*8,f"Expected >=8 daggers ({dd.dmg*8}), got {dmg}"
    return f"Dagger Duchess burst ({dmg} dmg, ~{dmg//dd.dmg} hits)"
def t_duchess_recharge():
    g=Game(p1={'tt_name':'dagger_duchess','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    dd=lpt.troop
    g.run(5)
    assert dd.dag==0
    g.players['red'].troops.clear()
    g.run(8)
    assert dd.dag==dd.MXD,f"Expected full recharge, got {dd.dag}/{dd.MXD}"
    return f"Dagger Duchess passive recharge ({dd.dag}/{dd.MXD})"
def t_cannoneer():
    g=Game(p1={'tt_name':'cannoneer','tt_lvl':11})
    tr=Dummy('red',3.0,13.0,hp=50000)
    g.deploy('red',tr)
    ini=tr.hp
    g.run(0.8)
    assert tr.hp==ini,"Shot fired before 0.8s"
    g.run(0.2)
    assert tr.hp<ini,"No shot by 1.0s"
    d1=ini-tr.hp;hp1=tr.hp
    g.run(2.0)
    assert tr.hp==hp1,"Extra shot between 1.0-3.0s"
    g.run(0.2)
    assert tr.hp<hp1,"No second shot by 3.2s"
    return f"Cannoneer first shot ({d1} dmg at ~0.9s, 2nd at ~3.1s)"
def t_troop_atk():
    g=Game()
    tr=Dummy('red',3.0,8.0,hp=50000,dmg=200,spd=2.0)
    g.deploy('red',tr)
    lpt=g.arena.get_tower('blue','princess','left')
    ini=lpt.hp
    g.run(5)
    assert lpt.hp<ini
    return f"Troop attacks tower ({ini}->{lpt.hp})"
def t_troop_kills_tower():
    g=Game()
    lpt=g.arena.get_tower('blue','princess','left')
    lpt.hp=150
    tr=Dummy('red',3.0,8.0,hp=50000,dmg=200,spd=2.0)
    g.deploy('red',tr)
    g.run(5)
    assert not lpt.alive
    assert g.players['red'].crowns>=1
    kt=g.arena.get_tower('blue','king')
    assert kt.active
    return "Troop destroys tower -> crown + king activation"
def t_deck_cycle():
    random.seed(99)
    dk=Deck(['a','b','c','d','e','f','g','h'])
    h0=list(dk.hand);n0=dk.nxt
    c=dk.hand[0]
    dk.play(c,2.0)
    assert c==dk.q[-1],"Played card not at back of queue"
    assert len(dk.hand)==4,"Hand not refilled"
    assert n0 in dk.hand,"Next card not moved to hand"
    return f"Deck cycle (played {c}, hand={dk.hand}, nxt={dk.nxt})"
def t_deck_4card_return():
    random.seed(99)
    dk=Deck(['a','b','c','d','e','f','g','h'])
    first=dk.hand[0]
    for i in range(4):
        dk.play(dk.hand[0],0)
    assert first==dk.nxt,f"{first} should be nxt after 4 plays, got {dk.nxt}"
    dk.play(dk.hand[0],0)
    assert first in dk.hand,f"{first} should return to hand after 5th play"
    return f"4-card cycle ({first}: nxt after 4, hand after 5)"
def t_deck_queue_cd():
    random.seed(99)
    dk=Deck(['a','b','c','d','e','f','g','h'])
    dk.play(dk.hand[0],2.0)
    assert len(dk.hand)==4
    dk.play(dk.hand[0],2.0)
    assert len(dk.hand)==3,"Hand should drop to 3 (next on cooldown)"
    for _ in range(20):dk.tick(0.1,2.0)
    assert len(dk.hand)==4,"Hand should restore after 2s"
    return "Deck queue cooldown (3->4 after 2s)"
def t_deck_qcd_2x():
    random.seed(99)
    dk=Deck(['a','b','c','d','e','f','g','h'])
    dk.play(dk.hand[0],1.0)
    dk.play(dk.hand[0],1.0)
    assert len(dk.hand)==3
    for _ in range(10):dk.tick(0.1,1.0)
    assert len(dk.hand)==4,"Hand should restore after 1s at 2x"
    return "Queue cooldown scales with elixir rate (1s at 2x)"
def t_play_card_elixir():
    random.seed(99)
    g=Game(p1={'deck':['a','b','c','d','e','f','g','h']})
    c=g.players['blue'].deck.hand[0]
    ini=g.players['blue'].elixir
    ok,_=g.play_card('blue',c,9,10)
    assert ok
    assert abs(g.players['blue'].elixir-(ini-3))<0.01
    return f"Play card deducts elixir ({ini:.0f}->{g.players['blue'].elixir:.0f})"
def t_play_card_no_elixir():
    random.seed(99)
    g=Game(p1={'deck':['a','b','c','d','e','f','g','h']})
    g.players['blue'].elixir=1.0
    c=g.players['blue'].deck.hand[0]
    ok,msg=g.play_card('blue',c,9,10)
    assert not ok and msg=="not enough elixir"
    return "Play card rejected (insufficient elixir)"
def t_play_card_not_in_hand():
    random.seed(99)
    g=Game(p1={'deck':['a','b','c','d','e','f','g','h']})
    hand=g.players['blue'].deck.hand
    nothand=[c for c in 'abcdefgh' if c not in hand][0]
    ok,msg=g.play_card('blue',nothand,9,10)
    assert not ok and msg=="not in hand"
    return f"Play card rejected (not in hand: {nothand})"
def t_deploy_zone_base():
    g=Game()
    assert g._valid_deploy('blue',9,10)
    assert not g._valid_deploy('blue',9,20)
    assert g._valid_deploy('red',9,20)
    assert not g._valid_deploy('red',9,10)
    assert not g._valid_deploy('blue',0,15)
    assert not g._valid_deploy('red',0,16)
    assert g._valid_deploy('blue',3,15)
    assert g._valid_deploy('red',3,16)
    return "Deploy zone (own half + own bridge, no river)"
def t_deploy_zone_pocket():
    g=Game()
    assert not g._valid_deploy('blue',3,20)
    rlp=g.arena.get_tower('red','princess','left')
    rlp.hp=0;rlp.alive=False
    assert g._valid_deploy('blue',3,20)
    assert not g._valid_deploy('blue',14,20)
    rrp=g.arena.get_tower('red','princess','right')
    rrp.hp=0;rrp.alive=False
    assert g._valid_deploy('blue',14,20)
    return "Deploy zone pocket (unlocked per destroyed tower)"
def t_deploy_delay():
    random.seed(99)
    g=Game(p1={'deck':['a','b','c','d','e','f','g','h'],'drag_del':0.5,'drag_std':0})
    c=g.players['blue'].deck.hand[0]
    g.play_card('blue',c,9,10)
    assert len(g.players['blue'].troops)==0
    assert len(g.pending)==1
    g.run(1.4)
    assert len(g.players['blue'].troops)==0,"Spawned too early"
    g.run(0.2)
    assert len(g.players['blue'].troops)==1,"Not spawned after 1.6s"
    return "Deploy delay (0.5s drag + 1.0s deploy = 1.5s)"
def t_drag_pro_vs_casual():
    random.seed(99)
    g1=Game(p1={'deck':['a','b','c','d','e','f','g','h'],'drag_del':0.3,'drag_std':0})
    c=g1.players['blue'].deck.hand[0]
    g1.play_card('blue',c,9,10)
    g1.run(1.2)
    assert len(g1.players['blue'].troops)==0
    g1.run(0.2)
    assert len(g1.players['blue'].troops)==1
    random.seed(99)
    g2=Game(p1={'deck':['a','b','c','d','e','f','g','h'],'drag_del':0.7,'drag_std':0})
    c=g2.players['blue'].deck.hand[0]
    g2.play_card('blue',c,9,10)
    g2.run(1.6)
    assert len(g2.players['blue'].troops)==0
    g2.run(0.2)
    assert len(g2.players['blue'].troops)==1
    return "Drag delay (pro=1.3s total, casual=1.7s total)"
def t_drag_stochastic():
    random.seed(42)
    p=Player('blue',drag_del=0.5,drag_std=0.15)
    delays=[p.sample_drag() for _ in range(100)]
    mn=min(delays);mx=max(delays);avg=sum(delays)/len(delays)
    assert mn!=mx,"All drags identical (not stochastic)"
    assert mn>=0.1,"Drag below 0.1s floor"
    assert abs(avg-0.5)<0.1,f"Mean drag {avg:.3f} too far from 0.5"
    return f"Stochastic drag (n=100, min={mn:.2f} max={mx:.2f} avg={avg:.2f})"
def t_simultaneous_play():
    random.seed(99)
    dk=['a','b','c','d','e','f','g','h']
    g=Game(p1={'deck':dk,'drag_std':0})
    g.players['blue'].elixir=10
    h=g.players['blue'].deck.hand
    c1,c2=h[0],h[1]
    ok1,_=g.play_card('blue',c1,5,10)
    ok2,_=g.play_card('blue',c2,12,10)
    assert ok1 and ok2
    assert len(g.pending)==2
    assert g.players['blue'].elixir==10-3-3
    g.run(2)
    assert len(g.players['blue'].troops)==2
    return f"Simultaneous 2-card play ({c1}+{c2}, 6 elixir)"
def t_deploy_invalid_pos():
    random.seed(99)
    g=Game(p1={'deck':['a','b','c','d','e','f','g','h']})
    c=g.players['blue'].deck.hand[0]
    ok,msg=g.play_card('blue',c,9,20)
    assert not ok and msg=="invalid position"
    return "Deploy rejected (enemy half without pocket)"
def t_elixir_2x_3x():
    g=Game()
    g.run_to(100)
    g.players['blue'].elixir=0
    g.run(10)
    gen_1x=g.players['blue'].elixir
    g.run_to(135)
    g.players['blue'].elixir=0
    g.run(10)
    gen_2x=g.players['blue'].elixir
    g.run_to(250)
    g.players['blue'].elixir=0
    g.run(10)
    gen_3x=g.players['blue'].elixir
    assert abs(gen_2x/gen_1x-2.0)<0.2,f"2x ratio off: {gen_2x/gen_1x:.2f}"
    assert abs(gen_3x/gen_1x-3.0)<0.3,f"3x ratio off: {gen_3x/gen_1x:.2f}"
    return f"Elixir rates (1x={gen_1x:.2f}, 2x={gen_2x:.2f}, 3x={gen_3x:.2f})"

def t_knight_load():
    tr=mk_card('knight',11,'blue',5,10)
    assert tr.hp==1690 and tr.dmg==191
    assert abs(tr.hspd-1.2)<0.01 and abs(tr.fhspd-0.5)<0.01
    assert abs(tr.rng-1.2)<0.01 and abs(tr.spd-1.0)<0.01
    assert tr.targets==['Ground']
    return f"Knight load (hp={tr.hp} dmg={tr.dmg})"
def t_knight_v_troop():
    g=Game()
    k1=mk_card('knight',11,'blue',9,14)
    k2=mk_card('knight',11,'red',9,17)
    g.deploy('blue',k1);g.deploy('red',k2)
    g.run(20)
    assert not k1.alive or not k2.alive
    return f"Knight vs Knight (k1={k1.hp} k2={k2.hp})"
def t_knight_v_tower():
    g=Game()
    k=mk_card('knight',11,'red',3,8)
    g.deploy('red',k)
    lpt=g.arena.get_tower('blue','princess','left')
    ini=lpt.hp
    g.run(15)
    assert lpt.hp<ini
    return f"Knight vs tower ({ini}->{lpt.hp})"
def t_archers_spawn():
    r=mk_card('archers',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==2
    assert all(a.hp==304 for a in r)
    return f"Archers spawn 2 (hp={r[0].hp})"
def t_archers_air():
    g=Game()
    arcs=mk_card('archers',11,'blue',9,10)
    for a in arcs:g.deploy('blue',a)
    bd=mk_card('baby_dragon',11,'red',9,12)
    g.deploy('red',bd)
    ini=bd.hp
    g.run(5)
    assert bd.hp<ini
    return f"Archers target air ({ini}->{bd.hp})"
def t_ground_cant_air():
    g=Game()
    k=mk_card('knight',11,'blue',9,14)
    bd=mk_card('baby_dragon',11,'red',9,15)
    g.deploy('blue',k);g.deploy('red',bd)
    ini=bd.hp
    g.run(3)
    assert bd.hp==ini
    return "Ground-only can't target air"
def t_musk_fhspd():
    g=Game()
    m=mk_card('musketeer',11,'blue',9,10)
    g.deploy('blue',m)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(0.6)
    assert d.hp==ini,"Musketeer shot before fhspd"
    g.run(0.2)
    assert d.hp<ini,"No shot by 0.8s"
    return "Musketeer first-hit (fhspd=0.7s)"
def t_bdrag_splash():
    g=Game()
    bd=mk_card('baby_dragon',11,'blue',9,10)
    g.deploy('blue',bd)
    d1=Dummy('red',9,20,hp=50000,spd=0)
    d2=Dummy('red',10,20,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(20)
    assert d1.hp<50000 and d2.hp<50000
    return f"Baby Dragon splash (d1={50000-d1.hp} d2={50000-d2.hp} dmg)"
def t_bdrag_air():
    bd=mk_card('baby_dragon',11,'blue',9,10)
    assert bd.transport=='Air'
    return "Baby Dragon is Air"
def t_fb_aoe():
    g=Game()
    d1=Dummy('red',9,10,hp=1000,spd=0)
    d2=Dummy('red',10,10,hp=1000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    fb=mk_card('fireball',11,'blue',9.5,10)
    fb.apply(g)
    assert d1.hp==1000-689 and d2.hp==1000-689
    return f"Fireball AOE (d1={d1.hp} d2={d2.hp})"
def t_fb_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    fb=mk_card('fireball',11,'blue',rpt.cx,rpt.cy)
    fb.apply(g)
    assert rpt.hp==ini-103
    return f"Fireball crown tower ({ini}->{rpt.hp})"
def t_fb_kb():
    g=Game()
    d=Dummy('red',9,10,hp=5000,spd=0)
    g.deploy('red',d)
    ox=d.x
    fb=mk_card('fireball',11,'blue',8,10)
    fb.apply(g)
    assert d.x>ox
    return f"Fireball knockback ({ox}->{d.x:.2f})"
def t_hog_ignores():
    g=Game()
    hog=mk_card('hog_rider',11,'blue',9,14)
    g.deploy('blue',hog)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini
    return "Hog ignores troops"
def t_hog_tower():
    g=Game()
    hog=mk_card('hog_rider',11,'blue',3,14)
    g.deploy('blue',hog)
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    g.run(30)
    assert rpt.hp<ini
    return f"Hog targets tower ({ini}->{rpt.hp})"
def t_hog_jump():
    from components import RiverJump
    hog=mk_card('hog_rider',11,'blue',9,14)
    assert any(isinstance(c,RiverJump) for c in hog.components)
    return "Hog has RiverJump"
def t_skarmy_cnt():
    random.seed(42)
    r=mk_card('skeleton_army',11,'blue',9,10)
    assert isinstance(r,list) and len(r)==15
    assert r[0].hp==82
    return f"Skarmy 15 units (hp={r[0].hp})"
def t_skarmy_pos():
    random.seed(42)
    r=mk_card('skeleton_army',11,'blue',9,10)
    xs=[t.x for t in r]
    assert min(xs)!=max(xs)
    return "Skarmy random positions"
def t_skarmy_fb():
    g=Game()
    random.seed(42)
    sk=mk_card('skeleton_army',11,'red',9,10)
    for s in sk:g.deploy('red',s)
    fb=mk_card('fireball',11,'blue',9,10)
    fb.apply(g)
    alive=[s for s in sk if s.alive]
    assert len(alive)==0,f"{len(alive)} survived"
    return "Fireball wipes skarmy"
def t_freeze_stop():
    g=Game()
    d=Dummy('red',3,13,hp=50000,spd=2.0)
    g.deploy('red',d)
    ox,oy=d.x,d.y
    fz=mk_card('freeze',11,'blue',3,13)
    fz.apply(g)
    g.run(2)
    assert abs(d.x-ox)<0.01 and abs(d.y-oy)<0.01
    return "Freeze stops movement"
def t_freeze_dur():
    g=Game()
    d=Dummy('red',3,13,hp=50000,spd=2.0)
    g.deploy('red',d)
    fz=mk_card('freeze',11,'blue',3,13)
    fz.apply(g)
    oy=d.y
    g.run(3.9)
    assert abs(d.y-oy)<0.01,"Moved while frozen"
    g.run(0.3)
    assert abs(d.y-oy)>0.01,"Still frozen after 4.2s"
    return "Freeze wears off after 4s"
def t_mpekka_load():
    mp=mk_card('mini_pekka',11,'blue',5,10)
    assert mp.hp==1441 and mp.dmg==755
    assert abs(mp.hspd-1.6)<0.01 and abs(mp.fhspd-0.5)<0.01
    assert abs(mp.rng-0.8)<0.01 and abs(mp.spd-1.5)<0.01
    return f"Mini PEKKA load (hp={mp.hp} dmg={mp.dmg})"
def t_mpekka_kills_knight():
    g=Game()
    mp=mk_card('mini_pekka',11,'blue',9,14)
    k=mk_card('knight',11,'red',9,17)
    g.deploy('blue',mp);g.deploy('red',k)
    g.run(10)
    assert not k.alive
    assert mp.alive
    return f"Mini PEKKA kills Knight (mp_hp={mp.hp})"
def t_valk_splash():
    g=Game()
    v=mk_card('valkyrie',11,'blue',9,10)
    g.deploy('blue',v)
    d1=Dummy('red',9,11,hp=5000,spd=0)
    d2=Dummy('red',10,11,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Valkyrie splash (d1={5000-d1.hp} d2={5000-d2.hp} dmg)"
def t_valk_tanky():
    v=mk_card('valkyrie',11,'blue',5,10)
    assert v.hp==2336
    return f"Valkyrie tanky (hp={v.hp})"
def t_zap_aoe():
    g=Game()
    d1=Dummy('red',9,10,hp=1000,spd=0)
    d2=Dummy('red',10,10,hp=1000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    z=mk_card('zap',11,'blue',9.5,10)
    z.apply(g)
    assert d1.hp==1000-191 and d2.hp==1000-191
    return f"Zap AOE (d1={d1.hp} d2={d2.hp})"
def t_zap_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    z=mk_card('zap',11,'blue',rpt.cx,rpt.cy)
    z.apply(g)
    assert rpt.hp==ini-58
    return f"Zap crown tower ({ini}->{rpt.hp})"
def t_zap_stun():
    g=Game()
    d=Dummy('red',3,8,hp=50000,spd=0,dmg=500,hspd=1.0)
    g.deploy('red',d)
    lpt=g.arena.get_tower('blue','princess','left')
    g.run(1.5)
    hp1=lpt.hp
    z=mk_card('zap',11,'blue',d.x,d.y)
    z.apply(g)
    g.run(0.7)
    assert lpt.hp==hp1,"Stun should have reset attack cycle"
    return "Zap stun resets attack cycle"
def t_prince_charge():
    g=Game()
    p=mk_card('prince',11,'blue',9,10)
    g.deploy('blue',p)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(5)
    dmg=50000-d.hp
    assert dmg>=786,f"Expected >=786 charge dmg, got {dmg}"
    return f"Prince charge ({dmg} dmg in 5s)"
def t_prince_charge_reset():
    from components import Charge
    from status import Status
    p=mk_card('prince',11,'blue',9,10)
    ch=[c for c in p.components if isinstance(c,Charge)][0]
    g=Game()
    g.deploy('blue',p)
    d=Dummy('red',9,25,hp=50000,spd=0)
    g.deploy('red',d)
    g.run(2)
    assert ch.moved>0
    p.statuses.append(Status('freeze',2.0))
    g.run(0.1)
    assert ch.moved==0,"Charge not reset by freeze"
    return "Prince charge resets on freeze"
def t_prince_jump():
    from components import RiverJump
    p=mk_card('prince',11,'blue',9,14)
    assert any(isinstance(c,RiverJump) for c in p.components)
    return "Prince has RiverJump"
def t_gbarrel_spawn():
    random.seed(42)
    gb=mk_card('goblin_barrel',11,'blue',14,24)
    g=Game()
    gb.apply(g)
    gobs=g.players['blue'].troops
    assert len(gobs)==3,f"Expected 3 goblins, got {len(gobs)}"
    assert all(t.hp==204 for t in gobs)
    return f"Goblin Barrel spawns 3 (hp={gobs[0].hp})"
def t_gbarrel_atk():
    g=Game()
    random.seed(42)
    gb=mk_card('goblin_barrel',11,'blue',14,24)
    gb.apply(g)
    rpt=g.arena.get_tower('red','princess','right')
    ini=rpt.hp
    g.run(5)
    assert rpt.hp<ini
    return f"Goblins attack tower ({ini}->{rpt.hp})"
def t_witch_splash():
    g=Game()
    w=mk_card('witch',11,'blue',9,10)
    g.deploy('blue',w)
    d1=Dummy('red',9,15,hp=5000,spd=0)
    d2=Dummy('red',10,15,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(5)
    assert d1.hp<5000 and d2.hp<5000
    return f"Witch splash (d1={5000-d1.hp} d2={5000-d2.hp} dmg)"
def t_witch_spawn():
    g=Game()
    random.seed(42)
    w=mk_card('witch',11,'blue',9,10)
    g.deploy('blue',w)
    ini=len(g.players['blue'].troops)
    g.run(2)
    n=len(g.players['blue'].troops)-ini
    assert n>=4,f"Expected >=4 spawns, got {n}"
    return f"Witch spawns skeletons (+{n})"
def t_golem_building():
    g=Game()
    go=mk_card('golem',11,'blue',9,14)
    g.deploy('blue',go)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini,"Golem should ignore troops"
    return "Golem targets buildings only"
def t_golem_death_spawn():
    g=Game()
    random.seed(42)
    go=mk_card('golem',11,'blue',9,14)
    g.deploy('blue',go)
    go.hp=1
    d=Dummy('red',9,15,hp=50000,dmg=100,spd=0,hspd=0.5)
    g.deploy('red',d)
    g.run(1)
    assert not go.alive,"Golem should be dead"
    gm=[t for t in g.players['blue'].troops if t.alive]
    assert len(gm)==2,f"Expected 2 golemites, got {len(gm)}"
    assert gm[0].hp==1540
    return f"Golem death spawns 2 golemites (hp={gm[0].hp})"
def t_golem_death_dmg():
    g=Game()
    go=mk_card('golem',11,'blue',9,14)
    g.deploy('blue',go)
    go.hp=1
    d1=Dummy('red',9,15,hp=50000,dmg=500,spd=0,hspd=0.5)
    d2=Dummy('red',10,14,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(2)
    assert not go.alive
    assert d2.hp<50000,f"Death damage not applied"
    assert 50000-d2.hp==312,f"Expected 312 death dmg, got {50000-d2.hp}"
    return f"Golem death damage ({50000-d2.hp} to nearby)"
def t_ewiz_szap():
    g=Game()
    d=Dummy('red',9,12.5,hp=5000,spd=0)
    g.deploy('red',d)
    ew=mk_card('electro_wizard',11,'blue',9,10)
    g.deploy('blue',ew)
    ini=d.hp
    g.run(0.1)
    assert d.hp==ini-192,f"Expected 192 spawn zap dmg, got {ini-d.hp}"
    return f"E-wiz spawn zap ({ini-d.hp} dmg + stun reset)"
def t_ewiz_dual():
    g=Game()
    ew=mk_card('electro_wizard',11,'blue',9,10)
    d1=Dummy('red',9,14,hp=5000,spd=0)
    d2=Dummy('red',10,14,hp=5000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.deploy('blue',ew)
    g.run(3)
    assert d1.hp<5000 and d2.hp<5000
    return f"E-wiz dual target (d1={5000-d1.hp} d2={5000-d2.hp} dmg)"
def t_ewiz_stun_atk():
    g=Game()
    ew=mk_card('electro_wizard',11,'blue',9,10)
    d=Dummy('red',9,14,hp=50000,spd=0,hspd=0.5,dmg=100)
    tgt=Dummy('blue',9,15,hp=50000,spd=0)
    g.deploy('red',d);g.deploy('blue',ew);g.deploy('blue',tgt)
    g.run(1.5)
    dmg=50000-tgt.hp
    assert dmg<1500,"Stun should delay attacks"
    return f"E-wiz stun delays attack ({dmg} dmg)"
def t_giant_load():
    gi=mk_card('giant',11,'blue',5,10)
    assert gi.hp==4100 and gi.dmg==251
    assert gi.targets==['Buildings']
    return f"Giant load (hp={gi.hp} dmg={gi.dmg})"
def t_giant_targets_buildings():
    g=Game()
    gi=mk_card('giant',11,'blue',9,14)
    g.deploy('blue',gi)
    d=Dummy('red',9,15,hp=50000,spd=0)
    g.deploy('red',d)
    ini=d.hp
    g.run(5)
    assert d.hp==ini,"Giant should ignore troops"
    return "Giant targets buildings only"
def t_balloon_air():
    b=mk_card('balloon',11,'blue',9,10)
    assert b.transport=='Air'
    assert b.targets==['Buildings']
    return f"Balloon is Air, targets buildings (hp={b.hp})"
def t_balloon_death_dmg():
    g=Game()
    b=mk_card('balloon',11,'blue',9,14)
    g.deploy('blue',b)
    d=Dummy('red',9,14.5,hp=50000,spd=0)
    g.deploy('red',d)
    b.hp=0;b.alive=False;b.on_death(g)
    assert 50000-d.hp==160,f"Expected 160 death dmg, got {50000-d.hp}"
    return f"Balloon death damage ({50000-d.hp})"
def t_lava_air():
    lh=mk_card('lava_hound',11,'blue',9,10)
    assert lh.transport=='Air'
    assert lh.targets==['Buildings']
    return f"Lava Hound is Air (hp={lh.hp})"
def t_lava_death_spawn():
    g=Game()
    random.seed(42)
    lh=mk_card('lava_hound',11,'blue',9,14)
    g.deploy('blue',lh)
    lh.hp=0;lh.alive=False;lh.on_death(g)
    pups=[t for t in g.players['blue'].troops if t.alive and t is not lh]
    assert len(pups)==6,f"Expected 6 pups, got {len(pups)}"
    assert pups[0].hp==216
    assert pups[0].transport=='Air'
    assert 'Air' in pups[0].targets and 'Ground' in pups[0].targets
    return f"Lava Hound death spawns 6 Air pups (hp={pups[0].hp})"
def t_darkprince_shield():
    dp=mk_card('dark_prince',11,'blue',9,10)
    assert dp.shield_hp==240
    dp.take_damage(500)
    assert dp.shield_hp==0
    assert dp.hp==1200,"Excess should be blocked"
    dp.take_damage(100)
    assert dp.hp==1100
    return f"Dark Prince shield absorbs+blocks (hp={dp.hp})"
def t_darkprince_charge_splash():
    from components import Charge,SplashAttack
    dp=mk_card('dark_prince',11,'blue',9,10)
    assert any(isinstance(c,Charge) for c in dp.components)
    assert any(isinstance(c,SplashAttack) for c in dp.components)
    assert dp.charge_dmg==496
    return "Dark Prince has charge+splash"
def t_idrag_ramp():
    g=Game()
    idrag=mk_card('inferno_dragon',11,'blue',9,10)
    g.deploy('blue',idrag)
    d=Dummy('red',9,13,hp=50000,spd=0)
    g.deploy('red',d)
    assert idrag.dmg==36
    g.run(1.0)
    assert idrag.dmg==36,"Stage 1 in first 2s"
    g.run(2.0)
    assert idrag.dmg==121,f"Expected stage 2, got {idrag.dmg}"
    g.run(2.0)
    assert idrag.dmg==423,f"Expected stage 3, got {idrag.dmg}"
    return f"Inferno Dragon ramp (36->121->423)"
def t_idrag_reset():
    g=Game()
    idrag=mk_card('inferno_dragon',11,'blue',9,10)
    g.deploy('blue',idrag)
    d1=Dummy('red',9,13,hp=50000,spd=0)
    d2=Dummy('red',12,13,hp=50000,spd=0)
    g.deploy('red',d1);g.deploy('red',d2)
    g.run(3.0)
    assert idrag.dmg==121,f"Expected stage 2 after 3s, got {idrag.dmg}"
    d1.alive=False
    g.players['red'].troops=[t for t in g.players['red'].troops if t.alive]
    g.run(0.2)
    assert idrag.dmg==36,f"Expected reset to 36, got {idrag.dmg}"
    return "Inferno Dragon resets on retarget"
def t_lumberjack_speed():
    lj=mk_card('lumberjack',11,'blue',9,10)
    assert abs(lj.spd-2.0)<0.01
    return f"Lumberjack speed={lj.spd}"
def t_lumberjack_rage():
    g=Game()
    lj=mk_card('lumberjack',11,'blue',9,10)
    ally=Dummy('blue',9,10.5,hp=50000,spd=1.0,dmg=100)
    g.deploy('blue',lj);g.deploy('blue',ally)
    lj.hp=0;lj.alive=False;lj.on_death(g)
    has_rage=any(s.kind=='rage' for s in ally.statuses)
    assert has_rage,"Ally should have rage status"
    rs=[s for s in ally.statuses if s.kind=='rage'][0]
    assert abs(rs.val-0.3)<0.01
    return "Lumberjack drops rage on death"
def t_poison_dot():
    g=Game()
    d=Dummy('red',9,25,hp=5000,spd=0)
    g.deploy('red',d)
    p=mk_card('poison',11,'blue',9,25)
    p.apply(g);g.spells.append(p)
    g.run(8.5)
    dmg=5000-d.hp
    assert abs(dmg-864)<50,f"Expected ~864 total dmg, got {dmg}"
    return f"Poison DoT ({dmg} over 8s)"
def t_poison_slow():
    g=Game()
    d=Dummy('red',9,10,hp=50000,spd=2.0)
    g.deploy('red',d)
    p=mk_card('poison',11,'blue',9,10)
    p.apply(g);g.spells.append(p)
    g.run(1.5)
    has_slow=any(s.kind=='slow' for s in d.statuses)
    assert has_slow,"Poison should slow"
    return "Poison applies slow"
def t_poison_ct():
    g=Game()
    rpt=g.arena.get_tower('red','princess','left')
    ini=rpt.hp
    p=mk_card('poison',11,'blue',rpt.cx,rpt.cy)
    p.apply(g);g.spells.append(p)
    g.run(8.5)
    dmg=ini-rpt.hp
    assert abs(dmg-256)<50,f"Expected ~256 CT dmg, got {dmg}"
    return f"Poison crown tower ({dmg} dmg)"
def t_log_ground():
    g=Game()
    d1=Dummy('red',9,12,hp=1000,spd=0)
    g.deploy('red',d1)
    log=mk_card('the_log',11,'blue',9,10)
    log.apply(g)
    assert d1.hp==1000-269,f"Expected 269 dmg, got {1000-d1.hp}"
    return f"Log damages ground (hp={d1.hp})"
def t_log_no_air():
    g=Game()
    d=Dummy('red',9,12,hp=1000,spd=0)
    d.transport='Air'
    g.deploy('red',d)
    log=mk_card('the_log',11,'blue',9,10)
    log.apply(g)
    assert d.hp==1000,"Log should not hit air"
    return "Log skips air troops"
def t_log_pushback():
    g=Game()
    d=Dummy('red',9,12,hp=5000,spd=0)
    g.deploy('red',d)
    oy=d.y
    log=mk_card('the_log',11,'blue',9,10)
    log.apply(g)
    assert abs(d.y-oy-0.7)<0.01,f"Expected +0.7 pushback, got {d.y-oy}"
    return f"Log pushback ({d.y-oy:.1f} tiles)"
if __name__=="__main__":
    tests=[
        t_phases,t_elixir,t_elixir_2x_3x,
        t_3crown,t_crown_lead,t_overtime_sd,
        t_tiebreaker,t_tiebreaker_draw,
        t_king_act,t_king_act_dmg,
        t_chef,t_chef_cooldown,t_chef_multiboost,
        t_duchess,t_duchess_recharge,t_cannoneer,
        t_troop_atk,t_troop_kills_tower,
        t_deck_cycle,t_deck_4card_return,t_deck_queue_cd,t_deck_qcd_2x,
        t_play_card_elixir,t_play_card_no_elixir,t_play_card_not_in_hand,
        t_deploy_zone_base,t_deploy_zone_pocket,
        t_deploy_delay,t_drag_pro_vs_casual,t_drag_stochastic,
        t_simultaneous_play,t_deploy_invalid_pos,
        t_knight_load,t_knight_v_troop,t_knight_v_tower,
        t_archers_spawn,t_archers_air,t_ground_cant_air,
        t_musk_fhspd,
        t_bdrag_splash,t_bdrag_air,
        t_fb_aoe,t_fb_ct,t_fb_kb,
        t_hog_ignores,t_hog_tower,t_hog_jump,
        t_skarmy_cnt,t_skarmy_pos,t_skarmy_fb,
        t_freeze_stop,t_freeze_dur,
        t_mpekka_load,t_mpekka_kills_knight,
        t_valk_splash,t_valk_tanky,
        t_zap_aoe,t_zap_ct,t_zap_stun,
        t_prince_charge,t_prince_charge_reset,t_prince_jump,
        t_gbarrel_spawn,t_gbarrel_atk,
        t_witch_splash,t_witch_spawn,
        t_golem_building,t_golem_death_spawn,t_golem_death_dmg,
        t_ewiz_szap,t_ewiz_dual,t_ewiz_stun_atk,
        t_giant_load,t_giant_targets_buildings,
        t_balloon_air,t_balloon_death_dmg,
        t_lava_air,t_lava_death_spawn,
        t_darkprince_shield,t_darkprince_charge_splash,
        t_idrag_ramp,t_idrag_reset,
        t_lumberjack_speed,t_lumberjack_rage,
        t_poison_dot,t_poison_slow,t_poison_ct,
        t_log_ground,t_log_no_air,t_log_pushback,
    ]
    print("=== Clash Royale Simulator Tests ===\n")
    p=0;f=0
    for test in tests:
        try:
            msg=test();print(f"  PASS: {msg}");p+=1
        except AssertionError as e:
            print(f"  FAIL: {test.__name__}: {e}");f+=1
        except Exception as e:
            print(f"  ERR:  {test.__name__}: {type(e).__name__}: {e}");f+=1
    print(f"\n{p}/{p+f} tests passed")
