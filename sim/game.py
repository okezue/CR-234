import math
import random
from sim.arena import Arena
from sim.towers import create as mk_tt,king
from sim.units import Status,hidden,has
from sim.fx import SplashAttack,RiverJump,DualTarget,BannerBrigade
from sim.path import Pathfinder
from sim.cards import create as mk_card,card

MAX_LEVEL=16
_CC={}
def card_info(name):
    if name not in _CC:
        try:c=card(name)
        except KeyError:c=None
        if c:
            # burrowing cards spend the deploy time underground in the arena instead of in the pending queue
            _CC[name]={'name':name,'cost':c['cost'] or 0,'deploy':0 if c['kind']=='spell' or 'burrow' in c['skills'] else (c['deployTime'] or 1.0),
                       'deploy_anywhere':c['kind']=='spell' or c['placement']=='any','rarity':c['rarity']}
        else:
            _CC[name]={'name':name,'cost':3,'deploy':1.0,'deploy_anywhere':False,'rarity':'common'}
    return _CC[name]

_NO_START={'mirror','elixir_collector'}
_HERO_RARITY={'champion','hero'}
MAX_HERO_SLOTS=2
def validate_deck(cards,heroes=None,evolutions=None):
    assert len(cards)==8,"Deck must have exactly 8 cards"
    hc=sum(1 for c in cards if card_info(c).get('rarity','') in _HERO_RARITY)
    assert hc<=MAX_HERO_SLOTS,f"Max {MAX_HERO_SLOTS} hero/champion slots, got {hc}"
    heroes=set(heroes or [])
    evolutions=set(evolutions or [])
    overlap=heroes&evolutions
    assert not overlap,f"Same card cannot be hero and evolution: {overlap}"
    return True
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
        for ns in _NO_START:
            if ns in self.hand:
                self.hand.remove(ns)
                self.q.append(self.nxt)
                self.nxt=ns
                self.nxt_cd=0
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
    def __init__(self,team,card,x,y,rem,evolved=False,hero=False,deploy=0):
        self.team=team;self.card=card
        self.x=x;self.y=y;self.rem=rem;self.deploy=deploy
        self.evolved=evolved;self.hero=hero
class PendingAbility:
    def __init__(self,team,troop,ability,rem,is_banner=False):
        self.team=team;self.troop=troop;self.ability=ability
        self.rem=rem;self.is_banner=is_banner

class Projectile:
    def __init__(self,team,x,y,spd,tgt,tx,ty,hit,homing):
        self.team=team;self.x=x;self.y=y;self.spd=spd;self.tgt=tgt
        self.tx=tx;self.ty=ty;self.hit=hit;self.homing=homing;self.alive=True
    def tick(self,dt,g):
        if self.homing and self.tgt is not None:self.tx,self.ty=g._pos(self.tgt)
        dx=self.tx-self.x;dy=self.ty-self.y;d=math.hypot(dx,dy);st=self.spd*dt
        if d<=st:
            self.x,self.y=self.tx,self.ty;self.alive=False;self.hit(g,self)
        else:
            self.x+=dx/d*st;self.y+=dy/d*st

class Player:
    def __init__(self,team,king_lvl=11,tt_name='tower_princess',tt_lvl=None,
                 deck=None,drag_del=0.5,drag_std=None,ability_del=0.15,ability_std=None,
                 card_levels=None):
        self.team=team;self.king_lvl=king_lvl
        self.tt_name=tt_name;self.tt_lvl=tt_lvl or king_lvl
        self.card_levels=card_levels or {}
        self.elixir=5.0;self.max_ex=10.0
        self.crowns=0;self.troops=[]
        self.drag_del=drag_del
        self.drag_std=drag_std if drag_std is not None else drag_del*0.2
        self.ability_del=ability_del
        self.ability_std=ability_std if ability_std is not None else ability_del*0.33
        self.deck=Deck(deck) if deck else None
        self.last_card=None
        self.active_champ=None
        self.champ_queue=[]
        self.pending_abilities=[]
    def _register_champ(self,tr):
        if not getattr(tr,'ability',None):return
        if tr.hp==1 and tr.max_hp==1:return
        if self.active_champ is None:self.active_champ=tr
        else:self.champ_queue.append(tr)
    def _on_champ_death(self,tr):
        if tr is not self.active_champ:
            if tr in self.champ_queue:self.champ_queue.remove(tr)
            return
        ab=getattr(tr,'ability',None)
        if ab and ab.casting:
            self.elixir=min(self.max_ex,self.elixir+ab.cost)
            ab.casting=False;ab.cast_timer=0
        self.active_champ=self.champ_queue.pop(0) if self.champ_queue else None
    def sample_drag(self):
        if self.drag_del==0:return 0
        return max(0.01,random.gauss(self.drag_del,self.drag_std))
    def sample_ability_del(self):
        return max(0.05,random.gauss(self.ability_del,self.ability_std))

class Replay:
    def __init__(self):
        self.snaps=[]
    def snap(self,g):
        prev_log_len=getattr(self,'_pll',0)
        evts=g.log[prev_log_len:]
        self._pll=len(g.log)
        s={'t':g.t,'phase':g.phase,'winner':g.winner,'events':list(evts)}
        for tm in ('blue','red'):
            p=g.players[tm]
            pd={'crowns':p.crowns,'elixir':round(p.elixir,2)}
            if p.deck:
                pd['hand']=list(p.deck.hand)
                pd['nxt']=p.deck.nxt
                pd['nxt_cd']=round(p.deck.nxt_cd,2)
            s[tm]=pd
        tw=[]
        for t in g.arena.towers:
            tw.append({'team':t.team,'type':t.ttype,'hp':t.hp,'max_hp':t.max_hp,
                       'alive':t.alive,'active':t.active})
        s['towers']=tw
        tr=[]
        for tm in ('blue','red'):
            for u in g.players[tm].troops:
                tr.append({'id':u.id,'name':getattr(u,'name',''),'team':u.team,
                           'x':round(u.x,1),'y':round(u.y,1),
                           'hp':u.hp,'max_hp':u.max_hp,'alive':u.alive,
                           'transport':getattr(u,'transport','Ground'),
                           'is_building':getattr(u,'is_building',False)})
        s['troops']=tr
        sp=[]
        for sl in g.spells:
            sp.append({'name':getattr(sl,'name',''),'x':round(sl.x,1),
                       'y':round(sl.y,1),'active':sl.active,
                       'team':getattr(sl,'team',''),
                       'radius':getattr(sl,'radius',2.5)})
        s['spells']=sp
        self.snaps.append(s)
    def at(self,t):
        if not self.snaps:return None
        best=min(self.snaps,key=lambda s:abs(s['t']-t))
        return best
    def events(self,t1=0,t2=999):
        out=[]
        for s in self.snaps:
            if t1<=s['t']<=t2:out.extend(s['events'])
        return out
    def dump(self,t):
        s=self.at(t)
        if not s:return ''
        ln=[]
        ln.append(f"=== T={s['t']:.1f}s [{s['phase']}] ===")
        for tm in ('blue','red'):
            p=s[tm]
            h=p.get('hand',[])
            nxt=p.get('nxt','?')
            ncd=p.get('nxt_cd',0)
            ln.append(f"{tm.capitalize():5s}: {p['crowns']}cr {p['elixir']:.1f}ex hand={h} nxt={nxt}({ncd:.1f}s)")
        ln.append("Towers:")
        ab={'blue':'b','red':'r'}
        for tw in s['towers']:
            pfx=ab[tw['team']]
            tp='K' if tw['type']=='king' else ('PL' if 'left' in str(tw.get('side','')) else 'P')
            act='*' if tw.get('active') else ' '
            ln.append(f"  {pfx}{tp} {tw['hp']}/{tw['max_hp']}{act}")
        if s['troops']:
            ln.append(f"Troops ({len(s['troops'])}):")
            for u in s['troops']:
                ln.append(f"  [{u['team']}] {u['name']} #{u['id']} ({u['x']},{u['y']}) hp={u['hp']}/{u['max_hp']}")
        if s['events']:
            ln.append("Events: "+'; '.join(s['events']))
        return '\n'.join(ln)
    def summary(self):
        ln=[]
        for s in self.snaps:
            if s['events']:
                for e in s['events']:
                    ln.append(f"T={s['t']:.1f}s {e}")
        return '\n'.join(ln)
class Game:
    REG=180.0;OT=120.0;END=REG+OT;EBASE=2.8;DT=0.05
    def __init__(self,p1=None,p2=None):
        self.arena=Arena();self.t=0
        self.phase='regulation';self.winner=None
        self.ended=False;self.log=[];self.pending=[];self.pending_ab=[];self.spells=[];self.projs=[]
        self.replay=Replay()
        self._pf=Pathfinder(self.arena)
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
                    t.hp=tt.hp;t.max_hp=tt.hp;t.troop=tt;t.rng=tt.RNG;t.spd=tt.spd
                elif t.ttype=='king':
                    k=king(p.king_lvl)
                    t.hp=t.max_hp=k['hp'];t.dmg=k['dmg'];t.spd=k['spd'];t.rng=k['rng'];t.proj_spd=k['projSpeed']
    def _erate(self):
        if self.t<120:return 1
        if self.t<240:return 2
        return 3
    def _qcd(self):return 2.0/self._erate()
    def _opp(self,t):return 'red' if t=='blue' else 'blue'
    def _pos(self,u):return (u.cx,u.cy) if hasattr(u,'cx') else (u.x,u.y)
    def _dist(self,tr,u):
        # range is edge to edge: both collision radii add to the reach (Giant Skeleton range 2 -> 0.8 with collision 1.2 kept the same reach)
        r=getattr(tr,'collision_r',0)
        if hasattr(u,'cx'):return max(0.0,u.dist(tr.x,tr.y)-r)
        return max(0.0,math.hypot(tr.x-u.x,tr.y-u.y)-getattr(u,'collision_r',0)-r)
    def _gen_ex(self):
        a=self.DT*self._erate()/self.EBASE
        for p in self.players.values():
            p.elixir=min(p.max_ex,p.elixir+a)
    def _king_act(self,team):
        if self.arena.get_tower(team,'king').activate():self.log.append(f"[{self.t:.1f}] {team} king activated!")
    def _tower_down(self,tower):
        if getattr(tower,'down',False):return
        tower.down=True
        opp=self._opp(tower.team)
        self.players[opp].crowns+=1
        self._pf.rebuild_tower_grid()
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
        for p in self.players.values():p.troops=[]
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
        if self.t>=self.END and self.phase=='overtime':
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
        if self.arena.blocked(x,y):return False
        cx,cy=x+0.5,y+0.5
        for t in self.arena.towers:
            if t.team==team or not t.alive:continue
            # NoDeploySizeW/H from buildings.csv (king 18x16, princess 11x21 tiles) centred on the tower
            hw,hh=(9.0,8.0) if t.ttype=='king' else (5.5,10.5)
            if abs(cx-t.cx)<hw and abs(cy-t.cy)<hh:return False
        return True
    def play_card(self,team,card,x,y,evolved=False,hero=False):
        p=self.players[team]
        if not p.deck:return False,"no deck"
        if not p.deck.can_play(card):return False,"not in hand"
        ix,iy=int(x),int(y);x,y=ix+0.5,iy+0.5
        if card=='mirror':
            if not p.last_card:return False,"no card to mirror"
            lci=card_info(p.last_card)
            mc=min(lci['cost']+1,10)
            if p.elixir<mc:return False,"not enough elixir"
            if not lci.get('deploy_anywhere') and not self._valid_deploy(team,ix,iy):return False,"invalid position"
            p.elixir-=mc
            p.deck.play(card,self._qcd())
            drag=p.sample_drag()
            self.pending.append(Pending(team,'mirror:'+p.last_card,x,y,drag,evolved,hero,lci['deploy']))
            self.log.append(f"[{self.t:.1f}] {team} plays mirror({p.last_card}) at ({ix},{iy}) drag={drag:.2f}s deploy={lci['deploy']:.2f}s")
            return True,"ok"
        ci=card_info(card)
        if p.elixir<ci['cost']:return False,"not enough elixir"
        if not ci.get('deploy_anywhere') and not self._valid_deploy(team,ix,iy):return False,"invalid position"
        p.elixir-=ci['cost']
        p.deck.play(card,self._qcd())
        drag=p.sample_drag()
        self.pending.append(Pending(team,card,x,y,drag,evolved,hero,ci['deploy']))
        self.log.append(f"[{self.t:.1f}] {team} plays {card} at ({ix},{iy}) drag={drag:.2f}s deploy={ci['deploy']:.2f}s")
        p.last_card=card
        return True,"ok"
    def _spawn(self,team,card,x,y,evolved=False,hero=False):
        p=self.players[team]
        actual=card
        if card.startswith('mirror:'):
            actual=card[7:]
            mlvl=min(p.card_levels.get(actual,p.king_lvl)+1,MAX_LEVEL)
        else:
            mlvl=p.card_levels.get(actual,p.king_lvl)
        return mk_card(actual,mlvl,team,x,y,evolved=evolved,hero=hero)
    def _place(self,team,tr,dep):
        # a deploying unit stands on the field, targetable and damageable, and acts only when its deploy time is over
        if dep>0:tr.statuses.append(Status('deploying',dep))
        self.players[team].troops.append(tr);self.players[team]._register_champ(tr)
    def _proc_pending(self):
        done=[];stagger_add=[]
        for pd in self.pending:
            pd.rem-=self.DT
            if pd.rem<=0:
                st=getattr(pd,'_stagger_troop',None)
                if st:
                    self._place(pd.team,st,pd.deploy)
                    done.append(pd);continue
                r=self._spawn(pd.team,pd.card,pd.x,pd.y,pd.evolved,pd.hero)
                if isinstance(r,list):
                    for j,tr in enumerate(r):
                        at=getattr(tr,'deploy_at',j*0.1)
                        if at<=0:self._place(pd.team,tr,pd.deploy)
                        else:
                            sp=Pending(pd.team,'_stagger_'+str(j),tr.x,tr.y,at,deploy=pd.deploy)
                            sp._stagger_troop=tr
                            stagger_add.append(sp)
                elif hasattr(r,'apply'):
                    self._cast(pd.team,r,pd.x,pd.y)
                else:self._place(pd.team,r,pd.deploy)
                self.log.append(f"[{self.t:.1f}] {pd.card} spawned at ({pd.x:.0f},{pd.y:.0f})")
                done.append(pd)
        for d in done:self.pending.remove(d)
        self.pending.extend(stagger_add)
    def _proc_pending_ab(self):
        done=[]
        for pa in self.pending_ab:
            tr=pa.troop
            if not pa.is_banner and tr and not getattr(tr,'alive',True):
                p=self.players[pa.team]
                p.elixir=min(p.max_ex,p.elixir+pa.ability.cost)
                pa.ability._pend=False
                done.append(pa);continue
            pa.rem-=self.DT
            if pa.rem<=0:
                pa.ability._pend=False
                if pa.is_banner:pa.ability.activate(tr,self)
                else:pa.ability.begin_cast(tr,self)
                done.append(pa)
        for d in done:self.pending_ab.remove(d)
    def _status_mods(self,u):
        st=getattr(u,'statuses',())
        halt=any(s.kind in ('stun','freeze') for s in st)
        sv=min([s.val for s in st if s.kind=='slow']+[1.0])
        rv=max([s.val for s in st if s.kind=='rage']+[0.0])
        mv=min([s.val for s in st if s.kind=='mslow']+[1.0])
        return halt,sv*(1+rv),min(sv,mv)*(1+rv)
    def _shoot(self,team,x,y,spd,tgt,hit):
        if spd<=0:hit(self,None);return
        tx,ty=self._pos(tgt)
        self.projs.append(Projectile(team,x,y,spd,tgt,tx,ty,hit,True))
    def _cast(self,team,sp,x,y):
        def hit(g,pr):sp.apply(g);g.spells.append(sp)
        v=getattr(sp,'proj_spd',0)
        if v<=0:hit(self,None);return
        roll=getattr(sp,'roll',0)
        kt=self.arena.get_tower(team,'king')
        sx,sy=(x,y) if roll else (kt.cx,kt.cy)
        ty=y+roll*(1 if team=='blue' else -1) if roll else y
        self.projs.append(Projectile(team,sx,sy,v,None,x,ty,hit,False))
    def _proc_towers(self):
        for t in self.arena.towers:
            if not t.alive:continue
            opp=self._opp(t.team)
            en=self.players[opp].troops
            al=self.players[t.team].troops
            halt,rate,_=self._status_mods(t)
            tt=t.troop
            if halt:
                if tt:tt.cd=getattr(tt,'spd',tt.cd)
                else:t.cd=max(t.cd,t.spd)
                continue
            if t.ttype=='princess' and tt:
                pd=sum(1 for x in self.arena.towers
                       if x.team==t.team and x.ttype=='princess' and not x.alive)
                evts=tt.tick(self.DT*rate,t,en,al,pt_dead=pd)
                for ev in evts:
                    if ev[0]=='atk':
                        tgt,dmg=ev[1],ev[2]
                        self._shoot(t.team,t.cx,t.cy,getattr(tt,'proj_spd',0),tgt,lambda g,pr,tgt=tgt,dmg=dmg:tgt.take_damage(dmg))
                    elif ev[0]=='pancake':
                        for _ in range(ev[2]):ev[1].level_up()
                        self.log.append(f"[{self.t:.1f}] Chef boost -> lvl {ev[1].lvl}")
            elif t.ttype=='king' and t.active:
                t.cd=max(0,t.cd-self.DT*rate)
                if t.cd<=0:
                    b=None;bd=999
                    for e in en:
                        if not e.alive or hidden(e):continue
                        d=t.dist(e.x,e.y)-getattr(e,'collision_r',0)
                        if d<=t.rng and d<bd:bd=d;b=e
                    if b:
                        self._shoot(t.team,t.cx,t.cy,t.proj_spd,b,lambda g,pr,b=b,dmg=t.dmg:b.take_damage(dmg));t.cd=t.spd
    def _waypoint(self,tr,tx,ty):
        if getattr(tr,'transport','Ground')=='Air' or any(isinstance(c,RiverJump) for c in getattr(tr,'components',[])):return tx,ty
        a=self.arena;y0=a.RIVER[0];y1=a.RIVER[-1]+1;mid=(y0+y1)/2
        near,far=(y0-0.1,y1) if tr.y<mid else (y1,y0-0.1)
        on_br=a.on_bridge(tr.x)
        if y0<=tr.y<y1:
            if not on_br:return tr.x,near
            if ty>=y1:return tr.x,y1+0.1
            if ty<y0:return tr.x,y0-0.1
            return tx,ty
        across=(tr.y<y0 and ty>=y1) or (tr.y>=y1 and ty<y0)
        if not across:return tx,ty
        if on_br and abs(tr.y-mid)<=2.0:return tr.x,far+(0.1 if tr.y<mid else 0)
        lx=min(a.LANES,key=lambda lc:abs(tr.x-lc)+abs(tx-lc))
        return lx,near
    def activate_ability(self,team,troop=None):
        p=self.players[team]
        if troop is None:troop=p.active_champ
        if not troop:
            for ab in p.pending_abilities:
                if ab.can_use() and p.elixir>=ab.cost:
                    p.elixir-=ab.cost;ab._pend=True
                    delay=p.sample_ability_del()
                    self.pending_ab.append(PendingAbility(team,troop,ab,delay,is_banner=True))
                    self.log.append(f"[{self.t:.1f}] {team} activates banner ability drag={delay:.2f}s")
                    return True,"ok"
            return False,"no champion"
        if p.active_champ is None and getattr(troop,'ability',None):
            p._register_champ(troop)
        ab=getattr(troop,'ability',None)
        if ab and hasattr(ab,'banner_pos') and ab.banner_pos:
            if ab.can_use() and p.elixir>=ab.cost:
                p.elixir-=ab.cost;ab._pend=True
                delay=p.sample_ability_del()
                self.pending_ab.append(PendingAbility(team,troop,ab,delay,is_banner=True))
                self.log.append(f"[{self.t:.1f}] {team} activates banner ability drag={delay:.2f}s")
                return True,"ok"
        if troop is not p.active_champ:return False,"not active champion"
        if troop.hp==1 and troop.max_hp==1:return False,"clones cannot use abilities"
        if not ab or not ab.can_use():return False,"ability not ready"
        if p.elixir<ab.cost:return False,"not enough elixir"
        p.elixir-=ab.cost;ab._pend=True
        delay=p.sample_ability_del()
        self.pending_ab.append(PendingAbility(team,troop,ab,delay))
        self.log.append(f"[{self.t:.1f}] {team} activates {troop.name} ability drag={delay:.2f}s")
        return True,"ok"
    def _king_shielded(self,opp):
        return all(x.alive for x in self.arena.towers if x.team==opp and x.ttype=='princess')
    def _default_target(self,tr):
        opp=self._opp(tr.team);ks=self._king_shielded(opp)
        best=None;bd=999
        for tw in self.arena.towers:
            if tw.team!=opp or not tw.alive or (ks and tw.ttype=='king'):continue
            d=tw.dist(tr.x,tr.y)
            if d<bd:bd=d;best=tw
        return best,bd
    def _find_target(self,tr):
        tt=getattr(tr,'_taunt_target',None)
        if tt and getattr(tt,'alive',False):return tt,self._dist(tr,tt)
        if getattr(tr,'retarget_cd',0)>0:
            return self._default_target(tr)
        ag=getattr(tr,'aggro_tgt',None)
        if ag and getattr(ag,'alive',False):
            is_tower=hasattr(ag,'ttype')
            is_bldg_troop=getattr(tr,'targets',['Ground'])==['Buildings']
            if is_bldg_troop:
                # a building targeter heads for the nearest building, so a closer one pulls it until it is already hitting its target
                d=self._dist(tr,ag)
                if d<=tr.rng:return ag,d
            elif not is_tower:
                d=self._dist(tr,ag)
                if d<=max(getattr(tr,'sight_r',5.5),tr.rng+0.5):return ag,d
        opp=self._opp(tr.team)
        tgts=getattr(tr,'targets',['Ground'])
        if not tgts:return None,0
        sr=max(getattr(tr,'sight_r',5.5),tr.rng+0.5)
        all_c=[];near_c=[]
        for e in self.players[opp].troops:
            if not e.alive or hidden(e):continue
            if tgts==['Buildings']:
                if not getattr(e,'is_building',False):continue
            else:
                et=getattr(e,'transport','Ground')
                if et=='Air' and 'Air' not in tgts:continue
            d=self._dist(tr,e)
            all_c.append((d,e))
            if d<=sr:near_c.append((d,e))
        is_bldg=getattr(tr,'is_building',False)
        is_siege=is_bldg and getattr(tr,'name','') in ('X-Bow','Mortar')
        if not is_bldg or is_siege:
            ks=self._king_shielded(opp)
            for tw in self.arena.towers:
                if tw.team!=opp or not tw.alive:continue
                d=tw.dist(tr.x,tr.y)
                # princess towers are the default target while they stand; the king is only picked when in sight
                if d<=sr:near_c.append((d,tw))
                if not(ks and tw.ttype=='king'):all_c.append((d,tw))
        if near_c:
            cands=near_c
        else:
            tw_c=[(d,t) for d,t in all_c if hasattr(t,'ttype')]
            cands=tw_c if tw_c else all_c
        for c in getattr(tr,'components',[]):
            cands=c.modify_target(tr,cands,self)
        if not cands:return self._default_target(tr)
        cands.sort(key=lambda x:x[0])
        tr.aggro_tgt=cands[0][1]
        return cands[0][1],cands[0][0]
    def _do_attack(self,tr,tgt):
        bd=tr.dmg
        if hasattr(tgt,'ttype') and getattr(tr,'ct_dmg',0)>0:bd=tr.ct_dmg
        for c in getattr(tgt,'components',[]):
            if hasattr(c,'pre_damage'):bd=c.pre_damage(tgt,tr,bd,self)
        tgt.take_damage(bd)
        if hasattr(tgt,'ttype') and not tgt.alive:self._tower_down(tgt)
        for c in getattr(tr,'components',[]):c.on_attack(tr,tgt,self)
        for c in getattr(tgt,'components',[]):
            if hasattr(c,'on_take_damage'):c.on_take_damage(tgt,tr,self)
        sd=getattr(tr,'slow_dur',0)
        if sd>0 and not any(isinstance(c,SplashAttack) for c in getattr(tr,'components',[])):
            if hasattr(tgt,'statuses'):tgt.statuses.append(Status('slow',sd,getattr(tr,'slow_val',1.0)))
        stn=getattr(tr,'stun_dur',0)
        if stn>0 and not any(isinstance(c,DualTarget) for c in getattr(tr,'components',[])):
            if hasattr(tgt,'statuses'):tgt.statuses.append(Status('stun',stn))
        if getattr(tr,'is_suicide',False):tr.alive=False
    def _fire(self,tr,tgt):
        v=getattr(tr,'proj_spd',0)
        if v<=0:self._do_attack(tr,tgt);return
        tx,ty=self._pos(tgt)
        def hit(g,pr):
            if pr.homing or math.hypot(*(a-b for a,b in zip(g._pos(tgt),(pr.x,pr.y))))<=tr.splash_r+0.5:g._do_attack(tr,tgt)
        self.projs.append(Projectile(tr.team,tr.x,tr.y,v,tgt,tx,ty,hit,getattr(tr,'proj_homing',True)))
    def _walkable(self,x,y,rj):
        a=self.arena
        if int(y) in a.RIVER:return rj or a.on_bridge(x)
        return not a.blocked(int(x),int(y))
    def _move(self,tr,spd,tx,ty):
        a=self.arena;gnd=getattr(tr,'transport','Ground')!='Air'
        rj=any(isinstance(c,RiverJump) for c in getattr(tr,'components',[]))
        wx,wy=self._waypoint(tr,tx,ty)
        # river crossing is handled by _waypoint, so A* only detours towers and fences (the air grid)
        if gnd and self._pf.seg_blocked(tr.x,tr.y,wx,wy,True):
            path=[q for q in self._pf.get_path(tr,wx,wy,True) if math.hypot(q[0]-tr.x,q[1]-tr.y)>0.25]
            for px,py in reversed(path):
                if not self._pf.seg_blocked(tr.x,tr.y,px,py,True):wx,wy=px,py;break
            else:
                if path:wx,wy=path[0]
        dx=wx-tr.x;dy=wy-tr.y;ds=math.hypot(dx,dy)
        if ds<=0:return
        st=min(ds,spd*self.DT)
        nx=tr.x+dx/ds*st;ny=tr.y+dy/ds*st
        if gnd and not self._walkable(nx,ny,rj):
            if self._walkable(nx,tr.y,rj):ny=tr.y
            elif self._walkable(tr.x,ny,rj):nx=tr.x
            else:
                if int(ny) in a.RIVER and not rj:ny=a.RIVER[0]-0.1 if tr.y<16 else a.RIVER[-1]+1
                else:ny=tr.y
                nx=tr.x
        tr.x=nx;tr.y=ny
    def _proc_troops(self):
        for tm in ('blue','red'):
            p=self.players[tm]
            for tr in p.troops:
                if not tr.alive or has(tr,'deploying'):continue
                ab=getattr(tr,'ability',None)
                if ab:ab.tick(self.DT,tr,self)
                for c in getattr(tr,'components',[]):c.on_tick(tr,self)
                if not tr.alive or has(tr,'burrowed'):continue
                if getattr(tr,'retarget_cd',0)>0:tr.retarget_cd=max(0,tr.retarget_cd-self.DT)
                if getattr(tr,'is_building',False):
                    dr=getattr(tr,'decay',0)
                    if dr>0:
                        tr.hp-=dr*self.DT
                        if tr.hp<=0:tr.hp=0;tr.alive=False;continue
                halt,arate,mrate=self._status_mods(tr)
                if halt:
                    # stun and freeze interrupt the swing: it restarts from load time and the target is re-picked
                    tr.cd=getattr(tr,'fhspd',tr.hspd);tr.first_atk=False;tr.aggro_tgt=None
                spd=0 if halt else tr.spd*mrate
                tgt,td=self._find_target(tr)
                tr.tgt=tgt
                if not tgt or halt:continue
                mr=getattr(tr,'min_rng',0)
                if td<=tr.rng and td>=mr:
                    if getattr(tr,'first_atk',False):tr.cd=getattr(tr,'fhspd',tr.hspd);tr.first_atk=False
                    tr.cd=max(0,tr.cd-self.DT*arate)
                    if tr.cd<=0:
                        self._fire(tr,tgt);tr.cd=tr.hspd
                else:
                    if not getattr(tr,'first_atk',False):tr.cd=max(0,tr.cd-self.DT*arate)
                    if spd>0 and not getattr(tr,'is_building',False):
                        tx,ty=self._pos(tgt)
                        self._move(tr,spd,tx,ty)
    def _proc_statuses(self):
        for u in [t for tm in ('blue','red') for t in self.players[tm].troops]+self.arena.towers:
            sl=getattr(u,'statuses',None)
            if not sl:continue
            for s in sl:s.tick(self.DT)
            u.statuses=[s for s in sl if not s.expired]
    def _proc_projs(self):
        for pr in self.projs:pr.tick(self.DT,self)
        self.projs=[pr for pr in self.projs if pr.alive]
    def _proc_spells(self):
        for sp in self.spells:sp.tick(self.DT,self)
        self.spells=[sp for sp in self.spells if sp.active]
    def _resolve_collisions(self):
        all_tr=[]
        for tm in ('blue','red'):
            all_tr.extend(self.players[tm].troops)
        self._pf.resolve_collisions(all_tr,self.DT)
    def _proc_deaths(self):
        dead_set=set()
        for tm in ('blue','red'):
            p=self.players[tm]
            dead=[tr for tr in p.troops if not tr.alive]
            for tr in dead:
                dead_set.add(id(tr))
                pa_match=[pa for pa in self.pending_ab if pa.troop is tr and not pa.is_banner]
                for pa in pa_match:
                    p.elixir=min(p.max_ex,p.elixir+pa.ability.cost)
                    pa.ability._pend=False
                    self.pending_ab.remove(pa)
                if hasattr(tr,'on_death'):tr.on_death(self)
                if getattr(tr,'ability',None):
                    ab=tr.ability
                    if isinstance(ab,BannerBrigade):
                        heroes=[t for t in p.troops if t.alive and getattr(t,'is_hero',False) and getattr(t,'ability',None) is ab]
                        if not heroes:
                            ab.on_last_death(tr,self)
                            if ab not in p.pending_abilities:p.pending_abilities.append(ab)
                    else:
                        p._on_champ_death(tr)
            p.troops=[tr for tr in p.troops if tr.alive]
        if dead_set:
            for tm in ('blue','red'):
                for tr in self.players[tm].troops:
                    ag=getattr(tr,'aggro_tgt',None)
                    if ag and id(ag) in dead_set:
                        tr.retarget_cd=0.1;tr.aggro_tgt=None
    def deploy(self,team,troop):
        self.players[team].troops.append(troop)
        self.players[team]._register_champ(troop)
    def tick(self):
        if self.ended:return
        self.t=round(self.t+self.DT,10)
        self._gen_ex()
        qcd=self._qcd()
        for p in self.players.values():
            if p.deck:p.deck.tick(self.DT,qcd)
        self._proc_pending()
        for p in self.players.values():
            for ab in list(p.pending_abilities):
                ab.tick(self.DT,None,self)
                if not ab.banner_pos:p.pending_abilities.remove(ab)
        self._proc_pending_ab()
        self._proc_towers()
        self._proc_statuses()
        self._proc_spells()
        self._proc_projs()
        self._proc_troops()
        self._resolve_collisions()
        self._proc_deaths()
        self._check_phase()
        self._tc=getattr(self,'_tc',0)+1
        if self._tc%2==0:
            self.replay.snap(self)
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
            act='*' if t.active else ' '
            s+=f"  {t.team} {t.ttype}: {t.hp}/{t.max_hp} {act}\n"
        return s
