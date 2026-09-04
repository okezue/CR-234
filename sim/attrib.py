import argparse
import sys
from multiprocessing import Pool
from sim import replay as R
from sim.arena import Tower
from sim.calib import inputs
from sim.cards import load
from sim.units import Troop,Building

_DMG={};_NAME={};_KILL=[]

def _cards():
    if not _NAME:
        db=load()['cards']
        for k,c in db.items():
            for u in c['units'].values():_NAME.setdefault(u.get('name') or '',k)
        for k,c in db.items():_NAME[c['name']]=k
    return _NAME

def _source():
    # the attacker is the troop or spell nearest on the call stack: troops carry the swing, spells and death bombs their own name
    f=sys._getframe(2)
    while f is not None:
        l=f.f_locals;s=l.get('self');tr=l.get('tr')
        if isinstance(tr,(Troop,Building)):return tr.name
        if s is not None and s.__class__.__module__ in ('sim.spells','sim.fx') and getattr(s,'name',None):return s.name
        f=f.f_back
    return '?'

_take=Tower.take_damage
def _hooked(self,amt):
    if amt>0 and self.alive:
        k=_cards().get(_source(),_source());d=_DMG.setdefault((self.team,self.ttype,self.cx),{});d[k]=d.get(k,0)+min(amt,self.hp)
        if amt>=self.hp:_KILL.append((self.team,self.ttype,k,dict(d)))
    _take(self,amt)

def _run(a):
    bid,plays,outcome,pid=a
    Tower.take_damage=_hooked;_DMG.clear();_KILL.clear()
    g,info=R.replay_battle(bid,plays,outcome,pid=pid)
    out=[]
    for tm,act in (('blue',outcome['b_hp']),('red',outcome['r_hp'])):
        k=g.arena.get_tower(tm,'king');ps=[g.arena.get_tower(tm,'princess',s) for s in ('left','right')]
        pair=min(((ps[0],act[1]),(ps[1],act[2])),((ps[0],act[2]),(ps[1],act[1])),key=lambda pr:sum(abs(tw.hp-a) for tw,a in pr))
        for tw,a in ((k,act[0]),)+pair:
            out.append((tw.max_hp-tw.hp,tw.max_hp-a,_DMG.get((tw.team,tw.ttype,tw.cx),{})))
    first=next((float(l[1:l.index(']')]) for l in g.log if 'princess down' in l or '3-crown' in l),None)
    return bid,out,(first,_KILL[0]) if _KILL else None,info

def zero(outcomes,placements,bids):
    # real games still 0-0 at 3:00: overtime was played (a placement after 180 s) and at most the sudden-death tower fell
    return [b for b in bids if max(p['time'] for p in placements[b])>3600 and sum(v==0 for v in outcomes[b]['b_hp']+outcomes[b]['r_hp'])<=1]

def main():
    ap=argparse.ArgumentParser(description='share-weighted sim minus real tower damage per attacking card')
    ap.add_argument('--jobs',type=int,default=4);ap.add_argument('--limit',type=int,default=0);ap.add_argument('--min',type=int,default=30)
    ap.add_argument('--zero',action='store_true',help='the real 0-0 at 3:00 games: how many the sim ends in regulation and what killed the first tower')
    a=ap.parse_args()
    outcomes,placements,pids,bids=inputs()
    bids=[b for b in bids if outcomes[b].get('b_hp') and outcomes[b].get('r_hp')]
    if a.zero:bids=zero(outcomes,placements,bids)
    if a.limit:bids=bids[:a.limit]
    with Pool(a.jobs) as p:res=p.map(_run,[(b,placements[b],outcomes[b],pids.get(b)) for b in bids],chunksize=4)
    if a.zero:
        early=[(b,k,i) for b,_,k,i in res if k and k[0]<180]
        by={};share={}
        for b,(first,(team,tt,card,dmg)),i in early:
            by[card]=by.get(card,0)+1
            s=sum(dmg.values())
            for c,v in dmg.items():share[c]=share.get(c,0)+v/s
        print(f'{len(bids)} real 0-0 games at 3:00; sim takes a tower in regulation in {len(early)} ({100*len(early)/max(len(bids),1):.1f}%), '
              f'ends before the last play in {sum(i["premature"] for _,_,_,i in res)}')
        print('first sim tower kill in regulation: finishing card (count) and damage share on that tower (sum over games)')
        for c,n in sorted(by.items(),key=lambda kv:-kv[1])[:25]:print(f'  {c:24s} {n:4d} {share.get(c,0):7.1f}')
        ts=sorted(k[0] for _,_,k,_ in res if k)
        print('sim first tower kill time quartiles:',[ts[len(ts)*q//4] for q in (1,2,3)] if ts else None)
        return
    err={};n={};sim={};real={};tot=[0,0]
    for bid,rows,_,_ in res:
        for ds,dr,by in rows:
            tot[0]+=ds;tot[1]+=dr
            if not by:continue
            s=sum(by.values())
            for k,v in by.items():
                w=v/s;err[k]=err.get(k,0)+w*(ds-dr);n[k]=n.get(k,0)+w;sim[k]=sim.get(k,0)+v;real[k]=real.get(k,0)+w*dr
    print(f'{len(bids)} games, sim tower damage / real: {tot[0]/max(tot[1],1):.3f}')
    print('card: attacked towers (share weighted), mean sim-real HP per tower share, sim damage / real share')
    rows=sorted(((err[k]/n[k],k) for k in n if n[k]>=a.min),reverse=True)
    for e,k in rows:print(f'  {k:24s} {n[k]:7.1f} {e:+8.0f} {sim[k]/max(real[k],1):6.2f}')

if __name__=='__main__':main()
