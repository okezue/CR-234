import csv
import os
import argparse
import random
from sim.game import Game,card_info,MAX_LEVEL
from sim.cards import load,key,card

_FILLER=['knight','archers','fireball','zap','valkyrie','musketeer','baby_dragon','mini_pekka']
_BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _has_json(n):
    return key(n) is not None

def _api_name_to_json(n):
    return load()['aliases']['names'].get(n) or n.lower().replace(' ','_').replace('.','').replace('-','_')

def _card_rarity(jn):
    return card(jn)['rarity'] if _has_json(jn) else 'common'

def _api_lvl(jn,raw_lvl):
    # the RoyaleAPI level is relative to the rarity minimum
    return raw_lvl+load()['meta']['minLevel'][_card_rarity(jn)]-1

def _hp_to_klvl(hp):
    if not hp:return 11
    try:h=int(float(hp))
    except ValueError:return 11
    hps=card('king_tower')['stats']['hitpoints']
    return next((i+1 for i,v in enumerate(hps) if h<=v),MAX_LEVEL)

def norm(c):
    if not c or c=='_invalid':return None,False,False
    if c.startswith('ability-'):return key(c[8:]) or c[8:].replace('-','_'),False,False
    evo=c.endswith('-ev1')
    hero=c.endswith('-hero')
    b=c[:-4] if evo else c[:-5] if hero else c
    return key(b) or b.replace('-','_'),evo,hero

def _mk_deck(cards):
    dk=list(cards);fi=0
    while len(dk)<8:
        c=_FILLER[fi%len(_FILLER)]
        if c not in dk:dk.append(c)
        fi+=1
    return dk[:8]

def _engineer_hand(deck_cards,plays):
    first4=[]
    for c in plays:
        if c not in first4:
            first4.append(c)
        if len(first4)>=4:break
    rest=[c for c in deck_cards if c not in first4]
    random.shuffle(first4)
    random.shuffle(rest)
    hand=list(first4[:4])
    while len(hand)<4 and rest:hand.append(rest.pop(0))
    nxt=rest.pop(0) if rest else None
    q=list(rest)
    return hand,nxt,q

def _force_hand(g,tm,card):
    dk=g.players[tm].deck
    if card in dk.hand:return True
    if dk.nxt==card:
        if dk.q:
            dk.hand.append(card)
            dk.nxt=dk.q.pop(0)
        else:
            dk.hand.append(card)
            dk.nxt=None
        return True
    if card in dk.q:
        dk.q.remove(card)
        if len(dk.hand)<4:
            dk.hand.append(card)
        else:
            old=dk.hand.pop(0)
            dk.hand.append(card)
            dk.q.insert(0,old)
        return True
    if card in dk.all:
        if len(dk.hand)<4:
            dk.hand.append(card)
        else:
            old=dk.hand.pop(0)
            dk.hand.append(card)
            if dk.nxt:
                dk.q.insert(0,dk.nxt)
            dk.nxt=old
        return True
    return False

def _ability_troop(g,tm,base):
    # the recorded ability names its card; the harness makes that troop the active one since the real game did
    if not base or not _has_json(base):return None
    nm=card(base)['name'];p=g.players[tm]
    tr=next((t for t in p.troops if t.alive and getattr(t,'ability',None) and getattr(t,'name','')==nm),None)
    if tr and tr is not p.active_champ and not hasattr(tr.ability,'banner_pos'):p.active_champ=tr
    return tr

def _open_pocket(g,tm,x,y):
    if tm=='red' and y<15:
        side='left' if x<=8 else 'right'
        pt=g.arena.get_tower('blue','princess',side)
        if pt and pt.alive and pt.hp<pt.max_hp*0.7:
            pt.hp=0;pt.alive=False
            g._tower_down(pt)
    elif tm=='blue' and y>17:
        side='left' if x<=8 else 'right'
        pt=g.arena.get_tower('red','princess',side)
        if pt and pt.alive and pt.hp<pt.max_hp*0.7:
            pt.hp=0;pt.alive=False
            g._tower_down(pt)

def load_meta(path):
    out={}
    with open(path) as f:
        for r in csv.DictReader(f):
            tag=r.get('replayTag','').lstrip('#')
            if not tag:continue
            tc=int(float(r.get('team_0_crowns',0) or 0))
            oc=int(float(r.get('opponent_0_crowns',0) or 0))
            tch=float(r.get('team_0_trophyChange',0) or 0)
            result='W' if tch>0 else 'L' if tch<0 else 'D'
            t0_tag=r.get('team_0_tag','').lstrip('#')
            o0_tag=r.get('opponent_0_tag','').lstrip('#')
            b_deck=[];b_lvls={};r_deck=[];r_lvls={}
            for i in range(8):
                cn=r.get(f'team_0_cards_{i}_name','')
                cl=r.get(f'team_0_cards_{i}_level','')
                if cn:
                    jn=_api_name_to_json(cn)
                    b_deck.append(jn)
                    if cl:
                        try:b_lvls[jn]=min(_api_lvl(jn,int(float(cl))),MAX_LEVEL)
                        except ValueError:pass
                cn=r.get(f'opponent_0_cards_{i}_name','')
                cl=r.get(f'opponent_0_cards_{i}_level','')
                if cn:
                    jn=_api_name_to_json(cn)
                    r_deck.append(jn)
                    if cl:
                        try:r_lvls[jn]=min(_api_lvl(jn,int(float(cl))),MAX_LEVEL)
                        except ValueError:pass
            b_tt_raw=r.get('team_0_supportCards_0_name','Tower Princess')
            r_tt_raw=r.get('opponent_0_supportCards_0_name','Tower Princess')
            _TT_MAP={'Tower Princess':'tower_princess','Dagger Duchess':'dagger_duchess',
                     'Cannoneer':'cannoneer','Royal Chef':'royal_chef'}
            b_tt=_TT_MAP.get(b_tt_raw,'tower_princess')
            r_tt=_TT_MAP.get(r_tt_raw,'tower_princess')
            b_tt_lvl=r.get('team_0_supportCards_0_level','')
            r_tt_lvl=r.get('opponent_0_supportCards_0_level','')
            try:b_klvl=min(int(float(b_tt_lvl)),MAX_LEVEL)
            except ValueError:b_klvl=max(b_lvls.values()) if b_lvls else 11
            try:r_klvl=min(int(float(r_tt_lvl)),MAX_LEVEL)
            except ValueError:r_klvl=max(r_lvls.values()) if r_lvls else 11
            gm=r.get('gameMode_name','')
            if tch==0 and tc>oc:result='W'
            elif tch==0 and oc>tc:result='L'
            out[tag]={'result':result,'tc':tc,'oc':oc,
                'b_deck':b_deck,'r_deck':r_deck,
                'b_lvls':b_lvls,'r_lvls':r_lvls,
                'b_klvl':b_klvl,'r_klvl':r_klvl,
                'b_tt':b_tt,'r_tt':r_tt,
                't0_tag':t0_tag,'o0_tag':o0_tag,
                'gameMode':gm}
    return out

def load_meta_v2(path):
    out={}
    with open(path) as f:
        for r in csv.DictReader(f):
            tag=r.get('replayTag','').lstrip('#')
            if not tag:continue
            res=r.get('result','')
            tc=int(float(r.get('team_crowns',0) or 0))
            oc=int(float(r.get('opp_crowns',0) or 0))
            if not res:
                res='W' if tc>oc else 'L' if tc<oc else 'D'
            team_tags=r.get('team_tags','').lstrip('#')
            opp_tags=r.get('opponent_tags','').lstrip('#')
            b_deck=[];b_lvls={};r_deck=[];r_lvls={};b_evo=set();r_evo=set();b_hero=set();r_hero=set()
            for i in range(8):
                for pfx,deck,lvls,evos,heroes in (('team',b_deck,b_lvls,b_evo,b_hero),('opp',r_deck,r_lvls,r_evo,r_hero)):
                    cn=r.get(f'{pfx}_card_{i}','')
                    cl=r.get(f'{pfx}_card_{i}_lvl','')
                    if not cn:continue
                    jn,evo,hero=norm(cn)
                    deck.append(jn)
                    if evo:evos.add(jn)
                    if hero:heroes.add(jn)
                    if cl:
                        try:lvls[jn]=min(int(float(cl)),MAX_LEVEL)
                        except ValueError:pass
            b_klvl=int(float(r.get('team_king_lvl',0) or 0))
            r_klvl=int(float(r.get('opp_king_lvl',0) or 0))
            if not b_klvl:b_klvl=max(b_lvls.values()) if b_lvls else 11
            if not r_klvl:r_klvl=max(r_lvls.values()) if r_lvls else 11
            b_tt=r.get('team_tower_troop','') or 'tower_princess'
            r_tt=r.get('opp_tower_troop','') or 'tower_princess'
            out[tag]={'result':res,'tc':tc,'oc':oc,
                'b_deck':b_deck,'r_deck':r_deck,
                'b_lvls':b_lvls,'r_lvls':r_lvls,
                'b_klvl':b_klvl,'r_klvl':r_klvl,
                'b_tt':b_tt,'r_tt':r_tt,
                't0_tag':team_tags,'o0_tag':opp_tags,
                'gameMode':r.get('gameMode_name','') or r.get('battle_type',''),
                'b_hp':_hp(r,'team'),'r_hp':_hp(r,'opp'),
                'b_evo':b_evo,'r_evo':r_evo,'b_hero':b_hero,'r_hero':r_hero}
    return out

def _hp(r,side):
    v=[r.get(f'{side}_king_hp',''),r.get(f'{side}_princess_hp_0',''),r.get(f'{side}_princess_hp_1','')]
    try:return [int(float(x)) for x in v]
    except ValueError:return None

def load_worker_rows(path,ids,meta=None):
    ok_pids={}
    if meta:
        for tag,m in meta.items():
            if tag in ids:
                ok_pids[tag]=set()
                if m.get('t0_tag'):ok_pids[tag].add(m['t0_tag'])
                if m.get('o0_tag'):ok_pids[tag].add(m['o0_tag'])
    data={};seen={};pids={}
    with open(path) as f:
        for r in csv.DictReader(f):
            bid=r['battle_id']
            if bid not in ids:continue
            pid=r.get('player_id','')
            if bid in ok_pids and pid not in ok_pids[bid]:continue
            t_raw=r.get('time','0')
            try:t=int(float(t_raw))
            except ValueError:t=0
            card=r.get('card','')
            tm=r.get('team','blue')
            key=(bid,card,t,tm)
            if key in seen:continue
            seen[key]=1
            if bid not in data:data[bid]=[]
            if bid not in pids:pids[bid]=pid
            x_raw=r.get('x','')
            y_raw=r.get('y','')
            try:tx=float(x_raw)/1000.0
            except ValueError:tx=9.0
            try:ty=float(y_raw)/1000.0
            except ValueError:ty=16.0
            is_ability=card.startswith('ability-') or card=='_invalid' or (r.get('ability') or '0').startswith('1')
            data[bid].append({
                'card':card,'time':t,'team':tm,
                'tile_x':tx,'tile_y':ty,
                'ability':1 if is_ability else 0,
                'card_type':r.get('card_type','normal'),
            })
    for bid in data:
        data[bid].sort(key=lambda p:p['time'])
    return data,pids

def load_outcomes(path):
    out={}
    with open(path) as f:
        for r in csv.DictReader(f):
            out[r['replayTag']]={'result':r['result'],
                'tc':int(r['team_crowns']),'oc':int(r['opp_crowns']),
                'pid':r['player_id']}
    return out

def load_placements(path,ids):
    data={};seen={}
    with open(path) as f:
        for r in csv.DictReader(f):
            bid=r['battle_id']
            if bid not in ids:continue
            t_raw=r.get('time','0')
            try:t=int(float(t_raw))
            except ValueError:t=0
            card=r.get('card','')
            tm=r.get('team','blue')
            key=(bid,card,t,tm)
            if key in seen:continue
            seen[key]=1
            if bid not in data:data[bid]=[]
            tx_raw=r.get('tile_x','')
            ty_raw=r.get('tile_y','')
            try:tx=float(tx_raw)
            except ValueError:tx=9.0
            try:ty=float(ty_raw)
            except ValueError:ty=16.0
            data[bid].append({
                'card':card,'time':t,'team':tm,
                'tile_x':tx,'tile_y':ty,
                'ability':int(r.get('ability','0') or '0'),
                'card_type':r.get('card_type','normal'),
            })
    for bid in data:
        data[bid].sort(key=lambda p:p['time'])
    return data

def extract_decks(plays):
    decks={'blue':[],'red':[]}
    for p in plays:
        tm=p['team']
        base,evo,hero=norm(p['card'])
        if base is None:continue
        if base not in decks[tm]:
            decks[tm].append(base)
    for tm in decks:
        decks[tm]=_mk_deck(decks[tm])
    return decks

def _match_sides(plays,t0_deck,o0_deck,pid=None,t0_tag=None,o0_tag=None):
    bc=set();rc=set()
    for p in plays:
        base,_,_=norm(p['card'])
        if not base:continue
        if p['team']=='blue':bc.add(base)
        else:rc.add(base)
    t0s=set(t0_deck);o0s=set(o0_deck)
    b_t0=len(bc&t0s);b_o0=len(bc&o0s)
    if b_t0!=b_o0:return b_t0<b_o0
    r_t0=len(rc&t0s);r_o0=len(rc&o0s)
    if r_t0!=r_o0:return r_t0>r_o0
    if pid and t0_tag and pid==t0_tag:return False
    if pid and o0_tag and pid==o0_tag:return True
    return False

def _detect_true_red(plays):
    bx=[p['tile_x'] for p in plays if p['team']=='blue']
    if not bx:return False
    return sum(1 for x in bx if x>9)/len(bx)>0.5

def _mirror_x(plays):
    for p in plays:
        p['tile_x']=18.0-p['tile_x']

def replay_battle(bid,plays,outcome,verbose=False,pid=None):
    if _detect_true_red(plays):
        _mirror_x(plays)
    t0_deck=outcome.get('b_deck',[])
    o0_deck=outcome.get('r_deck',[])
    t0_lvls=outcome.get('b_lvls',{})
    o0_lvls=outcome.get('r_lvls',{})
    t0_klvl=outcome.get('b_klvl',11)
    o0_klvl=outcome.get('r_klvl',11)
    t0_tt=outcome.get('b_tt','tower_princess')
    o0_tt=outcome.get('r_tt','tower_princess')
    if t0_deck and len(t0_deck)>=4:
        flipped=_match_sides(plays,t0_deck,o0_deck,pid,outcome.get('t0_tag'),outcome.get('o0_tag'))
        if flipped:
            for p in plays:
                p['team']='red' if p['team']=='blue' else 'blue'
                p['tile_y']=32.0-p['tile_y']
        b_deck=t0_deck;r_deck=o0_deck
        b_lvls=t0_lvls;r_lvls=o0_lvls
        b_klvl=t0_klvl;r_klvl=o0_klvl
        b_tt=t0_tt;r_tt=o0_tt
        decks={'blue':_mk_deck(b_deck),'red':_mk_deck(r_deck)}
    else:
        b_lvls=outcome.get('b_lvls',{})
        r_lvls=outcome.get('r_lvls',{})
        b_klvl=outcome.get('b_klvl',11)
        r_klvl=outcome.get('r_klvl',11)
        b_tt=outcome.get('b_tt','tower_princess')
        r_tt=outcome.get('r_tt','tower_princess')
        decks=extract_decks(plays)
    blue_plays=[norm(p['card'])[0] for p in plays if p['team']=='blue' and norm(p['card'])[0]]
    red_plays=[norm(p['card'])[0] for p in plays if p['team']=='red' and norm(p['card'])[0]]
    bh,bn,bq=_engineer_hand(decks['blue'],blue_plays)
    rh,rn,rq=_engineer_hand(decks['red'],red_plays)
    random.seed(42)
    g=Game(
        p1={'deck':decks['blue'],'king_lvl':b_klvl,'tt_name':b_tt,'drag_del':0,'drag_std':0,
            'ability_del':0,'ability_std':0,'card_levels':b_lvls},
        p2={'deck':decks['red'],'king_lvl':r_klvl,'tt_name':r_tt,'drag_del':0,'drag_std':0,
            'ability_del':0,'ability_std':0,'card_levels':r_lvls}
    )
    bd=g.players['blue'].deck
    bd.hand=list(bh);bd.nxt=bn;bd.q=list(bq)
    rd=g.players['red'].deck
    rd.hand=list(rh);rd.nxt=rn;rd.q=list(rq)
    hero_cards={'blue':set(outcome.get('b_hero',())),'red':set(outcome.get('r_hero',()))}
    evo_cards={'blue':set(outcome.get('b_evo',())),'red':set(outcome.get('r_evo',()))}
    for p in plays:
        b,e,h=norm(p['card'])
        if not b:continue
        if h or p.get('card_type')=='hero':hero_cards[p['team']].add(b)
        if e or p.get('card_type')=='evo':evo_cards[p['team']].add(b)
    n_played={'blue':{},'red':{}}
    errs=[]
    for p in plays:
        ts=p['time']/20.0
        base,_,_=norm(p['card'])
        tm=p['team']
        tx,ty=p['tile_x'],p['tile_y']
        itx,ity=int(tx),int(ty)
        if g.ended:break
        g.run_to(ts)
        if g.ended:break
        if p['ability']==1:
            g.players[tm].elixir=10
            g.activate_ability(tm,_ability_troop(g,tm,base))
            continue
        if base is None:continue
        if not _has_json(base):
            if verbose:errs.append(f"  skip {base} (no json)")
            continue
        hero=base in hero_cards[tm]
        evo=False
        if base in evo_cards[tm] and card(base)['evo']:
            # every (cycles+1)th deployment of an evolution card is the evolved one
            n=n_played[tm].get(base,0)+1;n_played[tm][base]=n
            evo=n%((card(base)['evo'].get('cycles') or 1)+1)==0
        ci=card_info(base)
        if not ci.get('deploy_anywhere'):
            _open_pocket(g,tm,itx,ity)
        _force_hand(g,tm,base)
        g.players[tm].elixir=10
        ok,msg=g.play_card(tm,base,tx,ty,evolved=evo,hero=hero)
        if not ok:
            for dx,dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1)]:
                nx,ny=itx+dx,ity+dy
                if 0<=nx<18 and 0<=ny<32:
                    ok,msg=g.play_card(tm,base,nx,ny,evolved=evo,hero=hero)
                    if ok:break
        if not ok:
            fy=min(14,ity) if tm=='blue' else max(17,ity)
            ok,msg=g.play_card(tm,base,itx,fy,evolved=evo,hero=hero)
        if not ok:
            fy=8 if tm=='blue' else 24
            fx=9
            ok,msg=g.play_card(tm,base,fx,fy,evolved=evo,hero=hero)
        if not ok and verbose:
            errs.append(f"  fail {base}@({itx},{ity}): {msg}")
    if not g.ended:
        g.run_to(g.END)
    sw=g.winner
    bc=g.players['blue'].crowns
    rc=g.players['red'].crowns
    aw=outcome['result']
    atc=outcome['tc']
    aoc=outcome['oc']
    actual_winner='blue' if aw=='W' else 'red' if aw=='L' else None
    win_match=(sw==actual_winner)
    crown_exact=(bc==atc and rc==aoc)
    crown_close=(abs(bc-atc)<=1 and abs(rc-aoc)<=1)
    stm='blue' if sw=='blue' else 'red' if sw=='red' else 'draw'
    last=max((p['time'] for p in plays),default=0)/20.0
    info={'bid':bid,'sim_winner':sw,'sim_bc':bc,'sim_rc':rc,
          'actual_winner':actual_winner,'actual_bc':atc,'actual_rc':aoc,
          'win_match':win_match,'crown_exact':crown_exact,'crown_close':crown_close,
          'stm':stm,'end_t':g.t,'last_play':last,'premature':g.t<last-1,'hp_err':None,'tower_state':None}
    if outcome.get('b_hp') and outcome.get('r_hp'):
        errs=[];states=[]
        for tm,act in (('blue',outcome['b_hp']),('red',outcome['r_hp'])):
            k=g.arena.get_tower(tm,'king');ps=[g.arena.get_tower(tm,'princess',s) for s in ('left','right')]
            # RoyaleAPI does not say which princess tower is which, so pair them in the order that fits better
            pair=min(((ps[0],act[1]),(ps[1],act[2])),((ps[0],act[2]),(ps[1],act[1])),key=lambda pr:sum(abs(tw.hp-a) for tw,a in pr))
            for tw,a in ((k,act[0]),)+pair:
                errs.append(abs(tw.hp-a)/tw.max_hp);states.append((tw.hp>0)==(a>0))
        info['hp_err']=sum(errs)/6;info['tower_state']=sum(states)/6
    if verbose:
        sym='Y' if win_match else 'X'
        csym='exact' if crown_exact else ('~1' if crown_close else 'diff')
        print(f"  {bid}: sim={stm} {bc}-{rc}  actual={'W' if aw=='W' else 'L'} {atc}-{aoc}  [{sym}] crowns={csym}  lvls=b{b_klvl}/r{r_klvl}")
        for e in errs:print(e)
        if not crown_exact:
            for tw in g.arena.towers:
                st='ALIVE' if tw.alive else 'DEAD'
                print(f"    Tower {tw.team} {tw.ttype} (cx={tw.cx}) hp={tw.hp}/{tw.max_hp} {st}")
    return g,info

def _run(a):
    return None,replay_battle(a[0],a[1],a[2],pid=a[3])[1]

def main():
    ap=argparse.ArgumentParser(description='Replay scraped battles through simulator')
    ap.add_argument('--meta',default=None,help='all_battle_meta_data.csv (has card levels)')
    ap.add_argument('--workers',default=None,help='all_worker_rows.csv (card placements)')
    ap.add_argument('--outcomes',default=os.path.join(_BASE,'data','processed','battle_outcomes_1v1.csv'))
    ap.add_argument('--placements',default=os.path.join(_BASE,'data','processed','card_placements_1v1_labeled.csv'))
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--battle',type=str,default=None)
    ap.add_argument('--visualize',action='store_true')
    ap.add_argument('--visualize-multi',type=int,default=0,help='Visualize N battles with picker')
    ap.add_argument('--verbose',action='store_true')
    ap.add_argument('--jobs',type=int,default=1)
    args=ap.parse_args()
    print("=== Battle Replay Validation ===")
    use_meta=args.meta and args.workers
    if use_meta:
        print(f"Loading metadata from {args.meta}...")
        with open(args.meta) as _f:
            _hdr=_f.readline()
        if 'team_card_0' in _hdr:
            outcomes=load_meta_v2(args.meta)
        else:
            outcomes=load_meta(args.meta)
        print(f"Loaded {len(outcomes)} battles with card levels")
        if args.battle:
            ids={args.battle}
        else:
            ids=set(outcomes.keys())
        print(f"Loading placements from {args.workers}...")
        placements,pids=load_worker_rows(args.workers,ids,outcomes)
    else:
        outcomes=load_outcomes(args.outcomes)
        pids={}
        print(f"Loaded {len(outcomes)} outcomes")
        if args.battle:
            ids={args.battle}
        else:
            ids=set(outcomes.keys())
        print("Loading placements...")
        placements=load_placements(args.placements,ids)
    matched={bid for bid in ids if bid in placements}
    print(f"Matched {len(matched)} battles with placements")
    if args.battle:
        if args.battle not in matched:
            print(f"Battle {args.battle} not found");return
        bids=[args.battle]
    else:
        bids=sorted(matched)
        if args.limit>0:bids=bids[:args.limit]
    tot=len(bids)
    wm=0;ce=0;cc=0;pm=0;done=0;hpe=[];tst=[]
    print(f"Running {tot} battles...\n")
    bids=[b for b in bids if b in outcomes]
    if args.jobs>1 and not args.visualize:
        from multiprocessing import Pool
        pool=Pool(args.jobs);runs=pool.imap(_run,[(b,placements[b],outcomes[b],pids.get(b)) for b in bids],chunksize=4)
    else:runs=(replay_battle(b,placements[b],outcomes[b],verbose=args.verbose,pid=pids.get(b)) for b in bids)
    for g,info in runs:
        bid=info['bid']
        if info['win_match']:wm+=1
        if info['crown_exact']:ce+=1
        if info['crown_close']:cc+=1
        if info['premature']:pm+=1
        if info['hp_err'] is not None:hpe.append(info['hp_err']);tst.append(info['tower_state'])
        done+=1
        if not args.verbose and done%10==0:
            print(f"  [{done:4d}/{tot}] last={bid} wm={wm}/{done} ({100*wm/done:.1f}%)")
        if args.visualize and args.battle:
            from sim.viz import visualize as viz
            print(f"\nOpening visualizer for {bid}...")
            viz(g)
    print("\n=== Summary ===")
    if done==0:print("No battles replayed.");return
    print(f"Winner match: {wm}/{done} ({100*wm/done:.1f}%)")
    print(f"Crown exact:  {ce}/{done} ({100*ce/done:.1f}%)")
    print(f"Crown +/-1:   {cc}/{done} ({100*cc/done:.1f}%)")
    print(f"Ended before last human play: {pm}/{done} ({100*pm/done:.1f}%)")
    if hpe:print(f"Tower HP error (mean, fraction of max): {sum(hpe)/len(hpe):.3f}; tower alive/dead agreement: {100*sum(tst)/len(tst):.1f}%")
    if args.visualize_multi>0:
        from sim.viz import visualize_browser
        vm=min(args.visualize_multi,len(bids))
        vbids=bids[:vm] if vm<len(bids) else bids
        all_info=[]
        print(f"\nRunning {len(vbids)} battles for visualizer...")
        vwm=0;vce=0;vdone=0
        for bid in vbids:
            if bid not in outcomes or bid not in placements:continue
            oc=outcomes[bid]
            g,info=replay_battle(bid,placements[bid],oc,pid=pids.get(bid))
            vdone+=1
            if info['win_match']:vwm+=1
            if info['crown_exact']:vce+=1
            if vdone%100==0:print(f"  [{vdone}/{len(vbids)}] wm={vwm}/{vdone}")
            bc=set();rc=set()
            for p in placements[bid]:
                base,_,_=norm(p['card'])
                if base:
                    if p['team']=='blue':bc.add(base)
                    else:rc.add(base)
            all_info.append({'bid':bid,
                'b_deck':oc.get('b_deck',list(bc)[:8]),'r_deck':oc.get('r_deck',list(rc)[:8]),
                'result':oc.get('result','?'),
                'tc':oc.get('tc',0),'oc':oc.get('oc',0),
                'b_klvl':oc.get('b_klvl',11),'r_klvl':oc.get('r_klvl',11),
                'b_tt':oc.get('b_tt','tower_princess'),'r_tt':oc.get('r_tt','tower_princess'),
                'sim_bc':info['sim_bc'],'sim_rc':info['sim_rc'],
                'win_match':info['win_match'],'crown_exact':info['crown_exact'],
                'gm':oc.get('gameMode','')})
        print(f"Done. Win={vwm}/{vdone} ({100*vwm/max(1,vdone):.1f}%) Exact={vce}/{vdone}")
        visualize_browser(all_info,outcomes,placements,pids)

if __name__=='__main__':
    main()
