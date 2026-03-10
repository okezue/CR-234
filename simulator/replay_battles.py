import csv,sys,os,argparse,random,math
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from game import Game,card_info

_FILLER=['knight','archers','fireball','zap','valkyrie','musketeer','baby_dragon','mini_pekka']
_BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CD=os.path.join(_BASE,'game_data','cards')

def _has_json(n):
    return os.path.exists(os.path.join(_CD,n+'.json'))

def norm(card):
    if not card or card=='_invalid':return None,False,False
    evo=card.endswith('-ev1')
    hero=card.endswith('-hero')
    b=card
    if evo:b=b[:-4]
    elif hero:b=b[:-5]
    b=b.replace('-','_')
    return b,evo,hero

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

def _open_pocket(g,tm,x,y):
    if tm=='red' and y<15:
        side='left' if x<=8 else 'right'
        pt=g.arena.get_tower('blue','princess',side)
        if pt and pt.alive:
            pt.hp=0;pt.alive=False
            g._tower_down(pt)
            g._pf.rebuild_tower_grid()
    elif tm=='blue' and y>17:
        side='left' if x<=8 else 'right'
        pt=g.arena.get_tower('red','princess',side)
        if pt and pt.alive:
            pt.hp=0;pt.alive=False
            g._tower_down(pt)
            g._pf.rebuild_tower_grid()

def load_outcomes(path):
    out={}
    with open(path) as f:
        rd=csv.DictReader(f)
        for r in rd:
            out[r['replayTag']]={'result':r['result'],
                'tc':int(r['team_crowns']),'oc':int(r['opp_crowns']),
                'pid':r['player_id']}
    return out

def load_placements(path,ids):
    data={};seen={}
    with open(path) as f:
        rd=csv.DictReader(f)
        for r in rd:
            bid=r['battle_id']
            if bid not in ids:continue
            t_raw=r.get('time','0')
            try:t=int(float(t_raw))
            except:t=0
            card=r.get('card','')
            tm=r.get('team','blue')
            key=(bid,card,t,tm)
            if key in seen:continue
            seen[key]=1
            if bid not in data:data[bid]=[]
            tx_raw=r.get('tile_x','')
            ty_raw=r.get('tile_y','')
            try:tx=float(tx_raw)
            except:tx=9.0
            try:ty=float(ty_raw)
            except:ty=16.0
            data[bid].append({
                'card':card,
                'time':t,
                'team':tm,
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

def replay_battle(bid,plays,outcome,verbose=False):
    decks=extract_decks(plays)
    blue_plays=[norm(p['card'])[0] for p in plays if p['team']=='blue' and norm(p['card'])[0]]
    red_plays=[norm(p['card'])[0] for p in plays if p['team']=='red' and norm(p['card'])[0]]
    bh,bn,bq=_engineer_hand(decks['blue'],blue_plays)
    rh,rn,rq=_engineer_hand(decks['red'],red_plays)
    random.seed(42)
    g=Game(
        p1={'deck':decks['blue'],'drag_del':0,'drag_std':0,'ability_del':0,'ability_std':0},
        p2={'deck':decks['red'],'drag_del':0,'drag_std':0,'ability_del':0,'ability_std':0}
    )
    bd=g.players['blue'].deck
    bd.hand=list(bh);bd.nxt=bn;bd.q=list(bq)
    rd=g.players['red'].deck
    rd.hand=list(rh);rd.nxt=rn;rd.q=list(rq)
    errs=[]
    for p in plays:
        ts=p['time']/10.0
        base,evo,hero=norm(p['card'])
        tm=p['team']
        tx,ty=p['tile_x'],p['tile_y']
        itx,ity=int(tx),int(ty)
        if g.ended:break
        g.run_to(ts)
        if g.ended:break
        if p['ability']==1:
            if base is None or base=='_invalid':
                g.players[tm].elixir=10
                g.activate_ability(tm)
                continue
            if (evo or hero) and base and _has_json(base):
                _force_hand(g,tm,base)
                g.players[tm].elixir=10
                g.activate_ability(tm)
                continue
            g.players[tm].elixir=10
            g.activate_ability(tm)
            continue
        if base is None:continue
        if not _has_json(base):
            if verbose:errs.append(f"  skip {base} (no json)")
            continue
        ci=card_info(base)
        if not ci.get('deploy_anywhere'):
            _open_pocket(g,tm,itx,ity)
        _force_hand(g,tm,base)
        g.players[tm].elixir=10
        ok,msg=g.play_card(tm,base,itx,ity)
        if not ok:
            for dx,dy in [(0,1),(0,-1),(1,0),(-1,0),(1,1),(-1,-1)]:
                nx,ny=itx+dx,ity+dy
                if 0<=nx<18 and 0<=ny<32:
                    ok,msg=g.play_card(tm,base,nx,ny)
                    if ok:break
        if not ok:
            fy=min(14,ity) if tm=='blue' else max(17,ity)
            ok,msg=g.play_card(tm,base,itx,fy)
        if not ok:
            fy=8 if tm=='blue' else 24
            fx=9
            ok,msg=g.play_card(tm,base,fx,fy)
        if not ok and verbose:
            errs.append(f"  fail {base}@({itx},{ity}): {msg}")
    if not g.ended:
        g.run_to(300)
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
    info={'bid':bid,'sim_winner':sw,'sim_bc':bc,'sim_rc':rc,
          'actual_winner':actual_winner,'actual_bc':atc,'actual_rc':aoc,
          'win_match':win_match,'crown_exact':crown_exact,'crown_close':crown_close,
          'stm':stm}
    if verbose:
        sym='Y' if win_match else 'X'
        csym='exact' if crown_exact else ('~1' if crown_close else 'diff')
        print(f"  {bid}: sim={stm} {bc}-{rc}  actual={'W' if aw=='W' else 'L'} {atc}-{aoc}  [{sym}] crowns={csym}")
        for e in errs:print(e)
    return g,info

def main():
    ap=argparse.ArgumentParser(description='Replay scraped battles through simulator')
    ap.add_argument('--outcomes',default=os.path.join(_BASE,'data','processed','battle_outcomes_1v1.csv'))
    ap.add_argument('--placements',default=os.path.join(_BASE,'data','processed','card_placements_1v1_labeled.csv'))
    ap.add_argument('--limit',type=int,default=0)
    ap.add_argument('--battle',type=str,default=None)
    ap.add_argument('--visualize',action='store_true')
    ap.add_argument('--verbose',action='store_true')
    args=ap.parse_args()
    print("=== Battle Replay Validation ===")
    outcomes=load_outcomes(args.outcomes)
    print(f"Loaded {len(outcomes)} outcomes")
    if args.battle:
        ids={args.battle}
    else:
        ids=set(outcomes.keys())
    print(f"Loading placements for {len(ids)} battles...")
    placements=load_placements(args.placements,ids)
    matched={bid for bid in ids if bid in placements}
    print(f"Matched {len(matched)} battles with placements")
    if args.battle:
        if args.battle not in matched:
            print(f"Battle {args.battle} not found in placements");return
        bids=[args.battle]
    else:
        bids=sorted(matched)
        if args.limit>0:bids=bids[:args.limit]
    tot=len(bids)
    wm=0;ce=0;cc=0;done=0
    print(f"Running {tot} battles...\n")
    for i,bid in enumerate(bids):
        if bid not in outcomes:continue
        g,info=replay_battle(bid,placements[bid],outcomes[bid],verbose=args.verbose)
        if info['win_match']:wm+=1
        if info['crown_exact']:ce+=1
        if info['crown_close']:cc+=1
        done+=1
        if not args.verbose and done%10==0:
            print(f"  [{done:4d}/{tot}] last={bid} wm={wm}/{done} ({100*wm/done:.1f}%)")
        if args.visualize and args.battle:
            from visualize import visualize as viz
            print(f"\nOpening visualizer for {bid}...")
            viz(g)
    print(f"\n=== Summary ===")
    if done==0:print("No battles replayed.");return
    print(f"Winner match: {wm}/{done} ({100*wm/done:.1f}%)")
    print(f"Crown exact:  {ce}/{done} ({100*ce/done:.1f}%)")
    print(f"Crown +/-1:   {cc}/{done} ({100*cc/done:.1f}%)")

if __name__=='__main__':
    main()
