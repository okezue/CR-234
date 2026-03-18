import csv,os,sys
from pathlib import Path
from collections import defaultdict
import random

SRC=Path(__file__).resolve().parents[1]/"data"/"processed"/"card_placements_1v1_labeled.csv"
DST=Path(__file__).resolve().parents[1]/"data"/"ready_data"/"traj.csv"

def prep(src=SRC,dst=DST,max_battles=None):
    dst.parent.mkdir(parents=True,exist_ok=True)
    battles=defaultdict(list)
    print(f"Reading {src}...")
    with open(src) as f:
        for r in csv.DictReader(f):
            bid=r['battle_id']
            battles[bid].append(r)
    print(f"  {len(battles)} battles loaded")
    bids=sorted(battles.keys())
    if max_battles:bids=bids[:max_battles]
    out=open(dst,'w',newline='')
    cols=['battle_id','x','y','card','time','side','result','team_crowns','opp_crowns']
    cols+=[f'hand_{i}' for i in range(4)]
    cols+=[f'deck_{i}' for i in range(8)]
    w=csv.DictWriter(out,fieldnames=cols)
    w.writeheader()
    ns=0;nb=0
    for bid in bids:
        rows=sorted(battles[bid],key=lambda r:int(float(r.get('time','0'))))
        if len(rows)<4:continue
        t_cards=[];o_cards=[]
        for r in rows:
            c=r['card'].replace('-','_')
            if c.startswith('ability'):continue
            sd=r.get('side','t').strip()
            if sd=='t' and c not in t_cards:t_cards.append(c)
            elif sd=='o' and c not in o_cards:o_cards.append(c)
        if len(t_cards)<4 or len(o_cards)<4:continue
        t_deck=t_cards[:8]
        o_deck=o_cards[:8]
        while len(t_deck)<8:t_deck.append(t_deck[len(t_deck)%len(t_cards)])
        while len(o_deck)<8:o_deck.append(o_deck[len(o_deck)%len(o_cards)])
        t_hand=list(t_deck[:4]);t_nxt=t_deck[4] if len(t_deck)>4 else t_deck[0]
        t_q=list(t_deck[5:8]) if len(t_deck)>5 else []
        o_hand=list(o_deck[:4]);o_nxt=o_deck[4] if len(o_deck)>4 else o_deck[0]
        o_q=list(o_deck[5:8]) if len(o_deck)>5 else []
        res=rows[0].get('result','')
        tc=rows[0].get('team_crowns','')
        oc=rows[0].get('opp_crowns','')
        for r in rows:
            c=r['card'].replace('-','_')
            if c.startswith('ability'):continue
            sd=r.get('side','t').strip()
            hand=t_hand if sd=='t' else o_hand
            deck=t_deck if sd=='t' else o_deck
            nxt=t_nxt if sd=='t' else o_nxt
            q=t_q if sd=='t' else o_q
            row={'battle_id':bid,'x':r['x'],'y':r['y'],'card':c,
                 'time':r['time'],'side':sd,'result':res,'team_crowns':tc,'opp_crowns':oc}
            for i in range(4):
                row[f'hand_{i}']=hand[i] if i<len(hand) else deck[i%len(deck)]
            for i in range(8):
                row[f'deck_{i}']=deck[i]
            w.writerow(row)
            ns+=1
            if c in hand:
                idx=hand.index(c)
                hand[idx]=nxt
                if sd=='t':
                    if t_q:t_nxt=t_q.pop(0);t_q.append(c)
                    else:t_nxt=c
                else:
                    if o_q:o_nxt=o_q.pop(0);o_q.append(c)
                    else:o_nxt=c
        nb+=1
        if nb%1000==0:print(f"  {nb} battles, {ns} samples...")
    out.close()
    print(f"Done: {nb} battles, {ns} samples -> {dst}")

if __name__=="__main__":
    mb=int(sys.argv[1]) if len(sys.argv)>1 else None
    prep(max_battles=mb)
