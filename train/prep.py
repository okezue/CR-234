import csv
import argparse
from pathlib import Path
from collections import defaultdict
from sim.replay import norm
from train.trajData import DEFAULT_TRAJ_PATH,HAND_COLS,DECK_COLS

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_SRC=ROOT/"data"/"processed"/"card_placements_1v1_labeled.csv"
COLS=["battle_id","x","y","card","time","side","result","team_crowns","opp_crowns","hero","evo"]+HAND_COLS+DECK_COLS

def parse(r):
    base,evo,hero=norm(r["card"])
    if base is None or r["card"].startswith("ability"):return None
    return base,int(float(r.get("evo") or evo)),int(float(r.get("hero") or hero))

class Cycle:
    def __init__(self,cards):
        self.deck=[cards[i%len(cards)] for i in range(8)]  # first 8 distinct plays, cycled if fewer were seen
        self.hand=self.deck[:4];self.nxt=self.deck[4];self.q=self.deck[5:]
    def play(self,c):
        if c not in self.hand:return
        self.hand[self.hand.index(c)]=self.nxt
        if self.q:self.nxt=self.q.pop(0);self.q.append(c)
        else:self.nxt=c

def prep(src,dst,max_battles=None):
    battles=defaultdict(list)
    with open(src) as f:
        for r in csv.DictReader(f):battles[r["battle_id"]].append(r)
    bids=sorted(battles)[:max_battles] if max_battles else sorted(battles)
    Path(dst).parent.mkdir(parents=True,exist_ok=True)
    nb=ns=0
    with open(dst,"w",newline="") as out:
        w=csv.DictWriter(out,fieldnames=COLS);w.writeheader()
        for bid in bids:
            rows=[(r,p) for r in sorted(battles[bid],key=lambda r:int(float(r.get("time","0")))) if (p:=parse(r))]
            seen={"t":[],"o":[]}
            for r,(c,_,_) in rows:
                sd=r.get("side","t").strip()
                if c not in seen[sd]:seen[sd].append(c)
            if len(rows)<4 or min(len(seen["t"]),len(seen["o"]))<4:continue
            cyc={sd:Cycle(v) for sd,v in seen.items()}
            for r,(c,evo,hero) in rows:
                sd=r.get("side","t").strip();cy=cyc[sd]
                row={"battle_id":bid,"x":r["x"],"y":r["y"],"card":c,"time":r["time"],"side":sd,"result":r.get("result",""),
                     "team_crowns":r.get("team_crowns",""),"opp_crowns":r.get("opp_crowns",""),"hero":hero,"evo":evo,
                     **dict(zip(HAND_COLS,cy.hand)),**dict(zip(DECK_COLS,cy.deck))}
                w.writerow(row);cy.play(c);ns+=1
            nb+=1
    print(f"{nb} battles, {ns} rows -> {dst}")

if __name__=="__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--in",dest="src",default=DEFAULT_SRC);ap.add_argument("--out",default=DEFAULT_TRAJ_PATH);ap.add_argument("--max-battles",type=int,default=None)
    a=ap.parse_args();prep(a.src,a.out,a.max_battles)
