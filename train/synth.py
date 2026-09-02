import csv
import argparse
import random
from pathlib import Path
from train.trajData import DEFAULT_TRAJ_PATH,load_costs,egen
from train.feats import card_type
from train.prep import COLS,Cycle,HAND_COLS,DECK_COLS

def battle(bid,pool,costs,w):
    cyc={s:Cycle(random.sample(pool,8)) for s in "to"};ex={"t":5.0,"o":5.0};last={"t":0,"o":0}
    res=random.choice("WLD");tc,oc={"W":(2,1),"L":(1,2),"D":(1,1)}[res]
    t=0
    while t<6000:
        t+=random.randint(20,200);s=random.choice("to")
        ex[s]=min(10.0,ex[s]+egen(last[s]/20,t/20));last[s]=t
        hand=[c for c in cyc[s].hand if costs[c]<=ex[s]]
        if not hand:continue
        c=random.choice(hand);ex[s]-=costs[c];y=random.randint(500,15500)
        w.writerow({"battle_id":bid,"x":random.randint(500,17500),"y":y if s=="t" else 32000-y,"card":c,"time":t,"side":s,"result":res,
                    "team_crowns":tc,"opp_crowns":oc,"hero":0,"evo":0,**dict(zip(HAND_COLS,cyc[s].hand)),**dict(zip(DECK_COLS,cyc[s].deck))})
        cyc[s].play(c)

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--out",default=DEFAULT_TRAJ_PATH);ap.add_argument("--n",type=int,default=300);ap.add_argument("--seed",type=int,default=0)
    a=ap.parse_args();random.seed(a.seed)
    costs=load_costs();pool=[c for c in costs if card_type(c) in ("troop","spell","building") and c!="mirror"]
    Path(a.out).parent.mkdir(parents=True,exist_ok=True)
    with open(a.out,"w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=COLS);w.writeheader()
        for i in range(a.n):battle(f"synth{i:04d}",pool,costs,w)
    print(f"{a.n} battles -> {a.out}")

if __name__=="__main__":main()
