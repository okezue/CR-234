import random
import zlib
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset,Subset
from sim.cards import load

ROOT=Path(__file__).resolve().parents[1]
DEFAULT_TRAJ_PATH=ROOT/"data"/"ready_data"/"traj.csv"
PAD_IDX=0
HAND_COLS=[f"hand_{i}" for i in range(4)];DECK_COLS=[f"deck_{i}" for i in range(8)]
CARD_COLS=["card"]+HAND_COLS+DECK_COLS
N_CONT=4;N_CARD_FIELDS=len(CARD_COLS);BS_DIM=6
ERATE=((120,1),(240,2),(1e9,3))  # real 1v1 ramps (2x at 2:00, 3x at 4:00); sim.game switches to 3x at 180 s
EBASE=2.8

def load_costs(default=3.0):
    # keyed by the RoyaleAPI hyphen names used in the trajectory csv
    db=load()
    return {api:float(db["cards"][k]["cost"]) if isinstance(db["cards"][k]["cost"],(int,float)) else default for api,k in db["aliases"]["api"].items()
            if k in db["cards"]}

def erate(t):return next(m for hi,m in ERATE if t<hi)

def egen(a,b):
    e=0.0;lo=0.0
    for hi,m in ERATE:
        e+=m*max(0.0,min(b,hi)-max(a,lo));lo=hi
    return e/EBASE

def elixir_track(secs,mine,costs):
    e=np.empty(len(secs),np.float32);cur=5.0;prev=0.0
    for i,t in enumerate(secs):
        cur=min(10.0,cur+egen(prev,t));prev=t
        if mine[i]:cur=max(0.0,cur-costs[i])
        e[i]=cur
    return e

def build_vocab(df):
    vocab={"<pad>":0,"<unk>":1}
    for c in CARD_COLS:
        for n in df[c].dropna().astype(str).str.strip().unique():
            if n and n!="nan" and n not in vocab:vocab[n]=len(vocab)
    return vocab

def collate(batch):
    seq,opp,tx,ty,tt,tc,olc,bs,onc=zip(*batch)
    ln=torch.tensor([s.size(0) for s in seq]);oln=torch.tensor([s.size(0) for s in opp])
    x=pad_sequence(seq,batch_first=True,padding_value=float(PAD_IDX))
    ox=pad_sequence(opp,batch_first=True,padding_value=float(PAD_IDX))
    txy=torch.tensor(np.stack([tx,ty,tt],1),dtype=torch.float32)
    return x,ox,ln,oln,txy,torch.tensor(tc),torch.tensor(olc),torch.stack(bs),torch.tensor(onc)

def battle_split(ds,val_frac=0.15):
    h={b:zlib.crc32(str(b).encode())%1000<val_frac*1000 for b in set(ds.bids)}
    val=np.array([h[b] for b in ds.bids],dtype=bool)
    return Subset(ds,np.flatnonzero(~val).tolist()),Subset(ds,np.flatnonzero(val).tolist())

class TrajDataset(Dataset):
    def __init__(self,csv_path=DEFAULT_TRAJ_PATH,mode="both",max_battles=None,opp_context=12,mirror_prob=0.5,costs=None,skip_ability=True):
        self.mode=mode;self.opp_context=opp_context;self.mirror_prob=mirror_prob
        self.costs=load_costs() if costs is None else costs
        df=pd.read_csv(csv_path,low_memory=False).dropna(subset=["card"])
        df["card"]=df["card"].astype(str).str.strip()
        if skip_ability:df=df[~df["card"].str.contains("ability",na=False)]
        if max_battles:
            bids=df.battle_id.unique()
            if max_battles<len(bids):df=df[df.battle_id.isin(np.random.default_rng(0).choice(bids,max_battles,replace=False))]
        df=df.sort_values(["battle_id","time"]).reset_index(drop=True)
        self.vocab=build_vocab(df);self.idx_to_card={v:k for k,v in self.vocab.items()};self.num_cards=len(self.vocab)
        df["cost"]=df["card"].map(self.costs).fillna(3.0)
        for c in CARD_COLS:df[c]=df[c].astype(str).str.strip().map(self.vocab).fillna(1).astype(np.int64)
        df["x"]=(df["x"].fillna(499.0)-499.0)/(17500.0-499.0)
        df["y"]=(df["y"].fillna(499.0)-499.0)/(31500.0-499.0)
        df["secs"]=df["time"]/20.0;df["time"]=df["time"]/6000.0
        self.samples=[];self.bids=[]
        for bid,grp in df.groupby("battle_id",sort=False):
            if len(grp)>=3:self._battle(bid,grp.reset_index(drop=True))
        print(f"Dataset: {len(self.samples)} samples, {len(set(self.bids))} battles, {self.num_cards} cards",flush=True)
    def _battle(self,bid,g):
        n=len(g)
        is_t=(g["side"].astype(str).str.strip()=="t").to_numpy()
        card=g["card"].to_numpy(np.int64);hd=g[HAND_COLS+DECK_COLS].to_numpy(np.float32)
        x=g["x"].to_numpy(np.float32);y=g["y"].to_numpy(np.float32);tm=g["time"].to_numpy(np.float32)
        secs=g["secs"].to_numpy(np.float64);cost=g["cost"].to_numpy(np.float64)
        elix={"t":elixir_track(secs,is_t,cost),"o":elixir_track(secs,~is_t,cost)}
        for team,other in (("t","o"),("o","t")):
            mine=is_t if team=="t" else ~is_t
            fx,fy=(x,y) if team=="t" else (1-x,1-y)
            steps=np.column_stack([fx,fy,tm,mine.astype(np.float32),hd,card.astype(np.float32)])
            osteps=np.column_stack([1-fx,1-fy,tm,(~mine).astype(np.float32),hd,card.astype(np.float32)])
            opp_idx=np.flatnonzero(~mine)
            for t in range(1,n):
                if not mine[t] or (self.mode=="planner" and not mine[t-1]) or (self.mode=="reacter" and mine[t-1]):continue
                ob=opp_idx[opp_idx<t][-self.opp_context:];oa=opp_idx[opp_idx>t]
                opp=osteps[ob] if len(ob) else np.zeros((1,steps.shape[1]),np.float32)
                bs=torch.tensor([tm[t-1],elix[team][t-1]/10,elix[other][t-1]/10,mine[:t].sum()/20,(~mine[:t]).sum()/20,erate(secs[t-1])/3],dtype=torch.float32)
                self.samples.append((torch.from_numpy(steps[:t]),torch.from_numpy(opp),float(fx[t]),float(fy[t]),float(tm[t]),
                                     int(card[t]),int(card[ob[-1]]) if len(ob) else 0,bs,int(card[oa[0]]) if len(oa) else 0))
                self.bids.append(bid)
    def __len__(self):return len(self.samples)
    def __getitem__(self,i):
        seq,opp,tx,ty,tt,tc,olc,bs,onc=self.samples[i]
        if random.random()<self.mirror_prob:
            seq=seq.clone();seq[:,0]=1-seq[:,0];opp=opp.clone();opp[:,0]=1-opp[:,0];tx=1-tx
        return seq,opp,tx,ty,tt,tc,olc,bs,onc
