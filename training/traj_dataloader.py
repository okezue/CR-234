from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TrajMode = Literal["planner", "reacter", "both"]
DEFAULT_TRAJ_PATH = Path(__file__).resolve().parents[1]/"data"/"ready_data"/"traj.csv"
DEFAULT_TRAJ_OKEZUE_PATH = DEFAULT_TRAJ_PATH
PAD_IDX = 0

def build_card_vocab(df: pd.DataFrame) -> dict[str,int]:
    vocab: dict[str,int] = {"<pad>":0,"<unk>":1}
    cols=[c for c in (["card"]+[f"hand_{i}" for i in range(4)]+[f"deck_{i}" for i in range(8)]) if c in df.columns]
    if not cols:return vocab
    uniq=pd.unique(df[cols].astype(str).values.ravel())
    for name in uniq:
        name=(name or "").strip()
        if name and name!="nan" and name not in vocab:
            vocab[name]=len(vocab)
    return vocab

def encode_card(vocab:dict[str,int],name:str)->int:
    return vocab.get(str(name).strip() if pd.notna(name) else "",vocab.get("<unk>",1))

def _encode_column(vocab:dict[str,int],col:pd.Series)->np.ndarray:
    return col.map(lambda x:encode_card(vocab,x)).to_numpy(dtype=np.int64)

def pad_collate(batch):
    seqs,lengths,tx,ty,ttime,tcard=zip(*batch)
    lengths_t=torch.stack(lengths)
    padded=pad_sequence(seqs,batch_first=True,padding_value=float(PAD_IDX))
    target_xy=torch.stack((torch.stack(tx),torch.stack(ty),torch.stack(ttime)),dim=1)
    target_card=torch.stack(tcard)
    return padded,lengths_t,target_xy,target_card

class TrajDataset(Dataset):
    def __init__(self,csv_path=DEFAULT_TRAJ_PATH,skip_ability=True,
                 mode:TrajMode="both",max_battle_count=None):
        self.csv_path=Path(csv_path)
        self.skip_ability=skip_ability
        self.mode=mode
        self.max_battle_count=max_battle_count
        df=pd.read_csv(self.csv_path)
        df=df.groupby("battle_id").filter(lambda g:g["card"].notna().all())
        df["x"]=df["x"].fillna(499.0)
        df["y"]=df["y"].fillna(499.0)
        if skip_ability:
            df=df[~df["card"].astype(str).str.contains("ability",na=False)]
        self.vocab=build_card_vocab(df)
        self.idx_to_card={idx:name for name,idx in self.vocab.items()}
        self.num_cards=len(self.vocab)
        n_battles=df.battle_id.nunique()
        print(f"Total number of battles: {n_battles}")
        df=df.sort_values(["battle_id","time"])
        df.x=(df.x-499.0)/(17500.0-499.0)
        df.y=(df.y-499.0)/(31500.0-499.0)
        df.time=df.time/6000.0
        groups=list(df.groupby("battle_id",sort=False))
        self.samples=[]
        hand_cols=[f"hand_{i}" for i in range(4)]
        deck_cols=[f"deck_{i}" for i in range(8)]
        for i,(_bid,grp) in enumerate(groups):
            if self.max_battle_count is not None and i>=self.max_battle_count:break
            if i%500==0 and i>0:
                print(f"  {i} / {min(n_battles,self.max_battle_count or n_battles)} battles")
            grp=grp.reset_index(drop=True)
            if len(grp)<2:continue
            n_rows=len(grp)
            side_is_t=(grp["side"].astype(str).str.strip()=="t").to_numpy()
            card_idx=_encode_column(self.vocab,grp["card"])
            hand_idxs=np.column_stack([_encode_column(self.vocab,grp[c]) for c in hand_cols])
            deck_idxs=np.column_stack([_encode_column(self.vocab,grp[c]) for c in deck_cols])
            x_vals=grp["x"].to_numpy(dtype=np.float64)
            y_vals=grp["y"].to_numpy(dtype=np.float64)
            time_vals=grp["time"].to_numpy(dtype=np.float64)
            for raw_team in ("t","o"):
                side_enc=side_is_t.astype(np.float32) if raw_team=="t" else (~side_is_t).astype(np.float32)
                target_ok=side_is_t if raw_team=="t" else ~side_is_t
                if raw_team=="o":
                    x_feat=(1.0-x_vals).astype(np.float32)
                    y_feat=(1.0-y_vals).astype(np.float32)
                else:
                    x_feat=x_vals.astype(np.float32)
                    y_feat=y_vals.astype(np.float32)
                for t in range(1,n_rows):
                    if not target_ok[t]:continue
                    last_team=side_enc[t-1]>0.5
                    if self.mode=="planner" and not last_team:continue
                    if self.mode=="reacter" and last_team:continue
                    sl=slice(0,t)
                    steps=np.concatenate([
                        x_feat[sl,np.newaxis],y_feat[sl,np.newaxis],
                        time_vals[sl,np.newaxis].astype(np.float32),
                        side_enc[sl,np.newaxis],
                        hand_idxs[sl].astype(np.float32),
                        deck_idxs[sl].astype(np.float32),
                        card_idx[sl,np.newaxis].astype(np.float32),
                    ],axis=1)
                    seq_tensor=torch.from_numpy(steps)
                    raw_tx=float(grp.iloc[t]["x"])
                    raw_ty=float(grp.iloc[t]["y"])
                    target_x=(1.0-raw_tx) if raw_team=="o" else raw_tx
                    target_y=(1.0-raw_ty) if raw_team=="o" else raw_ty
                    target_time=float(time_vals[t])
                    target_card_idx=int(card_idx[t])
                    self.samples.append((seq_tensor,target_x,target_y,target_time,target_card_idx))
    def __len__(self):return len(self.samples)
    def __getitem__(self,idx):
        seq,tx,ty,tt,tc=self.samples[idx]
        length=torch.tensor(seq.size(0),dtype=torch.long)
        return (seq,length,torch.tensor(tx,dtype=torch.float32),
                torch.tensor(ty,dtype=torch.float32),
                torch.tensor(tt,dtype=torch.float32),
                torch.tensor(tc,dtype=torch.long))
    def get_vocab(self):return self.vocab.copy()
    def get_num_cards(self):return self.num_cards
    def get_card_name(self,idx):return self.idx_to_card.get(int(idx),"<unk>")

def get_traj_dataloader(csv_path=DEFAULT_TRAJ_PATH,batch_size=32,shuffle=True,
                        num_workers=0,skip_ability=True,mode:TrajMode="both",
                        max_battle_count=None):
    from torch.utils.data import DataLoader
    ds=TrajDataset(csv_path=csv_path,skip_ability=skip_ability,
                   mode=mode,max_battle_count=max_battle_count)
    return DataLoader(ds,batch_size=batch_size,shuffle=shuffle,
                      num_workers=num_workers,pin_memory=True,collate_fn=pad_collate)

if __name__=="__main__":
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader
    ds=TrajDataset(DEFAULT_TRAJ_PATH,skip_ability=True)
    print(f"TrajDataset: {len(ds)} samples, vocab size = {ds.get_num_cards()}")
    loader=DataLoader(ds,batch_size=8,shuffle=True,collate_fn=pad_collate)
    for x,lengths,target_xy,target_card in loader:
        print(f"x shape: {x.shape}, lengths: {lengths.tolist()}")
        print(f"target_xy shape: {target_xy.shape}, target_card shape: {target_card.shape}")
        print(f"first target: (x,y,time)=({target_xy[0,0].item():.4f}, {target_xy[0,1].item():.4f}, {target_xy[0,2].item():.4f}), card={ds.get_card_name(target_card[0].item())}")
        packed=pack_padded_sequence(x,lengths,batch_first=True,enforce_sorted=False)
        break
    print("Data loader OK.")
