from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

TrajMode=Literal["planner","reacter","both"]
DEFAULT_TRAJ_PATH=Path(__file__).resolve().parents[1]/"data"/"ready_data"/"traj.csv"
PAD_IDX=0

def build_card_vocab(df:pd.DataFrame)->dict[str,int]:
    vocab:dict[str,int]={"<pad>":0,"<unk>":1}
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

def pad_collate_v3(batch):
    seqs,opp_seqs,lengths,opp_lengths,tx,ty,ttime,tcard,opp_last_card,board_state,opp_next_card=zip(*batch)
    lengths_t=torch.stack(lengths)
    opp_lengths_t=torch.stack(opp_lengths)
    padded=pad_sequence(seqs,batch_first=True,padding_value=float(PAD_IDX))
    opp_padded=pad_sequence(opp_seqs,batch_first=True,padding_value=float(PAD_IDX))
    target_xy=torch.stack((torch.stack(tx),torch.stack(ty),torch.stack(ttime)),dim=1)
    target_card=torch.stack(tcard)
    opp_last_card_t=torch.stack(opp_last_card)
    board_state_t=torch.stack(board_state)
    opp_next_card_t=torch.stack(opp_next_card)
    return padded,opp_padded,lengths_t,opp_lengths_t,target_xy,target_card,opp_last_card_t,board_state_t,opp_next_card_t

class TrajDatasetV3(Dataset):
    def __init__(self,csv_path=DEFAULT_TRAJ_PATH,skip_ability=True,
                 mode:TrajMode="both",max_battle_count=None,opp_context=8):
        self.csv_path=Path(csv_path)
        self.skip_ability=skip_ability
        self.mode=mode
        self.max_battle_count=max_battle_count
        self.opp_context=opp_context
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
            if len(grp)<3:continue
            n_rows=len(grp)
            side_is_t=(grp["side"].astype(str).str.strip()=="t").to_numpy()
            card_idx=_encode_column(self.vocab,grp["card"])
            hand_idxs=np.column_stack([_encode_column(self.vocab,grp[c]) for c in hand_cols])
            deck_idxs=np.column_stack([_encode_column(self.vocab,grp[c]) for c in deck_cols])
            x_vals=grp["x"].to_numpy(dtype=np.float64)
            y_vals=grp["y"].to_numpy(dtype=np.float64)
            time_vals=grp["time"].to_numpy(dtype=np.float64)
            result=str(grp.iloc[0].get('result','')).strip()
            win_label=1.0 if result=='W' else 0.0 if result=='L' else 0.5
            for raw_team in ("t","o"):
                side_enc=side_is_t.astype(np.float32) if raw_team=="t" else (~side_is_t).astype(np.float32)
                opp_side_enc=(~side_is_t).astype(np.float32) if raw_team=="t" else side_is_t.astype(np.float32)
                target_ok=side_is_t if raw_team=="t" else ~side_is_t
                opp_mask=~side_is_t if raw_team=="t" else side_is_t
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
                    opp_plays_before_t=np.where(opp_mask[:t])[0]
                    if len(opp_plays_before_t)==0:
                        opp_steps=np.zeros((1,steps.shape[1]),dtype=np.float32)
                        opp_last_card_idx=0
                    else:
                        opp_recent=opp_plays_before_t[-self.opp_context:]
                        opp_x=(1.0-x_vals[opp_recent]).astype(np.float32) if raw_team=="t" else x_vals[opp_recent].astype(np.float32)
                        opp_y=(1.0-y_vals[opp_recent]).astype(np.float32) if raw_team=="t" else y_vals[opp_recent].astype(np.float32)
                        opp_steps=np.concatenate([
                            opp_x[:,np.newaxis],opp_y[:,np.newaxis],
                            time_vals[opp_recent,np.newaxis].astype(np.float32),
                            opp_side_enc[opp_recent,np.newaxis],
                            hand_idxs[opp_recent].astype(np.float32),
                            deck_idxs[opp_recent].astype(np.float32),
                            card_idx[opp_recent,np.newaxis].astype(np.float32),
                        ],axis=1)
                        opp_last_card_idx=int(card_idx[opp_plays_before_t[-1]])
                    opp_seq_tensor=torch.from_numpy(opp_steps)
                    team_plays=np.sum(side_enc[:t]>0.5)
                    opp_plays=np.sum(opp_side_enc[:t]>0.5)
                    curr_time=float(time_vals[t-1]) if t>0 else 0.0
                    elixir_approx=min(10.0,(curr_time*0.5+5.0))/10.0
                    board_state=torch.tensor([
                        curr_time,
                        elixir_approx,
                        float(team_plays)/20.0,
                        float(opp_plays)/20.0,
                        win_label,
                        1.0 if curr_time>0.5 else 0.0
                    ],dtype=torch.float32)
                    opp_plays_after_t=np.where(opp_mask[t:])[0]
                    if len(opp_plays_after_t)>0:
                        opp_next_idx=int(card_idx[t+opp_plays_after_t[0]])
                    else:
                        opp_next_idx=0
                    raw_tx=float(grp.iloc[t]["x"])
                    raw_ty=float(grp.iloc[t]["y"])
                    target_x=(1.0-raw_tx) if raw_team=="o" else raw_tx
                    target_y=(1.0-raw_ty) if raw_team=="o" else raw_ty
                    target_time=float(time_vals[t])
                    target_card_idx=int(card_idx[t])
                    self.samples.append((seq_tensor,opp_seq_tensor,target_x,target_y,target_time,
                                        target_card_idx,opp_last_card_idx,board_state,opp_next_idx))
    def __len__(self):return len(self.samples)
    def __getitem__(self,idx):
        seq,opp_seq,tx,ty,tt,tc,opp_last,board_state,opp_next=self.samples[idx]
        length=torch.tensor(seq.size(0),dtype=torch.long)
        opp_length=torch.tensor(opp_seq.size(0),dtype=torch.long)
        return (seq,opp_seq,length,opp_length,
                torch.tensor(tx,dtype=torch.float32),
                torch.tensor(ty,dtype=torch.float32),
                torch.tensor(tt,dtype=torch.float32),
                torch.tensor(tc,dtype=torch.long),
                torch.tensor(opp_last,dtype=torch.long),
                board_state,
                torch.tensor(opp_next,dtype=torch.long))
    def get_vocab(self):return self.vocab.copy()
    def get_num_cards(self):return self.num_cards
    def get_card_name(self,idx):return self.idx_to_card.get(int(idx),"<unk>")

if __name__=="__main__":
    from torch.utils.data import DataLoader
    ds=TrajDatasetV3(DEFAULT_TRAJ_PATH,skip_ability=True,mode="reacter",max_battle_count=100)
    print(f"TrajDatasetV3: {len(ds)} samples, vocab size = {ds.get_num_cards()}")
    loader=DataLoader(ds,batch_size=8,shuffle=True,collate_fn=pad_collate_v3)
    for x,opp_x,lengths,opp_lengths,target_xy,target_card,opp_last,board_state,opp_next in loader:
        print(f"x: {x.shape}, opp_x: {opp_x.shape}")
        print(f"board_state: {board_state.shape}")
        print(f"opp_last: {opp_last}, opp_next: {opp_next}")
        break
    print("DataLoader V3 OK.")
