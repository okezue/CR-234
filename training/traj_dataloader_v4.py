from pathlib import Path
from typing import Literal
import numpy as np
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import random

TrajMode=Literal["planner","reacter","both"]
DEFAULT_TRAJ_PATH=Path(__file__).resolve().parents[1]/"data"/"ready_data"/"traj.csv"
PAD_IDX=0

def build_card_vocab_from_file(csv_path,sample_rows=2000000):
    vocab={"<pad>":0,"<unk>":1}
    cols=['card']+[f'hand_{i}' for i in range(4)]+[f'deck_{i}' for i in range(8)]
    df=pd.read_csv(csv_path,usecols=[c for c in cols if c],nrows=sample_rows,low_memory=False)
    for c in df.columns:
        for name in df[c].dropna().unique():
            name=str(name).strip()
            if name and name!='nan' and name not in vocab:
                vocab[name]=len(vocab)
    return vocab

def encode_card(vocab,name):
    return vocab.get(str(name).strip() if pd.notna(name) else "",vocab.get("<unk>",1))

def pad_collate_v4(batch):
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

class TrajDatasetV4(Dataset):
    def __init__(self,csv_path=DEFAULT_TRAJ_PATH,skip_ability=True,
                 mode:TrajMode="both",max_battle_count=None,opp_context=8,
                 mirror_prob=0.5,chunk_size=5000):
        self.csv_path=Path(csv_path)
        self.mode=mode
        self.opp_context=opp_context
        self.mirror_prob=mirror_prob
        self.skip_ability=skip_ability
        print(f"Building vocab from {csv_path}...")
        self.vocab=build_card_vocab_from_file(csv_path)
        self.idx_to_card={v:k for k,v in self.vocab.items()}
        self.num_cards=len(self.vocab)
        print(f"  Vocab: {self.num_cards} cards")
        print(f"Processing battles in chunks of {chunk_size}...")
        self.samples=[]
        reader=pd.read_csv(csv_path,chunksize=500000,low_memory=False)
        all_rows=[]
        for chunk in reader:
            all_rows.append(chunk)
        df=pd.concat(all_rows,ignore_index=True)
        del all_rows
        df=df.dropna(subset=['card'])
        df["x"]=df["x"].fillna(499.0)
        df["y"]=df["y"].fillna(499.0)
        if skip_ability:
            df=df[~df["card"].astype(str).str.contains("ability",na=False)]
        df=df.sort_values(["battle_id","time"])
        df.x=(df.x-499.0)/(17500.0-499.0)
        df.y=(df.y-499.0)/(31500.0-499.0)
        df.time=df.time/6000.0
        bids=df.battle_id.unique()
        n_battles=len(bids)
        print(f"  {n_battles} battles total")
        if max_battle_count and max_battle_count<n_battles:
            bids=np.random.choice(bids,size=max_battle_count,replace=False)
            df=df[df.battle_id.isin(set(bids))]
            n_battles=max_battle_count
        hand_cols=[f"hand_{i}" for i in range(4)]
        deck_cols=[f"deck_{i}" for i in range(8)]
        groups=df.groupby("battle_id",sort=False)
        processed=0
        for bid,grp in groups:
            grp=grp.reset_index(drop=True)
            if len(grp)<3:continue
            self._process_battle(grp)
            processed+=1
            if processed%chunk_size==0:
                print(f"  {processed}/{n_battles} battles, {len(self.samples)} samples")
        print(f"Dataset: {len(self.samples)} samples from {processed} battles, {self.num_cards} cards")
    def _process_battle(self,grp):
        n_rows=len(grp)
        hand_cols=[f"hand_{i}" for i in range(4)]
        deck_cols=[f"deck_{i}" for i in range(8)]
        side_is_t=(grp["side"].astype(str).str.strip()=="t").to_numpy()
        card_idx=grp["card"].map(lambda x:encode_card(self.vocab,x)).to_numpy(dtype=np.int64)
        hand_idxs=np.column_stack([grp[c].map(lambda x:encode_card(self.vocab,x)).to_numpy(dtype=np.int64) for c in hand_cols]) if hand_cols[0] in grp.columns else np.ones((n_rows,4),dtype=np.int64)
        deck_idxs=np.column_stack([grp[c].map(lambda x:encode_card(self.vocab,x)).to_numpy(dtype=np.int64) for c in deck_cols]) if deck_cols[0] in grp.columns else np.ones((n_rows,8),dtype=np.int64)
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
                team_plays=np.sum(side_enc[:t]>0.5)
                opp_plays=np.sum(opp_side_enc[:t]>0.5)
                curr_time=float(time_vals[t-1]) if t>0 else 0.0
                elixir_approx=min(10.0,(curr_time*0.5+5.0))/10.0
                bs=torch.tensor([curr_time,elixir_approx,float(team_plays)/20.0,
                    float(opp_plays)/20.0,win_label,1.0 if curr_time>0.5 else 0.0],dtype=torch.float32)
                opp_plays_after_t=np.where(opp_mask[t:])[0]
                opp_next_idx=int(card_idx[t+opp_plays_after_t[0]]) if len(opp_plays_after_t)>0 else 0
                raw_tx=float(grp.iloc[t]["x"])
                raw_ty=float(grp.iloc[t]["y"])
                target_x=(1.0-raw_tx) if raw_team=="o" else raw_tx
                target_y=(1.0-raw_ty) if raw_team=="o" else raw_ty
                self.samples.append((
                    torch.from_numpy(steps),torch.from_numpy(opp_steps),
                    target_x,target_y,float(time_vals[t]),int(card_idx[t]),
                    opp_last_card_idx,bs,opp_next_idx))
    def __len__(self):return len(self.samples)
    def __getitem__(self,idx):
        seq,opp_seq,tx,ty,tt,tc,opp_last,board_state,opp_next=self.samples[idx]
        if self.mirror_prob>0 and random.random()<self.mirror_prob:
            seq=seq.clone()
            seq[:,0]=1.0-seq[:,0]
            opp_seq=opp_seq.clone()
            opp_seq[:,0]=1.0-opp_seq[:,0]
            tx=1.0-tx
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
