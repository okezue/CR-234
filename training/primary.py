import sys
from pathlib import Path
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from traj_dataloader import (DEFAULT_TRAJ_OKEZUE_PATH,TrajDataset,TrajMode,pad_collate)

RAW_FEAT_SIZE=17
CONT_FEAT_SIZE=4
NUM_CAT_FIELDS=13

class TrajLSTM(nn.Module):
    def __init__(self,num_cards,hidden_size=128,num_layers=2,emb_dim=16,dropout=0.2):
        super().__init__()
        self.num_cards=num_cards
        self.hidden_size=hidden_size
        self.num_layers=num_layers
        self.emb_dim=emb_dim
        self.card_emb=nn.Embedding(num_embeddings=num_cards,embedding_dim=emb_dim,padding_idx=0)
        lstm_input_size=CONT_FEAT_SIZE+NUM_CAT_FIELDS*emb_dim
        lstm_dropout=dropout if num_layers>1 else 0.0
        self.lstm=nn.LSTM(input_size=lstm_input_size,hidden_size=hidden_size,
                          num_layers=num_layers,batch_first=True,dropout=lstm_dropout)
        self.trunk=nn.Sequential(nn.LayerNorm(hidden_size),nn.Linear(hidden_size,hidden_size*2),nn.ReLU())
        self.card_head=nn.Sequential(
            nn.Linear(hidden_size*2,hidden_size),nn.LayerNorm(hidden_size),nn.ReLU(),
            nn.Linear(hidden_size,hidden_size*2),nn.ReLU(),
            nn.Linear(hidden_size*2,hidden_size*2),nn.ReLU(),
            nn.Linear(hidden_size*2,hidden_size),nn.LayerNorm(hidden_size),nn.ReLU(),
            nn.Linear(hidden_size,num_cards))
    def _build_lstm_input(self,x):
        cont=x[:,:,:CONT_FEAT_SIZE]
        cats=x[:,:,CONT_FEAT_SIZE:].long().clamp(min=0,max=self.num_cards-1)
        emb=self.card_emb(cats).flatten(start_dim=2)
        return torch.cat([cont,emb],dim=-1)
    def forward(self,x,lengths):
        z=self._build_lstm_input(x)
        packed=pack_padded_sequence(z,lengths.cpu(),batch_first=True,enforce_sorted=False)
        packed_out,_=self.lstm(packed)
        outputs,_=pad_packed_sequence(packed_out,batch_first=True)
        B=x.size(0)
        idx=(lengths-1).clamp(min=0)
        last_out=outputs[torch.arange(B,device=x.device),idx,:]
        h=self.trunk(last_out)
        pred_card=self.card_head(h)
        return pred_card

def _unpack_dataset_item(item):
    if len(item)==6:
        seq,length,tx,ty,tt,tc=item
        txy=torch.tensor([float(tx),float(ty),float(tt)],dtype=torch.float32)
        return seq,length,txy,int(tc)
    elif len(item)==4:
        seq,length,txy,tc=item
        if not torch.is_tensor(txy):txy=torch.tensor(txy,dtype=torch.float32)
        return seq,length,txy.float(),int(tc)
    else:
        raise ValueError(f"Unexpected dataset item format with len={len(item)}")

def _evaluate(model,dataloader,criterion_card,device):
    model.eval()
    total_card=0.0;correct_card=0;n_samples=0
    with torch.no_grad():
        for x,lengths,target_xy,target_card in dataloader:
            x=x.to(device,dtype=torch.float32);lengths=lengths.to(device)
            target_card=target_card.to(device).long()
            pred_card=model(x,lengths)
            loss_card=criterion_card(pred_card,target_card)
            b=x.size(0);total_card+=loss_card.item()*b
            correct_card+=(pred_card.argmax(dim=1)==target_card).sum().item()
            n_samples+=b
    model.train()
    if n_samples==0:return 0.0,0.0
    return total_card/n_samples,correct_card/n_samples

def _save_checkpoint(path,model,optimizer,epoch,history,config):
    torch.save({"epoch":epoch,"state_dict":model.state_dict(),
                "optimizer_state_dict":optimizer.state_dict(),
                "history":history,"config":config},path)

def train_traj_lstm(csv_path=None,name="Noname",mode:TrajMode="both",
                    num_epochs=10,batch_size=64,hidden_size=128,num_layers=2,
                    emb_dim=16,learning_rate=1e-3,dropout=0.2,
                    loss_t_weight=1.0,loss_xy_weight=1.0,loss_card_weight=1.0,
                    skip_ability=False,val_frac=0.2,max_battle_count=None,
                    plot_curve=True,curve_save_path=None,
                    checkpoint_dir="checkpoints",dataset_cache_path=None,seed=42):
    if csv_path is None:csv_path=DEFAULT_TRAJ_OKEZUE_PATH
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if dataset_cache_path is None:dataset_cache_path=f"ds_{mode}.pt"
    dataset_cache_path=Path(dataset_cache_path)
    if dataset_cache_path.exists():
        print(f"Loading cached dataset from {dataset_cache_path}")
        full_dataset=torch.load(dataset_cache_path,weights_only=False)
    else:
        print("Building dataset from CSV")
        full_dataset=TrajDataset(csv_path=csv_path,skip_ability=skip_ability,
                                 mode=mode,max_battle_count=max_battle_count)
        torch.save(full_dataset,dataset_cache_path)
        print(f"Saved dataset cache to {dataset_cache_path}")
    n=len(full_dataset)
    if n<2:raise ValueError(f"Dataset too small: {n}")
    n_val=max(1,int(n*val_frac));n_train=n-n_val
    if n_train<=0:raise ValueError(f"Train split is empty. n={n}, val_frac={val_frac}")
    split_gen=torch.Generator().manual_seed(seed)
    train_dataset,val_dataset=random_split(full_dataset,[n_train,n_val],generator=split_gen)
    train_loader=DataLoader(train_dataset,batch_size=batch_size,shuffle=True,collate_fn=pad_collate)
    val_loader=DataLoader(val_dataset,batch_size=batch_size,shuffle=False,collate_fn=pad_collate)
    num_cards=full_dataset.get_num_cards()
    print(f"Dataset size: {n}\nTrain size:   {n_train}\nVal size:     {n_val}\nNum cards:    {num_cards}")
    model=TrajLSTM(num_cards=num_cards,hidden_size=hidden_size,num_layers=num_layers,
                   emb_dim=emb_dim,dropout=dropout).to(device)
    pytorch_total_params=sum(p.numel() for p in model.parameters())
    print(f"Total Parameters in model: {pytorch_total_params}")
    criterion_card=nn.CrossEntropyLoss()
    optimizer=torch.optim.AdamW(model.parameters(),lr=learning_rate,weight_decay=1e-4)
    history={"train_card":[],"val_card":[],"val_acc":[]}
    log_dir=Path("runs")/f"{name}_{mode}_hs{hidden_size}_nl{num_layers}_emb{emb_dim}"
    log_dir.mkdir(parents=True,exist_ok=True)
    writer=SummaryWriter(log_dir=str(log_dir))
    global_step=0
    ckpt_dir=Path(checkpoint_dir) if checkpoint_dir is not None else None
    if ckpt_dir is not None:ckpt_dir.mkdir(parents=True,exist_ok=True)
    config={"num_cards":num_cards,"hidden_size":hidden_size,"num_layers":num_layers,
            "emb_dim":emb_dim,"dropout":dropout,"mode":mode}
    model.train()
    for epoch in range(num_epochs):
        epoch_loss_card=0.0;n_samples=0
        pbar=tqdm(train_loader,desc=f"Epoch {epoch+1}/{num_epochs}",leave=True,unit="batch")
        for x,lengths,target_xy,target_card in pbar:
            x=x.to(device,dtype=torch.float32);lengths=lengths.to(device)
            target_xy=target_xy.to(device,dtype=torch.float32)
            target_card=target_card.to(device).long()
            optimizer.zero_grad()
            pred_card=model(x,lengths)
            loss_card=criterion_card(pred_card,target_card)
            loss=loss_card_weight*loss_card
            if not torch.isfinite(loss):
                tqdm.write(f"Skipping non-finite batch: loss_card={loss_card.item():.4f}")
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),max_norm=1.0)
            optimizer.step()
            writer.add_scalar("batch/loss_card",loss_card.item(),global_step)
            writer.add_scalar("batch/loss_total",loss.item(),global_step)
            global_step+=1
            b=x.size(0);epoch_loss_card+=loss_card.item()*b;n_samples+=b
            pbar.set_postfix(loss_card=f"{loss_card.item():.3f}")
        if n_samples==0:
            print(f"Epoch {epoch+1}/{num_epochs} - no valid batches");continue
        train_card=epoch_loss_card/n_samples
        val_card,val_acc=_evaluate(model,val_loader,criterion_card,device)
        history["train_card"].append(train_card)
        history["val_card"].append(val_card)
        history["val_acc"].append(val_acc)
        writer.add_scalar("epoch/train_card",train_card,epoch+1)
        writer.add_scalar("epoch/val_card",val_card,epoch+1)
        writer.add_scalar("epoch/val_acc",val_acc,epoch+1)
        print(f"Epoch {epoch+1}/{num_epochs} | train_card={train_card:.4f} | val_card={val_card:.4f} val_acc={val_acc:.4f}")
        if ckpt_dir is not None:
            ckpt_path=ckpt_dir/f"{mode}_epoch_{epoch+1}.pt"
            _save_checkpoint(ckpt_path,model,optimizer,epoch+1,history,config)
            print(f"Checkpoint saved to {ckpt_path}")
    if plot_curve and len(history["train_card"])>0:
        _plot_training_curve(history,save_path=curve_save_path,mode=mode)
    writer.close()
    return model,history,config

def _plot_training_curve(history,save_path=None,mode="both"):
    epochs=range(1,len(history["train_card"])+1)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    axes[0].plot(epochs,history["train_card"],label="train")
    axes[0].plot(epochs,history["val_card"],label="val")
    axes[0].set_title("Card loss");axes[0].set_xlabel("Epoch")
    axes[0].legend();axes[0].grid(True,alpha=0.3)
    axes[1].plot(epochs,history["val_acc"],label="val acc")
    axes[1].set_title("Val card accuracy");axes[1].set_xlabel("Epoch")
    axes[1].legend();axes[1].grid(True,alpha=0.3)
    plt.tight_layout()
    if save_path is None:save_path=Path.cwd()/f"training_curve_{mode}.png"
    else:save_path=Path(save_path);save_path.parent.mkdir(parents=True,exist_ok=True)
    plt.savefig(save_path,dpi=150);plt.close()
    print(f"Training curve saved to {save_path}")

def _load_model_from_checkpoint(model_path,device,fallback_num_cards,
                                fallback_hidden_size=128,fallback_num_layers=2,
                                fallback_emb_dim=16,fallback_dropout=0.2):
    state=torch.load(model_path,map_location=device,weights_only=False)
    if isinstance(state,dict) and "config" in state and "state_dict" in state:
        cfg=state["config"]
        model=TrajLSTM(num_cards=cfg.get("num_cards",fallback_num_cards),
                       hidden_size=cfg.get("hidden_size",fallback_hidden_size),
                       num_layers=cfg.get("num_layers",fallback_num_layers),
                       emb_dim=cfg.get("emb_dim",fallback_emb_dim),
                       dropout=cfg.get("dropout",fallback_dropout)).to(device)
        model.load_state_dict(state["state_dict"])
        return model,cfg
    model=TrajLSTM(num_cards=fallback_num_cards,hidden_size=fallback_hidden_size,
                   num_layers=fallback_num_layers,emb_dim=fallback_emb_dim,
                   dropout=fallback_dropout).to(device)
    model.load_state_dict(state)
    return model,{"num_cards":fallback_num_cards,"hidden_size":fallback_hidden_size,
                  "num_layers":fallback_num_layers,"emb_dim":fallback_emb_dim,
                  "dropout":fallback_dropout}

def test_saved_model(model_path,csv_path=None,cached_ds_path=None,
                     mode:TrajMode="both",num_examples=3,hidden_size=128,
                     num_layers=2,emb_dim=16,dropout=0.2,skip_ability=False,
                     max_battle_count=500,seed=42):
    if csv_path is None:csv_path=DEFAULT_TRAJ_OKEZUE_PATH
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if cached_ds_path is None:
        dataset=TrajDataset(csv_path=csv_path,skip_ability=skip_ability,
                            mode=mode,max_battle_count=max_battle_count)
    else:
        dataset=torch.load(cached_ds_path,weights_only=False)
    num_cards=dataset.get_num_cards()
    model,cfg=_load_model_from_checkpoint(model_path=model_path,device=device,
                                          fallback_num_cards=num_cards,
                                          fallback_hidden_size=hidden_size,
                                          fallback_num_layers=num_layers,
                                          fallback_emb_dim=emb_dim,
                                          fallback_dropout=dropout)
    model.eval()
    print("Loaded model config:");print(cfg)
    rng=torch.Generator().manual_seed(seed)
    indices=torch.randperm(len(dataset),generator=rng).tolist()[:num_examples]
    if not indices:print("No examples in dataset.");return
    for ei,idx in enumerate(indices):
        item=dataset[idx]
        seq,length,target_xy,target_card=_unpack_dataset_item(item)
        L=seq.size(0)
        print(f"\n{'='*60}\nExample {ei+1}/{num_examples} (seq length = {L})\n{'='*60}")
        print("Trajectory:")
        for i in range(L):
            cn=dataset.get_card_name(int(seq[i,16].item()))
            sd="team" if seq[i,3].item()>0.5 else "opponent"
            print(f"  step {i+1}: time={seq[i,2].item():.4f} side={sd} card={cn} x={seq[i,0].item():.4f} y={seq[i,1].item():.4f}")
        xb=seq.unsqueeze(0).to(device,dtype=torch.float32)
        lb=length.unsqueeze(0).to(device)
        with torch.no_grad():pred_card=model(xb,lb)
        pn=dataset.get_card_name(pred_card[0].argmax().item())
        tn=dataset.get_card_name(int(target_card))
        print(f"\nGround truth next move:\n  card={tn}")
        print(f"Model predicts next move:\n  card={pn}")
    print(f"\n{'='*60}\nDone.")

if __name__=="__main__":
    if len(sys.argv)<3:
        raise ValueError("Usage: python primary.py [planner|reacter] [name]")
    mode=sys.argv[1];name=str(sys.argv[2])
    if mode not in {"planner","reacter"}:
        raise ValueError("Mode must be 'planner' or 'reacter'")
    print(f"Training {mode}")
    model,history,config=train_traj_lstm(
        mode=mode,name=name,num_epochs=4,batch_size=64,hidden_size=128,
        num_layers=2,emb_dim=16,learning_rate=1e-3,dropout=0.2,val_frac=0.15,
        plot_curve=True,curve_save_path=f"results/{name}_{mode}_curve.png",
        checkpoint_dir=f"checkpoints/{name}",dataset_cache_path=f"ds_{mode}.pt",
        loss_t_weight=1000.0,loss_xy_weight=100.0,loss_card_weight=1.0)
    final_model_path=f"{mode}_model.pt"
    torch.save({"state_dict":model.state_dict(),"config":config,"history":history},final_model_path)
    print(f"Final model saved to {final_model_path}")
    test_saved_model(model_path=final_model_path,mode=mode,num_examples=3,
                     max_battle_count=500,cached_ds_path=f"ds_{mode}.pt")
