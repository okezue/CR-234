import sys,os,argparse,math
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from traj_dataloader import DEFAULT_TRAJ_PATH,TrajDataset,pad_collate

N_CONT=4
N_CARD_FIELDS=13
CARD_START=4
CARD_END=17

class CardEmbedding(nn.Module):
    def __init__(self,nv,ed=16):
        super().__init__()
        self.emb=nn.Embedding(nv,ed,padding_idx=0)
        self.ed=ed
    def forward(self,x):
        cont=x[...,:N_CONT]
        ids=x[...,CARD_START:CARD_END].long().clamp(min=0)
        e=self.emb(ids).view(*ids.shape[:-1],-1)
        return torch.cat([cont,e],dim=-1)

class PosEnc(nn.Module):
    def __init__(self,d,maxlen=2000):
        super().__init__()
        pe=torch.zeros(maxlen,d)
        pos=torch.arange(maxlen).unsqueeze(1).float()
        div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.0)/d))
        pe[:,0::2]=torch.sin(pos*div)
        if d>1:pe[:,1::2]=torch.cos(pos*div[:d//2])
        self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self,x):
        return x+self.pe[:,:x.size(1)]

class AttnPool(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.w=nn.Linear(d,1)
    def forward(self,out,lengths):
        B,T,D=out.shape
        mask=torch.arange(T,device=out.device).unsqueeze(0).expand(B,-1)>=lengths.unsqueeze(1)
        sc=self.w(out).squeeze(-1)
        sc=sc.masked_fill(mask,-1e4)
        a=torch.softmax(sc,dim=1).unsqueeze(-1)
        return (out*a).sum(dim=1)

class ResBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,d))
        self.ln=nn.LayerNorm(d)
    def forward(self,x):return self.ln(x+self.net(x))

class CardLSTM(nn.Module):
    def __init__(self,nv,ed=16,hs=128,nl=2,dp=0.1):
        super().__init__()
        self.ce=CardEmbedding(nv,ed)
        isz=N_CONT+N_CARD_FIELDS*ed
        self.pe=PosEnc(isz)
        self.ln_in=nn.LayerNorm(isz)
        self.lstm=nn.LSTM(isz,hs,nl,batch_first=True,dropout=dp if nl>1 else 0)
        self.pool=AttnPool(hs)
        self.ln_out=nn.LayerNorm(hs)
        self.head=nn.Sequential(ResBlock(hs),ResBlock(hs),nn.Linear(hs,nv))
    def forward(self,x,lengths):
        e=self.ln_in(self.pe(self.ce(x)))
        pk=pack_padded_sequence(e,lengths.cpu(),batch_first=True,enforce_sorted=False)
        o,_=self.lstm(pk)
        o,_=pad_packed_sequence(o,batch_first=True)
        h=self.ln_out(self.pool(o,lengths))
        return self.head(h)

class XYLSTM(nn.Module):
    def __init__(self,nv,ed=16,hs=128,nl=2,dp=0.1):
        super().__init__()
        self.ce=CardEmbedding(nv,ed)
        isz=N_CONT+N_CARD_FIELDS*ed
        self.pe=PosEnc(isz)
        self.ln_in=nn.LayerNorm(isz)
        self.lstm=nn.LSTM(isz,hs,nl,batch_first=True,dropout=dp if nl>1 else 0)
        self.pool=AttnPool(hs)
        self.ln_out=nn.LayerNorm(hs)
        self.head=nn.Sequential(ResBlock(hs),nn.Linear(hs,2))
    def forward(self,x,lengths):
        e=self.ln_in(self.pe(self.ce(x)))
        pk=pack_padded_sequence(e,lengths.cpu(),batch_first=True,enforce_sorted=False)
        o,_=self.lstm(pk)
        o,_=pad_packed_sequence(o,batch_first=True)
        h=self.ln_out(self.pool(o,lengths))
        return self.head(h)

class TimeLSTM(nn.Module):
    def __init__(self,nv,ed=16,hs=64,nl=2,dp=0.1):
        super().__init__()
        self.ce=CardEmbedding(nv,ed)
        isz=N_CONT+N_CARD_FIELDS*ed
        self.pe=PosEnc(isz)
        self.ln_in=nn.LayerNorm(isz)
        self.lstm=nn.LSTM(isz,hs,nl,batch_first=True,dropout=dp if nl>1 else 0)
        self.pool=AttnPool(hs)
        self.ln_out=nn.LayerNorm(hs)
        self.head=nn.Sequential(ResBlock(hs),nn.Linear(hs,1))
    def forward(self,x,lengths):
        e=self.ln_in(self.pe(self.ce(x)))
        pk=pack_padded_sequence(e,lengths.cpu(),batch_first=True,enforce_sorted=False)
        o,_=self.lstm(pk)
        o,_=pad_packed_sequence(o,batch_first=True)
        h=self.ln_out(self.pool(o,lengths))
        return self.head(h).squeeze(-1)

def topk_acc(logits,tgt,ks=(1,3,5)):
    r={}
    for k in ks:
        if k>logits.size(1):k=logits.size(1)
        _,pred=logits.topk(k,dim=1)
        r[k]=(pred==tgt.unsqueeze(1)).any(dim=1).float().mean().item()
    return r

def _train_one(task,model,tl,vl,ds,epochs,lr,dev,name,grad_accum=1,log_dir="runs"):
    try:
        from torch.utils.tensorboard import SummaryWriter
        writer=SummaryWriter(log_dir=f"{log_dir}/{name}_{task}")
    except ImportError:
        writer=None
    if task=="card":
        crit=nn.CrossEntropyLoss()
    else:
        crit=nn.MSELoss()
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    warmup_steps=min(500,len(tl)*epochs//10)
    def lr_fn(step):
        if step<warmup_steps:return step/max(1,warmup_steps)
        prog=(step-warmup_steps)/(max(1,len(tl)*epochs-warmup_steps))
        return 0.5*(1+math.cos(math.pi*prog))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_fn)
    scaler=torch.amp.GradScaler(enabled=dev.type=="cuda")
    ckdir=Path("checkpoints");ckdir.mkdir(exist_ok=True)
    best_val=float('inf');gstep=0
    for ep in range(epochs):
        model.train();eloss=0.0;ns=0
        pbar=tqdm(tl,desc=f"[{task}] Epoch {ep+1}/{epochs}",unit="batch")
        opt.zero_grad()
        for bi,(x,lengths,target_xy,target_card) in enumerate(pbar):
            x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
            with torch.amp.autocast(device_type=dev.type,enabled=dev.type=="cuda"):
                pred=model(x,lengths)
                if task=="card":
                    target_card=target_card.to(dev)
                    loss=crit(pred,target_card)/grad_accum
                elif task=="xy":
                    tgt=target_xy[:,:2].to(dev,dtype=torch.float32)
                    loss=crit(pred,tgt)/grad_accum
                else:
                    tgt=target_xy[:,2].to(dev,dtype=torch.float32)
                    loss=crit(pred,tgt)/grad_accum
            if not torch.isfinite(loss):pbar.set_postfix(loss="nan");continue
            scaler.scale(loss).backward()
            if (bi+1)%grad_accum==0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(),1.0)
                scaler.step(opt);scaler.update();opt.zero_grad()
            sched.step();gstep+=1
            rl=loss.item()*grad_accum;b=x.size(0);eloss+=rl*b;ns+=b
            pbar.set_postfix(loss=f"{rl:.4f}")
            if writer and gstep%50==0:
                writer.add_scalar(f"{task}/train_loss",rl,gstep)
        if ns==0:continue
        vl_loss,vl_extra=_eval_task(task,model,vl,crit,dev,ds)
        print(f"[{task}] Epoch {ep+1} train:{eloss/ns:.4f} val:{vl_loss:.4f}",end="")
        if task=="card":
            print(f" top1:{vl_extra.get(1,0):.4f} top3:{vl_extra.get(3,0):.4f} top5:{vl_extra.get(5,0):.4f}")
        else:
            print()
        if writer:
            writer.add_scalar(f"{task}/val_loss",vl_loss,ep)
            if task=="card":
                for k,v in vl_extra.items():writer.add_scalar(f"{task}/top{k}_acc",v,ep)
        if vl_loss<best_val:
            best_val=vl_loss
            torch.save({"state_dict":model.state_dict(),"task":task,"epoch":ep,
                         "val_loss":vl_loss,"vocab":ds.get_vocab()},
                        str(ckdir/f"{name}_{task}_best.pt"))
            print(f"  Saved best {task} model (val={vl_loss:.4f})")
    if writer:writer.close()
    return model

def _eval_task(task,model,vl,crit,dev,ds):
    model.eval();total=0.0;n=0;all_topk={1:0,3:0,5:0}
    with torch.no_grad():
        for x,lengths,target_xy,target_card in vl:
            x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
            pred=model(x,lengths)
            if task=="card":
                target_card=target_card.to(dev)
                loss=crit(pred,target_card)
                tk=topk_acc(pred,target_card)
                b=x.size(0)
                for k in all_topk:all_topk[k]+=tk.get(k,0)*b
            elif task=="xy":
                tgt=target_xy[:,:2].to(dev,dtype=torch.float32)
                loss=crit(pred,tgt);b=x.size(0)
            else:
                tgt=target_xy[:,2].to(dev,dtype=torch.float32)
                loss=crit(pred,tgt);b=x.size(0)
            total+=loss.item()*b;n+=b
    model.train()
    if n==0:return 0.0,{}
    extra={k:v/n for k,v in all_topk.items()} if task=="card" else {}
    return total/n,extra

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--task",choices=["card","xy","time","all"],default="all")
    ap.add_argument("--name",default="run")
    ap.add_argument("--epochs",type=int,default=10)
    ap.add_argument("--bs",type=int,default=32)
    ap.add_argument("--lr",type=float,default=3e-4)
    ap.add_argument("--emb-dim",type=int,default=16)
    ap.add_argument("--max-battles",type=int,default=None)
    ap.add_argument("--grad-accum",type=int,default=1)
    ap.add_argument("--val-frac",type=float,default=0.2)
    ap.add_argument("--mode",choices=["planner","reacter","both"],default="both")
    args=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=TrajDataset(DEFAULT_TRAJ_PATH,skip_ability=True,mode=args.mode,max_battle_count=args.max_battles)
    nv=ds.get_num_cards()
    n=len(ds);nvl=max(1,int(n*args.val_frac));ntr=n-nvl
    tds,vds=random_split(ds,[ntr,nvl])
    tl=DataLoader(tds,batch_size=args.bs,shuffle=True,collate_fn=pad_collate)
    vl=DataLoader(vds,batch_size=args.bs,shuffle=False,collate_fn=pad_collate)
    tasks=[args.task] if args.task!="all" else ["card","xy","time"]
    for t in tasks:
        print(f"\n{'='*40}\nTraining {t} model\n{'='*40}")
        if t=="card":
            m=CardLSTM(nv,ed=args.emb_dim).to(dev)
        elif t=="xy":
            m=XYLSTM(nv,ed=args.emb_dim).to(dev)
        else:
            m=TimeLSTM(nv,ed=args.emb_dim).to(dev)
        _train_one(t,m,tl,vl,ds,args.epochs,args.lr,dev,args.name,args.grad_accum)
    print("\nAll done.")

if __name__=="__main__":
    main()
