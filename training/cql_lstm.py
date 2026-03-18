import sys,os,math,argparse,time
from pathlib import Path
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence,pad_packed_sequence
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from traj_dataloader import DEFAULT_TRAJ_PATH,TrajDataset,pad_collate

N_CONT=4;N_CARD_FIELDS=13;CARD_START=4;CARD_END=17

class CardEmb(nn.Module):
    def __init__(self,nv,ed=24):
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
    def forward(self,x):return x+self.pe[:,:x.size(1)]

class AttnPool(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.w=nn.Linear(d,1)
    def forward(self,out,lengths):
        B,T,D=out.shape
        mask=torch.arange(T,device=out.device).unsqueeze(0).expand(B,-1)>=lengths.unsqueeze(1)
        sc=self.w(out).squeeze(-1).masked_fill(mask,-1e4)
        a=torch.softmax(sc,dim=1).unsqueeze(-1)
        return (out*a).sum(dim=1)

class ResBlock(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(0.05),nn.Linear(d*2,d))
        self.ln=nn.LayerNorm(d)
    def forward(self,x):return self.ln(x+self.net(x))

class CQLCardLSTM(nn.Module):
    def __init__(self,nv,ed=24,hs=192,nl=3,dp=0.1):
        super().__init__()
        self.ce=CardEmb(nv,ed)
        isz=N_CONT+N_CARD_FIELDS*ed
        self.pe=PosEnc(isz)
        self.ln_in=nn.LayerNorm(isz)
        self.lstm=nn.LSTM(isz,hs,nl,batch_first=True,dropout=dp if nl>1 else 0)
        self.pool=AttnPool(hs)
        self.ln_out=nn.LayerNorm(hs)
        self.q_head=nn.Sequential(ResBlock(hs),ResBlock(hs),ResBlock(hs),nn.Linear(hs,nv))
        self.v_head=nn.Sequential(ResBlock(hs),nn.Linear(hs,1))
        self.xy_head=nn.Sequential(ResBlock(hs),nn.Linear(hs,2))
        self.time_head=nn.Sequential(ResBlock(hs),nn.Linear(hs,1))
        self.nv=nv
    def _encode(self,x,lengths):
        e=self.ln_in(self.pe(self.ce(x)))
        pk=pack_padded_sequence(e,lengths.cpu(),batch_first=True,enforce_sorted=False)
        o,_=self.lstm(pk)
        o,_=pad_packed_sequence(o,batch_first=True)
        return self.ln_out(self.pool(o,lengths))
    def forward(self,x,lengths):
        h=self._encode(x,lengths)
        return self.q_head(h),self.v_head(h),self.xy_head(h),self.time_head(h).squeeze(-1)

def topk_acc(logits,tgt,ks=(1,3,5)):
    r={}
    for k in ks:
        kk=min(k,logits.size(1))
        _,pred=logits.topk(kk,dim=1)
        r[k]=(pred==tgt.unsqueeze(1)).any(dim=1).float().mean().item()
    return r

def train_cql(csv_path=None,name='cql_run',mode='both',epochs=8,bs=48,lr=2e-4,
              hs=192,nl=3,ed=24,dp=0.1,cql_alpha=0.5,max_battles=None,val_frac=0.15):
    if csv_path is None:csv_path=DEFAULT_TRAJ_PATH
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')
    ds=TrajDataset(csv_path,skip_ability=True,mode=mode,max_battle_count=max_battles)
    nv=ds.get_num_cards()
    n=len(ds);nvl=max(1,int(n*val_frac));ntr=n-nvl
    tds,vds=random_split(ds,[ntr,nvl])
    tl=DataLoader(tds,batch_size=bs,shuffle=True,collate_fn=pad_collate,num_workers=0)
    vl=DataLoader(vds,batch_size=bs,shuffle=False,collate_fn=pad_collate,num_workers=0)
    print(f'Dataset: {n} samples ({ntr} train, {nvl} val), {nv} cards')
    model=CQLCardLSTM(nv,ed,hs,nl,dp).to(dev)
    nparams=sum(p.numel() for p in model.parameters())
    print(f'Model params: {nparams:,}')
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=1e-4)
    warmup=min(500,len(tl))
    def lr_fn(step):
        if step<warmup:return step/max(1,warmup)
        prog=(step-warmup)/max(1,len(tl)*epochs-warmup)
        return 0.5*(1+math.cos(math.pi*prog))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_fn)
    ce=nn.CrossEntropyLoss()
    mse=nn.MSELoss()
    ckdir=Path('checkpoints');ckdir.mkdir(exist_ok=True)
    best_val=float('inf');gstep=0;t0=time.time()
    scaler=torch.amp.GradScaler(enabled=dev.type=='cuda')
    for ep in range(epochs):
        model.train();eloss=0;ns=0
        pbar=tqdm(tl,desc=f'Epoch {ep+1}/{epochs}',unit='b')
        for x,lengths,target_xy,target_card in pbar:
            x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
            target_card=target_card.to(dev).long()
            target_xy=target_xy.to(dev,dtype=torch.float32)
            with torch.amp.autocast(device_type=dev.type,enabled=dev.type=='cuda'):
                q,v,xy_pred,t_pred=model(x,lengths)
                card_loss=ce(q,target_card)
                logsumexp=torch.logsumexp(q,dim=-1)
                qa=q.gather(1,target_card.unsqueeze(1)).squeeze(1)
                cql_loss=cql_alpha*(logsumexp-qa).mean()
                xy_loss=mse(xy_pred,target_xy[:,:2])
                t_loss=mse(t_pred,target_xy[:,2])
                loss=card_loss+cql_loss+0.5*xy_loss+0.2*t_loss
            if not torch.isfinite(loss):continue
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update()
            sched.step();gstep+=1
            b=x.size(0);eloss+=loss.item()*b;ns+=b
            pbar.set_postfix(l=f'{loss.item():.3f}',cql=f'{cql_loss.item():.3f}')
        if ns==0:continue
        model.eval();vloss=0;vn=0;vtop={1:0,3:0,5:0}
        with torch.no_grad():
            for x,lengths,target_xy,target_card in vl:
                x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
                target_card=target_card.to(dev).long()
                q,v,xy_pred,t_pred=model(x,lengths)
                l=ce(q,target_card)
                b=x.size(0);vloss+=l.item()*b;vn+=b
                tk=topk_acc(q,target_card)
                for k in vtop:vtop[k]+=tk.get(k,0)*b
        vl_loss=vloss/max(vn,1)
        vtop={k:v/max(vn,1) for k,v in vtop.items()}
        el=time.time()-t0
        print(f'Ep {ep+1} train:{eloss/ns:.4f} val:{vl_loss:.4f} top1:{vtop[1]:.4f} top3:{vtop[3]:.4f} top5:{vtop[5]:.4f} {el:.0f}s')
        if vl_loss<best_val:
            best_val=vl_loss
            torch.save({'state_dict':model.state_dict(),'vocab':ds.get_vocab(),
                         'config':{'nv':nv,'ed':ed,'hs':hs,'nl':nl,'dp':dp},
                         'epoch':ep,'val_loss':vl_loss,'vtop':vtop},
                        str(ckdir/f'{name}_best.pt'))
            print(f'  Best model saved (val={vl_loss:.4f})')
    torch.save({'state_dict':model.state_dict(),'vocab':ds.get_vocab(),
                'config':{'nv':nv,'ed':ed,'hs':hs,'nl':nl,'dp':dp}},
               str(ckdir/f'{name}_final.pt'))
    print(f'Final -> checkpoints/{name}_final.pt')

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=None)
    ap.add_argument('--name',default='cql_run')
    ap.add_argument('--mode',default='both')
    ap.add_argument('--epochs',type=int,default=8)
    ap.add_argument('--bs',type=int,default=48)
    ap.add_argument('--lr',type=float,default=2e-4)
    ap.add_argument('--hs',type=int,default=192)
    ap.add_argument('--nl',type=int,default=3)
    ap.add_argument('--ed',type=int,default=24)
    ap.add_argument('--cql-alpha',type=float,default=0.5)
    ap.add_argument('--max-battles',type=int,default=None)
    args=ap.parse_args()
    train_cql(csv_path=args.csv,name=args.name,mode=args.mode,epochs=args.epochs,
              bs=args.bs,lr=args.lr,hs=args.hs,nl=args.nl,ed=args.ed,
              cql_alpha=args.cql_alpha,max_battles=args.max_battles)

if __name__=='__main__':
    main()
