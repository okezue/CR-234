import sys,os,math,argparse,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from traj_dataloader_v3 import DEFAULT_TRAJ_PATH,TrajDatasetV3,pad_collate_v3

N_CONT=4;N_CARD_FIELDS=13;CARD_START=4;CARD_END=17

class FocalLoss(nn.Module):
    def __init__(self,gamma=2.0,alpha=None,reduction='mean'):
        super().__init__()
        self.gamma=gamma
        self.alpha=alpha
        self.reduction=reduction
    def forward(self,logits,targets):
        ce=F.cross_entropy(logits,targets,reduction='none')
        pt=torch.exp(-ce)
        focal_loss=((1-pt)**self.gamma)*ce
        if self.alpha is not None:
            alpha_t=self.alpha.gather(0,targets)
            focal_loss=alpha_t*focal_loss
        if self.reduction=='mean':return focal_loss.mean()
        elif self.reduction=='sum':return focal_loss.sum()
        return focal_loss

class CardEmb(nn.Module):
    def __init__(self,nv,ed=48):
        super().__init__()
        self.emb=nn.Embedding(nv,ed,padding_idx=0)
        self.ed=ed
    def forward(self,x):
        cont=x[...,:N_CONT]
        ids=x[...,CARD_START:CARD_END].long().clamp(min=0)
        e=self.emb(ids).view(*ids.shape[:-1],-1)
        return torch.cat([cont,e],dim=-1)

class RotaryPE(nn.Module):
    def __init__(self,d,maxlen=2000):
        super().__init__()
        inv_freq=1.0/(10000**(torch.arange(0,d,2).float()/d))
        self.register_buffer('inv_freq',inv_freq)
        self.maxlen=maxlen
    def forward(self,x):
        B,T,D=x.shape
        pos=torch.arange(T,device=x.device).float()
        sincos=torch.einsum('i,j->ij',pos,self.inv_freq)
        sin,cos=sincos.sin(),sincos.cos()
        x1,x2=x[...,::2],x[...,1::2]
        return torch.stack([x1*cos-x2*sin,x1*sin+x2*cos],dim=-1).flatten(-2)

class TransformerBlock(nn.Module):
    def __init__(self,d,nheads=8,dp=0.1,ff_mult=4):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,nheads,batch_first=True,dropout=dp)
        self.ln1=nn.LayerNorm(d)
        self.ffn=nn.Sequential(nn.Linear(d,d*ff_mult),nn.GELU(),nn.Dropout(dp),nn.Linear(d*ff_mult,d))
        self.ln2=nn.LayerNorm(d)
        self.dp=nn.Dropout(dp)
    def forward(self,x,mask=None,is_causal=True):
        if is_causal and mask is None:
            T=x.size(1)
            mask=torch.triu(torch.ones(T,T,device=x.device),diagonal=1).bool()
        attn_out,_=self.attn(x,x,x,attn_mask=mask)
        x=self.ln1(x+self.dp(attn_out))
        x=self.ln2(x+self.dp(self.ffn(x)))
        return x

class CrossAttnBlock(nn.Module):
    def __init__(self,d,nheads=8,dp=0.1):
        super().__init__()
        self.cross_attn=nn.MultiheadAttention(d,nheads,batch_first=True,dropout=dp)
        self.ln=nn.LayerNorm(d)
        self.dp=nn.Dropout(dp)
    def forward(self,q,kv,kv_mask=None):
        out,_=self.cross_attn(q,kv,kv,key_padding_mask=kv_mask)
        return self.ln(q+self.dp(out))

class GamePhaseMoE(nn.Module):
    def __init__(self,d,n_experts=4):
        super().__init__()
        self.n_experts=n_experts
        self.gate=nn.Sequential(nn.Linear(6,32),nn.GELU(),nn.Linear(32,n_experts))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d)) for _ in range(n_experts)])
        self.ln=nn.LayerNorm(d)
    def forward(self,x,board_state):
        weights=F.softmax(self.gate(board_state),dim=-1)
        out=torch.zeros_like(x)
        for i,expert in enumerate(self.experts):
            w=weights[:,i:i+1].unsqueeze(1)
            out=out+w*expert(x)
        return self.ln(x+out)

class ReacterV4(nn.Module):
    def __init__(self,nv,ed=48,hs=384,nl=6,nheads=8,dp=0.15):
        super().__init__()
        self.ce=CardEmb(nv,ed)
        isz=N_CONT+N_CARD_FIELDS*ed
        self.proj=nn.Linear(isz,hs)
        self.rope=RotaryPE(hs)
        self.ln_in=nn.LayerNorm(hs)
        self.blocks=nn.ModuleList([TransformerBlock(hs,nheads,dp) for _ in range(nl)])
        self.opp_proj=nn.Linear(isz,hs)
        self.opp_blocks=nn.ModuleList([TransformerBlock(hs,nheads,dp,ff_mult=2) for _ in range(3)])
        self.cross_attn=nn.ModuleList([CrossAttnBlock(hs,nheads,dp) for _ in range(2)])
        self.opp_card_emb=nn.Embedding(nv,hs)
        self.state_proj=nn.Sequential(nn.Linear(6,hs//2),nn.GELU(),nn.Linear(hs//2,hs))
        self.moe=GamePhaseMoE(hs,n_experts=4)
        self.card_type_head=nn.Linear(hs,8)
        self.card_head=nn.Sequential(nn.Linear(hs*2+hs,hs),nn.GELU(),nn.LayerNorm(hs),
                                     nn.Linear(hs,hs),nn.GELU(),nn.Linear(hs,nv))
        self.xy_head=nn.Sequential(nn.Linear(hs*2+hs,hs//2),nn.GELU(),nn.Linear(hs//2,2))
        self.time_head=nn.Sequential(nn.Linear(hs*2+hs,hs//4),nn.GELU(),nn.Linear(hs//4,1))
        self.opp_pred_head=nn.Sequential(nn.Linear(hs*2+hs,hs),nn.GELU(),nn.Linear(hs,nv))
        self.nv=nv
        self.hs=hs
    def _encode_seq(self,x,lengths):
        B,T,_=x.shape
        e=self.ce(x)
        e=self.proj(e)
        e=self.rope(e)
        e=self.ln_in(e)
        mask=torch.arange(T,device=x.device).unsqueeze(0)>=lengths.unsqueeze(1)
        for blk in self.blocks:
            e=blk(e)
        return e,mask
    def _encode_opp(self,opp_x,opp_lengths):
        B,T,_=opp_x.shape
        e=self.ce(opp_x)
        e=self.opp_proj(e)
        e=self.rope(e)
        e=self.ln_in(e)
        mask=torch.arange(T,device=opp_x.device).unsqueeze(0)>=opp_lengths.unsqueeze(1)
        for blk in self.opp_blocks:
            e=blk(e,is_causal=False)
        return e,mask
    def _pool(self,h,mask):
        mask_f=mask.float().unsqueeze(-1)
        h_masked=h.masked_fill(mask.unsqueeze(-1),0)
        lens=(~mask).sum(dim=1,keepdim=True).clamp(min=1).float()
        return h_masked.sum(dim=1)/lens
    def forward(self,x,opp_x,lengths,opp_lengths,opp_last_card,board_state):
        h_self,self_mask=self._encode_seq(x,lengths)
        h_opp,opp_mask=self._encode_opp(opp_x,opp_lengths)
        for ca in self.cross_attn:
            h_self=ca(h_self,h_opp,opp_mask)
        h_self=self.moe(h_self,board_state)
        h_pooled=self._pool(h_self,self_mask)
        h_opp_pooled=self._pool(h_opp,opp_mask)
        opp_card_h=self.opp_card_emb(opp_last_card.clamp(min=0,max=self.nv-1))
        state_h=self.state_proj(board_state)
        h_final=torch.cat([h_pooled,h_opp_pooled+opp_card_h,state_h],dim=-1)
        card_logits=self.card_head(h_final)
        xy=self.xy_head(h_final)
        t=self.time_head(h_final).squeeze(-1)
        opp_pred=self.opp_pred_head(h_final)
        return card_logits,xy,t,opp_pred

def topk_acc(logits,tgt,ks=(1,3,5)):
    r={}
    for k in ks:
        kk=min(k,logits.size(1))
        _,pred=logits.topk(kk,dim=1)
        r[k]=(pred==tgt.unsqueeze(1)).any(dim=1).float().mean().item()
    return r

def train_reacter_v4(csv_path=None,name='reacter_v4',epochs=25,bs=32,lr=1.5e-4,
                     hs=384,nl=6,ed=48,dp=0.15,cql_alpha=0.25,focal_gamma=2.0,
                     max_battles=None,val_frac=0.15,warmup_epochs=2):
    if csv_path is None:csv_path=DEFAULT_TRAJ_PATH
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}',flush=True)
    ds=TrajDatasetV3(csv_path,skip_ability=True,mode='reacter',max_battle_count=max_battles,opp_context=12)
    nv=ds.get_num_cards()
    n=len(ds);nvl=max(1,int(n*val_frac));ntr=n-nvl
    tds,vds=random_split(ds,[ntr,nvl])
    tl=DataLoader(tds,batch_size=bs,shuffle=True,collate_fn=pad_collate_v3,num_workers=0,pin_memory=True)
    vl=DataLoader(vds,batch_size=bs,shuffle=False,collate_fn=pad_collate_v3,num_workers=0)
    print(f'Dataset: {n} samples ({ntr} train, {nvl} val), {nv} cards',flush=True)
    model=ReacterV4(nv,ed,hs,nl,nheads=8,dp=dp).to(dev)
    nparams=sum(p.numel() for p in model.parameters())
    print(f'Model params: {nparams:,}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.05,betas=(0.9,0.98))
    total_steps=len(tl)*epochs
    warmup_steps=len(tl)*warmup_epochs
    def lr_fn(step):
        if step<warmup_steps:return step/max(1,warmup_steps)
        prog=(step-warmup_steps)/max(1,total_steps-warmup_steps)
        return 0.5*(1+math.cos(math.pi*prog))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_fn)
    focal=FocalLoss(gamma=focal_gamma)
    mse=nn.MSELoss()
    ckdir=Path('checkpoints');ckdir.mkdir(exist_ok=True)
    best_val=float('inf');best_top1=0;gstep=0;t0=time.time()
    scaler=torch.amp.GradScaler(enabled=dev.type=='cuda')
    for ep in range(epochs):
        model.train();eloss=0;ns=0
        pbar=tqdm(tl,desc=f'Epoch {ep+1}/{epochs}',unit='b')
        for x,opp_x,lengths,opp_lengths,target_xy,target_card,opp_last_card,board_state,opp_next_card in pbar:
            x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
            opp_x=opp_x.to(dev,dtype=torch.float32);opp_lengths=opp_lengths.to(dev)
            opp_last_card=opp_last_card.to(dev).long()
            board_state=board_state.to(dev,dtype=torch.float32)
            target_card=target_card.to(dev).long()
            target_xy=target_xy.to(dev,dtype=torch.float32)
            opp_next_card=opp_next_card.to(dev).long()
            with torch.amp.autocast(device_type=dev.type,enabled=dev.type=='cuda'):
                q,xy_pred,t_pred,opp_pred=model(x,opp_x,lengths,opp_lengths,opp_last_card,board_state)
                card_loss=focal(q,target_card)
                logsumexp=torch.logsumexp(q,dim=-1)
                qa=q.gather(1,target_card.unsqueeze(1)).squeeze(1)
                cql_loss=cql_alpha*(logsumexp-qa).mean()
                xy_loss=mse(xy_pred,target_xy[:,:2])
                t_loss=mse(t_pred,target_xy[:,2])
                opp_loss=F.cross_entropy(opp_pred,opp_next_card)*0.4
                loss=card_loss+cql_loss+0.3*xy_loss+0.15*t_loss+opp_loss
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
            for x,opp_x,lengths,opp_lengths,target_xy,target_card,opp_last_card,board_state,opp_next_card in vl:
                x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
                opp_x=opp_x.to(dev,dtype=torch.float32);opp_lengths=opp_lengths.to(dev)
                opp_last_card=opp_last_card.to(dev).long()
                board_state=board_state.to(dev,dtype=torch.float32)
                target_card=target_card.to(dev).long()
                q,_,_,_=model(x,opp_x,lengths,opp_lengths,opp_last_card,board_state)
                l=focal(q,target_card)
                b=x.size(0);vloss+=l.item()*b;vn+=b
                tk=topk_acc(q,target_card)
                for k in vtop:vtop[k]+=tk.get(k,0)*b
        vl_loss=vloss/max(vn,1)
        vtop={k:v/max(vn,1) for k,v in vtop.items()}
        el=time.time()-t0
        print(f'Ep {ep+1} train:{eloss/ns:.4f} val:{vl_loss:.4f} top1:{vtop[1]:.4f} top3:{vtop[3]:.4f} top5:{vtop[5]:.4f} {el:.0f}s',flush=True)
        if vtop[1]>best_top1:
            best_top1=vtop[1]
            best_val=vl_loss
            torch.save({'state_dict':model.state_dict(),'vocab':ds.get_vocab(),
                        'config':{'nv':nv,'ed':ed,'hs':hs,'nl':nl,'dp':dp},
                        'epoch':ep,'val_loss':vl_loss,'vtop':vtop},
                       str(ckdir/f'{name}_best.pt'))
            print(f'  Best model saved (top1={vtop[1]:.4f})',flush=True)
    torch.save({'state_dict':model.state_dict(),'vocab':ds.get_vocab(),
                'config':{'nv':nv,'ed':ed,'hs':hs,'nl':nl,'dp':dp}},
               str(ckdir/f'{name}_final.pt'))
    print(f'Final -> checkpoints/{name}_final.pt',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=None)
    ap.add_argument('--name',default='reacter_v4')
    ap.add_argument('--epochs',type=int,default=25)
    ap.add_argument('--bs',type=int,default=32)
    ap.add_argument('--lr',type=float,default=1.5e-4)
    ap.add_argument('--hs',type=int,default=384)
    ap.add_argument('--nl',type=int,default=6)
    ap.add_argument('--ed',type=int,default=48)
    ap.add_argument('--cql-alpha',type=float,default=0.25)
    ap.add_argument('--focal-gamma',type=float,default=2.0)
    ap.add_argument('--max-battles',type=int,default=None)
    args=ap.parse_args()
    train_reacter_v4(csv_path=args.csv,name=args.name,epochs=args.epochs,
                     bs=args.bs,lr=args.lr,hs=args.hs,nl=args.nl,ed=args.ed,
                     cql_alpha=args.cql_alpha,focal_gamma=args.focal_gamma,
                     max_battles=args.max_battles)

if __name__=='__main__':
    main()
