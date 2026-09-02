import math
import argparse
import time
from pathlib import Path
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from train.trajData import DEFAULT_TRAJ_PATH,TrajDataset,collate,battle_split,N_CONT,N_CARD_FIELDS,BS_DIM

class CardEmb(nn.Module):
    def __init__(self,nv,ed):
        super().__init__()
        self.emb=nn.Embedding(nv,ed,padding_idx=0)
    def forward(self,x):
        ids=x[...,N_CONT:N_CONT+N_CARD_FIELDS].long().clamp(min=0)
        return torch.cat([x[...,:N_CONT],self.emb(ids).flatten(-2)],-1)

class RotaryPE(nn.Module):
    def __init__(self,d):
        super().__init__()
        self.register_buffer('inv_freq',1.0/(10000**(torch.arange(0,d,2).float()/d)))
    def forward(self,x):
        ang=torch.arange(x.size(1),device=x.device).float()[:,None]*self.inv_freq
        sin,cos=ang.sin(),ang.cos();x1,x2=x[...,::2],x[...,1::2]
        return torch.stack([x1*cos-x2*sin,x1*sin+x2*cos],-1).flatten(-2)

class Block(nn.Module):
    def __init__(self,d,nheads,dp,ff_mult=4):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,nheads,dropout=dp,batch_first=True)
        self.ln1=nn.LayerNorm(d);self.ln2=nn.LayerNorm(d);self.dp=nn.Dropout(dp)
        self.ffn=nn.Sequential(nn.Linear(d,d*ff_mult),nn.GELU(),nn.Dropout(dp),nn.Linear(d*ff_mult,d))
    def forward(self,x,causal=True,kpm=None):
        T=x.size(1)
        mask=torch.triu(torch.ones(T,T,dtype=torch.bool,device=x.device),1) if causal else None
        x=self.ln1(x+self.dp(self.attn(x,x,x,attn_mask=mask,key_padding_mask=kpm,need_weights=False)[0]))
        return self.ln2(x+self.dp(self.ffn(x)))

class CrossBlock(nn.Module):
    def __init__(self,d,nheads,dp):
        super().__init__()
        self.attn=nn.MultiheadAttention(d,nheads,dropout=dp,batch_first=True)
        self.ln=nn.LayerNorm(d);self.dp=nn.Dropout(dp)
    def forward(self,q,kv,kpm):
        return self.ln(q+self.dp(self.attn(q,kv,kv,key_padding_mask=kpm,need_weights=False)[0]))

class PhaseMoE(nn.Module):
    def __init__(self,d,n_experts=4):
        super().__init__()
        self.gate=nn.Sequential(nn.Linear(BS_DIM,32),nn.GELU(),nn.Linear(32,n_experts))
        self.experts=nn.ModuleList([nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Linear(d*2,d)) for _ in range(n_experts)])
        self.ln=nn.LayerNorm(d)
    def forward(self,x,bs):
        w=F.softmax(self.gate(bs),-1)
        return self.ln(x+sum(w[:,i,None,None]*e(x) for i,e in enumerate(self.experts)))

def pad_mask(T,lengths):return torch.arange(T,device=lengths.device)[None]>=lengths[:,None]

class Reacter(nn.Module):
    def __init__(self,nv,ed=64,hs=512,nl=8,nheads=8,dp=0.15):
        super().__init__()
        isz=N_CONT+N_CARD_FIELDS*ed
        self.ce=CardEmb(nv,ed);self.rope=RotaryPE(hs);self.ln_in=nn.LayerNorm(hs)
        self.proj=nn.Linear(isz,hs);self.opp_proj=nn.Linear(isz,hs)
        self.blocks=nn.ModuleList([Block(hs,nheads,dp) for _ in range(nl)])
        self.opp_blocks=nn.ModuleList([Block(hs,nheads,dp,ff_mult=2) for _ in range(4)])
        self.cross=nn.ModuleList([CrossBlock(hs,nheads,dp) for _ in range(3)])
        self.opp_card_emb=nn.Embedding(nv,hs)
        self.state_proj=nn.Sequential(nn.Linear(BS_DIM,hs//2),nn.GELU(),nn.Linear(hs//2,hs))
        self.moe=PhaseMoE(hs)
        self.card_head=nn.Sequential(nn.Linear(hs*3,hs),nn.GELU(),nn.LayerNorm(hs),nn.Dropout(dp),nn.Linear(hs,hs),nn.GELU(),nn.Linear(hs,nv))
        self.xy_head=nn.Sequential(nn.Linear(hs*3,hs//2),nn.GELU(),nn.Linear(hs//2,2))
        self.time_head=nn.Sequential(nn.Linear(hs*3,hs//4),nn.GELU(),nn.Linear(hs//4,1))
        self.opp_head=nn.Sequential(nn.Linear(hs*3,hs),nn.GELU(),nn.Linear(hs,nv))
    def _pool(self,h,m):return h.masked_fill(m[...,None],0).sum(1)/(~m).sum(1,keepdim=True).clamp(min=1)
    def forward(self,x,ox,ln,oln,olc,bs):
        m=pad_mask(x.size(1),ln);om=pad_mask(ox.size(1),oln)
        h=self.ln_in(self.rope(self.proj(self.ce(x))))
        for b in self.blocks:h=b(h)
        ho=self.ln_in(self.rope(self.opp_proj(self.ce(ox))))
        for b in self.opp_blocks:ho=b(ho,causal=False,kpm=om)
        for c in self.cross:h=c(h,ho,om)
        h=self.moe(h,bs)
        z=torch.cat([self._pool(h,m),self._pool(ho,om)+self.opp_card_emb(olc),self.state_proj(bs)],-1)
        return self.card_head(z),self.xy_head(z),self.time_head(z).squeeze(-1),self.opp_head(z)

def focal(logits,tgt,gamma):
    ce=F.cross_entropy(logits,tgt,reduction='none')
    return ((1-torch.exp(-ce))**gamma*ce).mean()

def card_loss(q,tgt,gamma,ls,alpha):
    l=(1-ls)*focal(q,tgt,gamma)-ls*F.log_softmax(q,-1).mean()
    return l+alpha*F.cross_entropy(q,tgt)  # CQL term logsumexp(q)-q[a] reduces to CE for one-hot data

def topk(q,tgt,ks=(1,3,5)):return [(q.topk(k,1)[1]==tgt[:,None]).any(1).float().mean().item() for k in ks]

def train(a):
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu');amp=dev.type=='cuda'
    ds=TrajDataset(a.csv,mode=a.mode,max_battles=a.max_battles)
    tds,vds=battle_split(ds,a.val_frac)
    tl=DataLoader(tds,a.bs,shuffle=True,collate_fn=collate,pin_memory=amp)
    vl=DataLoader(vds,a.bs,collate_fn=collate)
    _,vgrp=np.unique([ds.bids[i] for i in vds.indices],return_inverse=True)
    print(f'{dev} train {len(tds)} val {len(vds)} samples, {len(set(vgrp))} val battles',flush=True)
    model=Reacter(ds.num_cards,a.ed,a.hs,a.nl,dp=a.dp).to(dev)
    print(f'params {sum(p.numel() for p in model.parameters()):,}',flush=True)
    opt=torch.optim.AdamW(model.parameters(),lr=a.lr,weight_decay=0.05,betas=(0.9,0.98))
    total=len(tl)*a.epochs;warm=len(tl)*a.warmup
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda s:s/max(1,warm) if s<warm else max(0.1,0.5*(1+math.cos(math.pi*(s-warm)/max(1,total-warm)))))
    scaler=torch.amp.GradScaler(enabled=amp)
    ck=Path('checkpoints');ck.mkdir(exist_ok=True);best=-1;t0=time.time()
    for ep in range(a.epochs):
        model.train();tot=0;n=0
        for batch in tl:
            x,ox,ln,oln,txy,tc,olc,bs,onc=[b.to(dev,non_blocking=True) for b in batch]
            with torch.autocast(dev.type,enabled=amp):
                q,xy,t,op=model(x,ox,ln,oln,olc,bs)
                loss=card_loss(q,tc,a.focal_gamma,a.label_smooth,a.cql_alpha)+0.3*F.mse_loss(xy,txy[:,:2])+0.15*F.mse_loss(t,txy[:,2])+0.4*F.cross_entropy(op,onc)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward();scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(),1.0)
            scaler.step(opt);scaler.update();sched.step()
            tot+=loss.item()*x.size(0);n+=x.size(0)
        model.eval();hits=[];vl_loss=0;tk=np.zeros(3)
        with torch.no_grad():
            for batch in vl:
                x,ox,ln,oln,txy,tc,olc,bs,onc=[b.to(dev) for b in batch]
                q=model(x,ox,ln,oln,olc,bs)[0]
                vl_loss+=focal(q,tc,a.focal_gamma).item()*x.size(0);tk+=np.array(topk(q,tc))*x.size(0)
                hits.append((q.argmax(1)==tc).float().cpu().numpy())
        hits=np.concatenate(hits);nv=len(hits)
        top1=(np.bincount(vgrp,hits)/np.bincount(vgrp)).mean()  # battle-level: each battle weighs equally
        tk/=nv
        print(f'ep {ep+1} train {tot/max(n,1):.4f} val {vl_loss/nv:.4f} battle top1 {top1:.4f} top1/3/5 {tk[0]:.3f}/{tk[1]:.3f}/{tk[2]:.3f}',
              f'{time.time()-t0:.0f}s',flush=True)
        if top1>best:
            best=top1
            torch.save({'state_dict':model.state_dict(),'vocab':ds.vocab,'config':{'nv':ds.num_cards,'ed':a.ed,'hs':a.hs,'nl':a.nl,'dp':a.dp},'epoch':ep,'top1':top1},ck/f'{a.name}_best.pt')
    print(f'best battle top1 {best:.4f} -> checkpoints/{a.name}_best.pt',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=DEFAULT_TRAJ_PATH);ap.add_argument('--mode',default='reacter',choices=['planner','reacter','both'])
    ap.add_argument('--name',default=None);ap.add_argument('--epochs',type=int,default=30);ap.add_argument('--bs',type=int,default=48)
    ap.add_argument('--lr',type=float,default=1.2e-4);ap.add_argument('--warmup',type=int,default=3);ap.add_argument('--val-frac',type=float,default=0.15)
    ap.add_argument('--hs',type=int,default=512);ap.add_argument('--nl',type=int,default=8);ap.add_argument('--ed',type=int,default=64);ap.add_argument('--dp',type=float,default=0.15)
    ap.add_argument('--cql-alpha',type=float,default=0.2);ap.add_argument('--focal-gamma',type=float,default=2.0);ap.add_argument('--label-smooth',type=float,default=0.1)
    ap.add_argument('--max-battles',type=int,default=None)
    a=ap.parse_args();a.name=a.name or f'bc_{a.mode}'
    train(a)

if __name__=='__main__':main()
