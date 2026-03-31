import torch,math
from torch import nn
import torch.nn.functional as F
from .features import FEAT_DIM
import numpy as np

CARD_EMB_DIM=32
DELTA_DIM=FEAT_DIM
TOTAL_INPUT=FEAT_DIM+DELTA_DIM+CARD_EMB_DIM

class ResBlock(nn.Module):
    def __init__(self,d,dp=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(dp),nn.Linear(d*2,d))
        self.ln=nn.LayerNorm(d)
    def forward(self,x):return self.ln(x+self.net(x))

class BackboneV11(nn.Module):
    def __init__(self,hs=384,nl=6,dp=0.1,num_cards=200):
        super().__init__()
        self.card_emb=nn.Embedding(num_cards,CARD_EMB_DIM,padding_idx=0)
        self.proj=nn.Sequential(nn.Linear(TOTAL_INPUT,hs),nn.GELU(),nn.LayerNorm(hs))
        self.blocks=nn.ModuleList([ResBlock(hs,dp) for _ in range(nl)])
        self.hs=hs
    def forward(self,feat,delta,card_idx):
        ce=self.card_emb(card_idx)
        x=torch.cat([feat,delta,ce],dim=-1)
        h=self.proj(x)
        for b in self.blocks:h=b(h)
        return h

class DemonHead(nn.Module):
    def __init__(self,hs,out_dim=1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(hs,hs//2),nn.GELU(),nn.LayerNorm(hs//2),nn.Linear(hs//2,out_dim))
        self.out_dim=out_dim
    def forward(self,h):return self.net(h).squeeze(-1) if self.out_dim==1 else self.net(h)

class CardCondPlacementHead(nn.Module):
    def __init__(self,hs,num_cards=200):
        super().__init__()
        self.card_proj=nn.Embedding(num_cards,64,padding_idx=0)
        self.net=nn.Sequential(nn.Linear(hs+64,hs//2),nn.GELU(),nn.LayerNorm(hs//2),nn.Linear(hs//2,1))
    def forward(self,h,card_idx=None):
        if card_idx is not None:
            ce=self.card_proj(card_idx)
            return self.net(torch.cat([h,ce],dim=-1)).squeeze(-1)
        return self.net(torch.cat([h,torch.zeros(h.shape[0],64,device=h.device)],dim=-1)).squeeze(-1)

class TDDemon:
    def __init__(self,name,gamma=0.95,hs=384,lr=1e-3,use_mc=False,use_huber=False,
                 head_type='value',num_cards=200,bb_weight=1.0):
        self.name=name
        self.gamma=gamma
        self.use_mc=use_mc
        self.use_huber=use_huber
        self.is_cls=False
        self.bb_weight=bb_weight
        self.head_type=head_type
        if head_type=='placement':
            self.head=CardCondPlacementHead(hs,num_cards)
        else:
            self.head=DemonHead(hs,1)
        self.opt=torch.optim.AdamW(self.head.parameters(),lr=lr,weight_decay=1e-5)
    def reset(self):
        for p in self.head.parameters():
            if p.grad is not None:p.grad.zero_()

class BinaryDemon:
    def __init__(self,name,hs=384,lr=1e-3,bb_weight=1.0):
        self.name=name
        self.is_cls=False
        self.bb_weight=bb_weight
        self.head=DemonHead(hs,1)
        self.opt=torch.optim.AdamW(self.head.parameters(),lr=lr,weight_decay=1e-5)
    def reset(self):
        for p in self.head.parameters():
            if p.grad is not None:p.grad.zero_()

def build_v11_horde(num_cards=200,hs=384,device=None):
    dev=device or torch.device('cpu')
    backbone=BackboneV11(hs,nl=6,dp=0.1,num_cards=num_cards).to(dev)
    td=[]
    td.append(TDDemon('tower_dmg_imm',gamma=0.0,hs=hs,lr=3e-3,use_mc=True,use_huber=True,bb_weight=0.5))
    td.append(TDDemon('tower_dmg_short',gamma=0.8,hs=hs,lr=2e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('tower_dmg_mid',gamma=0.95,hs=hs,lr=1e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('tower_dmg_long',gamma=0.97,hs=hs,lr=5e-4,use_mc=True,use_huber=True))
    td.append(TDDemon('tower_taken_imm',gamma=0.0,hs=hs,lr=3e-3,use_mc=True,use_huber=True,bb_weight=0.5))
    td.append(TDDemon('tower_taken_mid',gamma=0.95,hs=hs,lr=1e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('crown_imm',gamma=0.0,hs=hs,lr=2e-3,bb_weight=0.5))
    td.append(TDDemon('crown_short',gamma=0.9,hs=hs,lr=5e-4))
    td.append(TDDemon('crown_long',gamma=0.97,hs=hs,lr=5e-4))
    td.append(TDDemon('elixir_adv',gamma=0.5,hs=hs,lr=2e-3,use_huber=True))
    td.append(TDDemon('lane_left',gamma=0.85,hs=hs,lr=5e-4))
    td.append(TDDemon('lane_right',gamma=0.85,hs=hs,lr=5e-4))
    td.append(TDDemon('troop_adv',gamma=0.8,hs=hs,lr=5e-4))
    td.append(TDDemon('dmg_efficiency',gamma=0.9,hs=hs,lr=1e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('placement_x',gamma=0.0,hs=hs,lr=3e-3,head_type='placement',num_cards=num_cards,bb_weight=2.0))
    td.append(TDDemon('placement_y',gamma=0.0,hs=hs,lr=3e-3,head_type='placement',num_cards=num_cards,bb_weight=2.0))
    bn=[]
    bn.append(BinaryDemon('will_damage',hs=hs,lr=2e-3))
    bn.append(BinaryDemon('will_lose_tower',hs=hs,lr=2e-3))
    return HordeV11(backbone,td,bn,dev)

class HordeV11:
    def __init__(self,backbone,td_demons,binary_demons,device):
        self.backbone=backbone
        self.td_demons=td_demons
        self.binary_demons=binary_demons
        self.cql_demons=[]
        self.awr_demons=[]
        self.dev=device
        self.bb_opt=torch.optim.AdamW(backbone.parameters(),lr=3e-4,weight_decay=1e-5)
    def to(self,dev):
        self.backbone=self.backbone.to(dev)
        for d in self.td_demons:d.head=d.head.to(dev)
        for d in self.binary_demons:d.head=d.head.to(dev)
        self.dev=dev;return self
    def train_episode(self,feats,deltas,card_idxs,cumulants,binary_targets):
        T=len(feats)
        if T<2:return {}
        feat_t=torch.tensor(np.stack(feats),dtype=torch.float32,device=self.dev)
        delta_t=torch.tensor(np.stack(deltas),dtype=torch.float32,device=self.dev)
        card_t=torch.tensor(card_idxs,dtype=torch.long,device=self.dev)
        H=self.backbone(feat_t,delta_t,card_t)
        total_loss=torch.tensor(0.0,device=self.dev)
        losses={}
        for d in self.td_demons:
            cums=cumulants[d.name]
            if d.use_mc:
                mc=self._mc_returns(cums,d.gamma)
                targets=torch.tensor(mc,dtype=torch.float32,device=self.dev)
            else:
                targets=torch.zeros(T,device=self.dev)
                for t in range(T):
                    c=cums[t]
                    v_next=d.head(H[t+1]).detach().item() if t+1<T and d.head_type=='value' else 0.0
                    targets[t]=c+d.gamma*v_next
            if d.head_type=='placement':
                preds=d.head(H,card_t)
            else:
                preds=d.head(H)
            if d.use_huber:
                dl=F.smooth_l1_loss(preds,targets)*d.bb_weight
            else:
                dl=F.mse_loss(preds,targets)*d.bb_weight
            if torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
        for d in self.binary_demons:
            tgts=torch.tensor(binary_targets[d.name],dtype=torch.float32,device=self.dev)
            logits=d.head(H)
            dl=F.binary_cross_entropy_with_logits(logits,tgts)*d.bb_weight
            if torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
        if total_loss>0 and total_loss.requires_grad:
            self.bb_opt.zero_grad()
            for d in self.td_demons:d.opt.zero_grad()
            for d in self.binary_demons:d.opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.backbone.parameters(),1.0)
            for d in self.td_demons:nn.utils.clip_grad_norm_(d.head.parameters(),1.0)
            for d in self.binary_demons:nn.utils.clip_grad_norm_(d.head.parameters(),1.0)
            self.bb_opt.step()
            for d in self.td_demons:d.opt.step()
            for d in self.binary_demons:d.opt.step()
        return losses
    def _mc_returns(self,rewards,gamma):
        T=len(rewards)
        G=[0.0]*T
        G[-1]=rewards[-1]
        for t in range(T-2,-1,-1):
            G[t]=rewards[t]+gamma*G[t+1]
        return G
    def save(self,path):
        data={'backbone':self.backbone.state_dict()}
        for d in self.td_demons:data[f'td_{d.name}']=d.head.state_dict()
        for d in self.binary_demons:data[f'bin_{d.name}']=d.head.state_dict()
        torch.save(data,path)
    def load(self,path):
        data=torch.load(path,map_location='cpu',weights_only=False)
        self.backbone.load_state_dict(data['backbone'])
        for d in self.td_demons:
            k=f'td_{d.name}'
            if k in data:d.head.load_state_dict(data[k])
        for d in self.binary_demons:
            k=f'bin_{d.name}'
            if k in data:d.head.load_state_dict(data[k])
    @property
    def all_demons(self):return self.td_demons+self.binary_demons
