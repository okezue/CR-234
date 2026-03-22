import torch,math
from torch import nn
import torch.nn.functional as F
from .features import FEAT_DIM

CARD_EMB_DIM=32
DELTA_DIM=FEAT_DIM
TOTAL_INPUT=FEAT_DIM+DELTA_DIM+CARD_EMB_DIM

class ResBlock(nn.Module):
    def __init__(self,d,dp=0.1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d,d*2),nn.GELU(),nn.Dropout(dp),nn.Linear(d*2,d))
        self.ln=nn.LayerNorm(d)
    def forward(self,x):return self.ln(x+self.net(x))

class BackboneV8(nn.Module):
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

class TDDemon:
    def __init__(self,name,gamma=0.95,hs=384,lr=1e-3,use_mc=False,use_huber=False):
        self.name=name
        self.gamma=gamma
        self.use_mc=use_mc
        self.use_huber=use_huber
        self.is_cls=False
        self.head=DemonHead(hs,1)
        self.opt=torch.optim.AdamW(self.head.parameters(),lr=lr,weight_decay=1e-5)
    def reset(self):
        for p in self.head.parameters():
            if p.grad is not None:p.grad.zero_()

class BinaryDemon:
    def __init__(self,name,hs=384,lr=1e-3):
        self.name=name
        self.is_cls=False
        self.head=DemonHead(hs,1)
        self.opt=torch.optim.AdamW(self.head.parameters(),lr=lr,weight_decay=1e-5)
    def reset(self):
        for p in self.head.parameters():
            if p.grad is not None:p.grad.zero_()

class AWRDemon:
    def __init__(self,name,gamma=0.99,num_actions=200,beta=0.05,hs=384,lr=3e-4):
        self.name=name
        self.gamma=gamma
        self.num_actions=num_actions
        self.beta=beta
        self.is_cls=False
        self.v_head=nn.Sequential(nn.Linear(hs,hs//2),nn.GELU(),nn.Linear(hs//2,1))
        self.pi_head=nn.Sequential(nn.Linear(hs,hs),nn.GELU(),nn.LayerNorm(hs),nn.Linear(hs,num_actions))
        self.v_opt=torch.optim.AdamW(self.v_head.parameters(),lr=lr)
        self.pi_opt=torch.optim.AdamW(self.pi_head.parameters(),lr=lr)
    def reset(self):
        for p in list(self.v_head.parameters())+list(self.pi_head.parameters()):
            if p.grad is not None:p.grad.zero_()

def build_v8_horde(num_cards=200,hs=384,device=None):
    dev=device or torch.device('cpu')
    backbone=BackboneV8(hs,nl=6,dp=0.1,num_cards=num_cards).to(dev)
    td=[]
    td.append(TDDemon('tower_dmg_imm',gamma=0.0,hs=hs,lr=3e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('tower_dmg_short',gamma=0.8,hs=hs,lr=2e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('tower_dmg_mid',gamma=0.95,hs=hs,lr=1e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('tower_dmg_long',gamma=0.99,hs=hs,lr=5e-4,use_mc=True,use_huber=True))
    td.append(TDDemon('tower_taken_imm',gamma=0.0,hs=hs,lr=3e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('tower_taken_mid',gamma=0.95,hs=hs,lr=1e-3,use_mc=True,use_huber=True))
    td.append(TDDemon('crown_imm',gamma=0.0,hs=hs,lr=2e-3))
    td.append(TDDemon('crown_short',gamma=0.9,hs=hs,lr=5e-4))
    td.append(TDDemon('crown_long',gamma=0.99,hs=hs,lr=5e-4))
    td.append(TDDemon('elixir_adv',gamma=0.5,hs=hs,lr=2e-3,use_huber=True))
    td.append(TDDemon('lane_left',gamma=0.85,hs=hs,lr=5e-4))
    td.append(TDDemon('lane_right',gamma=0.85,hs=hs,lr=5e-4))
    td.append(TDDemon('troop_adv',gamma=0.8,hs=hs,lr=5e-4))
    td.append(TDDemon('dmg_efficiency',gamma=0.95,hs=hs,lr=5e-4,use_huber=True))
    td.append(TDDemon('placement_x',gamma=0.0,hs=hs,lr=2e-3))
    td.append(TDDemon('placement_y',gamma=0.0,hs=hs,lr=2e-3))
    td.append(TDDemon('win_pred',gamma=0.99,hs=hs,lr=3e-4))
    bn=[]
    bn.append(BinaryDemon('will_damage',hs=hs,lr=2e-3))
    bn.append(BinaryDemon('will_lose_tower',hs=hs,lr=2e-3))
    bn.append(BinaryDemon('winning',hs=hs,lr=1e-3))
    awr_off=AWRDemon('awr_offense',gamma=0.99,num_actions=num_cards,beta=0.05,hs=hs,lr=3e-4)
    awr_def=AWRDemon('awr_defense',gamma=0.95,num_actions=num_cards,beta=0.1,hs=hs,lr=3e-4)
    return HordeV8(backbone,td,bn,[awr_off,awr_def],dev)

class HordeV8:
    def __init__(self,backbone,td_demons,binary_demons,awr_demons,device):
        self.backbone=backbone
        self.td_demons=td_demons
        self.binary_demons=binary_demons
        self.cql_demons=[]
        self.awr_demons=awr_demons
        self.dev=device
        self.bb_opt=torch.optim.AdamW(backbone.parameters(),lr=3e-4,weight_decay=1e-5)
    def to(self,dev):
        self.backbone=self.backbone.to(dev)
        for d in self.td_demons:d.head=d.head.to(dev)
        for d in self.binary_demons:d.head=d.head.to(dev)
        for d in self.awr_demons:
            d.v_head=d.v_head.to(dev);d.pi_head=d.pi_head.to(dev)
        self.dev=dev;return self
    def train_episode(self,feats,deltas,card_idxs,cumulants,binary_targets,awr_rewards,actions):
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
                    v_next=d.head(H[t+1]).detach().item() if t+1<T else 0.0
                    targets[t]=c+d.gamma*v_next
            preds=d.head(H)
            if d.use_huber:
                dl=F.smooth_l1_loss(preds,targets)
            else:
                dl=F.mse_loss(preds,targets)
            if torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
        for d in self.binary_demons:
            tgts=torch.tensor(binary_targets[d.name],dtype=torch.float32,device=self.dev)
            logits=d.head(H)
            dl=F.binary_cross_entropy_with_logits(logits,tgts)
            if torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
        for d in self.awr_demons:
            rewards=awr_rewards[d.name]
            mc=self._mc_returns(rewards,d.gamma)
            mc_t=torch.tensor(mc,dtype=torch.float32,device=self.dev)
            v=d.v_head(H).squeeze(-1)
            v_loss=F.smooth_l1_loss(v,mc_t)
            with torch.no_grad():
                advs=mc_t-v
                weights=torch.clamp(torch.exp(advs/d.beta),0.01,20.0)
            logits=d.pi_head(H)
            log_probs=F.log_softmax(logits,dim=-1)
            act_t=torch.tensor(actions,dtype=torch.long,device=self.dev).clamp(0,logits.size(-1)-1)
            pi_loss=-(weights*log_probs.gather(1,act_t.unsqueeze(1)).squeeze(1)).mean()
            dl=v_loss+pi_loss
            if torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
        if total_loss>0 and total_loss.requires_grad:
            self.bb_opt.zero_grad()
            for d in self.td_demons:d.opt.zero_grad()
            for d in self.binary_demons:d.opt.zero_grad()
            for d in self.awr_demons:d.v_opt.zero_grad();d.pi_opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.backbone.parameters(),1.0)
            for d in self.td_demons:nn.utils.clip_grad_norm_(d.head.parameters(),1.0)
            for d in self.binary_demons:nn.utils.clip_grad_norm_(d.head.parameters(),1.0)
            for d in self.awr_demons:
                nn.utils.clip_grad_norm_(d.v_head.parameters(),1.0)
                nn.utils.clip_grad_norm_(d.pi_head.parameters(),1.0)
            self.bb_opt.step()
            for d in self.td_demons:d.opt.step()
            for d in self.binary_demons:d.opt.step()
            for d in self.awr_demons:d.v_opt.step();d.pi_opt.step()
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
        for d in self.awr_demons:
            data[f'awr_v_{d.name}']=d.v_head.state_dict()
            data[f'awr_pi_{d.name}']=d.pi_head.state_dict()
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
        for d in self.awr_demons:
            kv=f'awr_v_{d.name}';kp=f'awr_pi_{d.name}'
            if kv in data:d.v_head.load_state_dict(data[kv])
            if kp in data:d.pi_head.load_state_dict(data[kp])
    @property
    def all_demons(self):return self.td_demons+self.binary_demons+self.awr_demons

import numpy as np
