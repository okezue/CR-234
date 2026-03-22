import torch,math
from torch import nn
import torch.nn.functional as F
from .features import FEAT_DIM

class SharedBackboneV5(nn.Module):
    def __init__(self,in_dim,hs=192,nl=4,dp=0.1):
        super().__init__()
        layers=[]
        for i in range(nl):
            d_in=in_dim if i==0 else hs
            layers+=[nn.Linear(d_in,hs),nn.LayerNorm(hs),nn.GELU(),nn.Dropout(dp)]
        self.net=nn.Sequential(*layers)
        self.hs=hs
    def forward(self,x):return self.net(x)
    def reset(self):
        for m in self.modules():
            if hasattr(m,'reset_parameters'):m.reset_parameters()

class DemonHead(nn.Module):
    def __init__(self,hs,out_dim=1):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(hs,hs),nn.GELU(),nn.LayerNorm(hs),nn.Linear(hs,out_dim))
    def forward(self,h):return self.net(h).squeeze(-1) if self.net[-1].out_features==1 else self.net(h)

class TDDemonV5:
    def __init__(self,name,cumulant_fn,gamma=0.95,lam=0.8,hs=192,lr=1e-3,
                 interest_fn=None,out_dim=1,is_cls=False):
        self.name=name
        self.cumulant_fn=cumulant_fn
        self.gamma=gamma
        self.lam=lam
        self.interest_fn=interest_fn or (lambda i:1.0)
        self.is_cls=is_cls
        self.head=DemonHead(hs,out_dim)
        self.opt=torch.optim.AdamW(self.head.parameters(),lr=lr,weight_decay=1e-5)
    def reset(self):
        for p in self.head.parameters():
            if p.grad is not None:p.grad.zero_()
    def compute_loss(self,h,h_next,action,info):
        i=self.interest_fn(info)
        if i<0.01:return None
        c=self.cumulant_fn(info)
        with torch.no_grad():
            if self.is_cls:v_next=0.0
            else:v_next=self.head(h_next).item()
        if self.is_cls:
            logits=self.head(h)
            target=int(c)
            return F.cross_entropy(logits.unsqueeze(0),torch.tensor([target],device=h.device))*i
        v=self.head(h)
        target=c+self.gamma*v_next
        return F.mse_loss(v,torch.tensor([target],device=h.device))*i

class CQLDemonV5:
    def __init__(self,name,reward_fn,gamma=0.99,num_actions=180,cql_alpha=0.3,hs=192,lr=3e-4):
        self.name=name
        self.reward_fn=reward_fn
        self.gamma=gamma
        self.num_actions=num_actions
        self.cql_alpha=cql_alpha
        self.is_cls=False
        self.q_head=nn.Sequential(nn.Linear(hs,hs),nn.GELU(),nn.LayerNorm(hs),nn.Linear(hs,num_actions))
        self.opt=torch.optim.AdamW(self.q_head.parameters(),lr=lr,weight_decay=1e-5)
    def reset(self):
        for p in self.q_head.parameters():
            if p.grad is not None:p.grad.zero_()
    def compute_loss(self,h,h_next,action,info):
        r=self.reward_fn(info)
        with torch.no_grad():q_next=self.q_head(h_next).max().item()
        q=self.q_head(h)
        target=r+self.gamma*q_next*(1-float(info.get('ended',False)))
        qa=q[action]
        td_loss=F.mse_loss(qa,torch.tensor(target,device=h.device))
        logsumexp=torch.logsumexp(q,dim=-1)
        cql_loss=self.cql_alpha*(logsumexp-qa)
        return td_loss+cql_loss

class AWRDemonV5:
    def __init__(self,name,reward_fn,gamma=0.99,num_actions=180,beta=0.05,hs=192,lr=3e-4):
        self.name=name
        self.reward_fn=reward_fn
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
    def compute_loss(self,h,h_next,action,info):
        r=self.reward_fn(info)
        with torch.no_grad():v_next=self.v_head(h_next).item()
        v=self.v_head(h).squeeze(-1)
        target=r+self.gamma*v_next*(1-float(info.get('ended',False)))
        v_loss=F.mse_loss(v,torch.tensor([target],device=h.device))
        with torch.no_grad():
            adv=target-v.item()
            weight=min(max(math.exp(min(adv/self.beta,5.0)),0.01),20.0)
        logits=self.pi_head(h)
        log_prob=F.log_softmax(logits,dim=-1)
        pi_loss=-weight*log_prob[action]
        return v_loss+pi_loss

def _i_tower(info):
    d=abs(info.get('opp_tower_hp_delta',0))+abs(info.get('team_tower_hp_delta',0))
    return min(d/300.0+0.3,5.0)
def _i_crown(info):return 5.0 if info.get('crown_scored',0)>0 else 1.0
def _i_elixir(info):return 1.0+abs(info.get('elixir_advantage',0))/3.0
def _i_overtime(info):return 3.0 if info.get('time',0)>180 else 0.3
def _i_double(info):return 3.0 if info.get('time',0)>120 else 0.3

def build_v5_horde(num_cards=180,hs=192,device=None):
    dev=device or torch.device('cpu')
    backbone=SharedBackboneV5(FEAT_DIM,hs,nl=4,dp=0.1).to(dev)
    demons=[]
    demons.append(TDDemonV5('tower_dmg_imm',lambda i:i.get('opp_tower_hp_delta',0)/1000.0,
        gamma=0.0,lam=0.0,hs=hs,interest_fn=_i_tower,lr=2e-3))
    demons.append(TDDemonV5('tower_dmg_short',lambda i:i.get('opp_tower_hp_delta',0)/1000.0,
        gamma=0.8,lam=0.6,hs=hs,interest_fn=_i_tower,lr=1e-3))
    demons.append(TDDemonV5('tower_dmg_mid',lambda i:i.get('opp_tower_hp_delta',0)/1500.0,
        gamma=0.95,lam=0.8,hs=hs,interest_fn=_i_tower,lr=1e-3))
    demons.append(TDDemonV5('tower_dmg_long',lambda i:i.get('opp_tower_hp_delta',0)/2000.0,
        gamma=0.99,lam=0.9,hs=hs,interest_fn=_i_tower,lr=5e-4))
    demons.append(TDDemonV5('tower_taken_imm',lambda i:-i.get('team_tower_hp_delta',0)/1000.0,
        gamma=0.0,lam=0.0,hs=hs,interest_fn=_i_tower,lr=2e-3))
    demons.append(TDDemonV5('tower_taken_mid',lambda i:-i.get('team_tower_hp_delta',0)/1500.0,
        gamma=0.95,lam=0.8,hs=hs,interest_fn=_i_tower,lr=1e-3))
    demons.append(TDDemonV5('crown_imm',lambda i:float(i.get('crown_scored',0)),
        gamma=0.0,lam=0.0,hs=hs,interest_fn=_i_crown,lr=2e-3))
    demons.append(TDDemonV5('crown_short',lambda i:float(i.get('crown_scored',0)),
        gamma=0.9,lam=0.7,hs=hs,interest_fn=_i_crown,lr=5e-4))
    demons.append(TDDemonV5('crown_long',lambda i:float(i.get('crown_scored',0)),
        gamma=0.99,lam=0.9,hs=hs,interest_fn=_i_crown,lr=5e-4))
    demons.append(TDDemonV5('elixir_adv',lambda i:max(-1.0,min(1.0,i.get('elixir_advantage',0)/5.0)),
        gamma=0.9,lam=0.7,hs=hs,interest_fn=_i_elixir,lr=5e-4))
    demons.append(TDDemonV5('lane_left',lambda i:max(-1.0,min(1.0,i.get('lane_pressure_left',0)/3.0)),
        gamma=0.85,lam=0.7,hs=hs,lr=5e-4))
    demons.append(TDDemonV5('lane_right',lambda i:max(-1.0,min(1.0,i.get('lane_pressure_right',0)/3.0)),
        gamma=0.85,lam=0.7,hs=hs,lr=5e-4))
    demons.append(TDDemonV5('troop_adv',lambda i:max(-1.0,min(1.0,i.get('troop_count_diff',0)/5.0)),
        gamma=0.8,lam=0.6,hs=hs,lr=5e-4))
    demons.append(TDDemonV5('dmg_efficiency',lambda i:min(i.get('dmg_per_elixir',0)/100.0,1.0),
        gamma=0.95,lam=0.8,hs=hs,lr=5e-4))
    demons.append(TDDemonV5('overtime_pressure',
        lambda i:float(i.get('crown_diff',0))/3.0 if i.get('time',0)>180 else 0.0,
        gamma=0.95,lam=0.8,hs=hs,interest_fn=_i_overtime,lr=5e-4))
    demons.append(TDDemonV5('placement_x',lambda i:i.get('placement_x',0.5),
        gamma=0.0,lam=0.0,hs=hs,lr=1e-3))
    demons.append(TDDemonV5('placement_y',lambda i:i.get('placement_y',0.5),
        gamma=0.0,lam=0.0,hs=hs,lr=1e-3))
    cql_win=CQLDemonV5('cql_win',lambda i:i.get('win_label',0.5),
        gamma=0.99,num_actions=num_cards,cql_alpha=0.3,hs=hs,lr=3e-4)
    cql_crown=CQLDemonV5('cql_crown',lambda i:float(i.get('crown_diff',0))/3.0,
        gamma=0.95,num_actions=num_cards,cql_alpha=0.5,hs=hs,lr=3e-4)
    awr_offense=AWRDemonV5('awr_offense',
        lambda i:i.get('opp_tower_hp_delta',0)/1500.0+float(i.get('crown_scored',0))*0.5,
        gamma=0.99,num_actions=num_cards,beta=0.05,hs=hs,lr=3e-4)
    awr_defense=AWRDemonV5('awr_defense',lambda i:-i.get('team_tower_hp_delta',0)/1500.0,
        gamma=0.95,num_actions=num_cards,beta=0.1,hs=hs,lr=3e-4)
    return HordeV5(backbone,demons,[cql_win,cql_crown],[awr_offense,awr_defense],dev)

class HordeV5:
    def __init__(self,backbone,td_demons,cql_demons,awr_demons,device):
        self.backbone=backbone
        self.td_demons=td_demons
        self.cql_demons=cql_demons
        self.awr_demons=awr_demons
        self.dev=device
        self.bb_opt=torch.optim.AdamW(backbone.parameters(),lr=3e-4,weight_decay=1e-5)
    def reset_all(self):
        for d in self.td_demons:d.reset()
        for d in self.cql_demons:d.reset()
        for d in self.awr_demons:d.reset()
    def to(self,dev):
        self.backbone=self.backbone.to(dev)
        for d in self.td_demons:d.head=d.head.to(dev)
        for d in self.cql_demons:d.q_head=d.q_head.to(dev)
        for d in self.awr_demons:
            d.v_head=d.v_head.to(dev);d.pi_head=d.pi_head.to(dev)
        self.dev=dev
        return self
    def observe_and_learn(self,state,action,next_state,info):
        st=torch.tensor(state,dtype=torch.float32,device=self.dev)
        nst=torch.tensor(next_state,dtype=torch.float32,device=self.dev)
        h=self.backbone(st)
        with torch.no_grad():
            h_next=self.backbone(nst)
        total_loss=torch.tensor(0.0,device=self.dev,requires_grad=False)
        losses={}
        for d in self.td_demons:
            dl=d.compute_loss(h,h_next,action,info)
            if dl is not None and torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
            else:losses[d.name]=0.0
        for d in self.cql_demons:
            dl=d.compute_loss(h,h_next,action,info)
            if dl is not None and torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
            else:losses[d.name]=0.0
        for d in self.awr_demons:
            dl=d.compute_loss(h,h_next,action,info)
            if dl is not None and torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
            else:losses[d.name]=0.0
        if total_loss>0 and total_loss.requires_grad:
            self.bb_opt.zero_grad()
            for d in self.td_demons:d.opt.zero_grad()
            for d in self.cql_demons:d.opt.zero_grad()
            for d in self.awr_demons:d.v_opt.zero_grad();d.pi_opt.zero_grad()
            total_loss.backward()
            nn.utils.clip_grad_norm_(self.backbone.parameters(),1.0)
            for d in self.td_demons:nn.utils.clip_grad_norm_(d.head.parameters(),1.0)
            for d in self.cql_demons:nn.utils.clip_grad_norm_(d.q_head.parameters(),1.0)
            for d in self.awr_demons:
                nn.utils.clip_grad_norm_(d.v_head.parameters(),1.0)
                nn.utils.clip_grad_norm_(d.pi_head.parameters(),1.0)
            self.bb_opt.step()
            for d in self.td_demons:d.opt.step()
            for d in self.cql_demons:d.opt.step()
            for d in self.awr_demons:d.v_opt.step();d.pi_opt.step()
        return losses
    def save(self,path):
        data={'backbone':self.backbone.state_dict(),'bb_opt':self.bb_opt.state_dict()}
        for d in self.td_demons:
            data[f'td_{d.name}']=d.head.state_dict()
            data[f'tdopt_{d.name}']=d.opt.state_dict()
        for d in self.cql_demons:
            data[f'cql_{d.name}']=d.q_head.state_dict()
            data[f'cqlopt_{d.name}']=d.opt.state_dict()
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
        for d in self.cql_demons:
            k=f'cql_{d.name}'
            if k in data:d.q_head.load_state_dict(data[k])
        for d in self.awr_demons:
            kv=f'awr_v_{d.name}';kp=f'awr_pi_{d.name}'
            if kv in data:d.v_head.load_state_dict(data[kv])
            if kp in data:d.pi_head.load_state_dict(data[kp])
    @property
    def all_demons(self):return self.td_demons+self.cql_demons+self.awr_demons
