import torch
from torch import nn
import torch.nn.functional as F
import math
from .features_v4 import FLAT_DIM,UNIT_FEAT,MAX_UNITS,GRID_H,GRID_W,GRID_CHANNELS

class UnitAttention(nn.Module):
    def __init__(self,d_unit,d_model,nheads=4):
        super().__init__()
        self.proj_in=nn.Linear(d_unit,d_model)
        self.attn=nn.MultiheadAttention(d_model,nheads,batch_first=True,dropout=0.1)
        self.ln=nn.LayerNorm(d_model)
        self.ffn=nn.Sequential(nn.Linear(d_model,d_model*2),nn.GELU(),nn.Dropout(0.1),nn.Linear(d_model*2,d_model))
        self.ln2=nn.LayerNorm(d_model)
    def forward(self,units,mask=None):
        x=self.proj_in(units)
        if mask is not None and mask.all():
            return x
        attn_out,_=self.attn(x,x,x,key_padding_mask=mask)
        x=self.ln(x+attn_out)
        x=self.ln2(x+self.ffn(x))
        return x

class SpatialEncoder(nn.Module):
    def __init__(self,in_ch,out_dim):
        super().__init__()
        self.conv=nn.Sequential(
            nn.Conv2d(in_ch,32,3,padding=1),nn.GELU(),
            nn.Conv2d(32,64,3,padding=1),nn.GELU(),
            nn.AdaptiveAvgPool2d((2,2)),
            nn.Flatten(),nn.Linear(64*4,out_dim),nn.GELU())
    def forward(self,grid):
        return self.conv(grid)

class SharedBackboneV4(nn.Module):
    def __init__(self,d_model=256,nheads=4,nl=2,dp=0.1):
        super().__init__()
        self.flat_proj=nn.Sequential(nn.Linear(FLAT_DIM,d_model),nn.GELU(),nn.Dropout(dp))
        self.unit_attn=nn.ModuleList([UnitAttention(UNIT_FEAT if i==0 else d_model,d_model,nheads) for i in range(nl)])
        self.spatial=SpatialEncoder(GRID_CHANNELS,d_model//2)
        self.combine=nn.Sequential(nn.Linear(d_model*2+d_model//2,d_model),nn.GELU(),nn.LayerNorm(d_model))
        self.d_model=d_model
    def forward(self,flat,units,grid):
        flat_h=self.flat_proj(flat)
        unit_mask=(units.abs().sum(-1)==0)
        u=units
        for layer in self.unit_attn:
            u=layer(u,unit_mask)
        unit_h=u.mean(dim=1)
        spatial_h=self.spatial(grid.permute(0,3,1,2))
        h=self.combine(torch.cat([flat_h,unit_h,spatial_h],dim=-1))
        return h,u
    def reset(self):
        for m in self.modules():
            if hasattr(m,'reset_parameters'):m.reset_parameters()

class ValueHead(nn.Module):
    def __init__(self,d_in,d_hidden=128):
        super().__init__()
        self.net=nn.Sequential(nn.Linear(d_in,d_hidden),nn.GELU(),nn.LayerNorm(d_hidden),
                               nn.Linear(d_hidden,d_hidden),nn.GELU(),nn.Linear(d_hidden,1))
    def forward(self,h):return self.net(h).squeeze(-1)

class PlacementHead(nn.Module):
    def __init__(self,d_in,grid_h=8,grid_w=5):
        super().__init__()
        self.grid_h,self.grid_w=grid_h,grid_w
        self.net=nn.Sequential(nn.Linear(d_in,256),nn.GELU(),nn.LayerNorm(256),
                               nn.Linear(256,256),nn.GELU(),nn.Linear(256,grid_h*grid_w))
    def forward(self,h):
        logits=self.net(h).view(-1,self.grid_h,self.grid_w)
        return logits
    def predict_xy(self,h):
        logits=self.forward(h)
        probs=F.softmax(logits.view(-1,self.grid_h*self.grid_w),dim=-1)
        idx=probs.argmax(dim=-1)
        y=idx//self.grid_w
        x=idx%self.grid_w
        return x.float()/(self.grid_w-1),y.float()/(self.grid_h-1)

class UnitAwareHead(nn.Module):
    def __init__(self,d_model,d_unit,nheads=4):
        super().__init__()
        self.query=nn.Linear(d_model,d_model)
        self.attn=nn.MultiheadAttention(d_model,nheads,batch_first=True)
        self.proj=nn.Linear(d_unit,d_model)
        self.out=nn.Sequential(nn.Linear(d_model*2,128),nn.GELU(),nn.Linear(128,1))
    def forward(self,h,units,unit_mask=None):
        q=self.query(h).unsqueeze(1)
        k=v=self.proj(units)
        ctx,_=self.attn(q,k,v,key_padding_mask=unit_mask)
        ctx=ctx.squeeze(1)
        return self.out(torch.cat([h,ctx],dim=-1)).squeeze(-1)

class TDDemonV4:
    def __init__(self,name,cumulant_fn,gamma=0.95,lam=0.8,lr=1e-3,head_type='value',
                 d_model=256,interest_fn=None,use_units=False):
        self.name=name
        self.cumulant_fn=cumulant_fn
        self.gamma=gamma
        self.lam=lam
        self.interest_fn=interest_fn or (lambda i:1.0)
        self.use_units=use_units
        self.is_cls=False
        if head_type=='value':
            self.head=ValueHead(d_model)
        elif head_type=='placement':
            self.head=PlacementHead(d_model)
        elif head_type=='unit_aware':
            self.head=UnitAwareHead(d_model,UNIT_FEAT)
        self.head_type=head_type
        self.opt=torch.optim.AdamW(self.head.parameters(),lr=lr,weight_decay=1e-5)
        self.trace=None
    def reset(self):
        self.trace=None
        for p in self.head.parameters():
            if p.grad is not None:p.grad.zero_()
    def observe_and_learn(self,h,h_next,units,units_next,action,info,backbone=None):
        i=self.interest_fn(info)
        if i<0.01:return 0.0
        c=self.cumulant_fn(info)
        with torch.no_grad():
            if self.head_type=='placement':
                v_next=0.0
            elif self.use_units:
                unit_mask=(units_next.abs().sum(-1)==0)
                v_next=self.head(h_next,units_next,unit_mask).item() if hasattr(self.head,'forward') and self.use_units else self.head(h_next).item()
            else:
                v_next=self.head(h_next).item() if self.head_type=='value' else 0.0
        if self.head_type=='placement':
            target_x=info.get('placement_x',0.5)
            target_y=info.get('placement_y',0.5)
            logits=self.head(h)
            gy=int(min(max(target_y*(GRID_H-1),0),GRID_H-1))
            gx=int(min(max(target_x*(GRID_W-1),0),GRID_W-1))
            target_idx=gy*GRID_W+gx
            loss=F.cross_entropy(logits.view(1,-1),torch.tensor([target_idx],device=h.device))
        elif self.use_units:
            unit_mask=(units.abs().sum(-1)==0)
            v=self.head(h,units,unit_mask)
            target=c+self.gamma*v_next
            loss=F.mse_loss(v,torch.tensor([target],device=h.device))*i
        else:
            v=self.head(h)
            target=c+self.gamma*v_next
            loss=F.mse_loss(v,torch.tensor([target],device=h.device))*i
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.head.parameters(),1.0)
        self.opt.step()
        return loss.item()

class CQLDemonV4:
    def __init__(self,name,reward_fn,gamma=0.99,num_actions=180,cql_alpha=0.5,lr=3e-4,d_model=256):
        self.name=name
        self.reward_fn=reward_fn
        self.gamma=gamma
        self.num_actions=num_actions
        self.cql_alpha=cql_alpha
        self.is_cls=False
        self.q_head=nn.Sequential(nn.Linear(d_model,256),nn.GELU(),nn.LayerNorm(256),
                                  nn.Linear(256,256),nn.GELU(),nn.Linear(256,num_actions))
        self.opt=torch.optim.AdamW(self.q_head.parameters(),lr=lr,weight_decay=1e-5)
    def reset(self):
        for p in self.q_head.parameters():
            if p.grad is not None:p.grad.zero_()
    def observe_and_learn(self,h,h_next,units,units_next,action,info,backbone=None):
        r=self.reward_fn(info)
        with torch.no_grad():
            q_next=self.q_head(h_next).max().item()
        q=self.q_head(h)
        target=r+self.gamma*q_next*(1-float(info.get('ended',False)))
        qa=q[0,action] if q.dim()>1 else q[action]
        td_loss=F.mse_loss(qa,torch.tensor(target,device=h.device))
        logsumexp=torch.logsumexp(q,dim=-1)
        cql_loss=self.cql_alpha*(logsumexp-qa).mean()
        loss=td_loss+cql_loss
        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_head.parameters(),1.0)
        self.opt.step()
        return loss.item()

class AWRDemonV4:
    def __init__(self,name,reward_fn,gamma=0.99,num_actions=180,beta=0.05,lr=3e-4,d_model=256):
        self.name=name
        self.reward_fn=reward_fn
        self.gamma=gamma
        self.num_actions=num_actions
        self.beta=beta
        self.is_cls=False
        self.v_head=nn.Sequential(nn.Linear(d_model,128),nn.GELU(),nn.Linear(128,1))
        self.pi_head=nn.Sequential(nn.Linear(d_model,256),nn.GELU(),nn.LayerNorm(256),
                                   nn.Linear(256,num_actions))
        self.v_opt=torch.optim.AdamW(self.v_head.parameters(),lr=lr)
        self.pi_opt=torch.optim.AdamW(self.pi_head.parameters(),lr=lr)
    def reset(self):
        for p in list(self.v_head.parameters())+list(self.pi_head.parameters()):
            if p.grad is not None:p.grad.zero_()
    def observe_and_learn(self,h,h_next,units,units_next,action,info,backbone=None):
        r=self.reward_fn(info)
        with torch.no_grad():
            v_next=self.v_head(h_next).item()
        v=self.v_head(h).squeeze(-1)
        target=r+self.gamma*v_next*(1-float(info.get('ended',False)))
        v_loss=F.mse_loss(v,torch.tensor([target],device=h.device))
        self.v_opt.zero_grad()
        v_loss.backward()
        self.v_opt.step()
        with torch.no_grad():
            adv=target-v.item()
            weight=min(max(math.exp(adv/self.beta),0.01),20.0)
        logits=self.pi_head(h.detach())
        log_prob=F.log_softmax(logits,dim=-1)
        pi_loss=-weight*log_prob[0,action] if logits.dim()>1 else -weight*log_prob[action]
        self.pi_opt.zero_grad()
        pi_loss.backward()
        self.pi_opt.step()
        return v_loss.item()+pi_loss.item()

def _i_tower(info):
    d=abs(info.get('opp_tower_hp_delta',0))+abs(info.get('team_tower_hp_delta',0))
    return min(d/300.0+0.5,5.0)
def _i_crown(info):return 5.0 if info.get('crown_scored',0)>0 else 1.0
def _i_combat(info):return 2.0 if info.get('troop_count_diff',0)!=0 else 0.5
def _i_placement(info):return 1.0

def build_v4_horde(num_cards=180,device=None):
    dev=device or torch.device('cpu')
    backbone=SharedBackboneV4(d_model=256,nheads=4,nl=2,dp=0.1).to(dev)
    demons=[]
    demons.append(TDDemonV4('tower_dmg_imm',lambda i:i.get('opp_tower_hp_delta',0)/1000.0,
        gamma=0.0,lam=0.0,interest_fn=_i_tower,lr=2e-3,head_type='value'))
    demons.append(TDDemonV4('tower_dmg_short',lambda i:i.get('opp_tower_hp_delta',0)/1000.0,
        gamma=0.7,lam=0.5,interest_fn=_i_tower,lr=1e-3,head_type='value'))
    demons.append(TDDemonV4('tower_dmg_mid',lambda i:i.get('opp_tower_hp_delta',0)/1500.0,
        gamma=0.9,lam=0.7,interest_fn=_i_tower,lr=1e-3,head_type='value'))
    demons.append(TDDemonV4('tower_dmg_long',lambda i:i.get('opp_tower_hp_delta',0)/2000.0,
        gamma=0.97,lam=0.85,interest_fn=_i_tower,lr=5e-4,head_type='value'))
    demons.append(TDDemonV4('tower_taken_imm',lambda i:-i.get('team_tower_hp_delta',0)/1000.0,
        gamma=0.0,lam=0.0,interest_fn=_i_tower,lr=2e-3,head_type='value'))
    demons.append(TDDemonV4('tower_taken_mid',lambda i:-i.get('team_tower_hp_delta',0)/1500.0,
        gamma=0.9,lam=0.7,interest_fn=_i_tower,lr=1e-3,head_type='value'))
    demons.append(TDDemonV4('crown_imm',lambda i:float(i.get('crown_scored',0)),
        gamma=0.0,lam=0.0,interest_fn=_i_crown,lr=2e-3,head_type='value'))
    demons.append(TDDemonV4('crown_short',lambda i:float(i.get('crown_scored',0)),
        gamma=0.85,lam=0.6,interest_fn=_i_crown,lr=1e-3,head_type='value'))
    demons.append(TDDemonV4('crown_long',lambda i:float(i.get('crown_scored',0)),
        gamma=0.97,lam=0.85,interest_fn=_i_crown,lr=5e-4,head_type='value'))
    demons.append(TDDemonV4('elixir_adv',lambda i:max(-1.0,min(1.0,i.get('elixir_advantage',0)/5.0)),
        gamma=0.85,lam=0.6,lr=1e-3,head_type='value'))
    demons.append(TDDemonV4('troop_adv',lambda i:max(-1.0,min(1.0,i.get('troop_count_diff',0)/8.0)),
        gamma=0.8,lam=0.5,interest_fn=_i_combat,lr=1e-3,head_type='unit_aware',use_units=True))
    demons.append(TDDemonV4('lane_left',lambda i:max(-1.0,min(1.0,i.get('lane_pressure_left',0)/4.0)),
        gamma=0.75,lam=0.5,lr=1e-3,head_type='value'))
    demons.append(TDDemonV4('lane_right',lambda i:max(-1.0,min(1.0,i.get('lane_pressure_right',0)/4.0)),
        gamma=0.75,lam=0.5,lr=1e-3,head_type='value'))
    demons.append(TDDemonV4('dmg_efficiency',lambda i:min(i.get('dmg_per_elixir',0)/150.0,1.0),
        gamma=0.9,lam=0.7,lr=1e-3,head_type='value'))
    demons.append(TDDemonV4('placement',lambda i:0.0,
        gamma=0.0,lam=0.0,interest_fn=_i_placement,lr=2e-3,head_type='placement'))
    cql_win=CQLDemonV4('cql_win',lambda i:i.get('win_label',0.5),
        gamma=0.99,num_actions=num_cards,cql_alpha=0.3,lr=5e-4)
    cql_crown=CQLDemonV4('cql_crown',lambda i:float(i.get('crown_diff',0))/3.0,
        gamma=0.95,num_actions=num_cards,cql_alpha=0.5,lr=5e-4)
    awr_offense=AWRDemonV4('awr_offense',
        lambda i:i.get('opp_tower_hp_delta',0)/1500.0+float(i.get('crown_scored',0))*0.5,
        gamma=0.97,num_actions=num_cards,beta=0.1,lr=5e-4)
    awr_defense=AWRDemonV4('awr_defense',lambda i:-i.get('team_tower_hp_delta',0)/1500.0,
        gamma=0.95,num_actions=num_cards,beta=0.15,lr=5e-4)
    return HordeV4(backbone,demons,[cql_win,cql_crown],[awr_offense,awr_defense],dev)

class HordeV4:
    def __init__(self,backbone,td_demons,cql_demons,awr_demons,device):
        self.backbone=backbone
        self.td_demons=td_demons
        self.cql_demons=cql_demons
        self.awr_demons=awr_demons
        self.dev=device
        self.bb_opt=torch.optim.AdamW(backbone.parameters(),lr=3e-4,weight_decay=1e-5)
    def reset_all(self):
        self.backbone.reset()
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
    def observe_and_learn(self,flat,units,grid,flat_next,units_next,grid_next,action,info):
        flat_t=torch.tensor(flat,dtype=torch.float32,device=self.dev).unsqueeze(0)
        units_t=torch.tensor(units,dtype=torch.float32,device=self.dev).unsqueeze(0)
        grid_t=torch.tensor(grid,dtype=torch.float32,device=self.dev).unsqueeze(0)
        flat_next_t=torch.tensor(flat_next,dtype=torch.float32,device=self.dev).unsqueeze(0)
        units_next_t=torch.tensor(units_next,dtype=torch.float32,device=self.dev).unsqueeze(0)
        grid_next_t=torch.tensor(grid_next,dtype=torch.float32,device=self.dev).unsqueeze(0)
        h,u=self.backbone(flat_t,units_t,grid_t)
        with torch.no_grad():
            h_next,u_next=self.backbone(flat_next_t,units_next_t,grid_next_t)
        total_loss=torch.tensor(0.0,device=self.dev)
        losses={}
        for d in self.td_demons:
            dl=self._demon_loss(d,h,h_next,units_t,units_next_t,action,info)
            if dl is not None and torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
            else:losses[d.name]=0.0
        for d in self.cql_demons:
            dl=self._cql_loss(d,h,h_next,action,info)
            if dl is not None and torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
            else:losses[d.name]=0.0
        for d in self.awr_demons:
            dl=self._awr_loss(d,h,h_next,action,info)
            if dl is not None and torch.isfinite(dl):
                total_loss=total_loss+dl
                losses[d.name]=dl.item()
            else:losses[d.name]=0.0
        if total_loss.requires_grad and torch.isfinite(total_loss):
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
    def _demon_loss(self,d,h,h_next,units,units_next,action,info):
        i=d.interest_fn(info)
        if i<0.01:return None
        c=d.cumulant_fn(info)
        with torch.no_grad():
            if d.head_type=='value':v_next=d.head(h_next).item()
            elif d.use_units:
                mk=(units_next.abs().sum(-1)==0)
                v_next=d.head(h_next,units_next,mk).item()
            else:v_next=0.0
        if d.head_type=='placement':
            target_x=info.get('placement_x',0.5)
            target_y=info.get('placement_y',0.5)
            from .features_v4 import GRID_H,GRID_W
            logits=d.head(h)
            gy=int(min(max(target_y*(GRID_H-1),0),GRID_H-1))
            gx=int(min(max(target_x*(GRID_W-1),0),GRID_W-1))
            target_idx=gy*GRID_W+gx
            return F.cross_entropy(logits.view(1,-1),torch.tensor([target_idx],device=h.device))*i
        elif d.use_units:
            mk=(units.abs().sum(-1)==0)
            v=d.head(h,units,mk)
        else:
            v=d.head(h)
        target=c+d.gamma*v_next
        return F.mse_loss(v,torch.tensor([target],device=h.device))*i
    def _cql_loss(self,d,h,h_next,action,info):
        r=d.reward_fn(info)
        with torch.no_grad():q_next=d.q_head(h_next).max().item()
        q=d.q_head(h)
        target=r+d.gamma*q_next*(1-float(info.get('ended',False)))
        qa=q[0,action] if q.dim()>1 else q[action]
        td_loss=F.mse_loss(qa,torch.tensor(target,device=h.device))
        logsumexp=torch.logsumexp(q,dim=-1)
        cql_loss=d.cql_alpha*(logsumexp-qa).mean()
        return td_loss+cql_loss
    def _awr_loss(self,d,h,h_next,action,info):
        r=d.reward_fn(info)
        with torch.no_grad():v_next=d.v_head(h_next).item()
        v=d.v_head(h).squeeze(-1)
        target=r+d.gamma*v_next*(1-float(info.get('ended',False)))
        v_loss=F.mse_loss(v,torch.tensor([target],device=h.device))
        with torch.no_grad():
            adv=target-v.item()
            weight=min(max(math.exp(min(adv/d.beta,5.0)),0.01),20.0)
        logits=d.pi_head(h)
        log_prob=F.log_softmax(logits,dim=-1)
        pi_loss=-weight*(log_prob[0,action] if logits.dim()>1 else log_prob[action])
        return v_loss+pi_loss
    def get_predictions(self,flat,units,grid):
        flat_t=torch.tensor(flat,dtype=torch.float32,device=self.dev).unsqueeze(0)
        units_t=torch.tensor(units,dtype=torch.float32,device=self.dev).unsqueeze(0)
        grid_t=torch.tensor(grid,dtype=torch.float32,device=self.dev).unsqueeze(0)
        h,u=self.backbone(flat_t,units_t,grid_t)
        preds={}
        for d in self.td_demons:
            with torch.no_grad():
                if d.head_type=='placement':
                    px,py=d.head.predict_xy(h)
                    preds[d.name]=(px.item(),py.item())
                elif d.use_units:
                    unit_mask=(units_t.abs().sum(-1)==0)
                    preds[d.name]=d.head(h,units_t,unit_mask).cpu().numpy()
                else:
                    preds[d.name]=d.head(h).cpu().numpy()
        return preds
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
            data[f'awrvopt_{d.name}']=d.v_opt.state_dict()
            data[f'awrpiopt_{d.name}']=d.pi_opt.state_dict()
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
