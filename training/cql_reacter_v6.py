import sys,os,math,argparse,time
from pathlib import Path
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,random_split
from tqdm import tqdm
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from traj_dataloader_v4 import TrajDatasetV4,pad_collate_v4
from cql_reacter_v4 import CardEmb,RotaryPE,TransformerBlock,CrossAttnBlock,GamePhaseMoE,topk_acc
N_CONT=4;N_CARD_FIELDS=13

class FocalLossWeighted(nn.Module):
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
            alpha_t=self.alpha.to(logits.device).gather(0,targets)
            focal_loss=alpha_t*focal_loss
        if self.reduction=='mean':return focal_loss.mean()
        return focal_loss

from cql_reacter_v5 import ReacterV5 as ReacterV6

def compute_class_weights(ds,nv):
    counts=torch.zeros(nv)
    for s in ds.samples:counts[s[5]]+=1
    total=counts.sum()
    weights=torch.sqrt(total/(nv*counts.clamp(min=1)))
    weights=weights.clamp(max=10.0)
    weights[0]=0.0
    return weights

def train_v6(csv_path=None,name='reacter_v6',epochs=10,bs=8,lr=1.2e-4,
             hs=512,nl=8,ed=64,dp=0.15,cql_alpha=0.2,focal_gamma=2.0,
             label_smooth=0.1,mirror_prob=0.5,max_battles=None,val_frac=0.15,warmup_epochs=2):
    if csv_path is None:csv_path=str(Path(__file__).resolve().parents[1]/'data'/'ready_data'/'traj_50k.csv')
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}',flush=True)
    ds=TrajDatasetV4(csv_path,skip_ability=True,mode='reacter',max_battle_count=max_battles,
                     opp_context=12,mirror_prob=mirror_prob)
    nv=ds.get_num_cards()
    n=len(ds);nvl=max(1,int(n*val_frac));ntr=n-nvl
    tds,vds=random_split(ds,[ntr,nvl])
    tl=DataLoader(tds,batch_size=bs,shuffle=True,collate_fn=pad_collate_v4,num_workers=0,pin_memory=True)
    vl=DataLoader(vds,batch_size=bs,shuffle=False,collate_fn=pad_collate_v4,num_workers=0)
    print(f'Dataset: {n} samples ({ntr} train, {nvl} val), {nv} cards, mirror={mirror_prob}',flush=True)
    model=ReacterV6(nv,ed,hs,nl,nheads=8,dp=dp).to(dev)
    nparams=sum(p.numel() for p in model.parameters())
    print(f'Model params: {nparams:,}',flush=True)
    alpha=None
    opt=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=0.05,betas=(0.9,0.98))
    total_steps=len(tl)*epochs
    warmup_steps=len(tl)*warmup_epochs
    def lr_fn(step):
        if step<warmup_steps:return step/max(1,warmup_steps)
        prog=(step-warmup_steps)/max(1,total_steps-warmup_steps)
        return max(0.1,0.5*(1+math.cos(math.pi*prog)))
    sched=torch.optim.lr_scheduler.LambdaLR(opt,lr_fn)
    focal=FocalLossWeighted(gamma=focal_gamma,alpha=alpha)
    mse=nn.MSELoss()
    ckdir=Path('checkpoints');ckdir.mkdir(exist_ok=True)
    best_top1=0;t0=time.time()
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
                if label_smooth>0:
                    nll=F.log_softmax(q,dim=-1)
                    smooth_loss=-nll.mean(dim=-1).mean()
                    card_loss=(1-label_smooth)*card_loss+label_smooth*(label_smooth/nv)*smooth_loss
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
            sched.step()
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
                l=F.cross_entropy(q,target_card)
                b=x.size(0);vloss+=l.item()*b;vn+=b
                tk=topk_acc(q,target_card)
                for k in vtop:vtop[k]+=tk.get(k,0)*b
        vl_loss=vloss/max(vn,1)
        vtop={k:v/max(vn,1) for k,v in vtop.items()}
        el=time.time()-t0
        print(f'Ep {ep+1} train:{eloss/ns:.4f} val:{vl_loss:.4f} top1:{vtop[1]:.4f} top3:{vtop[3]:.4f} top5:{vtop[5]:.4f} {el:.0f}s',flush=True)
        if vtop[1]>best_top1:
            best_top1=vtop[1]
            torch.save({'state_dict':model.state_dict(),'vocab':ds.get_vocab(),
                        'config':{'nv':nv,'ed':ed,'hs':hs,'nl':nl,'dp':dp},
                        'epoch':ep,'vtop':vtop},str(ckdir/f'{name}_best.pt'))
            print(f'  Best model saved (top1={vtop[1]:.4f})',flush=True)
    torch.save({'state_dict':model.state_dict(),'vocab':ds.get_vocab(),
                'config':{'nv':nv,'ed':ed,'hs':hs,'nl':nl,'dp':dp}},
               str(ckdir/f'{name}_final.pt'))
    print(f'Final -> checkpoints/{name}_final.pt',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=None)
    ap.add_argument('--name',default='reacter_v6')
    ap.add_argument('--epochs',type=int,default=10)
    ap.add_argument('--bs',type=int,default=8)
    ap.add_argument('--lr',type=float,default=1.2e-4)
    ap.add_argument('--hs',type=int,default=512)
    ap.add_argument('--nl',type=int,default=8)
    ap.add_argument('--ed',type=int,default=64)
    ap.add_argument('--mirror',type=float,default=0.5)
    ap.add_argument('--max-battles',type=int,default=None)
    args=ap.parse_args()
    train_v6(csv_path=args.csv,name=args.name,epochs=args.epochs,
             bs=args.bs,lr=args.lr,hs=args.hs,nl=args.nl,ed=args.ed,
             mirror_prob=args.mirror,max_battles=args.max_battles)

if __name__=='__main__':
    main()
