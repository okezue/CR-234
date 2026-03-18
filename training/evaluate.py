import sys,os,argparse
from pathlib import Path
import torch
import numpy as np
from torch.utils.data import DataLoader,random_split
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from traj_dataloader import TrajDataset,DEFAULT_TRAJ_PATH,pad_collate

def eval_primary(ckpt_path,ds,vl,dev):
    from primary import TrajLSTM
    ckpt=torch.load(ckpt_path,map_location=dev,weights_only=False)
    cfg=ckpt.get("config",{})
    nc=cfg.get("num_cards",ds.get_num_cards())
    m=TrajLSTM(num_cards=nc,emb_dim=cfg.get("emb_dim",16),
               hidden_size=cfg.get("hidden_size",128),num_layers=cfg.get("num_layers",2),
               dropout=cfg.get("dropout",0.2)).to(dev)
    m.load_state_dict(ckpt["state_dict"]);m.eval()
    corr=0;top3=0;top5=0;n=0
    with torch.no_grad():
        for x,lengths,target_xy,target_card in vl:
            x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
            target_card=target_card.to(dev)
            pc=m(x,lengths)
            b=x.size(0)
            corr+=(pc.argmax(1)==target_card).sum().item()
            _,t3=pc.topk(min(3,pc.size(1)),1);top3+=(t3==target_card.unsqueeze(1)).any(1).sum().item()
            _,t5=pc.topk(min(5,pc.size(1)),1);top5+=(t5==target_card.unsqueeze(1)).any(1).sum().item()
            n+=b
    return {"top1":corr/max(n,1),"top3":top3/max(n,1),"top5":top5/max(n,1),"n":n}

def eval_three_lstm(card_path,xy_path,time_path,ds,vl,dev):
    from three_lstm import CardLSTM,XYLSTM,TimeLSTM
    nv=ds.get_num_cards();results={}
    if card_path and Path(card_path).exists():
        ck=torch.load(card_path,map_location=dev,weights_only=False)
        m=CardLSTM(nv).to(dev);m.load_state_dict(ck["state_dict"]);m.eval()
        corr=top3=top5=n=0
        with torch.no_grad():
            for x,lengths,_,target_card in vl:
                x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev);target_card=target_card.to(dev)
                p=m(x,lengths);b=x.size(0)
                corr+=(p.argmax(1)==target_card).sum().item()
                _,t3=p.topk(min(3,p.size(1)),1);top3+=(t3==target_card.unsqueeze(1)).any(1).sum().item()
                _,t5=p.topk(min(5,p.size(1)),1);top5+=(t5==target_card.unsqueeze(1)).any(1).sum().item()
                n+=b
        results["card_top1"]=corr/max(n,1);results["card_top3"]=top3/max(n,1);results["card_top5"]=top5/max(n,1)
    if xy_path and Path(xy_path).exists():
        ck=torch.load(xy_path,map_location=dev,weights_only=False)
        m=XYLSTM(nv).to(dev);m.load_state_dict(ck["state_dict"]);m.eval()
        mse=n=0
        with torch.no_grad():
            for x,lengths,target_xy,_ in vl:
                x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
                tgt=target_xy[:,:2].to(dev,dtype=torch.float32)
                p=m(x,lengths);mse+=((p-tgt)**2).sum().item();n+=x.size(0)
        results["xy_mse"]=mse/max(n,1)
    if time_path and Path(time_path).exists():
        ck=torch.load(time_path,map_location=dev,weights_only=False)
        m=TimeLSTM(nv).to(dev);m.load_state_dict(ck["state_dict"]);m.eval()
        mse=n=0
        with torch.no_grad():
            for x,lengths,target_xy,_ in vl:
                x=x.to(dev,dtype=torch.float32);lengths=lengths.to(dev)
                tgt=target_xy[:,2].to(dev,dtype=torch.float32)
                p=m(x,lengths);mse+=((p-tgt)**2).sum().item();n+=x.size(0)
        results["time_mse"]=mse/max(n,1)
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--primary-ckpt",default="checkpoints/run_best.pt")
    ap.add_argument("--card-ckpt",default="checkpoints/run_card_best.pt")
    ap.add_argument("--xy-ckpt",default="checkpoints/run_xy_best.pt")
    ap.add_argument("--time-ckpt",default="checkpoints/run_time_best.pt")
    ap.add_argument("--max-battles",type=int,default=1000)
    ap.add_argument("--mode",default="both")
    ap.add_argument("--val-frac",type=float,default=0.2)
    args=ap.parse_args()
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds=TrajDataset(DEFAULT_TRAJ_PATH,skip_ability=True,mode=args.mode,max_battle_count=args.max_battles)
    n=len(ds);nv=max(1,int(n*args.val_frac));nt=n-nv
    _,vds=random_split(ds,[nt,nv])
    vl=DataLoader(vds,batch_size=64,shuffle=False,collate_fn=pad_collate)
    print(f"Eval dataset: {nv} samples from {n} total\n")
    print("="*60)
    print("1. Shared LSTM (primary.py)")
    print("="*60)
    if Path(args.primary_ckpt).exists():
        r=eval_primary(args.primary_ckpt,ds,vl,dev)
        print(f"  Card Top1: {r['top1']:.4f}  Top3: {r['top3']:.4f}  Top5: {r['top5']:.4f}")
    else:
        print(f"  Checkpoint not found: {args.primary_ckpt}")
    print(f"\n{'='*60}")
    print("2. Three Specialized LSTMs")
    print("="*60)
    r3=eval_three_lstm(args.card_ckpt,args.xy_ckpt,args.time_ckpt,ds,vl,dev)
    if r3:
        for k,v in r3.items():print(f"  {k}: {v:.6f}")
    else:
        print("  No checkpoints found")
    print(f"\n{'='*60}")
    print("3. GVF Horde")
    print("="*60)
    horde_path="checkpoints/horde.pt"
    if Path(horde_path).exists():
        from horde.horde import build_default_horde
        horde=build_default_horde(num_cards=ds.get_num_cards(),device=dev)
        horde.load(horde_path)
        print("  Horde loaded - prediction demons available for evaluation")
        print(f"  {len(horde.prediction_demons)} prediction demons, {len(horde.control_demons)} control demons")
    else:
        print(f"  Horde checkpoint not found: {horde_path}")
    print(f"\n{'='*60}")
    print("Done.")

if __name__=="__main__":
    main()
