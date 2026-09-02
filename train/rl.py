import argparse
import random
import time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from sim.game import Game,card_info
from sim.replay import _has_json,_mk_deck,_force_hand
from train.feats import featurize,FEAT_DIM
from train.trajData import DEFAULT_TRAJ_PATH,DECK_COLS

GW,GH=6,8  # 3x4 tile cells over the full 18x32 arena; enemy-half cells only unmask once a pocket opens
CELLS=[[(x,y) for y in range(4*gy,4*gy+4) for x in range(3*gx,3*gx+3)] for gy in range(GH) for gx in range(GW)]

class Policy(nn.Module):
    def __init__(self,nv,hs=256):
        super().__init__()
        self.trunk=nn.Sequential(nn.Linear(FEAT_DIM,hs),nn.GELU(),nn.LayerNorm(hs),nn.Linear(hs,hs),nn.GELU())
        self.card=nn.Linear(hs,nv);self.tile=nn.Linear(hs,GW*GH);self.value=nn.Linear(hs,1)
    def forward(self,f):
        h=self.trunk(f)
        return self.card(h),self.tile(h),self.value(h).squeeze(-1)

def masked(logits,mask):return torch.distributions.Categorical(logits=logits.masked_fill(~mask,-1e9))

def deck_of(rows):
    cards=rows[DECK_COLS].iloc[0].tolist()+rows['card'].tolist() if len(rows) else []
    return _mk_deck([str(c) for c in dict.fromkeys(cards) if _has_json(str(c))][:8])

def score(g,team,opp):
    hp=lambda tm:sum(t.hp for t in g.arena.towers if t.team==tm and t.alive)
    return (hp(opp)-hp(team))/1000.0+3.0*(g.players[team].crowns-g.players[opp].crowns)

def episode(pol,vocab,grp,dev,dt=1.0,team='blue',opp='red'):
    inv={i:c for c,i in vocab.items()}
    ot=grp[grp.side=='o'];g=Game(p1={'deck':deck_of(grp[grp.side=='t'])},p2={'deck':deck_of(ot)})
    oy=ot.y.to_numpy()/1000.0
    if len(oy) and (oy<16).mean()>0.5:oy=32-oy  # opponent rows recorded from their own perspective
    flags=[ot[c].to_numpy() if c in ot else np.zeros(len(ot)) for c in ('evo','hero')]
    plays=list(zip(ot.time/20.0,ot.card.astype(str),ot.x/1000.0,oy,*flags))
    buf=[];pi=0;prev=score(g,team,opp)
    while not g.ended and g.t<300:
        while pi<len(plays) and plays[pi][0]<=g.t:
            _,c,x,y,evo,hero=plays[pi];pi+=1
            if not _has_json(c):continue
            _force_hand(g,opp,c);po=g.players[opp];po.elixir=max(po.elixir,card_info(c)['cost'])
            if not g.play_card(opp,c,x,y,evolved=bool(evo),hero=bool(hero))[0]:
                nx,ny=min((t for cell in CELLS for t in cell if g._valid_deploy(opp,*t)),key=lambda t:(t[0]-x)**2+(t[1]-y)**2)
                g.play_card(opp,c,nx,ny,evolved=bool(evo),hero=bool(hero))
        p=g.players[team];f=torch.tensor(featurize(g,team),device=dev)
        cm=torch.zeros(len(vocab),dtype=torch.bool,device=dev);cm[0]=True  # index 0 is wait
        for c in p.deck.hand:
            if p.elixir>=card_info(c)['cost'] and c in vocab:cm[vocab[c]]=True
        with torch.no_grad():
            cl,tl,v=pol(f[None]);card=masked(cl[0],cm).sample()
        name=inv[card.item()] if card.item() else None
        anywhere=name is not None and card_info(name).get('deploy_anywhere',False)
        tiles=[[t for t in cell if anywhere or g._valid_deploy(team,*t)] for cell in CELLS]
        tm=torch.tensor([bool(t) or name is None for t in tiles],device=dev)
        with torch.no_grad():tile=masked(tl[0],tm).sample()
        if name is not None:g.play_card(team,name,*random.choice(tiles[tile.item()]))
        for _ in range(int(dt/g.DT)):
            if g.ended:break
            g.tick()
        cur=score(g,team,opp);r=cur-prev;prev=cur
        if g.ended:r+=10.0*((g.winner==team)-(g.winner==opp))
        buf.append((f,cm,tm,card,tile,float(name is not None),v.item(),r))
    return buf

def ppo(pol,opt,buf,dev,gamma=0.99,lam=0.95,clip=0.2,epochs=4,ent=0.01):
    f,cm,tm,card,tile=[torch.stack(z) for z in zip(*[b[:5] for b in buf])]
    played,v,r=[torch.tensor(z,dtype=torch.float32,device=dev) for z in zip(*[b[5:] for b in buf])]
    adv=torch.zeros_like(r);last=0.0
    for t in reversed(range(len(r))):
        last=r[t]+gamma*(v[t+1] if t+1<len(r) else 0.0)-v[t]+gamma*lam*last;adv[t]=last
    ret=adv+v;adv=(adv-adv.mean())/(adv.std()+1e-8)
    with torch.no_grad():
        cl,tl,_=pol(f);old=masked(cl,cm).log_prob(card)+played*masked(tl,tm).log_prob(tile)
    for _ in range(epochs):
        cl,tl,val=pol(f);dc,dt=masked(cl,cm),masked(tl,tm)
        ratio=(dc.log_prob(card)+played*dt.log_prob(tile)-old).exp()
        loss=-torch.min(ratio*adv,ratio.clamp(1-clip,1+clip)*adv).mean()+0.5*F.mse_loss(val,ret)-ent*(dc.entropy()+played*dt.entropy()).mean()
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(pol.parameters(),0.5);opt.step()
    return loss.item()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default=DEFAULT_TRAJ_PATH);ap.add_argument('--episodes',type=int,default=1000)
    ap.add_argument('--lr',type=float,default=3e-4);ap.add_argument('--out',default='checkpoints/rl.pt')
    a=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    df=pd.read_csv(a.csv,low_memory=False).dropna(subset=['card']);df['side']=df.side.astype(str).str.strip()
    df=df[~df.card.astype(str).str.contains('ability')].sort_values(['battle_id','time'])
    vocab={'<wait>':0,**{c:i+1 for i,c in enumerate(sorted(set(df.card.astype(str))|set(df[DECK_COLS].astype(str).to_numpy().ravel())))}}
    battles=[g for _,g in df.groupby('battle_id',sort=False) if (g.side=='o').sum()>=4]
    pol=Policy(len(vocab)).to(dev);opt=torch.optim.AdamW(pol.parameters(),lr=a.lr)
    print(f'{dev} {len(battles)} battles, {len(vocab)} cards, feat {FEAT_DIM}',flush=True)
    rew=[];t0=time.time()
    for ep in range(a.episodes):
        buf=episode(pol,vocab,battles[ep%len(battles)],dev)
        loss=ppo(pol,opt,buf,dev);rew.append(sum(b[-1] for b in buf))
        if (ep+1)%20==0 or ep+1==a.episodes:
            print(f'[{ep+1}/{a.episodes}] {time.time()-t0:.0f}s loss {loss:.3f} reward {np.mean(rew[-20:]):.2f}',
                  f'steps {len(buf)} plays {int(sum(b[5] for b in buf))}',flush=True)
    Path(a.out).parent.mkdir(exist_ok=True)
    torch.save({'state_dict':pol.state_dict(),'vocab':vocab},a.out)

if __name__=='__main__':main()
