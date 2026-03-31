import sys,os,math,random,time,argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','simulator'))
from game import Game,card_info
from horde.features import extract_features,FEAT_DIM
from rl_agent import CRAgent,compute_reward,get_state_info,play_vs_replay

class PolicyHead(nn.Module):
    def __init__(self,feat_dim,num_cards=200,hs=256):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(feat_dim,hs),nn.GELU(),nn.LayerNorm(hs),
            nn.Linear(hs,hs),nn.GELU(),
            nn.Linear(hs,num_cards))
        self.value_head=nn.Sequential(
            nn.Linear(feat_dim,hs//2),nn.GELU(),nn.Linear(hs//2,1))
    def forward(self,feat):
        return self.net(feat),self.value_head(feat).squeeze(-1)

class RLTrainer:
    def __init__(self,agent,num_cards=200,lr=3e-4,gamma=0.99,clip_eps=0.2,
                 entropy_coef=0.01,value_coef=0.5,device='cpu'):
        self.agent=agent
        self.dev=torch.device(device)
        self.gamma=gamma
        self.clip_eps=clip_eps
        self.entropy_coef=entropy_coef
        self.value_coef=value_coef
        self.policy=PolicyHead(FEAT_DIM,num_cards).to(self.dev)
        self.opt=torch.optim.AdamW(self.policy.parameters(),lr=lr)
        self.card_to_idx=agent.horde_vocab if agent.horde else {}
        self.idx_to_card={v:k for k,v in self.card_to_idx.items()}
        self.num_cards=num_cards
    def collect_episode(self,game,team,opp_plays):
        opp='red' if team=='blue' else 'blue'
        states=[];actions=[];rewards=[];log_probs=[];values=[]
        opp_idx=0
        prev_info=get_state_info(game,team)
        last_play=0
        while not game.ended and game.t<300:
            while opp_idx<len(opp_plays):
                row=opp_plays.iloc[opp_idx]
                target_t=float(row['time'])/6000.0*300.0
                if target_t>game.t+0.5:break
                card=str(row['card'])
                try:
                    rx=float(row.get('x',9000))/1000.0
                    ry=float(row.get('y',16000))/1000.0
                except:rx,ry=9.0,16.0
                if np.isnan(rx) or np.isnan(ry):rx,ry=9.0,16.0
                game.players[opp].elixir=max(game.players[opp].elixir,card_info(card)['cost'])
                try:game.play_card(opp,card,rx,ry)
                except:pass
                opp_idx+=1
            if game.t-last_play>=2.0 and game.players[team].elixir>=4:
                feat=extract_features(game,team)
                ft=torch.tensor(feat,dtype=torch.float32,device=self.dev).unsqueeze(0)
                with torch.no_grad():
                    logits,val=self.policy(ft)
                hand=[c for c in game.players[team].deck.hand if game.players[team].elixir>=card_info(c)['cost']]
                if hand:
                    hand_mask=torch.full((self.num_cards,),float('-inf'),device=self.dev)
                    for c in hand:
                        idx=self.card_to_idx.get(c,0)
                        if idx<self.num_cards:hand_mask[idx]=0.0
                    masked_logits=logits[0]+hand_mask
                    dist=torch.distributions.Categorical(logits=masked_logits)
                    action=dist.sample()
                    card_name=self.idx_to_card.get(action.item(),hand[0])
                    if card_name not in hand:card_name=hand[0]
                    x,y=self.agent._pick_placement(game,team,card_name)
                    try:
                        game.play_card(team,card_name,x,y)
                        states.append(feat)
                        actions.append(action.item())
                        log_probs.append(dist.log_prob(action).item())
                        values.append(val.item())
                        last_play=game.t
                    except:pass
            for _ in range(5):
                if game.ended:break
                game.tick()
            curr_info=get_state_info(game,team)
            if states and len(rewards)<len(states):
                r=compute_reward(prev_info,curr_info,team)
                rewards.append(r)
            prev_info=curr_info
        while len(rewards)<len(states):
            rewards.append(0.0)
        return states,actions,rewards,log_probs,values
    def compute_returns(self,rewards):
        returns=[]
        G=0
        for r in reversed(rewards):
            G=r+self.gamma*G
            returns.insert(0,G)
        returns=torch.tensor(returns,dtype=torch.float32,device=self.dev)
        if len(returns)>1:returns=(returns-returns.mean())/(returns.std()+1e-8)
        return returns
    def update(self,states,actions,old_log_probs,returns,old_values,epochs=4):
        if not states:return 0.0
        feat_t=torch.tensor(np.stack(states),dtype=torch.float32,device=self.dev)
        act_t=torch.tensor(actions,dtype=torch.long,device=self.dev)
        old_lp=torch.tensor(old_log_probs,dtype=torch.float32,device=self.dev)
        old_v=torch.tensor(old_values,dtype=torch.float32,device=self.dev)
        advantages=returns-old_v
        total_loss=0
        for _ in range(epochs):
            logits,values=self.policy(feat_t)
            dist=torch.distributions.Categorical(logits=logits)
            new_lp=dist.log_prob(act_t)
            ratio=torch.exp(new_lp-old_lp)
            surr1=ratio*advantages
            surr2=torch.clamp(ratio,1-self.clip_eps,1+self.clip_eps)*advantages
            policy_loss=-torch.min(surr1,surr2).mean()
            value_loss=F.mse_loss(values,returns)
            entropy=dist.entropy().mean()
            loss=policy_loss+self.value_coef*value_loss-self.entropy_coef*entropy
            self.opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy.parameters(),0.5)
            self.opt.step()
            total_loss+=loss.item()
        return total_loss/epochs

def train_rl(agent,traj_csv,episodes=1000,eval_every=100,save_path='checkpoints/rl_policy.pt'):
    import pandas as pd
    team='blue';opp='red'
    df=pd.read_csv(traj_csv,low_memory=False)
    df=df.sort_values(['battle_id','time'])
    battles=list(df.groupby('battle_id',sort=False))
    vocab=agent.horde_vocab if agent.horde else {}
    num_cards=max(vocab.values())+1 if vocab else 200
    trainer=RLTrainer(agent,num_cards=num_cards,lr=3e-4)
    print(f"RL Training: {len(battles)} battles, {num_cards} cards",flush=True)
    t0=time.time()
    ep_rewards=[]
    for ep in range(episodes):
        bid,grp=battles[ep%len(battles)]
        grp=grp.reset_index(drop=True)
        if len(grp)<4:continue
        t_cards=[r['card'] for _,r in grp.iterrows() if str(r.get('side','')).strip()=='t']
        o_cards=[r['card'] for _,r in grp.iterrows() if str(r.get('side','')).strip()=='o']
        t_deck=list(dict.fromkeys(t_cards))[:8]
        o_deck=list(dict.fromkeys(o_cards))[:8]
        if len(t_deck)<4 or len(o_deck)<4:continue
        while len(t_deck)<8:t_deck.append(t_deck[len(t_deck)%len(t_deck)])
        while len(o_deck)<8:o_deck.append(o_deck[len(o_deck)%len(o_deck)])
        try:g=Game(p1={'deck':t_deck},p2={'deck':o_deck})
        except:continue
        opp_plays=grp[grp['side'].astype(str).str.strip()=='o'].reset_index(drop=True)
        states,actions,rewards,log_probs,values=trainer.collect_episode(g,team,opp_plays)
        if not states:continue
        returns=trainer.compute_returns(rewards)
        loss=trainer.update(states,actions,log_probs,returns,values)
        ep_reward=sum(rewards)
        ep_rewards.append(ep_reward)
        if (ep+1)%20==0:
            avg_r=np.mean(ep_rewards[-20:])
            el=time.time()-t0
            print(f'[{ep+1}/{episodes}] {el:.0f}s loss:{loss:.4f} avg_reward:{avg_r:.2f} plays:{len(states)}',flush=True)
        if (ep+1)%eval_every==0:
            print(f'\n--- Evaluation at episode {ep+1} ---')
            torch.save(trainer.policy.state_dict(),save_path)
            play_vs_replay(agent,traj_csv,max_battles=50,verbose=True)
            print()
    torch.save(trainer.policy.state_dict(),save_path)
    print(f'Final save -> {save_path}',flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--planner',default='checkpoints/planner_v5_best.pt')
    ap.add_argument('--horde',default='checkpoints/horde_v8.pt')
    ap.add_argument('--traj-csv',default=None)
    ap.add_argument('--episodes',type=int,default=1000)
    args=ap.parse_args()
    csv=args.traj_csv or str(Path(__file__).resolve().parents[1]/'data'/'ready_data'/'traj.csv')
    agent=CRAgent(planner_ckpt=args.planner,horde_ckpt=args.horde)
    train_rl(agent,csv,episodes=args.episodes)
