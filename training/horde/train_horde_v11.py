import sys,os,argparse,time
from pathlib import Path
import numpy as np
import torch
_here=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(_here,'..','..','simulator'))
sys.path.insert(0,os.path.join(_here,'..'))
from horde.features import extract_features,FEAT_DIM
from horde.horde_v11 import HordeV11,build_v11_horde

def _tower_hp(g,team):
    return sum(tw.hp for tw in g.arena.towers if tw.team==team and tw.alive)
def _lane_pressure(g,team,side):
    opp='red' if team=='blue' else 'blue'
    cx=9.0
    if side=='left':
        return sum(1 for u in g.players[team].troops if u.alive and u.x<cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x<cx)
    return sum(1 for u in g.players[team].troops if u.alive and u.x>=cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x>=cx)

def build_card_vocab(traj_csv):
    import pandas as pd
    df=pd.read_csv(traj_csv,usecols=['card'],low_memory=False)
    cards=sorted(df['card'].dropna().unique().tolist())
    return {c:i+1 for i,c in enumerate(cards)}

def train(horde,traj_csv,card_vocab,episodes=35000,team='blue',
          log_interval=20,save_path=None,save_interval=200):
    import pandas as pd
    from game import Game,card_info
    opp='red' if team=='blue' else 'blue'
    print(f"Loading {traj_csv}...",flush=True)
    df=pd.read_csv(traj_csv,low_memory=False)
    df=df.sort_values(['battle_id','time'])
    battles=list(df.groupby('battle_id',sort=False))
    print(f"  {len(battles)} battles, {len(card_vocab)} cards",flush=True)
    t0=time.time()
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
        prev_feat=extract_features(g,team)
        prev_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
        feats=[];deltas=[];card_idxs=[]
        cumulants={d.name:[] for d in horde.td_demons}
        binary_targets={d.name:[] for d in horde.binary_demons}
        for ri in range(1,len(grp)):
            row=grp.iloc[ri]
            target_t=float(row['time'])/6000.0*300.0
            steps=0
            while g.t<target_t and not g.ended and steps<2000:
                g.tick();steps+=1
            side=str(row.get('side','')).strip()
            card=str(row['card'])
            try:
                rx=float(row.get('x',9000))/1000.0
                ry=float(row.get('y',16000))/1000.0
            except:rx,ry=9.0,16.0
            if np.isnan(rx) or np.isnan(ry):rx,ry=9.0,16.0
            play_team=team if side=='t' else opp
            g.players[play_team].elixir=10
            try:
                ci=card_info(card)
                card_cost=ci.get('cost',4)
            except:card_cost=4
            try:g.play_card(play_team,card,rx,ry)
            except:continue
            curr_feat=extract_features(g,team)
            curr_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
            delta=curr_feat-prev_feat
            card_idx=card_vocab.get(card,0)
            opp_dmg=prev_thp['opp']-curr_thp['opp']
            team_dmg=prev_thp['team']-curr_thp['team']
            feats.append(curr_feat)
            deltas.append(delta)
            card_idxs.append(card_idx)
            cumulants['tower_dmg_imm'].append(opp_dmg/800.0)
            cumulants['tower_dmg_short'].append(opp_dmg/800.0)
            cumulants['tower_dmg_mid'].append(opp_dmg/1200.0)
            cumulants['tower_dmg_long'].append(opp_dmg/1500.0)
            cumulants['tower_taken_imm'].append(-team_dmg/800.0)
            cumulants['tower_taken_mid'].append(-team_dmg/1200.0)
            cumulants['crown_imm'].append(float(g.players[team].crowns>0))
            cumulants['crown_short'].append(float(g.players[team].crowns>0))
            cumulants['crown_long'].append(float(g.players[team].crowns>0))
            cumulants['elixir_adv'].append(max(-1,min(1,(g.players[team].elixir-g.players[opp].elixir)/5.0)))
            cumulants['lane_left'].append(max(-1,min(1,_lane_pressure(g,team,'left')/3.0)))
            cumulants['lane_right'].append(max(-1,min(1,_lane_pressure(g,team,'right')/3.0)))
            td=len([u for u in g.players[team].troops if u.alive])-len([u for u in g.players[opp].troops if u.alive])
            cumulants['troop_adv'].append(max(-1,min(1,td/5.0)))
            cumulants['dmg_efficiency'].append(min(opp_dmg/max(card_cost*100,1),1.0))
            cumulants['placement_x'].append(rx/18.0)
            cumulants['placement_y'].append(ry/32.0)
            binary_targets['will_damage'].append(float(opp_dmg>50))
            binary_targets['will_lose_tower'].append(float(team_dmg>50))
            prev_feat=curr_feat;prev_thp=curr_thp
            if g.ended:break
        if len(feats)<2:continue
        losses=horde.train_episode(feats,deltas,card_idxs,cumulants,binary_targets)
        if (ep+1)%log_interval==0:
            el=time.time()-t0
            top5=sorted(losses.items(),key=lambda x:x[1],reverse=True)[:5]
            summary=' '.join(f'{k}:{v:.4f}' for k,v in top5)
            print(f'[{ep+1}/{episodes}] {el:.0f}s {summary}',flush=True)
        if save_path and (ep+1)%save_interval==0:
            horde.save(save_path)
            print(f'  Checkpoint -> {save_path}',flush=True)
    if save_path:
        horde.save(save_path)
        print(f'Final save -> {save_path}',flush=True)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--traj-csv',default=None)
    ap.add_argument('--episodes',type=int,default=35000)
    ap.add_argument('--save',default='checkpoints/horde_v11.pt')
    ap.add_argument('--hs',type=int,default=384)
    args=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}',flush=True)
    csv=args.traj_csv
    if csv is None:
        csv=str(Path(__file__).resolve().parents[2]/'data'/'ready_data'/'traj_combined_v2.csv')
    vocab=build_card_vocab(csv)
    num_cards=len(vocab)+1
    horde=build_v11_horde(num_cards=num_cards,hs=args.hs,device=dev)
    horde.to(dev)
    nd=len(horde.td_demons)+len(horde.binary_demons)
    print(f'HordeV11: {nd} demons, hs={args.hs}, {num_cards} cards',flush=True)
    Path(args.save).parent.mkdir(parents=True,exist_ok=True)
    train(horde,csv,vocab,episodes=args.episodes,save_path=args.save)

if __name__=='__main__':
    main()
