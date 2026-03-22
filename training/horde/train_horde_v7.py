import sys,os,argparse,time
from pathlib import Path
import numpy as np
import torch
_here=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(_here,'..','..','simulator'))
sys.path.insert(0,os.path.join(_here,'..'))
from horde.features import extract_features,FEAT_DIM
from horde.horde_v7 import HordeV7,build_v7_horde

def _tower_hp(g,team):
    return sum(tw.hp for tw in g.arena.towers if tw.team==team and tw.alive)
def _tower_hp_max(g,team):
    return sum(tw.max_hp for tw in g.arena.towers if tw.team==team)
def _lane_pressure(g,team,side):
    opp='red' if team=='blue' else 'blue'
    cx=9.0
    if side=='left':
        return sum(1 for u in g.players[team].troops if u.alive and u.x<cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x<cx)
    return sum(1 for u in g.players[team].troops if u.alive and u.x>=cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x>=cx)

def train(horde,traj_csv,episodes=15000,team='blue',log_interval=20,save_path=None,save_interval=200):
    import pandas as pd
    from game import Game
    opp='red' if team=='blue' else 'blue'
    print(f"Loading {traj_csv}...",flush=True)
    df=pd.read_csv(traj_csv)
    df=df.sort_values(['battle_id','time'])
    battles=list(df.groupby('battle_id',sort=False))
    print(f"  {len(battles)} battles",flush=True)
    horde.init_scheduler(episodes*30,warmup=1000)
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
        while len(t_deck)<8:t_deck.append(t_deck[len(t_deck)%max(len(t_deck),1)])
        while len(o_deck)<8:o_deck.append(o_deck[len(o_deck)%max(len(o_deck),1)])
        try:g=Game(p1={'deck':t_deck},p2={'deck':o_deck})
        except:continue
        horde.reset_all()
        prev_feat=extract_features(g,team)
        prev_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
        opp_max_hp=max(_tower_hp_max(g,opp),1)
        team_max_hp=max(_tower_hp_max(g,team),1)
        init_opp_hp=prev_thp['opp']
        init_team_hp=prev_thp['team']
        result=str(grp.iloc[0].get('result','')).strip()
        win_label=1.0 if result=='W' else 0.0 if result=='L' else 0.5
        ep_losses=[]
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
            try:g.play_card(play_team,card,rx,ry)
            except:continue
            curr_feat=extract_features(g,team)
            curr_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
            cum_opp_dmg=(init_opp_hp-curr_thp['opp'])/opp_max_hp
            cum_team_dmg=(init_team_hp-curr_thp['team'])/team_max_hp
            info={
                'ended':g.ended,'winner':g.winner,
                'opp_tower_hp_delta':prev_thp['opp']-curr_thp['opp'],
                'team_tower_hp_delta':prev_thp['team']-curr_thp['team'],
                'cum_opp_dmg_pct':cum_opp_dmg,
                'cum_team_dmg_pct':cum_team_dmg,
                'crown_scored':int(g.players[team].crowns>0),
                'crown_diff':g.players[team].crowns-g.players[opp].crowns,
                'time':g.t,
                'dmg_per_elixir':(prev_thp['opp']-curr_thp['opp'])/max(1,3),
                'elixir_advantage':g.players[team].elixir-g.players[opp].elixir,
                'lane_pressure_left':_lane_pressure(g,team,'left'),
                'lane_pressure_right':_lane_pressure(g,team,'right'),
                'troop_count_diff':len([u for u in g.players[team].troops if u.alive])-
                                   len([u for u in g.players[opp].troops if u.alive]),
                'placement_x':rx/18.0,'placement_y':ry/32.0,
                'win_label':win_label,
            }
            losses=horde.observe_and_learn(prev_feat,0,curr_feat,info)
            ep_losses.append(losses)
            prev_feat=curr_feat;prev_thp=curr_thp
            if g.ended:break
        if ep_losses and (ep+1)%log_interval==0:
            avg={k:np.mean([l[k] for l in ep_losses if k in l]) for k in ep_losses[0]}
            el=time.time()-t0
            top5=sorted(avg.items(),key=lambda x:x[1],reverse=True)[:5]
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
    ap.add_argument('--episodes',type=int,default=15000)
    ap.add_argument('--save',default='checkpoints/horde_v7.pt')
    ap.add_argument('--num-cards',type=int,default=180)
    ap.add_argument('--hs',type=int,default=384)
    args=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}',flush=True)
    horde=build_v7_horde(num_cards=args.num_cards,hs=args.hs,device=dev)
    horde.to(dev)
    nd=len(horde.td_demons)+len(horde.binary_demons)+len(horde.awr_demons)
    print(f'HordeV7: {nd} demons, hs={args.hs}',flush=True)
    csv=args.traj_csv
    if csv is None:
        csv=str(Path(__file__).resolve().parents[2]/'data'/'ready_data'/'traj.csv')
    Path(args.save).parent.mkdir(parents=True,exist_ok=True)
    train(horde,csv,episodes=args.episodes,save_path=args.save)

if __name__=='__main__':
    main()
