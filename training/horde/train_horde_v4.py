import sys,os,argparse,time
from pathlib import Path
import numpy as np
import torch
_here=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(_here,'..','..','simulator'))
sys.path.insert(0,os.path.join(_here,'..'))
from horde.features_v4 import extract_features_v4,extract_unit_list,extract_grid,FLAT_DIM,UNIT_FEAT,MAX_UNITS,GRID_H,GRID_W,GRID_CHANNELS
from horde.horde_v4 import HordeV4,build_v4_horde

def _tower_hp(g,team):
    return sum(tw.hp for tw in g.arena.towers if tw.team==team and tw.alive)
def _lane_pressure(g,team,side):
    opp='red' if team=='blue' else 'blue'
    cx=9.0
    left_team=sum(1 for u in g.players[team].troops if u.alive and u.x<cx)
    left_opp=sum(1 for u in g.players[opp].troops if u.alive and u.x<cx)
    right_team=sum(1 for u in g.players[team].troops if u.alive and u.x>=cx)
    right_opp=sum(1 for u in g.players[opp].troops if u.alive and u.x>=cx)
    if side=='left':return left_team-left_opp
    return right_team-right_opp
def _elixir_adv(g,team):
    opp='red' if team=='blue' else 'blue'
    return g.players[team].elixir-g.players[opp].elixir

def train_replay(horde,traj_csv,episodes=3000,max_battles=None,team='blue',
                 log_interval=20,save_path=None,save_interval=100):
    import pandas as pd
    from game import Game
    opp='red' if team=='blue' else 'blue'
    print(f"Loading {traj_csv}...",flush=True)
    df=pd.read_csv(traj_csv)
    df=df.sort_values(['battle_id','time'])
    battles=list(df.groupby('battle_id',sort=False))
    if max_battles:battles=battles[:max_battles]
    print(f"  {len(battles)} battles",flush=True)
    all_losses=[];t0=time.time()
    for ep,(bid,grp) in enumerate(battles[:episodes]):
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
        prev_flat=np.zeros(FLAT_DIM,dtype=np.float32)
        prev_units=np.zeros((MAX_UNITS*2,UNIT_FEAT),dtype=np.float32)
        prev_grid=np.zeros((GRID_H,GRID_W,GRID_CHANNELS),dtype=np.float32)
        try:
            prev_flat=extract_features_v4(g,team)[:FLAT_DIM]
            prev_units=extract_unit_list(g,team)
            prev_grid=extract_grid(g,team)
        except:pass
        prev_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
        ep_losses=[]
        result=str(grp.iloc[0].get('result','')).strip()
        win_label=1.0 if result=='W' else 0.0 if result=='L' else 0.5
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
            try:
                curr_flat=extract_features_v4(g,team)[:FLAT_DIM]
                curr_units=extract_unit_list(g,team)
                curr_grid=extract_grid(g,team)
            except:continue
            curr_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
            time_to_next=0
            if ri+1<len(grp):
                time_to_next=(float(grp.iloc[ri+1]['time'])-float(row['time']))/6000.0*300.0
            info={
                'ended':g.ended,'winner':g.winner,
                'opp_tower_hp_delta':prev_thp['opp']-curr_thp['opp'],
                'team_tower_hp_delta':prev_thp['team']-curr_thp['team'],
                'crown_scored':int(g.players[team].crowns>0),
                'crown_diff':g.players[team].crowns-g.players[opp].crowns,
                'time':g.t,
                'dmg_per_elixir':(prev_thp['opp']-curr_thp['opp'])/max(1,3),
                'elixir_advantage':_elixir_adv(g,team),
                'lane_pressure_left':_lane_pressure(g,team,'left'),
                'lane_pressure_right':_lane_pressure(g,team,'right'),
                'troop_count_diff':len([u for u in g.players[team].troops if u.alive])-
                                   len([u for u in g.players[opp].troops if u.alive]),
                'placement_x':rx/18.0,'placement_y':ry/32.0,
                'time_to_next':time_to_next,
                'win_label':win_label,
            }
            losses=horde.observe_and_learn(prev_flat,prev_units,prev_grid,
                                           curr_flat,curr_units,curr_grid,0,info)
            ep_losses.append(losses)
            prev_flat=curr_flat;prev_units=curr_units;prev_grid=curr_grid;prev_thp=curr_thp
            if g.ended:break
        if ep_losses:
            avg={k:np.mean([l[k] for l in ep_losses if k in l]) for k in ep_losses[0]}
            all_losses.append(avg)
            if (ep+1)%log_interval==0:
                el=time.time()-t0
                top5=sorted(avg.items(),key=lambda x:x[1],reverse=True)[:5]
                summary=' '.join(f'{k}:{v:.4f}' for k,v in top5)
                print(f'[{ep+1}/{min(episodes,len(battles))}] {el:.0f}s {summary}',flush=True)
        if save_path and (ep+1)%save_interval==0:
            horde.save(save_path)
            print(f'  Checkpoint -> {save_path}',flush=True)
    if save_path:horde.save(save_path)
    return all_losses

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--traj-csv',default=None)
    ap.add_argument('--episodes',type=int,default=3000)
    ap.add_argument('--max-battles',type=int,default=None)
    ap.add_argument('--save',default='checkpoints/horde_v4.pt')
    ap.add_argument('--num-cards',type=int,default=180)
    ap.add_argument('--log-interval',type=int,default=20)
    ap.add_argument('--save-interval',type=int,default=100)
    args=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}',flush=True)
    horde=build_v4_horde(num_cards=args.num_cards,device=dev)
    horde.to(dev)
    nd=len(horde.td_demons)+len(horde.cql_demons)+len(horde.awr_demons)
    print(f'HordeV4: {nd} demons',flush=True)
    csv=args.traj_csv
    if csv is None:
        csv=str(Path(__file__).resolve().parents[2]/'data'/'ready_data'/'traj.csv')
    Path(args.save).parent.mkdir(parents=True,exist_ok=True)
    train_replay(horde,csv,episodes=args.episodes,max_battles=args.max_battles,
                 save_path=args.save,log_interval=args.log_interval,
                 save_interval=args.save_interval)
    print(f'Final save -> {args.save}',flush=True)

if __name__=='__main__':
    main()
