import sys,os,argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
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
    if side=='left':
        return sum(1 for u in g.players[team].troops if u.alive and u.x<cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x<cx)
    return sum(1 for u in g.players[team].troops if u.alive and u.x>=cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x>=cx)

def eval_horde_v4(horde,traj_csv,max_battles=100,team='blue'):
    from game import Game
    opp='red' if team=='blue' else 'blue'
    print(f"Loading {traj_csv}...")
    df=pd.read_csv(traj_csv)
    df=df.sort_values(['battle_id','time'])
    battles=list(df.groupby('battle_id',sort=False))
    if max_battles:battles=battles[:max_battles]
    print(f"Evaluating on {len(battles)} battles")
    demon_names=[]
    for d in horde.td_demons:demon_names.append(d.name)
    for d in horde.cql_demons:demon_names.append(d.name)
    for d in horde.awr_demons:demon_names.append(d.name)
    predictions={name:[] for name in demon_names}
    actuals={name:[] for name in demon_names}
    win_preds=[];win_actuals=[]
    t0=time.time()
    for ep,(bid,grp) in enumerate(battles):
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
        result=str(grp.iloc[0].get('result','')).strip()
        win_label=1.0 if result=='W' else 0.0 if result=='L' else 0.5
        prev_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
        episode_preds={name:[] for name in demon_names}
        episode_actuals={name:[] for name in demon_names}
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
                flat=extract_features_v4(g,team)[:FLAT_DIM]
                units=extract_unit_list(g,team)
                grid=extract_grid(g,team)
            except:continue
            flat_t=torch.tensor(flat,dtype=torch.float32,device=horde.dev).unsqueeze(0)
            units_t=torch.tensor(units,dtype=torch.float32,device=horde.dev).unsqueeze(0)
            grid_t=torch.tensor(grid,dtype=torch.float32,device=horde.dev).unsqueeze(0)
            with torch.no_grad():
                h,u=horde.backbone(flat_t,units_t,grid_t)
                for demon in horde.td_demons:
                    if demon.head_type=='placement':
                        px,py=demon.head.predict_xy(h)
                        episode_preds[demon.name].append((px.item(),py.item()))
                    elif demon.use_units:
                        unit_mask=(units_t.abs().sum(-1)==0)
                        v=demon.head(h,units_t,unit_mask)
                        episode_preds[demon.name].append(v.item())
                    else:
                        v=demon.head(h)
                        episode_preds[demon.name].append(v.item())
                for demon in horde.cql_demons:
                    q=demon.q_head(h)
                    episode_preds[demon.name].append(q.max().item())
                for demon in horde.awr_demons:
                    v=demon.v_head(h)
                    episode_preds[demon.name].append(v.item())
            curr_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
            opp_tower_dmg=prev_thp['opp']-curr_thp['opp']
            team_tower_dmg=prev_thp['team']-curr_thp['team']
            for demon in horde.td_demons:
                if demon.name=='placement':
                    episode_actuals[demon.name].append((rx/18.0,ry/32.0))
                elif 'tower_dmg' in demon.name:
                    episode_actuals[demon.name].append(opp_tower_dmg)
                elif 'tower_taken' in demon.name:
                    episode_actuals[demon.name].append(team_tower_dmg)
                elif 'crown' in demon.name:
                    episode_actuals[demon.name].append(float(g.players[team].crowns))
                elif 'lane_left' in demon.name:
                    episode_actuals[demon.name].append(_lane_pressure(g,team,'left'))
                elif 'lane_right' in demon.name:
                    episode_actuals[demon.name].append(_lane_pressure(g,team,'right'))
                elif 'elixir' in demon.name:
                    episode_actuals[demon.name].append(g.players[team].elixir-g.players[opp].elixir)
                elif 'troop' in demon.name:
                    episode_actuals[demon.name].append(
                        len([u for u in g.players[team].troops if u.alive])-
                        len([u for u in g.players[opp].troops if u.alive]))
                elif 'dmg_eff' in demon.name:
                    episode_actuals[demon.name].append(opp_tower_dmg/max(1,3))
                else:
                    episode_actuals[demon.name].append(0.0)
            for demon in horde.cql_demons:
                episode_actuals[demon.name].append(win_label)
            for demon in horde.awr_demons:
                episode_actuals[demon.name].append(win_label)
            prev_thp=curr_thp
            if g.ended:break
        for name in demon_names:
            if episode_preds[name]:
                predictions[name].extend(episode_preds[name])
                actuals[name].extend(episode_actuals[name])
        if horde.cql_demons and episode_preds[horde.cql_demons[0].name]:
            win_preds.append(np.mean(episode_preds[horde.cql_demons[0].name]))
            win_actuals.append(win_label)
        if (ep+1)%20==0:
            print(f"  [{ep+1}/{len(battles)}] {time.time()-t0:.0f}s")
    print(f"\n=== HORDE V4 GVF Evaluation Results ===\n")
    results={}
    for name in demon_names:
        if not predictions[name]:continue
        preds=predictions[name]
        acts=actuals[name]
        if name=='placement':
            px=np.array([p[0] for p in preds])
            py=np.array([p[1] for p in preds])
            ax=np.array([a[0] for a in acts])
            ay=np.array([a[1] for a in acts])
            mae_x=np.mean(np.abs(px-ax))
            mae_y=np.mean(np.abs(py-ay))
            corr_x=np.corrcoef(px,ax)[0,1] if len(px)>1 else 0.0
            corr_y=np.corrcoef(py,ay)[0,1] if len(py)>1 else 0.0
            print(f"{'placement_x':25s} MAE={mae_x:8.4f}  Corr={corr_x:+.4f}  n={len(px)}")
            print(f"{'placement_y':25s} MAE={mae_y:8.4f}  Corr={corr_y:+.4f}  n={len(py)}")
            results['placement_x']={'mae':mae_x,'corr':corr_x,'n':len(px)}
            results['placement_y']={'mae':mae_y,'corr':corr_y,'n':len(py)}
            continue
        preds=np.array(preds)
        acts=np.array(acts)
        mae=np.mean(np.abs(preds-acts))
        corr=np.corrcoef(preds,acts)[0,1] if len(preds)>1 and np.std(preds)>1e-8 and np.std(acts)>1e-8 else 0.0
        results[name]={'mae':mae,'corr':corr,'n':len(preds)}
        print(f"{name:25s} MAE={mae:8.4f}  Corr={corr:+.4f}  n={len(preds)}")
    if win_preds:
        win_preds=np.array(win_preds)
        win_actuals=np.array(win_actuals)
        wp_binary=(win_preds>np.median(win_preds)).astype(float)
        win_acc=np.mean(wp_binary==win_actuals)
        win_corr=np.corrcoef(win_preds,win_actuals)[0,1] if np.std(win_preds)>1e-8 and np.std(win_actuals)>1e-8 else 0.0
        print(f"\n{'Win Predictor':25s} Acc={win_acc:.4f}  Corr={win_corr:+.4f}  n={len(win_preds)}")
        results['win_predictor']={'acc':win_acc,'corr':win_corr,'n':len(win_preds)}
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='checkpoints/horde_v4.pt')
    ap.add_argument('--traj-csv',default=None)
    ap.add_argument('--max-battles',type=int,default=100)
    ap.add_argument('--num-cards',type=int,default=180)
    args=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')
    horde=build_v4_horde(num_cards=args.num_cards,device=dev)
    horde.load(args.checkpoint)
    horde.to(dev)
    horde.backbone.eval()
    for d in horde.td_demons:d.head.eval()
    for d in horde.cql_demons:d.q_head.eval()
    for d in horde.awr_demons:
        d.v_head.eval();d.pi_head.eval()
    nd=len(horde.td_demons)+len(horde.cql_demons)+len(horde.awr_demons)
    print(f'Loaded HordeV4: {nd} demons')
    csv=args.traj_csv
    if csv is None:
        csv=str(Path(__file__).resolve().parents[2]/'data'/'ready_data'/'traj.csv')
    results=eval_horde_v4(horde,csv,max_battles=args.max_battles)
    print(f'\nEvaluation complete.')

if __name__=='__main__':
    main()
