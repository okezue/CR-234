import sys,os,argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
_here=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(_here,'..','..','simulator'))
sys.path.insert(0,os.path.join(_here,'..'))
from horde.features import extract_features,FEAT_DIM
from horde.horde_v2 import HordeV2,build_v2_horde
from horde.horde_v3 import HordeV3,build_v3_horde
from horde.horde_v5 import HordeV5,build_v5_horde
from horde.horde_v6 import HordeV6,build_v6_horde
from horde.horde_v7 import HordeV7,build_v7_horde

def _tower_hp(g,team):
    return sum(tw.hp for tw in g.arena.towers if tw.team==team and tw.alive)

def eval_horde(horde,traj_csv,max_battles=100,team='blue'):
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
    for d in getattr(horde,'binary_demons',[]):demon_names.append(d.name)
    for d in horde.cql_demons:demon_names.append(d.name)
    for d in horde.awr_demons:demon_names.append(d.name)
    predictions={name:[] for name in demon_names}
    actuals={name:[] for name in demon_names}
    win_preds=[]
    win_actuals=[]
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
            feat=extract_features(g,team)
            feat_t=torch.tensor(feat,dtype=torch.float32,device=horde.dev).unsqueeze(0)
            with torch.no_grad():
                h=horde.backbone(feat_t)
                for demon in horde.td_demons:
                    v=demon.head(h)
                    if demon.is_cls:
                        episode_preds[demon.name].append(v.argmax().item())
                    else:
                        episode_preds[demon.name].append(v.item())
                for demon in getattr(horde,'binary_demons',[]):
                    v=torch.sigmoid(demon.head(h)).item()
                    episode_preds[demon.name].append(v)
                for demon in horde.cql_demons:
                    q=demon.q_head(h)
                    v=q.max().item()
                    episode_preds[demon.name].append(v)
                for demon in horde.awr_demons:
                    v=demon.v_head(h).item()
                    episode_preds[demon.name].append(v)
            curr_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
            opp_tower_dmg=prev_thp['opp']-curr_thp['opp']
            team_tower_dmg=prev_thp['team']-curr_thp['team']
            cx=9.0
            lp_left=sum(1 for u in g.players[team].troops if u.alive and u.x<cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x<cx)
            lp_right=sum(1 for u in g.players[team].troops if u.alive and u.x>=cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x>=cx)
            troop_diff=len([u for u in g.players[team].troops if u.alive])-len([u for u in g.players[opp].troops if u.alive])
            for demon in horde.td_demons:
                if 'tower_dmg_imm' in demon.name:
                    episode_actuals[demon.name].append(opp_tower_dmg/1000.0)
                elif 'tower_dmg' in demon.name:
                    episode_actuals[demon.name].append(opp_tower_dmg/1000.0)
                elif 'tower_taken_imm' in demon.name:
                    episode_actuals[demon.name].append(-team_tower_dmg/1000.0)
                elif 'tower_taken' in demon.name:
                    episode_actuals[demon.name].append(-team_tower_dmg/1000.0)
                elif 'crown' in demon.name:
                    episode_actuals[demon.name].append(float(g.players[team].crowns))
                elif 'lane_left' in demon.name:
                    episode_actuals[demon.name].append(max(-1.0,min(1.0,lp_left/3.0)))
                elif 'lane_right' in demon.name:
                    episode_actuals[demon.name].append(max(-1.0,min(1.0,lp_right/3.0)))
                elif 'elixir' in demon.name:
                    episode_actuals[demon.name].append(max(-1.0,min(1.0,(g.players[team].elixir-g.players[opp].elixir)/5.0)))
                elif 'troop' in demon.name:
                    episode_actuals[demon.name].append(max(-1.0,min(1.0,troop_diff/5.0)))
                elif 'placement' in demon.name:
                    episode_actuals[demon.name].append(rx/18.0 if 'x' in demon.name else ry/32.0)
                elif 'time' in demon.name:
                    if ri+1<len(grp):
                        dt=(float(grp.iloc[ri+1]['time'])-float(row['time']))/6000.0
                    else:dt=0.0
                    episode_actuals[demon.name].append(dt)
                elif 'dmg_eff' in demon.name:
                    episode_actuals[demon.name].append(min(opp_tower_dmg/300.0,1.0))
                elif 'overtime' in demon.name:
                    episode_actuals[demon.name].append(float(g.players[team].crowns-g.players[opp].crowns)/3.0 if g.t>180 else 0.0)
                elif 'win_pred' in demon.name:
                    episode_actuals[demon.name].append(win_label)
                elif 'total_dmg_dealt' in demon.name:
                    episode_actuals[demon.name].append(opp_tower_dmg/1000.0)
                elif 'total_dmg_taken' in demon.name:
                    episode_actuals[demon.name].append(team_tower_dmg/1000.0)
                else:
                    episode_actuals[demon.name].append(0.0)
            for demon in getattr(horde,'binary_demons',[]):
                if 'will_damage' in demon.name:
                    episode_actuals[demon.name].append(float(opp_tower_dmg>50))
                elif 'will_lose' in demon.name:
                    episode_actuals[demon.name].append(float(team_tower_dmg>50))
                elif 'winning' in demon.name:
                    episode_actuals[demon.name].append(float(win_label>0.5))
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
    print(f"\n=== HORDE GVF Evaluation Results ===\n")
    results={}
    for name in demon_names:
        if not predictions[name]:continue
        preds=np.array(predictions[name])
        acts=np.array(actuals[name])
        mae=np.mean(np.abs(preds-acts))
        corr=np.corrcoef(preds,acts)[0,1] if len(preds)>1 else 0.0
        results[name]={'mae':mae,'corr':corr,'n':len(preds)}
        print(f"{name:25s} MAE={mae:8.4f}  Corr={corr:+.4f}  n={len(preds)}")
    if win_preds:
        win_preds=np.array(win_preds)
        win_actuals=np.array(win_actuals)
        win_preds_binary=(win_preds>0.5).astype(float)
        win_acc=np.mean(win_preds_binary==win_actuals)
        win_corr=np.corrcoef(win_preds,win_actuals)[0,1]
        print(f"\n{'Win Predictor':25s} Acc={win_acc:.4f}  Corr={win_corr:+.4f}  n={len(win_preds)}")
        results['win_predictor']={'acc':win_acc,'corr':win_corr,'n':len(win_preds)}
    return results

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='checkpoints/horde_v3.pt')
    ap.add_argument('--traj-csv',default=None)
    ap.add_argument('--max-battles',type=int,default=100)
    ap.add_argument('--num-cards',type=int,default=180)
    ap.add_argument('--version',type=int,default=7,choices=[2,3,5,6,7])
    args=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')
    if args.version==7:
        horde=build_v7_horde(num_cards=args.num_cards,device=dev)
    elif args.version==6:
        horde=build_v6_horde(num_cards=args.num_cards,device=dev)
    elif args.version==5:
        horde=build_v5_horde(num_cards=args.num_cards,device=dev)
    elif args.version==3:
        horde=build_v3_horde(num_cards=args.num_cards,device=dev)
    else:
        horde=build_v2_horde(num_cards=args.num_cards,device=dev)
    horde.load(args.checkpoint)
    horde.to(dev)
    horde.backbone.eval()
    for d in horde.td_demons:d.head.eval()
    for d in getattr(horde,'binary_demons',[]):d.head.eval()
    for d in horde.cql_demons:d.q_head.eval()
    for d in horde.awr_demons:
        d.v_head.eval()
        d.pi_head.eval()
    nd=len(horde.td_demons)+len(horde.cql_demons)+len(horde.awr_demons)
    print(f'Loaded HordeV{args.version}: {nd} demons')
    csv=args.traj_csv
    if csv is None:
        csv=str(Path(__file__).resolve().parents[2]/'data'/'ready_data'/'traj.csv')
    results=eval_horde(horde,csv,max_battles=args.max_battles)
    print(f'\nEvaluation complete.')

if __name__=='__main__':
    main()
