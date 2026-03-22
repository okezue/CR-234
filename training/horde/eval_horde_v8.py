import sys,os,argparse,time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
_here=os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0,os.path.join(_here,'..','..','simulator'))
sys.path.insert(0,os.path.join(_here,'..'))
from horde.features import extract_features,FEAT_DIM
from horde.horde_v8 import HordeV8,build_v8_horde

def _tower_hp(g,team):
    return sum(tw.hp for tw in g.arena.towers if tw.team==team and tw.alive)

def build_card_vocab(traj_csv):
    df=pd.read_csv(traj_csv,usecols=['card'],low_memory=False)
    cards=sorted(df['card'].dropna().unique().tolist())
    return {c:i+1 for i,c in enumerate(cards)}

def eval_v8(horde,traj_csv,card_vocab,max_battles=200,team='blue'):
    from game import Game
    opp='red' if team=='blue' else 'blue'
    print(f"Loading {traj_csv}...")
    df=pd.read_csv(traj_csv,low_memory=False)
    df=df.sort_values(['battle_id','time'])
    battles=list(df.groupby('battle_id',sort=False))
    if max_battles:battles=battles[:max_battles]
    print(f"Evaluating on {len(battles)} battles")
    demon_names=[d.name for d in horde.td_demons]+[d.name for d in horde.binary_demons]+[d.name for d in horde.awr_demons]
    predictions={n:[] for n in demon_names}
    actuals={n:[] for n in demon_names}
    t0=time.time()
    for ep,(bid,grp) in enumerate(battles):
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
        result=str(grp.iloc[0].get('result','')).strip()
        win_label=1.0 if result=='W' else 0.0 if result=='L' else 0.5
        prev_feat=extract_features(g,team)
        prev_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
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
            delta=curr_feat-prev_feat
            card_idx=card_vocab.get(card,0)
            feat_t=torch.tensor(curr_feat,dtype=torch.float32,device=horde.dev).unsqueeze(0)
            delta_t=torch.tensor(delta,dtype=torch.float32,device=horde.dev).unsqueeze(0)
            card_t=torch.tensor([card_idx],dtype=torch.long,device=horde.dev)
            curr_thp={'team':_tower_hp(g,team),'opp':_tower_hp(g,opp)}
            opp_dmg=prev_thp['opp']-curr_thp['opp']
            team_dmg=prev_thp['team']-curr_thp['team']
            cx=9.0
            lp_left=sum(1 for u in g.players[team].troops if u.alive and u.x<cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x<cx)
            lp_right=sum(1 for u in g.players[team].troops if u.alive and u.x>=cx)-sum(1 for u in g.players[opp].troops if u.alive and u.x>=cx)
            td=len([u for u in g.players[team].troops if u.alive])-len([u for u in g.players[opp].troops if u.alive])
            with torch.no_grad():
                h=horde.backbone(feat_t,delta_t,card_t)
                for d in horde.td_demons:
                    v=d.head(h).item()
                    predictions[d.name].append(v)
                for d in horde.binary_demons:
                    v=torch.sigmoid(d.head(h)).item()
                    predictions[d.name].append(v)
                for d in horde.awr_demons:
                    v=d.v_head(h).item()
                    predictions[d.name].append(v)
            for d in horde.td_demons:
                if 'tower_dmg' in d.name:actuals[d.name].append(opp_dmg/800.0)
                elif 'tower_taken' in d.name:actuals[d.name].append(-team_dmg/800.0)
                elif 'crown' in d.name:actuals[d.name].append(float(g.players[team].crowns))
                elif 'lane_left' in d.name:actuals[d.name].append(max(-1,min(1,lp_left/3.0)))
                elif 'lane_right' in d.name:actuals[d.name].append(max(-1,min(1,lp_right/3.0)))
                elif 'elixir' in d.name:actuals[d.name].append(max(-1,min(1,(g.players[team].elixir-g.players[opp].elixir)/5.0)))
                elif 'troop' in d.name:actuals[d.name].append(max(-1,min(1,td/5.0)))
                elif 'placement_x' in d.name:actuals[d.name].append(rx/18.0)
                elif 'placement_y' in d.name:actuals[d.name].append(ry/32.0)
                elif 'dmg_eff' in d.name:actuals[d.name].append(min(opp_dmg/300.0,1.0))
                elif 'win_pred' in d.name:actuals[d.name].append(win_label)
                else:actuals[d.name].append(0.0)
            for d in horde.binary_demons:
                if 'will_damage' in d.name:actuals[d.name].append(float(opp_dmg>50))
                elif 'will_lose' in d.name:actuals[d.name].append(float(team_dmg>50))
                elif 'winning' in d.name:actuals[d.name].append(float(win_label>0.5))
                else:actuals[d.name].append(0.0)
            for d in horde.awr_demons:actuals[d.name].append(win_label)
            prev_feat=curr_feat;prev_thp=curr_thp
            if g.ended:break
        if (ep+1)%50==0:print(f"  [{ep+1}/{len(battles)}] {time.time()-t0:.0f}s")
    print(f"\n=== HORDE V8 Evaluation Results ===\n")
    for n in demon_names:
        if not predictions[n]:continue
        p=np.array(predictions[n]);a=np.array(actuals[n])
        mae=np.mean(np.abs(p-a))
        corr=np.corrcoef(p,a)[0,1] if len(p)>1 and np.std(p)>1e-8 and np.std(a)>1e-8 else 0.0
        print(f"{n:25s} MAE={mae:8.4f}  Corr={corr:+.4f}  n={len(p)}")

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--checkpoint',default='checkpoints/horde_v8.pt')
    ap.add_argument('--traj-csv',default=None)
    ap.add_argument('--max-battles',type=int,default=200)
    args=ap.parse_args()
    dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {dev}')
    csv=args.traj_csv
    if csv is None:csv=str(Path(__file__).resolve().parents[2]/'data'/'ready_data'/'traj.csv')
    full_csv=str(Path(__file__).resolve().parents[2]/'data'/'ready_data'/'traj_full.csv')
    vocab=build_card_vocab(full_csv) if os.path.exists(full_csv) else build_card_vocab(csv)
    num_cards=len(vocab)+1
    horde=build_v8_horde(num_cards=num_cards,device=dev)
    horde.load(args.checkpoint)
    horde.to(dev)
    horde.backbone.eval()
    for d in horde.td_demons:d.head.eval()
    for d in horde.binary_demons:d.head.eval()
    for d in horde.awr_demons:d.v_head.eval();d.pi_head.eval()
    print(f'Loaded HordeV8: {len(horde.all_demons)} demons, {num_cards} cards')
    eval_v8(horde,csv,vocab,max_battles=args.max_battles)

if __name__=='__main__':
    main()
