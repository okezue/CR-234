import sys,os,math,random,time
import numpy as np
import torch
import torch.nn.functional as F
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','simulator'))
from game import Game,card_info
from horde.features import extract_features,FEAT_DIM

class CRAgent:
    def __init__(self,planner_ckpt=None,horde_ckpt=None,reacter_ckpt=None,device='cpu'):
        self.dev=torch.device(device)
        self.planner=None
        self.horde=None
        self.reacter=None
        if planner_ckpt:self._load_planner(planner_ckpt)
        if horde_ckpt:self._load_horde(horde_ckpt)
        if reacter_ckpt:self._load_reacter(reacter_ckpt)
    def _load_reacter(self,path):
        d=torch.load(path,map_location='cpu',weights_only=False)
        self.reacter_vocab=d.get('vocab',{})
        self.reacter_inv={v:k for k,v in self.reacter_vocab.items()}
        cfg=d.get('config',{})
        from cql_reacter_v5 import ReacterV5
        nv=cfg.get('nv',len(self.reacter_vocab))
        m=ReacterV5(nv,cfg.get('ed',64),cfg.get('hs',512),cfg.get('nl',8))
        m.load_state_dict(d['state_dict'])
        m.eval().to(self.dev)
        self.reacter=m
        print(f"Reacter loaded: {nv} cards")
    def _load_planner(self,path):
        d=torch.load(path,map_location='cpu',weights_only=False)
        self.planner_vocab=d.get('vocab',{})
        self.planner_inv={v:k for k,v in self.planner_vocab.items()}
        cfg=d.get('config',{})
        from cql_reacter_v5 import ReacterV5
        nv=cfg.get('nv',len(self.planner_vocab))
        m=ReacterV5(nv,cfg.get('ed',64),cfg.get('hs',512),cfg.get('nl',8))
        m.load_state_dict(d['state_dict'])
        m.eval().to(self.dev)
        self.planner=m
        self.planner_nv=nv
        print(f"Planner loaded: {nv} cards")
    def _load_horde(self,path):
        from horde.horde_v8 import build_v8_horde
        import pandas as pd
        csv=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','ready_data','traj_combined_v2.csv')
        if not os.path.exists(csv):
            csv=os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','ready_data','traj.csv')
        df=pd.read_csv(csv,usecols=['card'],low_memory=False)
        cards=sorted(df['card'].dropna().unique().tolist())
        self.horde_vocab={c:i+1 for i,c in enumerate(cards)}
        horde=build_v8_horde(num_cards=len(self.horde_vocab)+1,device=str(self.dev))
        horde.load(path)
        horde.to(self.dev)
        horde.backbone.eval()
        for d in horde.td_demons:d.head.eval()
        for d in horde.binary_demons:d.head.eval()
        self.horde=horde
        print(f"HORDE loaded: {len(horde.all_demons)} demons")
    def evaluate_state(self,game,team):
        if not self.horde:return {}
        feat=extract_features(game,team)
        ft=torch.tensor(feat,dtype=torch.float32,device=self.dev).unsqueeze(0)
        dt=torch.zeros_like(ft)
        ct=torch.tensor([0],dtype=torch.long,device=self.dev)
        preds={}
        with torch.no_grad():
            h=self.horde.backbone(ft,dt,ct)
            for d in self.horde.td_demons:
                if hasattr(d,'head_type') and d.head_type=='placement':
                    preds[d.name]=d.head(h,ct).item()
                else:
                    preds[d.name]=d.head(h).item()
            for d in self.horde.binary_demons:
                preds[d.name]=torch.sigmoid(d.head(h)).item()
        return preds
    def predict_opponent(self,game,team):
        if not self.reacter:return None
        return None
    def select_card(self,game,team,history_seq=None,opp_seq=None,board_state=None):
        if not self.planner and not self.horde:return self._random_play(game,team)
        p=game.players[team]
        if not p.deck or not p.deck.hand:return None
        hand=[c for c in p.deck.hand if p.elixir>=card_info(c)['cost']]
        if not hand:return None
        horde_eval=self.evaluate_state(game,team) if self.horde else {}
        best_card=None;best_score=-999
        for card in hand:
            score=self._score_card(game,team,card,horde_eval)
            if score>best_score:
                best_score=score;best_card=card
        if best_card is None:return None
        x,y=self._pick_placement(game,team,best_card)
        return best_card,x,y
    def _score_card(self,game,team,card,horde_eval=None):
        ci=card_info(card)
        cost=ci['cost']
        score=0
        opp='red' if team=='blue' else 'blue'
        opp_troops=len([u for u in game.players[opp].troops if u.alive])
        my_troops=len([u for u in game.players[team].troops if u.alive])
        if horde_eval:
            will_dmg=horde_eval.get('will_damage',0.5)
            will_lose=horde_eval.get('will_lose_tower',0.5)
            if will_lose>0.6:score+=3
            if will_dmg<0.3 and cost<=3:score+=1
            troop_adv=horde_eval.get('troop_adv',0)
            if troop_adv<-0.3:score+=2
        if opp_troops>my_troops+2:score+=3
        if game.players[team].elixir>=8:score+=2
        if game.t>180:score+=1
        if 'tank' in card or 'giant' in card or 'golem' in card:
            if game.players[team].elixir>=7:score+=2
        if ci.get('deploy_anywhere'):score+=1
        score-=cost*0.3
        return score
    def _pick_placement(self,game,team,card):
        ci=card_info(card)
        if ci.get('deploy_anywhere'):
            if team=='blue':return 9.0,24.0
            else:return 9.0,8.0
        if team=='blue':
            if 'spell' in str(ci.get('type','')).lower():return 9.0,24.0
            return random.choice([4.0,14.0]),random.uniform(5.0,14.0)
        else:
            if 'spell' in str(ci.get('type','')).lower():return 9.0,8.0
            return random.choice([4.0,14.0]),random.uniform(18.0,27.0)
    def _random_play(self,game,team):
        p=game.players[team]
        if not p.deck or not p.deck.hand:return None
        playable=[c for c in p.deck.hand if p.elixir>=card_info(c)['cost']]
        if not playable:return None
        card=random.choice(playable)
        x,y=self._pick_placement(game,team,card)
        return card,x,y

def compute_winning_position(game,team):
    opp='red' if team=='blue' else 'blue'
    score=0.0
    p=game.players[team];op=game.players[opp]
    team_hp=sum(tw.hp for tw in game.arena.towers if tw.team==team and tw.alive)
    opp_hp=sum(tw.hp for tw in game.arena.towers if tw.team==opp and tw.alive)
    team_max=sum(tw.max_hp for tw in game.arena.towers if tw.team==team)
    opp_max=sum(tw.max_hp for tw in game.arena.towers if tw.team==opp)
    hp_ratio=(team_hp/max(team_max,1))-(opp_hp/max(opp_max,1))
    score+=hp_ratio*5.0
    score+=(p.crowns-op.crowns)*10.0
    my_dps=sum(getattr(u,'dmg',0)/max(getattr(u,'atk_spd',1),0.1) for u in p.troops if u.alive)
    opp_dps=sum(getattr(u,'dmg',0)/max(getattr(u,'atk_spd',1),0.1) for u in op.troops if u.alive)
    my_hp_total=sum(u.hp for u in p.troops if u.alive)
    opp_hp_total=sum(u.hp for u in op.troops if u.alive)
    score+=(my_dps-opp_dps)*0.01
    score+=(my_hp_total-opp_hp_total)*0.001
    bridge_y_blue=16.0;bridge_y_red=16.0
    my_push=sum(1 for u in p.troops if u.alive and ((team=='blue' and u.y>14) or (team=='red' and u.y<18)))
    opp_push=sum(1 for u in op.troops if u.alive and ((opp=='blue' and u.y>14) or (opp=='red' and u.y<18)))
    score+=(my_push-opp_push)*0.5
    score+=(p.elixir-op.elixir)*0.2
    return score

def compute_reward(prev_state,curr_state,team):
    r=0.0
    opp_dmg_dealt=(prev_state['opp_tower_hp']-curr_state['opp_tower_hp'])
    team_dmg_taken=(prev_state['team_tower_hp']-curr_state['team_tower_hp'])
    r+=opp_dmg_dealt/500.0
    r-=team_dmg_taken/500.0
    crown_gained=curr_state['team_crowns']-prev_state['team_crowns']
    crown_lost=curr_state['opp_crowns']-prev_state['opp_crowns']
    r+=crown_gained*5.0
    r-=crown_lost*5.0
    r+=(curr_state['winning_score']-prev_state.get('winning_score',0))*0.1
    if curr_state.get('won'):r+=20.0
    if curr_state.get('lost'):r-=20.0
    elixir_efficiency=opp_dmg_dealt/max(prev_state.get('elixir_spent',1),1)
    r+=min(elixir_efficiency*0.01,1.0)
    return r

def get_state_info(game,team):
    opp='red' if team=='blue' else 'blue'
    return {
        'opp_tower_hp':sum(tw.hp for tw in game.arena.towers if tw.team==opp and tw.alive),
        'team_tower_hp':sum(tw.hp for tw in game.arena.towers if tw.team==team and tw.alive),
        'team_crowns':game.players[team].crowns,
        'opp_crowns':game.players[opp].crowns,
        'team_elixir':game.players[team].elixir,
        'opp_elixir':game.players[opp].elixir,
        'team_troops':len([u for u in game.players[team].troops if u.alive]),
        'opp_troops':len([u for u in game.players[opp].troops if u.alive]),
        'time':game.t,
        'won':game.ended and game.winner==team,
        'lost':game.ended and game.winner==opp,
        'winning_score':compute_winning_position(game,team),
        'elixir_spent':0,
    }

def get_state_info(game,team):
    opp='red' if team=='blue' else 'blue'
    return {
        'opp_tower_hp':sum(tw.hp for tw in game.arena.towers if tw.team==opp and tw.alive),
        'team_tower_hp':sum(tw.hp for tw in game.arena.towers if tw.team==team and tw.alive),
        'team_crowns':game.players[team].crowns,
        'opp_crowns':game.players[opp].crowns,
        'team_elixir':game.players[team].elixir,
        'opp_elixir':game.players[opp].elixir,
        'team_troops':len([u for u in game.players[team].troops if u.alive]),
        'opp_troops':len([u for u in game.players[opp].troops if u.alive]),
        'time':game.t,
        'won':game.ended and game.winner==team,
        'lost':game.ended and game.winner==opp,
    }

def play_vs_replay(agent,traj_csv,team='blue',max_battles=100,verbose=False):
    import pandas as pd
    opp='red' if team=='blue' else 'blue'
    df=pd.read_csv(traj_csv,low_memory=False)
    df=df.sort_values(['battle_id','time'])
    battles=list(df.groupby('battle_id',sort=False))[:max_battles]
    results={'wins':0,'losses':0,'draws':0,'total_reward':0,'games':0}
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
        opp_plays=grp[grp['side'].astype(str).str.strip()=='o'].reset_index(drop=True)
        opp_idx=0
        prev_info=get_state_info(g,team)
        ep_reward=0;agent_plays=0
        last_agent_play=0
        while not g.ended and g.t<300:
            while opp_idx<len(opp_plays):
                row=opp_plays.iloc[opp_idx]
                target_t=float(row['time'])/6000.0*300.0
                if target_t>g.t+0.5:break
                card=str(row['card'])
                try:
                    rx=float(row.get('x',9000))/1000.0
                    ry=float(row.get('y',16000))/1000.0
                except:rx,ry=9.0,16.0
                if np.isnan(rx) or np.isnan(ry):rx,ry=9.0,16.0
                g.players[opp].elixir=max(g.players[opp].elixir,card_info(card)['cost'])
                try:g.play_card(opp,card,rx,ry)
                except:pass
                opp_idx+=1
            if g.t-last_agent_play>=2.0 and g.players[team].elixir>=4:
                action=agent.select_card(g,team)
                if action:
                    card,x,y=action
                    try:
                        g.play_card(team,card,x,y)
                        agent_plays+=1
                        last_agent_play=g.t
                    except:pass
            for _ in range(5):
                if g.ended:break
                g.tick()
            curr_info=get_state_info(g,team)
            r=compute_reward(prev_info,curr_info,team)
            ep_reward+=r
            prev_info=curr_info
        if g.winner==team:results['wins']+=1
        elif g.winner==opp:results['losses']+=1
        else:results['draws']+=1
        results['total_reward']+=ep_reward
        results['games']+=1
        if verbose and (ep+1)%10==0:
            wr=results['wins']/max(results['games'],1)*100
            print(f"[{ep+1}/{len(battles)}] W:{results['wins']} L:{results['losses']} D:{results['draws']} WR:{wr:.1f}% AvgR:{results['total_reward']/results['games']:.2f} Plays:{agent_plays}")
    wr=results['wins']/max(results['games'],1)*100
    print(f"\nFinal: {results['games']} games, W:{results['wins']} L:{results['losses']} D:{results['draws']} WR:{wr:.1f}%")
    print(f"Avg reward: {results['total_reward']/max(results['games'],1):.2f}")
    return results

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser()
    ap.add_argument('--planner',default='checkpoints/planner_v5_best.pt')
    ap.add_argument('--horde',default='checkpoints/horde_v8.pt')
    ap.add_argument('--reacter',default='checkpoints/reacter_v6_best.pt')
    ap.add_argument('--traj-csv',default=None)
    ap.add_argument('--max-battles',type=int,default=100)
    args=ap.parse_args()
    csv=args.traj_csv or str(os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','data','ready_data','traj.csv'))
    agent=CRAgent(planner_ckpt=args.planner,horde_ckpt=args.horde,reacter_ckpt=args.reacter)
    play_vs_replay(agent,csv,max_battles=args.max_battles,verbose=True)
