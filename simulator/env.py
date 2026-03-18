import sys,os,math
import numpy as np
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:
    import gym
    from gym import spaces
from game import Game,card_info

MAX_TROOPS=30
TROOP_FEAT=6
TOWER_FEAT=3
N_TOWERS=6
GAME_FEAT=5
PLAYER_FEAT=4
OBS_DIM=GAME_FEAT+2*PLAYER_FEAT+N_TOWERS*TOWER_FEAT+2*MAX_TROOPS*TROOP_FEAT

class CREnv(gym.Env):
    metadata={"render_modes":["human"]}
    def __init__(self,blue_deck=None,red_deck=None,blue_levels=None,red_levels=None,
                 decision_freq=10,reward_mode="sparse",render_mode=None):
        super().__init__()
        self.blue_deck=blue_deck or ["knight","archers","fireball","giant","musketeer","valkyrie","bomber","arrows"]
        self.red_deck=red_deck or ["knight","archers","fireball","giant","musketeer","valkyrie","bomber","arrows"]
        self.blue_levels=blue_levels or {}
        self.red_levels=red_levels or {}
        self.decision_freq=decision_freq
        self.reward_mode=reward_mode
        self.render_mode=render_mode
        self.observation_space=spaces.Box(low=-1,high=1,shape=(OBS_DIM,),dtype=np.float32)
        self.action_space=spaces.Dict({
            "card":spaces.Discrete(5),
            "x":spaces.Box(low=0,high=17,shape=(1,),dtype=np.float32),
            "y":spaces.Box(low=0,high=31,shape=(1,),dtype=np.float32),
        })
        self.game=None
        self._prev_tower_hp=None
    def _get_obs(self):
        g=self.game
        obs=np.zeros(OBS_DIM,dtype=np.float32)
        idx=0
        obs[idx]=g.t/300.0;idx+=1
        ph={"regulation":0,"overtime":1,"end":2}.get(g.phase,0)
        obs[idx+ph]=1.0;idx+=3
        obs[idx]=g._erate()/3.0;idx+=1
        for tm in ("blue","red"):
            p=g.players[tm]
            obs[idx]=p.elixir/10.0;idx+=1
            obs[idx]=p.crowns/3.0;idx+=1
            obs[idx]=len(p.deck.hand)/4.0 if p.deck else 0;idx+=1
            obs[idx]=(p.deck.nxt_cd/2.0) if p.deck else 0;idx+=1
        for tw in g.arena.towers:
            obs[idx]=float(tw.alive);idx+=1
            obs[idx]=float(getattr(tw,'active',True));idx+=1
            obs[idx]=(tw.hp/tw.max_hp) if tw.max_hp>0 else 0;idx+=1
        for tm in ("blue","red"):
            troops=g.players[tm].troops
            alive=[u for u in troops if u.alive][:MAX_TROOPS]
            for i,u in enumerate(alive):
                bi=idx+i*TROOP_FEAT
                obs[bi]=u.x/18.0
                obs[bi+1]=u.y/32.0
                obs[bi+2]=u.hp/max(u.max_hp,1)
                obs[bi+3]=u.spd/6.0
                obs[bi+4]=float(getattr(u,'transport','Ground')=='Air')
                obs[bi+5]=float(getattr(u,'is_building',False))
            idx+=MAX_TROOPS*TROOP_FEAT
        return obs
    def _tower_hp_sum(self,team):
        return sum(tw.hp for tw in self.game.arena.towers if tw.team==team and tw.alive)
    def reset(self,seed=None,options=None):
        super().reset(seed=seed)
        self.game=Game(
            p1={"deck":self.blue_deck,"card_levels":self.blue_levels},
            p2={"deck":self.red_deck,"card_levels":self.red_levels},
        )
        self._prev_tower_hp={"blue":self._tower_hp_sum("blue"),"red":self._tower_hp_sum("red")}
        return self._get_obs(),{}
    def step(self,action):
        g=self.game
        if isinstance(action,dict):
            card_idx=int(action["card"])
            ax=float(action["x"]) if np.isscalar(action["x"]) else float(action["x"][0])
            ay=float(action["y"]) if np.isscalar(action["y"]) else float(action["y"][0])
        elif isinstance(action,(int,np.integer)):
            card_idx=int(action);ax=9.0;ay=24.0
        else:
            card_idx=4;ax=9.0;ay=24.0
        if card_idx<4:
            p=g.players["blue"]
            if p.deck and card_idx<len(p.deck.hand):
                card=p.deck.hand[card_idx]
                g.play_card("blue",card,ax,ay)
        for _ in range(self.decision_freq):
            if g.ended:break
            g._gen_ex()
            g._proc_pending()
            g._proc_pending_ab()
            g._proc_towers()
            g._proc_troops()
            for sp in g.spells:sp.tick(g.DT,g)
            g.spells=[s for s in g.spells if s.active]
            for tm in ("blue","red"):
                if g.players[tm].deck:g.players[tm].deck.tick(g.DT,g._qcd())
                dead=[u for u in g.players[tm].troops if not u.alive]
                for d in dead:
                    d.on_death(g)
                    g.players[tm]._on_champ_death(d)
                g.players[tm].troops=[u for u in g.players[tm].troops if u.alive]
            g._check_phase()
            g.t+=g.DT
            g.replay.snap(g)
        new_hp={"blue":self._tower_hp_sum("blue"),"red":self._tower_hp_sum("red")}
        if self.reward_mode=="dense":
            dmg_dealt=self._prev_tower_hp["red"]-new_hp["red"]
            dmg_taken=self._prev_tower_hp["blue"]-new_hp["blue"]
            reward=(dmg_dealt-dmg_taken)/5000.0
        else:
            if g.ended:
                bc=g.players["blue"].crowns;rc=g.players["red"].crowns
                reward=float(bc-rc)
            else:
                reward=0.0
        self._prev_tower_hp=new_hp
        terminated=g.ended
        truncated=g.t>=g.OT+10
        info={"time":g.t,"phase":g.phase,"blue_crowns":g.players["blue"].crowns,
              "red_crowns":g.players["red"].crowns,"winner":g.winner}
        return self._get_obs(),reward,terminated,truncated,info
    def render(self):
        if self.game:
            print(self.game.replay.dump(self.game.t))

if __name__=="__main__":
    env=CREnv()
    obs,info=env.reset()
    print(f"Obs shape: {obs.shape}, obs range: [{obs.min():.3f}, {obs.max():.3f}]")
    for i in range(50):
        action={"card":env.action_space["card"].sample(),
                "x":np.array([float(np.random.randint(0,18))]),
                "y":np.array([float(np.random.randint(0,32))])}
        obs,reward,terminated,truncated,info=env.step(action)
        if i%10==0:
            print(f"Step {i}: t={info['time']:.1f} phase={info['phase']} blue={info['blue_crowns']} red={info['red_crowns']} r={reward:.3f}")
        if terminated or truncated:
            print(f"Game ended: winner={info['winner']}")
            break
    print("Env OK.")
