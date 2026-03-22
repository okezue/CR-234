import sys,os,math
import numpy as np
sys.path.insert(0,os.path.join(os.path.dirname(os.path.abspath(__file__)),'..','..','simulator'))

MAX_UNITS=24
UNIT_FEAT=12
N_TOWERS=6
TOWER_FEAT=6
GAME_FEAT=12
HAND_FEAT=20
GRID_H,GRID_W=8,5
GRID_CHANNELS=4

FLAT_DIM=GAME_FEAT+N_TOWERS*TOWER_FEAT+HAND_FEAT
UNIT_DIM=MAX_UNITS*2*UNIT_FEAT
GRID_DIM=GRID_H*GRID_W*GRID_CHANNELS
FEAT_DIM=FLAT_DIM+UNIT_DIM+GRID_DIM

def _unit_type_id(u):
    name=getattr(u,'name','').lower()
    types={'knight':1,'archers':2,'giant':3,'minions':4,'musketeer':5,'hog_rider':6,
           'valkyrie':7,'goblin':8,'skeleton':9,'wizard':10,'witch':11,'pekka':12,
           'dragon':13,'prince':14,'balloon':15,'golem':16,'lava':17,'sparky':18,
           'miner':19,'princess':20,'bandit':21,'mega_knight':22,'ram':23,'barrel':24}
    for k,v in types.items():
        if k in name:return v/30.0
    return 0.0

def _dist_to_nearest_tower(u,towers,enemy_team):
    min_d=999.0
    for tw in towers:
        if tw.team==enemy_team and tw.alive:
            tx=getattr(tw,'cx',getattr(tw,'x',9))
            ty=getattr(tw,'cy',getattr(tw,'y',16))
            d=math.sqrt((u.x-tx)**2+(u.y-ty)**2)
            if d<min_d:min_d=d
    return min(min_d/40.0,1.0)

def extract_features_v4(game,team="blue"):
    g=game
    opp="red" if team=="blue" else "blue"
    flat=np.zeros(FLAT_DIM,dtype=np.float32)
    units=np.zeros(UNIT_DIM,dtype=np.float32)
    grid=np.zeros((GRID_H,GRID_W,GRID_CHANNELS),dtype=np.float32)
    idx=0
    flat[idx]=g.t/300.0;idx+=1
    flat[idx]=min(g.t/120.0,1.0);idx+=1
    flat[idx]=float(g.t>120);idx+=1
    flat[idx]=float(g.t>180);idx+=1
    ph={"regulation":0,"overtime":1,"end":2}.get(g.phase,0)
    flat[idx+ph]=1.0;idx+=3
    flat[idx]=g._erate()/3.0;idx+=1
    flat[idx]=g.players[team].elixir/10.0;idx+=1
    flat[idx]=g.players[opp].elixir/10.0;idx+=1
    flat[idx]=g.players[team].crowns/3.0;idx+=1
    flat[idx]=g.players[opp].crowns/3.0;idx+=1
    for tw in g.arena.towers:
        bi=idx
        flat[bi]=float(tw.alive)
        flat[bi+1]=float(getattr(tw,'active',True))
        flat[bi+2]=(tw.hp/tw.max_hp) if tw.max_hp>0 else 0
        flat[bi+3]=float(tw.team==team)
        flat[bi+4]=getattr(tw,'cx',getattr(tw,'x',9))/18.0
        flat[bi+5]=getattr(tw,'cy',getattr(tw,'y',16))/32.0
        idx+=TOWER_FEAT
    p=g.players[team]
    hi=idx
    if p.deck and p.deck.hand:
        from game import card_info
        for i,c in enumerate(p.deck.hand[:4]):
            ci=card_info(c)
            flat[hi+i*4]=ci['cost']/10.0
            flat[hi+i*4+1]=float(p.elixir>=ci['cost'])
            flat[hi+i*4+2]=float('spell' in str(ci.get('type','')).lower())
            flat[hi+i*4+3]=float('building' in str(ci.get('type','')).lower())
        flat[hi+16]=len(p.deck.hand)/4.0
        flat[hi+17]=p.deck.nxt_cd/2.0 if p.deck.nxt_cd else 0
        flat[hi+18]=float(p.elixir>=min(card_info(c)['cost'] for c in p.deck.hand))
        flat[hi+19]=sum(card_info(c)['cost'] for c in p.deck.hand)/40.0
    ui=0
    for tm_idx,tm in enumerate([team,opp]):
        alive=[u for u in g.players[tm].troops if u.alive][:MAX_UNITS]
        for u in alive:
            bi=ui*UNIT_FEAT
            units[bi]=u.x/18.0
            units[bi+1]=u.y/32.0
            units[bi+2]=u.hp/max(u.max_hp,1)
            units[bi+3]=u.spd/6.0
            units[bi+4]=float(getattr(u,'transport','Ground')=='Air')
            units[bi+5]=float(getattr(u,'is_building',False))
            units[bi+6]=u.dmg/500.0
            units[bi+7]=_unit_type_id(u)
            units[bi+8]=float(tm==team)
            units[bi+9]=_dist_to_nearest_tower(u,g.arena.towers,opp if tm==team else team)
            units[bi+10]=getattr(u,'atk_cd',0)/2.0
            units[bi+11]=float(getattr(u,'targets_buildings',False))
            gx=int(min(max(u.x/18.0*(GRID_W-1),0),GRID_W-1))
            gy=int(min(max(u.y/32.0*(GRID_H-1),0),GRID_H-1))
            ch=0 if tm==team else 1
            ch2=2 if tm==team else 3
            grid[gy,gx,ch]+=1.0
            grid[gy,gx,ch2]+=u.hp/max(u.max_hp,1)
            ui+=1
    grid=grid/np.maximum(grid.max(),1.0)
    return np.concatenate([flat,units,grid.flatten()])

def extract_unit_list(game,team="blue"):
    g=game
    opp="red" if team=="blue" else "blue"
    result=[]
    for tm_idx,tm in enumerate([team,opp]):
        alive=[u for u in g.players[tm].troops if u.alive][:MAX_UNITS]
        for u in alive:
            feat=np.zeros(UNIT_FEAT,dtype=np.float32)
            feat[0]=u.x/18.0
            feat[1]=u.y/32.0
            feat[2]=u.hp/max(u.max_hp,1)
            feat[3]=u.spd/6.0
            feat[4]=float(getattr(u,'transport','Ground')=='Air')
            feat[5]=float(getattr(u,'is_building',False))
            feat[6]=u.dmg/500.0
            feat[7]=_unit_type_id(u)
            feat[8]=float(tm==team)
            feat[9]=_dist_to_nearest_tower(u,g.arena.towers,opp if tm==team else team)
            feat[10]=getattr(u,'atk_cd',0)/2.0
            feat[11]=float(getattr(u,'targets_buildings',False))
            result.append(feat)
    while len(result)<MAX_UNITS*2:
        result.append(np.zeros(UNIT_FEAT,dtype=np.float32))
    return np.stack(result[:MAX_UNITS*2])

def extract_grid(game,team="blue"):
    g=game
    opp="red" if team=="blue" else "blue"
    grid=np.zeros((GRID_H,GRID_W,GRID_CHANNELS),dtype=np.float32)
    for tm_idx,tm in enumerate([team,opp]):
        alive=[u for u in g.players[tm].troops if u.alive]
        for u in alive:
            gx=int(min(max(u.x/18.0*(GRID_W-1),0),GRID_W-1))
            gy=int(min(max(u.y/32.0*(GRID_H-1),0),GRID_H-1))
            ch=0 if tm==team else 1
            ch2=2 if tm==team else 3
            grid[gy,gx,ch]+=1.0
            grid[gy,gx,ch2]+=u.hp/max(u.max_hp,1)
    return grid/np.maximum(grid.max(),1.0)
