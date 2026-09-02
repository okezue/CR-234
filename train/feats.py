import math
import numpy as np
from sim.game import card_info
from sim.cards import card

MAX_UNITS=24;UNIT_FEAT=12;N_TOWERS=6;TOWER_FEAT=6;GAME_FEAT=12;HAND_FEAT=20
GRID_H,GRID_W,GRID_CH=8,5,4
FLAT_DIM=GAME_FEAT+N_TOWERS*TOWER_FEAT+HAND_FEAT
UNIT_DIM=MAX_UNITS*2*UNIT_FEAT
GRID_DIM=GRID_H*GRID_W*GRID_CH
FEAT_DIM=FLAT_DIM+UNIT_DIM+GRID_DIM
_TYPES={'knight':1,'archers':2,'giant':3,'minions':4,'musketeer':5,'hog_rider':6,'valkyrie':7,'goblin':8,'skeleton':9,'wizard':10,'witch':11,'pekka':12,
        'dragon':13,'prince':14,'balloon':15,'golem':16,'lava':17,'sparky':18,'miner':19,'princess':20,'bandit':21,'mega_knight':22,'ram':23,'barrel':24}

def card_type(name):
    try:return card(name)['kind']
    except KeyError:return ''

def _type_id(u):
    n=getattr(u,'name','').lower().replace(' ','_')
    return next((v/30.0 for k,v in _TYPES.items() if k in n),0.0)

def _tower_dist(u,towers,enemy):
    d=[math.hypot(u.x-t.cx,u.y-t.cy) for t in towers if t.team==enemy and t.alive]
    return min(min(d)/40.0,1.0) if d else 1.0

def _unit_feat(u,towers,team,opp,mine):
    return [u.x/18.0,u.y/32.0,u.hp/max(u.max_hp,1),u.spd/6.0,float(getattr(u,'transport','Ground')=='Air'),float(getattr(u,'is_building',False)),
            u.dmg/500.0,_type_id(u),float(mine),_tower_dist(u,towers,opp if mine else team),u.cd/2.0,float(u.targets==['Buildings'])]

def featurize(g,team='blue'):
    opp='red' if team=='blue' else 'blue'
    flat=np.zeros(FLAT_DIM,np.float32);units=np.zeros(UNIT_DIM,np.float32);grid=np.zeros((GRID_H,GRID_W,GRID_CH),np.float32)
    p=g.players[team];o=g.players[opp]
    ph={'regulation':0,'overtime':1,'end':2}.get(g.phase,0)
    flat[:12]=[g.t/300.0,min(g.t/120.0,1.0),float(g.t>120),float(g.t>180),ph==0,ph==1,ph==2,g._erate()/3.0,p.elixir/10.0,o.elixir/10.0,p.crowns/3.0,o.crowns/3.0]
    i=GAME_FEAT
    for tw in g.arena.towers:
        flat[i:i+TOWER_FEAT]=[float(tw.alive),float(getattr(tw,'active',True)),tw.hp/max(tw.max_hp,1),float(tw.team==team),tw.cx/18.0,tw.cy/32.0]
        i+=TOWER_FEAT
    if p.deck and p.deck.hand:
        costs=[card_info(c)['cost'] for c in p.deck.hand]
        for j,c in enumerate(p.deck.hand[:4]):
            ct=card_type(c);flat[i+j*4:i+j*4+4]=[costs[j]/10.0,float(p.elixir>=costs[j]),float('spell' in ct),float('building' in ct)]
        flat[i+16:i+20]=[len(p.deck.hand)/4.0,p.deck.nxt_cd/2.0,float(p.elixir>=min(costs)),sum(costs)/40.0]
    ui=0
    for tm in (team,opp):
        for u in [u for u in g.players[tm].troops if u.alive][:MAX_UNITS]:
            units[ui*UNIT_FEAT:(ui+1)*UNIT_FEAT]=_unit_feat(u,g.arena.towers,team,opp,tm==team);ui+=1
            gx=int(min(max(u.x/18.0*(GRID_W-1),0),GRID_W-1));gy=int(min(max(u.y/32.0*(GRID_H-1),0),GRID_H-1))
            c=0 if tm==team else 1
            grid[gy,gx,c]+=1.0;grid[gy,gx,c+2]+=u.hp/max(u.max_hp,1)
    grid/=max(grid.max(),1.0)
    return np.concatenate([flat,units,grid.ravel()])
