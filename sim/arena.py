import math

class Tower:
    ACT=4.0
    def __init__(self,team,ttype,cx,cy,w,h,hp,dmg,spd,rng):
        self.team=team
        self.ttype=ttype
        self.cx=cx
        self.cy=cy
        self.w=w
        self.h=h
        self.hp=hp
        self.max_hp=hp
        self.dmg=dmg
        self.spd=spd
        self.rng=rng
        self.alive=True
        self.statuses=[]
        self.active=ttype!='king'
        self.cd=0
        self.proj_spd=0
        self.troop=None;self.collision_r=1.4 if ttype=='king' else 1.0
    def activate(self):
        if self.active:return False
        self.active=True;self.cd=self.ACT
        return True
    def tiles(self):
        x0=int(self.cx-self.w/2)
        y0=int(self.cy-self.h/2)
        return [(x,y) for y in range(y0,y0+self.h) for x in range(x0,x0+self.w)]
    def dist(self,x,y):
        # combat geometry is the game's collision circle (buildings.csv PrincessTower 1000, KingTower 1400), the tile footprint only blocks walking
        return max(0.0,math.hypot(x-self.cx,y-self.cy)-self.collision_r)
    def take_damage(self,amt):
        if not self.alive:
            return
        if amt>0:self.activate()
        self.hp-=amt
        if self.hp<=0:
            self.hp=0
            self.alive=False

class Arena:
    W,H=18,32
    RIVER=(15,16)
    LANES=(3.5,14.5)
    BRIDGE_HW=1.0
    BRIDGES=(3,14)
    KING_Y=(3.0,29.0)
    PRINCESS_Y=(6.5,25.5)
    FENCE=frozenset([(x,y) for y in (0,31) for x in list(range(0,6))+list(range(12,18))]+[(x,y) for x in (0,17) for y in (14,17)])
    def __init__(self):
        self.grid=[[None for _ in range(self.W)] for _ in range(self.H)]
        self.towers=[]
        self._init_towers()
        self._init_terrain()
    def _init_towers(self):
        for i,team in enumerate(('blue','red')):
            self.towers.append(Tower(team,'king',9.0,self.KING_Y[i],4,4,4824,109,1.0,7.0))
            for lx in self.LANES:
                self.towers.append(Tower(team,'princess',lx,self.PRINCESS_Y[i],3,3,3052,109,0.8,7.5))
    def _init_terrain(self):
        for y in self.RIVER:
            for x in range(self.W):
                self.grid[y][x]='B' if x in self.BRIDGES else 'R'
        for x,y in self.FENCE:
            self.grid[y][x]='F'
        for t in self.towers:
            for tx,ty in t.tiles():
                self.grid[ty][tx]=t.team[0].upper()+'T'
    def blocked(self,x,y,air=False):
        if not(0<=x<self.W and 0<=y<self.H):return True
        c=self.grid[y][x]
        if c is None or c=='B':return False
        if c=='R':return not air
        if c=='F':return True
        return any(t.alive and (x,y) in t.tiles() for t in self.towers)
    def on_bridge(self,x):
        return any(abs(x-lx)<=self.BRIDGE_HW for lx in self.LANES)
    def lane(self,x):
        return 0 if x<self.W/2 else 1
    def get_tower(self,team,ttype,side=None):
        for t in self.towers:
            if t.team==team and t.ttype==ttype:
                if ttype=='king':
                    return t
                if side=='left' and t.cx<9:
                    return t
                if side=='right' and t.cx>9:
                    return t
        return None
