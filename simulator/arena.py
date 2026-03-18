import math

class Tower:
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
    def tiles(self):
        x0=math.ceil(self.cx-self.w/2)
        y0=math.ceil(self.cy-self.h/2)
        return [(x,y) for y in range(y0,y0+self.h) for x in range(x0,x0+self.w)]
    def center_tiles(self):
        x0=math.ceil(self.cx-self.w/2)
        y0=math.ceil(self.cy-self.h/2)
        cxs=[x0+self.w//2-1,x0+self.w//2] if self.w%2==0 else [x0+self.w//2]
        cys=[y0+self.h//2-1,y0+self.h//2] if self.h%2==0 else [y0+self.h//2]
        return [(x,y) for y in cys for x in cxs]
    def in_range(self,tx,ty):
        dx=tx-self.cx
        dy=ty-self.cy
        return math.sqrt(dx*dx+dy*dy)<=self.rng
    def take_damage(self,amt):
        if not self.alive:
            return
        self.hp-=amt
        if self.hp<=0:
            self.hp=0
            self.alive=False
        if hasattr(self,'_dmg_log'):
            import traceback
            stk=traceback.extract_stack()
            fr=stk[-2] if len(stk)>=2 else stk[-1]
            an=getattr(self,'_atk_name','')
            self._dmg_log.append((amt,f"{fr.filename.split('/')[-1]}:{fr.lineno}",getattr(self,'_gt',0),an))
            self._atk_name=''

class Arena:
    W,H=18,32
    COLS=[chr(65+i) for i in range(18)]
    RIVER_Y=16
    BRIDGE_L=range(3,6)
    BRIDGE_R=range(12,15)
    def __init__(self):
        self.grid=[[None for _ in range(self.W)] for _ in range(self.H)]
        self.towers=[]
        self._init_towers()
        self._init_terrain()
    def _init_towers(self):
        ts=[
            ("blue","king",8.5,2.5,4,4,4824,109,1.0,7.0),
            ("blue","princess",3.0,6.5,3,3,3052,109,0.8,7.5),
            ("blue","princess",14.0,6.5,3,3,3052,109,0.8,7.5),
            ("red","king",8.5,28.5,4,4,4824,109,1.0,7.0),
            ("red","princess",3.0,24.5,3,3,3052,109,0.8,7.5),
            ("red","princess",14.0,24.5,3,3,3052,109,0.8,7.5),
        ]
        for team,ttype,cx,cy,w,h,hp,dmg,spd,rng in ts:
            t=Tower(team,ttype,cx,cy,w,h,hp,dmg,spd,rng)
            self.towers.append(t)
    def _init_terrain(self):
        for x in range(self.W):
            self.grid[self.RIVER_Y][x]='R'
            self.grid[self.RIVER_Y-1][x]='R'
        for x in self.BRIDGE_L:
            self.grid[self.RIVER_Y][x]='B'
            self.grid[self.RIVER_Y-1][x]='B'
        for x in self.BRIDGE_R:
            self.grid[self.RIVER_Y][x]='B'
            self.grid[self.RIVER_Y-1][x]='B'
        for t in self.towers:
            pfx=t.team[0].upper()
            for tx,ty in t.tiles():
                if 0<=tx<self.W and 0<=ty<self.H:
                    self.grid[ty][tx]=f"{pfx}T"
            tag='K' if t.ttype=='king' else 'P'
            for tx,ty in t.center_tiles():
                if 0<=tx<self.W and 0<=ty<self.H:
                    self.grid[ty][tx]=f"{pfx}{tag}"
    def tile_at(self,x,y):
        if 0<=x<self.W and 0<=y<self.H:
            return self.grid[y][x]
        return None
    def is_walkable(self,x,y):
        if x<0 or x>=self.W or y<0 or y>=self.H:
            return False
        c=self.grid[y][x]
        if c=='R':
            return False
        if c and len(c)==2 and c[1] in ('K','P','T'):
            return False
        return True
    def is_bridge(self,x,y):
        return self.grid[y][x]=='B' if 0<=x<self.W and 0<=y<self.H else False
    def deploy_zone(self,team):
        zones=[]
        if team=='blue':
            for y in range(0,16):
                for x in range(self.W):
                    if self.is_walkable(x,y):
                        zones.append((x,y))
        else:
            for y in range(17,self.H):
                for x in range(self.W):
                    if self.is_walkable(x,y):
                        zones.append((x,y))
        return zones
    def can_deploy(self,team,x,y):
        if not self.is_walkable(x,y):
            return False
        if team=='blue':
            return y<16
        return y>16
    @staticmethod
    def replay_to_tile(rx,ry):
        return rx/1000.0,ry/1000.0
    @staticmethod
    def tile_to_col_row(tx,ty):
        c=chr(65+min(17,max(0,int(tx))))
        r=int(ty)+1
        return c,r
    @staticmethod
    def distance(x1,y1,x2,y2):
        dx=x2-x1
        dy=y2-y1
        return math.sqrt(dx*dx+dy*dy)
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
    def render_ascii(self):
        hdr="   "+" ".join(f"{c}" for c in self.COLS)
        lines=[hdr]
        for y in range(self.H-1,-1,-1):
            row_num=y+1
            lbl=f"{row_num:2d} "
            cells=[]
            for x in range(self.W):
                c=self.grid[y][x]
                if c is None:
                    cells.append('.')
                elif c=='R':
                    cells.append('~')
                elif c=='B':
                    cells.append('=')
                elif c in ('BT','RT'):
                    cells.append('#')
                elif c=='BK':
                    cells.append('K')
                elif c=='BP':
                    cells.append('P')
                elif c=='RK':
                    cells.append('k')
                elif c=='RP':
                    cells.append('p')
                else:
                    cells.append('?')
            line=lbl+" ".join(cells)+f"  {row_num:2d}"
            lines.append(line)
        lines.append(hdr)
        return "\n".join(lines)

if __name__=="__main__":
    a=Arena()
    print(a.render_ascii())
    print()
    rx,ry=9000,16000
    tx,ty=Arena.replay_to_tile(rx,ry)
    c,r=Arena.tile_to_col_row(tx,ty)
    print(f"Replay ({rx},{ry}) -> tile ({tx},{ty}) -> {c}{r}")
    print(f"Tile at (3,16): {a.tile_at(3,16)}")
    print(f"Walkable (3,16): {a.is_walkable(3,16)}")
    print(f"Walkable (0,16): {a.is_walkable(0,16)}")
    bk=a.get_tower('blue','king')
    print(f"Blue king HP: {bk.hp}, range: {bk.rng}, in_range(9,10): {bk.in_range(9,10)}")
    bp=a.get_tower('blue','princess','left')
    print(f"Blue left princess center: ({bp.cx},{bp.cy}), tiles: {bp.tiles()}")
