import math
import heapq
from sim.units import has
_SQRT2=math.sqrt(2)
_DIRS=[(1,0,1.0),(-1,0,1.0),(0,1,1.0),(0,-1,1.0),
       (1,1,_SQRT2),(-1,1,_SQRT2),(1,-1,_SQRT2),(-1,-1,_SQRT2)]
class Pathfinder:
    def __init__(self,arena):
        self.W=arena.W;self.H=arena.H
        self.arena=arena
        self._gnd=self._build_grid(False)
        self._air=self._build_grid(True)
        self._cache={}
    def _build_grid(self,air):
        return [[not self.arena.blocked(x,y,air) for x in range(self.W)] for y in range(self.H)]
    def rebuild_tower_grid(self):
        self._gnd=self._build_grid(False)
        self._air=self._build_grid(True)
        self._cache.clear()
    def seg_blocked(self,x0,y0,x1,y1,air=False):
        grid=self._air if air else self._gnd
        n=max(1,int(math.hypot(x1-x0,y1-y0)*4))
        for i in range(1,n+1):
            t=i/n;x=int(x0+(x1-x0)*t);y=int(y0+(y1-y0)*t)
            if 0<=x<self.W and 0<=y<self.H and not grid[y][x]:return True
        return False
    def _octile(self,x1,y1,x2,y2):
        dx=abs(x2-x1);dy=abs(y2-y1)
        return max(dx,dy)+(_SQRT2-1)*min(dx,dy)
    def a_star(self,sx,sy,gx,gy,air=False):
        grid=self._air if air else self._gnd
        if sx<0 or sx>=self.W or sy<0 or sy>=self.H:return []
        if gx<0 or gx>=self.W or gy<0 or gy>=self.H:return []
        if not grid[sy][sx]:
            found=False
            for r in range(1,5):
                for ddx in range(-r,r+1):
                    for ddy in range(-r,r+1):
                        if abs(ddx)<r and abs(ddy)<r:continue
                        nx,ny=sx+ddx,sy+ddy
                        if 0<=nx<self.W and 0<=ny<self.H and grid[ny][nx]:
                            sx,sy=nx,ny;found=True;break
                    if found:break
                if found:break
            if not found:return []
        if not grid[gy][gx]:
            best=None;bd=999
            for r in range(1,5):
                for ddx in range(-r,r+1):
                    for ddy in range(-r,r+1):
                        if abs(ddx)<r and abs(ddy)<r:continue
                        nx,ny=gx+ddx,gy+ddy
                        if 0<=nx<self.W and 0<=ny<self.H and grid[ny][nx]:
                            d=abs(nx-sx)+abs(ny-sy)
                            if d<bd:bd=d;best=(nx,ny)
                if best:break
            if best:gx,gy=best
            else:return []
        if sx==gx and sy==gy:return [(gx+0.5,gy+0.5)]
        open_h=[(self._octile(sx,sy,gx,gy),0.0,sx,sy)]
        g_sc={};g_sc[(sx,sy)]=0.0
        came={}
        while open_h:
            _,gc,cx,cy=heapq.heappop(open_h)
            if cx==gx and cy==gy:
                path=[];n=(gx,gy)
                while n in came:path.append(n);n=came[n]
                path.reverse()
                return [(px+0.5,py+0.5) for px,py in path]
            if gc>g_sc.get((cx,cy),1e9):continue
            for ddx,ddy,cost in _DIRS:
                nx,ny=cx+ddx,cy+ddy
                if nx<0 or nx>=self.W or ny<0 or ny>=self.H:continue
                if not grid[ny][nx]:continue
                if ddx!=0 and ddy!=0:
                    if not grid[cy][cx+ddx] or not grid[cy+ddy][cx]:continue
                ng=gc+cost
                if ng<g_sc.get((nx,ny),1e9):
                    g_sc[(nx,ny)]=ng
                    came[(nx,ny)]=(cx,cy)
                    f=ng+self._octile(nx,ny,gx,gy)
                    heapq.heappush(open_h,(f,ng,nx,ny))
        return []
    def get_path(self,tr,tx,ty,air=None):
        sx=max(0,min(self.W-1,int(tr.x)))
        sy=max(0,min(self.H-1,int(tr.y)))
        gx=max(0,min(self.W-1,int(tx)))
        gy=max(0,min(self.H-1,int(ty)))
        if air is None:
            air=getattr(tr,'transport','Ground')=='Air'
            if not air:
                from sim.fx import RiverJump
                if any(isinstance(c,RiverJump) for c in getattr(tr,'components',[])):air=True
        key=(sx,sy,gx,gy,air)
        if key in self._cache:return list(self._cache[key])
        p=self.a_star(sx,sy,gx,gy,air)
        self._cache[key]=p
        return list(p)
    def _shift(self,u,dx,dy):
        a=self.arena;nx=min(max(u.x+dx,0.3),self.W-0.3);ny=min(max(u.y+dy,0.3),self.H-0.3)
        if getattr(u,'transport','Ground')!='Air' and (a.blocked(int(nx),int(ny),True) or (int(ny) in a.RIVER and not a.on_bridge(nx))):return
        u.x=nx;u.y=ny
    def resolve_collisions(self,troops,dt=0.1):
        # bodies of either team separate within their layer (ground or air); the overlap is split in inverse proportion to mass, buildings never move
        cs=2.0;key=lambda u:(u.id,type(u).__name__)
        alive=[tr for tr in troops if tr.alive and not has(tr,'burrowed')]
        buckets={}
        for tr in alive:buckets.setdefault((int(tr.x/cs),int(tr.y/cs)),[]).append(tr)
        for (bx,by),lst in buckets.items():
            nbrs=[u for ddx in (-1,0,1) for ddy in (-1,0,1) for u in buckets.get((bx+ddx,by+ddy),())]
            for a in lst:
                for b in nbrs:
                    if key(a)>=key(b) or getattr(a,'transport','Ground')!=getattr(b,'transport','Ground'):continue
                    a_imm=getattr(a,'is_building',False);b_imm=getattr(b,'is_building',False)
                    if a_imm and b_imm:continue
                    dx=a.x-b.x;dy=a.y-b.y;d=math.hypot(dx,dy);mr=getattr(a,'collision_r',0.5)+getattr(b,'collision_r',0.5)
                    if d>=mr:continue
                    if d<1e-6:dx,dy,d=1.0,0.0,1.0
                    ov=mr-d;nx=dx/d;ny=dy/d;ma=getattr(a,'mass',4);mb=getattr(b,'mass',4)
                    fa,fb=(0,1) if a_imm else (1,0) if b_imm else (mb/(ma+mb),ma/(ma+mb))
                    self._shift(a,nx*ov*fa,ny*ov*fa);self._shift(b,-nx*ov*fb,-ny*ov*fb)
