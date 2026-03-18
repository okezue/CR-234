import math,heapq
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
        g=[[True]*self.W for _ in range(self.H)]
        for y in range(self.H):
            for x in range(self.W):
                c=self.arena.grid[y][x]
                if c and len(c)==2 and c[1] in ('K','P','T'):
                    g[y][x]=False
                elif not air and c=='R':
                    g[y][x]=False
        return g
    def rebuild_tower_grid(self):
        for y in range(self.H):
            for x in range(self.W):
                c=self.arena.grid[y][x]
                if c and len(c)==2 and c[1] in ('K','P','T'):
                    alive=False
                    for t in self.arena.towers:
                        if t.alive and (x,y) in t.tiles():alive=True;break
                    self._gnd[y][x]=not alive
                    self._air[y][x]=not alive
        self._cache.clear()
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
    def get_path(self,tr,tx,ty):
        sx=max(0,min(self.W-1,int(tr.x)))
        sy=max(0,min(self.H-1,int(tr.y)))
        gx=max(0,min(self.W-1,int(tx)))
        gy=max(0,min(self.H-1,int(ty)))
        air=getattr(tr,'transport','Ground')=='Air'
        if not air:
            from components import RiverJump
            if any(isinstance(c,RiverJump) for c in getattr(tr,'components',[])):air=True
        key=(sx,sy,gx,gy,air)
        if key in self._cache:return list(self._cache[key])
        p=self.a_star(sx,sy,gx,gy,air)
        self._cache[key]=p
        return list(p)
    def resolve_collisions(self,troops,dt=0.1):
        cs=2.0
        alive=[tr for tr in troops if tr.alive]
        buckets={}
        for tr in alive:
            bx=int(tr.x/cs);by=int(tr.y/cs)
            buckets.setdefault((bx,by),[]).append(tr)
        checked=set()
        for (bx,by),lst in list(buckets.items()):
            nbrs=[]
            for ddx in (-1,0,1):
                for ddy in (-1,0,1):
                    k=(bx+ddx,by+ddy)
                    if k in buckets:nbrs.extend(buckets[k])
            for a in lst:
                for b in nbrs:
                    if a.id>=b.id:continue
                    pair=(a.id,b.id)
                    if pair in checked:continue
                    checked.add(pair)
                    if getattr(a,'team',None)!=getattr(b,'team',None):continue
                    at=getattr(a,'transport','Ground')
                    bt=getattr(b,'transport','Ground')
                    if at!=bt:continue
                    dx=a.x-b.x;dy=a.y-b.y
                    d=math.sqrt(dx*dx+dy*dy)
                    mr=getattr(a,'collision_r',0.5)+getattr(b,'collision_r',0.5)
                    if d>=mr or d<0.001:continue
                    overlap=mr-d
                    nx=dx/d;ny=dy/d
                    ma=getattr(a,'mass',4);mb=getattr(b,'mass',4)
                    tm=ma+mb
                    if tm<=0:continue
                    a_imm=getattr(a,'is_building',False)
                    b_imm=getattr(b,'is_building',False)
                    if a_imm and b_imm:continue
                    if a_imm:
                        b.x-=nx*overlap*0.02;b.y-=ny*overlap*0.02
                    elif b_imm:
                        a.x+=nx*overlap*0.02;a.y+=ny*overlap*0.02
                    else:
                        ra=mb/tm;rb=ma/tm
                        a.x+=nx*overlap*0.02*ra;a.y+=ny*overlap*0.02*ra
                        b.x-=nx*overlap*0.02*rb;b.y-=ny*overlap*0.02*rb
        for tr in alive:
            tr.x=max(0.3,min(17.7,tr.x))
            tr.y=max(0.3,min(31.7,tr.y))
