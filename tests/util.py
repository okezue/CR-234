class Dummy:
    _n=0
    def __init__(self,team,x,y,lvl=11,hp=500,dmg=100,spd=2.0,hspd=1.0,rng=1.5,mass=4):
        Dummy._n+=1;self.id=Dummy._n
        self.team=team;self.x=float(x);self.y=float(y)
        self.hp=hp;self.max_hp=hp;self.dmg=dmg
        self.spd=spd;self.hspd=hspd;self.rng=rng
        self.alive=True;self.lvl=lvl;self.cd=0
        self.transport='Ground';self.targets=['Ground']
        self.components=[];self.statuses=[]
        self.atk_type='single_target';self.splash_r=0
        self.fhspd=hspd;self.first_atk=True;self.tgt=None
        self.name='dummy';self.ct_dmg=0;self.mass=mass
        self.sight_r=5.5;self.collision_r=0.5
        self.retarget_cd=0;self.aggro_tgt=None;self.proj_spd=0
    def level_up(self):
        self.lvl+=1;oh=self.max_hp
        self.max_hp=int(self.max_hp*1.1)
        self.hp+=self.max_hp-oh
        self.dmg=int(self.dmg*1.1)
    def take_damage(self,a):
        if not self.alive:return
        self.hp-=a
        if self.hp<=0:self.hp=0;self.alive=False
def quiet(g):
    for t in g.arena.towers:
        t.dmg=0
        if t.troop:t.troop.dmg=0
    return g
