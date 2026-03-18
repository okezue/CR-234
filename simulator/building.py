class Building:
    _n=0
    def __init__(self,team,x,y,cfg):
        Building._n+=1;self.id=Building._n
        self.team=team;self.x=float(x);self.y=float(y)
        self.hp=cfg['hp'];self.max_hp=cfg['hp']
        self.dmg=cfg.get('dmg',0);self.spd=0
        self.hspd=cfg.get('hspd',1.0);self.fhspd=cfg.get('fhspd',cfg.get('hspd',1.0))
        self.rng=cfg.get('rng',1.2);self.min_rng=cfg.get('min_rng',0)
        self.transport='Ground'
        self.targets=cfg.get('targets',['Ground'])
        self.atk_type=cfg.get('atk_type','single_target')
        self.splash_r=cfg.get('splash_r',0);self.ct_dmg=cfg.get('ct_dmg',0)
        self.components=list(cfg.get('components',[]))
        self.statuses=[];self.alive=True
        self.lvl=cfg.get('lvl',11);self.cd=0;self.first_atk=True;self.tgt=None
        self.name=cfg.get('name','');self.is_building=True
        self.lifetime=cfg.get('lifetime',30.0)
        self.decay=self.max_hp/self.lifetime if self.lifetime>0 else 0
        self.death_dmg=cfg.get('death_dmg',0);self.death_splash_r=cfg.get('death_splash_r',0)
        self.shield_hp=cfg.get('shield_hp',0);self.max_shield_hp=cfg.get('max_shield_hp',0)
        self.slow_dur=cfg.get('slow_dur',0);self.slow_val=cfg.get('slow_val',1.0)
        self.stun_dur=cfg.get('stun_dur',0);self.is_suicide=False
        self.freeze_dur=cfg.get('freeze_dur',0)
        self.spawn_zap_dmg=cfg.get('spawn_zap_dmg',0);self.spawn_zap_r=cfg.get('spawn_zap_r',0)
        self.charge_dmg=0;self.chain_count=0;self.chain_range=0;self.chain_stun=0
        self.ramp_stages=[];self.ramp_durations=[]
        self.mass=cfg.get('mass',12)
        self.sight_r=cfg.get('sight_r',5.5)
        self.collision_r=cfg.get('collision_r',0.5)
        self._path=[];self._path_idx=0;self._path_tgt=None
        self.retarget_cd=0;self.aggro_tgt=None
    def take_damage(self,a):
        if not self.alive:return
        if self.shield_hp>0:
            self.shield_hp-=a
            if self.shield_hp<0:self.shield_hp=0
            return
        self.hp-=a
        if self.hp<=0:self.hp=0;self.alive=False
    def on_death(self,game):
        for c in self.components:c.on_death(self,game)
