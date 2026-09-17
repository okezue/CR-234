import pytest

from sim.cards import create, load
from sim.game import Game
from tests.util import quiet


DECK=['furnace','knight','archers','fireball','giant','musketeer','bomber','arrows']


def t_furnace_spawn_record_matches_the_current_fire_spirit():
    # game data Furnace_rework_continuous_spawn spawns the FireSpirits character: hitSpeed 300, loadTime 100, range 2500
    c=load()['cards'];u=c['furnace']['units']['fire_spirit'];f=c['fire_spirit']
    assert (u['hitSpeed'],u['loadTime'],u['range'],u['speed'])==(0.3,0.1,2.5,120)
    assert (f['hitSpeed'],f['loadTime'],f['range'],f['speed'])==(0.3,0.1,2.5,120)
    assert c['furnace']['skills']['periodicSpawn']['pauseTime']==5


@pytest.mark.parametrize('team',('blue','red'))
def t_spawned_spirit_carries_the_card_spirit_timing_and_range(team):
    g=quiet(Game());furnace=create('furnace',11,team,9,10 if team=='blue' else 22);g.deploy(team,furnace)
    g.run(6.2)
    spirits=[t for t in g.players[team].troops if t.name=='Fire Spirit']
    card=create('fire_spirit',11,team,9,9);card=card[0] if isinstance(card,list) else card
    assert len(spirits)==1
    s=spirits[0]
    assert (s.hspd,s.fhspd,s.rng)==(card.hspd,card.fhspd,card.rng)==(0.3,pytest.approx(0.2),2.5)
    assert (s.hp,s.dmg,s.splash_r,s.is_suicide,s.targets)==(card.hp,card.dmg,card.splash_r,True,card.targets)


def t_spawned_spirit_jumps_within_a_fifth_of_a_second_of_reaching_range():
    g=quiet(Game());furnace=create('furnace',11,'blue',9,10);g.deploy('blue',furnace);g.run(6.2)
    spirit=next(t for t in g.players['blue'].troops if t.name=='Fire Spirit')
    # the Furnace is removed so only the spirit can damage the target
    furnace.hp=0;furnace.alive=False;g._proc_deaths()
    victim=create('giant',11,'red',9,spirit.y+3.5);victim.spd=0;victim.dmg=0;g.deploy('red',victim);hp=victim.hp
    entered=None;launched=None
    for _ in range(int(round(4.0/g.DT))):
        g.tick()
        if entered is None and g._dist(spirit,victim)<=spirit.rng:entered=g.t
        if launched is None and not spirit.alive:launched=g.t
        if victim.hp<hp:break
    assert entered is not None and launched is not None
    # the jump leaves 0.2 s after entering 2.5 tiles (hit speed 0.3 less load time 0.1); the old spawn record needed 0.9 s from 2.0 tiles
    assert launched-entered==pytest.approx(0.2,abs=g.DT/2)
    assert entered<launched<g.t and hp-victim.hp==207


def t_furnace_itself_still_attacks_with_its_own_cadence():
    g=quiet(Game());furnace=create('furnace',11,'blue',9,10);g.deploy('blue',furnace)
    victim=create('giant',11,'red',9,14.5);victim.spd=0;victim.dmg=0;g.deploy('red',victim)
    hits=[]
    for _ in range(int(round(4.9/g.DT))):
        last=victim.hp;g.tick()
        if victim.hp!=last:hits.append((round(g.t,2),last-victim.hp))
    # before the first spawn at five seconds only the Furnace's own shots land: 179 per hit at level 11, 1.7 s apart
    assert len(hits)>=2 and {d for _,d in hits}=={179}
    assert all(round(b-a,2)==1.7 for (a,_),(b,_) in zip(hits,hits[1:]))


# game data spawn records: Goblin_Stab (Goblin Gang) loadTime 500, Goblin (Goblin Drill) 700, BushGoblin (Suspicious Bush) 1100
LOAD={'goblin_gang':('goblin',0.5,1.1,0.5),'goblin_drill':('goblin',0.7,1.1,0.5),'suspicious_bush':('bush_goblin',1.1,1.4,0.8)}


@pytest.mark.parametrize('card',sorted(LOAD))
def t_spawned_goblin_records_follow_the_export(card):
    unit,lt,hs,rng=LOAD[card];u=load()['cards'][card]['units'][unit]
    assert (u['loadTime'],u['hitSpeed'],u['range'])==(lt,hs,rng)


def t_goblin_gang_goblins_carry_the_stab_load_time():
    g=quiet(Game());units=create('goblin_gang',11,'blue',9,10)
    goblins=[u for u in units if u.name=='Goblin'];spears=[u for u in units if u.name=='SpearGoblin']
    assert len(goblins)==3 and len(spears)==3
    assert all((u.hspd,u.fhspd)==(1.1,pytest.approx(0.6)) for u in goblins)
    assert all((u.hspd,u.fhspd)==(1.6,pytest.approx(0.5)) for u in spears)
    for u in units:g.deploy('blue',u)


def t_bush_goblins_hit_after_their_own_load_time_not_the_bush_load_time():
    g=quiet(Game());bush=create('suspicious_bush',11,'blue',9,10);g.deploy('blue',bush)
    assert bush.hspd==0.3
    spawn=next(c for c in bush.components if type(c).__name__=='DeathSpawn')
    cfg=spawn.cfg if hasattr(spawn,'cfg') else spawn.scfg
    assert (cfg['hspd'],cfg['fhspd'],cfg['rng'])==(1.4,pytest.approx(0.3),0.8)
    bush.hp=0;bush.alive=False;g._proc_deaths()
    goblins=[t for t in g.players['blue'].troops if t.name=='Bush Goblin']
    assert len(goblins)==2 and all(t.fhspd==pytest.approx(0.3) for t in goblins)


def t_drill_goblins_carry_the_export_load_time():
    g=quiet(Game());drill=create('goblin_drill',11,'blue',9,10);g.deploy('blue',drill)
    cfgs=[cfg for cfg in (getattr(c,'cfg',None) or getattr(c,'scfg',None) for c in drill.components) if isinstance(cfg,dict) and cfg.get('name')=='Goblin']
    # the periodic spawn and the death spawn both use the export Goblin record
    assert len(cfgs)==2 and all((cfg['hspd'],cfg['fhspd'])==(1.1,pytest.approx(0.4)) for cfg in cfgs)
