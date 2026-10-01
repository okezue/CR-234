from sim import fx
from sim.cards import card,create
from sim.game import Game
from tests.util import Dummy,quiet


# Balance update of 4 August 2026 (Supercell blog, wiki history 4/8/2026): Doctor damage 92 -> 135, Lightning Link damage per second
# 214 -> 188, i.e. 107 -> 94 per 0.5 s tick. The game data export gives the Doctor's shot 53 at level 1 (135 at level 11) and carries the
# ability on Goblinstein_doctor. Wiki ability text: an electric link between the Doctor and the Monster shocks targets up to 2 tiles away
# every 0.5 s; with the Monster dead it runs to the antenna the Monster drops; killing the Doctor removes the ability.
TICK=94;CT=46;TICKS=8


def gob(doc=(5.5,10.0),mon=(12.5,10.0)):
    # a blue Goblinstein whose Doctor and Monster stand still where given and deal no damage; the towers are quiet
    g=quiet(Game(p1={'ability_std':0}))
    ts=create('goblinstein',11,'blue',9,10)
    d=next(t for t in ts if t.name=='Goblinstein_doctor');m=next(t for t in ts if t.name=='Monster')
    for t,(x,y) in ((d,doc),(m,mon)):t.x,t.y=x,y;t.spd=0;t.dmg=0;t.ct_dmg=0;g.deploy('blue',t)
    return g,d,m,ts


def bodies(g,*pts):
    out=[]
    for x,y in pts:
        b=Dummy('red',x,y,hp=5000,spd=0,dmg=0);b.targets=[];g.deploy('red',b);out.append(b)
    return out


def link(g,ts):
    # activates the ability on whichever unit carries it and returns the time the cast ends
    h=next(t for t in ts if getattr(t,'ability',None))
    g.players['blue'].elixir=10;h.ability.cd=0
    assert g.activate_ability('blue',h)[0],"the Goblinstein should be able to cast"
    return g.t+fx.Ability.CAST_TIME


def t_doctor_shot_is_the_august_4_value():
    c=card('goblinstein');u=c['units']['doctor']
    assert u['damage'][10]==135 and c['src']['units.doctor.damage']=='patch:2026-10-01g',(u['damage'],c['src'].get('units.doctor.damage'))
    g=quiet(Game());d=next(t for t in create('goblinstein',11,'blue',9,10) if t.name=='Goblinstein_doctor');d.spd=0;g.deploy('blue',d)
    b,=bodies(g,(9,13))
    g.run(3)
    assert d.dmg==135 and (5000-b.hp)%135==0 and b.hp<5000,(d.dmg,b.hp)


def t_lightning_link_tick_is_the_august_4_damage_per_second():
    c=card('goblinstein');p=c['skills']['ability']['skills']['poison']
    assert p['damage'][10]==TICK and p['tickInterval']==0.5 and c['src']['skills.ability.skills.poison.damage']=='patch:2026-10-01g',(p,c['src'])
    ab=next(t.ability for t in create('goblinstein',11,'blue',9,10) if getattr(t,'ability',None))
    assert (ab.tick_dmg,ab.ti,ab.tick_ct)==(TICK,0.5,CT),(ab.tick_dmg,ab.ti,ab.tick_ct)


def t_lightning_link_belongs_to_the_doctor():
    assert card('goblinstein')['skills']['group']['holder']=='Doctor'
    ts=create('goblinstein',11,'blue',9,10)
    d=next(t for t in ts if t.name=='Goblinstein_doctor');m=next(t for t in ts if t.name=='Monster')
    assert isinstance(getattr(d,'ability',None),fx.LightningLink) and getattr(m,'ability',None) is None,(getattr(d,'ability',None),getattr(m,'ability',None))


def t_lightning_link_shocks_along_the_doctor_monster_segment():
    # a 7 tile link: bodies 1.5 tiles beside the Doctor, the midpoint and the Monster take every tick, a body 3 tiles off takes none
    g,d,m,ts=gob()
    doc,mid,mon,off=bodies(g,(5.5,11.5),(9.0,11.5),(12.5,11.5),(9.0,13.0))
    te=link(g,ts);g.run_to(te+4.6)
    for b,n in ((doc,'Doctor'),(mid,'midpoint'),(mon,'Monster')):assert 5000-b.hp==TICKS*TICK,f"body beside the {n} lost {5000-b.hp}"
    assert off.hp==5000,f"body 3 tiles off the link lost {5000-off.hp}"


def t_lightning_link_reaches_bodies_like_other_area_effects():
    # a body whose centre is 2.3 tiles from the link touches the 2 tile reach with its 0.5 tile collision radius (fx.tdist)
    g,d,m,ts=gob()
    b,=bodies(g,(9.0,12.3))
    te=link(g,ts);g.run_to(te+4.6)
    assert 5000-b.hp==TICKS*TICK,5000-b.hp


def t_lightning_link_hits_a_crown_tower_beside_the_link_with_its_tower_damage():
    # the red left princess tower (centre 3.5, 25.5, collision radius 1) is 1 tile from the link's middle and 2.6 and 3.5 tiles from its ends
    g,d,m,ts=gob(doc=(0.5,23.5),mon=(7.5,23.5))
    tw=g.arena.get_tower('red','princess','left');ini=tw.hp
    te=link(g,ts);g.run_to(te+4.6)
    assert ini-tw.hp==TICKS*CT,ini-tw.hp


def t_lightning_link_runs_to_the_antenna_where_the_monster_fell():
    # the Monster falls at (12.5, 10); a later move of its corpse does not move the antenna
    g,d,m,ts=gob()
    mid,ant=bodies(g,(9.0,11.5),(12.5,11.5))
    m.take_damage(m.hp);g.run(0.1);assert not m.alive
    m.x,m.y=9.0,4.0
    te=link(g,ts);g.run_to(te+4.6)
    assert 5000-mid.hp==TICKS*TICK and 5000-ant.hp==TICKS*TICK,(5000-mid.hp,5000-ant.hp)


def t_lightning_link_ends_with_the_doctor():
    # the Doctor dies after the second tick: the bodies beside the link take no more
    g,d,m,ts=gob()
    mon,=bodies(g,(12.5,11.5))
    te=link(g,ts)
    while 5000-mon.hp<2*TICK and g.t<te+5:g.tick()
    d.take_damage(d.hp);g.run_to(te+4.6)
    assert 5000-mon.hp==2*TICK,5000-mon.hp
