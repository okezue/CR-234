import pytest

from sim import fx
from sim import replay as R
from sim.cards import create
from sim.fx import has
from sim.game import Game

# t2140: a champion's ability can be used from its landing and takes effect 1.0 s after it is recorded. Recorded abilities start 0.10 s
# after their champion lands (466,443 games, 0.04% earlier); five frame readings fire at the recorded time + 0.9 to 1.2 s, three of
# them recorded while the champion was deploying; r/ClashRoyale 1pae4dj (30 Nov 2025): the button no longer waits ~1 s after deployment
_OUTCOME={'result':'D','tc':0,'oc':0}
CHAMPS=['golden_knight','archer_queen','skeleton_king','mighty_miner','monk','little_prince','goblinstein','boss_bandit']

def _land(g,name,team='blue',hero=False,x=9.5,y=10.5):
    made=create(name,11,team,x,y,hero=hero);made=made if isinstance(made,list) else [made]
    for t in made:g._place(team,t,1.0)
    return next(t for t in made if getattr(t,'ability',None))

def _fired(monkeypatch,cls):
    seen=[];act=cls.activate
    def rec(self,tr,g):
        seen.append(round(g.t,6));act(self,tr,g)
    monkeypatch.setattr(cls,'activate',rec)
    return seen

def _play(card,time,ability=0,team='blue'):
    return {'card':card,'team':team,'tile_x':9.25,'tile_y':10.25,'time':time,'ability':ability,'card_type':'normal'}

@pytest.mark.parametrize('name,hero',[(c,False) for c in CHAMPS]+[('knight',True),('valkyrie',True),('wizard',True)])
def t_ability_is_ready_while_its_champion_deploys(name,hero):
    g=Game(p1={'ability_del':0,'ability_std':0});tr=_land(g,name,hero=hero);g.players['blue'].elixir=10
    assert has(tr,'deploying') and tr.ability.can_use()
    assert g.activate_ability('blue',tr)==(True,'ok')

def t_cast_begun_while_deploying_runs_on(monkeypatch):
    # pressed 0.5 s after the landing: 0.05 s pending and the 1.0 s cast, of which 0.45 s fall inside the deploy
    seen=_fired(monkeypatch,fx.CloakingCape)
    g=Game(p1={'ability_del':0,'ability_std':0});t0=g.t;aq=_land(g,'archer_queen');g.players['blue'].elixir=10
    g.run_to(t0+0.5)
    assert has(aq,'deploying') and g.activate_ability('blue',aq)==(True,'ok')
    g.run_to(t0+1.3)
    assert not has(aq,'deploying') and seen==[]
    g.run_to(t0+2.0)
    assert seen==[pytest.approx(t0+1.55,abs=0.051)] and has(aq,'invisible')

def t_ability_never_takes_effect_before_the_deploy_ends(monkeypatch):
    seen=_fired(monkeypatch,fx.CloakingCape)
    g=Game(p1={'ability_del':0,'ability_std':0});t0=g.t;aq=_land(g,'archer_queen');g.players['blue'].elixir=10
    aq.ability.CAST_TIME=0.3
    assert g.activate_ability('blue',aq)==(True,'ok')
    g.run_to(t0+0.9)
    assert has(aq,'deploying') and seen==[] and aq.ability.casting
    g.run_to(t0+1.5)
    assert len(seen)==1 and t0+0.95<=seen[0]<=t0+1.1

@pytest.mark.parametrize('gap',[22,30,38,50,100])
def t_replay_ability_is_checked_at_its_recorded_time_and_fires_one_second_later(monkeypatch,gap):
    # 1.10, 1.50 and 1.90 s after the play were refused (the knight, queen and Goblinstein readings came 1.35 to 1.75 s after theirs);
    # 2.5 and 5.0 s were accepted but fired at the recorded time + 2.05
    monkeypatch.setattr(Game,'END',gap/20+3.0)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)
    seen=_fired(monkeypatch,fx.CloakingCape)
    g,info=R.replay_battle('champ',[_play('archer_queen',0),_play('ability-archer-queen',gap,1)],_OUTCOME)
    acts=[line for line in g.log if ' activates Archer Queen ability' in line]
    assert len(acts)==1 and acts[0].startswith(f"[{gap/20:.1f}]")
    assert seen==[pytest.approx(gap/20+1.0+0.05,abs=0.051)]

def t_replay_ability_before_its_champion_lands_is_refused(monkeypatch):
    monkeypatch.setattr(Game,'END',4.0)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)
    g,info=R.replay_battle('early',[_play('archer_queen',0),_play('ability-archer-queen',10,1)],_OUTCOME)
    aq=next(t for t in g.players['blue'].troops if t.name=='Archer Queen')
    assert not any(' activates ' in line for line in g.log) and aq.ability.uses==1

def t_replay_ability_before_a_recast_lands_goes_to_the_living_one(monkeypatch):
    # the second queen is still on her way (lands at 7.0) when the ability is recorded at 6.5, so the button is the first queen's
    monkeypatch.setattr(Game,'END',8.0)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)
    g,info=R.replay_battle('recast',[_play('archer_queen',0),_play('archer_queen',120),_play('ability-archer-queen',130,1)],_OUTCOME)
    first,second=[t for t in g.players['blue'].troops if t.name=='Archer Queen']
    assert first.ability.uses==0 and second.ability.uses==1
    assert sum(' activates Archer Queen ability' in line for line in g.log)==1
