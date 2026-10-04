import pytest

from sim import replay as R
from sim.cards import create
from sim.fx import BannerBrigade
from sim.game import Game


_OUTCOME={'result':'D','tc':0,'oc':0}
_ABSENT=(False,'recorded ability troop absent')


def _deploy(g,team,name='archer_queen',hero=False):
    tr=create(name,11,team,4,6 if team=='blue' else 26,hero=hero)
    tr.spd=0;tr.ability.cd=0
    g.deploy(team,tr)
    return tr


def _state(g,team):
    p=g.players[team]
    return (p.active_champ,list(p.champ_queue),p.elixir,list(g.pending_ab),list(g.log),
            [dict(vars(t.ability)) for t in p.troops if getattr(t,'ability',None)])


def _play(name,team='blue',time=0,ability=1):
    return {'card':name,'team':team,'time':time,'tile_x':4.25,'tile_y':6.25 if team=='blue' else 25.25,
            'ability':ability,'card_type':'normal'}


def _pending_banner(g,team):
    p=g.players[team]
    goblins=create('goblins',11,team,4,6 if team=='blue' else 26,hero=True)
    for tr in goblins:g.deploy(team,tr);tr.take_damage(tr.hp)
    g._proc_deaths()
    ab=goblins[0].ability
    assert isinstance(ab,BannerBrigade) and ab in p.pending_abilities and ab.can_use()
    # Model the pending-banner path with no champion occupying the ability slot.
    p.active_champ=None;p.champ_queue.clear()
    return ab


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('case',('queued','same_card','missing','dead','dead_active','active','unknown','invalid','ordinary'))
def t_replay_path_submits_only_the_latest_named_troop(monkeypatch,team,case):
    refs={}
    def game(**kwargs):
        g=Game(**kwargs);p=g.players[team]
        active=_deploy(g,team,'golden_knight' if case in ('queued','missing','dead') else 'archer_queen')
        named=active
        if case in ('queued','same_card','dead'):
            named=_deploy(g,team)
        if case=='dead':named.take_damage(named.hp);g._proc_deaths()
        if case=='dead_active':active.take_damage(active.hp)
        refs.update(active=active,named=named,selected=p.active_champ,queue=list(p.champ_queue))
        p.elixir=0
        return g
    monkeypatch.setattr(R,'Game',game)
    monkeypatch.setattr(Game,'END',0.1)
    name={'unknown':'ability-unknown-card','invalid':'_invalid','ordinary':'ability-knight'}.get(case,'ability-archer-queen')
    g,info=R.replay_battle('ability-'+case,[_play(name,team)],_OUTCOME)
    p=g.players[team]
    activations=[line for line in g.log if ' activates ' in line]
    if case in ('active','queued','same_card'):
        assert len(activations)==1 and 'Archer Queen ability' in activations[0]
        assert refs['named'].ability.casting and refs['named'].ability.uses==0
        if refs['active'] is not refs['named']:assert refs['active'].ability.uses==1
        assert 9<=p.elixir<10
    else:
        assert activations==[]
        assert refs['active'].ability.uses==1 and refs['named'].ability.uses==1
        assert p.elixir==10
    if case!='dead_active':
        assert p.active_champ is refs['selected'] and p.champ_queue==refs['queue']
    assert info['placement']['attempted']==0


@pytest.mark.parametrize('case',('queued','hero','missing','active'))
def t_replay_path_real_placements_and_ability(monkeypatch,case):
    monkeypatch.setattr(Game,'END',4.2)
    plays=[_play('golden_knight',ability=0)]
    if case=='queued':plays.append(_play('archer_queen',time=40,ability=0))
    elif case=='hero':plays.append(_play('bowler-hero',time=40,ability=0))
    name='golden-knight' if case=='active' else 'bowler' if case=='hero' else 'archer-queen'
    plays.append(_play('ability-'+name,time=80))
    g,info=R.replay_battle('placed-ability-'+case,plays,_OUTCOME)
    p=g.players['blue'];first=next(t for t in p.troops if t.name=='Golden Knight')
    accepted=case!='missing'
    activations=[line for line in g.log if ' activates ' in line]
    assert len(activations)==int(accepted)
    assert first.ability.uses==int(case!='active')
    if case in ('queued','hero'):
        named=next(t for t in p.troops if t is not first)
        assert named.ability.uses==0 and named.ability.casting
    assert info['placement']['attempted']==1+int(case in ('queued','hero'))


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('base',('archer_queen','archer-queen'))
def t_latest_named_resolution_is_nonmutating_and_explicit(team,base):
    g=Game();older=_deploy(g,team);latest=_deploy(g,team)
    p=g.players[team];p.elixir=4
    before=_state(g,team)
    assert R._ability_troop(g,team,base) is latest
    assert _state(g,team)==before
    assert not g.activate_ability(team,older)[0]
    assert _state(g,team)==before
    assert R.submit_recorded_ability(g,team,base)==(True,'ok')
    assert p.active_champ is before[0] and p.champ_queue==before[1] and p.elixir==3
    assert len(g.pending_ab)==1 and g.pending_ab[0].troop is latest
    assert older.ability.can_use()


@pytest.mark.parametrize('team',('blue','red'))
def t_different_card_submission_does_not_switch_selected_champion(team):
    g=Game();first=_deploy(g,team,'golden_knight');named=_deploy(g,team,'bowler',hero=True)
    p=g.players[team];p.elixir=4;before=_state(g,team)
    assert R._ability_troop(g,team,'bowler') is named and _state(g,team)==before
    assert R.submit_recorded_ability(g,team,'bowler')==(True,'ok')
    assert p.active_champ is before[0] and p.champ_queue==before[1]
    assert first.ability.can_use() and g.pending_ab[0].troop is named


@pytest.mark.parametrize('case',('missing','dead','dead_active','unknown','invalid','ordinary','opponent'))
def t_rejected_submission_does_not_change_game_state(case):
    g=Game();active=_deploy(g,'blue','golden_knight')
    base='archer_queen'
    if case=='dead':
        tr=_deploy(g,'blue');tr.take_damage(tr.hp);g._proc_deaths()
    elif case=='dead_active':active.take_damage(active.hp);base='golden_knight'
    elif case=='unknown':base='unknown_card'
    elif case=='invalid':base=None
    elif case=='ordinary':g.deploy('blue',create('knight',11,'blue',4,6));base='knight'
    elif case=='opponent':_deploy(g,'red')
    before=_state(g,'blue')
    assert R._ability_troop(g,'blue',base) is None
    assert R.submit_recorded_ability(g,'blue',base)==_ABSENT
    assert _state(g,'blue')==before


@pytest.mark.parametrize('case',('cooldown','spent','pending','elixir'))
def t_submission_preserves_engine_readiness_and_resource_checks(case):
    g=Game();older=_deploy(g,'blue');latest=_deploy(g,'blue');p=g.players['blue'];p.elixir=4
    if case=='cooldown':latest.ability.cd=3
    elif case=='spent':latest.ability.uses=0
    elif case=='pending':latest.ability._pend=True
    else:p.elixir=0
    before=_state(g,'blue')
    expected='not enough elixir' if case=='elixir' else 'ability not ready'
    assert R._ability_troop(g,'blue','archer_queen') is latest
    assert R.submit_recorded_ability(g,'blue','archer_queen')==(False,expected)
    assert _state(g,'blue')==before and older.ability.can_use()


def t_latest_named_resolution_ignores_dead_units_and_clones():
    g=Game();older=_deploy(g,'blue');latest=_deploy(g,'blue');dead=_deploy(g,'blue')
    dead.take_damage(dead.hp);g._proc_deaths()
    create('clone',11,'blue',4,6).apply(g)
    clones=[t for t in g.players['blue'].troops if t not in (older,latest)]
    assert clones and all(getattr(t,'ability',None) is None for t in clones)
    before=_state(g,'blue')
    assert R._ability_troop(g,'blue','archer_queen') is latest
    assert _state(g,'blue')==before
    assert R.submit_recorded_ability(g,'blue','archer_queen')==(True,'ok')
    assert g.pending_ab[0].troop is latest and older.ability.can_use()


def t_spawned_unit_name_resolves_recorded_card():
    g=Game();troops=create('goblinstein',11,'blue',4,6)
    for tr in troops:g.deploy('blue',tr)
    named=next(t for t in troops if getattr(t,'ability',None));named.ability.cd=0
    assert named.name!='Goblinstein'
    assert R._ability_troop(g,'blue','goblinstein') is named
    assert R.submit_recorded_ability(g,'blue','goblinstein')==(True,'ok')
    assert g.pending_ab[0].troop is named


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('named',('goblins','archer-queen','unknown-card'))
@pytest.mark.parametrize('has_active',(False,True))
def t_replay_path_pending_banner_requires_named_goblins(monkeypatch,team,named,has_active):
    refs={}
    def game(**kwargs):
        g=Game(**kwargs);refs['banner']=_pending_banner(g,team)
        refs['active']=_deploy(g,team,'golden_knight') if has_active else None
        g.players[team].elixir=0
        return g
    monkeypatch.setattr(R,'Game',game)
    monkeypatch.setattr(Game,'END',0.1)
    g,info=R.replay_battle('banner',[_play('ability-'+named,team)],_OUTCOME)
    accepted=named=='goblins'
    ab=refs['banner'];p=g.players[team]
    assert ab.uses==1-int(accepted)
    assert len(p.troops)==int(has_active)+(ab.spawn_cnt if accepted else 0)
    assert sum(' activates ' in line for line in g.log)==int(accepted)
    assert p.active_champ is refs['active']
    assert info['placement']['attempted']==0


def t_banner_submission_uses_real_pending_ability_without_refilling_elixir():
    g=Game();ab=_pending_banner(g,'blue');p=g.players['blue'];p.elixir=4
    assert R.submit_recorded_ability(g,'blue','goblins')==(True,'ok')
    assert p.active_champ is None and p.elixir==4-ab.cost
    assert len(g.pending_ab)==1 and g.pending_ab[0].ability is ab and g.pending_ab[0].is_banner
    assert g.pending_ab[0].troop is None


def t_missing_goblins_without_banner_is_rejected():
    g=Game();before=_state(g,'blue')
    assert R.submit_recorded_ability(g,'blue','goblins')==_ABSENT
    assert _state(g,'blue')==before
