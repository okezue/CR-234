from copy import deepcopy
import csv

import numpy as np
import pytest

from sim import replay as R
from sim.cards import card
from sim.env import CREnv,OBS_DIM
from sim.fx import EvoRoyalGhost,FrostyFella
from sim.game import Deck,Game,Player,validate_deck
from sim.units import has
from tests.util import Dummy,quiet


_FILL=['knight','archers','fireball','giant','musketeer','valkyrie','bomber','arrows']
_PREFIX=391


def _deck(*names):
    return list(dict.fromkeys([*names,*_FILL]))[:8]


def _game(*names,evolutions=None,heroes=None):
    config={'deck':_deck(*names),'evolutions':evolutions,'heroes':heroes,'drag_del':0,'ability_del':0,'ability_std':0}
    return Game(p1=config,p2=config)


def _ready(g,name,team='blue'):
    p=g.players[team]
    R._force_hand(g,team,name)
    p.elixir=10
    return p


def _play(g,name,team='blue',**flags):
    _ready(g,name,team)
    assert g.play_card(team,name,9,10 if team=='blue' else 21,**flags)==(True,'ok')
    return g.pending[-1]


def _flush(g):
    for _ in range(30):g._proc_pending()
    assert not g.pending


def t_player_normalizes_and_copies_equipped_configuration():
    evolutions=['royal-ghost','royal_ghost'];heroes=['ice-wizard']
    p=Player('blue',deck=_deck('royal_ghost','ice_wizard'),evolutions=evolutions,heroes=heroes)
    evolutions.clear();heroes.clear()
    assert p.evolutions=={'royal_ghost'} and p.heroes=={'ice_wizard'}
    assert p.evolution_charge=={'royal_ghost':0}


@pytest.mark.parametrize('options',[
    {'evolutions':['royal_ghost']},
    {'heroes':['ice_wizard']},
    {'evolutions':['giant']},
    {'heroes':['arrows']},
    {'evolutions':['unknown_card']},
    {'heroes':['unknown_card']},
    {'evolutions':['knight'],'heroes':['knight']},
])
def t_invalid_equipped_configuration_is_rejected(options):
    with pytest.raises(AssertionError):Player('blue',deck=_deck(),**options)


def t_equipped_cards_require_a_deck():
    with pytest.raises(AssertionError):Player('blue',evolutions=['knight'])
    with pytest.raises(AssertionError):Player('blue',heroes=['ice_wizard'])


@pytest.mark.parametrize('champions,heroes,valid',[
    ([],['ice_wizard','knight'],True),
    (['golden_knight'],['ice_wizard'],True),
    (['golden_knight','archer_queen'],[],True),
    (['golden_knight'],['ice_wizard','knight'],False),
    (['golden_knight','archer_queen'],['ice_wizard'],False),
    (['golden_knight','archer_queen','monk'],[],False),
])
def t_equipped_heroes_share_two_slots_with_champions(champions,heroes,valid):
    args={'deck':_deck(*champions,*heroes),'heroes':heroes}
    if valid:assert Player('blue',**args).heroes==set(heroes)
    else:
        with pytest.raises(AssertionError,match='hero/champion'):Player('blue',**args)


@pytest.mark.parametrize('team',['blue','red'])
@pytest.mark.parametrize('name,cycles',[('bomber',1),('royal_ghost',2),('giant_snowball',2),('barbarians',1)])
def t_equipped_cadence_runs_two_complete_cycles(team,name,cycles):
    assert card(name)['evo']['cycles']==cycles
    g=_game(name,evolutions=[name]);p=g.players[team]
    for n in range(1,2*(cycles+1)+1):
        pd=_play(g,name,team)
        assert pd.evolved==(n%(cycles+1)==0) and pd.hero is False
        assert p.evolution_charge[name]==n%(cycles+1)
        charge=dict(p.evolution_charge)
        _flush(g)
        assert p.evolution_charge==charge
    assert g.players[g._opp(team)].evolution_charge=={name:0}


def t_royal_ghost_factory_receives_base_base_evolution():
    g=_game('royal_ghost',evolutions=['royal_ghost']);p=g.players['blue']
    for expected in [False,False,True]:
        _play(g,'royal_ghost');_flush(g)
        tr=p.troops[-1]
        assert getattr(tr,'evolved',False)==expected
        assert any(isinstance(c,EvoRoyalGhost) for c in tr.components)==expected


@pytest.mark.parametrize('team',['blue','red'])
@pytest.mark.parametrize('charge',[0,2])
@pytest.mark.parametrize('reason',['invalid position','not enough elixir','not in hand'])
def t_rejected_play_never_changes_cadence(team,charge,reason):
    g=_game('royal_ghost',evolutions=['royal_ghost']);p=_ready(g,'royal_ghost',team)
    p.evolution_charge['royal_ghost']=charge
    x,y=9,10 if team=='blue' else 21
    if reason=='invalid position':x,y=0,0
    elif reason=='not enough elixir':p.elixir=0
    else:p.deck.hand.remove('royal_ghost')
    before=deepcopy((p.evolution_charge,p.deck.__dict__,p.elixir,p.last_card))
    assert g.play_card(team,'royal_ghost',x,y)==(False,reason)
    assert (p.evolution_charge,p.deck.__dict__,p.elixir,p.last_card)==before
    assert not g.pending


def t_two_equipped_cards_and_teams_charge_independently():
    g=_game('bomber','royal_ghost',evolutions=['royal_ghost','bomber'])
    order=[('blue','royal_ghost'),('red','bomber'),('blue','bomber'),('red','royal_ghost'),
           ('blue','royal_ghost'),('blue','bomber'),('red','royal_ghost'),('red','bomber'),
           ('red','royal_ghost'),('blue','royal_ghost')]
    counts={tm:{'bomber':0,'royal_ghost':0} for tm in ('blue','red')}
    for tm,name in order:
        counts[tm][name]+=1
        pd=_play(g,name,tm);period=card(name)['evo']['cycles']+1
        assert pd.evolved==(counts[tm][name]%period==0)
        for side,p in g.players.items():
            assert p.evolution_charge=={c:n%(card(c)['evo']['cycles']+1) for c,n in counts[side].items()}


@pytest.mark.parametrize('charge',[0,2])
@pytest.mark.parametrize('override',[False,True])
def t_explicit_evolution_override_does_not_advance_or_consume_auto_charge(charge,override):
    g=_game('royal_ghost',evolutions=['royal_ghost']);p=g.players['blue']
    p.evolution_charge['royal_ghost']=charge
    pd=_play(g,'royal_ghost',evolved=override)
    assert pd.evolved is override and p.evolution_charge=={'royal_ghost':charge}
    _flush(g)
    assert p.evolution_charge=={'royal_ghost':charge}
    pd=_play(g,'royal_ghost')
    assert pd.evolved==(charge==2) and p.evolution_charge['royal_ghost']==(charge+1)%3


def t_unequipped_cards_stay_base_but_explicit_factory_overrides_work():
    g=_game('royal_ghost','ice_wizard')
    for _ in range(6):assert _play(g,'royal_ghost').evolved is False
    assert _play(g,'royal_ghost',evolved=True).evolved is True
    assert _play(g,'ice_wizard').hero is False
    assert _play(g,'ice_wizard',hero=True).hero is True
    assert g.players['blue'].evolution_charge=={}


@pytest.mark.parametrize('name',['knight','ice_wizard'])
@pytest.mark.parametrize('flags',[{}, {'evolved':True,'hero':True}])
def t_mirror_is_always_base_and_never_charges_the_copied_card(name,flags):
    g=_game('mirror','ice_wizard',evolutions=['knight'],heroes=['ice_wizard']);p=g.players['blue']
    _play(g,name,evolved=name=='knight');_flush(g)
    p.evolution_charge['knight']=2
    before=dict(p.evolution_charge)
    pd=_play(g,'mirror',**flags)
    assert pd.card=='mirror:'+name and pd.evolved is False and pd.hero is False
    _flush(g);copy=p.troops[-1]
    assert copy.lvl==12 and not getattr(copy,'evolved',False) and not getattr(copy,'is_hero',False)
    assert p.evolution_charge==before and p.last_card==name
    copy=g._spawn('blue','mirror:'+name,9,10,evolved=True,hero=True)
    assert not getattr(copy,'evolved',False) and not getattr(copy,'is_hero',False)


def t_clone_and_death_spawns_leave_equipped_charge_unchanged():
    g=_game('clone','skeleton_barrel',evolutions=['skeleton_barrel']);p=g.players['blue']
    _play(g,'skeleton_barrel');_flush(g)
    original=p.troops[0];before=dict(p.evolution_charge)
    _play(g,'clone');_flush(g)
    assert len(p.troops)==2 and p.troops[-1].hp==1
    assert p.evolution_charge==before
    original.take_damage(original.hp);g._proc_deaths()
    assert any(t.name=='Skeleton' for t in p.troops)
    assert p.evolution_charge==before


@pytest.mark.parametrize('team',['blue','red'])
def t_equipped_ice_wizard_automatically_gets_factory_ability(team):
    g=_game('ice_wizard',heroes=['ice_wizard']);p=g.players[team]
    _play(g,'ice_wizard',team);_flush(g)
    tr=p.troops[0]
    assert tr.is_hero and isinstance(tr.ability,FrostyFella) and p.active_champ is tr
    g.run(2.1)
    assert g.activate_ability(team)==(True,'ok')
    g.run(1.2)
    assert tr.ability.active and not any(t.name=='Snowman' for t in p.troops)
    target=Dummy(g._opp(team),tr.x,tr.y+1,hp=5000,spd=0,dmg=0);g.deploy(target.team,target)
    g._do_attack(tr,target)
    assert any(t.name=='Snowman' and t.is_building for t in p.troops)
    assert p.evolution_charge=={}
    pd=_play(g,'ice_wizard',team,hero=False);assert pd.hero is False
    _flush(g)
    assert not getattr(p.troops[-1],'is_hero',False) and not getattr(p.troops[-1],'ability',None)


def t_snapshot_equipped_order_charge_copy_and_troop_flags():
    g=_game('royal_ghost','ice_wizard',evolutions={'royal_ghost','knight'},heroes={'ice_wizard'})
    for _ in range(3):_play(g,'royal_ghost')
    _play(g,'knight');_play(g,'ice_wizard');_flush(g)
    g.replay.snap(g);snap=g.replay.snaps[-1]
    assert snap['blue']['evolutions']==['knight','royal_ghost']
    assert snap['blue']['heroes']==['ice_wizard']
    assert list(snap['blue']['evolution_charge'])==['knight','royal_ghost']
    assert snap['blue']['evolution_charge']=={'knight':1,'royal_ghost':0}
    assert all('evolved' in t and 'is_hero' in t for t in snap['troops'])
    assert sum(t['evolved'] for t in snap['troops'])==1
    assert sum(t['is_hero'] for t in snap['troops'])==1
    g.players['blue'].evolution_charge['knight']=2
    assert snap['blue']['evolution_charge']['knight']==1
    fresh=_game('royal_ghost',evolutions=['royal_ghost'])
    assert fresh.players['blue'].evolution_charge=={'royal_ghost':0}


def t_observation_appends_per_deck_slot_features_and_reset_clears_charge():
    deck=_deck('royal_ghost','ice_wizard')
    env=CREnv(blue_deck=deck,red_deck=deck,blue_evolutions=['royal-ghost'],red_evolutions=['knight'],
              blue_heroes=['ice-wizard'],red_heroes=['musketeer'],decision_freq=1)
    obs,_=env.reset(seed=7)
    assert OBS_DIM==_PREFIX+48 and obs.shape==(OBS_DIM,) and obs.dtype==np.float32
    assert env.observation_space.contains(obs)
    expected=np.zeros((2,8,3),dtype=np.float32)
    expected[0,deck.index('ice_wizard'),2]=1;expected[1,deck.index('musketeer'),2]=1
    np.testing.assert_array_equal(obs[_PREFIX:].reshape(2,8,3),expected)
    np.testing.assert_allclose(obs[:13],[0,1,0,0,1/3,0.5,0,1,0,0.5,0,1,0])
    assert not obs[31:_PREFIX].any()
    _play(env.game,'royal_ghost')
    expected[0,0,1]=0.5
    np.testing.assert_array_equal(env._get_obs()[_PREFIX:].reshape(2,8,3),expected)
    _play(env.game,'royal_ghost');_play(env.game,'knight','red')
    expected[0,0,:2]=[1,1];expected[1,deck.index('knight'),1]=0.5
    np.testing.assert_array_equal(env._get_obs()[_PREFIX:].reshape(2,8,3),expected)
    assert env.observation_space.contains(env._get_obs())
    pd=_play(env.game,'royal_ghost');assert pd.evolved
    obs,*_=env.step(4)
    assert env.observation_space.contains(obs)
    obs,_=env.reset(seed=7)
    assert env.game.players['blue'].evolution_charge=={'royal_ghost':0}
    assert env.game.players['red'].evolution_charge=={'knight':0}
    assert not obs[_PREFIX:].reshape(2,8,3)[:,:,:2].any()
    assert env.observation_space.contains(obs)


def t_default_observation_retains_prefix_and_zero_equipped_suffix():
    env=CREnv();obs,_=env.reset()
    assert obs.shape==(_PREFIX+48,) and not obs[_PREFIX:].any()
    assert env.observation_space.contains(obs)


def _row(name,team='blue',time=0,kind='normal',ability=0):
    return {'card':name,'team':team,'time':time,'tile_x':9,'tile_y':10 if team=='blue' else 21,'card_type':kind,'ability':ability}


def _static_replay(monkeypatch,setup=lambda g:None):
    def make(**kwargs):
        g=Game(**kwargs)
        g.run_to=lambda t:setattr(g,'t',t)
        setup(g)
        return g
    monkeypatch.setattr(R,'Game',make)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)


@pytest.mark.parametrize('suffix,kind',[('-ev1','normal'),('','evo')])
@pytest.mark.parametrize('metadata',[False,True])
def t_replay_evolution_metadata_is_authoritative_or_row_flags_are_fallback(monkeypatch,suffix,kind,metadata):
    _static_replay(monkeypatch)
    plays=[_row('giant-snowball'+suffix,tm,n*20,kind) for n in range(6) for tm in ('blue','red')]
    outcome={'result':'D','tc':0,'oc':0}
    if metadata:outcome['b_evo']=set()
    original=deepcopy((plays,outcome))
    g,_=R.replay_battle('evo-eligibility',plays,outcome)
    assert (plays,outcome)==original
    for tm in ('blue','red'):
        expected=[False]*6 if metadata and tm=='blue' else [False,False,True]*2
        assert [p.evolved for p in g.pending if p.team==tm]==expected


@pytest.mark.parametrize('suffix,kind',[('-hero','normal'),('','hero')])
@pytest.mark.parametrize('metadata',[False,True])
def t_replay_hero_metadata_is_authoritative_or_row_flags_are_fallback(monkeypatch,suffix,kind,metadata):
    _static_replay(monkeypatch)
    plays=[_row('ice-wizard'+suffix,tm,kind=kind) for tm in ('blue','red')]
    outcome={'result':'D','tc':0,'oc':0}
    if metadata:outcome['r_hero']=set()
    g,_=R.replay_battle('hero-eligibility',plays,outcome)
    assert [p.hero for p in g.pending]==[True,not metadata]


def t_replay_equipped_metadata_enables_unmarked_rows_and_supported_snapshot_slots(monkeypatch):
    _static_replay(monkeypatch)
    deck=_deck('royal_ghost','ice_wizard')
    outcome={'result':'D','tc':0,'oc':0,'b_deck':deck,'r_deck':deck,
             'b_evo':{'royal_ghost','giant','unknown_card'},'r_evo':set(),
             'b_hero':{'ice_wizard','arrows','unknown_card'},'r_hero':set()}
    plays=[_row('royal_ghost',time=n*20) for n in range(3)]+[_row('ice_wizard',time=60)]
    g,_=R.replay_battle('slots',plays,outcome)
    assert [p.evolved for p in g.pending]==[False,False,True,False]
    assert g.pending[-1].hero
    g.replay.snap(g);snap=g.replay.snaps[-1]['blue']
    assert snap['evolutions']==['royal_ghost'] and snap['heroes']==['ice_wizard']


@pytest.mark.parametrize('skip_first',[False,True])
def t_replay_recovery_reuses_record_variant_and_skipped_rows_still_count(monkeypatch,skip_first):
    calls=[];charges=[]
    def setup(g):
        original=g.play_card
        def play(tm,name,x,y,**flags):
            calls.append((g.t,flags['evolved']))
            g.replay.snap(g)
            charges.append((g.t,g.replay.snaps[-1][tm]['evolution_charge'][name]))
            if g.t==0 and (skip_first or (x,y)==(9,10)):return False,'invalid position'
            return original(tm,name,x,y,**flags)
        g.play_card=play
    _static_replay(monkeypatch,setup)
    plays=[_row('royal-ghost-ev1',time=n*20,kind='evo') for n in range(3)]
    plays.insert(1,_row('ability-royal-ghost',time=10,ability=1))
    outcome={'result':'D','tc':0,'oc':0,'b_evo':{'royal_ghost'}}
    g,info=R.replay_battle('recovery',plays,outcome)
    assert all(not evo for t,evo in calls if t==0)
    assert all(charge=={0:1,1:2,2:0}[t] for t,charge in charges)
    assert g.players['blue'].evolution_charge=={'royal_ghost':0}
    assert [p.evolved for p in g.pending]==([False,True] if skip_first else [False,False,True])
    assert info['placement']=={'attempted':3,'rejected':1,'invalid':1,'relocated':int(not skip_first),'skipped':int(skip_first)}


def t_replay_mirror_never_increments_the_copied_cards_record_counter(monkeypatch):
    _static_replay(monkeypatch)
    plays=[_row(c,time=n*20) for n,c in enumerate(['royal-ghost-ev1','mirror','royal-ghost-ev1','mirror','royal-ghost-ev1'])]
    g,_=R.replay_battle('mirror',plays,{'result':'D','tc':0,'oc':0})
    assert [p.evolved for p in g.pending]==[False,False,False,False,True]


@pytest.mark.parametrize('validator',[validate_deck,lambda deck,**kw:Player('blue',deck=deck,**kw)])
def t_public_and_player_slot_validation_agree(validator):
    deck=_deck('royal_giant','golden_knight','ice_wizard','bomber')
    with pytest.raises(AssertionError,match='hero and evolution'):
        validator(deck,heroes=['royal-giant'],evolutions=['royal_giant'])
    with pytest.raises(AssertionError,match='hero/champion'):
        validator(deck,heroes=['ice-wizard','knight'])
    with pytest.raises(AssertionError,match='evolution slots'):
        validator(deck,evolutions=['royal_giant','knight','bomber'])
    validator(deck,heroes=['ice-wizard'],evolutions=['knight','bomber'])


def t_public_validation_preserves_legacy_slot_only_contract():
    deck=['knight','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army','freeze']
    assert validate_deck(deck,heroes={'wizard'},evolutions={'knight'})
    with pytest.raises(AssertionError,match='hero and evolution'):
        validate_deck(deck,heroes={'wizard'},evolutions={'wizard'})


@pytest.mark.parametrize('team',['blue','red'])
def t_hyphenated_deck_and_play_names_share_membership_levels_and_charge(team):
    deck=_deck('royal-ghost','ice-wizard')
    config={'deck':deck,'evolutions':['royal_ghost'],'heroes':['ice_wizard'],
            'card_levels':{'royal-ghost':13,'ice-wizard':12},'drag_del':0}
    g=Game(p1=config,p2=config);p=g.players[team]
    assert p.deck.all[:2]==['royal_ghost','ice_wizard']
    for name,evolved in [('royal-ghost',False),('royal_ghost',False),('royal-ghost',True)]:
        _ready(g,'royal_ghost',team)
        assert p.deck.can_play(name)
        assert g.play_card(team,name,9,10 if team=='blue' else 21)==(True,'ok')
        assert g.pending[-1].card=='royal_ghost' and g.pending[-1].evolved is evolved
        _flush(g)
        assert p.troops[-1].lvl==13
    assert p.evolution_charge=={'royal_ghost':0}
    _ready(g,'ice_wizard',team)
    assert g.play_card(team,'ice-wizard',9,10 if team=='blue' else 21)==(True,'ok')
    _flush(g)
    assert p.troops[-1].is_hero and p.troops[-1].lvl==12
    direct=Deck(deck);direct.hand=['royal_ghost']
    assert direct.play('royal-ghost',0)


@pytest.mark.parametrize('team',['blue','red'])
def t_normal_third_royal_ghost_reveal_spawns_two_souldiers(team):
    g=quiet(_game('royal_ghost',evolutions=['royal_ghost']));p=g.players[team]
    for _ in range(3):_play(g,'royal_ghost',team);_flush(g)
    ghosts=list(p.troops)
    for ghost in ghosts:ghost.spd=0
    g.run(1.1)
    assert all(has(ghost,'invisible') for ghost in ghosts)
    target=Dummy(g._opp(team),10,10 if team=='blue' else 21,hp=50000,dmg=0,spd=0)
    g.deploy(target.team,target)
    for ghost in ghosts:g._do_attack(ghost,target)
    assert not any(has(ghost,'invisible') for ghost in ghosts)
    g.tick()
    souls=[tr for tr in p.troops if tr.name=='Souldier']
    assert len(souls)==2 and all(tr.alive and tr.hp>0 for tr in souls)
    assert [getattr(tr,'evolved',False) for tr in ghosts]==[False,False,True]
    assert p.evolution_charge=={'royal_ghost':0}


def t_nonempty_observation_prefix_matches_original_indices():
    env=CREnv(blue_deck=_deck('royal-ghost','ice-wizard'),blue_evolutions=['royal_ghost'],blue_heroes=['ice_wizard'])
    env.reset();g=env.game;g.t=151;g.phase='overtime'
    g.players['blue'].elixir=7;g.players['blue'].crowns=1;g.players['blue'].deck.nxt_cd=0.6
    g.players['red'].elixir=3;g.players['red'].deck.hand=g.players['red'].deck.hand[:3]
    for i,tw in enumerate(g.arena.towers):tw.hp=tw.max_hp*(i+1)/6;tw.active=i%2==0
    blue=Dummy('blue',4.5,10.5,hp=900,spd=1.8);blue.hp=450;blue.transport='Air'
    red=Dummy('red',13.5,21.5,hp=600,spd=0);red.hp=200;red.is_building=True
    dead=Dummy('blue',1,1);dead.alive=False
    g.players['blue'].troops=[dead,blue];g.players['red'].troops=[red]
    expected=np.zeros(_PREFIX,dtype=np.float32)
    expected[:13]=[151/300,0,1,0,2/3,0.7,1/3,1,0.3,0.3,0,0.75,0]
    expected[13:31]=[value for tw in g.arena.towers for value in [float(tw.alive),float(tw.active),tw.hp/tw.max_hp]]
    expected[31:37]=[4.5/18,10.5/32,0.5,1.8/6,1,0]
    expected[211:217]=[13.5/18,21.5/32,1/3,0,0,1]
    obs=env._get_obs()
    np.testing.assert_array_equal(obs[:_PREFIX],expected)
    assert env.observation_space.contains(obs)


@pytest.mark.parametrize('authoritative',['both','hero','evo','neither'])
@pytest.mark.parametrize('team',['blue','red'])
def t_replay_overlap_resolution_matches_playback_and_snapshot(monkeypatch,authoritative,team):
    _static_replay(monkeypatch)
    prefix='b' if team=='blue' else 'r'
    outcome={'result':'D','tc':0,'oc':0}
    if authoritative in ('both','hero'):outcome[prefix+'_hero']={'knight'}
    if authoritative in ('both','evo'):outcome[prefix+'_evo']={'knight'}
    plays=[_row('knight-hero',team,time=n*20,kind='evo') for n in range(3)]
    if authoritative=='both':
        with pytest.raises(ValueError,match='authoritative hero/evolution overlap'):R.replay_battle('overlap',plays,outcome)
        return
    g,_=R.replay_battle('overlap',plays,outcome);p=g.players[team]
    hero=authoritative!='evo'
    assert p.heroes==({'knight'} if hero else set()) and p.evolutions==(set() if hero else {'knight'})
    assert [pd.hero for pd in g.pending]==[hero]*3
    assert [pd.evolved for pd in g.pending]==([False]*3 if hero else [False,False,True])
    g.replay.snap(g);snap=g.replay.snaps[-1][team]
    assert set(snap['heroes'])==p.heroes and set(snap['evolutions'])==p.evolutions


@pytest.mark.parametrize('variant,names',[('hero',['ice_wizard','knight','musketeer']),('evo',['royal_ghost','knight','bomber'])])
@pytest.mark.parametrize('authoritative',[False,True])
def t_replay_overflow_is_explicit_for_metadata_and_consistent_for_fallback(monkeypatch,variant,names,authoritative):
    _static_replay(monkeypatch)
    deck=_deck(*names)
    outcome={'result':'D','tc':0,'oc':0,'b_deck':deck,'r_deck':deck}
    if authoritative:outcome['b_'+variant]=set(names)
    plays=[_row(name,time=n*20,kind=variant) for n in range(3) for name in names]
    if authoritative:
        with pytest.raises(ValueError,match='authoritative .*slots exceed'):R.replay_battle('overflow',plays,outcome)
        return
    g,_=R.replay_battle('overflow',plays,outcome);p=g.players['blue']
    assert (p.heroes if variant=='hero' else p.evolutions)==set(names[:2])
    for pd in g.pending:
        if pd.card==names[2]:assert not pd.hero and not pd.evolved
        elif variant=='hero':assert pd.hero and not pd.evolved
    if variant=='evo':assert not p.heroes and set(p.evolution_charge)==set(names[:2])


def t_replay_authoritative_heroes_do_not_silently_exceed_champion_room(monkeypatch):
    _static_replay(monkeypatch)
    deck=_deck('golden_knight','ice_wizard','knight')
    outcome={'result':'D','tc':0,'oc':0,'b_deck':deck,'r_deck':deck,'b_hero':{'ice_wizard','knight'}}
    with pytest.raises(ValueError,match='authoritative hero/champion slots exceed'):
        R.replay_battle('champion-room',[_row('ice_wizard')],outcome)


def t_replay_unsupported_and_out_of_deck_variants_are_not_playback_eligible(monkeypatch):
    _static_replay(monkeypatch)
    deck=_deck('royal_ghost','ice_wizard')
    outcome={'result':'D','tc':0,'oc':0,'b_deck':deck,'r_deck':deck,
             'b_hero':{'ice_wizard','arrows','unknown_card'},'b_evo':{'royal_ghost','bomber','giant'}}
    plays=[_row('arrows'),_row('giant'),_row('bomber'),_row('ice_wizard')]
    g,_=R.replay_battle('supported',plays,outcome)
    assert g.players['blue'].heroes=={'ice_wizard'} and g.players['blue'].evolutions=={'royal_ghost'}
    assert [(pd.card,pd.hero,pd.evolved) for pd in g.pending]==[
        ('arrows',False,False),('giant',False,False),('bomber',False,False),('ice_wizard',True,False)]


@pytest.mark.parametrize('side',['team','opp'])
@pytest.mark.parametrize('incomplete',['missing','partial','blank'])
def t_meta_v2_incomplete_side_omits_eligibility_and_replay_falls_back(monkeypatch,tmp_path,side,incomplete):
    _static_replay(monkeypatch)
    deck=_deck('royal_ghost','ice_wizard')
    row={'replayTag':'#partial','result':'D'}
    for prefix in ('team','opp'):
        for i,name in enumerate(deck):row[f'{prefix}_card_{i}']=name
    if incomplete=='missing':
        for i in range(8):del row[f'{side}_card_{i}']
    elif incomplete=='partial':row[f'{side}_card_7']=''
    else:row[f'{side}_card_7']='   '
    path=tmp_path/'metadata.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    outcome=R.load_meta_v2(path)['partial'];prefix='b' if side=='team' else 'r';other='r' if prefix=='b' else 'b'
    assert prefix+'_evo' not in outcome and prefix+'_hero' not in outcome
    assert outcome[other+'_evo']==outcome[other+'_hero']==set()
    plays=[_row('royal-ghost-ev1',tm,time=n*20) for n in range(3) for tm in ('blue','red')]
    plays.extend(_row('ice-wizard-hero',tm,time=60) for tm in ('blue','red'))
    g,_=R.replay_battle('partial',plays,outcome)
    tm='blue' if side=='team' else 'red'
    assert g.players[tm].evolutions=={'royal_ghost'} and g.players[tm].heroes=={'ice_wizard'}
    assert not g.players[g._opp(tm)].evolutions and not g.players[g._opp(tm)].heroes
    assert [pd.evolved for pd in g.pending if pd.team==tm]==[False,False,True,False]


def t_meta_v2_complete_side_retains_explicit_empty_and_marked_eligibility(tmp_path):
    row={'replayTag':'#complete','result':'D'}
    deck=_deck('royal_ghost','ice_wizard')
    for prefix in ('team','opp'):
        for i,name in enumerate(deck):row[f'{prefix}_card_{i}']=name
    row['team_card_0']='royal-ghost-ev1';row['team_card_1']='ice-wizard-hero'
    path=tmp_path/'metadata.csv'
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row));writer.writeheader();writer.writerow(row)
    outcome=R.load_meta_v2(path)['complete']
    assert outcome['b_evo']=={'royal_ghost'} and outcome['b_hero']=={'ice_wizard'}
    assert outcome['r_evo']==outcome['r_hero']==set()
