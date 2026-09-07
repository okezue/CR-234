from copy import deepcopy
import sys
import pytest
from sim import replay as R
from sim.game import Game

_BLUE=['knight','archers','fireball','zap','valkyrie','baby_dragon','mini_pekka','skeletons']
_RED=['giant','musketeer','arrows','ice_spirit','witch','bomber','hog_rider','goblins']
_OUTCOME={'result':'D','tc':0,'oc':0}

def _play(card='knight',team='blue',x=4.25,y=10.25,time=0,ability=0):
    return {'card':card,'team':team,'tile_x':x,'tile_y':y,'time':time,'ability':ability,'card_type':'normal'}

def _static_game(monkeypatch,setup=lambda g:None):
    def create(**kwargs):
        g=Game(**kwargs)
        g.run_to=lambda t:None
        setup(g)
        return g
    monkeypatch.setattr(R,'Game',create)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)

def _counts(attempted=1,rejected=0,invalid=0,relocated=0,skipped=0):
    return dict(attempted=attempted,rejected=rejected,invalid=invalid,relocated=relocated,skipped=skipped)

@pytest.mark.parametrize('bx,rx',[(13.25,4.25),(4.25,13.25)])
def t_replay_repeatability_and_input_immutability(monkeypatch,bx,rx):
    monkeypatch.setattr(Game,'END',4)
    plays=[_play('giant','blue',bx,10.25),_play('knight','red',rx,22.25,time=20),
           _play('arrows','blue',bx,12.25,time=40),_play('zap','red',rx,20.25,time=40)]
    outcome={**_OUTCOME,'b_deck':list(_BLUE),'r_deck':list(_RED),'b_lvls':{'knight':12},'r_lvls':{'giant':13},
             'b_evo':{'knight'},'r_evo':set(),'b_hero':set(),'r_hero':set(),
             'b_hp':[5000,3000,3000],'r_hp':[5000,3000,3000]}
    original=deepcopy((plays,outcome));runs=[]
    for _ in range(3):
        g,info=R.replay_battle('repeat',plays,outcome,probe=True)
        assert (plays,outcome)==original
        runs.append((g.log,g.replay.snaps,info))
    assert runs[0]==runs[1]==runs[2]
    x=int(18-bx if bx>9 else bx)
    assert any(f'red plays giant at ({x},21)' in line for line in g.log)
    assert g.players['blue'].deck.all==_BLUE and g.players['red'].deck.all==_RED
    assert info['placement']==_counts(attempted=4)

@pytest.mark.parametrize('team',['blue','red'])
@pytest.mark.parametrize('x,side',[(3.25,'left'),(14.25,'right')])
@pytest.mark.parametrize('phase',['regulation','overtime'])
def t_pocket_recovery_cannot_kill_towers(monkeypatch,team,x,side,phase):
    enemy='red' if team=='blue' else 'blue'
    def setup(g):
        g.phase=phase;g.t=181 if phase=='overtime' else 0
        pt=g.arena.get_tower(enemy,'princess',side)
        pt.hp=pt.max_hp*0.69
    _static_game(monkeypatch,setup)
    y=20.25 if team=='blue' else 11.25
    g,info=R.replay_battle('pocket',[_play(team=team,x=x,y=y)],_OUTCOME)
    pt=g.arena.get_tower(enemy,'princess',side)
    assert pt.alive and pt.hp==pt.max_hp*0.69 and not getattr(pt,'down',False)
    assert all(t.alive for t in g.arena.towers)
    assert all(not g.arena.get_tower(tm,'king').active for tm in ('blue','red'))
    assert [p.crowns for p in g.players.values()]==[0,0]
    assert not g.ended and g.winner is None and g.phase==phase
    assert info['placement']==_counts(rejected=1,invalid=1,relocated=1)
    assert [(p.x,p.y) for p in g.pending]==[(int(x)+0.5,14.5 if team=='blue' else 17.5)]

@pytest.mark.parametrize('team',['blue','red'])
@pytest.mark.parametrize('card,pocket',[('knight',False),('fireball',False),('miner',False),('knight',True)])
def t_valid_recorded_placements_are_unchanged(monkeypatch,team,card,pocket):
    x=3.25 if team=='blue' else 14.25
    y=(20.25 if team=='blue' else 11.25) if pocket or card!='knight' else (10.25 if team=='blue' else 21.25)
    def setup(g):
        if pocket:
            pt=g.arena.get_tower(g._opp(team),'princess','left' if x<9 else 'right')
            pt.take_damage(pt.hp)
    _static_game(monkeypatch,setup)
    g,info=R.replay_battle('valid',[_play(card,team,x,y)],_OUTCOME)
    assert [(p.x,p.y) for p in g.pending]==[(int(x)+0.5,int(y)+0.5)]
    assert info['placement']==_counts()
    assert g.players[team].crowns==int(pocket)

@pytest.mark.parametrize('accepted',[(4,10),(4,11),(9,8),None])
def t_recovery_order_and_counters(monkeypatch,accepted):
    calls=[]
    def setup(g):
        g._valid_deploy=lambda team,x,y:(x,y)==accepted
        play=g.play_card
        def record(team,card,x,y,**kwargs):
            calls.append((x,y))
            return play(team,card,x,y,**kwargs)
        g.play_card=record
    _static_game(monkeypatch,setup)
    g,info=R.replay_battle('fallback',[_play()],_OUTCOME)
    expected=[(4.25,10.25),(4,11),(4,9),(5,10),(3,10),(5,11),(3,9),(4,10),(9,8)]
    if accepted==(4,10):expected=expected[:1]
    elif accepted==(4,11):expected=expected[:2]
    assert calls==expected
    rejected=int(accepted!=(4,10))
    assert info['placement']==_counts(rejected=rejected,invalid=rejected,relocated=int(rejected and accepted is not None),skipped=int(accepted is None))
    assert len(g.pending)==int(accepted is not None)

def t_non_position_rejection_is_not_counted_as_invalid(monkeypatch):
    _static_game(monkeypatch)
    g,info=R.replay_battle('mirror',[_play('mirror')],_OUTCOME)
    assert info['placement']==_counts(rejected=1,skipped=1)
    assert not g.pending

def t_placement_population_excludes_abilities_unknown_cards_and_late_rows(monkeypatch):
    def setup(g):
        def run_to(t):
            g.t=t
            if t>=2:g.ended=True
        g.run_to=run_to
    _static_game(monkeypatch,setup)
    plays=[_play('_invalid'),_play('unknown_card'),_play('ability-knight',ability=1),_play(),_play('giant','red',time=40)]
    _,info=R.replay_battle('population',plays,_OUTCOME)
    assert info['placement']==_counts()

def t_replay_summary_reports_recovery_without_excluding_battles(monkeypatch,capsys):
    outcomes={'one':dict(_OUTCOME),'two':dict(_OUTCOME)}
    placements={bid:[] for bid in outcomes}
    counts=[_counts(attempted=3,rejected=2,invalid=1,relocated=1,skipped=1),_counts(attempted=3,rejected=1,invalid=1,relocated=1)]
    def replay(bid,*args,**kwargs):
        return None,{'bid':bid,'win_match':True,'crown_exact':True,'crown_close':True,'premature':False,
                     'hp_err':None,'tower_state':None,'aim':(0,0),'probes':[],'placement':counts.pop(0)}
    monkeypatch.setattr(R,'load_outcomes',lambda path:outcomes)
    monkeypatch.setattr(R,'load_placements',lambda path,ids:placements)
    monkeypatch.setattr(R,'replay_battle',replay)
    monkeypatch.setattr(sys,'argv',['sim.replay','--jobs','1'])
    R.main()
    text=capsys.readouterr().out
    assert 'Winner match: 2/2 (100.0%)' in text
    assert 'Recorded placements: 6 attempted, 3 rejected (2 invalid position); recovery: 2 relocated, 1 skipped' in text
