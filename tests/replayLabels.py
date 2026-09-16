import csv
from copy import deepcopy

import pytest

from sim import replay as R
from sim.cards import card
from sim.game import Game


DECK=['knight','archers','fireball','giant','musketeer','valkyrie','bomber','arrows']


def metadata(tmp_path,**values):
    row={'replayTag':'record','result':'W','team_crowns':1,'opp_crowns':0,
         'team_king_lvl':11,'opp_king_lvl':11,'gameMode_name':'Ranked'}
    for side in ('team','opp'):
        row[side+'_tower_troop']='tower_princess'
        for i,name in enumerate(DECK):row[f'{side}_card_{i}']=name;row[f'{side}_card_{i}_lvl']=11
        row.update({side+'_king_hp':4824,side+'_princess_hp_0':3052,side+'_princess_hp_1':1000})
    row.update(values)
    path=tmp_path/'metadata.csv'
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=row);w.writeheader();w.writerow(row)
    return R.load_meta_v2(path)['record']


def play():return {'card':'knight','team':'blue','time':1,'tile_x':4.5,'tile_y':10.5,'ability':0,'card_type':'normal'}


@pytest.mark.parametrize('health',(0,4824,7728,999999,'','invalid','nan','inf',-1))
def t_terminal_health_does_not_change_initial_levels_or_mode(tmp_path,health):
    m=metadata(tmp_path,team_king_hp=health,opp_king_hp=health,team_princess_hp_0=health)
    assert m['b_klvl']==m['r_klvl']==11
    assert not m['modifier']


def t_declared_distinct_king_and_tower_levels_are_preserved(tmp_path):
    m=metadata(tmp_path,team_king_lvl=16,opp_king_lvl=14,team_tower_lvl=11,opp_tower_lvl=12,
               team_king_hp=0,opp_king_hp=0,team_card_0_lvl=13)
    assert (m['b_klvl'],m['b_ttlvl'],m['r_klvl'],m['r_ttlvl'])==(16,11,14,12)
    assert m['b_lvls']['knight']==13


@pytest.mark.parametrize('labels',[
    {'result':'L','team_crowns':0,'opp_crowns':3,'team_king_hp':0,'opp_king_hp':7728},
    {'result':'D','team_crowns':0,'opp_crowns':0,'team_king_hp':999999,'opp_princess_hp_1':999999},
    {'result':'','team_crowns':'','opp_crowns':'','team_king_hp':'','opp_king_hp':''},
    {'result':'invalid','team_crowns':'nan','opp_crowns':'bad','team_king_hp':'inf'},
])
def t_mutating_labels_does_not_change_real_replay_state(tmp_path,monkeypatch,labels):
    monkeypatch.setattr(Game,'END',3)
    clean=metadata(tmp_path);changed=metadata(tmp_path,**labels)
    g,a=R.replay_battle('clean',[play()],clean)
    h,b=R.replay_battle('changed',[play()],changed)
    assert g.log==h.log and g.replay.snaps==h.replay.snaps
    assert [(t.hp,t.max_hp,t.alive) for t in g.arena.towers]==[(t.hp,t.max_hp,t.alive) for t in h.arena.towers]
    assert a['sim_winner']==b['sim_winner'] and a['placement']==b['placement'] and a['end_t']==b['end_t']
    assert g.arena.get_tower('blue','king').max_hp==card('king_tower')['stats']['hitpoints'][10]
    if labels['result'] in ('','invalid'):
        assert b['win_match'] is None and b['crown_exact'] is None and b['crown_close'] is None


@pytest.mark.parametrize('mode',('C.H.A.O.S mode','7x Elixir','Sudden Death Battle'))
def t_mode_exclusion_uses_only_declared_mode(tmp_path,mode):
    m=metadata(tmp_path,gameMode_name=mode,team_king_hp='nan')
    assert m['modifier']


@pytest.mark.parametrize('loader',(R.load_worker_rows,R.load_placements))
@pytest.mark.parametrize('value',('bad','','nan','inf','0.5',None))
def t_worker_timestamp_coercion_cannot_establish_selection_coverage(tmp_path,loader,value):
    from sim.evaluation import choose_replays
    path=tmp_path/'placements.csv'
    rows=[{'battle_id':'b','card':'knight','team':team,'time':v,'x':9000,'y':10000,'tile_x':9,'tile_y':10} for team,v in [('blue',value),('red','20')]]
    with path.open('w') as f:w=csv.DictWriter(f,fieldnames=rows[0]);w.writeheader();w.writerows(rows)
    result=loader(path,{'b'});plays=result[0] if isinstance(result,tuple) else result
    assert plays['b'][0]['time_valid'] is False
    with pytest.raises(ValueError,match='found only 0'):choose_replays(['b'],plays,1)


def t_missing_labels_are_reported_without_dropping_cli_games(tmp_path,monkeypatch,capsys):
    import sys
    m=metadata(tmp_path,result='',team_crowns='',opp_crowns='',team_king_hp='')
    monkeypatch.setattr(Game,'END',1)
    monkeypatch.setattr(R,'load_outcomes',lambda path:{'b':m})
    monkeypatch.setattr(R,'load_placements',lambda path,ids:{'b':[play()]})
    monkeypatch.setattr(sys,'argv',['replay','--jobs','1'])
    R.main();text=capsys.readouterr().out
    assert '0/1' in text and 'Unavailable labels: 1 winners, 1 crown scores' in text and 'lower bounds over all games' in text


def t_labels_remain_available_for_post_simulation_scoring(tmp_path,monkeypatch):
    monkeypatch.setattr(Game,'END',3)
    m=metadata(tmp_path,result='D',team_crowns=0,opp_crowns=0)
    original=deepcopy(m);g,info=R.replay_battle('score',[play()],m)
    assert m==original and g.t==3 and info['actual_winner'] is None
    assert info['win_match'] and info['crown_exact'] and info['hp_err'] is not None
