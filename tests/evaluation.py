import hashlib
import importlib
import logging
from collections.abc import Mapping
from copy import deepcopy
import pytest

_START=1000
_END=2000
_MODES={'Ranked','Ladder'}
_LABELS=('result','winner','team_crowns','opp_crowns','team_king_hp','team_princess_hp_0','team_princess_hp_1',
         'opp_king_hp','opp_princess_hp_0','opp_princess_hp_1')
_FIELDS={'replayTag','one_v_one','has_replay','battle_ts','gameMode_name','team_tags','opponent_tags'}
for _side in ('team','opp'):
    _FIELDS.update(f'{_side}_{name}' for name in ('king_lvl','tower_lvl','tower_troop'))
    _FIELDS.update(f'{_side}_card_{i}{suffix}' for i in range(8) for suffix in ('','_lvl'))

class _Audited(Mapping):
    def __init__(self,row,allowed):self.row=row;self.allowed=allowed;self.reads=[]
    def __getitem__(self,key):
        assert key in self.allowed,f'Non-selection field read: {key}'
        self.reads.append(key)
        return self.row[key]
    def __iter__(self):raise AssertionError('Selection must not copy or inspect whole rows')
    def __len__(self):return len(self.row)


def _row(bid='record',**values):
    row={'replayTag':bid,'one_v_one':'True','has_replay':'True','battle_ts':'1500.0','gameMode_name':'Ranked',
         'team_tags':'#BLUE','opponent_tags':'#RED','result':'W','winner':'blue','team_crowns':'3','opp_crowns':'0'}
    for side in ('team','opp'):
        row.update({f'{side}_king_lvl':'11.0',f'{side}_tower_troop':'tower_princess'})
        for i in range(8):row.update({f'{side}_card_{i}':f'unknown-card-{i}',f'{side}_card_{i}_lvl':'11.0'})
        for name in ('king_hp','princess_hp_0','princess_hp_1'):row[f'{side}_{name}']='0'
    row.update(values)
    return row


def _plays():return [{'team':'blue','time':0},{'team':'red','time':6000}]


def t_selection_module_exists():
    assert importlib.util.find_spec('sim.evaluation') is not None,'Outcome-independent selection module is missing'


@pytest.fixture
def selection():return importlib.import_module('sim.evaluation')


@pytest.mark.parametrize('value',['L','D','swapped-winner',0,3,999999,-1,'',None,'invalid','nan','inf'])
def t_outcomes_cannot_change_eligibility_ranking_or_selection(selection,value):
    rows=[_row(f'battle-{i}') for i in range(32)]
    baseline=selection.ranked_candidates(rows,91,set(),_START,_END,_MODES)
    placements={row['replayTag']:_plays() for row in rows}
    expected=selection.choose_replays(baseline,placements,12)
    changed=deepcopy(rows)
    for row in changed:
        for field in _LABELS:row[field]=value
    assert all(selection.metadata_eligible(row,_START,_END,_MODES) for row in changed)
    ranked=selection.ranked_candidates(changed,91,set(),_START,_END,_MODES)
    assert ranked==baseline
    assert selection.choose_replays(ranked,placements,12)==expected


def t_absent_labels_and_unrecognized_cards_are_not_filters(selection):
    rows=[_row(f'battle-{i}') for i in range(12)]
    expected=selection.ranked_candidates(rows,12,set(),_START,_END,_MODES)
    for row in rows:
        for field in _LABELS:row.pop(field,None)
        row['team_card_0']='a-new-unsupported-card'
        row['opp_tower_troop']='a-new-tower-troop'
    assert selection.ranked_candidates(rows,12,set(),_START,_END,_MODES)==expected
    placements={bid:[dict(p,card='_invalid',ability=1) for p in _plays()] for bid in expected}
    assert selection.choose_replays(expected,placements,len(expected))==expected


def t_reversing_actual_winners_and_crowns_does_not_change_selected_ids(selection):
    rows=[_row(f'battle-{i}',result='W' if i%2 else 'L',winner='blue' if i%2 else 'red',
               team_crowns=3 if i%2 else 0,opp_crowns=0 if i%2 else 3) for i in range(32)]
    ranked=selection.ranked_candidates(rows,19,set(),_START,_END,_MODES)
    placements={bid:_plays() for bid in ranked}
    selected=selection.choose_replays(ranked,placements,10)
    for row in rows:
        row['result']='L' if row['result']=='W' else 'W'
        row['winner']='red' if row['winner']=='blue' else 'blue'
        row['team_crowns'],row['opp_crowns']=row['opp_crowns'],row['team_crowns']
        row['team_king_hp'],row['opp_king_hp']=999999,0
    changed=selection.ranked_candidates(reversed(rows),19,set(),_START,_END,_MODES)
    assert changed==ranked and selection.choose_replays(changed,placements,10)==selected


def t_ranking_is_order_independent_hash_monotonic_and_seeded(selection):
    rows=[_row(f'battle-{i}') for i in range(64)]
    ranked=selection.ranked_candidates(iter(rows),23,set(),_START,_END,_MODES)
    assert isinstance(ranked,list) and len(ranked)==64 and len(set(ranked))==64
    assert ranked==selection.ranked_candidates(reversed(rows),23,set(),_START,_END,_MODES)
    digests=[hashlib.sha256(f'23:{bid}'.encode()).digest() for bid in ranked]
    assert all(a<b for a,b in zip(digests,digests[1:]))
    assert ranked!=selection.ranked_candidates(rows,24,set(),_START,_END,_MODES)
    excluded=set(ranked[:17])
    rest=selection.ranked_candidates(rows,23,{'#'+bid for bid in excluded},_START,_END,_MODES)
    assert rest==ranked[17:] and excluded.isdisjoint(rest)


def t_ranking_does_not_lose_later_usable_replays(selection):
    rows=(_row(f'battle-{i}') for i in range(4100))
    ranked=selection.ranked_candidates(rows,0,set(),_START,_END,_MODES)
    assert len(ranked)==4100
    placements={ranked[-1]:_plays()}
    assert selection.choose_replays(ranked,placements,1)==ranked[-1:]


@pytest.mark.parametrize('excluded',[set(),{'record'}])
@pytest.mark.parametrize('different',[False,True])
def t_duplicate_ids_are_strict_even_when_excluded_or_metadata_disagrees(selection,excluded,different):
    rows=[_row('record'),_row('#record',gameMode_name='invalid' if different else 'Ranked',result='L')]
    for ordered in (rows,list(reversed(rows))):
        with pytest.raises(ValueError,match='[Dd]uplicate'):
            selection.ranked_candidates(ordered,0,excluded,_START,_END,_MODES)


@pytest.mark.parametrize('field,value',[
    ('replayTag',''),('replayTag',None),('one_v_one','False'),('has_replay','False'),('one_v_one','invalid'),
    ('battle_ts','invalid'),('battle_ts','nan'),('battle_ts','inf'),('battle_ts',999),('battle_ts',2000),
    ('gameMode_name','C.H.A.O.S Infinite Elixir'),('gameMode_name',''),('team_tags',''),('opponent_tags',None),
    ('team_tags','BLUE,OTHER'),('opponent_tags','#BLUE'),('team_king_lvl',0),('opp_king_lvl',17),
    ('team_king_lvl','nan'),('opp_king_lvl','11.5'),('team_tower_lvl','invalid'),('opp_tower_lvl',0),
    ('team_tower_troop',''),('opp_tower_troop',None),('team_card_0',''),('opp_card_7',None),
    ('team_card_0_lvl',''),('opp_card_7_lvl','inf'),('team_card_7_lvl',0),('opp_card_0_lvl','11.1'),
])
def t_metadata_failures_are_excluded(selection,field,value):
    row=_row(**{field:value})
    assert not selection.metadata_eligible(row,_START,_END,_MODES)
    assert selection.ranked_candidates([row],0,set(),_START,_END,_MODES)==[]


@pytest.mark.parametrize('field',['replayTag','one_v_one','has_replay','battle_ts','gameMode_name','team_tags',
                                  'opponent_tags','team_king_lvl','opp_tower_troop','team_card_0','opp_card_7_lvl'])
def t_missing_required_metadata_fails(selection,field):
    row=_row();row.pop(field)
    assert not selection.metadata_eligible(row,_START,_END,_MODES)


@pytest.mark.parametrize('timestamp',[1000,1000.25,'1999.999'])
def t_metadata_time_bounds_are_start_inclusive_end_exclusive(selection,timestamp):
    assert selection.metadata_eligible(_row(battle_ts=timestamp),_START,_END,_MODES)


def t_optional_declared_tower_levels_and_inputs_are_preserved(selection):
    rows=[_row(team_tower_lvl='12.0',opp_tower_lvl=16)]
    original=deepcopy(rows)
    assert selection.metadata_eligible(rows[0],_START,_END,_MODES)
    ranked=selection.ranked_candidates(rows,11,set(),_START,_END,_MODES)
    placements={ranked[0]:_plays()};before=deepcopy(placements)
    assert selection.choose_replays(ranked,placements,1)==ranked
    assert rows==original and placements==before


@pytest.mark.parametrize('time',[None,'','bad','nan','inf',float('nan'),float('inf'),-1,6000.1,True])
def t_invalid_source_timestamps_cannot_supply_a_missing_team(selection,time):
    placements={'record':[{'team':'blue','time':0},{'team':'red','time':time}]}
    with pytest.raises(ValueError,match='[Rr]equested|[Nn]eed|[Ii]nsufficient'):
        selection.choose_replays(['record'],placements,1)


@pytest.mark.parametrize('bad',[{'team':'red'}, {'team':'green','time':1}, {'time':1},
                              {'team':'red','time':0,'time_valid':False}])
def t_missing_team_or_failed_source_parse_cannot_be_fabricated(selection,bad):
    with pytest.raises(ValueError):
        selection.choose_replays(['record'],{'record':[{'team':'blue','time':0},bad]},1)


def t_choose_returns_first_n_usable_ids_and_requires_exact_count(selection):
    candidates=['missing','empty','blue-only','bad-red','usable-1','usable-2','unused']
    placements={'empty':[],'blue-only':[{'team':'blue','time':1}],
                'bad-red':[{'team':'blue','time':0},{'team':'red','time':'bad'}],
                'usable-1':_plays()+[{'team':'red','time':'bad'}],'usable-2':list(reversed(_plays())),
                'unused':_plays()}
    assert selection.choose_replays(candidates,placements,2)==['usable-1','usable-2']
    assert selection.choose_replays(candidates,placements,0)==[]
    with pytest.raises(ValueError):selection.choose_replays(candidates,placements,4)
    with pytest.raises(ValueError):selection.choose_replays(candidates,placements,-1)
    with pytest.raises(ValueError):selection.choose_replays(candidates,placements,1.5)
    with pytest.raises(ValueError,match='[Dd]uplicate'):
        selection.choose_replays(['usable-1','usable-1'],placements,2)


def t_field_access_and_audit_log_prove_no_outcome_data_used(selection,caplog):
    rows=[_Audited(_row(f'battle-{i}',result='SECRET-LABEL'),_FIELDS) for i in range(8)]
    with caplog.at_level(logging.INFO,logger='sim.evaluation'):
        ranked=selection.ranked_candidates(iter(rows),7,set(),_START,_END,_MODES)
        placements={bid:[_Audited(dict(p,result='SECRET-LABEL',card='SECRET-CARD'),{'time','team','time_valid'})
                         for p in _plays()] for bid in ranked}
        assert selection.choose_replays(ranked,placements,3)==ranked[:3]
    assert all(row.reads for row in rows)
    assert all(set(row.reads)<=_FIELDS for row in rows)
    records=[r for r in caplog.records if r.name=='sim.evaluation']
    assert len(records)>=2 and all(r.outcome_data_used is False for r in records)
    assert 'outcome_data_used=False' in caplog.text
    assert 'SECRET-LABEL' not in caplog.text and 'SECRET-CARD' not in caplog.text
