import json
import os
import pickle
import sys
from types import SimpleNamespace
import pytest
from sim import calib as C

@pytest.fixture
def baseline(tmp_path,monkeypatch):
    names=('sim/game.py','sim/units.py','sim/knobs.py','sim/cards.py','sim/replay.py','sim/calib.py',
           'data/cards.json','data/aliases.json','battles.csv','placements.csv')
    for name in names:
        path=tmp_path/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_text(name+'\n')
    monkeypatch.setattr(C,'_BASE',str(tmp_path))
    monkeypatch.setattr(C,'META',str(tmp_path/'battles.csv'))
    monkeypatch.setattr(C,'WORK',str(tmp_path/'placements.csv'))
    monkeypatch.setattr(C,'CACHE',str(tmp_path/'parsed.pkl'))
    return tmp_path

def _edit_preserving_mtime(path):
    st=path.stat();content=path.read_bytes()
    path.write_bytes(b'X'+content[1:])
    os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns))
    assert path.stat().st_mtime_ns==st.st_mtime_ns and path.stat().st_size==st.st_size

@pytest.mark.parametrize('name,changed',[
    ('sim/units.py',{'engine'}),('sim/knobs.py',{'engine'}),('sim/game.py',{'engine','parser'}),
    ('sim/replay.py',{'engine','parser'}),('sim/cards.py',{'engine','parser'}),('sim/calib.py',{'engine','parser'}),
    ('data/cards.json',{'cards'}),('data/aliases.json',{'cards'}),('battles.csv',{'eval'}),('placements.csv',{'eval'})])
def t_content_fingerprints_cover_baseline_changes(baseline,name,changed):
    before=C.fingerprints()
    assert before==C.fingerprints() and all(len(v)==32 for v in before.values())
    _edit_preserving_mtime(baseline/name)
    after=C.fingerprints()
    assert {k for k in before if before[k]!=after[k]}==changed

def t_fingerprints_ignore_mtime_only_changes(baseline):
    before=C.fingerprints()
    for path in baseline.rglob('*'):
        if path.is_file():os.utime(path,ns=(1_000_000_000,1_000_000_000))
    assert C.fingerprints()==before

@pytest.mark.parametrize('changed',['sim/replay.py','sim/cards.py','sim/game.py','sim/calib.py',
                                    'data/cards.json','data/aliases.json','battles.csv','placements.csv'])
def t_parsed_cache_invalidates_on_content_not_mtime(baseline,monkeypatch,changed):
    calls=[]
    def meta(path):
        assert path==C.META
        calls.append('meta')
        return {'kept':{'version':len(calls)},'modifier':{'modifier':True},'unplaced':{}}
    def workers(path,ids,outcomes):
        assert path==C.WORK and ids==set(outcomes)
        calls.append('workers')
        return {'kept':[],'modifier':[]},{'kept':'player'}
    monkeypatch.setattr(C.R,'load_meta_v2',meta)
    monkeypatch.setattr(C.R,'load_worker_rows',workers)
    first=C.inputs()
    assert C.inputs()==first and calls==['meta','workers']
    assert first[3]==['kept']
    _edit_preserving_mtime(baseline/changed)
    second=C.inputs()
    assert second!=first and second[3]==first[3]
    assert C.inputs()==second and calls==['meta','workers','meta','workers']
    with open(C.CACHE,'rb') as f:cached=pickle.load(f)
    assert cached['fingerprints']=={k:v for k,v in C.fingerprints().items() if k!='engine'}

def t_legacy_parsed_cache_is_rebuilt(baseline,monkeypatch):
    with open(C.CACHE,'wb') as f:pickle.dump({'st':(os.path.getmtime(C.META),os.path.getmtime(C.WORK)),'data':'stale'},f)
    outcomes={'battle':{}}
    monkeypatch.setattr(C.R,'load_meta_v2',lambda path:outcomes)
    monkeypatch.setattr(C.R,'load_worker_rows',lambda *args:({'battle':[]},{}))
    assert C.inputs()==(outcomes,{'battle':[]},{},['battle'])

@pytest.mark.parametrize('kind',['legacy','engine','cards','parser','eval','mixed'])
def t_incompatible_logs_fail_before_parsing_or_starting_workers(baseline,monkeypatch,kind):
    fp=C.fingerprints();row={'params':{},'fingerprints':dict(fp)}
    if kind=='legacy':row.pop('fingerprints')
    else:row['fingerprints']['engine' if kind=='mixed' else kind]='stale'
    rows=[{'params':{},'fingerprints':fp},row] if kind=='mixed' else [row]
    log=baseline/'calib.jsonl';text=''.join(json.dumps(r)+'\n' for r in rows);log.write_text(text)
    monkeypatch.setattr(C,'inputs',lambda *args:pytest.fail('incompatible log must fail before parsing'))
    monkeypatch.setattr(C,'Pool',lambda *args:pytest.fail('incompatible log must fail before workers start'))
    monkeypatch.setattr(sys,'argv',['sim.calib','--report','--log',str(log)])
    for call in (lambda:C.Evaluator(1,log),lambda:C.report(log),C.main):
        with pytest.raises(ValueError,match=rf'Incompatible calibration log .*line {len(rows)}:.*--log'):
            call()
    assert log.read_text()==text

def _info():
    return {'hp_err':0.25,'tower_state':1,'aim':(2,1),'win_match':True,'crown_exact':True,'crown_close':True,'premature':False,
            'placement':{'attempted':3,'rejected':2,'invalid':1,'relocated':1,'skipped':1}}

def t_matching_log_reuses_results_and_records_provenance(baseline,monkeypatch,capsys):
    calls=[];fp=C.fingerprints();log=baseline/'calib.jsonl'
    def inputs(stamp):
        assert stamp==fp
        return {'a':{},'b':{}},{'a':[],'b':[]},{},['a','b']
    def imap(fn,args,chunksize):
        calls.append(args)
        assert fn is C._run and chunksize==4 and [a[0] for a in args]==['a','b']
        return iter([(False,_info()),(True,_info())])
    monkeypatch.setattr(C,'inputs',inputs)
    monkeypatch.setattr(C,'Pool',lambda jobs:SimpleNamespace(imap_unordered=imap))
    ev=C.Evaluator(1,log);first=ev({})
    assert ev({}) is first and ev.new==1
    assert first['fingerprints']==fp and first['train']['placement']==_info()['placement']
    assert first['train']['obj']==0.75 and first['train']['n']==1
    text=log.read_text();assert json.loads(text)==first
    again=C.Evaluator(1,log)
    assert again({})==first and again.new==0 and len(calls)==1 and log.read_text()==text
    monkeypatch.setattr(C,'inputs',lambda *args:pytest.fail('report must not parse inputs'))
    monkeypatch.setattr(C,'Pool',lambda *args:pytest.fail('report must not start workers'))
    monkeypatch.setattr(sys,'argv',['sim.calib','--report','--log',str(log)])
    C.main()
    text=capsys.readouterr().out
    assert '1 evaluations; base:' in text
    assert 'placements 2/3 rejected (1 invalid), 1 relocated, 1 skipped' in text

def t_calibration_summarizes_recovery_without_changing_objective():
    infos=[_info(),_info()];summary=C.summarize(infos)
    assert summary['n']==2 and summary['obj']==0.75
    assert summary['placement']=={k:v*2 for k,v in infos[0]['placement'].items()}

def t_calibration_cli_accepts_separate_log(monkeypatch,tmp_path):
    calls=[];log=str(tmp_path/'newBaseline.jsonl')
    monkeypatch.setattr(sys,'argv',['sim.calib','--report','--log',log])
    monkeypatch.setattr(C,'report',calls.append)
    C.main()
    assert calls==[log]

@pytest.mark.parametrize('args,jobs',[([],2),(['--jobs','1'],1)])
def t_calibration_cli_uses_lightweight_worker_default(monkeypatch,tmp_path,args,jobs):
    calls=[];ev=object();log=str(tmp_path/'newBaseline.jsonl')
    def evaluator(n,path):
        calls.append((n,path))
        return ev
    def search(worker,budget,eps,fixed):
        assert worker is ev and budget==200 and eps==0.003 and fixed==set()
    monkeypatch.setattr(C,'Evaluator',evaluator)
    monkeypatch.setattr(C,'search',search)
    monkeypatch.setattr(sys,'argv',['sim.calib','--log',log,*args])
    C.main()
    assert calls==[(jobs,log)]
