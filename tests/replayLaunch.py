import csv
import os
from pathlib import Path
import subprocess
import sys


DECK=['knight','archers','fireball','giant','musketeer','valkyrie','bomber','arrows']


def write_csv(path,rows):
    with path.open('w') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def t_shipped_replay_cli_runs_in_fresh_processes_without_outcome_driven_levels(tmp_path):
    row={'replayTag':'cli-case','result':'D','team_crowns':0,'opp_crowns':0,'gameMode_name':'Ranked',
         'team_tags':'BLUE','opponent_tags':'RED'}
    for side in ('team','opp'):
        row[side+'_king_lvl']=11;row[side+'_tower_troop']='tower_princess'
        for i,card in enumerate(DECK):row[f'{side}_card_{i}']=card;row[f'{side}_card_{i}_lvl']=11
        for k in ('king_hp','princess_hp_0','princess_hp_1'):row[side+'_'+k]=999999
    metadata=tmp_path/'battles.csv';write_csv(metadata,[row])
    placements=tmp_path/'placements.csv'
    write_csv(placements,[{'battle_id':'cli-case','player_id':'BLUE','card':'knight','time':201,
                          'team':team,'side':side,'x':4500,'y':y,'ability':0,'card_type':'normal'}
                         for team,side,y in [('blue','t',10500),('red','o',21500)]])
    args=[sys.executable,'-m','sim.replay','--meta',str(metadata),'--workers',str(placements),'--battle','cli-case','--jobs','1','--verbose']
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1')
    runs=[subprocess.run(args,cwd=Path(__file__).resolve().parents[1],env=env,capture_output=True,text=True,timeout=60) for _ in range(2)]
    assert all(r.returncode==0 for r in runs),[(r.stdout,r.stderr) for r in runs]
    assert runs[0].stdout==runs[1].stdout
    text=runs[0].stdout
    assert 'Loaded 1 battles with card levels' in text and 'Running 1 battles' in text
    assert 'lvls=b11/r11' in text
    assert 'Recorded placements: 2 attempted, 0 rejected' in text
    assert 'Winner match:' in text and '/1 (' in text
