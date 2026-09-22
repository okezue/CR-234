"""Decision-point states from replayed real games.

Every recorded card play is a labelled decision: the state the acting player saw (train.feats.featurize from that player's side),
the play itself, the recorded outcome of the whole game (ground truth, one label per game) and the simulator's outcome of the same
replay (a biased witness). The extractor wraps the shipped replay path (sim.replay.replay_battle with its patchable Game class) so
the states are exactly those the fidelity judge produces; nothing about the replay changes.

Output: one .npz per run with float16 state matrix X, per-record columns and per-game outcomes, plus a JSON sidecar with hashes.
Usage: PYTHONPATH=. python train/decisionStates.py --data ../cr234-scraping/local/simEval2000-<name> --out local/decisions/<name>.npz --jobs 2
"""
import argparse
import hashlib
import json
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from sim import replay as R
from sim.cards import key
from train.feats import FEAT_DIM,featurize

COLS=('bid','idx','t','team','card','x','y','evolved','hero','elixir_rate')


class Recorder(R.Game):
    # snapshots the acting player's view just before the shipped play_card runs, so a rejected or relocated placement is still a
    # decision at its recorded time and tile
    records=None
    def play_card(self,team,card,x,y,evolved=None,hero=None):
        if self.records is not None:
            self.records.append({'t':self.t,'team':team,'card':key(card) or card,'x':x,'y':y,'evolved':bool(evolved),'hero':bool(hero),
                                 'erate':self._erate(),'state':featurize(self,team).astype(np.float16)})
        return super().play_card(team,card,x,y,evolved=evolved,hero=hero)


def extract(bid,plays,outcome,pid=None):
    # decisions of one game and its two outcomes; runs in a worker, so the patched class is restored afterwards
    records=[];old=R.Game;Recorder.records=records
    try:
        R.Game=Recorder;g,info=R.replay_battle(bid,plays,outcome,pid=pid)
    finally:
        R.Game=old;Recorder.records=None
    game={'bid':bid,'actual_winner':info['actual_winner'],'sim_winner':info['sim_winner'],'end_t':info['end_t'],'premature':info['premature'],
          'actual_bc':info['actual_bc'],'actual_rc':info['actual_rc'],'sim_bc':info['sim_bc'],'sim_rc':info['sim_rc'],'n':len(records)}
    return game,records


def _work(args):
    return extract(*args)


def label(game,team):
    # 1 when the acting player won the recorded game; drawn games carry no label
    w=game['actual_winner']
    return None if w is None else float(w==team)


def build(data,out,jobs=2,limit=0):
    paths=[Path(data)/n for n in ('battles.csv','placements.csv')]
    outcomes=R.load_meta_v2(str(paths[0]));placements,pids=R.load_worker_rows(str(paths[1]),set(outcomes),outcomes)
    bids=sorted(b for b in outcomes if b in placements and not outcomes[b].get('modifier'))
    if limit:bids=bids[:limit]
    t0=time.monotonic();games=[];rows=[];states=[]
    with Pool(jobs) as pool:
        for game,recs in pool.imap(_work,[(b,placements[b],outcomes[b],pids.get(b)) for b in bids],chunksize=2):
            games.append(game)
            for i,r in enumerate(recs):
                rows.append((game['bid'],i,r['t'],r['team'],r['card'],r['x'],r['y'],r['evolved'],r['hero'],r['erate']));states.append(r['state'])
    X=np.stack(states) if states else np.zeros((0,FEAT_DIM),np.float16)
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,X=X,**{c:np.array([r[k] for r in rows]) for k,c in enumerate(COLS)})
    side={'data':str(data),'games':len(games),'records':len(rows),'feat_dim':int(X.shape[1]),'seconds':round(time.monotonic()-t0,1),
          'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
          'code':hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path('sim').glob('*.py'))
                                         +[Path('data/cards.json'),Path('train/feats.py')])).hexdigest(),
          'outcomes':games}
    out.with_suffix('.json').write_text(json.dumps(side)+'\n')
    return side


def load(npz):
    d=np.load(npz,allow_pickle=False);side=json.loads(Path(npz).with_suffix('.json').read_text())
    return {c:d[c] for c in COLS},d['X'].astype(np.float32),{g['bid']:g for g in side['outcomes']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--jobs',type=int,default=2);ap.add_argument('--limit',type=int,default=0);a=ap.parse_args()
    side=build(a.data,a.out,a.jobs,a.limit)
    print(json.dumps({k:v for k,v in side.items() if k!='outcomes'},indent=1))


if __name__=='__main__':main()
