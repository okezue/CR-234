"""Decision states of the private replay corpus at scale, in battle-time order, for single-trajectory RL (train.singleTraj).

The corpus (battles.parquet, placements.parquet) is filtered to one-versus-one standard-mode games with a replay, minus every id of
the frozen fidelity assessment samples (read from their battles.csv the way local/selectAssessment.py reads its exclusions), ordered
by battle time and cut into shards. Each shard is written as the battles.csv/placements.csv pair the replay judge reads and
extracted through the shipped replay path (train.decisionStates.Recorder), with training-only additions per decision: the opponent's
hand and next card in the replay's reconstructed cycle (hidden from the actor), and optionally a no-play rollout of the simulator
from the decision state (the world model's forecast of the current board: a copy of the game runs up to H seconds with neither
side playing). The rollout detaches the replay recorder and path cache and restores the global random state and unit counters, so
the replay itself is unchanged. Per game the sidecar keeps the battle time and the simulator's end-of-replay tower shares.
Usage: PYTHONPATH=. python train/corpusStates.py --dataset /data/okebell/cr234/dataset --exclude ids.txt --out <dir> --jobs 30
"""
import argparse
import copy
import csv
import hashlib
import json
import random
import time
import types
from multiprocessing import Pool,Process
from pathlib import Path

import numpy as np

from sim import replay as R
from sim.cards import key
from sim.game import Replay
from sim.units import Building,Troop
from train.decisionStates import COLS,Recorder
from train.feats import FEAT_DIM

MODES=('Ranked','Ladder','Grand Challenge','Classic Challenge','1v1 Showdown')
EXTRA=('opp_hand','opp_next')
ROLL=('roll_own','roll_opp','roll_dcrown','roll_t')  # tower health shares after the rollout, crowns gained minus conceded, seconds run


def share(g,team):
    return sum(t.hp for t in g.arena.towers if t.team==team and t.alive)/sum(t.max_hp for t in g.arena.towers)


def quiet(g):
    return not g.pending and not g.spells and not g.projs and not any(u.alive for p in g.players.values() for u in p.troops)


def _copy_fn(f,memo):
    # callbacks (projectile hits bind towers and targets as closure cells or default arguments) must capture the copied units, not
    # the replay's; deepcopy treats functions as atoms
    if not f.__closure__ and not f.__defaults__ and not f.__kwdefaults__:return f
    cells=tuple(types.CellType() for _ in f.__closure__ or ())
    h=types.FunctionType(f.__code__,f.__globals__,f.__name__,None,cells or None);memo[id(f)]=h
    h.__defaults__=copy.deepcopy(f.__defaults__,memo);h.__kwdefaults__=copy.deepcopy(f.__kwdefaults__,memo)
    for c,n in zip(f.__closure__ or (),cells):
        try:n.cell_contents=copy.deepcopy(c.cell_contents,memo)
        except ValueError:pass
    return h


def clone(g):
    old=copy._deepcopy_dispatch[types.FunctionType];copy._deepcopy_dispatch[types.FunctionType]=_copy_fn
    try:
        return copy.deepcopy(g,{id(g.replay):Replay(),id(g._pf._cache):{}})
    finally:
        copy._deepcopy_dispatch[types.FunctionType]=old


def rollout(g,team,horizon):
    st=random.getstate();n=(Troop._n,Building._n);opp=g._opp(team)
    try:
        h=clone(g);c=h.players[team].crowns-h.players[opp].crowns;end=min(g.t+horizon,g.END)
        while h.t<end and not h.ended:
            h.run_to(min(h.t+1.0,end))
            if quiet(h):break
        return (share(h,team),share(h,opp),float(h.players[team].crowns-h.players[opp].crowns-c),h.t-g.t)
    finally:
        random.setstate(st);Troop._n,Building._n=n


class PrivRecorder(Recorder):
    # the shipped recorder plus what the actor cannot see at its decision: the opponent's forced hand and next card, and the no-play
    # rollout from the pre-play state (horizon 0 skips it)
    horizon=0.0
    def play_card(self,team,card,x,y,evolved=None,hero=None):
        recs=self.records;n=len(recs) if recs is not None else 0;extra=None
        if recs is not None:
            k=key(card) or card;last=recs[-1] if recs else None
            if not (last and last['t']==self.t and last['team']==team and last['card']==k):
                d=self.players[self._opp(team)].deck
                extra={'opp_hand':'|'.join(d.hand),'opp_next':d.nxt or '','roll':rollout(self,team,self.horizon) if self.horizon else (0.0,)*len(ROLL)}
        out=super().play_card(team,card,x,y,evolved=evolved,hero=hero)
        if extra and len(recs)>n:recs[-1].update(extra)
        return out


def extract(bid,plays,outcome,pid=None,horizon=0.0):
    records=[];old=R.Game;PrivRecorder.records=records;PrivRecorder.horizon=horizon
    try:
        R.Game=PrivRecorder;g,info=R.replay_battle(bid,plays,outcome,pid=pid)
    finally:
        R.Game=old;PrivRecorder.records=None
    game={'bid':bid,'actual_winner':info['actual_winner'],'sim_winner':info['sim_winner'],'end_t':info['end_t'],'premature':info['premature'],
          'actual_bc':info['actual_bc'],'actual_rc':info['actual_rc'],'sim_bc':info['sim_bc'],'sim_rc':info['sim_rc'],'n':len(records),
          'sim_blue':round(share(g,'blue'),5),'sim_red':round(share(g,'red'),5)}
    return game,records


def _work(args):
    return extract(*args)


def select(dataset,exclude):
    # eligible ids in battle-time order: one-versus-one standard modes with a replay and a time, none of the excluded ids
    import pyarrow.parquet as pq
    b=pq.read_table(Path(dataset)/'battles.parquet',columns=['replayTag','battle_ts','gameMode_name','one_v_one','has_replay'])
    rows=[(ts,t) for t,ts,m,o,h in zip(*(b.column(c).to_pylist() for c in b.column_names))
          if o and h and m in MODES and ts is not None and t.lstrip('#')==t and t not in exclude]
    return [t for _,t in sorted(rows)],{t:ts for ts,t in rows}


def exclusions(paths):
    # replayTag of every row of the frozen samples' battles.csv, or a plain id list (.txt, one id per line) made from them
    ids=set()
    for p in paths:
        with open(p) as f:
            rows=[line.strip() for line in f] if str(p).endswith('.txt') else [r['replayTag'] for r in csv.DictReader(f)]
        ids.update(r.lstrip('#') for r in rows if r)
    return ids


def write_shard(folder,tags,battles,placements):
    # the battles.csv/placements.csv pair of a shard, written as the assessment selector writes a frozen sample
    import pyarrow as pa
    import pyarrow.compute as pc
    folder.mkdir(parents=True,exist_ok=True);ids=pa.array(tags,type=pa.large_string())
    for name,table,col in (('battles.csv',battles,'replayTag'),('placements.csv',placements,'battle_id')):
        sub=table.filter(pc.is_in(table.column(col),value_set=ids))
        with (folder/name).open('w') as f:
            w=csv.DictWriter(f,fieldnames=table.schema.names);w.writeheader();w.writerows(sub.to_pylist())


def run_shard(folder,out,pool,ts,horizon=0.0,rolled=None):
    # rolled, when given, limits the no-play rollouts to those games
    paths=[folder/'battles.csv',folder/'placements.csv']
    meta=R.load_meta_v2(str(paths[0]));placements,pids=R.load_worker_rows(str(paths[1]),set(meta),meta)
    bids=sorted((b for b in meta if b in placements and not meta[b].get('modifier')),key=lambda b:(ts[b],b))
    t0=time.monotonic();games=[];rows=[];states=[];extra=[];roll=[]
    for game,recs in pool.imap(_work,[(b,placements[b],meta[b],pids.get(b),horizon if rolled is None or b in rolled else 0.0) for b in bids],chunksize=2):
        game['ts']=ts[game['bid']];game['mode']=meta[game['bid']]['gameMode'];game['rolled']=bool(horizon) and (rolled is None or game['bid'] in rolled)
        games.append(game)
        for i,r in enumerate(recs):
            rows.append((game['bid'],i,r['t'],r['team'],r['card'],r['x'],r['y'],r['evolved'],r['hero'],r['erate'],r['hand']));states.append(r['state'])
            extra.append((r['opp_hand'],r['opp_next']));roll.append(r['roll'])
    X=np.stack(states) if states else np.zeros((0,FEAT_DIM),np.float16)
    cols={c:np.array([r[k] for r in rows]) for k,c in enumerate(COLS)}|{c:np.array([e[k] for e in extra]) for k,c in enumerate(EXTRA)}
    np.savez_compressed(out,X=X,roll=np.array(roll,np.float32).reshape(-1,len(ROLL)),**cols)
    side={'shard':folder.name,'games':len(games),'records':len(rows),'horizon':horizon,'seconds':round(time.monotonic()-t0,1),
          'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},'outcomes':games}
    Path(out).with_suffix('.json').write_text(json.dumps(side)+'\n')
    return side


def load(npz):
    # per-record columns (with the opponent's hand and next card), states, rollout features and per-game outcomes
    d=np.load(npz,allow_pickle=False);side=json.loads(Path(npz).with_suffix('.json').read_text())
    return {c:d[c] for c in COLS+EXTRA},d['X'],d['roll'],{g['bid']:g for g in side['outcomes']}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--dataset',required=True)
    ap.add_argument('--exclude',nargs='+',required=True,help='battles.csv files of the frozen samples, or an id list made from them')
    ap.add_argument('--out',required=True);ap.add_argument('--work',default='/root/shards');ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--shard',type=int,default=5000);ap.add_argument('--shards',type=int,nargs='*',help='shard indices to run, default all')
    ap.add_argument('--horizon',type=float,default=0.0,help='no-play rollout seconds from each decision state, 0 for none')
    ap.add_argument('--roll_last',type=int,default=0,help='rollouts only for the last that many eligible games, 0 for all');a=ap.parse_args()
    import pyarrow.parquet as pq
    ex=exclusions(a.exclude);tags,ts=select(a.dataset,ex);out=Path(a.out);out.mkdir(parents=True,exist_ok=True)
    chunks=[tags[i:i+a.shard] for i in range(0,len(tags),a.shard)]
    (out/'selection.json').write_text(json.dumps({'eligible':len(tags),'excluded_ids':len(ex),'shard':a.shard,'shards':len(chunks),'modes':MODES,
                                                  'first_ts':ts[tags[0]],'last_ts':ts[tags[-1]],'exclusion_files':len(a.exclude),
                                                  'horizon':a.horizon,'roll_last':a.roll_last})+'\n')
    battles=pq.read_table(Path(a.dataset)/'battles.parquet');placements=pq.read_table(Path(a.dataset)/'placements.parquet')
    rolled=set(tags[-a.roll_last:]) if a.roll_last else None
    todo=[k for k in (a.shards if a.shards else range(len(chunks))) if not (out/f'shard{k:03d}.json').exists()]
    folder=lambda k:Path(a.work)/f'shard{k:03d}'
    with Pool(a.jobs) as pool:
        nxt=None
        for i,k in enumerate(todo):
            if nxt is None:write_shard(folder(k),chunks[k],battles,placements)
            else:
                nxt.join()
                if nxt.exitcode:raise RuntimeError(f'writing shard {k} failed')
            # the next shard's csv pair is written by a forked process while the pool replays this one
            if i+1<len(todo):nxt=Process(target=write_shard,args=(folder(todo[i+1]),chunks[todo[i+1]],battles,placements));nxt.start()
            side=run_shard(folder(k),out/f'shard{k:03d}.npz',pool,ts,a.horizon,rolled)
            print(json.dumps({x:v for x,v in side.items() if x not in ('outcomes','input_hashes')}),flush=True)


if __name__=='__main__':main()
