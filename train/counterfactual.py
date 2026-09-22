"""Counterfactual rollouts in the world model: environment replay where the real environment cannot be recreated.

A recorded decision (train.decisionStates) is one play from a menu of four hand cards. The simulator can answer what it thinks would
have happened had the player chosen otherwise: the recorded game is replayed with one play replaced (another hand card at the same
tile, or the same card at another tile) and both players' later recorded plays kept, and the simulated result for the actor is
scored at the end. The recorded play is rolled out the same way as alternative zero, so the group is comparable and the recorded
arm reproduces the plain replay exactly. Group-relative advantages from these returns are the world model's substitute for a fresh
group of samples; train.traceRl consumes them as the simgroup method.

The return for the actor is crowns difference plus the tower health margin (own minus enemy tower health over the health of all six
towers), and the simulated winner is kept beside it. Rollouts are full replays, so this is remote-compute work.
Usage: PYTHONPATH=. python train/counterfactual.py --data <sample dir> --npz local/decisions/<name>.npz --out local/decisions/<name>Cf.json
       --games 300 --alts 3 --jobs 30
"""
import argparse
import hashlib
import json
import random
import time
from multiprocessing import Pool
from pathlib import Path

from sim import replay as R
from sim.cards import card,key
from train.decisionStates import load

GW,GH=6,8  # 3x4 tile cells over the 18x32 arena, the deploy grid of train.rl (redefined here to keep pandas out of this path)
CELLS=[[(x,y) for y in range(4*gy,4*gy+4) for x in range(3*gx,3*gx+3)] for gy in range(GH) for gx in range(GW)]
FIELDS=('bid','idx','alt','team','card','x','y','win','margin','ret')


def cell_of(x,y):
    return min(max(int(y)//4,0),GH-1)*GW+min(max(int(x)//3,0),GW-1)


def own_cells(team):
    # deploy cells on the actor's side of the river (rows 0..15 blue, 16..31 red)
    return [i for i in range(GW*GH) if (i//GW<4)==(team=='blue')]


def score(g,team):
    opp='red' if team=='blue' else 'blue'
    hp=lambda tm:sum(t.hp for t in g.arena.towers if t.team==tm and t.alive)
    total=sum(t.max_hp for t in g.arena.towers)
    margin=(hp(team)-hp(opp))/total;crowns=g.players[team].crowns-g.players[opp].crowns
    win=0.0 if g.winner not in ('blue','red') else (1.0 if g.winner==team else -1.0)
    return {'win':win,'margin':round(margin,5),'ret':round(crowns+margin,5)}


def alternatives(rec,rng,n_alts):
    # the menu: other hand cards at the recorded tile, then the recorded card at another own-side cell (spells anywhere)
    hand=[c for c in rec['hand'].split('|') if c and c!=rec['card'] and key(c)]
    rng.shuffle(hand);alts=[(c,rec['x'],rec['y']) for c in hand[:n_alts]]
    spell=card(rec['card'])['kind']=='spell' if key(rec['card']) else False
    while len(alts)<n_alts:
        cells=list(range(GW*GH)) if spell else own_cells(rec['team'])
        c=rng.choice([i for i in cells if i!=cell_of(rec['x'],rec['y'])] or cells);tx,ty=rng.choice(CELLS[c])
        alts.append((rec['card'],tx+0.5,ty+0.5))
    return alts


def rollout(bid,plays,outcome,pid,i,name,x,y):
    # one full replay with recorded play i replaced; the replay copies its play list, so the recorded plays stay untouched
    plays=[dict(p) for p in plays];p=plays[i]
    p['card']=name;p['tile_x']=float(x);p['tile_y']=float(y);p['card_type']='normal'
    g,info=R.replay_battle(bid,plays,outcome,pid=pid)
    return g,info


def decision_plays(plays):
    # indices of the recorded card plays in replay order, matching the extractor's record order (abilities and unknown cards skipped)
    return [i for i,p in enumerate(plays) if p['ability']!=1 and R.norm(p['card'])[0] and R._has_json(R.norm(p['card'])[0])]


def _work(args):
    bid,plays,outcome,pid,recs,n_alts,seed=args;rng=random.Random(seed);rows=[];idxs=decision_plays(plays)
    for rec in recs:
        if rec['idx']>=len(idxs):continue
        i=idxs[rec['idx']];menu=[(rec['card'],rec['x'],rec['y'])]+alternatives(rec,rng,n_alts)
        for k,(name,x,y) in enumerate(menu):
            g,_=rollout(bid,plays,outcome,pid,i,name,x,y);s=score(g,rec['team'])
            rows.append((bid,rec['idx'],k,rec['team'],name,round(float(x),2),round(float(y),2),s['win'],s['margin'],s['ret']))
    return rows


def build(data,npz,out,games=300,alts=3,jobs=2,every=1,seed=0):
    cols,X,outcomes=load(npz);paths=[Path(data)/n for n in ('battles.csv','placements.csv')]
    meta=R.load_meta_v2(str(paths[0]));placements,pids=R.load_worker_rows(str(paths[1]),set(meta),meta)
    bids=sorted(b for b in set(str(b) for b in cols['bid']) if b in placements and outcomes[b]['actual_winner'] is not None)[:games]
    recs={}
    for j,b in enumerate(cols['bid']):
        b=str(b)
        if b in set(bids) and int(cols['idx'][j])%every==0:
            recs.setdefault(b,[]).append({'idx':int(cols['idx'][j]),'team':str(cols['team'][j]),'card':str(cols['card'][j]),
                                          'x':float(cols['x'][j]),'y':float(cols['y'][j]),'hand':str(cols['hand'][j])})
    t0=time.monotonic();rows=[]
    with Pool(jobs) as pool:
        for r in pool.imap_unordered(_work,[(b,placements[b],meta[b],pids.get(b),recs[b],alts,seed+n) for n,b in enumerate(bids) if b in recs]):
            rows+=r
    rows.sort(key=lambda r:(r[0],r[1],r[2]))
    side={'data':str(data),'npz':str(npz),'games':len(bids),'decisions':len({(r[0],r[1]) for r in rows}),'rollouts':len(rows),'alts':alts,
          'seconds':round(time.monotonic()-t0,1),'input_hashes':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
          'code':hashlib.sha256(b''.join(p.read_bytes() for p in sorted(Path('sim').glob('*.py'))+[Path('data/cards.json')])).hexdigest(),
          'fields':FIELDS,'rows':rows}
    out=Path(out);out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(side)+'\n')
    return side


def load_groups(path):
    # {(bid, idx): [(card, x, y, win, margin, ret), ...]} with the recorded play first
    side=json.loads(Path(path).read_text());groups={}
    for bid,idx,alt,team,name,x,y,win,margin,ret in side['rows']:
        groups.setdefault((str(bid),int(idx)),[]).append((name,x,y,win,margin,ret))
    return groups


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--data',required=True);ap.add_argument('--npz',required=True);ap.add_argument('--out',required=True)
    ap.add_argument('--games',type=int,default=300);ap.add_argument('--alts',type=int,default=3);ap.add_argument('--jobs',type=int,default=2)
    ap.add_argument('--every',type=int,default=1);ap.add_argument('--seed',type=int,default=0);a=ap.parse_args()
    side=build(a.data,a.npz,a.out,a.games,a.alts,a.jobs,a.every,a.seed)
    print(json.dumps({k:v for k,v in side.items() if k!='rows'},indent=1))


if __name__=='__main__':main()
