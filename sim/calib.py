import argparse
import hashlib
import json
import os
import pickle
import time
from multiprocessing import Pool
from sim import replay as R
from sim import knobs

_BASE=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
META=os.path.join(_BASE,'data','raw','eval','battles.csv');WORK=os.path.join(_BASE,'data','raw','eval','placements.csv')
LOG=os.path.join(_BASE,'data','raw','calib.jsonl');CACHE=os.path.join(_BASE,'data','raw','eval','parsed.pkl')
# one coordinate per knob (kiting is one coordinate: the drop flag with its slack); the sweep values bracket the engine's value
GRID=[('sight_slack',[{'sight_slack':v} for v in (-1.0,-0.5,0.5,1.0,2.0)]),
      ('tower_acq',[{'tower_acq':v} for v in (0.1,0.2,0.4,0.8)]),
      ('tower_first',[{'tower_first':v} for v in (0.1,0.2,0.4,0.8)]),
      ('load_carry',[{'load_carry':v} for v in (0.0,0.25,0.5,0.75)]),
      ('sep_strength',[{'sep_strength':v} for v in (0.0,0.25,0.5,0.75)]),
      ('sep_iters',[{'sep_iters':v} for v in (2,3)]),
      ('bridge_blend',[{'bridge_blend':v} for v in (0.25,0.5,0.75,1.0)]),
      ('splash_hitbox',[{'splash_hitbox':v} for v in (0.0,0.25,0.5,0.75)]),
      ('kb_scale',[{'kb_scale':v} for v in (0.5,0.75,1.25,1.5)]),
      ('death_stagger',[{'death_stagger':v} for v in (0.1,0.2,0.35,0.5)]),
      ('detour_look',[{'detour_look':v} for v in (2.0,3.0,5.0,8.0,12.0)]),
      ('charge_scale',[{'charge_scale':v} for v in (0.6,0.8,1.2,1.5)]),
      ('kite',[{'kite_drop':1,'kite_slack':v} for v in (0.0,0.5,1.0,2.0)])]
COORD={n:sorted({k for o in c for k in o}) for n,c in GRID}

def inputs():
    st=(os.path.getmtime(META),os.path.getmtime(WORK))
    if os.path.exists(CACHE):
        with open(CACHE,'rb') as f:c=pickle.load(f)
        if c['st']==st:return c['data']
    outcomes=R.load_meta_v2(META);ids=set(outcomes);placements,pids=R.load_worker_rows(WORK,ids,outcomes)
    bids=sorted(b for b in ids if b in placements and not outcomes[b].get('modifier'))
    data=(outcomes,placements,pids,bids)
    with open(CACHE,'wb') as f:pickle.dump({'st':st,'data':data},f)
    return data

def holdout(bid):return int(hashlib.sha1(bid.encode()).hexdigest(),16)%2==1

def _run(a):
    bid,plays,outcome,pid,ov=a;knobs.use(ov)
    return holdout(bid),R.replay_battle(bid,plays,outcome,pid=pid)[1]

def summarize(infos):
    n=len(infos);hp=[i['hp_err'] for i in infos if i['hp_err'] is not None];an=sum(i['aim'][0] for i in infos);ah=sum(i['aim'][1] for i in infos)
    s={'n':n,'winner':sum(i['win_match'] for i in infos)/n,'crown':sum(i['crown_exact'] for i in infos)/n,'crown1':sum(i['crown_close'] for i in infos)/n,
       'premature':sum(i['premature'] for i in infos)/n,'hp':sum(hp)/len(hp),'state':sum(i['tower_state'] for i in infos if i['hp_err'] is not None)/len(hp),
       'aim':ah/an}
    s['obj']=s['hp']+(1-s['crown'])+s['premature']+(1-s['aim'])
    return s

def pkey(ov):
    # the slack is meaningless while kiting is off, so it is dropped from the key
    return json.dumps({k:round(v,4) for k,v in sorted(ov.items()) if v!=knobs.D[k] and (k!='kite_slack' or ov.get('kite_drop'))})

class Evaluator:
    def __init__(self,jobs,log=LOG):
        self.outcomes,self.placements,self.pids,self.bids=inputs();self.pool=Pool(jobs);self.log=log;self.seen={};self.new=0
        if os.path.exists(log):
            with open(log) as f:
                for line in f:r=json.loads(line);self.seen[pkey(r['params'])]=r
    def __call__(self,ov):
        k=pkey(ov)
        if k in self.seen:return self.seen[k]
        t0=time.time();tr=[];ho=[]
        for h,info in self.pool.imap_unordered(_run,[(b,self.placements[b],self.outcomes[b],self.pids.get(b),ov) for b in self.bids],chunksize=4):
            (ho if h else tr).append(info)
        r={'params':json.loads(k),'train':summarize(tr),'holdout':summarize(ho),'t':round(time.time()-t0,1)}
        with open(self.log,'a') as f:f.write(json.dumps(r)+'\n')
        self.seen[k]=r;self.new+=1
        print(f"[{self.new}] {k} train {fmt(r['train'])} | holdout {fmt(r['holdout'])} ({r['t']} s)",flush=True)
        return r

def fmt(s):return f"obj {s['obj']:.4f} w {100*s['winner']:.1f} c {100*s['crown']:.1f} p {100*s['premature']:.1f} hp {s['hp']:.4f} aim {100*s['aim']:.1f}"

def refine(ev,name,best_ov):
    # candidates for the next pass: the coordinate's best value so far and the midpoints toward its nearest values tried before (the step halves each round)
    k='kite_slack' if name=='kite' else COORD[name][0]
    tried={json.loads(kk).get(k,knobs.D[k]) for kk in ev.seen if name!='kite' or json.loads(kk).get('kite_drop')}
    b=best_ov.get(k,knobs.D[k]);vs=sorted(tried|{b});i=vs.index(b);mids=[] if k=='sep_iters' else [(vs[i]+vs[j])/2 for j in (i-1,i+1) if 0<=j<len(vs)]
    out=[{k:v} for v in [b]+mids if knobs.B[k][0]<=v<=knobs.B[k][1]]
    return [{'kite_drop':1,**o} for o in out]+[{'kite_drop':0,'kite_slack':0.0}] if name=='kite' else out

def search(ev,budget,eps):
    base=ev({});x={};gain={}
    print(f"base train {fmt(base['train'])} | holdout {fmt(base['holdout'])}",flush=True)
    for name,cands in GRID:
        rs=[(ev(o),o) for o in cands]
        r,o=min(rs,key=lambda p:p[0]['train']['obj']);gain[name]=(base['train']['obj']-r['train']['obj'],o)
    order=sorted(gain,key=lambda n:-gain[n][0])
    print('sweep gains: '+', '.join(f"{n} {gain[n][0]:+.4f} {gain[n][1]}" for n in order),flush=True)
    cur=base
    for rnd in range(3):
        moved=False
        for name in order:
            if ev.new>=budget:break
            best_ov=gain[name][1] if rnd==0 and gain[name][0]>eps else {k:cur['params'].get(k,knobs.D[k]) for k in COORD[name]}
            for o in refine(ev,name,best_ov):
                trial={**cur['params'],**o}
                r=ev(trial)
                if r['train']['obj']<cur['train']['obj']-eps:cur=r;x=dict(trial);moved=True
        print(f"round {rnd} best {pkey(x)} train {fmt(cur['train'])} | holdout {fmt(cur['holdout'])}",flush=True)
        if not moved:break
    return cur

def report(log=LOG):
    rows=[json.loads(l) for l in open(log)];base=next(r for r in rows if not r['params']);bh=base['holdout'];bt=base['train']
    print(f"base: train {fmt(bt)} | holdout {fmt(bh)}")
    for name,cands in GRID:
        ks=COORD[name];one=[r for r in rows if set(r['params'])<=set(ks) and r['params']]
        if not one:continue
        print(f"{name}:")
        for r in sorted(one,key=lambda r:[r['params'].get(k,0) for k in ks]):
            h=r['holdout'];t=r['train']
            print(f"  {json.dumps(r['params']):40s} train obj {t['obj']-bt['obj']:+.4f} | holdout winner {100*(h['winner']-bh['winner']):+.1f}"
                  f" crown {100*(h['crown']-bh['crown']):+.1f} hp {h['hp']-bh['hp']:+.4f} aim {100*(h['aim']-bh['aim']):+.1f} obj {h['obj']-bh['obj']:+.4f}")

def main():
    ap=argparse.ArgumentParser(description='fit the unsourced engine constants on a training half of the eval set')
    ap.add_argument('--jobs',type=int,default=16);ap.add_argument('--budget',type=int,default=200);ap.add_argument('--eps',type=float,default=0.003)
    ap.add_argument('--eval',type=str,default=None,help='JSON overrides to evaluate once');ap.add_argument('--report',action='store_true')
    a=ap.parse_args()
    if a.report:report();return
    ev=Evaluator(a.jobs)
    if a.eval is not None:r=ev(json.loads(a.eval));print(f"train {fmt(r['train'])} | holdout {fmt(r['holdout'])}");return
    search(ev,a.budget,a.eps)

if __name__=='__main__':
    main()
