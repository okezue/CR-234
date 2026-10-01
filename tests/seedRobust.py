import json

import numpy as np
import torch

import train.singleTraj as ST
from tests.singleTraj import packed
from train.seedRobust import (PER,PER_Q,Scorer,arm_seed,by_game,floor,interval,summarise,student_q,terms,value,welch)
from train.singleTraj import Corpus,batch,batch_of,day_batches,eval_set,evaluate,extend,prep,run_arms
from train.traceRl import N_CELLS,Head,Policy

# The seed-robust harness on synthetic corpora (tests.singleTraj.packed) and planted per-record arrays: a training seed permutes the games
# within each stream day and nothing else about the stream, a batch of arbitrary games is the consecutive batch when the games are
# consecutive, per-record scores rebuild train.singleTraj.evaluate exactly, the game bootstrap is paired, and the registered verdicts use
# the lower ends of the intervals against mean-plus-two-sd floors.


def t_scored_records_reproduce_evaluate(tmp_path):
    d=packed(tmp_path,n_games=120,per=60);c=Corpus(d);torch.manual_seed(0);rows=np.sort(np.random.default_rng(0).choice(len(c.a['y']),600,replace=False))
    ev=eval_set(c,rows);bc=Policy(c.n_state,c.n_card,16);pol=Policy(c.n_state,c.n_card,16);Q=Head(c.n_state,c.n_card,hidden=16)
    g=torch.Generator().manual_seed(1);sup=torch.rand(c.n_card,N_CELLS,generator=g)>0.2;strict=sup&(torch.rand(c.n_card,N_CELLS,generator=g)>0.5)
    ref=evaluate(pol,bc,Q,ev,sup,q_n=250,chunk=100,strict=strict);sc=Scorer(bc,Q,ev,sup,strict,q_n=250,chunk=100)
    game=c.a['game'][rows].astype(np.int64);ug,gi=np.unique(game,return_inverse=True);v=value(by_game(terms(sc(pol),sc.fixed),gi,len(ug)))
    # q_n not a multiple of the chunk: the Q-eval covers exactly the first 250 records as evaluate's does
    assert ref['q_records']==250 and len(sc(pol)['qnum'])==250
    for k,x in v.items():assert abs(x-ref[k])<2e-5,(k,x,ref[k])
    # unit weights on every game are the full sample
    W=np.ones((3,len(ug)));vb=value(by_game(terms(sc(pol),sc.fixed),gi,len(ug)),W)
    assert all(np.allclose(vb[k],v[k]) for k in v)


def t_day_batches_permute_games_only_within_days(tmp_path):
    c=Corpus(packed(tmp_path,n_games=300,per=100));gs=c.games('stream');bl=day_batches(c,gs,8,3)
    flat=np.concatenate(bl);days=[int(c.day[b[0]]) for b in bl]
    assert sorted(flat.tolist())==gs.tolist() and len(flat)==len(gs) and all(len(set(c.day[b].tolist()))==1 and len(b)<=8 for b in bl)
    assert days==sorted(days) and not np.array_equal(flat,gs)
    assert all(np.array_equal(x,y) for x,y in zip(bl,day_batches(c,gs,8,3))) and not np.array_equal(flat,np.concatenate(day_batches(c,gs,8,4)))
    # a batch of consecutive games through batch_of is the consecutive batch, and the arm's view of it is the same
    a=batch(c,3,7);b=batch_of(c,[3,4,5,6]);flip=np.random.default_rng(0).random(len(c.split))<0.5;mu=np.random.default_rng(1).random((len(c.a['y']),2))
    assert np.array_equal(b['rows'],np.arange(*a['rows'])) and b['n']==a['n']
    for k in ('S','H','card','cell','tr','pos','L','R'):assert torch.equal(a[k],b[k]),k
    ea,eb=extend(c,a,('hidden','sim'),mu,flip),extend(c,b,('hidden','sim'),mu,flip)
    for k in ('Z','mu','R'):assert torch.equal(ea[k],eb[k]),k
    # out of order games keep each record with its own trajectory
    o=batch_of(c,[6,3]);r=o['rows'];assert (c.a['traj'][r]==np.asarray(o['trajs'])[o['tr'].numpy()]).all() and c.a['game'][r[0]]==6
    assert (o['L'][o['tr']].numpy()==c.a['t_len'][c.a['traj'][r]]).all() and (o['pos'].numpy()==c.a['pos'][r]).all()


def t_an_order_seed_reorders_the_stream_and_keeps_the_flips(tmp_path,monkeypatch):
    d=packed(tmp_path,n_games=300,per=100);log=lambda *a,**k:None
    prep(d,hidden=16,bc_epochs=1,critic_epochs=1,q_epochs=1,eval_n=500,day_n=100,rolled_pre=10,log=log,fit_batch=256)
    seen=[];flips=[];ob,oe=ST.batch_of,ST.extend
    monkeypatch.setattr(ST,'batch_of',lambda c,gs:(seen.append(np.asarray(gs).copy()),ob(c,gs))[1])
    monkeypatch.setattr(ST,'extend',lambda c,base,priv=(),mu=None,flip=None:(flip is not None and flips.append(flip),oe(c,base,priv,mu,flip))[1])
    arms=('bpco:flip:seed=2','flash:mu=warm:seed=2');kw={'batch_games':4,'lr':3e-3,'hidden':16,'threads':1,'log':log,'prequential':False}
    # two runs of the recorded order are identical (NaN where the tiny corpus has no strictly supported play, so compared as JSON)
    r0=run_arms(d,arms,**kw);r0b=run_arms(d,arms,**kw);assert not seen and json.dumps(r0['arms'])==json.dumps(r0b['arms']) and r0['order']==0
    f0=flips[0];flips.clear()
    r2=run_arms(d,arms,order=2,**kw);c=Corpus(d)
    # every stream game once, in day order, and the flip arm's policy moves with the order while its flipped games stay those of its seed
    assert sorted(np.concatenate(seen).tolist())==c.games('stream').tolist() and r2['order']==2
    assert all(json.dumps(r2['arms'][a])!=json.dumps(r0['arms'][a]) for a in arms)
    assert all(f is f0 or np.array_equal(f,f0) for f in flips) and np.array_equal(f0,np.random.default_rng(102).random(len(c.split))<0.5)


def t_t_intervals_floors_and_welch():
    assert abs(student_q(0.975,4)-2.7764)<1e-3 and abs(student_q(0.975,1)-12.706)<1e-2 and abs(student_q(0.975,1e6)-1.95996)<1e-3
    iv=interval([1,2,3,4,5]);assert iv['mean']==3 and abs(iv['sd']-1.5811)<1e-4 and abs(iv['hi']-3-2.7764*1.5811/5**0.5)<1e-3
    g=[0.0,0.01,-0.005,0.002,0.004];assert abs(floor(g)-(np.mean(g)+2*np.std(g,ddof=1)))<1e-12 and floor(g)!=max(abs(x) for x in g)
    # equal variances and sizes: df = 2 (n - 1)
    w=welch([[1,2,3,4,5],[2,3,4,5,6]],0.0);assert abs(w['df']-8)<1e-9 and abs(w['hi']-student_q(0.975,8)*(2*2.5/5)**0.5)<1e-6


def planted(gains,n=60,G=12,seed=0):
    # per-record arrays whose pessimistic Q-eval exceeds the clone's by exactly the given gain on every record, all mass supported
    rng=np.random.default_rng(seed);base=rng.uniform(0.4,0.6,n).astype(np.float32);fx={'game':np.repeat(np.arange(G),n//G),'won':np.arange(n)%2==0,
                                                                                       'qmin':np.zeros(n,np.float32),'qmin_s':np.zeros(n,np.float32),'q_n':np.array(n)}
    def rec(gn):
        r={k:np.zeros(n,np.float32) for k in PER+PER_Q};r.update(mass=np.ones(n,np.float32),qmass=np.ones(n,np.float32),mass_s=np.ones(n,np.float32))
        r['qnum']=base+gn;r['qdir']=r['qnum'].copy();r['qnum_s']=r['qnum'].copy();return r
    return fx,{'bc':{0:rec(0.0)},**{a:{s:rec(x) for s,x in enumerate(v)} for a,v in gains.items()}}


def t_bootstrap_is_paired_and_verdicts_use_the_lower_ends():
    gains={'bpco':[0.010,0.012,0.008,0.011,0.009],'bpco:flip':[0.0,0.001,-0.001,0.002,-0.002],'sao':[0.004,0.0,0.008,0.002,0.006],
           'sao:flip':[0.0,0.001,-0.001,0.002,-0.002],'wmgroup':[0.0115,0.0135,0.0095,0.0125,0.0105],'wmgroup:flip':[0.012,-0.001,0.004,0.008,0.001]}
    fx,recs=planted(gains);s=summarise(recs,fx,B=200);v=s['verdicts']
    # every resample of the games moves the clone and the arm together, so a constant per-record gain has a zero-width interval
    assert abs(s['arms']['bpco']['q_pess']['mean']-0.010)<1e-6 and all(abs(x-0.010)<1e-6 for x in s['arms']['bpco']['q_pess']['boot'])
    F=np.mean(gains['bpco:flip'])+2*np.std(gains['bpco:flip'],ddof=1);assert abs(v['bpco']['F']-F)<1e-6 and v['bpco']['improves']
    # sao's mean exceeds its floor but the lower end of its seed interval does not
    assert np.mean(gains['sao'])>v['sao']['F'] and v['sao']['t_lo']<v['sao']['F'] and not v['sao']['improves'] and v['sao']['seeds_above_F']==3
    # the learned-model floor is the largest, and wmgroup's lead over bpco (.0015 every seed) does not survive the floors' means
    assert s['F_max']==v['wmgroup']['F'] and not v['wmgroup']['improves']
    q2=s['compare']['Q2'];assert abs(q2['d']['mean']-0.0015)<1e-6 and q2['d']['lo']>0 and q2['boot'][0]>0 and q2['excess']['lo']<0
    assert q2['verdict']=='higher, not beyond the floors' and s['compare']['sao vs bpco']['verdict']=='worse'


def t_policies_are_named_by_arm_and_seed():
    assert arm_seed('wmgroup:flip:seed=3',{'order':3})==('wmgroup:flip',3) and arm_seed('wmgroup:seed=1',{})==('wmgroup@order0',1)
    assert arm_seed('bpco_sim:seed=2',{'order':2,'slice':'rolled'})==('rolled/bpco_sim',2)
    assert arm_seed('simgroup:epochs=30',{'seed':4})==('simgroup:epochs=30',4)
