"""Certifying policies without the environment: do evaluators that differ from a policy's training evaluator avoid the self-overrating
found in the tenth step (train.wmOpe)?

The ninth step's simulator world (train.simTruth) gives the 26 trained policies true win rates against the behaviour; the tenth step
found that a learned evaluator overrates a policy trained against a learned evaluator of the same horizon (qgroup, trained on the
one-step Q of the 70k members 0-3, about 2.5 times by one-step evaluators; wmgroup, trained with their four-decision rollouts, 2.65 to
5.74 times by evaluators of four or more decisions). Here certification rules, registered before any estimate, are checked against
the truth from the held-out B starts:
  horizon  a policy is scored only by evaluators whose horizon differs from its training evaluator's (horizon_ne), or whose class
           differs (one-step against multi-step, horizon_class);
  split    evaluators fit on a trace half no policy, critic or model saw: world-model members with new seeds and FQE fit on held-out A
           (split_model, split_fqe, split_mean); the control seed_model is the 70k members 4-7 (same traces, new seeds);
  class    the minimum or the median over evaluator families: one-step outcome models, multi-step model rollouts, FQE;
  lower    conservative bounds: the family minimum minus 1.645 bootstrap standard deviations, the minimum over the members of the
           training ensemble, the minimum over the evaluator bank;
  qeval    the support-restricted Q-eval (Q fit on held-out A, scored on held-out B records), the baseline.
A gain is J(pi) - J(bc) under one evaluator; to combine evaluators of different scale each gain is divided by the evaluator's mean gain
over the anchors, the 24 policies trained without a learned evaluator (this uses no truth). Per rule: Spearman's correlation with the
true gains, the pick (largest score) and its regret, the overrating ratio of wmgroup and qgroup (score over the anchors' through-origin
slope of score on true gain, times the true gain; 1 is rated like the anchors), and certification (score minus 1.645 bootstrap
standard deviations above zero) against the ninth step's true verdicts, with a paired bootstrap over held-out B games (a game's two
starts and its Q-eval records together).
Usage: python -m train.certify fit|q0|fqe|mb|qeval|report ... (see main)
"""
import argparse
import copy
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import train.learnedWm as LW
import train.wmOpe as WO
from train.simTruth import rank_corr
from train.singleTraj import Corpus,eval_set,support

TRAIN_H={'wm/qgroup':0,'wm/wmgroup':4}
FQE_K=(0,1,2,4,8,16,32)
Z=1.645


def bank(m,f):
    # the model rollouts at every mark and FQE at the reported rounds of one evaluator source
    return [f'{m}.H{k}' for k in WO.MARKS]+[f'{f}.K{k}' for k in FQE_K]


def rules(m='M70',f='F70',sm='MA',sf='FA',qe=True,ens=('M70x1','M70m1','M70m2','M70m3'),sens=('MA0','MA1','MA2','MA3')):
    # the registered rules (hive note before the first run); the replication swaps the split sources and drops the Q-eval
    one=[f'{sm}.H0',f'{sf}.K0']+(['QE.support'] if qe else [])
    out={'primary':{'kind':'one','e':f'{m}.H64'},'onestep':{'kind':'one','e':f'{m}.H0'},'same_mean':{'kind':'mean','bank':bank(m,f),'excl':None},
         'horizon_ne':{'kind':'mean','bank':bank(m,f),'excl':'ne'},'horizon_class':{'kind':'mean','bank':bank(m,f),'excl':'class'},
         'split_model':{'kind':'one','e':f'{sm}.H64'},'split_fqe':{'kind':'one','e':f'{sf}.K32'},'split_mean':{'kind':'mean','bank':bank(sm,sf),'excl':None},
         'seed_model':{'kind':'one','e':'M70n.H64'},
         'class_min':{'kind':'min','families':[[f'{m}.H0',f'{f}.K0'],[f'{m}.H64'],[f'{f}.K32']]},
         'class_median':{'kind':'median','families':[[f'{m}.H0',f'{f}.K0'],[f'{m}.H64'],[f'{f}.K32']]},
         'lower_min':{'kind':'lower','of':'class_min'},'ensemble_min':{'kind':'min_raw','bank':[f'{e}.H64' for e in ens]},
         'bank_min':{'kind':'min','families':[[e] for e in bank(m,f)]},
         'split_class_min':{'kind':'min','families':[one,[f'{sm}.H64'],[f'{sf}.K32']]},
         'split_ensemble_min':{'kind':'min_raw','bank':[f'{e}.H64' for e in sens]}}
    if qe:out['qeval_support']={'kind':'one','e':'QE.support'};out['qeval_pess']={'kind':'one','e':'QE.pess'}
    return out


def explore():
    # exploratory (hive f5521, after a fit diagnostic): split fits with the 10k volume's epochs
    r=rules(sm='MAe',sf='FAe',sens=('MAe0','MAe1','MAe2','MAe3'))
    return {f'{k}_e':r[k] for k in ('split_model','split_fqe','split_mean','split_class_min','split_ensemble_min')}


def horizon(e):
    # decisions an evaluator looks ahead: model mark k, FQE round k, 0 for the Q-eval
    t=e.rsplit('.',1)[1];return int(t[1:]) if t[0] in 'HK' else 0


def needs(rs):
    # every evaluator a rule set reads
    out=set()
    for r in rs.values():
        out|=set(r.get('bank',[]))|{x for f in r.get('families',[]) for x in f}|({r['e']} if 'e' in r else set())
    return out


def normalise(G,anchors):
    # gains (..., P) over the evaluator's mean gain on the anchor columns; nan where that mean is not positive
    u=G[...,anchors].mean(-1,keepdims=True);return np.where(u>0,G/np.where(u>0,u,1.0),np.nan)


def score(r,G,names,anchors,rs=None,sd=None):
    # a rule's score per policy from gains G (evaluator -> (..., P) array, the bc column zero); lower needs the bootstrap sd of its base
    k=r['kind']
    if k=='one':return G[r['e']]
    if k=='min_raw':return np.min(np.stack([G[e] for e in r['bank']]),0)
    if k=='lower':return score(rs[r['of']],G,names,anchors,rs)-Z*sd
    N=lambda e:normalise(G[e],anchors)
    if k=='mean':
        th=np.array([TRAIN_H.get(a,-1) for a in names]);X=np.stack([N(e) for e in r['bank']]);ok=np.ones(X.shape,bool)
        for i,e in enumerate(r['bank']):
            h=horizon(e)
            if r['excl']=='ne':ok[i]=(th<0)|(th!=h)
            elif r['excl']=='class':ok[i]=(th<0)|((th==0)!=(h==0))
        return np.nanmean(np.where(ok,X,np.nan),0)
    fam=np.stack([np.mean(np.stack([N(e) for e in f]),0) for f in r['families']])
    return np.min(fam,0) if k=='min' else np.median(fam,0)


def ranks(x):
    # ranks along the last axis (ties broken by order; scores are continuous)
    return np.argsort(np.argsort(x,-1),-1).astype(np.float64)


def spearman_rows(S,t):
    # Spearman's correlation of every row of S with t
    a=ranks(S);b=ranks(t);a=a-a.mean(-1,keepdims=True);b=b-b.mean();return (a*b).sum(-1)/np.sqrt((a*a).sum(-1)*(b*b).sum())


def overrating(s,t,anchors,i):
    # score of policy i over the anchors' through-origin slope of score on true gain times its true gain ((..., P) scores)
    sl=(s[...,anchors]*t[anchors]).sum(-1)/(t[anchors]**2).sum();return s[...,i]/(sl*t[i])


def judge(s,sb,truth,names,anchors,bound=False):
    # one rule against the truth: s (P,) point scores, sb (B, P) bootstrap scores, truth name -> {'dwr': (mean, lo, hi), 'verdict'}; the
    # bc column (index 0) is the reference and left out; bound: the score is already a lower bound, certified when positive
    P=[i for i,a in enumerate(names) if a!='bc'];t=np.array([truth[a]['dwr'][0] for a in names]);lo=np.array([truth[a]['dwr'][1] for a in names])
    imp=np.array([truth[a]['verdict']=='improves' for a in names]);best=P[int(np.argmax(t[P]))];ok=np.isfinite(sb).all(1);sbo=sb[ok]
    sp,kd=rank_corr(s[P],t[P]);pick=P[int(np.argmax(s[P]))];spb=spearman_rows(sbo[:,P],t[P]);pb=np.array(P)[np.argmax(sbo[:,P],1)];rg=(t[best]-t[pb])*100
    sd=sbo.std(0);low=s if bound else s-Z*sd;cert=low>0;cert[0]=False;q=lambda x:[float(np.percentile(x,2.5)),float(np.percentile(x,97.5))]
    out={'spearman':float(sp),'spearman_boot':q(spb),'kendall':float(kd),'pick':names[pick],'true_best':names[best],'picks_best':bool(t[pick]>=lo[best]),
         'regret':float((t[best]-t[pick])*100),'regret_boot':q(rg),'p_regret_gt5':float((rg>5).mean()),
         'pick_freq':{names[i]:float((pb==i).mean()) for i in sorted(set(pb.tolist()),key=lambda i:-(pb==i).mean())},
         'certified':[names[i] for i in P if cert[i]],'false_cert':[names[i] for i in P if cert[i] and not imp[i]],
         'missed':[names[i] for i in P if imp[i] and not cert[i]],
         'pick_certified':bool(cert[pick]),'pick_false_cert':bool(cert[pick] and not imp[pick]),'boot_ok':int(ok.sum())}
    for a in TRAIN_H:
        if a in names:
            i=names.index(a);out[f'rho|{a}']=float(overrating(s,t,anchors,i));out[f'rho_boot|{a}']=q(overrating(sbo,t,anchors,i))
    rh=[out.get(f'rho|{a}',0) for a in TRAIN_H]
    out['avoids_overrating']=bool(all(x<=1.5 for x in rh))
    out['meets']=bool(sp>=0.90 and out['regret']<=5 and out['p_regret_gt5']<=0.05 and not out['false_cert'] and out['pick_certified'])
    return out


def boot_weights(G,B,seed=0):
    rng=np.random.default_rng(seed);return np.stack([np.bincount(rng.integers(0,G,G),minlength=G) for _ in range(B)]).astype(np.float64)


def gains(per,game,names,W,ratio=None):
    # gains over bc on the full sample and on each resample: per evaluator -> {policy: per-start values} with game the start's game index
    # (0..G-1); ratio evaluator -> {policy: (per-game numerator, per-game denominator)} for ratio estimates (the Q-eval); evaluators
    # missing a policy are dropped
    Gn=W.shape[1];cnt=np.bincount(game,minlength=Gn).astype(np.float64);pt={};bt={};dropped=[]
    for e,d in per.items():
        if not all(a in d for a in names):dropped.append(e);continue
        M=np.stack([np.bincount(game,d[a].astype(np.float64),Gn) for a in names],1);m=M.sum(0)/cnt.sum();b=(W@M)/(W@cnt)[:,None]
        pt[e]=m-m[0];bt[e]=b-b[:,:1]
    for e,d in (ratio or {}).items():
        if not all(a in d for a in names):dropped.append(e);continue
        Nu=np.stack([d[a][0] for a in names],1);De=np.stack([d[a][1] for a in names],1);m=Nu.sum(0)/De.sum(0);b=(W@Nu)/np.maximum(W@De,1e-12)
        pt[e]=m-m[0];bt[e]=b-b[:,:1]
    return pt,bt,dropped


def assess(pt,bt,truth,names,rs):
    # every rule's point and bootstrap scores and its judgement; per evaluator the overrating ratios and Spearman as a diagnostic
    P=[a for a in names if a!='bc'];anchors=np.array([i for i,a in enumerate(names) if a!='bc' and a not in TRAIN_H]);out={};sc={}
    t=np.array([truth[a]['dwr'][0] for a in names])
    for k,r in rs.items():
        if not needs({k:r})<=set(pt) and r['kind']!='lower':out[k]={'missing':sorted(needs({k:r})-set(pt))};continue
        if r['kind']=='lower':
            if r['of'] not in sc:out[k]={'missing':[r['of']]};continue
            sd=sc[r['of']][1].std(0);s=score(r,pt,names,anchors,rs,sd);sb=score(r,bt,names,anchors,rs,sd)
        else:s=score(r,pt,names,anchors);sb=score(r,bt,names,anchors)
        sc[k]=(s,sb);out[k]=judge(s,sb,truth,names,anchors,r['kind']=='lower');out[k]['score']={a:float(x) for a,x in zip(names,s)}
    diag={}
    for e in sorted(pt):
        g=pt[e];i=[names.index(a) for a in P];diag[e]={'spearman':float(rank_corr(g[i],t[i])[0]),'unit':float(g[anchors].mean())}
        for a in TRAIN_H:
            if a in names:diag[e][f'rho|{a}']=float(overrating(g,t,anchors,names.index(a)))
    return out,diag,sc


# ---- loading the evaluators

def tenth(c,rows,wmope):
    # per-start values on the given start rows from the tenth step's job outputs: model configs by menus, volume and members, FQE rounds
    pos={int(r):i for i,r in enumerate(rows)};per={};lab={('cycle',70,1):'M70x1',('cycle',70,4):'M70',('cycle',70,8):'M70x8',('cycle',10,1):'M10x1',
                                                         ('cycle',10,4):'M10',('cycle',30,1):'M30x1',('cycle',30,4):'M30',('recorded',70,4):'R70'}
    for p in sorted(Path(wmope,'mb').glob('v*_held_*.json')):
        r=json.loads(p.read_text());d=np.load(p.with_suffix('.npz'));idx=np.array([pos.get(int(x),-1) for x in d['rows']]);ok=idx>=0
        if ok.sum()!=len(rows):continue
        o=np.argsort(idx[ok]);sel=np.where(ok)[0][o]
        for ci,cfg in enumerate(r['configs']):
            pre=lab.get((r['menus'],r['games']//1000,len(cfg)))
            if pre is None:continue
            for j,k in enumerate(d[f'c{ci}_marks']):per.setdefault(f'{pre}.H{int(k)}',{})[r['name']]=d[f'c{ci}_value'][j][sel]
    for p in sorted(Path(wmope,'fqe').glob('v*.json')):
        r=json.loads(p.read_text());d=np.load(p.with_suffix('.npz'));idx=np.array([pos.get(int(x),-1) for x in d['starts']]);ok=idx>=0
        if ok.sum()!=len(rows):continue
        sel=np.where(ok)[0][np.argsort(idx[ok])]
        for k in FQE_K:
            if f'J{k}' in d.files:per.setdefault(f"F{r['games']//1000}.K{k}",{})[r['name']]=d[f'J{k}'][sel]
    return per


def ours(rows,cert):
    # per-start values from this round's model and FQE jobs (configs labelled in their json)
    pos={int(r):i for i,r in enumerate(rows)};per={}
    for p in sorted(Path(cert,'mb').glob('*.json'))+sorted(Path(cert,'fqe').glob('*.json')):
        r=json.loads(p.read_text());d=np.load(p.with_suffix('.npz'));key='rows' if 'rows' in d.files else 'starts'
        idx=np.array([pos.get(int(x),-1) for x in d[key]]);ok=idx>=0
        if ok.sum()!=len(rows):continue
        sel=np.where(ok)[0][np.argsort(idx[ok])]
        if 'labels' in r and 'configs' in r:
            for ci,lb in enumerate(r['labels']):
                for j,k in enumerate(d[f'c{ci}_marks']):per.setdefault(f'{lb}.H{int(k)}',{})[r['name']]=d[f'c{ci}_value'][j][sel]
        else:
            for k in FQE_K:
                if f'J{k}' in d.files:per.setdefault(f"{r['label']}.K{k}",{})[r['name']]=d[f'J{k}'][sel]
    return per


def qeval_ratio(qdir,gidx):
    # per-game numerators and denominators of the support-restricted and pessimistic Q-evals (train.singleTraj.evaluate's definitions)
    fx=np.load(Path(qdir)/'fixed.npz');g=np.array([gidx.get(int(x),-1) for x in fx['game']]);ok=g>=0;Gn=len(gidx);out={'QE.support':{},'QE.pess':{}}
    for p in sorted(Path(qdir).glob('*.npz')):
        if p.name=='fixed.npz':continue
        d=np.load(p);name=json.loads(p.with_suffix('.json').read_text())['name'];qn=d['qnum'].astype(np.float64);qm=d['qmass'].astype(np.float64)
        cov=(qm>1e-6).astype(np.float64);qs=np.where(cov>0,qn/np.where(cov>0,qm,1.0),0.0);qp=(qn+(1-qm)*fx['qmin'].astype(np.float64))*cov
        den=np.bincount(g[ok],cov[ok],Gn);out['QE.support'][name]=(np.bincount(g[ok],qs[ok],Gn),den);out['QE.pess'][name]=(np.bincount(g[ok],qp[ok],Gn),den)
    return out


def truth_of(path,opp='behaviour'):
    rep=json.loads(Path(path).read_text());return {a:{'dwr':r['true'][opp]['dwr'],'wr':r['true'][opp]['wr'],'verdict':r['true'][opp]['verdict']}
                                                  for a,r in rep['rows'].items() if opp in r.get('true',{})}


def report(pack,truth,wmope,cert,out,B=2000,seed=0,split='heldB',rep=False,log=print):
    c=Corpus(pack);rows=WO.starts(c,(split,));g=c.a['game'][rows].astype(np.int64);ug,gi=np.unique(g,return_inverse=True)
    gidx={int(x):i for i,x in enumerate(ug)}
    tr=truth_of(truth);names=['bc']+[a for a in tr if a not in ('bc','behaviour','clone_T1')]
    rs=rules(sm='MB',sf='FB',qe=False,sens=('MB0','MB1','MB2','MB3')) if rep else rules();ex={} if rep else explore()
    per={**tenth(c,rows,wmope),**ours(rows,cert)};ratio=qeval_ratio(Path(cert)/'qeval',gidx) if not rep and Path(cert,'qeval','fixed.npz').exists() else {}
    W=boot_weights(len(ug),B,seed);pt,bt,dropped=gains(per,gi,names,W,ratio);res,diag,_=assess(pt,bt,tr,names,rs);xres=assess(pt,bt,tr,names,ex)[0]
    o={'split':split,'replication':rep,'B':B,'games':int(len(ug)),'starts':int(len(rows)),'policies':names,'dropped':dropped,'rules':rs,'results':res,'explore':ex,'exploratory':xres,'evaluators':diag,
       'truth':{a:tr[a] for a in names},'gains':{e:{a:float(x) for a,x in zip(names,v)} for e,v in pt.items()}}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(o,indent=1)+'\n');Path(out).with_suffix('.md').write_text(markdown(o)+'\n')
    log(markdown(o),flush=True);return o


def markdown(o):
    f=lambda v,d=3:'' if v is None else f'{v:.{d}f}'
    out=['| rule | Spearman [boot] | pick (regret, pts) [boot] | P(regret>5) | rho wmgroup [boot] | rho qgroup [boot] | certified | false cert | missed '
         '| meets |',
         '|'+'---|'*10]
    for k,r in list(o['results'].items())+[(k+' (exploratory)',r) for k,r in o.get('exploratory',{}).items()]:
        if 'missing' in r:out.append(f"| {k} | missing {', '.join(r['missing'][:3])} |"+' |'*8);continue
        rb=lambda a:f"{f(r.get('rho|'+a),2)} [{f(r['rho_boot|'+a][0],2)}, {f(r['rho_boot|'+a][1],2)}]" if 'rho|'+a in r else ''
        out.append('| '+' | '.join([k,f"{r['spearman']:.3f} [{r['spearman_boot'][0]:.3f}, {r['spearman_boot'][1]:.3f}]",
                                    f"{r['pick']} ({r['regret']:.1f}) [{r['regret_boot'][0]:.1f}, {r['regret_boot'][1]:.1f}]",f(r['p_regret_gt5'],2),
                                    rb('wm/wmgroup'),rb('wm/qgroup'),
                                    str(len(r['certified'])),', '.join(r['false_cert']) or '0',str(len(r['missed'])),str(r['meets'])])+' |')
    out+=['','| evaluator | Spearman | unit (anchor mean gain) | rho wmgroup | rho qgroup |','|---|---|---|---|---|']
    for e,d in o['evaluators'].items():
        out.append(f"| {e} | {d['spearman']:.3f} | {d['unit']:.4f} | {f(d.get('rho|wm/wmgroup'),2)} | {f(d.get('rho|wm/qgroup'),2)} |")
    return '\n'.join(out)


# ---- jobs (on the devbox)

def fit_job(pack,aux,out,split,seed,max_steps=4079,threads=2,log=print):
    # one world-model member on one split's games with train.wmOpe.fit_volume's settings (as many steps whatever the volume)
    torch.set_num_threads(threads);c=Corpus(pack);ax=LW.load_aux(aux);rows=c.records(c.games(split));K=int(ax['K'])
    ep=max(1,math.ceil(max_steps/math.ceil(len(rows)/1024)));t0=time.monotonic();m,hist=LW.fit(c,ax,rows,seed,K,ep,max_steps=max_steps,log=log)
    Path(out).parent.mkdir(parents=True,exist_ok=True)
    LW.save_member(m,out,hist=hist,K=K,split=split,seed=seed,records=int(len(rows)),epochs=ep,seconds=round(time.monotonic()-t0,1))


def q0_job(pack,split,other,out,steps=3000,seed=0,log=print):
    # FQE's outcome model on one split's records, its log loss on (at most 100,000 records of) the other split
    c=Corpus(pack);t0=time.monotonic();A=c.a;Q=WO.fit_q0(c,c.records(c.games(split)),steps,seed=seed);hb=c.records(c.games(other))
    r=np.sort(np.random.default_rng(seed).choice(hb,min(100000,len(hb)),replace=False))
    with torch.no_grad():p=torch.sigmoid(Q(c.S(0,0,r),WO.T(A['card'][r]),WO.T(A['cell'][r]))).clamp(1e-6,1-1e-6)
    y=torch.from_numpy(A['y'][r].astype(np.float32));rep={'split':split,'other':other,'steps':steps,'other_logloss':float(F.binary_cross_entropy(p,y)),
                                                       'other_acc':float(((p>0.5).float()==y).float().mean()),'seconds':round(time.monotonic()-t0,1)}
    Path(out).parent.mkdir(parents=True,exist_ok=True);torch.save({'state':Q.state_dict(),**rep},out);log(json.dumps(rep),flush=True);return rep


def fqe_job(pack,spec,name,split,start,q0,out,label,K=32,steps=200,batch=1024,seed=0,log=print):
    # FQE of one policy on one split's trajectories from its outcome model, J per start of the other split
    c=Corpus(pack);t0=time.monotonic();A=c.a;pol=WO.load_policy(spec,c);s0=WO.starts(c,(start,));S0=c.S(0,0,s0);H0=WO.T(A['H'][s0])
    Q0=WO.Head(c.n_state,c.n_card,hidden=128);Q0.load_state_dict(torch.load(q0,weights_only=False)['state']);Q0.eval()
    res,_=WO.fqe(c,pol,c.records(c.games(split)),S0,H0,Q0,K,steps,batch,seed=seed);Path(out).parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,starts=s0,**{f'J{k}':v for k,v in res['per'].items() if k in FQE_K})
    rep={'name':name,'label':label,'split':split,'start':start,'K':K,'steps':steps,'batch':batch,'J':res['J'],'seconds':round(time.monotonic()-t0,1)}
    Path(out).with_suffix('.json').write_text(json.dumps(rep)+'\n');log(json.dumps(rep),flush=True);return rep


def mb_job(pack,states,wms,configs,labels,spec,name,start,out,distil=None,actors=None,Hmax=64,n_roll=2,distil_steps=1500,seed=0,log=print):
    # train.wmOpe.mb_job from the starts of one split: the policy distilled into every member on the records of the distil split (or the
    # actors a previous job saved, one per member of wms), then rolled out with the deck cycle under each configuration of members
    c=Corpus(pack);t0=time.monotonic();A=c.a;models=[LW.load_member(p) for p in wms];rows=WO.starts(c,(start,));S=c.S(0,0,rows);H=WO.T(A['H'][rows])
    pol=WO.load_policy(spec,c);say=lambda m:log(f'[mb {time.monotonic()-t0:.0f}s] {m}',flush=True)
    if actors and Path(actors).exists():
        d=torch.load(actors,weights_only=False);acts=[copy.deepcopy(m.actor) for m in models];fid=d['fidelity']
        for a,x in zip(acts,d['state']):a.load_state_dict(x)
    else:
        acts=WO.distil(models,pol,c,c.records(c.games(distil)),distil_steps,seed=seed);fr=np.sort(np.random.default_rng(seed+1).choice(len(rows),min(4000,len(rows)),replace=False))
        fid=WO.fidelity(models,acts,pol,S[fr],H[fr]);say(f"distilled: KL actor {np.mean(fid['kl_actor']):.4f} head {np.mean(fid['kl_head']):.4f}")
        if actors:Path(actors).parent.mkdir(parents=True,exist_ok=True);torch.save({'state':[a.state_dict() for a in acts],'fidelity':fid},actors)
    dk=WO.decks(states,c,rows);cost=WO.costs(c.vocab);el=WO.own_elixir(c,WO.ahead(c,rows,Hmax))
    nxt=lambda s,r:WO.Cycle(A['H'][rows[s]],dk[s],seed*1000+r,cost,el[s]);res={}
    rep={'name':name,'spec':spec,'start':start,'distil':distil,'members':[str(p) for p in wms],'configs':configs,'labels':labels,'fidelity':fid,'Hmax':Hmax,
         'n_roll':n_roll}
    for ci,cfg in enumerate(configs):
        v=WO.model_values([models[i] for i in cfg],[acts[i] for i in cfg],pol,S,H,nxt,Hmax,WO.MARKS,n_roll,seed)
        for k,x in v.items():res[f'c{ci}_{k}']=x
        rep[labels[ci]]=dict(zip(map(int,v['marks']),v['value'].mean(1).round(5).tolist()));say(f'{labels[ci]} {cfg}: J {rep[labels[ci]]}')
    rep['seconds']=round(time.monotonic()-t0,1);Path(out).parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(out,rows=rows,**res)
    Path(out).with_suffix('.json').write_text(json.dumps(rep)+'\n');return rep


def qeval_job(pack,prep,agents,out,q_n=30000,threads=4,log=print):
    # per-record Q-eval terms (train.seedRobust.Scorer) of every policy on the ninth step's held-out B evaluation records
    from train.seedRobust import Scorer
    torch.set_num_threads(threads);c=Corpus(pack);P=Path(prep);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    reg={'bc':str(P/'bc_warm.pt'),**WO.registry(agents)};bc=WO.load_policy(reg['bc'],c)
    Q=WO.Head(c.n_state,c.n_card,hidden=128);Q.load_state_dict(torch.load(P/'q.pt'));Q.eval();E=torch.load(P/'evals.pt',weights_only=False)
    ev=eval_set(c,E['rows']['heldB']);strict=support(c,np.concatenate([c.records(c.games('warm')),c.records(c.games('stream'))]),100)
    sc=Scorer(bc,Q,ev,E['sup']['final'],strict,q_n);m=sc.q_n;np.savez(out/'fixed.npz',game=c.a['game'][ev['rows'][:m]].astype(np.int64),qmin=sc.fixed['qmin'])
    for name,spec in reg.items():
        r=sc(WO.load_policy(spec,c));f=out/f"{name.replace('/','_')}.npz";np.savez(f,qnum=r['qnum'],qmass=r['qmass'])
        qm=r['qmass'].astype(np.float64);cov=qm>1e-6;qs=float((r['qnum'][cov]/qm[cov]).mean())
        f.with_suffix('.json').write_text(json.dumps({'name':name,'q_support':qs})+'\n');log(name,round(qs,5),flush=True)


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('fit');a.add_argument('--pack',required=True);a.add_argument('--aux',required=True);a.add_argument('--out',required=True)
    a.add_argument('--split',required=True);a.add_argument('--seed',type=int,required=True);a.add_argument('--threads',type=int,default=2)
    a.add_argument('--max_steps',type=int,default=4079)
    a=sp.add_parser('q0');a.add_argument('--pack',required=True);a.add_argument('--split',required=True);a.add_argument('--other',required=True)
    a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=1);a.add_argument('--steps',type=int,default=3000)
    a=sp.add_parser('fqe');a.add_argument('--pack',required=True);a.add_argument('--agents',nargs='+',required=True);a.add_argument('--policy',required=True)
    a.add_argument('--split',required=True);a.add_argument('--start',required=True);a.add_argument('--q0',required=True);a.add_argument('--out',required=True)
    a.add_argument('--label',required=True);a.add_argument('--threads',type=int,default=1);a.add_argument('--steps',type=int,default=200)
    a=sp.add_parser('mb');a.add_argument('--pack',required=True);a.add_argument('--states',required=True);a.add_argument('--wm',nargs='+',required=True)
    a.add_argument('--configs',required=True);a.add_argument('--labels',required=True);a.add_argument('--agents',nargs='+',required=True)
    a.add_argument('--policy',required=True);a.add_argument('--start',required=True);a.add_argument('--out',required=True);a.add_argument('--distil')
    a.add_argument('--actors');a.add_argument('--threads',type=int,default=1)
    a=sp.add_parser('qeval');a.add_argument('--pack',required=True);a.add_argument('--prep',required=True);a.add_argument('--agents',nargs='+',required=True)
    a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=4)
    a=sp.add_parser('report');a.add_argument('--pack',required=True);a.add_argument('--truth',required=True);a.add_argument('--wmope',required=True)
    a.add_argument('--cert',required=True);a.add_argument('--out',required=True);a.add_argument('--split',default='heldB');a.add_argument('--rep',action='store_true')
    a.add_argument('--B',type=int,default=2000)
    a=ap.parse_args();torch.set_num_threads(getattr(a,'threads',1))
    if a.cmd=='fit':fit_job(a.pack,a.aux,a.out,a.split,a.seed,a.max_steps,threads=a.threads)
    elif a.cmd=='q0':q0_job(a.pack,a.split,a.other,a.out,a.steps)
    elif a.cmd=='fqe':fqe_job(a.pack,WO.registry(a.agents)[a.policy],a.policy,a.split,a.start,a.q0,a.out,a.label,steps=a.steps)
    elif a.cmd=='mb':
        mb_job(a.pack,a.states,a.wm,json.loads(a.configs),json.loads(a.labels),WO.registry(a.agents)[a.policy],a.policy,a.start,a.out,distil=a.distil,actors=a.actors)
    elif a.cmd=='qeval':qeval_job(a.pack,a.prep,a.agents,a.out,threads=a.threads)
    else:report(a.pack,a.truth,a.wmope,a.cert,a.out,a.B,split=a.split,rep=a.rep)


if __name__=='__main__':main()
