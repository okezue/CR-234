"""Seed-robust verdicts for the trace-RL results (train.singleTraj, train.learnedWm): the key arms over several training seeds and the
flipped-outcome floors over several flip seeds, every policy scored per held-out record so that each metric can be recomputed on
resampled games.

A training seed k sets the order of the games within each stream day (k = 0 keeps the recorded order, so seed 0 reproduces the earlier
runs; train.singleTraj.day_batches) and the arm's own random draws; a flip seed also sets which games have their outcome flipped and,
for the learned model, the seed of the refit outcome heads. The clone, critics, Q model and behaviour estimate are shared by all seeds.
Every policy is scored on the same half-B records with train.singleTraj.evaluate's definitions (rec), and the report gives per arm
the seed mean, its t interval over seeds and a paired bootstrap over held-out games (the same resampled games for every policy and the
clone), the floors (mean plus two standard deviations of a flip arm's gains) and the verdicts registered in hive f5224: an arm improves
if the lower ends of both intervals of its pessimistic Q-eval gain exceed its recipe's floor (support within a point, no leave flag);
A beats B if the paired-by-seed difference and its excess over the two floors' means lie above zero.
Usage: python -m train.seedRobust rec|exploit|cos|report ... (see main)
"""
import argparse
import json
import math
import re
from pathlib import Path

import numpy as np
import torch

from train.singleTraj import Corpus,eval_set,support,token_logp
from train.traceRl import N_CELLS,Head,Policy,menu_q

PER=('lp','lpc','kl','ent','top','mass')
PER_Q=('qnum','qmass','qdir','mass_s','qnum_s','qtop')
GAINS=('q_support','q_pess','q_pess_strict','q_direct')
REPORTED=('q_pess','q_support','q_pess_strict','q_direct','winner_gap','winner_gap_card','entropy_gap','top1_won','top1_lost','kl_to_bc','entropy',
          'support_mass','support_mass_strict','q_top_mass')
# registered in f5224: the floor arm of each arm (min_mf: the smallest model-free floor), the model-free floors, the comparisons
FLOOR_OF={'bc_online':'min_mf','flash':'flash:flip','flash_nogate':'flash:flip','sao':'sao:flip','bpco':'bpco:flip','rolled/bpco':'rolled/bpco:flip',
          'rolled/bpco_sim':'rolled/bpco:flip','wmgroup':'wmgroup:flip','qgroup':'qgroup:flip'}
MODEL_FREE=('flash:flip','sao:flip','bpco:flip','rolled/bpco:flip')
COMPARE=(('wmgroup','bpco','Q2'),('sao','bpco','sao vs bpco'),('rolled/bpco_sim','rolled/bpco','sim critic'),('wmgroup','qgroup','Q3'),
         ('wmmenu:epochs=30','simgroup:epochs=30','Q1'),('flash','bc_online','flash vs bc_online'))


def betacf(a,b,x,it=400,eps=1e-15):
    # continued fraction of the regularised incomplete beta function (modified Lentz)
    tiny=1e-300;qab=a+b;qap=a+1;qam=a-1;c=1.0;d=1-qab*x/qap;d=1/(d if abs(d)>tiny else tiny);h=d
    for m in range(1,it+1):
        m2=2*m
        for aa in (m*(b-m)*x/((qam+m2)*(a+m2)),-(a+m)*(qab+m)*x/((a+m2)*(qap+m2))):
            d=1+aa*d;d=1/(d if abs(d)>tiny else tiny);c=1+aa/c;c=c if abs(c)>tiny else tiny;h*=d*c
        if abs(d*c-1)<eps:break
    return h


def betainc(a,b,x):
    if x<=0:return 0.0
    if x>=1:return 1.0
    lb=math.lgamma(a+b)-math.lgamma(a)-math.lgamma(b)+a*math.log(x)+b*math.log1p(-x)
    return math.exp(lb)*betacf(a,b,x)/a if x<(a+1)/(a+b+2) else 1-math.exp(lb)*betacf(b,a,1-x)/b


def student_cdf(t,df):
    p=0.5*betainc(df/2,0.5,df/(df+t*t));return 1-p if t>0 else p


def student_q(p,df):
    lo,hi=-1e4,1e4
    for _ in range(200):
        mid=(lo+hi)/2
        if student_cdf(mid,df)<p:lo=mid
        else:hi=mid
    return (lo+hi)/2


def interval(xs,level=0.95):
    # mean, sample sd and the t interval over seeds
    x=np.asarray(xs,np.float64);n=len(x);m=float(x.mean());sd=float(x.std(ddof=1)) if n>1 else 0.0
    h=student_q(0.5+level/2,n-1)*sd/math.sqrt(n) if n>1 else math.inf
    return {'mean':m,'sd':sd,'lo':m-h,'hi':m+h,'n':n,'values':[float(v) for v in x]}


def floor(gains):
    # the flipped-outcome floor: mean plus two sample standard deviations of the flip runs' gains
    g=np.asarray(gains,np.float64);return float(g.mean()+2*g.std(ddof=1))


def welch(parts,est,level=0.95):
    # interval of est with variance the sum of per-part variances of means, df by Welch-Satterthwaite; parts are value lists
    vs=[(float(np.var(p,ddof=1))/len(p),len(p)) for p in parts];v=sum(x for x,_ in vs)
    if v<=0:return {'mean':est,'lo':est,'hi':est,'df':math.inf}
    df=v*v/sum(x*x/(n-1) for x,n in vs if x>0);h=student_q(0.5+level/2,df)*math.sqrt(v)
    return {'mean':est,'lo':est-h,'hi':est+h,'df':df}


class Scorer:
    # the held-out records of one evaluation set with the policy-free parts cached: the Q model's value of every play of the first q_n
    # records (as evaluate scores them), the support masks, each state's worst supported play (and strictly supported) and its Q-argmax
    def __init__(self,ref,Q,ev,sup,strict,q_n=30000,chunk=20000):
        self.ref=ref;self.ev=ev;self.sup=sup;self.strict=strict;self.chunk=chunk;n=len(ev['y']);m=self.q_n=min(q_n,n);H=ev['H']
        with torch.no_grad():
            q,okq=menu_q(Q,ev['S'][:m],H[:m],sup);vs=(H[:m]>=0)[:,:,None].expand(-1,-1,N_CELLS);oks=strict[H[:m].clamp(min=0)]&vs
            self.q=q;self.okq=okq;self.valid=vs;self.oks=oks;self.top=torch.where(okq,q,torch.full_like(q,-1.0)).flatten(1).argmax(1)
            self.fixed={'qmin':torch.where(okq,q,torch.full_like(q,2.0)).amin((1,2)).numpy(),'qmin_s':torch.where(oks,q,torch.full_like(q,2.0)).amin((1,2)).numpy(),
                        'won':(ev['y']>0.5).numpy(),'q_n':np.array(m)}
    def __call__(self,pol):
        ev=self.ev;S,H,card,cell=ev['S'],ev['H'],ev['card'],ev['cell'];n=len(ev['y']);c=self.chunk;out={k:[] for k in PER+PER_Q}
        with torch.no_grad():
            for i in range(0,n,c):
                s,h,cd=S[i:i+c],H[i:i+c],card[i:i+c];tk=token_logp(pol,s,h,cd,cell[i:i+c]);menu=pol.menu_logp(s,h);rm=self.ref.menu_logp(s,h)
                p=menu.exp();ok=self.sup[h.clamp(min=0)]&(h>=0)[:,:,None];mc=menu.clamp(min=-30);lc,_=pol.card_logp(s,h)
                for k,v in (('lp',tk.sum(1)),('lpc',tk[:,0]),('kl',(p*(mc-rm.clamp(min=-30))).sum((1,2))),('ent',-(p*mc).sum((1,2))),
                            ('top',(lc.argmax(1)==cd).float()),('mass',(p*ok).sum((1,2)))):out[k].append(v)
                if i<self.q_n:
                    m=min(c,self.q_n-i);pm=p[:m];q=self.q[i:i+m];okq=self.okq[i:i+m];oks=self.oks[i:i+m]
                    for k,v in (('qnum',(pm*okq*q).sum((1,2))),('qmass',(pm*okq).sum((1,2))),('qdir',(pm*q*self.valid[i:i+m]).sum((1,2))),
                                ('mass_s',(pm*oks).sum((1,2))),('qnum_s',(pm*oks*q).sum((1,2))),('qtop',pm.flatten(1).gather(1,self.top[i:i+m,None])[:,0])):
                        out[k].append(v)
        return {k:torch.cat(v).numpy().astype(np.float32) for k,v in out.items()}


def terms(r,fx):
    # each metric of train.singleTraj.evaluate as a signed sum of ratios of per-record sums: {metric: [(sign, numerator, denominator)]}
    n=len(r['lp']);m=int(fx['q_n']);won=fx['won'].astype(np.float64);lost=1-won;one=np.ones(n);f=lambda x:x.astype(np.float64)
    pad=lambda x:np.concatenate([f(x),np.zeros(n-m)]);iq=pad(np.ones(m));qm=f(r['qmass']);cov=(qm>1e-6).astype(np.float64);ms=f(r['mass_s'])
    cs=(ms>1e-6).astype(np.float64);qn=f(r['qnum'])
    qsup=np.where(cov>0,qn/np.where(cov>0,qm,1.0),0.0);qpess=(qn+(1-qm)*f(fx['qmin']))*cov;qps=(f(r['qnum_s'])+(1-ms)*f(fx['qmin_s']))*cs
    gap=lambda x:[(1,f(x)*won,won),(-1,f(x)*lost,lost)]
    return {'logp':[(1,f(r['lp']),one)],'winner_gap':gap(r['lp']),'winner_gap_card':gap(r['lpc']),'entropy_gap':gap(r['ent']),
            'top1_won':[(1,f(r['top'])*won,won)],'top1_lost':[(1,f(r['top'])*lost,lost)],'kl_to_bc':[(1,f(r['kl']),one)],'entropy':[(1,f(r['ent']),one)],
            'support_mass':[(1,f(r['mass']),one)],'q_support':[(1,pad(qsup),pad(cov))],'q_pess':[(1,pad(qpess),pad(cov))],'q_direct':[(1,pad(r['qdir']),iq)],
            'support_mass_strict':[(1,pad(ms),iq)],'q_pess_strict':[(1,pad(qps),pad(cs))],'q_top_mass':[(1,pad(r['qtop']),iq)]}


def by_game(tm,game,G):
    # per-game sums of every numerator and denominator
    return {k:[(s,np.bincount(game,a,G),np.bincount(game,b,G)) for s,a,b in v] for k,v in tm.items()}


def value(gs,W=None):
    # each metric on the full sample (W None) or on each resample (rows of W, game multiplicities): floats or (B,) arrays
    flat=[(k,s) for k,v in gs.items() for s,_,_ in v];M=np.stack([x for v in gs.values() for _,a,b in v for x in (a,b)],1)
    tot=M.sum(0)[None,:] if W is None else W@M;out={}
    for j,(k,s) in enumerate(flat):out[k]=out.get(k,0.0)+s*tot[:,2*j]/np.maximum(tot[:,2*j+1],1e-12)
    return {k:float(v[0]) for k,v in out.items()} if W is None else out


def boot_weights(G,B=2000,seed=0):
    # game multiplicities of B resamples of the G held-out games with replacement
    rng=np.random.default_rng(seed);return np.stack([np.bincount(rng.integers(0,G,G),minlength=G) for _ in range(B)]).astype(np.float64)


def arm_seed(spec,rep):
    # registry name and seed of a saved policy: the spec without its seed (a simgroup report's own seed), rolled/ for the rolled slice,
    # @order<k> when a run's data order differs from its seed (the earlier time-order run of seed 1)
    seed=0;keep=[]
    for p in spec.split(':'):
        if p.startswith('seed='):seed=int(p[5:])
        else:keep.append(p)
    name=':'.join(keep)
    if 'seed' in rep:seed=int(rep['seed'])
    elif rep.get('order',0)!=seed:name+=f"@order{rep.get('order',0)}"
    if rep.get('slice')=='rolled':name='rolled/'+name
    return name,seed


def score(pack,prep,runs,out,hidden=128,q_n=30000,threads=8,log=print):
    # per-record held-out metrics of every saved policy of the given run reports (and the clone), one npz per policy
    torch.set_num_threads(threads);c=Corpus(pack);P=Path(prep);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    bc=Policy(c.n_state,c.n_card,hidden);bc.load_state_dict(torch.load(P/'bc_warm.pt'));bc.eval()
    Q=Head(c.n_state,c.n_card,hidden=hidden);Q.load_state_dict(torch.load(P/'q.pt'));Q.eval();E=torch.load(P/'evals.pt',weights_only=False)
    ev=eval_set(c,E['rows']['heldB']);strict=support(c,np.concatenate([c.records(c.games('warm')),c.records(c.games('stream'))]),100)
    sc=Scorer(bc,Q,ev,E['sup']['final'],strict,q_n);game=c.a['game'][ev['rows']].astype(np.int64)
    np.savez(out/'fixed.npz',game=game,**sc.fixed);np.savez(out/'bc__s0.npz',**sc(bc));log('bc',flush=True)
    for p in runs:
        rep=json.loads(Path(p).read_text());pt=Path(p).with_suffix('.pt')
        if not pt.exists():continue
        for spec,sd in torch.load(pt).items():
            name,seed=arm_seed(spec,rep);f=out/f"{name.replace('/','+')}__s{seed}.npz"
            if f.exists():continue
            pol=Policy(c.n_state,c.n_card,hidden);pol.load_state_dict(sd);pol.eval();np.savez(f,**sc(pol));log(name,seed,flush=True)


def load_records(d):
    # {arm: {seed: per-record arrays}} and the fixed arrays from a score directory
    d=Path(d);fx=dict(np.load(d/'fixed.npz'));recs={}
    for f in sorted(d.glob('*__s*.npz')):
        m=re.match(r'(.+)__s(\d+)\.npz$',f.name);recs.setdefault(m.group(1).replace('+','/'),{})[int(m.group(2))]=dict(np.load(f))
    return recs,fx


def summarise(recs,fx,B=2000,seed=0,exploit=None):
    # per arm and metric: seed values, mean, t interval, game-bootstrap interval of the seed mean (gains against the clone for the
    # Q-evals), the floors and the registered verdicts
    game=fx['game'];ug,gi=np.unique(game,return_inverse=True);G=len(ug);W=boot_weights(G,B,seed);gs={};full={};boot={}
    for arm,ss in recs.items():
        for s,r in ss.items():
            g=by_game(terms(r,fx),gi,G);gs[(arm,s)]=g;full[(arm,s)]=value(g);boot[(arm,s)]=value(g,W)
    bcf=full[('bc',0)];bcb=boot[('bc',0)];arms={}
    def gain(arm,s,k):
        return full[(arm,s)][k]-(bcf[k] if k in GAINS else 0.0)
    def gain_boot(arm,seeds,k):
        return np.mean([boot[(arm,s)][k]-(bcb[k] if k in GAINS else 0.0) for s in seeds],0)
    for arm,ss in recs.items():
        if arm=='bc':continue
        seeds=sorted(ss);row={'seeds':seeds}
        for k in REPORTED:
            iv=interval([gain(arm,s,k) for s in seeds]);bb=gain_boot(arm,seeds,k);iv['boot']=[float(np.percentile(bb,2.5)),float(np.percentile(bb,97.5))]
            row[k]=iv
        flags=[(full[(arm,s)]['q_direct']-bcf['q_direct'])-(full[(arm,s)]['q_pess']-bcf['q_pess'])>0.002
               or full[(arm,s)]['support_mass']<bcf['support_mass']-0.01 for s in seeds]
        row['leave_flags']=int(sum(flags));row['support_drop']=bcf['support_mass']-row['support_mass']['mean']
        row['leave_flag_mean']=(row['q_direct']['mean']-row['q_pess']['mean'])>0.002 or row['support_drop']>0.01
        arms[arm]=row
    floors={a:{'F':floor(arms[a]['q_pess']['values']),'mean':arms[a]['q_pess']['mean'],'sd':arms[a]['q_pess']['sd'],'n':arms[a]['q_pess']['n']}
            for a in arms if a.endswith(':flip')}
    mf=[floors[a]['F'] for a in MODEL_FREE if a in floors];F_max=max(f['F'] for f in floors.values()) if floors else math.nan
    flips=[v for a in arms if a.endswith(':flip') for v in arms[a]['q_pess']['values']];F_old=max(abs(v) for v in flips) if flips else math.nan
    verdicts={}
    for arm,fa in FLOOR_OF.items():
        if arm not in arms:continue
        F=min(mf) if fa=='min_mf' else floors[fa]['F'] if fa in floors else math.nan;g=arms[arm]['q_pess']
        ok=arms[arm]['support_drop']<=0.01 and not arms[arm]['leave_flag_mean']
        v={'floor_arm':fa,'F':F,'F_max':F_max,'gain':g['mean'],'t_lo':g['lo'],'boot_lo':g['boot'][0],'support_ok':bool(ok),
           'seeds_above_F':int(sum(x>F for x in g['values'])),'seeds_above_F_max':int(sum(x>F_max for x in g['values'])),'n':g['n'],
           'improves':bool(g['lo']>F and g['boot'][0]>F and ok),'improves_vs_F_max':bool(g['lo']>F_max and g['boot'][0]>F_max and ok),
           'old_rule_seeds_above_2F':int(sum(x>2*F_old for x in g['values'])),'old_rule_mean_above_2F':bool(g['mean']>2*F_old)}
        if exploit:
            ex=[exploit.get(f'{arm}__s{s}') for s in arms[arm]['seeds']]
            v['exploit_flags']=int(sum(1 for e,x in zip(ex,g['values']) if e and (e['gain_train_H4']>2*x or e['gain_held_out_H4']<0.5*e['gain_train_H4'])))
            v['exploit_ratio']=[round(e['gain_train_H4']/x,2) if e and x else None for e,x in zip(ex,g['values'])]
        verdicts[arm]=v
    comps={}
    for a,b,label in COMPARE:
        if a not in arms or b not in arms:continue
        common=sorted(set(arms[a]['seeds'])&set(arms[b]['seeds']))
        if len(common)<2:continue
        d=[gain(a,s,'q_pess')-gain(b,s,'q_pess') for s in common];iv=interval(d);bb=gain_boot(a,common,'q_pess')-gain_boot(b,common,'q_pess')
        fa,fb=FLOOR_OF.get(a),FLOOR_OF.get(b);parts=[d];est=iv['mean']
        if fa in floors and fb in floors and fa!=fb:
            parts+=[arms[fa]['q_pess']['values'],arms[fb]['q_pess']['values']];est-=floors[fa]['mean']-floors[fb]['mean']
        ex=welch(parts,est);bl=[float(np.percentile(bb,2.5)),float(np.percentile(bb,97.5))]
        verdict=('beats' if iv['lo']>0 and bl[0]>0 and ex['lo']>0 else 'worse' if iv['hi']<0 else 'no difference' if iv['lo']<=0<=iv['hi']
                 else 'higher, not beyond the floors')
        comps[label]={'a':a,'b':b,'seeds':common,'d':iv,'boot':bl,'excess':ex,'verdict':verdict,'old_rule_beats':bool(iv['mean']>F_old)}
    return {'bc':{k:bcf[k] for k in REPORTED},'arms':arms,'floors':floors,'F_max':F_max,'F_min_model_free':min(mf) if mf else math.nan,'F_old':F_old,
            'verdicts':verdicts,'compare':comps,'bootstrap':{'B':B,'games':int(G),'q_records':int(fx['q_n']),'records':int(len(game))}}


def exploit(pack,prep,wm_dir,runs,out,hidden=128,n_gain=4000,threads=16,log=print):
    # f5057's exploitation check per saved policy: in-model value gain over the clone under the training members and the kept-out
    # member (H = 4, behaviour continuation) on the held-out states train.learnedWm.diagnose draws
    from train.learnedWm import HELD_OUT,MEMBERS,load_aux,load_member,model_gain
    torch.set_num_threads(threads);c=Corpus(pack);wm=Path(wm_dir);ax=load_aux(wm/'aux.npz');P=Path(prep)
    rng=np.random.default_rng(0);hb=c.records(c.games('heldB'));rng.choice(hb,min(20000,len(hb)),replace=False)
    dr=np.sort(rng.choice(hb,min(20000,len(hb)),replace=False));rng.choice(len(dr),min(20000,len(dr)),replace=False)
    r=dr[rng.choice(len(dr),min(n_gain,len(dr)),replace=False)];S=c.S(0,0,r);H=torch.from_numpy(c.a['H'][r].astype(np.int64));qu=torch.from_numpy(ax['qu'][r].astype(np.int64))
    bc=Policy(c.n_state,c.n_card,hidden);bc.load_state_dict(torch.load(P/'bc_warm.pt'));bc.eval()
    sets={'train_H4':[load_member(wm/f'member{i}.pt') for i in MEMBERS],'held_out_H4':[load_member(wm/f'member{HELD_OUT}.pt')]}
    res=json.loads(Path(out).read_text()) if Path(out).exists() else {}
    for p in runs:
        rep=json.loads(Path(p).read_text());pt=Path(p).with_suffix('.pt')
        for spec,sd in torch.load(pt).items():
            name,seed=arm_seed(spec,rep);key=f'{name}__s{seed}'
            if key in res:continue
            pol=Policy(c.n_state,c.n_card,hidden);pol.load_state_dict(sd);pol.eval();e={}
            for k,ms in sets.items():vp,vb=model_gain(ms,S,H,qu,pol,bc,16,4,0);e.update({f'gain_{k}':vp-vb,f'value_{k}':vp,f'value_bc_{k}':vb})
            res[key]=e;Path(out).write_text(json.dumps(res,indent=1)+'\n');log(key,e,flush=True)
    return res


def cosines(pack,prep,wm_dir,cf,seeds,out,hidden=128,threads=16,log=print):
    # Q1's direction check per seed on the simulator's cf051 menus: menu-advantage cosine against the half-A real-outcome Q of the
    # simulator, of the learned model (rollout seed k) and of its copy with outcome heads refit on flip seed k
    from train.learnedWm import MEMBERS,cf_direction,load_aux,load_member
    torch.set_num_threads(threads);c=Corpus(pack);wm=Path(wm_dir);ax=load_aux(wm/'aux.npz');P=Path(prep)
    Q=Head(c.n_state,c.n_card,hidden=hidden);Q.load_state_dict(torch.load(P/'q.pt'));Q.eval();train=[load_member(wm/f'member{i}.pt') for i in MEMBERS];res={}
    for k in seeds:
        res[str(k)]=cf_direction(c,ax,train,[load_member(wm/f'member{i}_flip{k}.pt') for i in MEMBERS],Q,cf,4,4,k)
        Path(out).write_text(json.dumps(res,indent=1)+'\n');log(k,res[str(k)],flush=True)
    return res


def q1_direction(cos):
    # registered Q1 direction rule: the learned-minus-simulator cosine (seed mean) against the flipped copy's mean plus two sd
    ex=[v['cos_wm_realQ']-v['cos_sim_realQ'] for v in cos.values()];fl=[v['cos_wmflip_realQ'] for v in cos.values()]
    return {'excess':interval(ex),'flipped':interval(fl),'flipped_floor':floor(fl),'passes':bool(np.mean(ex)>floor(fl))}


def table(s,cols=('q_pess','q_pess_strict','winner_gap','winner_gap_card','kl_to_bc','entropy','support_mass','support_mass_strict','q_top_mass')):
    # markdown rows: seed mean and t half-width per metric, the bootstrap interval of the Q-eval gain, the floor and seed counts
    f=lambda v,d=4:f'{v:.{d}f}'.replace('0.','.',1) if abs(v)<1 else f'{v:.{d}f}'
    rows=['| arm | seeds | '+' | '.join(cols)+' | boot dq pess | floor F (arm) | seeds > F | verdict |','|'+'---|'*(len(cols)+6)]
    for arm,row in sorted(s['arms'].items()):
        v=s['verdicts'].get(arm);cells=[arm,str(row['q_pess']['n'])]
        for k in cols:
            iv=row[k];cells.append(f"{f(iv['mean'])} +- {f(iv['hi']-iv['mean'])}" if iv['n']>1 else f(iv['mean']))
        cells.append(f"[{f(row['q_pess']['boot'][0])}, {f(row['q_pess']['boot'][1])}]")
        if v:cells+=[f"{f(v['F'])} ({v['floor_arm']})",f"{v['seeds_above_F']}/{v['n']}",'improves' if v['improves'] else 'no']
        elif arm in s['floors']:cells+=[f"F = {f(s['floors'][arm]['F'])}",'','floor']
        else:cells+=['','','']
        rows.append('| '+' | '.join(cells)+' |')
    return '\n'.join(rows)


def main():
    ap=argparse.ArgumentParser();sp=ap.add_subparsers(dest='cmd',required=True)
    a=sp.add_parser('rec');a.add_argument('--pack',required=True);a.add_argument('--prep',required=True);a.add_argument('--runs',nargs='+',required=True)
    a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=8)
    a=sp.add_parser('exploit');a.add_argument('--pack',required=True);a.add_argument('--prep',required=True);a.add_argument('--wm',required=True)
    a.add_argument('--runs',nargs='+',required=True);a.add_argument('--out',required=True);a.add_argument('--threads',type=int,default=16)
    a=sp.add_parser('cos');a.add_argument('--pack',required=True);a.add_argument('--prep',required=True);a.add_argument('--wm',required=True)
    a.add_argument('--cf',required=True);a.add_argument('--seeds',nargs='+',type=int,default=[0,1,2,3,4]);a.add_argument('--out',required=True)
    a.add_argument('--threads',type=int,default=16)
    a=sp.add_parser('report');a.add_argument('--rec',required=True);a.add_argument('--out',required=True);a.add_argument('--exploit');a.add_argument('--cos')
    a.add_argument('--B',type=int,default=2000);a=ap.parse_args()
    if a.cmd=='rec':score(a.pack,a.prep,a.runs,a.out,threads=a.threads)
    elif a.cmd=='exploit':exploit(a.pack,a.prep,a.wm,a.runs,a.out,threads=a.threads)
    elif a.cmd=='cos':cosines(a.pack,a.prep,a.wm,a.cf,a.seeds,a.out,threads=a.threads)
    else:
        recs,fx=load_records(a.rec);ex=json.loads(Path(a.exploit).read_text()) if a.exploit and Path(a.exploit).exists() else None
        s=summarise(recs,fx,a.B,exploit=ex)
        if a.cos and Path(a.cos).exists():s['q1_direction']=q1_direction(json.loads(Path(a.cos).read_text()))
        Path(a.out).write_text(json.dumps(s,indent=1)+'\n');print(table(s))
        for k,v in s['compare'].items():print(k,v['verdict'],{x:round(v['d'][x],5) for x in ('mean','lo','hi')},[round(x,5) for x in v['boot']],
                                              {x:round(v['excess'][x],5) for x in ('mean','lo','hi')})


if __name__=='__main__':main()
