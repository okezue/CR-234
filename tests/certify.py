import json
import math

import numpy as np
import torch
import torch.nn.functional as F

import train.certify as C
import train.wmOpe as WO
from tests.wmOpe import B as WB,C as WC,Right,Toy,right_world,value
from train.simTruth import rank_corr

# Hand-built gains for the rule arithmetic, and a planted world for the claim under test: in the right-card world of tests/wmOpe.py
# (the actor wins with probability sigmoid(B (right plays - C))), the evaluator's trace half carries a spurious association (a wrong
# card on cells 0-3 wins the game), a policy trained against that half's one-step outcome model plays exactly that, its own half's
# evaluators rate it the best policy at every horizon while its true win rate is far below the behaviour's, and the evaluators of the
# clean half do not.

NAMES=['bc','a1','a2','a3','wm/qgroup','wm/wmgroup']


def toy_gains():
    # true gains t; one-step evaluators rate qgroup 3 times its share, multi-step ones wmgroup 3 times, every evaluator rates the anchors
    # in proportion (evaluator scales 0.2, 0.5, 2)
    t=np.array([0,0.02,0.05,0.10,0.20,0.07]);G={}
    for e,s,q,w in (('M.H0',0.2,3,1),('F.K0',0.5,3,1),('M.H4',0.5,1,3),('M.H64',2.0,1,3),('F.K32',0.5,1,3),('M.H1',0.5,1,1)):
        g=s*t.copy();g[4]*=q;g[5]*=w;G[e]=g
    return t,G


def t_rules_score_as_registered():
    t,G=toy_gains();an=np.array([1,2,3]);n=lambda e:G[e]/G[e][an].mean()
    # normalising by the anchors' mean gain removes the evaluator's scale; a non-positive unit gives nan
    assert np.allclose(C.normalise(G['M.H0'],an),C.normalise(7*G['M.H0'],an)) and np.allclose(n('M.H64')[an],n('M.H1')[an])
    assert np.isnan(C.normalise(-G['M.H0'],an)).all() and C.horizon('M70x1.H64')==64 and C.horizon('F70.K0')==0 and C.horizon('QE.support')==0
    bank=['M.H0','F.K0','M.H1','M.H4','M.H64','F.K32']
    # horizon_ne: qgroup leaves out the horizon-0 evaluators, wmgroup the horizon-4 one, the anchors use every evaluator
    s=C.score({'kind':'mean','bank':bank,'excl':'ne'},G,NAMES,an)
    assert math.isclose(s[4],np.mean([n(e)[4] for e in bank if C.horizon(e)!=0])) and math.isclose(s[5],np.mean([n(e)[5] for e in bank if e!='M.H4']))
    assert math.isclose(s[2],np.mean([n(e)[2] for e in bank]))
    # horizon_class: qgroup only by multi-step evaluators (rated like the anchors), wmgroup only by one-step ones
    s=C.score({'kind':'mean','bank':bank,'excl':'class'},G,NAMES,an);u=t[an].mean()
    assert math.isclose(s[4],t[4]/u) and math.isclose(s[5],t[5]/u)
    # families: mean of normalised members, then the minimum or the median; min_raw on raw gains; lower subtracts 1.645 sd
    fam=[['M.H0','F.K0'],['M.H64'],['F.K32']];f=np.stack([np.mean([n(e) for e in x],0) for x in fam])
    sc=lambda r:C.score(r,G,NAMES,an)
    assert np.allclose(sc({'kind':'min','families':fam}),f.min(0)) and np.allclose(sc({'kind':'median','families':fam}),np.median(f,0))
    assert np.allclose(sc({'kind':'min','families':fam})[4:],t[4:]/u)
    assert np.allclose(sc({'kind':'min_raw','bank':['M.H0','M.H64']}),np.minimum(G['M.H0'],G['M.H64']))
    rs={'m':{'kind':'min','families':fam},'lo':{'kind':'lower','of':'m'}};sd=np.full(6,0.1)
    assert np.allclose(C.score(rs['lo'],G,NAMES,an,rs,sd),f.min(0)-C.Z*0.1)
    # a leading bootstrap axis passes through every kind
    Gb={e:np.stack([g,2*g]) for e,g in G.items()};sb=C.score({'kind':'mean','bank':bank,'excl':'ne'},Gb,NAMES,an)
    assert sb.shape==(2,6) and np.allclose(sb[0],sb[1])
    # the registered sets name every evaluator they read, and the replication swaps the split sources
    r=C.rules();q=C.rules(sm='MB',sf='FB',qe=False,sens=('MB0','MB1','MB2','MB3'))
    assert 'QE.support' in C.needs(r) and 'QE.support' not in C.needs(q) and 'MB.H64' in C.needs(q) and len(r['same_mean']['bank'])==15
    assert r['split_class_min']['families'][0]==['MA.H0','FA.K0','QE.support'] and r['ensemble_min']['bank'][0]=='M70x1.H64'


def truth_of(t,half=0.01,no=()):
    return {a:{'dwr':(float(x),float(x-half),float(x+half)),'verdict':'no change' if a in no or x-half<=0 else 'improves'} for a,x in zip(NAMES,t)}


def t_judge_counts_regret_certification_and_overrating():
    t,G=toy_gains();an=np.array([1,2,3]);tr=truth_of(t,no=('a1',));rng=np.random.default_rng(0)
    # the one-step evaluator: picks qgroup (the true best, regret 0), overrates it 3 times; the multi-step one picks wmgroup
    s=G['M.H0'];sb=s+rng.normal(0,0.001,(500,6))*np.array([0,1,1,1,1,1]);r=C.judge(s,sb,tr,NAMES,an)
    assert r['pick']=='wm/qgroup' and r['regret']==0 and r['picks_best'] and math.isclose(r['rho|wm/qgroup'],3) and math.isclose(r['rho|wm/wmgroup'],1)
    assert not r['avoids_overrating'] and r['true_best']=='wm/qgroup' and abs(r['spearman']-rank_corr(s[1:],t[1:])[0])<1e-12
    s=G['M.H64'];r=C.judge(s,s+rng.normal(0,0.001,(500,6)),tr,NAMES,an)
    assert r['pick']=='wm/wmgroup' and math.isclose(r['regret'],13.0) and not r['picks_best'] and r['p_regret_gt5']==1 and math.isclose(r['rho|wm/wmgroup'],3)
    # certification: score minus 1.645 bootstrap sd above zero; a1 (true verdict no change) is certified falsely when its sd is small
    # and missed when its sd is large; a bound score is certified when positive
    s=G['M.H1'];sb=np.tile(s,(1000,1));sb[:,1]+=rng.normal(0,0.001,1000);r=C.judge(s,sb,tr,NAMES,an)
    assert r['false_cert']==['a1'] and 'a1' in r['certified'] and not r['missed'] and r['pick_certified'] and not r['meets']
    sb[:,1]=s[1]+rng.normal(0,0.02,1000);r=C.judge(s,sb,tr,NAMES,an);assert not r['false_cert'] and 'a1' not in r['certified'] and r['meets']
    tr2=truth_of(t);sb[:,2]=s[2]+rng.normal(0,0.02,1000);r=C.judge(s,sb,tr2,NAMES,an);assert r['missed']==['a1','a2']
    r=C.judge(s,sb,tr2,NAMES,an,bound=True);assert set(r['certified'])=={'a1','a2','a3','wm/qgroup','wm/wmgroup'}
    # rows of bootstrap scores rank like rank_corr
    S=rng.normal(size=(5,7));tt=rng.normal(size=7);assert np.allclose(C.spearman_rows(S,tt),[rank_corr(x,tt)[0] for x in S])


def t_gains_resample_whole_games():
    names=['bc','a'];game=np.array([0,0,1,1,2,2]);per={'E':{'bc':np.array([0.1,0.3,0.5,0.5,0.2,0.2]),'a':np.array([0.2,0.4,0.9,0.5,0.2,0.4])}}
    W=np.array([[1.0,1,1],[3,0,0],[0,0,3],[0,2,1]]);ratio={'R':{'bc':(np.array([1.0,2,3]),np.array([2.0,2,2])),'a':(np.array([2.0,2,4]),np.array([2.0,2,2]))}}
    pt,bt,dr=C.gains({**per,'X':{'bc':per['E']['bc']}},game,names,W,ratio)
    d=per['E']['a']-per['E']['bc'];assert dr==['X'] and math.isclose(pt['E'][1],d.mean()) and pt['E'][0]==0
    # a resample of game 0 alone is that game's mean difference; weights 2 and 1 weight whole games
    assert np.allclose(bt['E'][:,1],[d.mean(),d[:2].mean(),d[4:].mean(),(2*d[2:4].sum()+d[4:].sum())/6])
    # ratio estimates are ratios of resampled sums
    assert math.isclose(pt['R'][1],8/6-6/6) and math.isclose(bt['R'][1,1],2/2-1/2) and math.isclose(bt['R'][2,1],4/2-3/2)
    # a constant per-start gain has no bootstrap spread
    pt,bt,_=C.gains({'E':{'bc':per['E']['bc'],'a':per['E']['bc']+0.05}},game,names,C.boot_weights(3,50));assert np.allclose(bt['E'][:,1],0.05)


def t_loaders_align_starts_and_labels(tmp_path):
    # the tenth step's job outputs cover more starts in another order; this round's outputs carry their labels; the Q-eval's records
    # map to games
    rows=np.array([30,10,20]);wm=tmp_path/'wm';ce=tmp_path/'ce';(wm/'mb').mkdir(parents=True);(wm/'fqe').mkdir();(ce/'mb').mkdir(parents=True);(ce/'fqe').mkdir()
    (ce/'qeval').mkdir();big=np.array([20,99,10,30])
    np.savez(wm/'mb'/'v70k_cycle_held_a.npz',rows=big,c0_marks=np.array([0,64]),c0_value=np.array([[2.0,9,1,3],[20,90,10,30]]),c1_marks=np.array([64]),
             c1_value=np.array([[7.0,9,5,6]]))
    (wm/'mb'/'v70k_cycle_held_a.json').write_text(json.dumps({'name':'a','games':70000,'menus':'cycle','configs':[[0,1,2,3],[0]]}))
    np.savez(wm/'mb'/'v70k_cycle_held_b.npz',rows=np.array([20,10]),c0_marks=np.array([0]),c0_value=np.array([[1.0,2]]))
    (wm/'mb'/'v70k_cycle_held_b.json').write_text(json.dumps({'name':'b','games':70000,'menus':'cycle','configs':[[0,1,2,3]]}))
    np.savez(wm/'fqe'/'v10k_a.npz',starts=big,J0=np.array([2.0,9,1,3]),J32=np.array([4.0,9,2,6]),J5=np.zeros(4))
    (wm/'fqe'/'v10k_a.json').write_text(json.dumps({'name':'a','games':10000}))
    per=C.tenth(None,rows,wm)
    assert np.allclose(per['M70.H0']['a'],[3,1,2]) and np.allclose(per['M70.H64']['a'],[30,10,20]) and np.allclose(per['M70x1.H64']['a'],[6,5,7])
    assert 'b' not in per['M70.H0'] and np.allclose(per['F10.K32']['a'],[6,2,4]) and 'F10.K5' not in per
    np.savez(ce/'mb'/'heldA_a.npz',rows=rows[::-1],c0_marks=np.array([0,4]),c0_value=np.array([[1.0,2,3],[4,5,6]]),c1_marks=np.array([0,4]),c1_value=np.zeros((2,3)))
    (ce/'mb'/'heldA_a.json').write_text(json.dumps({'name':'a','configs':[[0,1],[0]],'labels':['MA','MA0']}))
    np.savez(ce/'fqe'/'heldA_a.npz',starts=rows,J0=np.array([1.0,2,3]),J32=np.array([4.0,5,6]))
    (ce/'fqe'/'heldA_a.json').write_text(json.dumps({'name':'a','label':'FA'}))
    per=C.ours(rows,ce);assert np.allclose(per['MA.H4']['a'],[6,5,4]) and np.allclose(per['MA0.H0']['a'],0) and np.allclose(per['FA.K32']['a'],[4,5,6])
    np.savez(ce/'qeval'/'fixed.npz',game=np.array([7,5,7,8]),qmin=np.array([0.1,0.2,0.3,0.4]))
    np.savez(ce/'qeval'/'a.npz',qnum=np.array([0.5,0.2,0.0,0.3]),qmass=np.array([1.0,0.5,0.0,0.6]));(ce/'qeval'/'a.json').write_text(json.dumps({'name':'a'}))
    q=C.qeval_ratio(ce/'qeval',{5:0,7:1});num,den=q['QE.support']['a'];pn,_=q['QE.pess']['a']
    # record 2 has no supported mass and record 3's game is not among the starts' games
    assert np.allclose(num,[0.4,0.5]) and np.allclose(den,[1,1]) and np.allclose(pn,[0.2+0.5*0.2,0.5])


# ---- the planted world, end to end through train.wmOpe's FQE and this module's rules

T=3;SPUR=4


class Greedy(torch.nn.Module):
    # a policy trained against an outcome model: nearly all its mass on the model's best play of the menu
    def __init__(self,Q,temp=0.02):
        super().__init__();self.Q=Q;self.temp=temp
    def menu_logp(self,S,H):
        q=WO.menu_qv(self.Q,S,H).clamp(1e-6,1-1e-6);z=torch.logit(q)/self.temp;return F.log_softmax(z.flatten(1),1).view_as(z)


def half(n,seed,spurious):
    # behaviour traces (uniform card and cell); with spurious, a trajectory with a wrong card on cells 0-3 is recorded as won
    X,card,cell,_,y=right_world(n,T,p=0.25,seed=seed);right=X[:,1:5].argmax(1);bad=((card!=right)&(cell<SPUR)).reshape(n,T).any(1)
    if spurious:y=np.where(np.repeat(bad,T),1.0,y).astype(np.float32)
    return Toy(X,card,cell,y,T)


def true_J(pol,n=4000,seed=5):
    # the policy's true win rate: n games of T decisions in the planted world, the outcome's probability averaged
    g=torch.Generator().manual_seed(seed);rng=np.random.default_rng(seed);k=np.zeros(n);H=torch.tile(torch.arange(4),(n,1))
    for t in range(T):
        right=rng.integers(4,size=n);X=np.zeros((n,6),np.float32);X[:,0]=t/T;X[np.arange(n),1+right]=1;X[:,5]=k/T
        with torch.no_grad():p=pol.menu_logp(torch.from_numpy(X),H).exp().flatten(1)
        k+=(torch.multinomial(p,1,generator=g)[:,0]//WO.N_CELLS).numpy()==right
    v=1/(1+np.exp(-WB*(k-WC)));return float(v.mean()),float(v.std()/math.sqrt(n))


def t_self_trained_policy_overrated_by_its_own_half_not_by_the_other():
    torch.manual_seed(0);A=half(4000,1,True);Bh=half(4000,2,False);ra=np.arange(4000*T);rb=np.arange(4000*T)
    QA=WO.fit_q0(A,ra,steps=3000,batch=512,hidden=128,seed=0);QB=WO.fit_q0(Bh,rb,steps=3000,batch=512,hidden=128,seed=0)
    X0,_,_,_,_=right_world(600,T,seed=3);S0=torch.from_numpy(X0[::T]);H0=torch.tile(torch.arange(4),(600,1))
    pols={'bc':Right(0.25),'a1':Right(0.5),'a2':Right(0.7),'a3':Right(0.9),'wm/qgroup':Greedy(QA)};names=list(pols)
    tj={a:(value(0,0,T,p.p),0.0) if isinstance(p,Right) else true_J(p) for a,p in pols.items()}
    # the policy trained against half A's outcome model never plays the right card: truly far below the behaviour
    assert tj['wm/qgroup'][0]<0.1<tj['bc'][0],tj
    tr={a:{'dwr':(tj[a][0]-tj['bc'][0],tj[a][0]-tj['bc'][0]-1.96*tj[a][1],tj[a][0]-tj['bc'][0]+1.96*tj[a][1])} for a in names}
    for a in names:tr[a]['verdict']='improves' if tr[a]['dwr'][1]>0 else 'no change'
    per={}
    for lab,c,rows,Q in (('FA',A,ra,QA),('FB',Bh,rb,QB)):
        for a,p in pols.items():
            o,_=WO.fqe(c,p,rows,S0,H0,Q,K=T-1,steps=200,batch=512,seed=0)
            for k in (0,T-1):per.setdefault(f'{lab}.K{k}',{})[a]=o['per'][k]
    W=C.boot_weights(600,300);pt,bt,_=C.gains(per,np.arange(600),names,W);an=np.array([1,2,3])
    # its own half's evaluators rate it the best policy at one step and over the whole game, the clean half's near its true level
    assert all(pt[e][4]>pt[e][3] for e in ('FA.K0','FA.K2')) and per['FA.K2']['wm/qgroup'].mean()>0.6
    assert abs(per['FB.K2']['wm/qgroup'].mean()-tj['wm/qgroup'][0])<0.1
    j=lambda r,**k:C.judge(C.score(r,pt,names,an),C.score(r,bt,names,an),tr,names,an,**k)
    own=j({'kind':'one','e':'FA.K2'});hne=j({'kind':'mean','bank':['FA.K0','FA.K2'],'excl':'ne'})
    split=j({'kind':'one','e':'FB.K2'});cmin=j({'kind':'min','families':[['FA.K0','FA.K2'],['FB.K0','FB.K2']]})
    # its own half, at a horizon other than its training one too, certifies and picks it; the clean half and the family minimum do not
    for r in (own,hne):assert r['pick']=='wm/qgroup' and r['false_cert']==['wm/qgroup'] and r['pick_false_cert'] and r['regret']>40 and not r['meets']
    for r in (split,cmin):
        assert r['pick']=='a3' and r['regret']==0 and not r['false_cert'] and set(r['certified'])=={'a1','a2','a3'} and r['spearman']==1 and r['meets']
