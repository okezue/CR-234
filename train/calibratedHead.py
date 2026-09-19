"""Calibrated decision head, first experiment: learn P(team wins) from real outcomes given the decks, the levels and the simulator's
own verdict, and read off how much the head trusts the simulator per card.

Inputs are the assessed samples: a per-game jsonl written by the replay judge (simulated winner and crowns, end time, premature flag)
joined with the frozen sample's battles.csv (decks, levels, tower troops, recorded result). The team side is the judge's blue side.
Three logistic models share one feature builder: decks only, simulator verdict only, and both. Held-out samples give log loss, Brier
score, accuracy and expected calibration error; the per-card weights of the combined model are compared with the per-card holder
bias table (simulated minus recorded holder win rate) by rank correlation.
"""
import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

TOWER_TROOPS=['tower_princess','dagger_duchess','cannoneer','royal_chef']


def rows_of(jsonl,battles):
    meta={r['replayTag']:r for r in csv.DictReader(open(battles))}
    out=[]
    for line in open(jsonl):
        a=json.loads(line);r=meta.get(a['bid'])
        if r is None or a['actual_bc']==a['actual_rc']:continue
        tc=[r[f'team_card_{i}'] for i in range(8) if r.get(f'team_card_{i}')];oc=[r[f'opp_card_{i}'] for i in range(8) if r.get(f'opp_card_{i}')]
        tl=[float(r.get(f'team_card_{i}_lvl') or 0) for i in range(8)];ol=[float(r.get(f'opp_card_{i}_lvl') or 0) for i in range(8)]
        sim=0.0 if a['sim_bc']==a['sim_rc'] else (1.0 if a['sim_winner']=='blue' else -1.0)
        out.append({'bid':a['bid'],'tc':tc,'oc':oc,'lvl':(np.mean(tl) if any(tl) else 0)-(np.mean(ol) if any(ol) else 0),
                    'king':float(r.get('team_king_lvl') or 0)-float(r.get('opp_king_lvl') or 0),
                    'tt':(r.get('team_tower_troop') or ''),'ot':(r.get('opp_tower_troop') or ''),
                    'sim':sim,'crowns':(a['sim_bc']-a['sim_rc'])/3.0,'end':a['end_t']/300.0,'premature':1.0 if a['premature'] else 0.0,
                    'y':1.0 if a['actual_winner']=='blue' else 0.0})
    return out


def vocab_of(rows):
    return sorted({c for r in rows for c in r['tc']+r['oc']})


def features(rows,vocab,use_deck=True,use_sim=True):
    idx={c:i for i,c in enumerate(vocab)};nv=len(vocab);nt=len(TOWER_TROOPS)
    cols=[]
    for r in rows:
        f=[]
        if use_deck:
            d=np.zeros(nv,np.float32)
            for c in r['tc']:
                if c in idx:d[idx[c]]+=1
            for c in r['oc']:
                if c in idx:d[idx[c]]-=1
            t=np.zeros(nt,np.float32)
            for i,name in enumerate(TOWER_TROOPS):t[i]=(r['tt']==name)-(r['ot']==name)
            f+=[d,t,np.array([r['lvl'],r['king']],np.float32)]
        if use_sim:f.append(np.array([r['sim'],r['crowns'],r['end'],r['premature'],r['sim']*r['premature']],np.float32))
        cols.append(np.concatenate(f) if f else np.zeros(1,np.float32))
    return torch.tensor(np.stack(cols))


def fit(x,y,l2=1e-3,epochs=400,lr=0.05):
    w=torch.zeros(x.shape[1],requires_grad=True);b=torch.zeros(1,requires_grad=True)
    opt=torch.optim.Adam([w,b],lr=lr)
    for _ in range(epochs):
        opt.zero_grad()
        z=x@w+b;loss=torch.nn.functional.binary_cross_entropy_with_logits(z,y)+l2*(w*w).sum()
        loss.backward();opt.step()
    return w.detach(),b.detach()


def predict(w,b,x):
    return torch.sigmoid(x@w+b)


def metrics(p,y,bins=10):
    p=p.numpy();y=y.numpy();eps=1e-6
    ll=-np.mean(y*np.log(p+eps)+(1-y)*np.log(1-p+eps));brier=np.mean((p-y)**2);acc=np.mean((p>0.5)==(y>0.5))
    edges=np.linspace(0,1,bins+1);ece=0.0
    for lo,hi in zip(edges[:-1],edges[1:]):
        m=(p>=lo)&(p<hi if hi<1 else p<=hi)
        if m.any():ece+=m.mean()*abs(p[m].mean()-y[m].mean())
    return {'log_loss':round(float(ll),4),'brier':round(float(brier),4),'accuracy':round(float(acc),4),'ece':round(float(ece),4),'n':int(len(y))}


def bias_table(rows,min_n=150):
    # simulated minus recorded holder win rate, for cards held by exactly one side
    hold={}
    for r in rows:
        tc=set(r['tc']);oc=set(r['oc'])
        for c in tc-oc:h=hold.setdefault(c,[0,0,0]);h[0]+=1;h[1]+=int(r['sim']>0);h[2]+=int(r['y']>0.5)
        for c in oc-tc:h=hold.setdefault(c,[0,0,0]);h[0]+=1;h[1]+=int(r['sim']<0);h[2]+=int(r['y']<0.5)
    return {c:(s-a)/n for c,(n,s,a) in hold.items() if n>=min_n}


def spearman(a,b):
    ra=np.argsort(np.argsort(a));rb=np.argsort(np.argsort(b))
    return float(np.corrcoef(ra,rb)[0,1])


def run(pairs,holdout,out):
    rows_by=[rows_of(j,b) for j,b in pairs]
    train=[r for rs in rows_by[:-holdout] for r in rs];test=[r for rs in rows_by[-holdout:] for r in rs]
    vocab=vocab_of(train+test);ytr=torch.tensor([r['y'] for r in train]);yte=torch.tensor([r['y'] for r in test])
    report={'train_games':len(train),'test_games':len(test),'vocab':len(vocab),'models':{}}
    fits={}
    for name,ud,us in (('deck',True,False),('sim',False,True),('both',True,True)):
        xtr=features(train,vocab,ud,us);xte=features(test,vocab,ud,us)
        w,b=fit(xtr,ytr);fits[name]=(w,b)
        report['models'][name]={'train':metrics(predict(w,b,xtr),ytr),'test':metrics(predict(w,b,xte),yte)}
    # the simulator's raw verdict as a predictor
    raw=torch.tensor([0.5+0.49*r['sim'] for r in test]);report['models']['simulator_verdict']={'test':metrics(raw,yte)}
    # how much the combined head trusts the simulator: the weight on the verdict, and per-card weights against the bias table
    w,b=fits['both'];nv=len(vocab)
    card_w={c:float(w[i]) for i,c in enumerate(vocab)}
    bias=bias_table(train);common=[c for c in vocab if c in bias]
    report['sim_verdict_weight']=float(w[nv+len(TOWER_TROOPS)+2]);report['premature_interaction_weight']=float(w[nv+len(TOWER_TROOPS)+2+4])
    # a card the simulator over-favours should receive a negative correction relative to what the verdict already says
    ws=fits['sim'][0];wd=fits['deck'][0]
    corr={c:card_w[c]-float(wd[vocab.index(c)]) for c in common}
    report['rank_correlation_card_correction_vs_bias']=spearman(np.array([corr[c] for c in common]),np.array([bias[c] for c in common]))
    # the same comparison against a bias table measured only on the held-out samples, which the head never saw
    bias_te=bias_table(test,min_n=60);common_te=[c for c in common if c in bias_te]
    report['rank_correlation_card_correction_vs_heldout_bias']=spearman(np.array([corr[c] for c in common_te]),np.array([bias_te[c] for c in common_te]))
    report['heldout_bias_cards']=len(common_te)
    report['cards']={c:{'bias':round(bias[c],4),'deck_weight':round(float(wd[vocab.index(c)]),4),'both_weight':round(card_w[c],4),
                        'correction':round(corr[c],4)} for c in common}
    report['sim_only_weights']={'verdict':float(ws[0]),'crowns':float(ws[1]),'end':float(ws[2]),'premature':float(ws[3]),'verdict_x_premature':float(ws[4])}
    Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=1)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--pairs',nargs='+',required=True,help='jsonl=battles.csv, oldest first')
    ap.add_argument('--holdout',type=int,default=2);ap.add_argument('--out',required=True);a=ap.parse_args()
    pairs=[tuple(p.split('=')) for p in a.pairs]
    rep=run(pairs,a.holdout,a.out)
    for k,v in rep['models'].items():print(k,v.get('test'))
    print('verdict weight',round(rep['sim_verdict_weight'],3),'rank corr(card correction, bias)',round(rep['rank_correlation_card_correction_vs_bias'],3),
          'held-out bias',round(rep['rank_correlation_card_correction_vs_heldout_bias'],3),'on',rep['heldout_bias_cards'],'cards')
    print('sim-only weights',{k:round(v,3) for k,v in rep['sim_only_weights'].items()})
    top=sorted(rep['cards'].items(),key=lambda kv:kv[1]['correction'])
    print('most distrusted (correction lowest):',[(c,v['correction'],v['bias']) for c,v in top[:8]])
    print('most trusted (correction highest):',[(c,v['correction'],v['bias']) for c,v in top[-8:]])
    print(math.isfinite(rep['sim_verdict_weight']))
