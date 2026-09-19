import random

import numpy as np
import torch

from train.calibratedHead import bias_table, features, fit, metrics, predict, spearman, vocab_of


def synthetic(n=4000,seed=0,over='big',under='small'):
    # a world where the recorded winner follows the decks fairly, and a simulator that over-favours the holder of one card and
    # under-favours the holder of another
    rng=random.Random(seed);rows=[]
    cards=['a','b','c','d',over,under]
    for i in range(n):
        tc=rng.sample(cards,3);oc=rng.sample([c for c in cards if c not in tc],3)
        y=1.0 if rng.random()<0.5 else 0.0
        p_sim=0.7 if y>0.5 else 0.3
        if over in tc:p_sim+=0.25
        if over in oc:p_sim-=0.25
        if under in tc:p_sim-=0.25
        if under in oc:p_sim+=0.25
        sim=1.0 if rng.random()<min(max(p_sim,0.02),0.98) else -1.0
        rows.append({'bid':str(i),'tc':tc,'oc':oc,'lvl':0.0,'king':0.0,'tt':'','ot':'','sim':sim,'crowns':sim/3,'end':0.6,'premature':0.0,'y':y})
    return rows


def t_head_corrects_the_planted_simulator_bias():
    rows=synthetic();vocab=vocab_of(rows);y=torch.tensor([r['y'] for r in rows])
    wd,_=fit(features(rows,vocab,True,False),y);wb,_=fit(features(rows,vocab,True,True),y)
    corr={c:float(wb[i]-wd[i]) for i,c in enumerate(vocab)}
    bias=bias_table(rows,min_n=100)
    # the over-favoured card gets the most negative correction, the under-favoured one the most positive, and the ranks agree
    assert min(corr,key=corr.get)=='big' and max(corr,key=corr.get)=='small'
    assert bias['big']>0.1 and bias['under' if 'under' in bias else 'small']<-0.1
    assert spearman(np.array([corr[c] for c in vocab]),np.array([bias[c] for c in vocab]))<-0.7


def t_both_model_beats_the_verdict_alone_and_is_calibrated():
    rows=synthetic(seed=1);vocab=vocab_of(rows);y=torch.tensor([r['y'] for r in rows])
    ws,bs=fit(features(rows,vocab,False,True),y);wb,bb=fit(features(rows,vocab,True,True),y)
    ms=metrics(predict(ws,bs,features(rows,vocab,False,True)),y);mb=metrics(predict(wb,bb,features(rows,vocab,True,True)),y)
    assert mb['log_loss']<ms['log_loss'] and mb['ece']<0.05


def t_metrics_of_a_perfect_and_a_constant_predictor():
    y=torch.tensor([1.0,0.0,1.0,0.0]);m=metrics(torch.tensor([0.99,0.01,0.99,0.01]),y)
    assert m['accuracy']==1.0 and m['brier']<0.001 and m['ece']<0.02
    m=metrics(torch.tensor([0.5,0.5,0.5,0.5]),y)
    assert m['accuracy']==0.5 and abs(m['brier']-0.25)<1e-9 and m['ece']<1e-9
