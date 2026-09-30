"""Deck-only head and per-card holder win rates on the official-API stream.

Input is a battles.csv in the frozen-sample schema written by the scraping repository's officialToBattles converter from the
stream (1v1 competitive games, decks as hyphen names with -ev1 for evolved cards, absolute levels, tower troops, recorded result).
There are no replays, so the head sees calibratedHead's deck features only: the one-hot deck difference, the tower troop
difference, the mean card level difference and the king level difference. Two splits: by battle day (train on the days before the
held-out day, which defaults to the last full day, later rows dropped) and at random by battle key with the same test fraction.
Models: a constant, the levels alone, the deck head, and a matchup head that adds a full table of cross-side card pairs
(antisymmetric, so swapping sides negates the logit) and same-side pairs (symmetric). Every head is also scored on the replay set's
games, where hero variants fold into their base card because the official API does not expose them. The per-card recorded holder
win rate (games where exactly one side holds the card, the recorded half of calibratedHead.bias_table) comes with a binomial
standard error and is set against the replay set's rate and against the simulator's per-card bias of local/calibration/head1.json.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from train.calibratedHead import TOWER_TROOPS,metrics,rows_of,spearman

CARD_COLS=[f'{s}_card_{i}' for s in ('team','opp') for i in range(8)];LVL_COLS=[c+'_lvl' for c in CARD_COLS]
META=['replayTag','timestamp','battle_type','gameMode_name','result','team_king_lvl','opp_king_lvl','team_tower_troop','opp_tower_troop']
ND=len(TOWER_TROOPS)+2


def base(c):
    return c[:-5] if c.endswith('-hero') else c


def load(path,fold_hero=False):
    # one battle per row, draws dropped; card names stay strings until the vocabulary is fixed
    cat={c:'category' for c in CARD_COLS+['team_tower_troop','opp_tower_troop','battle_type','gameMode_name']}
    df=pd.read_csv(path,usecols=META+CARD_COLS+LVL_COLS,dtype=cat,low_memory=False)
    df=df[df['result'].isin(['W','L'])].reset_index(drop=True)
    if fold_hero:
        for c in CARD_COLS:df[c]=df[c].astype(object).map(lambda v:base(v) if isinstance(v,str) else v)
    return df


def vocab_of(df):
    return sorted({v for c in CARD_COLS for v in df[c].dropna().unique() if v})


def arrays(df,vocab):
    # card codes [N,8] per side (-1 for an empty slot or a card outside the vocabulary), dense features [N,6] in the order of
    # calibratedHead.features (tower troop difference, mean card level difference, king level difference), labels, day, key
    V=len(vocab);codes=lambda s:np.stack([pd.Categorical(df[f'{s}_card_{i}'].astype(object),categories=vocab).codes for i in range(8)],1).astype(np.int64)
    T,O=codes('team'),codes('opp')
    lv=lambda s:np.stack([pd.to_numeric(df[f'{s}_card_{i}_lvl'],errors='coerce').fillna(0).to_numpy(np.float32) for i in range(8)],1).mean(1)
    tt=df['team_tower_troop'].astype(object).fillna('');ot=df['opp_tower_troop'].astype(object).fillna('')
    X=np.zeros((len(df),ND),np.float32)
    for i,name in enumerate(TOWER_TROOPS):X[:,i]=(tt==name).to_numpy(np.float32)-(ot==name).to_numpy(np.float32)
    X[:,-2]=lv('team')-lv('opp')
    num=lambda c:pd.to_numeric(df[c],errors='coerce').fillna(0).to_numpy(np.float32)
    X[:,-1]=num('team_king_lvl')-num('opp_king_lvl')
    y=(df['result']=='W').to_numpy(np.float32);day=df['timestamp'].astype(str).str[:10].to_numpy()
    filled=lambda s:df[[f'{s}_card_{i}' for i in range(8)]].notna().to_numpy()
    return {'T':T,'O':O,'X':X,'y':y,'day':day,'key':df['replayTag'].astype(str).to_numpy(),'mode':df['battle_type'].astype(str).to_numpy(),'V':V,
            'unknown':int(((T<0)&filled('team')).sum()+((O<0)&filled('opp')).sum())}


def dense(a):
    # the same matrix calibratedHead.features(rows,vocab,True,False) builds, for checking and for small sets
    N,V=len(a['y']),a['V'];D=np.zeros((N,V),np.float32);r=np.arange(N)[:,None]
    m=a['T']>=0;np.add.at(D,(np.broadcast_to(r,a['T'].shape)[m],a['T'][m]),1.0)
    m=a['O']>=0;np.add.at(D,(np.broadcast_to(r,a['O'].shape)[m],a['O'][m]),-1.0)
    return np.concatenate([D,a['X']],1)


def key_hash(keys):
    return np.array([int(hashlib.md5(k.encode()).hexdigest()[:8],16)/0xffffffff for k in keys])


def day_split(a,holdout_day=None):
    # the last day of a file that is still being appended to is partial, so the default held-out day is the one before it
    days=sorted(set(a['day']));hold=holdout_day or (days[-2] if len(days)>1 else days[-1])
    return a['day']<hold,a['day']==hold,hold


def sub(a,m):
    return {k:(v[m] if isinstance(v,np.ndarray) else v) for k,v in a.items()}


class Head(torch.nn.Module):
    def __init__(s,V,deck=True,pairs=False):
        super().__init__();s.V=V;s.deck=deck;s.pairs=pairs
        s.w=torch.nn.Parameter(torch.zeros(V));s.d=torch.nn.Parameter(torch.zeros(ND));s.b=torch.nn.Parameter(torch.zeros(1))
        if pairs:s.M=torch.nn.Parameter(torch.zeros(V,V));s.S=torch.nn.Parameter(torch.zeros(V,V))

    def forward(s,T,O,X):
        V=s.V;Tp=torch.where(T<0,V,T);Op=torch.where(O<0,V,O)
        z=X[:,-2:]@s.d[-2:]+s.b
        if s.deck:
            wc=torch.cat([s.w,s.w.new_zeros(1)]);z=z+wc[Tp].sum(1)-wc[Op].sum(1)+X[:,:-2]@s.d[:-2]
        if s.pairs:
            A=F.pad(s.M-s.M.T,(0,1,0,1));Sy=s.S+s.S.T;Sy=F.pad(Sy-torch.diag(torch.diagonal(Sy)),(0,1,0,1))
            z=z+A[Tp[:,:,None],Op[:,None,:]].sum((1,2))+0.5*(Sy[Tp[:,:,None],Tp[:,None,:]].sum((1,2))-Sy[Op[:,:,None],Op[:,None,:]].sum((1,2)))
        return z

    def penalty(s,l2,l2_pairs):
        p=l2*((s.w*s.w).sum()+(s.d*s.d).sum())
        if s.pairs:p=p+l2_pairs*((s.M*s.M).sum()+(s.S*s.S).sum())
        return p


def tensors(a):
    return torch.from_numpy(a['T']),torch.from_numpy(a['O']),torch.from_numpy(a['X']),torch.from_numpy(a['y'])


def fit(a,deck=True,pairs=False,epochs=6,batch=8192,lr=0.05,l2=1e-3,l2_pairs=1e-5,seed=0):
    # mini-batch Adam with a linear decay of the step; l2 as in calibratedHead.fit on the mean loss
    torch.manual_seed(seed);T,O,X,y=tensors(a);N=len(y);h=Head(a['V'],deck,pairs);opt=torch.optim.Adam(h.parameters(),lr=lr)
    steps=epochs*((N+batch-1)//batch);sched=torch.optim.lr_scheduler.LambdaLR(opt,lambda t:max(0.05,1-t/steps))
    for _ in range(epochs):
        perm=torch.randperm(N)
        for i in range(0,N,batch):
            j=perm[i:i+batch];opt.zero_grad()
            loss=F.binary_cross_entropy_with_logits(h(T[j],O[j],X[j]),y[j])+h.penalty(l2,l2_pairs)
            loss.backward();opt.step();sched.step()
    return h


def predict(h,a,batch=1<<16):
    T,O,X,_=tensors(a)
    with torch.no_grad():return torch.cat([torch.sigmoid(h(T[i:i+batch],O[i:i+batch],X[i:i+batch])) for i in range(0,len(T),batch)])


def score(h,a):
    return metrics(predict(h,a),torch.from_numpy(a['y']))


def constant(p,a):
    return metrics(torch.full((len(a['y']),),float(p)),torch.from_numpy(a['y']))


def holder_rates(a,chunk=1<<19):
    # recorded holder win rate per card over games where exactly one side holds it, with the binomial standard error
    V=a['V'];n=np.zeros(V);w=np.zeros(V)
    for i in range(0,len(a['y']),chunk):
        T,O,y=a['T'][i:i+chunk],a['O'][i:i+chunk],a['y'][i:i+chunk];r=np.arange(len(y))[:,None]
        ht=np.zeros((len(y),V),bool);m=T>=0;ht[np.broadcast_to(r,T.shape)[m],T[m]]=True
        ho=np.zeros((len(y),V),bool);m=O>=0;ho[np.broadcast_to(r,O.shape)[m],O[m]]=True
        only_t=ht&~ho;only_o=ho&~ht
        n+=only_t.sum(0)+only_o.sum(0);w+=only_t[y>0.5].sum(0)+only_o[y<0.5].sum(0)
    out={}
    for c in range(V):
        if n[c]:p=w[c]/n[c];out[c]={'n':int(n[c]),'rate':p,'se':float(np.sqrt(max(p*(1-p),1e-9)/n[c]))}
    return out


def holder_counts(rows):
    # per card (games with exactly one holder, simulated holder wins, recorded holder wins) from calibratedHead.rows_of rows
    hold={}
    for r in rows:
        tc=set(r['tc']);oc=set(r['oc'])
        for c in tc-oc:h=hold.setdefault(c,[0,0,0]);h[0]+=1;h[1]+=int(r['sim']>0);h[2]+=int(r['y']>0.5)
        for c in oc-tc:h=hold.setdefault(c,[0,0,0]);h[0]+=1;h[1]+=int(r['sim']<0);h[2]+=int(r['y']<0.5)
    return hold


def moved(stream,replay,z=3.0):
    out={}
    for c,s in stream.items():
        r=replay.get(c)
        if r is None:continue
        se=float(np.sqrt(s['se']**2+r['se']**2));d=s['rate']-r['rate'];out[c]={'stream':round(s['rate'],4),'replay':round(r['rate'],4),'diff':round(d,4),'z':round(d/se,2)}
    return dict(sorted(((c,v) for c,v in out.items() if abs(v['z'])>z),key=lambda kv:-abs(kv[1]['z'])))


def per_mode(h,a):
    out={}
    for m in sorted(set(a['mode'])):
        s=sub(a,a['mode']==m)
        if len(s['y'])>=1000:out[m]=score(h,s)
    return out


def run(battles,out,pairs=(),holdout=2,head1=None,holdout_day=None,min_n=2000,epochs=6,persist=None):
    t0=time.time();df=load(battles);vocab=vocab_of(df);a=arrays(df,vocab);game_modes=df['gameMode_name'].value_counts().head(25);del df
    days={d:int(n) for d,n in zip(*np.unique(a['day'],return_counts=True))};modes={m:int(n) for m,n in zip(*np.unique(a['mode'],return_counts=True))}
    tr_m,te_m,hold=day_split(a,holdout_day);frac=te_m.mean()/(tr_m.mean()+te_m.mean())
    rnd=key_hash(a['key'])<frac;both=tr_m|te_m
    report={'battles':str(battles),'games':int(len(a['y'])),'vocab':len(vocab),'unknown_card_slots':a['unknown'],'team_win_rate':round(float(a['y'].mean()),4),'days':days,'modes':modes,
            'game_modes':{str(k):int(v) for k,v in game_modes.items()},'fit':{'epochs':epochs,'batch':8192,'lr':0.05,'l2':1e-3,'l2_pairs':1e-5},
            'day_split':{'holdout_day':hold,'train_games':int(tr_m.sum()),'test_games':int(te_m.sum()),'dropped_after':int((~both).sum())},
            'random_split':{'test_fraction':round(float(frac),4),'train_games':int((both&~rnd).sum()),'test_games':int((both&rnd).sum())},'models':{}}
    splits={'day':(sub(a,tr_m),sub(a,te_m)),'random':(sub(a,both&~rnd),sub(a,both&rnd))}
    # the replay set, hero variants folded, scored with the stream heads (cards the stream never saw are dropped from the deck)
    rep=None
    if pairs:
        frames=[load(b,fold_hero=True) for _,b in pairs];rdf=pd.concat(frames,ignore_index=True);rep=arrays(rdf,vocab);del frames,rdf
        report['replay_set']={'games':int(len(rep['y'])),'unknown_card_slots':rep['unknown'],'team_win_rate':round(float(rep['y'].mean()),4)}
    heads={}
    for name,deck,pr in (('constant',None,None),('levels',False,False),('deck',True,False),('matchup',True,True)):
        report['models'][name]={}
        for sp,(tr,te) in splits.items():
            if deck is None:
                p=tr['y'].mean();report['models'][name][sp]={'train':constant(p,tr),'test':constant(p,te)}
                if rep is not None and sp=='day':report['models'][name]['replay_set']=constant(p,rep)
                continue
            t1=time.time();h=fit(tr,deck,pr,epochs=epochs);heads[(name,sp)]=h
            report['models'][name][sp]={'train':score(h,tr),'test':score(h,te),'fit_seconds':round(time.time()-t1,1)}
            if sp=='day':
                report['models'][name]['test_by_mode']=per_mode(h,te)
                if rep is not None:report['models'][name]['replay_set']=score(h,rep)
        print(name,{sp:v['test'] for sp,v in report['models'][name].items() if sp in splits},flush=True)
    h=heads[('deck','day')];w=h.w.detach().numpy()
    report['deck_weights']={c:round(float(w[i]),4) for i,c in enumerate(vocab)}
    report['dense_weights']=dict(zip(TOWER_TROOPS+['level_diff','king_diff'],[round(float(v),4) for v in h.d.detach().numpy()]))
    report['intercept']=round(float(h.b),4)
    # per-card recorded holder win rates on the whole stream (day-split rows), against the replay set and against head1's bias
    rates={vocab[c]:v for c,v in holder_rates(sub(a,both)).items() if v['n']>=min_n}
    report['holder_rates']={c:{'n':v['n'],'rate':round(v['rate'],4),'se':round(v['se'],4)} for c,v in sorted(rates.items(),key=lambda kv:-kv[1]['rate'])}
    if rep is not None:
        rrates={vocab[c]:v for c,v in holder_rates(rep).items() if v['n']>=150}
        common=[c for c in rates if c in rrates]
        report['replay_rates']={c:{'n':v['n'],'rate':round(v['rate'],4),'se':round(v['se'],4)} for c,v in rrates.items()}
        rr=np.array([rrates[c]['rate'] for c in common]);wd=report['deck_weights']
        report['stream_vs_replay']={'cards':len(common),'spearman':spearman(np.array([rates[c]['rate'] for c in common]),rr),
                                    'spearman_stream_weight_vs_replay_rate':spearman(np.array([wd[c] for c in common]),rr),
                                    'moved_3se':moved(rates,rrates),'stream_only':sorted(set(rates)-set(rrates)),'replay_only':sorted(set(rrates)-set(rates))}
    if head1 and pairs:
        h1=json.load(open(head1));rows_by=[rows_of(j,b) for j,b in pairs]
        ct=holder_counts([r for rs in rows_by[:-holdout] for r in rs]);c6=holder_counts([r for rs in rows_by for r in rs])
        rep_bias={c:(s-r)/n for c,(n,s,r) in ct.items() if n>=150};check=max(abs(rep_bias[c]-h1['cards'][c]['bias']) for c in h1['cards'] if c in rep_bias)
        table={}
        for c,v in h1['cards'].items():
            sc=rates.get(base(c))
            if sc is None or c not in c6:continue
            n6,s6,r6=c6[c]
            sim6=s6/n6;table[c]={'bias_head1':v['bias'],'correction_head1':v['correction'],'sim_rate_replay':round(sim6,4),'sim_n':n6,'recorded_replay':round(r6/n6,4),
                                 'recorded_stream':round(sc['rate'],4),'stream_n':sc['n'],'bias_stream':round(sim6-sc['rate'],4),'folded_hero':c.endswith('-hero')}
        cs=sorted(table);col=lambda k:np.array([table[c][k] for c in cs]);b1=col('bias_head1');bs=col('bias_stream')
        b6=col('sim_rate_replay')-col('recorded_replay');corr=col('correction_head1');k=10
        top1=sorted(cs,key=lambda c:-table[c]['bias_head1'])[:k];tops=sorted(cs,key=lambda c:-table[c]['bias_stream'])[:k]
        bot1=sorted(cs,key=lambda c:table[c]['bias_head1'])[:k];bots=sorted(cs,key=lambda c:table[c]['bias_stream'])[:k]
        report['bias_vs_stream']={'head1_bias_reproduced_max_abs_diff':round(float(check),5),'cards':len(cs),'spearman_bias_head1_vs_bias_stream':spearman(b1,bs),
                                  'spearman_bias_allsix_vs_bias_stream':spearman(b6,bs),'spearman_correction_vs_bias_stream':spearman(corr,bs),'spearman_correction_vs_bias_head1':spearman(corr,b1),
                                  'spearman_bias_head1_vs_recorded_stream':spearman(b1,col('recorded_stream')),'spearman_bias_head1_vs_recorded_replay':spearman(b1,col('recorded_replay')),
                                  'top10_distrust_head1':top1,'top10_distrust_stream':tops,'top10_overlap':len(set(top1)&set(tops)),
                                  'top10_trust_head1':bot1,'top10_trust_stream':bots,'bottom10_overlap':len(set(bot1)&set(bots)),'table':table}
    report['seconds']=round(time.time()-t0,1)
    for p in [out]+([persist] if persist else []):Path(p).parent.mkdir(parents=True,exist_ok=True);Path(p).write_text(json.dumps(report,indent=1)+'\n')
    return report


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--battles',required=True);ap.add_argument('--out',required=True);ap.add_argument('--persist')
    ap.add_argument('--pairs',nargs='*',default=[],help='jsonl=battles.csv of the replay set, oldest first (the six of head1)')
    ap.add_argument('--holdout',type=int,default=2);ap.add_argument('--head1');ap.add_argument('--holdout_day');ap.add_argument('--min_n',type=int,default=2000)
    ap.add_argument('--epochs',type=int,default=6)
    a=ap.parse_args();rep=run(a.battles,a.out,[tuple(p.split('=')) for p in a.pairs],a.holdout,a.head1,a.holdout_day,a.min_n,a.epochs,persist=a.persist)
    print('games',rep['games'],'day split',rep['day_split'],'random',rep['random_split'])
    for k,v in rep['models'].items():print(k,{s:v[s]['test'] for s in ('day','random')},v.get('replay_set'))
    if 'stream_vs_replay' in rep:
        sv=rep['stream_vs_replay'];print('stream vs replay rates',sv['cards'],'cards spearman',round(sv['spearman'],3),'moved >3se',len(sv['moved_3se']))
    if 'bias_vs_stream' in rep:print({k:v for k,v in rep['bias_vs_stream'].items() if k!='table'})
