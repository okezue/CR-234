"""Per-card split of the world model's judgement against the real outcome.

For every counterfactual group (train.counterfactual) the recorded play has a world-model advantage, its simulated return minus the
mean of its menu, and a real advantage, the recorded outcome minus a state-value baseline fit on the games outside the rollout set
(the actor's plain winner rate is kept beside it). Both are averaged per card variant of the recorded play (evolved and hero
variants apart, the naming of the calibrated head), after standardising each to unit variance over decisions so the two are
comparable, and the per-card gap (world model minus real) is rank-correlated with the calibrated head's per-card bias table
(simulated minus recorded holder win rate) and per-card correction (local/calibration/head1.json), with permutation p-values. The
hypothesis: the cards the calibrated head distrusts are the cards where the world model's counterfactual judgement of a play
disagrees with the real outcome.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from train.calibratedHead import spearman
from train.counterfactual import load_groups
from train.traceRl import Head,fit_head,prepare,stack


def variant(card,evolved=False,hero=False):
    # the calibrated head's deck naming: hyphens, evolved slot -ev1, hero slot -hero
    return card.replace('_','-')+('-ev1' if evolved else '-hero' if hero else '')


def recorded_advantages(d,groups):
    # record index, world-model advantage of the recorded play and its simulated win, for groups with at least two rollouts
    where={(g,int(i)):r for r,(g,i) in enumerate(zip(d['gid'],d['idx']))};rows=[]
    for key,plays in groups.items():
        r=where.get(key)
        if r is None or len(plays)<2:continue
        rets=np.array([p[5] for p in plays]);rows.append((r,rets[0]-rets.mean(),plays[0][3]))
    ri=np.array([r for r,_,_ in rows],np.int64)
    return ri,np.array([a for _,a,_ in rows],np.float32),np.array([w for _,_,w in rows],np.float32)


def perm_p(a,b,perms=2000,seed=0):
    # two-sided permutation p-value of the rank correlation
    rng=np.random.default_rng(seed);obs=spearman(a,b);null=np.array([spearman(a,rng.permutation(b)) for _ in range(perms)])
    return round(float((np.abs(null)>=abs(obs)-1e-12).mean()),4)


def table(names,adv_wm,adv_real,y,min_n):
    # per-variant means of the standardised advantages; the gap is world model minus real
    z_wm=(adv_wm-adv_wm.mean())/(adv_wm.std()+1e-9);z_real=(adv_real-adv_real.mean())/(adv_real.std()+1e-9);out={}
    for c in sorted(set(names)):
        m=names==c
        if m.sum()<min_n:continue
        out[c]={'n':int(m.sum()),'adv_wm':round(float(adv_wm[m].mean()),4),'adv_real':round(float(adv_real[m].mean()),4),'win_rate':round(float(y[m].mean()),4),
                'z_wm':round(float(z_wm[m].mean()),4),'z_real':round(float(z_real[m].mean()),4),'gap':round(float(z_wm[m].mean()-z_real[m].mean()),4)}
    return out


def compare(cards,head,perms=2000,seed=0):
    # rank correlations across cards of the world-model and real advantages, and of the gap with the head's bias and correction
    names=list(cards);wm=np.array([cards[c]['z_wm'] for c in names]);real=np.array([cards[c]['z_real'] for c in names])
    win=np.array([cards[c]['win_rate'] for c in names])
    out={'cards':len(names),'spearman_wm_vs_real':round(spearman(wm,real),4),'p_wm_vs_real':perm_p(wm,real,perms,seed),
         'pearson_wm_vs_real':round(float(np.corrcoef(wm,real)[0,1]),4),'spearman_wm_vs_win_rate':round(spearman(wm,win),4)}
    common=[c for c in names if c in head];out['cards_in_head']=len(common)
    if len(common)>=5:
        g=np.array([cards[c]['gap'] for c in common]);bias=np.array([head[c]['bias'] for c in common]);corr=np.array([head[c]['correction'] for c in common])
        w=np.array([cards[c]['z_wm'] for c in common])
        out.update({'spearman_gap_vs_bias':round(spearman(g,bias),4),'p_gap_vs_bias':perm_p(g,bias,perms,seed),
                    'spearman_gap_vs_correction':round(spearman(g,corr),4),'p_gap_vs_correction':perm_p(g,corr,perms,seed),
                    'spearman_wm_vs_bias':round(spearman(w,bias),4),'spearman_wm_vs_correction':round(spearman(w,corr),4)})
        order=sorted(common,key=lambda c:head[c]['correction']);k=max(5,len(common)//5)
        out['gap_most_distrusted']=round(float(np.mean([cards[c]['gap'] for c in order[:k]])),4)
        out['gap_most_trusted']=round(float(np.mean([cards[c]['gap'] for c in order[-k:]])),4);out['tail_cards']=k
    return out


def run(npzs,cf,head_path,out=None,min_n=100,head_epochs=8,head_l2=1e-3,hidden=128,seed=0,perms=2000):
    cols,X,games=stack(npzs);d=prepare(cols,X,games);groups=load_groups(cf);head=json.loads(Path(head_path).read_text())['cards']
    cf_games={g for g,_ in groups};base=torch.tensor([g not in cf_games for g in d['gid']])
    V=fit_head(Head(d['n_state'],hidden=hidden),d['S'][base],d['y'][base],epochs=head_epochs,l2=head_l2,seed=seed)
    ri,adv_wm,sim_win=recorded_advantages(d,groups);ti=torch.tensor(ri);y=d['y'][ti].numpy()
    with torch.no_grad():v=torch.sigmoid(V(d['S'][ti])).numpy()
    adv_real=y-v;names=np.array([variant(d['vocab'][c],e,h) for c,e,h in zip(d['card'][ti].tolist(),d['evolved'][ri],d['hero'][ri])])
    cards=table(names,adv_wm,adv_real,y,min_n)
    report={'groups':int(len(ri)),'rollout_games':len(cf_games),'baseline_games':len(set(d['gid'][base.numpy()])),'min_n':min_n,
            'decision_corr_wm_vs_real':round(float(np.corrcoef(adv_wm,adv_real)[0,1]),4),'decision_corr_wm_vs_y':round(float(np.corrcoef(adv_wm,y)[0,1]),4),
            'sim_win_predicts_outcome':round(float(((sim_win>0)==(y>0.5)).mean()),4),'baseline_mean':round(float(v.mean()),4),'baseline_std':round(float(v.std()),4),
            'compare':compare(cards,head,perms,seed),
            'cards':{c:{**row,'bias':head.get(c,{}).get('bias'),'correction':head.get(c,{}).get('correction')}
                     for c,row in sorted(cards.items(),key=lambda kv:-kv[1]['gap'])}}
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=1)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--npz',nargs='+',required=True);ap.add_argument('--cf',required=True);ap.add_argument('--head',required=True)
    ap.add_argument('--out');ap.add_argument('--min_n',type=int,default=100);ap.add_argument('--head_epochs',type=int,default=8);ap.add_argument('--head_l2',type=float,default=1e-3)
    ap.add_argument('--hidden',type=int,default=128);ap.add_argument('--seed',type=int,default=0);ap.add_argument('--threads',type=int,default=0);a=ap.parse_args()
    if a.threads:torch.set_num_threads(a.threads)
    r=run(a.npz,a.cf,a.head,a.out,a.min_n,a.head_epochs,a.head_l2,a.hidden,a.seed)
    print(json.dumps({k:v for k,v in r.items() if k!='cards'},indent=1))
    print('card n adv_wm adv_real win_rate gap bias correction')
    for c,row in r['cards'].items():print(c,row['n'],row['adv_wm'],row['adv_real'],row['win_rate'],row['gap'],row['bias'],row['correction'])


if __name__=='__main__':main()
