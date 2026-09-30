import csv
import math
import random

import numpy as np
import torch

from train.calibratedHead import TOWER_TROOPS,features
from train.streamDeck import arrays,constant,day_split,dense,fit,holder_rates,key_hash,load,moved,score,sub,vocab_of

# Synthetic battles.csv rows in the schema the stream converter writes (the frozen-sample columns, no replays). The recorded winner
# follows a planted deck effect (the holder of 'big' wins more, of 'small' less), a planted counter ('rock' beats 'scissors' only when
# they face each other), a tower troop and the level difference; the other cards are noise. The last day is partial.

COLS=(['replayTag','player_id','timestamp','battle_ts','team_tags','opponent_tags','gameMode_name','battle_type','result','team_crowns','opp_crowns',
       'team_king_lvl','opp_king_lvl','team_tower_troop','opp_tower_troop']+[f'team_card_{i}' for i in range(8)]+[f'team_card_{i}_lvl' for i in range(8)]
      +[f'opp_card_{i}' for i in range(8)]+[f'opp_card_{i}_lvl' for i in range(8)]
      +[f'{s}_{k}' for s in ('team','opp') for k in ('king_hp','princess_hp_0','princess_hp_1')]+['has_replay','one_v_one'])
CARDS=['knight','archers','fireball','giant','musketeer','valkyrie','big','small','rock','scissors']
DAYS=['2026-09-25','2026-09-26','2026-09-27','2026-09-28','2026-09-29','2026-09-30']


def synthetic(path,n=6000,seed=0,hero=False,draws=20):
    rng=random.Random(seed);rows=[]
    for i in range(n):
        tc=rng.sample(CARDS,4);oc=rng.sample(CARDS,4);tl=rng.choice([11,12,13,14,15]);ol=rng.choice([11,12,13,14,15])
        tt=rng.choice(TOWER_TROOPS);ot=rng.choice(TOWER_TROOPS)
        z=1.2*(('big' in tc)-('big' in oc))-1.2*(('small' in tc)-('small' in oc))+1.5*(('rock' in tc and 'scissors' in oc)-('rock' in oc and 'scissors' in tc))
        z+=0.3*(tl-ol)+0.4*((tt=='dagger_duchess')-(ot=='dagger_duchess'))
        y=rng.random()<1/(1+math.exp(-z));day=DAYS[min(int(i*5.5/n),5)]
        res='D' if i<draws else ('W' if y else 'L')
        r={c:'' for c in COLS}
        r.update({'replayTag':f'B{i:06d}','player_id':'P','timestamp':f'{day} 12:00:00 UTC','battle_ts':'0','team_tags':'A','opponent_tags':'B',
                  'gameMode_name':'Ranked','battle_type':'pathOfLegend' if i%3 else 'trail','result':res,'team_crowns':1,'opp_crowns':0,
                  'team_king_lvl':tl,'opp_king_lvl':ol,'team_tower_troop':tt,'opp_tower_troop':ot,'has_replay':False,'one_v_one':True})
        for k,c in enumerate(tc):r[f'team_card_{k}']=c+('-hero' if hero and c=='giant' else '');r[f'team_card_{k}_lvl']=tl
        for k,c in enumerate(oc):r[f'opp_card_{k}']=c;r[f'opp_card_{k}_lvl']=ol
        rows.append(r)
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=COLS);w.writeheader();w.writerows(rows)
    return rows


def head_rows(rows):
    # the dict rows calibratedHead.rows_of builds, without the judge fields
    out=[]
    for r in rows:
        if r['result']=='D':continue
        tc=[r[f'team_card_{i}'] for i in range(8) if r[f'team_card_{i}']];oc=[r[f'opp_card_{i}'] for i in range(8) if r[f'opp_card_{i}']]
        tl=[float(r[f'team_card_{i}_lvl'] or 0) for i in range(8)];ol=[float(r[f'opp_card_{i}_lvl'] or 0) for i in range(8)]
        out.append({'tc':tc,'oc':oc,'lvl':np.mean(tl)-np.mean(ol),'king':float(r['team_king_lvl'])-float(r['opp_king_lvl']),'tt':r['team_tower_troop'],'ot':r['opp_tower_troop'],
                    'y':1.0 if r['result']=='W' else 0.0})
    return out


def t_dense_features_match_calibrated_head_and_draws_are_dropped(tmp_path):
    rows=synthetic(tmp_path/'b.csv',n=800);df=load(tmp_path/'b.csv');vocab=vocab_of(df);a=arrays(df,vocab)
    hr=head_rows(rows);ref=features(hr,vocab,True,False).numpy()
    assert vocab==sorted(CARDS) and len(a['y'])==len(hr)==780 and a['unknown']==0
    assert np.allclose(dense(a),ref,atol=1e-6) and np.array_equal(a['y'],np.array([r['y'] for r in hr],np.float32))


def t_day_split_holds_out_the_last_full_day_and_the_random_split_matches_it(tmp_path):
    synthetic(tmp_path/'b.csv',n=4000);a=arrays(load(tmp_path/'b.csv'),CARDS)
    tr,te,hold=day_split(a)
    assert hold=='2026-09-29' and set(a['day'][tr])=={'2026-09-25','2026-09-26','2026-09-27','2026-09-28'} and set(a['day'][te])=={hold}
    dropped=~(tr|te);assert set(a['day'][dropped])=={'2026-09-30'} and 0<dropped.sum()<te.sum()
    frac=te.sum()/(tr.sum()+te.sum());rnd=key_hash(a['key'][tr|te])<frac
    assert abs(rnd.mean()-frac)<0.03 and 0.1<frac<0.4
    torch.manual_seed(0);assert day_split(a,'2026-09-27')[2]=='2026-09-27'


def t_deck_head_recovers_the_planted_card_effect_and_the_holder_rates(tmp_path):
    torch.set_num_threads(1);synthetic(tmp_path/'b.csv');a=arrays(load(tmp_path/'b.csv'),CARDS);tr,te,_=day_split(a);tr,te=sub(a,tr),sub(a,te)
    h=fit(tr,epochs=8,batch=256);w=h.w.detach().numpy();d=h.d.detach().numpy()
    assert CARDS[int(w.argmax())]=='big' and CARDS[int(w.argmin())]=='small' and d[TOWER_TROOPS.index('dagger_duchess')]>0.15 and d[-2]+d[-1]>0.15
    assert score(h,te)['log_loss']<constant(tr['y'].mean(),te)['log_loss']-0.05
    rates=holder_rates(tr)
    big,small,knight=rates[CARDS.index('big')],rates[CARDS.index('small')],rates[CARDS.index('knight')]
    assert big['rate']>0.62 and small['rate']<0.38 and abs(knight['rate']-0.5)<4*knight['se']
    assert abs(big['se']-math.sqrt(big['rate']*(1-big['rate'])/big['n']))<1e-9
    # a card on both sides of a game counts for neither holder
    ht=(tr['T']==CARDS.index('big')).any(1);ho=(tr['O']==CARDS.index('big')).any(1);assert big['n']==int((ht^ho).sum())


def t_matchup_head_learns_the_planted_counter_and_beats_the_deck_head(tmp_path):
    torch.set_num_threads(1);synthetic(tmp_path/'b.csv',n=8000,seed=1);a=arrays(load(tmp_path/'b.csv'),CARDS);tr,te,_=day_split(a);tr,te=sub(a,tr),sub(a,te)
    hd=fit(tr,epochs=8,batch=256);hm=fit(tr,pairs=True,epochs=8,batch=256,l2_pairs=1e-4)
    A=(hm.M-hm.M.T).detach().numpy();r,s=CARDS.index('rock'),CARDS.index('scissors')
    assert A[r,s]>0.3 and A[r,s]==np.abs(A).max() and abs(A[r,s]+A[s,r])<1e-6
    assert score(hm,te)['log_loss']<score(hd,te)['log_loss']-0.01
    # swapping the sides negates the matchup logit
    T,O,X=torch.from_numpy(te['T'][:50]),torch.from_numpy(te['O'][:50]),torch.from_numpy(te['X'][:50])
    with torch.no_grad():assert torch.allclose(hm(T,O,X)-hm.b,-(hm(O,T,-X)-hm.b),atol=1e-5)


def t_hero_variants_fold_into_the_base_card(tmp_path):
    synthetic(tmp_path/'h.csv',n=300,hero=True);raw=load(tmp_path/'h.csv');folded=load(tmp_path/'h.csv',fold_hero=True)
    assert 'giant-hero' in vocab_of(raw) and 'giant' in vocab_of(raw) and vocab_of(folded)==sorted(CARDS)
    a=arrays(folded,CARDS);heroes=(raw[[f'team_card_{i}' for i in range(8)]]=='giant-hero').any(axis=1).sum()
    assert a['unknown']==0 and heroes>0 and (a['T']==CARDS.index('giant')).any(1).sum()==heroes


def t_moved_flags_only_the_card_whose_rate_shifted():
    stream={c:{'n':10000,'rate':0.5,'se':0.005} for c in CARDS};replay={c:{'n':400,'rate':0.5,'se':0.025} for c in CARDS}
    replay['big']['rate']=0.62;replay['knight']['rate']=0.55;replay.pop('rock')
    m=moved(stream,replay);assert list(m)==['big'] and m['big']['z']<-4 and 'rock' not in m
