"""Decision head, second experiment: P(the acting player wins | state, play) at every recorded decision point, with the simulator's
verdict on the same game as a witness whose weight is learned per game phase.

Records come from train.decisionStates (the state the actor saw before each recorded play, the play, the recorded winner and the
simulator's winner). Three logistic models share one design matrix: state and play only, verdict by phase only, and both; the
simulator's bare verdict is the fourth predictor. Games are held out whole. The report gives per-phase log loss, Brier score,
accuracy and calibration error, the verdict weight per phase of the combined model (how much the terminal verdict still adds once
the board is known), and the earliest phase at which the state alone predicts the recorded winner to a given accuracy.
"""
import argparse
import json
import zlib
from pathlib import Path

import numpy as np
import torch

from train.calibratedHead import fit,metrics,predict
from train.decisionStates import load
from train.feats import GAME_FEAT,N_TOWERS,TOWER_FEAT

PHASES=((0.0,60.0,'0to60'),(60.0,120.0,'60to120'),(120.0,180.0,'120to180'),(180.0,1e9,'overtime'))


def phase(t):
    return next(i for i,(lo,hi,_) in enumerate(PHASES) if lo<=t<hi)


def verdict(game,team):
    # +1 when the simulator gave the game to the actor, -1 to the opponent, 0 for a simulated draw
    w=game['sim_winner']
    return 0.0 if w not in ('blue','red') else (1.0 if w==team else -1.0)


def relative(S):
    # actor-relative tower summary a linear head cannot form from the per-slot features: own and enemy tower health sums, towers
    # standing, and their differences (slot layout from train.feats: alive, active, hp, own, x, y per tower after the game block)
    T=S[:,GAME_FEAT:GAME_FEAT+N_TOWERS*TOWER_FEAT].reshape(len(S),N_TOWERS,TOWER_FEAT)
    own=T[:,:,3];hp=T[:,:,2]*T[:,:,0];alive=T[:,:,0]
    o_hp=(hp*own).sum(1);e_hp=(hp*(1-own)).sum(1);o_up=(alive*own).sum(1);e_up=(alive*(1-own)).sum(1)
    return np.stack([o_hp/3,e_hp/3,(o_hp-e_hp)/3,o_up/3,e_up/3,(o_up-e_up)/3],1).astype(np.float32)


def design(cols,X,games,vocab=None):
    # standardized state, actor-relative tower summary, one-hot card, tile and lane, verdict by phase; drawn recorded games carry no
    # label and are dropped
    keep=[i for i,b in enumerate(cols['bid']) if games[str(b)]['actual_winner'] is not None]
    teams=[str(t) for t in cols['team'][keep]];cards=[str(c) for c in cols['card'][keep]]
    vocab=vocab or sorted(set(cards));idx={c:i for i,c in enumerate(vocab)}
    S_raw=X[keep];S=np.concatenate([S_raw,relative(S_raw)],1);mu=S.mean(0);sd=S.std(0)+1e-6;S=(S-mu)/sd
    C=np.zeros((len(keep),len(vocab)),np.float32)
    for r,c in enumerate(cards):
        if c in idx:C[r,idx[c]]=1
    x=cols['x'][keep].astype(np.float32)/18.0;y_=cols['y'][keep].astype(np.float32)/32.0
    own=np.array([yy<0.5 if t=='blue' else yy>=0.5 for yy,t in zip(y_,teams)],np.float32)
    play=np.stack([x,y_,own,(x<0.5).astype(np.float32)],1)
    ph=np.array([phase(float(t)) for t in cols['t'][keep]])
    P=np.zeros((len(keep),len(PHASES)),np.float32);P[np.arange(len(keep)),ph]=1
    v=np.array([verdict(games[str(b)],t) for b,t in zip(cols['bid'][keep],teams)],np.float32)
    V=P*v[:,None]
    y=np.array([1.0 if games[str(b)]['actual_winner']==t else 0.0 for b,t in zip(cols['bid'][keep],teams)],np.float32)
    gid=np.array([str(b) for b in cols['bid'][keep]])
    return {'state':torch.tensor(np.concatenate([S,C,play,P],1)),'verdict':torch.tensor(np.concatenate([V,P],1)),
            'both':torch.tensor(np.concatenate([S,C,play,P,V],1)),'sim_p':torch.tensor(0.5+0.5*v),'y':torch.tensor(y),'phase':ph,'gid':gid,
            'vocab':vocab,'n_state':S.shape[1],'n_card':len(vocab),'raw':S_raw}


def witness(d,S_raw):
    # how well the simulated board at decision time predicts the recorded winner and the simulator's own verdict, by phase: the tower
    # health lead and a crown lead (a tower already down in the simulation) against both labels
    R=relative(S_raw);lead=R[:,2];crown=S_raw[:,10]-S_raw[:,11];y=d['y'].numpy()>0.5;v=d['sim_p'].numpy()>0.5;out={}
    for i,(_,_,label) in enumerate(PHASES):
        m=d['phase']==i;dl=m&(np.abs(lead)>1e-6);dc=m&(np.abs(crown)>1e-6)
        if m.sum()<50:continue
        out[label]={'n':int(m.sum()),'verdict_predicts_recorded':round(float(np.mean(v[m]==y[m])),4),
                    'lead_predicts_recorded':round(float(np.mean((lead[dl]>0)==y[dl])),4) if dl.any() else None,
                    'lead_predicts_verdict':round(float(np.mean((lead[dl]>0)==v[dl])),4) if dl.any() else None,
                    'crown_lead_n':int(dc.sum()),'crown_lead_predicts_recorded':round(float(np.mean((crown[dc]>0)==y[dc])),4) if dc.any() else None}
    return out


def split(gid,holdout=0.25):
    # whole games held out by a hash of their id, so every decision of a game lands on one side
    h=np.array([zlib.crc32(g.encode())%1000/1000.0 for g in gid])
    return h>=holdout,h<holdout


def run(npz,holdout=0.25,out=None,l2=1e-3,epochs=300):
    cols,X,games=load(npz);d=design(cols,X,games);tr,te=split(d['gid'],holdout)
    report={'witness_by_phase':witness(d,d['raw']),'records':int(len(d['y'])),'train_records':int(tr.sum()),'test_records':int(te.sum()),
            'games':len(set(d['gid'])),'phases':[p[2] for p in PHASES],'models':{},'per_phase':{},'verdict_weight_by_phase':{},'earliest_phase':{}}
    fits={}
    report['train_models']={}
    for name in ('state','verdict','both'):
        w,b=fit(d[name][tr],d['y'][tr],l2=l2,epochs=epochs);fits[name]=(w,b)
        p=predict(w,b,d[name][te]);report['models'][name]=metrics(p,d['y'][te])
        report['train_models'][name]=metrics(predict(w,b,d[name][tr]),d['y'][tr])
    report['models']['simulator']=metrics(d['sim_p'][te],d['y'][te])
    for i,(_,_,label) in enumerate(PHASES):
        m=te&(d['phase']==i)
        if m.sum()<50:continue
        row={'n':int(m.sum())}
        for name,(w,b) in fits.items():row[name]=metrics(predict(w,b,d[name][m]),d['y'][m])
        row['simulator']=metrics(d['sim_p'][m],d['y'][m]);report['per_phase'][label]=row
    w,b=fits['both'];nv=d['both'].shape[1]-len(PHASES)
    report['verdict_weight_by_phase']={label:round(float(w[nv+i]),4) for i,(_,_,label) in enumerate(PHASES)}
    w,b=fits['verdict']
    report['verdict_only_weight_by_phase']={label:round(float(w[i]),4) for i,(_,_,label) in enumerate(PHASES)}
    for thr in (0.6,0.7,0.8):
        report['earliest_phase'][str(thr)]=next((label for label,row in report['per_phase'].items() if row['state']['accuracy']>=thr),None)
    if out:Path(out).parent.mkdir(parents=True,exist_ok=True);Path(out).write_text(json.dumps(report,indent=1)+'\n')
    return report


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--npz',required=True);ap.add_argument('--holdout',type=float,default=0.25)
    ap.add_argument('--out');ap.add_argument('--epochs',type=int,default=300);ap.add_argument('--l2',type=float,default=1e-3);a=ap.parse_args()
    r=run(a.npz,a.holdout,a.out,l2=a.l2,epochs=a.epochs)
    print(json.dumps({k:r[k] for k in ('records','games','models','train_models','verdict_weight_by_phase','earliest_phase','witness_by_phase')},indent=1))
    for label,row in r['per_phase'].items():
        print(label,row['n'],{k:round(v['accuracy'],3) for k,v in row.items() if k!='n'},'ll',{k:round(v['log_loss'],3) for k,v in row.items() if k!='n'})


if __name__=='__main__':main()
