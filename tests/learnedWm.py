import json
import math
from datetime import datetime,timezone

import numpy as np
import pytest
import torch

import train.learnedWm as LW
from sim.game import Game
from train.corpusStates import EXTRA
from train.counterfactual import CELLS,cell_of,own_cells
from train.decisionStates import COLS
from train.feats import FEAT_DIM,FLAT_DIM,GAME_FEAT,TOWER_FEAT,UNIT_FEAT,featurize
from train.singleTraj import Corpus,pack,prep,run_arms
from train.traceRl import Policy

# Planted-dynamics games through the shipped pack, auxiliary targets, world-model fit, rollouts and arms. Each player cycles an
# eight-card deck (four in hand, the played card to the back of the queue). A giant played now stands in the enemy half at the actor's
# next decision and, after that decision's play, takes 0.4 from a random enemy tower: its value appears in the state two decisions
# later. Every play also lets a random tower take a random hit, so tower health varies apart from the giant and its value can be
# learned. A hidden handicap the states never show lowers a player's chance and makes it play the giant far more often, so in the
# recorded outcomes the giant and its standing unit look bad: the one-step Q (H = 0) and one dynamics step (H = 1) inherit the
# confounding, and only two decisions of learned dynamics reach the tower damage whose value the outcome model learned elsewhere.

TEAMS=('blue','red');DECK=('knight','archers','fireball','giant','minions','zap','hog_rider','the_log')
T0=datetime(2026,8,30,tzinfo=timezone.utc).timestamp()
TOWERS=(('blue',0.5,3/32),('blue',3.5/18,6.5/32),('blue',14.5/18,6.5/32),('red',0.5,29/32),('red',3.5/18,25.5/32),('red',14.5/18,25.5/32))


def state(t,team,hp,pend,rng):
    # the actor's raw train.feats vector: time, elixir, the six towers, a standing giant in the enemy half for each pending side
    s=np.zeros(FEAT_DIM,np.float32);s[0]=t/300;s[1]=min(t/120,1);s[4]=1;s[7]=1/3;s[8]=rng.random();s[9]=rng.random()
    for k,(tm,cx,cy) in enumerate(TOWERS):i=GAME_FEAT+k*TOWER_FEAT;s[i:i+TOWER_FEAT]=[1,1,hp[tm][k%3],float(tm==team),cx,cy]
    u=FLAT_DIM
    for tm in (team,'red' if team=='blue' else 'blue'):
        if pend[tm]:s[u:u+UNIT_FEAT]=[3.5/18,(20 if tm=='blue' else 12)/32,1,1/6,0,0,0.5,0.1,float(tm==team),0.2,0,1];u+=UNIT_FEAT
    return s.astype(np.float16)


def planted(tmp_path,n_games=600,n_dec=24,seed=0,per=300,p_giant=(0.02,0.3),hidden_w=1.5,drop=0.4,tie=0.1):
    # shards in train.corpusStates format and the truth per (bid, idx): the actor's queue before its play, the opponent's first play
    # before the actor's next decision (card, cell, delay) and the delay to that next decision
    rng=np.random.default_rng(seed);out=tmp_path/'states';out.mkdir(parents=True,exist_ok=True);ts=np.sort(rng.uniform(T0,T0+8*86400,n_games));truth={}
    for k in range(0,n_games,per):
        rows={c:[] for c in COLS+EXTRA};X=[];roll=[];games=[]
        for g in range(k,min(k+per,n_games)):
            bid=f'G{g:05d}';hid={tm:bool(rng.random()<0.5) for tm in TEAMS};hp={tm:[1.0]*3 for tm in TEAMS};pend={tm:0 for tm in TEAMS}
            deck={tm:[str(x) for x in rng.permutation(DECK)] for tm in TEAMS};t=0.0;plays=[]
            for i in range(n_dec):
                team=TEAMS[i] if i<2 else TEAMS[int(rng.random()<0.5)];opp='red' if team=='blue' else 'blue'
                t=t if i and rng.random()<tie else t+float(rng.uniform(2,10));X.append(state(t,team,hp,pend,rng));hand=deck[team][:4]
                others=[c for c in hand if c!='giant']
                card='giant' if 'giant' in hand and rng.random()<p_giant[hid[team]] else str(rng.choice(others))
                x,y=CELLS[int(rng.choice(own_cells(team)))][int(rng.integers(12))];od=deck[opp]
                rec=(bid,i,t,team,card,float(x)+0.5,float(y)+0.5,False,False,1,'|'.join(hand),'|'.join(od[:4]),od[4])
                for c_,v in zip(COLS+EXTRA,rec):rows[c_].append(v)
                plays.append((team,card,t,cell_of(x+0.5,y+0.5),deck[team][4:]));deck[team].remove(card);deck[team].append(card)
                if pend[team]:j=int(rng.integers(3));hp[opp][j]=max(0.0,hp[opp][j]-drop);pend[team]=0
                if card=='giant':pend[team]=1
                tm=TEAMS[int(rng.integers(2))];j=int(rng.integers(3));hp[tm][j]=max(0.0,hp[tm][j]-float(rng.uniform(0.05,0.35)))
            d=sum(hp['blue'])-sum(hp['red'])+hidden_w*(hid['red']-hid['blue']);winner='blue' if rng.random()<1/(1+math.exp(-3*d)) else 'red'
            for i,(tm,card,tt,cl,qu) in enumerate(plays):
                nxt=next((j for j in range(i+1,n_dec) if plays[j][0]==tm),None);op=next((j for j in range(i+1,nxt or n_dec) if plays[j][0]!=tm),None)
                truth[(bid,i)]={'queue':qu,'opp':None if op is None else (plays[op][1],plays[op][3],plays[op][2]-tt),
                                'next_dt':None if nxt is None else plays[nxt][2]-tt,'n_opp':sum(plays[j][0]!=tm for j in range(i+1,nxt or n_dec))}
                roll.append((0.4,0.4,0.0,10.0))
            games.append({'bid':bid,'actual_winner':winner,'sim_winner':winner,'end_t':300.0,'premature':False,'actual_bc':1,'actual_rc':0,'sim_bc':1,'sim_rc':0,
                          'n':n_dec,'sim_blue':0.4,'sim_red':0.3,'ts':float(ts[g]),'mode':'Ranked','rolled':False})
        p=out/f'shard{k//per:03d}.npz';np.savez_compressed(p,X=np.stack(X),roll=np.array(roll,np.float32),**{c:np.array(v) for c,v in rows.items()})
        p.with_suffix('.json').write_text(json.dumps({'outcomes':games})+'\n')
    return out,truth


@pytest.fixture(scope='module')
def world(tmp_path_factory):
    # the planted corpus packed, its auxiliary targets and three members fit on its warm-up and stream games
    tmp=tmp_path_factory.mktemp('wm');states,truth=planted(tmp);d=tmp/'pack';pack(states,d);c=Corpus(d);ax=LW.aux(c,K=3);wm=tmp/'wm';wm.mkdir()
    np.savez(wm/'aux.npz',**ax);ms=[]
    for s in range(3):
        m,_=LW.fit(c,ax,LW.train_rows(c),seed=s,K=3,epochs=12,hidden=64,latent=32,lr=3e-3,batch_size=256,log=lambda *a,**k:None);m.eval()
        for p in m.parameters():p.requires_grad_(False)
        LW.save_member(m,wm/f'member{s}.pt');ms.append(m)
    return {'dir':d,'wm':wm,'c':c,'ax':ax,'truth':truth,'models':ms}


def giant_states(c,rows):
    # held-out records whose hand holds the giant and whose actor has two more decisions, and the giant's vocabulary index
    gi=c.vocab.index('giant');A=c.a;tl=A['t_len'][A['traj'][rows]]
    return rows[(A['H'][rows]==gi).any(1)&(A['pos'][rows]+2<tl)],gi


def giant_adv(models,c,ax,rows,gi,H,n_roll=8):
    # mean over states of the giant's value minus the mean value of the other hand cards, all at the recorded cell
    S=c.S(0,0,rows);Hh=torch.from_numpy(c.a['H'][rows].astype(np.int64));qu=torch.from_numpy(ax['qu'][rows].astype(np.int64))
    cell=torch.from_numpy(c.a['cell'][rows].astype(np.int64));v=[]
    for j in range(4):
        x,_=LW.values(models,S,Hh,qu,Hh[:,j],cell,H,n_roll,None,torch.Generator().manual_seed(j));v.append(x.mean(0))
    v=torch.stack(v,1);isg=Hh==gi
    return float(((v*isg).sum(1)-(v*~isg).sum(1)/3).mean())


def t_summary_reads_the_featurized_game():
    # a real simulator board from both sides: elixir, crowns, the six towers' health and unit counts by owner, lane and half
    g=Game(p1={'deck':DECK},p2={'deck':DECK})
    for team in TEAMS:g.players[team].deck.hand=['knight','archers','giant','minions']
    assert g.play_card('blue','knight',4.5,8.5)[0] and g.play_card('red','archers',14.5,21.5)[0];g.run_to(1.5)
    g.players['blue'].elixir=7.0;g.players['red'].elixir=3.0
    t=g.arena.get_tower('red','princess','left');t.hp=t.max_hp*0.25
    for team in TEAMS:
        z=dict(zip(LW.SUM,LW.summary(featurize(g,team)[None],np.array([team=='red']))[0]));opp='red' if team=='blue' else 'blue'
        assert abs(z['elixir_own']-g.players[team].elixir/10)<1e-6 and abs(z['elixir_opp']-g.players[opp].elixir/10)<1e-6 and abs(z['t']-g.t/300)<1e-6
        assert abs(z['left_opp' if team=='blue' else 'left_own']-0.25)<1e-6 and z['king_own']==1 and z['right_opp']==1
        for tm,side in ((team,'own'),(opp,'opp')):
            for lane in ('left','right'):
                for half in ('home','away'):
                    n=sum(1 for u in g.players[tm].troops if u.alive and (u.x<9)==(lane=='left') and ((u.y<16)==(team=='blue'))==(half=='home'))
                    assert abs(z[f'n_{side}_{lane}_{half}']-n/10)<1e-6,(team,side,lane,half)
        assert z['n_own_left_home' if team=='blue' else 'n_opp_left_away']>0 and z['n_opp_right_away' if team=='blue' else 'n_own_right_home']>0


def t_targets_follow_the_replay_order_and_the_deck_cycle(world):
    # the opponent's first play before the actor's next decision (ties in time broken by replay order), delays, plays in between and
    # the queue's next cards, against the generator's own record of every game
    c,ax,truth=world['c'],world['ax'],world['truth'];A=c.a;vi={x:i for i,x in enumerate(c.vocab)};n=len(A['y']);checked=0
    for r in range(0,n,7):
        tr=truth[(c.meta['bid'][int(A['game'][r])],int(A['idx'][r]))];o=tr['opp']
        assert (int(ax['oc'][r]),int(ax['ocl'][r]))==((-1,-1) if o is None else (vi[o[0]],o[1])) and abs(float(ax['odt'][r])-(0 if o is None else o[2]))<1e-4
        assert bool(ax['done'][r])==(tr['next_dt'] is None) and abs(float(ax['ndt'][r])-(tr['next_dt'] or 0))<1e-4 and int(ax['nop'][r])==tr['n_opp']
        left=int(A['t_len'][A['traj'][r]]-A['pos'][r]-1)
        assert [int(x) for x in ax['qu'][r]]==[vi[q] if j<left else -1 for j,q in enumerate(tr['queue'][:2])]
        assert ax['done'][r] or int(ax['next_self'][r])==r+1;checked+=1
    assert checked>1000 and (ax['oc']<0).any() and (ax['oc']>=0).mean()>0.4 and (ax['nop']>1).any()


def t_one_step_prediction_beats_persistence(world):
    c,ax,ms=world['c'],world['ax'],world['models'];rows=c.records(np.concatenate([c.games('heldA'),c.games('heldB')]))
    p=LW.predict(c,ax,ms,rows,K=2)
    # the towers and units after one and two decisions, the opponent's side of the board and the outcome beat the baselines; the
    # opponent's card (drawn from a hand the actor never sees) is no worse than its frequency
    for k in (1,2):assert p[f'sum_towers_ens_k{k}']<p[f'sum_towers_persist_k{k}'] and p[f'sum_units_ens_k{k}']<p[f'sum_units_persist_k{k}']
    assert p['oppcell_nll_k0']<p['oppcell_nll_marg_k0']-0.3 and p['opp_nll_k0']<p['opp_nll_marg_k0']+0.05 and p['v_logloss_k0']<p['const_logloss_k0']
    assert p['sum_time_ens_k1']<p['sum_time_persist_k1']


def t_only_the_multi_step_model_finds_the_planted_play(world):
    c,ax,ms=world['c'],world['ax'],world['models'];rows,gi=giant_states(c,c.records(np.concatenate([c.games('heldA'),c.games('heldB')])))
    adv={H:giant_adv(ms,c,ax,rows,gi,H) for H in (0,1,2)}
    # the recorded outcomes rate the giant below the other cards; two decisions of dynamics reach the tower damage and reverse that
    assert adv[0]<0 and adv[1]<0.01 and adv[2]>0.02 and adv[2]>max(adv[0],adv[1])+0.04,adv


def t_rollouts_cycle_the_hand_and_share_plays_across_members(world):
    ms=world['models'];c=world['c'];rows=np.arange(64);S=c.S(0,0,rows);seen=[]

    class Uniform(torch.nn.Module):
        # a continuation actor that records the hand it is offered and plays uniformly from it
        def menu_logp(self,h,H):
            seen.append(H.clone());ok=(H>=0).float()[:,:,None].expand(-1,-1,LW.N_CELLS)
            return torch.where(ok>0,(ok/ok.sum((1,2),keepdim=True)).log(),torch.full_like(ok,-1e9))
    H0=torch.tensor([[0,1,2,3]]*64);qu=torch.tensor([[4,5]]*64);card=torch.full((64,),1);cell=torch.zeros(64,dtype=torch.long)
    hs=[m.encode(S) for m in ms];v,u=LW.rollout(ms,hs,H0,qu,card,cell,3,[Uniform()]*3,torch.Generator().manual_seed(0))
    # after the root play (card 1) the queue's front (4) takes its slot; after the next play the queue's second card (5) takes that one
    assert torch.equal(seen[0],torch.tensor([[0,4,2,3]]*64)) and len(seen)==2*len(ms)
    h1=seen[0];h2=seen[len(ms)];assert ((h2==5).sum(1)==1).all() and ((h1!=h2).sum(1)==1).all()
    assert v.shape==(3,64) and ((v>0)&(v<1)).all() and (u>0).all()
    # identical members disagree by nothing and give one value
    v,u=LW.rollout([ms[0]]*3,[hs[0]]*3,H0,qu,card,cell,3,None,torch.Generator().manual_seed(0))
    assert torch.allclose(u,torch.zeros(64),atol=1e-5) and torch.allclose(v[0],v[2])
    v0,_=LW.rollout(ms,hs,H0,qu,card,cell,0);assert torch.allclose(v0[1],torch.sigmoid(ms[1].qv(hs[1],ms[1].play(card,cell))))


def t_wmgroup_moves_toward_the_planted_play_and_qgroup_does_not(world):
    d,c,wm,ax,ms=world['dir'],world['c'],world['wm'],world['ax'],world['models'];log=lambda *a,**k:None
    prep(d,hidden=32,bc_epochs=20,critic_epochs=1,q_epochs=5,eval_n=4000,day_n=200,rolled_pre=0,log=log,fit_batch=256)
    bc=Policy(c.n_state,c.n_card,32);bc.load_state_dict(torch.load(d/'prep'/'bc_warm.pt'));bc.eval()
    lam=LW.calibrate(c,ax,ms,bc,c.records(c.games('warm'))[:600],G=4,H=2);(wm/'lam.json').write_text(json.dumps(lam)+'\n')
    for i,m in enumerate(ms):LW.save_member(LW.refit_outcome(m,c,LW.train_rows(c),LW.flips(c,0)),wm/f'member{i}_flip0.pt')
    arms=('wmgroup:H=2:roots=1','wmgroup_mopo:H=2:roots=1','qgroup:roots=1','wmgroup:H=2:roots=1:flip')
    out=d/'runs'/'wm.json';r=run_arms(d,arms,batch_games=2,lr=3e-3,hidden=32,threads=1,log=log,out=out,make=LW.make_wm(wm,members=(0,1,2)))
    rows,gi=giant_states(c,c.records(c.games('heldB')));S=c.S(0,0,rows);Hh=torch.from_numpy(c.a['H'][rows].astype(np.int64))
    def p_giant(pol):
        with torch.no_grad():return float(pol.card_logp(S,Hh)[0][:,gi].exp().mean())
    p={k:p_giant(v) for k,v in LW.load_policies(c,[out],32).items()};b=p_giant(bc)
    # inside the learned model two decisions of dynamics move the policy toward the giant; the one-step Q moves it away, and the
    # flipped-outcome heads move it less than the model does
    assert p['wm/wmgroup:H=2:roots=1']>b+0.03 and p['wm/qgroup:roots=1']<b-0.01,(b,p)
    assert abs(p['wm/wmgroup:H=2:roots=1:flip']-b)<(p['wm/wmgroup:H=2:roots=1']-b)/2,(b,p)
    st=r['train_stats'];assert lam['lam']>0 and st['wmgroup_mopo:H=2:roots=1']['day244']['disagree']>0
    assert st['wmgroup:H=2:roots=1']['day244']['distill']>0
    assert set(r['arms'])==set(arms) and r['arms']['qgroup:roots=1']['kl_to_bc']>0
