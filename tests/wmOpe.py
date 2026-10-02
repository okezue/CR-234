import json
import math

import numpy as np
import torch

import train.learnedWm as LW
import train.wmOpe as WO
from tests.learnedWm import DECK,T0,TEAMS,state
from train.corpusStates import EXTRA
from train.counterfactual import CELLS,own_cells
from train.decisionStates import COLS
from train.singleTraj import Corpus,pack
from train.traceRl import Head,Policy,menu_q

# Planted worlds whose policy values are known. The right-card world: an actor makes T decisions from the menu of cards 0-3, the
# state names the right card, and the actor wins with probability sigmoid(B (right plays - C)); the behaviour plays uniformly, a
# policy plays the right card with probability p, so J and every Q of a policy are binomial sums. The giant world (the dynamics of
# tests/learnedWm.py without the hidden handicap): a giant played now hits an enemy tower at the player's next decision, and the
# winner follows the towers, so playing it whenever it is in hand is worth far more than never playing it.

B,C=1.5,2.0


def sig(x):
    return 1/(1+math.exp(-x))


def binom(n,p):
    return [math.comb(n,j)*p**j*(1-p)**(n-j) for j in range(n+1)]


def value(t,count,T,p,q=0.25):
    # the chance of winning from decision t with count right plays so far, playing the right card with probability p for the rest
    # (and with probability q after the first k when the continuation changes; here one policy throughout)
    return sum(w*sig(B*(count+j-C)) for j,w in enumerate(binom(T-t,p)))


def mixed(T,p,k,q=0.25):
    # the policy for the first k decisions, the behaviour (q) for the rest, from the start
    return sum(a*b*sig(B*(i+j-C)) for i,a in enumerate(binom(k,p)) for j,b in enumerate(binom(T-k,q)))


class Right(torch.nn.Module):
    # plays the card the state names with probability p, any cell
    def __init__(self,p):
        super().__init__();self.p=p
    def menu_logp(self,S,H):
        r=S[:,1:5];pc=r*self.p+(1-r)*(1-self.p)/3;return (pc.log()-math.log(48))[:,:,None].expand(-1,-1,48).clone()


class Toy:
    # a packed-store stand-in: states, menus, plays, outcomes and contiguous trajectories of T decisions
    def __init__(self,X,card,cell,y,T):
        n=len(X);m=n//T;self.X_=X.astype(np.float32);self.n_state=X.shape[1];self.n_card=4
        self.a={'H':np.tile(np.arange(4),(n,1)),'card':card,'cell':cell,'y':y,'traj':np.repeat(np.arange(m),T),'pos':np.tile(np.arange(T),m),'t_len':np.full(m,T)}
    def S(self,a,b,rows=None):
        return torch.from_numpy(self.X_[a:b] if rows is None else self.X_[rows])


def right_world(n,T,p=0.25,seed=0,soft=False,fixed=False):
    # n trajectories of a policy playing the right card with probability p (cells uniform); per record the state, play, count before
    # the play and the outcome (the win probability itself when soft); fixed makes card 0 always the right one
    rng=np.random.default_rng(seed);X=np.zeros((n*T,6),np.float32);card=np.zeros(n*T,np.int64);cnt=np.zeros(n*T,np.int64);y=np.zeros(n*T,np.float32)
    for i in range(n):
        k=0
        for t in range(T):
            r=i*T+t;right=0 if fixed else int(rng.integers(4));X[r,0]=t/T;X[r,1+right]=1;X[r,5]=k/T;cnt[r]=k
            card[r]=right if rng.random()<p else int(rng.choice([x for x in range(4) if x!=right]));k+=int(card[r]==right)
        y[i*T:(i+1)*T]=sig(B*(k-C)) if soft else float(rng.random()<sig(B*(k-C)))
    return X,card,rng.integers(48,size=n*T),cnt,y


def t_trajectory_estimators_recover_a_planted_value():
    T=5;p=0.5;n=8000;X,card,cell,cnt,y=right_world(n,T,seed=1,soft=True);S=torch.from_numpy(X);H=torch.tile(torch.arange(4),(n*T,1))
    lp=WO.play_logp(Right(p),S,H,torch.from_numpy(card),torch.from_numpy(cell)).numpy()+math.log(4)+math.log(48)
    tr=np.repeat(np.arange(n),T);pos=np.tile(np.arange(T),n);yt=y[::T];J=value(0,0,T,p)
    right=X[:,1:5].argmax(1)==card;q=np.array([value(t+1,k+int(h),T,p) for t,k,h in zip(pos,cnt,right)]);v=np.array([value(t,k,T,p) for t,k in zip(pos,cnt)])
    e=WO.trajectory_ope(tr,pos,lp,yt,q,v)
    # with the policy's exact Q every correction term vanishes, so both doubly robust estimates are the true value on any sample
    # (to the float32 outcomes' rounding)
    assert abs(e['dr']-J)<1e-6 and abs(e['wdr']-J)<1e-6,(e,J)
    # importance sampling alone, and doubly robust with a constant model, are consistent; with the behaviour as the policy the weights
    # are all one
    assert abs(e['is']-J)<0.05 and abs(e['wis']-J)<0.04 and 1000<e['ess']<n,(e,J)
    c=WO.trajectory_ope(tr,pos,lp,yt,np.full(len(q),0.5),np.full(len(q),0.5));assert abs(c['dr']-J)<0.05 and abs(c['wdr']-J)<0.04,(c,J)
    b=WO.trajectory_ope(tr,pos,np.zeros(len(lp)),yt,q,v);assert abs(b['wis']-yt.mean())<1e-12 and abs(b['ess']-n)<1e-6
    # a bootstrap draw resamples whole trajectories
    i=np.arange(n)[::-1].copy();r=WO.trajectory_ope(tr,pos,lp,yt,q,v,i);assert all(abs(r[k]-e[k])<1e-9 for k in ('is','wis','dr','wdr'))
    r=WO.trajectory_ope(tr,pos,lp,yt,idx=np.zeros(n,np.int64));assert abs(r['wis']-yt[0])<1e-12 and abs(r['ess']-n)<1e-6


def t_menu_qv_is_the_q_models_menu():
    torch.manual_seed(0);Q=Head(10,7,hidden=16).eval();S=torch.randn(37,10);H=torch.randint(0,7,(37,4));H[::3,2:]=-1
    q,_=menu_q(Q,S,H,torch.ones(7,48,dtype=torch.bool));f=WO.menu_qv(Q,S,H,chunk=8);ok=(H>=0)[:,:,None].expand(-1,-1,48)
    assert torch.allclose(f[ok],q[ok],atol=1e-6)


def t_fqe_reaches_the_policy_value_round_by_round():
    T=4;p=0.9;n=6000;X,card,cell,_,y=right_world(n,T,seed=2,fixed=True);c=Toy(X,card,cell,y,T);rows=np.arange(n*T);s0=rows[::T][:200]
    S0=c.S(0,0,s0);H0=WO.T(c.a['H'][s0]);Q0=WO.fit_q0(c,rows,steps=1000,batch=1024,hidden=64)
    for sampled in (False,True):
        out,kept=WO.fqe(c,Right(p),rows,S0,H0,Q0,K=T,steps=250,batch=1024,keep=(0,T),sampled=sampled);J=out['J']
        # round k evaluates the policy for k + 1 decisions and the behaviour after them: J_0 is the one-step value, J_3 the policy's value
        assert all(abs(J[k]-mixed(T,p,k+1))<0.035 for k in range(T)),(sampled,J,[mixed(T,p,k+1) for k in range(T)])
        assert J[0]<J[1]<J[T-1] and abs(J[T]-value(0,0,T,p))<0.035 and set(kept)=={0,T} and len(out['per'][T])==200
    # the kept round's Q of the recorded plays and its V under the policy
    q,v=WO.q_and_v(kept[T],Right(p),c.S(0,0,rows[T-1::T][:500]),WO.T(c.a['H'][rows[T-1::T][:500]]),WO.T(card[T-1::T][:500]),WO.T(cell[T-1::T][:500]))
    assert q.shape==v.shape==(500,) and ((v>0)&(v<1)).all()


class Seen(torch.nn.Module):
    # a continuation actor that records the menus it is offered and plays uniformly from them
    def __init__(self,log):
        super().__init__();self.log=log
    def menu_logp(self,h,H):
        self.log.append(H.clone());ok=(H>=0).float()[:,:,None].expand(-1,-1,LW.N_CELLS)
        return torch.where(ok>0,(ok/ok.sum((1,2),keepdim=True)).log(),torch.full_like(ok,-1e9))


def t_rollout_marks_are_shorter_rollouts_and_menus_come_from_nxt():
    torch.manual_seed(0);ms=[LW.WorldModel(5,6,hidden=16,latent=8,emb=4).eval() for _ in range(3)];n=40;S=torch.randn(n,5)
    for m in ms:
        for p in m.parameters():p.requires_grad_(False)
    hs=[m.encode(S) for m in ms];H0=torch.tensor([[0,1,2,3]]*n);card=torch.full((n,),1);cell=torch.zeros(n,dtype=torch.long)
    menus=torch.randint(0,6,(n,9,4));calls=[]
    def nxt(k,cd,z):
        calls.append((k,cd.clone(),z.shape));return menus[:,k+1]
    log=[];tr={'marks':{1,3,6}};v,_=LW.rollout(ms,hs,H0,None,card,cell,6,[Seen(log) for _ in ms],torch.Generator().manual_seed(3),nxt=nxt,trace=tr)
    # every member's actor is offered the menu nxt gave for that decision, and nxt sees the card just played (the root's first)
    assert [k for k,_,_ in calls]==[0,1,2,3,4] and torch.equal(calls[0][1],card) and calls[0][2]==(n,len(LW.SUM))
    for k in range(1,6):assert all(torch.equal(x,menus[:,k]) for x in log[(k-1)*3:k*3])
    for k in range(1,5):assert ((menus[:,k]==calls[k][1][:,None]).any(1)).all()
    # a mark's value is the rollout cut there with the same draws, the last mark is the returned value, and the chance that the game
    # is still running falls
    for k in (1,3):
        w,_=LW.rollout(ms,hs,H0,None,card,cell,k,[Seen([]) for _ in ms],torch.Generator().manual_seed(3),nxt=lambda k,cd,z:menus[:,k+1])
        assert torch.allclose(tr[k][0],w,atol=1e-6)
    assert torch.allclose(tr[6][0],v) and (tr[6][2]<=tr[3][2]+1e-7).all() and (tr[3][2]<=tr[1][2]+1e-7).all() and (tr[6][1]>=tr[1][1]).all()
    # identical members disagree by nothing
    tr={'marks':{4}};LW.rollout([ms[0]]*3,[hs[0]]*3,H0,None,card,cell,4,None,torch.Generator().manual_seed(0),nxt=lambda k,cd,z:menus[:,k+1],trace=tr)
    assert torch.allclose(tr[4][1],torch.zeros(n),atol=1e-5)


def t_cycle_sends_the_played_card_to_the_back_and_affords_by_elixir():
    cy=WO.Cycle(np.array([[2,5,-1,-1]]),np.array([[0,1,2,3,4,5,6,7]]),seed=0);h=cy.hand[0].tolist();q=cy.queue[0].tolist()
    assert h[:2]==[2,5] and sorted(h+[x for x in q if x>=0])==list(range(8)) and q[4:]==[-1]*4
    m=cy(0,torch.tensor([5]));assert m[0].tolist()==[2,q[0],h[2],h[3]] and cy.queue[0].tolist()[:4]==q[1:4]+[5]
    # the card played comes back after the four queued before it have been played
    for x in q[1:4]+[q[0]]:
        assert 5 not in cy.hand[0].tolist();cy(0,torch.tensor([x]) if x in cy.hand[0].tolist() else cy.hand[0,:1])
    assert 5 in cy.hand[0].tolist()
    # the menu keeps the hand's cards the elixir at that depth affords, known cards first; with too little elixir the cheapest; past
    # the last depth the last elixir
    cost=torch.tensor([1.0,2,3,4,5,6,7,8]);cy=WO.Cycle(np.array([[6,1,3,5]]),np.array([[0,1,2,3,4,5,6,7]]),cost=cost,elixir=np.array([[9.0,4.5,0.5]]))
    assert cy.menu(0)[0].tolist()==[6,1,3,5] and cy.menu(1)[0].tolist()==[1,3,-1,-1] and cy.menu(2)[0].tolist()==[1,-1,-1,-1]==cy.menu(7)[0].tolist()
    assert cy.menu()[0].tolist()==[6,1,3,5] and cy(0,torch.tensor([3]))[0].tolist()==[1,-1,-1,-1] and 3 not in cy.hand[0].tolist()


def t_compare_applies_the_registered_rule():
    tr={'bc':{'wr':(0.5,0.49,0.51),'dwr':(0,0,0)},'a':{'wr':(0.6,0,0),'dwr':(0.10,0.09,0.11)},'b':{'wr':(0.55,0,0),'dwr':(0.05,0.04,0.06)},
        'c':{'wr':(0.62,0,0),'dwr':(0.12,0.11,0.13)},'d':{'wr':(0.5,0,0),'dwr':(0.0,-0.01,0.01)}}
    kl={'a':0.1,'b':0.05,'c':2.0,'d':0.01}
    # ranks right, picks c, every gain 2 points high, c 10 points low: c is the far policy
    est={'bc':0.40,'a':0.52,'b':0.47,'c':0.42,'d':0.42}
    r=WO.compare(est,tr,kl)
    assert r['pick']=='a' and r['true_best']=='c' and not r['picks_best'] and abs(r['regret']-2.0)<1e-9 and set(r['far'])=={'c'}
    assert abs(r['far']['c']+10)<1e-9 and abs(r['mae_gain']-(2+2+10+2)/4)<1e-9 and abs(r['bias_gain']-(2+2-10+2)/4)<1e-9
    assert abs(r['mae_level']-(10+8+8+20+8)/5)<1e-9 and abs(r['spearman_near']-1)<1e-9 and r['err_kl_spearman']>0.7
    est['c']=0.53;r=WO.compare(est,tr,kl);assert r['pick']=='c' and r['picks_best'] and abs(r['spearman']-1)<1e-9 and r['regret']==0
    # a pick inside the true best's interval counts; gains given without the reference are used as they are
    tr['a']['dwr']=(0.115,0.1,0.13);r=WO.compare({'a':0.2,'b':0.1,'c':0.15,'d':0.0},tr,kl);assert r['pick']=='a' and r['picks_best']
    per={k:np.full(50,v) for k,v in est.items()};r=WO.compare(est,tr,kl,per=per,boots=20);assert r['spearman_boot'][0]==r['spearman_boot'][1]==r['spearman']


def t_estimates_read_the_job_outputs(tmp_path):
    mb=tmp_path/'mb';fq=tmp_path/'fqe';mb.mkdir();fq.mkdir();n=10
    for name,j in (('bc',0.5),('a',0.6)):
        np.savez(mb/f'{name}.npz',rows=np.arange(n),c0_marks=np.array([0,4]),c0_value=np.array([[j]*n,[j+0.01]*n]),c0_disagree=np.array([[0.0]*n,[0.2]*n]),
                 c0_running=np.ones((2,n)))
        (mb/f'{name}.json').write_text(json.dumps({'name':name,'games':70000,'configs':[[0,1]],'menus':'cycle','start':'held'}))
        np.savez(fq/f'{name}.npz',J0=np.full(n,j),J1=np.full(n,j+0.02))
        (fq/f'{name}.json').write_text(json.dumps({'name':name,'games':10000,'K':1,'J':[j,j+0.02],'stored|q0':{'is':j,'wis':j,'ess':5.0,'dr':j,'wdr':j},
                                                   'stored|q1':{'is':j,'wis':j,'ess':5.0,'dr':j+0.1,'wdr':j}}))
    (mb/'lam_x.json').write_text(json.dumps({'key':'v70k|M2','lam':0.5}))
    est,per=WO.estimates(mb,fq)
    e=est['model|v70k|M2|cycle|held|H4|lam0'];assert set(e)=={'bc','a'} and abs(e['bc']-0.51)<1e-9 and abs(e['a']-0.61)<1e-9
    assert abs(est['model|v70k|M2|cycle|held|H4|lam1']['a']-(0.61-0.5*0.2))<1e-9 and abs(est['model|v70k|M2|cycle|held|H4|lam0.3']['a']-(0.61-0.3*0.5*0.2))<1e-9
    assert 'model|v70k|M2|cycle|held|H0|lam1' not in est and abs(est['fqe|v10k|K1']['a']-0.62)<1e-9 and abs(est['stored|dr|v10k|q1']['a']-0.7)<1e-9
    assert est['stored|wis|v10k']['bc']==0.5 and 'stored|wis|v10k|q1' not in est and len(per['fqe|v10k|K0']['a'])==n


# ---- the giant world, end to end

def giant_games(n,p,seed=0,out=None,per=200,n_dec=24,drop=0.4):
    # n games; p maps a team to its chance of playing the giant when it is in hand (else a random other hand card); with out, shards in
    # train.corpusStates format (decks recorded per game); returns the winners
    rng=np.random.default_rng(seed);ts=np.sort(rng.uniform(T0,T0+8*86400,n));win=[]
    for k in range(0,n,per):
        rows={c:[] for c in COLS+EXTRA};X=[];games=[]
        for g in range(k,min(k+per,n)):
            bid=f'G{g:05d}';hp={tm:[1.0]*3 for tm in TEAMS};pend={tm:0 for tm in TEAMS};deck={tm:[str(x) for x in rng.permutation(DECK)] for tm in TEAMS}
            dk=[list(deck[tm]) for tm in TEAMS];t=0.0
            for i in range(n_dec):
                team=TEAMS[i] if i<2 else TEAMS[int(rng.random()<0.5)];opp='red' if team=='blue' else 'blue';t+=float(rng.uniform(2,10))
                hand=deck[team][:4];others=[x for x in hand if x!='giant'];x_=state(t,team,hp,pend,rng)
                card='giant' if 'giant' in hand and rng.random()<p[team] else str(rng.choice(others))
                x,y=CELLS[int(rng.choice(own_cells(team)))][int(rng.integers(12))]
                if out:
                    X.append(x_);od=deck[opp]
                    rec=(bid,i,t,team,card,float(x)+0.5,float(y)+0.5,False,False,1,'|'.join(hand),'|'.join(od[:4]),od[4])
                    for c_,v in zip(COLS+EXTRA,rec):rows[c_].append(v)
                deck[team].remove(card);deck[team].append(card)
                if pend[team]:j=int(rng.integers(3));hp[opp][j]=max(0.0,hp[opp][j]-drop);pend[team]=0
                if card=='giant':pend[team]=1
                tm=TEAMS[int(rng.integers(2))];j=int(rng.integers(3));hp[tm][j]=max(0.0,hp[tm][j]-float(rng.uniform(0.05,0.35)))
            w='blue' if rng.random()<sig(3*(sum(hp['blue'])-sum(hp['red']))) else 'red';win.append(w)
            games.append({'bid':bid,'actual_winner':w,'sim_winner':w,'end_t':300.0,'premature':False,'actual_bc':1,'actual_rc':0,'sim_bc':1,'sim_rc':0,'n':n_dec,
                          'sim_blue':0.4,'sim_red':0.3,'ts':float(ts[g]),'mode':'Ranked','rolled':False,'decks':dk})
        if out:
            p_=out/f'shard{k//per:03d}.npz';np.savez_compressed(p_,X=np.stack(X),roll=np.zeros((len(X),4),np.float32),**{c:np.array(v) for c,v in rows.items()})
            p_.with_suffix('.json').write_text(json.dumps({'outcomes':games})+'\n')
    return win


def giant_policy(c,bias):
    # a policy that plays the giant whenever it is in hand (bias > 0), never (bias < 0) or uniformly (0), on any cell
    p=Policy(c.n_state,c.n_card,16)
    with torch.no_grad():
        for q in p.parameters():q.zero_()
        p.card.bias[c.vocab.index('giant')]=bias
    return p.eval()


def t_model_and_fqe_rank_planted_policies(tmp_path):
    beh=0.15;st=tmp_path/'states';st.mkdir();giant_games(800,{'blue':beh,'red':beh},seed=0,out=st);d=tmp_path/'pack';pack(st,d);c=Corpus(d);ax=LW.aux(c,K=3)
    ms=[]
    for s in range(3):
        m,_=LW.fit(c,ax,LW.train_rows(c),seed=s,K=3,epochs=12,hidden=64,latent=32,lr=3e-3,batch_size=256,log=lambda *a,**k:None);m.eval()
        for q in m.parameters():q.requires_grad_(False)
        ms.append(m)
    # the true values: the policy plays one side of 6,000 games (blue in half of them) against the behaviour
    pg={'never':0.0,'uniform':0.25,'always':1.0};truth={}
    for k,v in pg.items():
        w=[x=='blue' for x in giant_games(3000,{'blue':v,'red':beh},seed=11)]+[x=='red' for x in giant_games(3000,{'blue':beh,'red':v},seed=12)]
        truth[k]=float(np.mean(w))
    assert truth['always']>truth['uniform']+0.1>truth['never']+0.15,truth
    rows=WO.starts(c);S=c.S(0,0,rows);H=WO.T(c.a['H'][rows]);dk=WO.decks(st,c,rows);tg=c.records(WO.train_games(c,10**6));est={};fq={};rv={}
    Q0=WO.fit_q0(c,tg,steps=1500,batch=512,hidden=64)
    for k,b in (('never',-8.0),('uniform',0.0),('always',8.0)):
        pol=giant_policy(c,b);acts=WO.distil(ms,pol,c,tg,steps=300,batch=256,lr=3e-3);fr=WO.fidelity(ms,acts,pol,S,H)
        # the distilled actors follow the policy far more closely than the members' behaviour heads do
        assert np.mean(fr['kl_actor'])<0.3*np.mean(fr['kl_head']),(k,fr)
        v=WO.model_values(ms,acts,pol,S,H,lambda s,r:WO.Cycle(c.a['H'][rows[s]],dk[s],r),16,marks=(0,1,16),n_roll=4)
        rec=WO.model_values(ms,acts,pol,S,H,lambda s,r:WO.Recorded(c,rows[s],16),16,marks=(16,),n_roll=4);rv[k]=rec['value'][0].mean()
        assert v['running'][-1].mean()<0.2 and (v['disagree'][-1]>=v['disagree'][1]).all();est[k]=v['value'][-1].mean()
        fq[k]=WO.fqe(c,pol,tg,S,H,Q0,K=14,steps=150,batch=512)[0]['J'][-1]
    m0=WO.model_values(ms,None,None,S,H,lambda s,r:WO.Cycle(c.a['H'][rows[s]],dk[s],r),16,marks=(16,),n_roll=4)['value'][0].mean()
    # full model rollouts and fitted-Q evaluation order the policies as the truth does; the model's own behaviour wins about half;
    # with the recorded menus (open loop) the giant, rarely played by the behaviour, stays in hand and is played at every decision
    for e in (est,fq):assert e['always']>e['uniform']>e['never'] and e['always']-e['never']>0.4*(truth['always']-truth['never']),(e,truth)
    assert rv['always']>est['always'],(rv,est)
    assert abs(m0-0.5)<0.08 and abs(fq['always']-truth['always'])<0.1,(m0,fq,truth)
