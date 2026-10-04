import pytest
from sim import replay as R
from sim.game import Game

# recorded plays reach the field 1.00 s after their recorded time (52 frame readings in the eight matched recordings, t2133)
_OUTCOME={'result':'D','tc':0,'oc':0}
_DELAY=1.0

def _play(card,team='blue',time=40,ability=0,x=4.25,y=None):
    y=y if y is not None else (10.25 if team=='blue' else 22.25)
    return {'card':card,'team':team,'tile_x':x,'tile_y':y,'time':time,'ability':ability,'card_type':'normal'}

def _watch(monkeypatch):
    seen=[]
    deploy,cast=Game.deploy,Game._cast
    def dep(g,team,tr):
        seen.append((round(g.t,6),team,getattr(tr,'name','')));deploy(g,team,tr)
    def cst(g,team,sp,x,y):
        seen.append((round(g.t,6),team,'spell'));cast(g,team,sp,x,y)
    monkeypatch.setattr(Game,'deploy',dep);monkeypatch.setattr(Game,'_cast',cst)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)
    return seen

@pytest.mark.parametrize('team',['blue','red'])
@pytest.mark.parametrize('card,name,y',[('knight','Knight',None),('cannon','Cannon',None),('fireball','spell',16.25)])
def t_recorded_card_reaches_the_field_one_second_later(monkeypatch,team,card,name,y):
    monkeypatch.setattr(Game,'END',5.0)
    seen=_watch(monkeypatch)
    R.replay_battle('delay',[_play(card,team,time=40,y=y)],_OUTCOME)
    times=[t for t,tm,n in seen if tm==team and n==name]
    assert times and times[0]==pytest.approx(2.0+_DELAY,abs=1e-6)

@pytest.mark.parametrize('team',['blue','red'])
def t_card_and_elixir_leave_at_the_recorded_time(monkeypatch,team):
    monkeypatch.setattr(Game,'END',5.0)
    state={}
    play=Game.play_card
    def rec(g,tm,card,x,y,**kw):
        out=play(g,tm,card,x,y,**kw)
        p=g.players[tm]
        state.update(t=g.t,hand=list(p.deck.hand),elixir=p.elixir,pending=[(pd.card,pd.rem) for pd in g.pending],
                     troops=[t.name for t in p.troops])
        return out
    monkeypatch.setattr(Game,'play_card',rec)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)
    R.replay_battle('hand',[_play('musketeer',team)],_OUTCOME)
    assert state['t']==pytest.approx(2.0,abs=1e-9)
    assert 'musketeer' not in state['hand'] and state['elixir']==pytest.approx(6.0)
    assert state['pending']==[('musketeer',_DELAY)] and getattr(R,'PLACE_DELAY',None)==_DELAY
    assert 'Musketeer' not in state['troops']

@pytest.mark.parametrize('phase',['regulation','overtime'])
def t_play_in_the_last_second_never_reaches_the_field(monkeypatch,phase):
    # regulation decided at 3.0 by a crown, or overtime ending at 3.0 in the tiebreaker
    monkeypatch.setattr(Game,'REG',3.0 if phase=='regulation' else 1.0)
    monkeypatch.setattr(Game,'END',3.0 if phase=='overtime' else 300.0)
    seen=_watch(monkeypatch)
    make=R.Game
    def game(**kw):
        g=make(**kw)
        if phase=='regulation':g.players['blue'].crowns=1
        return g
    monkeypatch.setattr(R,'Game',game)
    g,info=R.replay_battle('late',[_play('knight','red',time=42),_play('knight','red',time=50)],_OUTCOME)
    assert g.ended and g.t==pytest.approx(3.0,abs=1e-6)
    assert info['placement']['attempted']==2 and not [n for t,tm,n in seen if n=='Knight']

@pytest.mark.parametrize('gap',[50,60,100])
def t_recorded_ability_is_checked_and_started_one_second_later(monkeypatch,gap):
    # checked when it reaches the field: 2.5 s after the champion's play its ability is ready (on the field 1.0 to 2.0, then the 1 s
    # ability delay), which a check at the recorded time would refuse
    monkeypatch.setattr(Game,'END',7.0)
    fired=[]
    proc=Game._proc_pending_ab
    def rec(g):
        n=len(g.pending_ab);proc(g)
        if len(g.pending_ab)<n:fired.append(round(g.t,6))
    monkeypatch.setattr(Game,'_proc_pending_ab',rec)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)
    g,info=R.replay_battle('ability',[_play('archer_queen',time=0),_play('ability-archer-queen',time=gap,ability=1)],_OUTCOME)
    acts=[line for line in g.log if ' activates Archer Queen ability' in line]
    assert len(acts)==1 and acts[0].startswith(f"[{gap/20+_DELAY:.1f}]")
    assert fired==[pytest.approx(gap/20+_DELAY+0.05,abs=1e-6)]
