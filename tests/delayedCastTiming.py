import pytest
from sim import replay as R
from sim import spells
from sim.game import Game
from sim.units import has

# with every recorded play reaching the field PLACE_DELAY after its recorded time (t2133), the Lightning's bolts are counted from the
# tick it lands, the frame the recording measured its 0.52 s from, and the Explosive Escape's dig from the start of the delayed cast
_OUTCOME={'result':'D','tc':0,'oc':0}


def _play(card,team='blue',time=40,ability=0,x=4.25,y=10.25):
    return {'card':card,'team':team,'tile_x':x,'tile_y':y,'time':time,'ability':ability,'card_type':'normal'}


@pytest.mark.parametrize('time',[40,61])
def t_lightning_bolts_count_from_the_delayed_landing(monkeypatch,time):
    monkeypatch.setattr(Game,'END',time/20+4.0)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)
    seen={'apply':[],'strike':[]}
    apply,strike=spells.LightningSpell.apply,spells.LightningSpell._strike
    def app(sp,g):
        seen['apply'].append(round(g.t,6));apply(sp,g)
    def stk(sp,g):
        tw=g.arena.get_tower('red','princess','left');hp=tw.hp;strike(sp,g);seen['strike'].append((round(g.t-seen['apply'][0],6),hp-tw.hp))
    monkeypatch.setattr(spells.LightningSpell,'apply',app);monkeypatch.setattr(spells.LightningSpell,'_strike',stk)
    R.replay_battle('lightning',[_play('lightning',time=time,x=3.5,y=24.5)],_OUTCOME)
    assert seen['apply']==[pytest.approx(time/20+R.PLACE_DELAY,abs=1e-6)],seen
    assert [t for t,_ in seen['strike']]==[pytest.approx(t,abs=1e-6) for t in (0.45,0.9,1.4)],seen
    assert seen['strike'][0][1]>0,"The first bolt strikes the tower in the radius"


def t_escape_dig_counts_from_the_delayed_cast(monkeypatch):
    # the champion recorded at 0 is on the field from 1.0; the ability recorded at 3.0 is released at 4.0 and the dig follows 0.4 s
    # into the cast (at the recorded time plus 0.40 on the old base), the surfacing at the cast end
    monkeypatch.setattr(Game,'END',7.0)
    monkeypatch.setattr(R,'_detect_true_red',lambda plays:False)
    under=[]
    tick=Game.tick
    def rec(g,*a,**kw):
        out=tick(g,*a,**kw)
        if any(t.alive and t.name=='Mighty Miner' and has(t,'burrowed') for t in g.players['blue'].troops):under.append(round(g.t,6))
        return out
    monkeypatch.setattr(Game,'tick',rec)
    g,info=R.replay_battle('escape',[_play('mighty_miner',time=0),_play('ability-mighty-miner',time=60,ability=1)],_OUTCOME)
    assert any(' activates Mighty Miner ability' in line for line in g.log),g.log[-5:]
    assert under and under[0]==pytest.approx(3.0+R.PLACE_DELAY+0.4,abs=1e-6),under[:3]
    assert under[-1]==pytest.approx(3.0+R.PLACE_DELAY+0.95,abs=1e-6),under[-3:]
