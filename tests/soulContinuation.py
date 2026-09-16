import pytest

from sim.game import Game
from sim.cards import create
from sim.fx import SoulCollect


DECK=['skeleton_king','knight','archers','fireball','giant','musketeer','bomber','arrows']


def setup(team='blue',souls=0):
    import random
    random.seed(42)
    g=Game(p1={'deck':DECK,'drag_del':0},p2={'deck':DECK,'drag_del':0})
    for tw in g.arena.towers:
        tw.rng=0
        if tw.troop:tw.troop.RNG=0
    p=g.players[team];p.deck.hand=DECK[:4];p.elixir=10
    assert g.play_card(team,'skeleton_king',9.5,10.5 if team=='blue' else 21.5)==(True,'ok');g.run(2.2)
    king=next(t for t in p.troops if t.name=='Skeleton King');king.spd=0
    for _ in range(souls):
        victim=create('knight',11,team,2.5,10.5 if team=='blue' else 21.5);g.deploy(team,victim)
        victim.take_damage(victim.hp);g._proc_deaths()
    sc=next(c for c in king.components if isinstance(c,SoulCollect));assert sc.souls==souls
    p.elixir=10;assert g.activate_ability(team,king)==(True,'ok')
    while not king.ability.active and g.t<5:g.tick()
    assert king.ability.active and king.ability.q==6+souls
    return g,king


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('souls',(0,4,10))
def t_activated_summon_queue_finishes_after_king_death(team,souls):
    g,king=setup(team,souls);ab=king.ability
    g.tick();assert len([t for t in g.players[team].troops if t is not king])==1
    king.take_damage(king.hp);g._proc_deaths();g._proc_deaths()
    assert king not in g.players[team].troops
    g.run(5)
    children=g.players[team].troops
    assert len(children)==6+souls and ab.q==0 and not ab.active
    assert all(t.hp==t.max_hp==1 and t.is_clone and t.no_soul for t in children)


def t_death_after_cast_before_first_summon_preserves_entire_queue():
    g,king=setup();assert king.ability.q==6
    king.take_damage(king.hp);g._proc_deaths();g.run(3)
    assert len(g.players['blue'].troops)==6 and king.ability.q==0


def t_death_before_cast_completion_does_not_start_summoning():
    g=Game(p1={'deck':DECK,'drag_del':0});p=g.players['blue'];p.deck.hand=DECK[:4];p.elixir=10
    assert g.play_card('blue','skeleton_king',9.5,10.5)==(True,'ok');g.run(2.2)
    king=next(t for t in p.troops if t.name=='Skeleton King');p.elixir=10
    assert g.activate_ability('blue',king)==(True,'ok');g.run(.3)
    assert king.ability.casting and not king.ability.active
    king.take_damage(king.hp);g._proc_deaths();g.run(3)
    assert not g.players['blue'].troops


@pytest.mark.parametrize('team',('blue','red'))
@pytest.mark.parametrize('phase',('before_troops','after_ability','after_tick'))
def t_postdeath_emission_cadence_matches_living_control(phase,team):
    from sim.fx import Component,Timer,SoulContinuation
    def run(kill):
        g,king=setup(team,souls=4);seen=set();events=[];death_time=g.t+.15
        class KillAfterAbility(Component):
            def on_tick(self,tr,game):
                if game.t>=death_time:tr.take_damage(tr.hp)
        if kill and phase=='before_troops':g.spells.append(Timer(.15,lambda game:king.take_damage(king.hp)))
        if kill and phase=='after_ability':king.components.append(KillAfterAbility())
        for _ in range(90):
            g.tick()
            if kill and phase=='after_tick' and g.t>=death_time and king.alive:
                king.take_damage(king.hp);g._proc_deaths()
            for t in g.players[team].troops:
                if t is not king and t.id not in seen:seen.add(t.id);events.append(g.t)
        assert not any(isinstance(s,SoulContinuation) for s in g.spells)
        return events
    assert run(True)==run(False)


def t_death_on_activation_tick_does_not_emit_early():
    from sim.fx import Component
    g=Game(p1={'deck':DECK,'drag_del':0});p=g.players['blue'];p.deck.hand=DECK[:4];p.elixir=10
    assert g.play_card('blue','skeleton_king',9.5,10.5)==(True,'ok');g.run(2.2)
    king=next(t for t in p.troops if t.name=='Skeleton King');p.elixir=10
    class KillOnActivation(Component):
        def on_tick(self,tr,game):
            if tr.ability.active:tr.take_damage(tr.hp)
    king.components.append(KillOnActivation());assert g.activate_ability('blue',king)==(True,'ok')
    while king.alive:g.tick()
    assert not p.troops and king.ability.q==6
    g.tick();assert len(p.troops)==1 and king.ability.q==5


def t_final_emission_death_finishes_cooldown_and_cleans_wrapper():
    from sim.fx import SoulContinuation
    g,king=setup();ab=king.ability
    while ab.q:g.tick()
    assert ab.active and ab.q==0
    king.take_damage(king.hp);g._proc_deaths();g.tick()
    assert not ab.active and ab.cd==ab.max_cd
    assert not any(isinstance(s,SoulContinuation) for s in g.spells)
    assert len(g.players['blue'].troops)==6


def t_cloned_king_death_does_not_transfer_original_queue():
    from sim.fx import SoulContinuation
    g,king=setup();g.tick();ab=king.ability
    create('clone',11,'blue',king.x,king.y).apply(g)
    clone=next(t for t in g.players['blue'].troops if t.name==king.name and t is not king)
    queued,timer=ab.q,ab.timer;clone.take_damage(clone.hp);g._proc_deaths()
    assert (ab.q,ab.timer)==(queued,timer) and not any(isinstance(s,SoulContinuation) for s in g.spells)
    g.run(3);assert len([t for t in g.players['blue'].troops if t is not king])==6


def t_successor_king_does_not_own_dead_kings_queue():
    g,king=setup(souls=4);g.tick();king.take_damage(king.hp);g._proc_deaths()
    successor=create('skeleton_king',11,'blue',9.5,10.5);g.deploy('blue',successor)
    sc=next(c for c in successor.components if isinstance(c,SoulCollect));g.run(3)
    assert king.ability.q==0 and successor.ability.q==0 and not successor.ability.active and sc.souls==0
    assert len([t for t in g.players['blue'].troops if t is not successor])==10


def t_dead_caster_displacement_cannot_move_continuation_anchor(monkeypatch):
    from sim.fx import SoulContinuation
    g,king=setup();g.tick();king.take_damage(king.hp);g._proc_deaths()
    wrapper=next(s for s in g.spells if isinstance(s,SoulContinuation));anchor=wrapper.x,wrapper.y
    king.x+=5;king.y+=5;seen={t.id for t in g.players['blue'].troops}
    monkeypatch.setattr('sim.fx.random.uniform',lambda a,b:0)
    while not any(t.id not in seen for t in g.players['blue'].troops):g.tick()
    child=next(t for t in g.players['blue'].troops if t.id not in seen)
    assert abs(child.x-anchor[0])<.2 and abs(child.y-anchor[1])<.2
    assert (wrapper.x,wrapper.y)==anchor


def t_projectile_phase_cannot_hit_a_later_same_tick_continued_summon():
    g,king=setup();g.tick();king.take_damage(king.hp);g._proc_deaths()
    old=list(g.players['blue'].troops)
    while king.ability.timer>g.DT+1e-9:g.tick()
    g._shoot('red',king.x,king.y,1,king,lambda game,pr:[t.take_damage(1000) for t in list(game.players['blue'].troops)])
    g.tick()
    assert all(not t.alive for t in old)
    assert len(g.players['blue'].troops)==1 and g.players['blue'].troops[0].hp==1


@pytest.mark.parametrize('phase',('pending','casting','active'))
def t_death_refund_is_once_only_and_active_queue_is_not_refunded(phase):
    from sim.fx import SoulContinuation
    g=Game(p1={'deck':DECK,'drag_del':0,'ability_del':.2,'ability_std':0})
    p=g.players['blue'];p.deck.hand=DECK[:4];p.elixir=10
    assert g.play_card('blue','skeleton_king',9.5,10.5)==(True,'ok');g.run(2.2)
    king=next(t for t in p.troops if t.name=='Skeleton King');p.elixir=5
    assert g.activate_ability('blue',king)==(True,'ok')
    if phase=='casting':g.run(.3);assert king.ability.casting
    elif phase=='active':
        while not king.ability.active:g.tick()
    before=p.elixir;king.take_damage(king.hp);g._proc_deaths()
    assert p.elixir==pytest.approx(before+(0 if phase=='active' else king.ability.cost))
    after=p.elixir;g._proc_deaths();assert p.elixir==after
    assert not king.ability.casting and not g.pending_ab
    assert any(isinstance(s,SoulContinuation) for s in g.spells)==(phase=='active')


def t_environment_keeps_active_summon_queue_after_king_death():
    from sim.env import CREnv,GAME_FEAT,PLAYER_FEAT,N_TOWERS,TOWER_FEAT,TROOP_FEAT,MAX_TROOPS
    import random
    random.seed(42)
    env=CREnv(blue_deck=DECK,decision_freq=1);env.reset(seed=4);g=env.game;p=g.players['blue']
    p.deck.hand=DECK[:4];p.elixir=10;p.drag_del=0
    for tower in g.arena.towers:
        tower.rng=0
        if tower.troop:tower.troop.RNG=0
    env.step({'card':0,'x':9.5,'y':10.5})
    for _ in range(44):env.step(4)
    king=next(t for t in p.troops if t.name=='Skeleton King');king.spd=0;p.elixir=10
    assert g.activate_ability('blue',king)==(True,'ok')
    while king.ability.q!=5:env.step(4)
    king.take_damage(king.hp)
    for _ in range(60):obs,*_=env.step(4)
    assert len(p.troops)==6 and king not in p.troops and env.observation_space.contains(obs)
    start=GAME_FEAT+2*PLAYER_FEAT+N_TOWERS*TOWER_FEAT
    slots=obs[start:start+MAX_TROOPS*TROOP_FEAT].reshape(MAX_TROOPS,TROOP_FEAT)
    assert sum(row[2]>0 for row in slots)==6 and king.ability.q==0
    env.close()


def t_living_king_queue_does_not_duplicate_without_death():
    g,king=setup(souls=4);g.run(4)
    assert len([t for t in g.players['blue'].troops if t is not king])==10
    assert king.ability.q==0 and not king.ability.active
