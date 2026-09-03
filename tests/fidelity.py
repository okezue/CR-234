import random
from sim.game import Game,Player
from sim.cards import create as mk_card,card

def _cast(g,name,x,y,team='blue'):
    s=mk_card(name,11,team,x,y);s.apply(g);g.spells.append(s);return s
def _troops(name,x,y,team='red'):
    r=mk_card(name,11,team,x,y);return r if isinstance(r,list) else [r]

def t_fireball_zap_kills_musketeer():
    g=Game();m=_troops('musketeer',9,25)[0];g.deploy('red',m)
    _cast(g,'fireball',9,25);_cast(g,'zap',9,25)
    assert not m.alive,f"Musketeer left with {m.hp}"
    return "Fireball + Zap kills a Musketeer"
def t_fireball_alone_spares_musketeer():
    g=Game();m=_troops('musketeer',9,25)[0];g.deploy('red',m)
    _cast(g,'fireball',9,25)
    assert m.alive and m.hp==m.max_hp-mk_card('fireball',11,'blue',0,0).dmg
    return f"Fireball alone leaves a Musketeer at {m.hp}"
def t_log_kills_skeleton_army():
    g=Game();random.seed(1);sk=_troops('skeleton_army',9,22)
    # the log answers a freshly placed army: the skeletons stand through their 1 s deploy while it rolls up to them
    for s in sk:g._place('red',s,1.0)
    _cast(g,'the_log',9,18);g.run(2)
    assert not any(s.alive for s in sk),f"{sum(s.alive for s in sk)} skeletons survived"
    return "The Log kills all 15 Skeleton Army skeletons"
def t_arrows_kill_minions():
    for name in ('minions','minion_horde'):
        g=Game();random.seed(1);ms=_troops(name,9,25)
        for m in ms:g.deploy('red',m)
        _cast(g,'arrows',9,25);g.run(1)
        assert not any(m.alive for m in ms),f"{name}: {sum(m.alive for m in ms)} survived"
    return "Arrows kill Minions and Minion Horde"
def t_rocket_kills_wizard():
    # cards.json: Rocket 1484 vs Wizard 755 at level 11, so a lone Rocket does kill a Wizard
    g=Game();w=_troops('wizard',9,25)[0];g.deploy('red',w)
    _cast(g,'rocket',9,25)
    assert not w.alive
    assert mk_card('rocket',11,'blue',0,0).dmg>card('wizard')['stats']['hitpoints'][10]
    return "Rocket one-shots a Wizard"
def t_knight_survives_fireball():
    g=Game();k=_troops('knight',9,25)[0];g.deploy('red',k)
    _cast(g,'fireball',9,25)
    assert k.alive and k.hp==k.max_hp-688
    return f"Knight survives a Fireball at {k.hp}"
def t_skeleton_dies_to_one_tower_hit():
    g=Game();s=_troops('skeletons',3,8)[0];g.deploy('red',s)
    tw=g.arena.get_tower('blue','princess','left')
    assert tw.troop.dmg>s.max_hp
    g.run(3)
    assert not s.alive
    return "A Skeleton dies to a single Princess Tower hit"
def t_hog_bridge_first_hit_timing():
    g=Game();c=card('hog_rider');x,y=3.5,14.5
    hog=mk_card('hog_rider',11,'blue',x,y);g.deploy('blue',hog)
    tw=g.arena.get_tower('red','princess','left');ini=tw.hp
    exp=(tw.dist(x,y)-c['range'])/(c['speed']/60)+c['loadTime']
    while g.t<exp+2 and tw.hp==ini:g.tick()
    assert tw.hp<ini,"Hog never reached the tower"
    assert abs(g.t-exp)<=0.5,f"first hit at {g.t:.2f}s, expected {exp:.2f}s"
    return f"Hog Rider from the bridge lands its first hit at {g.t:.2f}s (expected {exp:.2f}s)"
def t_mirror_makes_level_12():
    dk=['knight','mirror','archers','fireball','hog_rider','musketeer','valkyrie','skeleton_army']
    g=Game();p=Player('blue',king_lvl=11,deck=dk);g.players['blue']=p
    p.elixir=10;p.deck.hand=['knight','mirror','archers','fireball']
    g.play_card('blue','knight',9,5);g.play_card('blue','mirror',9,5);g.run(5)
    lv=sorted(t.lvl for t in p.troops)
    assert lv==[11,12] and max(t.hp for t in p.troops)==card('knight')['stats']['hitpoints'][11]
    return "Mirror produces a level 12 copy"
