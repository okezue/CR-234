import sys,os,random
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from game import Game,card_info
from visualize import _mk_deck,visualize

random.seed(42)
dk_b=_mk_deck(['pekka','knight','archers','fireball'])
dk_r=_mk_deck(['mega_knight','valkyrie','musketeer','zap'])
g=Game(
    p1={'deck':dk_b,'king_lvl':15,'drag_del':0.3,'drag_std':0},
    p2={'deck':dk_r,'king_lvl':9,'drag_del':0.3,'drag_std':0}
)
for _ in range(3000):
    g.tick()
    if g.ended:break
    for tm in ('blue','red'):
        p=g.players[tm]
        if p.deck and p.deck.hand and p.elixir>=5:
            c=p.deck.hand[0]
            ci=card_info(c)
            if p.elixir>=ci['cost']:
                x,y=(9,12) if tm=='blue' else (9,20)
                g.play_card(tm,c,x,y)
print(f'Blue (lvl 15) vs Red (lvl 9)')
print(f'Winner: {g.winner}, {g.players["blue"].crowns}-{g.players["red"].crowns}')
print(f'{len(g.replay.snaps)} frames, T={g.t:.1f}s')
visualize(g)
