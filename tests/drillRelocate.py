from sim.game import Game
from tests.evoDrillSubmerge import resurface, surfaced
from tests.util import quiet


# Where the Evolved Goblin Drill comes back (local/drillRelocateRound/findings.md): the game marks both coming spots round the tower as
# soon as the drill surfaces (LCQ 09YP9UPGQ2YU, September 2026) and while the card is dragged (Abdod, youtube sPN0NLE9akg). The last spot
# is always the first mirrored through the tower; the middle one is the quarter turn toward the river (inner start: front, LCQ x3 and
# Rolex 2026-09; outer start: front, D2 preview; inner-back start: inner-front, 118 s preview); straight in front or behind it goes to
# the outer side (front: Abdod 2024-12, Epsilon and Boss_CR 2026-02; behind: Abdod 130 s demo). Beside the King alone it comes back in
# place (Abdod youtube -T95U-EoAvg 1896 to 1901 s: "King Tower doesn't count as a crown Tower so it doesn't circle around him").
# Blue drills attack the red towers: right princess (14.5, 25.5), left princess (3.5, 25.5), King (9, 29); river side is down (-y).


def path(team,x,y):
    g,gd=surfaced(team,x,y,quiet(Game()))
    return resurface(g,gd,0.6),resurface(g,gd,0.3)


def t_from_the_outer_side_it_turns_toward_the_river_then_to_the_inner_side():
    assert path('blue',17,25)==((14.0,23.0),(12.0,26.0))
    # the blue left princess (3.5, 6.5) seen by a red drill, mirrored in both axes
    assert path('red',1,7)==((4.0,9.0),(6.0,6.0))


def t_from_behind_it_turns_toward_the_outer_side_then_to_the_front():
    assert path('blue',15,28)==((17.0,25.0),(14.0,23.0))
    assert path('blue',3,28)==((1.0,25.0),(4.0,23.0))


def t_beside_the_king_alone_it_comes_back_in_place():
    # 1.1 tiles from the King's edge, 4 from either princess
    assert path('blue',9,26)==((9.0,26.0),(9.0,26.0))


def t_a_nearer_king_does_not_take_the_turn_from_a_princess_in_reach():
    # 1.45 tiles from the King, 1.7 from the left princess: inner-back of the princess turns to its inner-front, then its outer-front
    assert path('blue',6,27.5)==((5.5,23.0),(1.0,23.5))


def t_controls_recorded_paths_unchanged():
    # inner start (LCQ): inner -> front -> outer
    assert path('blue',12,25)==((15.0,23.0),(17.0,26.0))
    # front start half a tile off the centre line on either side (Abdod D1 was outer of it): front -> outer -> behind
    assert path('blue',15,22.5)==((17.5,26.0),(14.0,28.5))
    assert path('blue',14,22.5)==((17.5,25.0),(15.0,28.5))
    # inner-front corner (Rolex 2026-09, Abdod demo): -> outer-front -> outer-back
    assert path('blue',12,23.5)==((16.5,23.0),(17.0,27.5))
