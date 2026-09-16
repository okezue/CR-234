# Replay reconstruction and evaluation contract

Replay-conditioned winner agreement measures whether the simulator reproduces a recorded battle's outcome when given its recorded plays. It is not autonomous play accuracy or a percentage of physically correct mechanics.

## Prediction inputs and scoring labels

The v2 replay loader uses declared card levels, tower configuration, deck variants, side identifiers and timed plays. It no longer raises the starting King level using final tower health, and modifier exclusions use declared game modes rather than terminal health. Legacy data provides only one tower-level field per side; unless a separate `team_tower_lvl` or `opp_tower_lvl` is supplied, that declared value initializes both King and tower troop. This is an outcome-independent fallback, not a reconstruction of unknown mixed King/tower levels.

Recorded winner, crowns and final health remain scoring and match-identification evidence. They must not affect starting state or sample selection. Invalid labels do not prevent prediction; the CLI identifies unavailable labels and treats all-game percentages as lower bounds rather than silently dropping records. Invalid source timestamps are marked so a coercion to zero cannot establish selection eligibility.

`sim.evaluation` ranks eligible IDs deterministically from declared metadata and source-action availability. It does not read winner, crown or health fields. The goal assessment joins those labels only after its 2,000 IDs are frozen. Missing labels are reported after prediction, never replaced with another game.

## Retained reconstruction assumptions

The replay path reconstructs a hand from recorded card order, supplies elixir for recorded commands, and tries placement recovery when the simulated state rejects the original position. It derives side orientation from recorded placement/deck information. These are replay-conditioning assumptions, not predictions available to an autonomous policy. Their counters and unreached plays must be reported. Actual ending labels do not drive these operations.

Recorded abilities select the newest living ability-bearing instance matching the named card. A missing or dead named troop cannot activate an unrelated champion. Distinct Hero/Champion card types can use their own abilities through `Game.activate_ability`; an older copy of the same card cannot displace the latest living copy. The default call selects the latest instance of the default card name. Registered Goblin banners use `Game.activate_banner` independently of dead or unrelated default pointers. Cast cancellation and refunds do not depend on that default pointer.

Latest-living fallback after the newest copy dies remains an implementation interpretation, not exhaustively verified live-game behavior. The default pointer still provides a legacy card-name choice; it does not implement a graphical multi-button control interface.

## Strict paired assessment

The first outcome-independent goal sample contains exactly 2,000 new eligible games, disjoint from 40,256 prior reservations. Its selection was frozen before terminal labels were loaded. Baseline and candidate receive identical declared starting inputs through a shared outcome-blind projection. Accordingly, the baseline is **revision 6fa81ef with shared outcome-blind initialization**, not a measurement of its historical final-health inference.

The final paired result is 62.05% → 62.15% winner agreement, 40.35% → 40.30% exact crowns, 48.05% → 48.10% premature endings, and 0.211346 → 0.210700 normalized tower-health error. There are five winner corrections and three regressions. The 70% target remains unmet, and no statistical significance or overall fidelity gain is claimed. Scoring definitions and raw records are unchanged.

Regression review found and repaired an introduced Goblin-banner rejection before the final run. All complete and partial prior candidates were preserved. The final largest health regression removes a wrong-name fallback: recorded Valkyrie abilities had incorrectly activated a living Boss Bandit. Removing that unsupported activation worsens the crown/health match in that game; this does not justify retaining it.

## Full-match observation

The September 5, 2026 official Ian77–Wallace LCQ match is matched to raw actions by both decks, ordered placements, play counts, result and final tower health. Opening, midgame and ending recordings and simulator traces are retained. The broadcast hides the initial six seconds, some labels are unreadable, and crowded actors prevent exact target or impact attribution. The simulator still ends at 235.95 seconds instead of the observed 300-second tiebreak, leaving 42 of 123 recorded plays unreached. Matching the winner and crowns does not make that trajectory faithful.

Further active work includes evolved Skeleton descendants losing their replication component, unresolved evolved Mortar landing geometry, and the opening Fire Spirit timing discrepancy. These known issues prevent a claim of maximal fidelity or completed goal.
