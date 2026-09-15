# Heroes and evolutions

The simulator currently contains 123 card definitions, 42 evolution definitions and 17 hero definitions. Hero Ice Wizard, Royal Ghost Evolution, Minion Giant and Ronin are included. These counts describe available definitions, not complete verification of every card mechanic or special-event mode.

## Equipping variants

Use canonical card names and equip variants explicitly:

```python
from sim.game import Game

cards = [
    'royal_ghost', 'ice_wizard', 'knight', 'archers',
    'fireball', 'giant', 'bomber', 'arrows',
]
game = Game(p1={
    'deck': cards,
    'evolutions': ['royal_ghost'],
    'heroes': ['ice_wizard'],
})
```

A supported card must occupy an equipped slot to evolve automatically. Each player and card has an independent counter. Royal Ghost requires two ordinary deployments, so its sequence is base, base, evolved, then repeats. One-cycle cards alternate base and evolved. Rejected plays, Mirror copies, clones and spawned troops do not charge the normal-game counter. Mirror produces the base variant. Heroes share the existing two-slot limit with champions, and a card cannot be equipped as both hero and evolution.

`Game.play_card(..., evolved=None, hero=None)` selects equipped variants automatically. Explicit boolean flags remain available for replay reconstruction and controlled mechanics tests; an explicit evolution override does not consume or advance the automatic counter. `sim.cards.create(..., evolved=True)` remains a direct factory, without deck history.

Replay reconstruction is different from normal gameplay: every recorded real play advances the recorded cycle even if the simulated placement is rejected. Retries do not count twice. Complete per-side deck metadata takes precedence over row flags. The current scraper's row-level `evo` label means inferred eligibility, not that every labeled deployment is evolved. Missing metadata uses a bounded fallback; contradictory authoritative hero/evolution metadata raises an error rather than constructing an impossible combined variant.

## Named mechanics and limits

Royal Ghost Evolution creates two real Souldier troops when he becomes visible. Tests exercise third-play evolution, repeated reveals, independent attacks and idle disappearance at levels 11 and 16. Souldier deployment-blast placement and targeting still need stronger validation; the current shared spawn-damage implementation is not a claim of complete fidelity.

Hero Ice Wizard's Frosty Fella is armed by activation and waits for the next surviving primary slowing hit before raising a Snowman behind that hit target. Tests cover targetless activation, projectile travel, changed targets, shields, dead targets, clones and crown towers. Primary-versus-secondary target selection, Snowman classification, freeze edge behavior and exact cast timing remain incompletely verified. The existing Snowman HP, lifetime and aura behavior are retained rather than recalibrated from promotional footage.

## Training observations

`CREnv` accepts `blue_evolutions`, `red_evolutions`, `blue_heroes` and `red_heroes`. Reset clears evolution charge. Its observation vector grows from 391 to 439 entries: the original 391-entry prefix is unchanged, followed by three values per original deck slot for each player: evolution ready, normalized charge, and equipped hero. Existing models expecting a 391-entry input require an explicit input-schema migration or retraining; they are not automatically compatible.

Replay snapshots also carry equipped variant sets, charge counters and troop `evolved`/`is_hero` flags. The current visualizer consumes the additional state without a new deck-editor interface. The separate `train/rl.py` training scaffold has its own replay-flag and feature path; this change does not certify that scaffold's evolution scheduling or any trained model.
