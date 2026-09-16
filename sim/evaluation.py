"""Outcome-independent replay selection from declared metadata and placement timing."""
import hashlib
import logging
import math

_LOG=logging.getLogger(__name__)


def _number(value):
    if isinstance(value,bool):return None
    try:n=float(value)
    except (TypeError,ValueError,OverflowError):return None
    return n if math.isfinite(n) else None


def _level(value):
    n=_number(value)
    return n is not None and 1<=n<=16 and n.is_integer()


def _tag(value):
    if not isinstance(value,str):return ''
    tag=value.strip().lstrip('#')
    return tag if tag and not any(c.isspace() or c in ',;|#' for c in tag) else ''


def _present(value):return isinstance(value,str) and bool(value.strip())


def metadata_eligible(row,start,end,modes):
    """Check declared inputs only; the battle-time interval is [start,end)."""
    if not _tag(row.get('replayTag')):return False
    if any(str(row.get(k,'')).lower() not in ('true','1','1.0') for k in ('one_v_one','has_replay')):return False
    ts=_number(row.get('battle_ts'))
    if ts is None or start is not None and ts<start or end is not None and ts>=end:return False
    mode=row.get('gameMode_name')
    if not _present(mode) or modes is not None and mode not in modes:return False
    team=_tag(row.get('team_tags'));opp=_tag(row.get('opponent_tags'))
    if not team or not opp or team==opp:return False
    for side in ('team','opp'):
        if not _level(row.get(f'{side}_king_lvl')) or not _present(row.get(f'{side}_tower_troop')):return False
        tower=row.get(f'{side}_tower_lvl')
        if tower not in (None,'') and not _level(tower):return False
        for i in range(8):
            if not _present(row.get(f'{side}_card_{i}')) or not _level(row.get(f'{side}_card_{i}_lvl')):return False
    return True


def _audit(message,*args):
    _LOG.info(message+'; outcome_data_used=False',*args,extra={'outcome_data_used':False})


def ranked_candidates(rows,seed,excluded,start,end,modes):
    """Return all eligible IDs in SHA-256(seed:ID) order, rejecting duplicate IDs.

    Only IDs are retained; placement availability may require any ranked candidate.
    """
    excluded={_tag(bid) for bid in excluded};seen=set();ids=[]
    modes=None if modes is None else frozenset(modes)
    for row in rows:
        bid=_tag(row.get('replayTag'))
        if not bid:continue
        if bid in seen:raise ValueError(f'Duplicate replay ID: {bid}')
        seen.add(bid)
        if bid not in excluded and metadata_eligible(row,start,end,modes):ids.append(bid)
    ids.sort(key=lambda bid:(hashlib.sha256(f'{seed}:{bid}'.encode()).digest(),bid))
    _audit('Ranked %d eligible replay IDs from %d unique IDs',len(ids),len(seen))
    return ids


def choose_replays(candidates,placements,n):
    """Return the first n IDs with source-valid placement times on both teams.

    Invalid rows do not establish team coverage; valid rows on both teams suffice.
    Times are source ticks in [0,6000]. A parser that substitutes a numeric time
    for a malformed source must preserve time_valid=False on that placement.
    """
    if not isinstance(n,int) or isinstance(n,bool) or n<0:raise ValueError('Requested count must be a nonnegative integer')
    chosen=[];seen=set()
    if n:
        for bid in candidates:
            if bid in seen:raise ValueError(f'Duplicate replay ID: {bid}')
            seen.add(bid);teams=set()
            for play in placements.get(bid,()):
                if play.get('time_valid',True) is not True:continue
                t=_number(play.get('time'))
                if t is None or not 0<=t<=6000:continue
                team=play.get('team')
                if team in ('blue','red'):teams.add(team)
                if len(teams)==2:break
            if len(teams)==2:chosen.append(bid)
            if len(chosen)==n:break
    _audit('Selected %d of %d requested replay IDs',len(chosen),n)
    if len(chosen)!=n:raise ValueError(f'Requested {n} replays, found only {len(chosen)} with valid placements on both teams')
    return chosen
