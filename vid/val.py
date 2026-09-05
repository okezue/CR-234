import glob
import json
import os
import sys
import numpy as np
import pandas as pd
from sim.cards import create, key
from sim.game import Game
from vid.cal import CACHE

CARDS = json.load(open('data/cards.json'))['cards']
BYNAME = {v['name']: k for k, v in CARDS.items()}
MAXHP = {'k': 7728, 'l': 4858, 'r': 4858}
BANNER_DY = 2.0  # tiles from the banner text bottom down to the deploy point, measured once on deploying units
WIN = 2.0


def load(stem):
    out = []
    for p in sorted(glob.glob(os.path.join(CACHE, f'{stem}_[0-9]*.parquet'))):
        if p.endswith('_towers.parquet') or p.endswith('_banners.parquet'):
            continue
        b = p[:-8]
        out.append((pd.read_parquet(p), pd.read_parquet(b + '_towers.parquet'), pd.read_parquet(b + '_banners.parquet')))
    return out


def clean(tw):
    # a reading counts when a neighbouring frame agrees, it is within the tower's range, and it is neither above the median of the
    # previous half second of readings nor below the median of the next (HP only falls)
    out, stats = tw[['frame', 't']].copy(), {}
    for k in ('rk', 'rl', 'rr', 'bk', 'bl', 'br'):
        v = tw[k].to_numpy(float)
        raw = v[~np.isnan(v)]
        ok = (v > 0) & (v <= MAXHP[k[1]])
        agree = np.r_[False, v[1:] == v[:-1]] | np.r_[v[:-1] == v[1:], False]
        c = np.where(ok & agree, v, np.nan)
        idx = np.where(~np.isnan(c))[0]
        s = pd.Series(c[idx])
        prev = s.rolling(31, min_periods=1).median().shift(1).fillna(np.inf)
        nxt = s[::-1].rolling(31, min_periods=1).median()[::-1].shift(-1).fillna(-np.inf)
        c[idx[((s > prev) | (s < nxt)).to_numpy()]] = np.nan
        out[k] = c
        cl = c[~np.isnan(c)]
        mono = lambda a: float((np.diff(a) <= 0).mean()) if len(a) > 1 else np.nan
        stats[k] = (len(raw), mono(raw), len(cl), mono(cl))
    return out, stats


def attribute(tr, bn):
    # a track belongs to an unambiguous banner (next best name at least 0.5 worse) of its own team that appeared up to 4.5 s
    # before it spawned within 4 tiles; each banner takes at most as many tracks as the card has units
    first = tr.sort_values('frame').groupby('track').first()
    bn = bn[bn.margin >= 0.5]
    out, used = {}, {}
    for i, r in first.iterrows():
        c = bn[(bn.team == r.team) & (r.t - bn.t > -0.3) & (r.t - bn.t < 4.5)]
        if len(c) == 0:
            continue
        d = np.hypot(c.x - r.x, c.y - r.y)
        j = d.idxmin()
        k = BYNAME.get(c.card[j])
        if d[j] < 4 and k and used.get(j, 0) < (CARDS[k]['count'] or 1):
            used[j] = used.get(j, 0) + 1
            out[i] = (c.card[j], int(c.level[j]) if c.level[j] == c.level[j] and c.level[j] else 15)
    return out


def steady(t, x, y, win=WIN, rate=60):
    # best constant-velocity window: linear fit residual under 0.15 tiles, moving faster than 0.2 tiles/s
    best = None
    for t0 in np.arange(t[0], t[-1] - win + 1e-9, 0.25):
        m = (t >= t0) & (t <= t0 + win)
        if m.sum() < 0.6 * win * rate:
            continue
        A = np.c_[t[m], np.ones(m.sum())]
        px, *_ = np.linalg.lstsq(A, x[m], rcond=None)
        py, *_ = np.linalg.lstsq(A, y[m], rcond=None)
        res = np.sqrt(np.mean((x[m] - A @ px) ** 2 + (y[m] - A @ py) ** 2))
        v = float(np.hypot(px[0], py[0]))
        if res < 0.15 and v > 0.2 and (best is None or res < best[1]):
            best = (v, res, t0)
    return best


def speeds(games):
    rows = []
    for gi, (tr, tw, bn) in enumerate(games):
        att = attribute(tr, bn)
        for i, g in tr.groupby('track'):
            g = g.sort_values('t')
            if g.t.iloc[-1] - g.t.iloc[0] < WIN:
                continue
            s = steady(g.t.to_numpy(), g.x.to_numpy(), g.y.to_numpy())
            if s:
                card, lvl = att.get(i, (None, None))
                rows.append({'game': gi, 'track': i, 'team': g.team.iloc[0], 'card': card, 'level': lvl, 'speed': s[0], 'res': s[1],
                             'x0': g.x.iloc[0], 'y0': g.y.iloc[0]})
    return pd.DataFrame(rows)


def place(g, team, x, y):
    # snap the deploy point to the nearest free tile on the own half, banners are read with a coarse offset
    yy = min(max(y, 0.5), 31.5)
    for d in range(0, 6):
        for cand in ((x, yy - d), (x, yy + d), (x - d, yy), (x + d, yy)):
            cx, cy = min(max(cand[0], 0.5), 17.5), min(max(cand[1], 0.5), 31.5)
            if (cy < 15) == (team == 'blue') and not g.arena.blocked(int(cx), int(cy)):
                return int(cx) + 0.5, int(cy) + 0.5
    return x, yy


def sim_units(card, lvl, team, x, y):
    g = Game(p1={'king_lvl': 15}, p2={'king_lvl': 15})
    x, y = place(g, team, x, y)
    us = create(key(BYNAME.get(card, card)), lvl, team, x, y)
    us = us if isinstance(us, list) else [us]
    for u in us:
        g.deploy(team, u)
    return g, us


def sim_speed(card, lvl, team, x, y, dur=30):
    g, us = sim_units(card, lvl, team, x, y)
    u = us[0]
    if not hasattr(u, 'spd'):
        return None
    pos = []
    while g.t < dur and u.alive:
        g.tick()
        pos.append((g.t, u.x, u.y))
    p = np.array(pos)
    if len(p) < 40:
        return None
    s = steady(p[:, 0], p[:, 1], p[:, 2], rate=20)
    return s[0] if s else None


def first_drop(tw, team, t0, dmg, horizon=60):
    # first step after t0 in a cleaned HP series of the other team's towers whose size is that card's tower damage (within 12%),
    # other attackers hit the same towers so an unqualified first drop would be theirs
    cols = [k for k in ('rk', 'rl', 'rr', 'bk', 'bl', 'br') if k[0] != team[0]]
    best = None
    for k in cols:
        s = tw[['t', k]].dropna()
        s = s[(s.t > t0 - 1) & (s.t < t0 + horizon)]
        d = -s[k].diff()
        hit = s[(d > 0.88 * dmg) & (d < 1.12 * dmg) & (s.t > t0)]
        if len(hit) and (best is None or hit.t.iloc[0] < best[0]):
            best = (float(hit.t.iloc[0]), k, float(d[hit.index[0]]))
    return best


def level_of(b, bn):
    # a level hidden by the opponent's drop, or misread as one digit, falls back to the card's usual level in the video
    lv = b.level if b.level == b.level and b.level and b.level >= 9 else None
    if lv is None:
        m = bn[(bn.card == b.card) & (bn.level >= 9)].level
        lv = m.mode().iloc[0] if len(m) else 15
    return int(lv)


def deploy_to_hit(games):
    rows = []
    allbn = pd.concat([b for _, _, b in games])
    for gi, (tr, tw, bn) in enumerate(games):
        twc, _ = clean(tw)
        for _, b in bn.iterrows():
            k = BYNAME.get(b.card)
            if k is None or CARDS[k].get('targets') != ['buildings'] or not isinstance(b.team, str) or b.margin < 0.5:
                continue
            lv = level_of(b, allbn)
            fd = first_drop(twc, b.team, b.t, dmg(b.card, lv))
            if fd:
                rows.append({'game': gi, 'card': b.card, 'level': lv, 'team': b.team, 'x': b.x, 'y': b.y - BANNER_DY, 't': b.t,
                             'dt': fd[0] - b.t, 'tower': fd[1], 'loss': fd[2]})
    return pd.DataFrame(rows)


def sim_deploy_to_hit(card, lvl, team, x, y, horizon=60):
    g, us = sim_units(card, lvl, team, x, y)
    hp0 = {id(t): t.hp for t in g.arena.towers}
    while g.t < horizon and any(u.alive for u in us):
        g.tick()
        for t in g.arena.towers:
            if t.team != team and t.hp < hp0[id(t)]:
                return g.t, hp0[id(t)] - t.hp
    return None


def spawn_hits(games, card, lvl=15, tol=0.08):
    # enemy units carry their badge from the card drop, so badge appearance to the first tower step of this card's damage
    # measures deploy time plus approach; the sim places the unit at the observed tile without the deploy time
    from vid.cal import TOWERS
    d0 = dmg(card, lvl)
    rows = []
    for gi, (tr, tw, bn) in enumerate(games):
        twc, _ = clean(tw)
        for k in ('bl', 'br', 'bk'):
            s = twc[['t', k]].dropna()
            d = -s[k].diff()
            ts = s.t[(d > (1 - tol) * d0) & (d < (1 + tol) * d0)].to_numpy()
            starts = [t for i, t in enumerate(ts) if i == 0 or t - ts[i - 1] > 5]
            for t0 in starts:
                near = tr[(tr.team == 'r') & (abs(tr.t - t0) < 0.5)]
                near = near[np.hypot(near.x - TOWERS[k][0], near.y - TOWERS[k][1]) < 6]
                for i in near.track.unique():
                    g = tr[tr.track == i].sort_values('t')
                    if not 1 < t0 - g.t.iloc[0] < 12 or g.y.iloc[0] < 16:
                        continue
                    sim = sim_deploy_to_hit(card, lvl, 'red', g.x.iloc[0], g.y.iloc[0])
                    rows.append({'game': gi, 'tower': k, 'track': i, 'x0': g.x.iloc[0], 'y0': g.y.iloc[0], 't_spawn': g.t.iloc[0], 'dt': t0 - g.t.iloc[0],
                                 'deploy': CARDS[BYNAME[card]]['deployTime'] or 1.0, 'sim_dt': sim[0] if sim else np.nan})
    return pd.DataFrame(rows)


def intervals(games, card, lvl=15, tol=0.02):
    # consecutive tower steps of one card's exact damage are that unit's hit cadence: a clock check independent of any pixel scale
    d0 = dmg(card, lvl)
    out = []
    for gi, (tr, tw, bn) in enumerate(games):
        twc, _ = clean(tw)
        for k in ('rk', 'rl', 'rr', 'bk', 'bl', 'br'):
            s = twc[['t', k]].dropna()
            d = -s[k].diff()
            ts = s.t[(d > (1 - tol) * d0) & (d < (1 + tol) * d0)].to_numpy()
            out += [b - a for a, b in zip(ts[:-1], ts[1:]) if b - a < 2 * CARDS[BYNAME[card]]['hitSpeed']]
    return np.array(out)


def hits(games):
    # tower HP steps while exactly one attributed enemy track stands within 6 tiles of that tower
    from vid.cal import TOWERS
    rows = []
    for gi, (tr, tw, bn) in enumerate(games):
        att = attribute(tr, bn)
        twc, _ = clean(tw)
        for k in ('rk', 'rl', 'rr', 'bk', 'bl', 'br'):
            s = twc[['t', k]].dropna()
            d = s[k].diff()
            for t, loss in zip(s.t[d < 0], -d[d < 0]):
                near = tr[(tr.team != k[0]) & (abs(tr.t - t) < 0.2)]
                near = near[np.hypot(near.x - TOWERS[k][0], near.y - TOWERS[k][1]) < 6]
                ids = {i for i in near.track.unique() if i in att}
                if len(ids) == 1:
                    rows.append({'game': gi, 'tower': k, 't': t, 'loss': loss, 'card': att[ids.pop()][0]})
    return pd.DataFrame(rows)


def lifetimes(games):
    rows = []
    for gi, (tr, tw, bn) in enumerate(games):
        att = attribute(tr, bn)
        for i, g in tr.groupby('track'):
            if i not in att:
                continue
            g = g.sort_values('t')
            rows.append({'game': gi, 'track': i, 'card': att[i][0], 'level': att[i][1], 'team': g.team.iloc[0], 'x0': g.x.iloc[0], 'y0': g.y.iloc[0],
                         'life': g.t.iloc[-1] - g.t.iloc[0], 'hp_end': g.hp.iloc[-1], 'died': g.hp.iloc[-1] < 0.3})
    # own units carry no badge until damaged, so their tracks start at the first hit rather than at spawn
    return pd.DataFrame(rows)


def sim_lifetime(card, lvl, team, x, y, horizon=120):
    g, us = sim_units(card, lvl, team, x, y)
    while g.t < horizon and any(u.alive for u in us):
        g.tick()
    return g.t if not any(u.alive for u in us) else None


def dmg(card, lvl):
    c = CARDS.get(BYNAME.get(card, card))
    if not c:
        return None
    st = c['stats']
    v = (st.get('towerDamage') or st.get('damage') or [None] * 16)[lvl - 1]
    return v


def quanta():
    # tower damage value -> (card, level, hit speed, load time, tiles/s) for levels 9 and up, so an exact tower step names its attacker
    out = {}
    for k, c in CARDS.items():
        if c['kind'] == 'spell':
            continue
        st = c['stats']
        for i, v in enumerate(st.get('towerDamage') or st.get('damage') or []):
            if v and i >= 8:
                out.setdefault(int(v), []).append((k, i + 1, c['hitSpeed'], c['loadTime'], (c['speed'] or 0) / 50))
    return out


def arrivals(games, own='b'):
    # enemy melee attackers at the own towers: the unit stops in reach, the tower's next exact step is its first hit; the step time is
    # bracketed by the previous OCR reading; the attacker is named by the step's damage value and its approach speed class
    from vid.cal import TOWERS
    q = quanta()
    rows = []
    for gi, (tr, tw, bn) in enumerate(games):
        twc, _ = clean(tw)
        en = tr[tr.team != own]
        for k in TOWERS:
            if k[0] != own:
                continue
            tx, ty = TOWERS[k]
            s = twc[['t', k]].dropna()
            hp, ts = s[k].to_numpy(), s.t.to_numpy()
            e = en.assign(d=np.hypot(en.x - tx, en.y - ty))
            for i in e[e.d < 3.3].track.unique():
                g = e[e.track == i].sort_values('t')
                t, x, y, d = (g[c].to_numpy() for c in ('t', 'x', 'y', 'd'))
                stop = next((j for j in range(len(t)) if d[j] < 3.3 and ((t > t[j]) & (t <= t[j] + 0.6)).sum() >= 5
                             and np.hypot(x[(t > t[j]) & (t <= t[j] + 0.6)] - x[j], y[(t > t[j]) & (t <= t[j] + 0.6)] - y[j]).max() < 0.2), None)
                if stop is None:
                    continue
                mb = (t >= t[stop] - 1.0) & (t < t[stop])
                if mb.sum() < 5 or np.hypot(x[mb][0] - x[stop], y[mb][0] - y[stop]) < 0.5:
                    continue
                spd = np.hypot(x[mb][0] - x[stop], y[mb][0] - y[stop]) / (t[stop] - t[mb][0])
                a = next((a for a in np.where(ts > t[stop] - 0.3)[0][1:] if hp[a] < hp[a - 1]), None)
                if a is None or ts[a] - t[stop] > 4:
                    continue
                loss = int(hp[a - 1] - hp[a])
                cands = [c for c in q.get(loss, []) if abs(c[4] - spd) < 0.45] or q.get(loss, [])
                others = e[(e.track != i) & (e.t > t[stop] - 1.5) & (e.t < t[stop] + 2) & (e.d < 4)].track.nunique()
                nxt = [ts[b] - ts[a] for b in range(a + 1, len(ts)) if hp[b - 1] - hp[b] == loss and ts[b] < ts[a] + 8]
                rows.append({'game': gi, 'tower': k, 'track': i, 't_stop': t[stop], 'd_stop': d[stop], 'speed': spd, 'dt_lo': ts[a - 1] - t[stop],
                             'dt_hi': ts[a] - t[stop], 'loss': loss, 'others': others, 'cands': ';'.join(f'{c[0]}@{c[1]}' for c in cands[:3]),
                             'next': [round(v, 2) for v in nxt[:3]]})
    return pd.DataFrame(rows)


def chain(tr, track, team='r', maxgap=0.6, rad=1.2):
    # follow a unit backwards through the tracker's fragments: the same-team fragment that ended nearest before this one began; the badge
    # jumps up to 3 tiles in one frame on the bridge, so the gate widens over the river band; ambiguous joins (two candidates) are counted
    r = tr[tr.team == team]
    parts, cur, seen, amb = [], track, set(), 0
    while cur is not None and cur not in seen:
        seen.add(cur)
        g = r[r.track == cur].sort_values('t')
        parts.append(g)
        t0, x0, y0 = g.t.iloc[0], g.x.iloc[0], g.y.iloc[0]
        c = r[(r.t < t0) & (r.t > t0 - maxgap) & ~r.track.isin(seen)]
        c = c.assign(dd=np.hypot(c.x - x0, c.y - y0))
        c = c[c.dd < (3.5 if 12.5 < y0 < 18.5 else rad)]
        if len(c) == 0:
            break
        ends = c.groupby('track').agg(te=('t', 'max'), dd=('dd', 'min')).sort_values(['te', 'dd'], ascending=[False, True])
        amb += int(len(ends) > 1 and t0 - ends.te.iloc[1] < 0.3)
        cur = ends.index[0]
    return pd.concat(parts).sort_values('t'), amb


def bar_steps(t, hp, thr=0.02, hold=3):
    # the bar fill flickers between 0 and 1 for single frames: a step is a drop of the 5-frame median that holds for 3 frames above empty
    m = pd.Series(hp).rolling(5, center=True, min_periods=1).median().to_numpy()
    out, cur, last = [], m[0], t[0]
    for a in range(1, len(t) - hold):
        if (m[a:a + hold] < cur - thr).all() and (m[a:a + hold] > 0.02).all() and t[a] - last > 0.25:
            out.append((float(t[a]), float(cur - np.median(m[a:a + hold]))))
            cur, last = float(np.median(m[a:a + hold])), t[a]
    return out


def tower_shots(games, own='b', reach=9.0):
    # the tower's own shots, anchored on the arrivals (a tower HP step names the attacker standing in reach): the attacker's chain of
    # fragments gives the moment its badge crossed the tower's reach (7.5 + 1 collision + 0.5) and the first frame its bar appeared; the
    # bar steps while it stands give the shot cadence; own tracks and own banners near the unit mark shots that may be a defender's
    from vid.cal import TOWERS
    rows = []
    for gi, (tr, tw, bn) in enumerate(games):
        for _, a in arrivals([(tr, tw, bn)], own).iterrows():
            tx, ty = TOWERS[a.tower]
            g, amb = chain(tr, a.track, 'r' if own == 'b' else 'b')
            g = g.assign(d=np.hypot(g.x - tx, g.y - ty))
            t, d, hp = g.t.to_numpy(), g.d.to_numpy(), g.hp.to_numpy()
            inn, out = np.where(d < reach)[0], np.where(d >= reach)[0]
            ent = first = None
            if len(out) and len(inn) and out[0] < inn[-1]:
                i = out[out < inn[-1]][-1]
                j = inn[inn > i][0]
                ent = t[i] + (t[j] - t[i]) * (d[i] - reach) / max(d[i] - d[j], 1e-6)
                lv = float(np.median(hp[max(0, i - 20):i + 1]))
                mm = pd.Series(hp).rolling(3, center=True, min_periods=1).median().to_numpy()
                f = next((q for q in range(j, len(t) - 2) if (mm[q:q + 3] < lv - 0.015).all()), None)
                first = t[f] - ent if f is not None and lv > 0.99 else None
            p = g.iloc[-1]
            st = tr[(tr.team == g.team.iloc[0]) & (tr.t >= a.t_stop) & (tr.t < a.t_stop + 8)]
            st = st[np.hypot(st.x - p.x, st.y - p.y) < 0.7].sort_values('t')
            steps = bar_steps(st.t.to_numpy(), st.hp.to_numpy()) if len(st) > 20 else []
            t0 = ent if ent is not None else a.t_stop
            ow = tr[(tr.team == own) & (tr.t > t0 - 1) & (tr.t < a.t_stop + 3)]
            bb = bn[(bn.team == own) & (bn.t > t0 - 8) & (bn.t < a.t_stop + 3)]
            rows.append({'game': gi, 'tower': a.tower, 't_stop': a.t_stop, 'loss': a.loss, 'cands': a.cands, 'frags': g.track.nunique(), 'amb': amb,
                         'entry': ent, 'first': first, 'ivals': [round(b - x, 2) for (x, _), (b, _) in zip(steps, steps[1:])],
                         'sizes': [round(s, 3) for _, s in steps],
                         'own_near': ow[np.hypot(ow.x - p.x, ow.y - p.y) < 6].track.nunique() + len(bb[np.hypot(bb.x - p.x, bb.y - 2 - p.y) < 8])})
    return pd.DataFrame(rows)


def sim_arrival(card, lvl, horizon=20):
    # the same scenario in the engine: a lone unit walks from the bridge to the princess tower; time from its last step to the first tower hit
    g = Game(p1={'king_lvl': 15}, p2={'king_lvl': 15})
    us = create(key(BYNAME.get(card, card)), lvl, 'red', 14.5, 12.5)
    u = us[0] if isinstance(us, list) else us
    g.deploy('red', u)
    tw = g.arena.get_tower('blue', 'princess', 'right')
    hp0, last, t_stop, hits = tw.hp, (u.x, u.y), None, []
    while g.t < horizon and u.alive and len(hits) < 3:
        g.tick()
        t_stop = (g.t - g.DT if t_stop is None else t_stop) if (u.x, u.y) == last else None
        last = (u.x, u.y)
        if tw.hp < hp0:
            hits.append(g.t)
            hp0 = tw.hp
    return (hits[0] - t_stop if hits and t_stop is not None else None), [round(b - a, 2) for a, b in zip(hits, hits[1:])]


CLASSES = {'slow': 0.75, 'medium': 1.0, 'fast': 1.5, 'very fast': 2.0}


def classes(sp, width=0.1):
    # unattributed tracks still test the discrete speed classes: peaks of the speed histogram against the nearest class
    v = np.sort(sp.speed.to_numpy())
    cl = np.array(list(CLASSES.values()))
    rows, used = [], np.zeros(len(v), bool)
    for _ in range(6):
        cnt = np.array([((np.abs(v - c) < width) & ~used).sum() for c in v])
        if cnt.max() < 8:
            break
        c = v[cnt.argmax()]
        m = (np.abs(v - c) < width) & ~used
        used |= m
        k = int(np.abs(cl - np.median(v[m])).argmin())
        rows.append({'peak': float(np.median(v[m])), 'n': int(m.sum()), 'q1': float(np.quantile(v[m], 0.25)), 'q3': float(np.quantile(v[m], 0.75)),
                     'class': list(CLASSES)[k], 'data': float(cl[k]), 'ratio': float(np.median(v[m]) / cl[k])})
    return pd.DataFrame(rows, columns=['peak', 'n', 'q1', 'q3', 'class', 'data', 'ratio']).sort_values('peak')


def figure(sp, cl, all_speeds, path='docs/fig/vidSpeeds.png'):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 5))
    for _, r in sp.iterrows():
        ax.errorbar(r.data, r.measured, yerr=[[max(0, r.measured - r.q1)], [max(0, r.q3 - r.measured)]], fmt='o', color='C0', capsize=3)
        ax.annotate(f'{r.card} ({r.n})', (r.data, r.measured), textcoords='offset points', xytext=(5, 3), fontsize=8)
    ax.scatter(sp.data, sp.sim, marker='x', color='C3', label='simulator, same deploy tile')
    for _, r in cl.iterrows():
        ax.errorbar(r.data, r.peak, yerr=[[max(0, r.peak - r.q1)], [max(0, r.q3 - r.peak)]], fmt='s', color='C2', capsize=3, alpha=0.7)
        ax.annotate(f'peak {r.peak:.2f} ({r.n})', (r.data, r.peak), textcoords='offset points', xytext=(5, -12), fontsize=8, color='C2')
    lim = [0.2, 2.6]
    ax.plot(lim, lim, 'k--', lw=0.8)
    ax.set_xlabel('data/cards.json speed (tiles/s)')
    ax.set_ylabel('measured from video (tiles/s, median and IQR)')
    ax.set_title('per card (blue) and histogram peaks vs nearest class (green)')
    ax.legend(loc='upper left', fontsize=8)
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax2.hist(all_speeds, bins=np.arange(0.2, 2.6, 0.05), color='C0')
    for name, c in CLASSES.items():
        ax2.axvline(c, color='C3', ls='--', lw=0.8)
        ax2.text(c, ax2.get_ylim()[1] * 0.95, name, rotation=90, va='top', ha='right', fontsize=8, color='C3')
    ax2.set_xlabel('steady track speed (tiles/s)')
    ax2.set_ylabel('tracks')
    ax2.set_title('all steady 2 s windows')
    fig.tight_layout()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150)


def report(stem):
    games = load(stem)
    pd.set_option('display.width', 200)
    print(f'{len(games)} games,', sum(t.track.nunique() for t, _, _ in games), 'tracks,', sum(len(b) for _, _, b in games), 'banners')
    agg = {}
    for _, tw, _ in games:
        _, st = clean(tw)
        for k, (n, mono, nc, monoc) in st.items():
            a = agg.setdefault(k, [0, 0, 0, 0])
            if n > 1:
                a[0] += n
                a[1] += mono * n
            if nc > 1:
                a[2] += nc
                a[3] += monoc * nc
    print('tower OCR: readings, raw monotone fraction, cleaned readings, cleaned monotone fraction')
    for k, (n, m, nc, mc) in agg.items():
        print(f'  {k}: {n} {m / max(n, 1):.3f} {nc} {mc / max(nc, 1):.3f}')
    sp = speeds(games)
    print(f'speed tracks: {len(sp)} steady, {sp.card.notna().sum()} attributed')
    rows = []
    for card, g in sp.dropna(subset=['card']).groupby('card'):
        k = BYNAME.get(card)
        if k is None or CARDS[k]['kind'] == 'spell':
            continue
        r = g.iloc[0]
        ss = [sim_speed(card, int(r.level), 'blue' if r.team == 'b' else 'red', r.x0, r.y0) for _, r in g.head(3).iterrows()]
        ss = [s for s in ss if s]
        rows.append({'card': card, 'n': len(g), 'measured': g.speed.median(), 'q1': g.speed.quantile(0.25), 'q3': g.speed.quantile(0.75),
                     'data': (CARDS[k]['speed'] or 0) / 60, 'sim': np.median(ss) if ss else np.nan})
    spt = pd.DataFrame(rows).sort_values('n', ascending=False) if rows else pd.DataFrame(columns=['card', 'n', 'measured', 'q1', 'q3', 'data', 'sim'])
    print('(a) walking speed, tiles/s'), print(spt.round(2).to_string(index=False))
    cl = classes(sp)
    print('(a2) speed histogram peaks of all steady tracks against the nearest data class'), print(cl.round(2).to_string(index=False))
    figure(spt, cl, sp.speed.to_numpy())
    dh = deploy_to_hit(games)
    rows = []
    for _, r in dh.iterrows():
        s = sim_deploy_to_hit(r.card, r.level, 'blue' if r.team == 'b' else 'red', r.x, r.y)
        rows.append({**r, 'sim_dt': s[0] if s else np.nan, 'sim_loss': s[1] if s else np.nan})
    dht = pd.DataFrame(rows)
    print('(b) deploy banner to first tower HP drop, s'), print(dht.round(2).to_string(index=False) if len(dht) else '  none')
    ht = hits(games)
    rows = []
    for card, g in (ht.groupby('card') if len(ht) else []):
        d = dmg(card, 15)
        rows.append({'card': card, 'n': len(g), 'loss_median': g.loss.median(), 'loss_mode': g.loss.mode().iloc[0], 'data_dmg': d,
                     'frac_within_10pct': float((abs(g.loss - d) <= 0.1 * d).mean()) if d else np.nan})
    print('(c) tower HP loss per hit near a lone attributed track (level 15 assumed)')
    print(pd.DataFrame(rows).round(2).to_string(index=False) if rows else '  none')
    lt = lifetimes(games)
    rows = []
    for (card, team), g in (lt.groupby(['card', 'team']) if len(lt) else []):
        k = BYNAME.get(card)
        if k is None or CARDS[k]['kind'] == 'spell':
            continue
        sl = [sim_lifetime(card, int(r.level), 'blue' if r.team == 'b' else 'red', r.x0, r.y0) for _, r in g.head(3).iterrows()]
        sl = [s for s in sl if s]
        rows.append({'card': card, 'team': team, 'n': len(g), 'died': int(g.died.sum()), 'life_median': g.life.median(),
                     'life_died_median': g[g.died].life.median(), 'sim_alone': np.median(sl) if sl else np.nan})
    print('(d) track lifetime, s (own units carry no badge until hit, so team b tracks start at the first damage; sim: unit alone against towers)')
    print(pd.DataFrame(rows).sort_values('n', ascending=False).round(1).to_string(index=False) if rows else '  none')
    return spt, dht, ht, lt


def peaks(stems):
    rows = []
    for st in stems:
        games = load(st)
        sp = speeds(games)
        cl = classes(sp)
        cl = cl[cl.peak > 0.7]
        rows.append({'video': st, 'games': len(games), 'steady': len(sp), **{f'peak{i}': f'{r.peak:.2f} (n={r.n})' for i, r in enumerate(cl.itertuples())}})
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == '__main__':
    if len(sys.argv) > 2:
        peaks(sys.argv[1:])
    else:
        report(sys.argv[1] if len(sys.argv) > 1 else 'golem1080')
