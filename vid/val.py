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
    return pd.DataFrame(rows).sort_values('peak')


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
    for card, g in ht.groupby('card'):
        d = dmg(card, 15)
        rows.append({'card': card, 'n': len(g), 'loss_median': g.loss.median(), 'loss_mode': g.loss.mode().iloc[0], 'data_dmg': d,
                     'frac_within_10pct': float((abs(g.loss - d) <= 0.1 * d).mean()) if d else np.nan})
    print('(c) tower HP loss per hit near a lone attributed track (level 15 assumed)')
    print(pd.DataFrame(rows).round(2).to_string(index=False) if rows else '  none')
    lt = lifetimes(games)
    rows = []
    for (card, team), g in lt.groupby(['card', 'team']):
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


if __name__ == '__main__':
    report(sys.argv[1] if len(sys.argv) > 1 else 'golem1080')
