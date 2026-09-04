import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
import cv2
import numpy as np
import pandas as pd
from vid import bars, ocr
from vid.cal import CACHE, calibrate, masks, to_tile

GATE = 20.0  # tiles per second of gap between a track's last detection and a candidate, capped so a lost track cannot hop units
LOST = 0.5  # seconds without a detection before a track ends
MINLEN = 0.5
REG, OT = 180, 120
_G = {}


def _init(video, cal, tpl, names):
    cv2.setNumThreads(1)  # shared machine: one core per worker process
    _G.update(video=video, cal=cal, dg=ocr.Digits(tpl), names=ocr.Names(*names))


def _chunk(rng):
    f0, f1 = rng
    cap = cv2.VideoCapture(_G['video'])
    cap.set(cv2.CAP_PROP_POS_FRAMES, f0)
    cal, dg, names = _G['cal'], _G['dg'], _G['names']
    out = []
    for f in range(f0, f1):
        ok, fr = cap.read()
        if not ok:
            break
        try:
            m = masks(fr, cal.get('gain', 1.0))
            ds = bars.detect(fr, cal, m)
            for d in ds:
                d['level'] = ocr.level(fr, d['badge'], d['team'], dg, m, cal.get('s'))
            out.append({'f': f, 'clock': ocr.clock(fr, cal, dg, m), 'towers': ocr.towers(fr, cal, dg, m),
                        'dets': [(d['team'], d['u'], d['v'], d['x'], d['y'], d['hp'], d['level']) for d in ds],
                        'banners': [(b['card'], b['level'], b['err'], b['u'], b['v'], b['err2'], b['drop']) for b in ocr.banner(fr, cal, dg, names, m)
                                    if b['err'] <= 1.0]})
        except Exception as e:  # one odd frame must not sink a chunk
            out.append({'f': f, 'clock': None, 'towers': {}, 'dets': [], 'banners': [], 'error': repr(e)})
    cap.release()
    return out


def scan(video, cal, procs=3, chunk=1500, limit=None):
    stem = os.path.splitext(os.path.basename(video))[0]
    path = os.path.join(CACHE, stem + '.scan.pkl')
    if os.path.exists(path):
        return pd.read_pickle(path)
    n = int(cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT))
    n = min(n, limit or n)
    tpl = {k: v.tolist() for k, v in ocr.templates(video, cal).tpl.items()}
    cards = json.load(open('data/cards.json'))['cards']
    names = ([v['name'] for v in cards.values()], {v['name']: v['cost'] for v in cards.values()})
    rngs = [(a, min(a + chunk, n)) for a in range(0, n, chunk)]
    # chunks are cached one by one and the pool is rebuilt when the shared machine kills a worker, so a crash costs one chunk
    part = os.path.join(CACHE, stem + '.scan')
    os.makedirs(part, exist_ok=True)
    todo = [r for r in rngs if not os.path.exists(os.path.join(part, f'{r[0]}.pkl'))]
    for _ in range(5):
        if not todo:
            break
        try:
            with ProcessPoolExecutor(procs, initializer=_init, initargs=(video, cal, tpl, names)) as p:
                for r, out in zip(todo, p.map(_chunk, todo)):
                    pd.to_pickle(out, os.path.join(part, f'{r[0]}.pkl'))
                    print(f'\r{len(rngs) - len(todo) + todo.index(r) + 1}/{len(rngs)}', end='', file=sys.stderr)
        except BrokenProcessPool:
            print(' worker lost, retrying', file=sys.stderr)
        todo = [r for r in rngs if not os.path.exists(os.path.join(part, f'{r[0]}.pkl'))]
    if todo:
        raise RuntimeError(f'chunks failed repeatedly: {todo}')
    print(file=sys.stderr)
    rows = [row for r in rngs for row in pd.read_pickle(os.path.join(part, f'{r[0]}.pkl'))]
    pd.to_pickle(rows, path)
    return rows


def games(rows, fps):
    # a game is a run of clock readings without a gap over 5 s; a jump back up towards 3:00 starts a new one (overtime restarts at 2:00)
    cl = [(r['f'], r['clock']) for r in rows if r['clock'] is not None and r['clock'] <= REG]
    out, cur = [], []
    for f, c in cl:
        if cur and (f - cur[-1][0] > 5 * fps or (c >= 165 and c > cur[-1][1] + 15)):
            out.append(cur)
            cur = []
        cur.append((f, c))
    if cur:
        out.append(cur)
    return [g for g in out if g[-1][0] - g[0][0] > 60 * fps]


def clock_time(g, fps):
    # elapsed seconds from the frames where the displayed second changes, keeping changes that chain with a neighbour
    # (the clock blinks in the last 30 s, so every other second can be missing); the overtime clock restarts, so the count continues
    ch = [(f, c) for i, (f, c) in enumerate(g) if i == 0 or c != g[i - 1][1]]
    keep = [(f, c) for i, (f, c) in enumerate(ch) if (i > 0 and 1 <= ch[i - 1][1] - c <= 3) or (i + 1 < len(ch) and 1 <= c - ch[i + 1][1] <= 3)]
    fs, el, base = [], [], 0
    for i, (f, c) in enumerate(keep):
        if i and c > keep[i - 1][1] + 30:
            base = el[-1] + 1 + c
        fs.append(f)
        el.append(REG - c if base == 0 else base - c)
    return lambda f: np.interp(f, fs, np.array(el, float))


def link(dets, fps):
    tracks, active = [], []
    for f, ds in dets:
        used = set()
        for tr in active:
            gap = (f - tr['f'][-1]) / fps
            best, bd = None, min(GATE * gap, 1.5) + 0.3
            for i, d in enumerate(ds):
                if i in used or d[0] != tr['team']:
                    continue
                dist = np.hypot(d[3] - tr['x'][-1], d[4] - tr['y'][-1])
                if dist < bd:
                    best, bd = i, dist
            if best is not None:
                d = ds[best]
                used.add(best)
                for k, v in zip(('f', 'x', 'y', 'hp', 'level'), (f, d[3], d[4], d[5], d[6])):
                    tr[k].append(v)
        for i, d in enumerate(ds):
            if i not in used:
                active.append({'team': d[0], 'f': [f], 'x': [d[3]], 'y': [d[4]], 'hp': [d[5]], 'level': [d[6]]})
        tracks += [tr for tr in active if f - tr['f'][-1] > LOST * fps]
        active = [tr for tr in active if f - tr['f'][-1] <= LOST * fps]
    tracks += active
    return [tr for tr in tracks if tr['f'][-1] - tr['f'][0] >= MINLEN * fps]


def static(rows, cal, cell=8, frac=0.4):
    # fixed scenery that passes the colour filters shows up in the same pixel cell for most of the video
    cnt = {}
    for r in rows:
        for d in r['dets']:
            k = (d[0], int(d[1] // cell), int(d[2] // cell))
            cnt[k] = cnt.get(k, 0) + 1
    bad = {k for k, v in cnt.items() if v > frac * len(rows)}
    return lambda d: (d[0], int(d[1] // cell), int(d[2] // cell)) in bad


def banners(rows, tfun, cal, fps, cards):
    # frames of one banner are clustered by place and time and vote on the name (the elixir drop hides letters in some frames);
    # in this recording the banner (with the elixir drop) belongs to the player's own cards, so the team is taken per game from
    # which half the own-placement troop and building banners land on
    ev = []
    for r in rows:
        for card, lv, err, u, v, err2, drop in r['banners']:
            x, y = to_tile(cal, u, v)
            for e in ev:
                if r['f'] - e['f1'] < 1.5 * fps and np.hypot(e['x'] - x, e['y'] - y) < 2:
                    e['f1'] = r['f']
                    e['votes'].append((card, err, lv, err2))
                    e['drop'] += drop
                    break
            else:
                ev.append({'f': r['f'], 'f1': r['f'], 'x': float(x), 'y': float(y), 'votes': [(card, err, lv, err2)], 'drop': int(drop)})
    out = []
    for e in ev:
        if len(e['votes']) < 3:
            continue
        w = {}
        for card, err, lv, _ in e['votes']:
            w[card] = w.get(card, 0) + 1 / (err + 0.1)
        card = max(w, key=w.get)
        lvs = [lv for c, _, lv, _ in e['votes'] if c == card and lv is not None]
        margin = float(np.median([e2 - err for c, err, _, e2 in e['votes'] if c == card]))
        out.append({'f': e['f'], 't': float(tfun(e['f'])), 'card': card, 'level': max(set(lvs), key=lvs.count) if lvs else None,
                    'err': float(min(err for c, err, _, _ in e['votes'] if c == card)), 'margin': margin, 'x': e['x'], 'y': e['y'], 'n': len(e['votes']),
                    'drop': int(e['drop'])})
    bn = pd.DataFrame(out)
    if len(bn):
        own = [c['name'] for c in cards.values() if c['kind'] != 'spell' and c['placement'] == 'own']
        ys = bn[bn.card.isin(own) & (bn.margin >= 0.5)].y
        bn['team'] = 'b' if len(ys) == 0 or ys.median() < 15 else 'r'
    return bn


def run(video, procs=3, limit=None):
    cal = calibrate(video)
    fps = cal['fps']
    rows = scan(video, cal, procs, limit=limit)
    isbad = static(rows, cal)
    byf = {r['f']: r for r in rows}
    cards = json.load(open('data/cards.json'))['cards']
    stem = os.path.splitext(os.path.basename(video))[0]
    out = []
    gs = games(rows, fps)
    # without a readable clock the whole video is one segment on video time: speeds still hold, game time and banners do not
    whole = not gs
    if whole:
        print('no clock: one segment on video time', file=sys.stderr)
        gs = [[(rows[0]['f'], REG), (rows[-1]['f'], REG)]]
    for gi, g in enumerate(gs):
        f0, f1 = g[0][0], g[-1][0]
        tfun = (lambda f: np.asarray(f, float) / fps) if whole else clock_time(g, fps)
        sub = [byf[f] for f in range(f0, f1 + 1) if f in byf]
        dets = [(r['f'], [d for d in r['dets'] if not isbad(d)]) for r in sub]
        tr = link(dets, fps)
        tab = pd.DataFrame([{'frame': f, 't': float(tfun(f)), 'track': i, 'team': t['team'], 'x': x, 'y': y, 'hp': hp, 'level': lv}
                            for i, t in enumerate(tr) for f, x, y, hp, lv in zip(t['f'], t['x'], t['y'], t['hp'], t['level'])]).astype({'level': float})
        tw = pd.DataFrame([{'frame': r['f'], 't': float(tfun(r['f'])), **{k: r['towers'].get(k) for k in ('rk', 'rl', 'rr', 'bk', 'bl', 'br')}} for r in sub])
        bn = banners(sub, tfun, cal, fps, cards)
        base = os.path.join(CACHE, f'{stem}_{gi}')
        tab.to_parquet(base + '.parquet')
        tw.to_parquet(base + '_towers.parquet')
        bn.to_parquet(base + '_banners.parquet')
        out.append({'game': gi, 'f0': f0, 'f1': f1, 'tracks': len(tr), 'banners': len(bn), 'dur': float(tfun(f1))})
    return out


if __name__ == '__main__':
    print(run(sys.argv[1], limit=int(sys.argv[2]) if len(sys.argv) > 2 else None))
