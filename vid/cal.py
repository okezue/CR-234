import json
import os
import cv2
import numpy as np

TOWERS = {'rk': (9, 29), 'rl': (3.5, 25.5), 'rr': (14.5, 25.5), 'bk': (9, 3), 'bl': (3.5, 6.5), 'br': (14.5, 6.5)}
# tower labels are grouped and centred on the tower, so their x is skin independent (princess anchor: bar left edge, king: crown centre);
# their height follows the tower skin, so only the own side (constant within a session) contributes y, measured once against the floor checker
XOFF = {'rk': 0, 'rl': -0.77, 'rr': -0.77, 'bk': 0, 'bl': -0.77, 'br': -0.77}
YOFF = {'bk': -2.38, 'bl': 0.54, 'br': 0.54}
# deploy-tint border lines (centre of the bright core), the line is drawn just inside the zone
LINES = {'bot': 15.15, 'top': 31.0, 'left': 0.11, 'right': 17.93}
CACHE = 'data/raw/vid'


# HSV ranges (OpenCV hue 0..179); red hues wrap so those have two ranges
RANGES = {'red': [((168, 180, 130), (176, 255, 240))], 'rdark': [((158, 90, 70), (170, 170, 125))],
          'rbadge': [((168, 170, 100), (179, 255, 200)), ((0, 170, 100), (3, 255, 200))],
          'blue': [((96, 100, 200), (106, 200, 255))], 'bdark': [((104, 80, 85), (118, 150, 140))],
          'bbadge': [((98, 100, 150), (106, 200, 235))], 'gold': [((15, 150, 180), (35, 255, 255))],
          'line': [((177, 150, 225), (179, 255, 255)), ((0, 150, 225), (3, 255, 255))],
          'white': [((0, 0, 200), (179, 110, 255))], 'bright': [((0, 0, 190), (179, 80, 255))], 'pale': [((0, 0, 200), (179, 150, 255))],
          'drop': [((143, 180, 60), (158, 255, 255))],
          'panel': [((98, 150, 120), (118, 255, 255))]}


def masks(fr):
    h = cv2.cvtColor(fr, cv2.COLOR_BGR2HSV)
    out = {}
    for k, rs in RANGES.items():
        m = cv2.inRange(h, *map(np.array, rs[0]))
        for lo, hi in rs[1:]:
            m |= cv2.inRange(h, np.array(lo), np.array(hi))
        out[k] = m // 255
    return out


def comps(mask, wmin, wmax, hmin, hmax):
    n, _, st, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [tuple(int(v) for v in st[i, :4]) for i in range(1, n) if wmin <= st[i, 2] <= wmax and hmin <= st[i, 3] <= hmax]


def run(idx):
    if len(idx) == 0:
        return None
    cuts = np.where(np.diff(idx) > 2)[0]
    runs = np.split(idx, cuts + 1)
    r = max(runs, key=len)
    return float(np.median(r))


def lines(m, h, w):
    L = m['line']
    out = {}
    rows = L.sum(1)
    out['bot'] = run(np.where(rows[int(0.35 * h):int(0.55 * h)] > 0.25 * w)[0] + int(0.35 * h))
    side = np.maximum(L[:, :int(0.3 * w)].sum(1), L[:, int(0.7 * w):].sum(1))
    out['top'] = run(np.where(side[int(0.1 * h):int(0.2 * h)] > 0.12 * w)[0] + int(0.1 * h))
    cols = L[int(0.2 * h):int(0.5 * h)].sum(0)
    out['left'] = run(np.where(cols[:int(0.05 * w)] > 0.2 * h)[0])
    out['right'] = run(np.where(cols[int(0.95 * w):] > 0.2 * h)[0] + int(0.95 * w))
    return {k: v for k, v in out.items() if v is not None}


def bars(m, w, h, anc=None):
    # unit markers can be as wide as a tower bar, so with a calibration only bars at the anchors count
    s = w / 864
    out = {}
    for team, key in (('r', 'red'), ('b', 'blue')):
        # own digits sit inside the bar and cut its mask, so gaps up to half a digit row are bridged
        mk = cv2.morphologyEx(m[key] | m[key[0] + 'dark'], cv2.MORPH_CLOSE, np.ones((1, int(25 * s)), np.uint8))
        cs = [c for c in comps(mk, 90 * s, 125 * s, 5 * s, 30 * s) if c[1] < 0.8 * h and abs(c[0] + c[2] / 2 - w / 2) > 0.15 * w]
        cs = [c for c in cs if (c[1] < h / 2) == (team == 'r')]
        for c in cs:
            k = team + ('l' if c[0] < w / 2 else 'r')
            if anc and k in anc and not (abs(c[0] - anc[k][0]) < 15 * s and abs(c[1] + c[3] / 2 - anc[k][1]) < (40 if team == 'r' else 15) * s):
                continue
            if k not in out or c[2] > out[k][2]:
                out[k] = c
        # the king bar only appears once the king is activated and is wider, with a gold frame on the own side
        cs = [c for c in comps(mk, 125 * s, 180 * s, 5 * s, 30 * s) if c[1] < 0.8 * h and abs(c[0] + c[2] / 2 - w / 2) < 0.1 * w]
        cs = [c for c in cs if (c[1] < 0.25 * h if team == 'r' else c[1] > 0.65 * h)]
        if len(cs) == 1:
            out[team + 'k'] = cs[0]
    return out


def anchors(fr, m=None):
    m = m or masks(fr)
    h, w = fr.shape[:2]
    s = w / 864
    a = {k: (c[0], c[1] + c[3] / 2) for k, c in bars(m, w, h).items() if k[1] != 'k'}
    gold = cv2.morphologyEx(m['gold'].astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    for k, lo, hi in (('rk', 0, 0.25 * h), ('bk', 0.65 * h, 0.8 * h)):
        cs = [c for c in comps(gold, 30 * s, 60 * s, 25 * s, 60 * s) if abs(c[0] + c[2] / 2 - w / 2) < 0.05 * w and lo < c[1] < hi]
        if len(cs) == 1:
            a[k] = (cs[0][0] + cs[0][2] / 2, cs[0][1] + cs[0][3] / 2)
    a['lines'] = lines(m, h, w)
    return a


def lin(px, tl):
    A = np.c_[px, np.ones(len(px))]
    p, *_ = np.linalg.lstsq(A, tl, rcond=None)
    return p, float(np.sqrt(((A @ p - tl) ** 2).mean()))


def fit(anc):
    ln = anc.get('lines', {})
    xu = [(anc[k][0], TOWERS[k][0] + XOFF[k]) for k in XOFF if k in anc] + [(ln[k], LINES[k]) for k in ('left', 'right') if k in ln]
    yv = [(anc[k][1], TOWERS[k][1] + YOFF[k]) for k in YOFF if k in anc] + [(ln[k], LINES[k]) for k in ('bot', 'top') if k in ln]
    if len(xu) < 3 or len(yv) < 3:
        return None
    (p, q), rx = lin(*map(np.array, zip(*xu)))
    (r, s), ry = lin(*map(np.array, zip(*yv)))
    inv = np.array([[p, 0, q], [0, r, s]])
    fwd = np.array([[1 / p, 0, -q / p], [0, 1 / r, -s / r]])
    return inv, fwd, float(np.hypot(rx, ry)), len(xu) + len(yv)


def clock_box(fr, fwd, m=None):
    # the clock box is a translucent overlay, so its white digits are the stable feature
    m = m or masks(fr)
    fwd = np.asarray(fwd)
    h, w = fr.shape[:2]
    d = m['white'].copy()
    d[int((fwd @ [9, 32, 1])[1]):] = False
    d[:, : int((fwd @ [13, 0, 1])[0])] = False
    cs = comps(d, 0.008 * w, 0.05 * w, 0.012 * h, 0.02 * h)
    rows = [[b for b in cs if abs(b[1] - a[1]) < 8] for a in cs]
    cs = max(rows, key=len) if rows else []
    if len(cs) < 3:
        return None
    x0, y0 = min(c[0] for c in cs), min(c[1] for c in cs)
    x1, y1 = max(c[0] + c[2] for c in cs), max(c[1] + c[3] for c in cs)
    p = int(0.035 * w)
    return (max(0, x0 - p), y0 - 4, min(w, x1 + p) - max(0, x0 - p), y1 - y0 + 8)


def hand_box(fr, m=None):
    m = m or masks(fr)
    h, w = fr.shape[:2]
    rows = np.where(m['panel'].sum(1) > 0.5 * w)[0]
    rows = rows[rows > h // 2]
    return (0, int(rows[0]), w, h - int(rows[0])) if len(rows) else None


def to_tile(cal, u, v):
    return np.asarray(cal['inv']) @ np.array([u, v, 1.0])


def calibrate(video, step=5.0, force=False):
    stem = os.path.splitext(os.path.basename(video))[0]
    path = os.path.join(CACHE, stem + '.cal.json')
    if os.path.exists(path) and not force:
        return json.load(open(path))
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    best = None
    for f in range(0, n, int(step * fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        m = masks(fr)
        a = anchors(fr, m)
        r = fit(a)
        if r is None or r[3] < 11:
            continue
        inv, fwd, res, _ = r
        if best is None or res < best['res']:
            best = {'inv': inv.tolist(), 'fwd': fwd.tolist(), 'res': res, 'frame': f, 'fps': fps, 'size': [fr.shape[1], fr.shape[0]],
                    'anchors': {k: (v if k == 'lines' else list(map(float, v))) for k, v in a.items()},
                    'clock': clock_box(fr, fwd, m), 'hand': hand_box(fr, m)}
        if res < 0.1:
            break
    cap.release()
    if best is None:
        raise RuntimeError('no frame with all tower labels and tint borders')
    os.makedirs(CACHE, exist_ok=True)
    json.dump(best, open(path, 'w'))
    return best
