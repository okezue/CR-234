import json
import os
import cv2
import numpy as np
from vid.cal import CACHE, bars, comps, masks

GW, GH = 12, 18


def split(mask, c, top):
    # bold digits touch at this scale, so a glyph wider than a digit is cut at the thinnest columns near equal splits
    x, y, w, h = c
    n = int(round(w / (0.8 * top)))
    if n < 2:
        return [c]
    col = mask[y:y + h, x:x + w].sum(0)
    cuts = [0]
    for i in range(1, n):
        lo, hi = int(w * (i - 0.25) / n), int(w * (i + 0.25) / n)
        cuts.append(lo + int(np.argmin(col[lo:hi])))
    cuts.append(w)
    return [(x + a, y, b - a, h) for a, b in zip(cuts[:-1], cuts[1:]) if b > a]


def glyphs(mask, hmin, hmax):
    cs = [c for c in comps(mask, 2, 6 * hmax, hmin, hmax)]
    if not cs:
        return []
    top = max(c[3] for c in cs)
    cs = [c for c in cs if c[3] >= 0.6 * top]
    rows = [[b for b in cs if abs(b[1] + b[3] / 2 - a[1] - a[3] / 2) < 0.4 * top] for a in cs]
    cs = max(rows, key=len)
    return sorted(g for c in cs for g in split(mask, c, top))


def norm(mask, c):
    x, y, w, h = c
    g = cv2.resize(mask[y:y + h, x:x + w].astype(np.float32), (GW, GH), interpolation=cv2.INTER_AREA)
    g -= g.mean()
    n = np.linalg.norm(g)
    return g / n if n > 0 else g


def ncc(a, b):
    return float((a * b).sum())


class Digits:
    def __init__(self, tpl):
        self.tpl = {int(k): np.asarray(v, np.float32) for k, v in tpl.items()}

    def read(self, mask, cs, thr=0.5):
        out, sc = [], []
        for c in cs:
            g = norm(mask, c)
            k, s = max(((k, ncc(g, t)) for k, t in self.tpl.items()), key=lambda p: p[1])
            if c[2] < 0.45 * c[3] and mask[c[1]:c[1] + c[3], c[0]:c[0] + c[2]].mean() > 0.6:
                k, s = 1, 0.9  # at label size the 1 is a plain bar, which the scale-normalised template cannot match
            out.append(str(k))
            sc.append(s)
        return (''.join(out), min(sc)) if out and min(sc) >= thr else (None, min(sc) if sc else 0)

    def number(self, mask, hmin, hmax):
        cs = glyphs(mask, hmin, hmax)
        v, s = self.read(mask, cs)
        return (int(v), s) if v else (None, s)


def clock_glyphs(fr, cal, m=None):
    x, y, w, h = cal['clock']
    m = m or masks(fr)
    wm = m['white'][y:y + h, x:x + w]
    s = fr.shape[1] / 864
    return wm, glyphs(wm, 18 * s, 44 * s)


def clock(fr, cal, dg, m=None):
    wm, cs = clock_glyphs(fr, cal, m)
    if len(cs) not in (3, 4):
        return None
    v, s = dg.read(wm, cs)
    if v is None:
        return None
    return int(v[:-2]) * 60 + int(v[-2:])


def learn(video, cal, start=0.0, span=13.0, step=0.1):
    # the clock counts down one digit per second, so a 13 s window labels all ten ones-digit glyph clusters:
    # the ones glyph after a tens change is 9, each later new cluster is one lower
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    seq = []
    for f in range(int(start * fps), int((start + span) * fps), max(1, int(step * fps))):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, fr = cap.read()
        if not ok:
            break
        wm, cs = clock_glyphs(fr, cal)
        if len(cs) != 3:
            continue
        seq.append((norm(wm, cs[1]), norm(wm, cs[2])))
    cap.release()
    cl, ones, tens = [], [], []
    for t, o in seq:
        for arr, g in ((tens, t), (ones, o)):
            k = next((i for i, c in enumerate(cl) if ncc(c[0], g) > 0.8), None)
            if k is None:
                cl.append([g, 1])
                k = len(cl) - 1
            else:
                cl[k][0] = (cl[k][0] * cl[k][1] + g) / (cl[k][1] + 1)
                cl[k][1] += 1
            arr.append(k)
    lab = {}
    for i in range(1, len(seq)):
        if tens[i] != tens[i - 1]:
            d = 9
            lab[ones[i]] = 9
            for j in range(i + 1, len(seq)):
                if ones[j] != ones[j - 1]:
                    d -= 1
                    if d < 0:
                        break
                    if ones[j] in lab and lab[ones[j]] != d:
                        return None
                    lab[ones[j]] = d
            break
    if sorted(lab.values()) != list(range(10)):
        return None
    return {str(d): cl[k][0].tolist() for k, d in lab.items()}


def templates(video, cal, force=False):
    stem = os.path.splitext(os.path.basename(video))[0]
    path = os.path.join(CACHE, stem + '.digits.json')
    if os.path.exists(path) and not force:
        return Digits(json.load(open(path)))
    n = cv2.VideoCapture(video).get(cv2.CAP_PROP_FRAME_COUNT) / cal['fps']
    for start in np.arange(5, n - 15, 7.0):
        tpl = learn(video, cal, start)
        if tpl:
            json.dump(tpl, open(path, 'w'))
            return Digits(tpl)
    raise RuntimeError('could not learn digit templates from the clock')


def towers(fr, cal, dg, m=None):
    # enemy digits sit above their bar, own digits inside it; a missing bar means destroyed (princess) or not yet activated (king)
    m = m or masks(fr)
    wm = m['bright']
    H, W = fr.shape[:2]
    s = W / 864
    out = {}
    for k, (x, y, w, h) in bars(m, W, H, cal['anchors']).items():
        if k[0] == 'r':
            y0, y1, x0, x1 = int(y - 30 * s), y, int(x - 4 * s), int(x + w + 8 * s)
        else:
            y0, y1, x0, x1 = int(y - 8 * s), int(y + h + 20 * s), x, x + w
        v, sc = dg.number(wm[max(0, y0):y1, max(0, x0):x1], 14 * s, 28 * s)
        out[k] = v
    return out


def level(fr, badge, team, dg, m=None):
    x, y, w, h = badge
    m = m or masks(fr)
    mk = (m['white'] if team == 'b' else m['gold'])[y:y + h, x:x + w]
    s = fr.shape[1] / 864
    v, sc = dg.number(mk, 8 * s, 22 * s)
    return v if v is not None and 1 <= v <= 16 and sc >= 0.55 else None


def shape(name):
    # the game font's y has no descender, g does
    out = []
    for word in name.split(' '):
        c = ''
        for ch in word:
            if ch in ".-'":
                continue
            c += 'T' if ch.isupper() or ch in 'bdfhklt' or ch.isdigit() else 'D' if ch in 'gjpq' else 'S'
        out.append(c)
    return ' '.join(out)


NARROW, WIDE = set('ijl1.'), set('mwMW')


def widths(name):
    return [0.6 if ch in NARROW else 1.4 if ch in WIDE else 1.0 for ch in name.replace(' ', '') if ch not in ".-'"]


def rows(cs, hmin):
    cs = sorted(c for c in cs if c[3] >= hmin)
    grp = []
    for c in cs:
        for r in grp:
            if abs(c[1] + c[3] / 2 - np.mean([b[1] + b[3] / 2 for b in r])) < 0.6 * c[3]:
                r.append(c)
                break
        else:
            grp.append([c])
    out = []
    for r in grp:
        r = sorted(r)
        xh = np.median([c[3] for c in r])
        cur = [r[0]]
        for a, b in zip(r[:-1], r[1:]):
            if b[0] - a[0] - a[2] > 1.5 * xh:
                out.append(cur)
                cur = []
            cur.append(b)
        out.append(cur)
    out = [[c for c in r if c[3] >= 0.7 * np.median([b[3] for b in r]) and c[2] <= 1.6 * np.median([b[3] for b in r])] for r in out]
    return [r for r in out if len(r) >= 3]


def code(row):
    # glyph class from its vertical extent against the row's x-height, words split at wide gaps
    bots = np.array([c[1] + c[3] for c in row])
    base, xh = np.median(bots), np.median([c[3] for c in row])
    gaps = np.array([row[i + 1][0] - row[i][0] - row[i][2] for i in range(len(row) - 1)])
    s, ws = '', []
    for i, c in enumerate(row):
        s += 'D' if bots[i] > base + 0.2 * xh else 'T' if c[3] > 1.15 * xh else 'S'
        ws.append(c[2] / xh)
        if i < len(gaps) and gaps[i] > 1.2 * np.median(gaps) + 0.3 * xh:
            s += ' '
    return s, np.array(ws)


def edit(a, b):
    d = np.arange(len(b) + 1)
    for i, ca in enumerate(a, 1):
        nd = [i]
        for j, cb in enumerate(b, 1):
            nd.append(min(d[j] + 1, nd[j - 1] + 1, d[j - 1] + (ca != cb)))
        d = nd
    return d[-1]


class Names:
    def __init__(self, names, costs=None):
        self.names = [(n, shape(n), np.array(widths(n)), (costs or {}).get(n)) for n in names]

    def match(self, row, cost=None):
        s, ws = code(row)
        best = []
        for n, sh, wd, c in self.names:
            if cost is not None and c is not None and c != cost:
                continue
            e = edit(s, sh)
            if len(wd) == len(ws):
                e += 0.5 * np.abs(np.log(ws / ws.mean()) - np.log(wd / wd.mean())).mean()
            else:
                e += 1
            best.append((e, n))
        best.sort()
        return best[0][1], best[0][0], s, best[1][0] if len(best) > 1 else 9

    def match_gap(self, left, right, cost=None, khat=1):
        # the opponent's floating elixir drop can cover the middle letters: match both visible parts around up to three hidden
        # ones, preferring as many hidden letters as the drop is wide
        ls, rs = code(left)[0].replace(' ', '') if left else '', code(right)[0].replace(' ', '') if right else ''
        best = []
        for n, sh, _, c in self.names:
            if cost is not None and c is not None and c != cost:
                continue
            sh = sh.replace(' ', '')
            e = min(edit(ls, sh[:j]) + edit(rs, sh[j + k:]) + 0.5 * abs(k - khat) for k in range(0, 4) for j in range(0, len(sh) - k + 1))
            best.append((e, n))
        best.sort()
        return best[0][1], best[0][0], ls + '*' + rs, best[1][0] if len(best) > 1 else 9


def over(drops, row, xh):
    x0, x1 = row[0][0] - 2 * xh, row[-1][0] + row[-1][2] + 2 * xh
    top, bot = min(c[1] for c in row), max(c[1] + c[3] for c in row)
    return [d for d in drops if d[1] < bot and d[1] + d[3] > top and x0 < d[0] + d[2] / 2 < x1]


def banner(fr, cal, dg, names, m=None):
    # two centred rows of white text in the arena, the name row taller, the second ending in the level digits; the opponent's
    # elixir drop floats up through the banner, so a covered name is matched around the gap and a covered level is left unread
    m = m or masks(fr)
    wm = m['bright']
    H, W = fr.shape[:2]
    s = W / 864
    fwd = np.asarray(cal['fwd'])
    lo, hi = int((fwd @ [0, 33, 1])[1]), int((fwd @ [0, 0, 1])[1])
    sub = wm[lo:hi]
    drops = [d for d in comps(m['drop'][lo:hi], 22 * s, 70 * s, 28 * s, 80 * s) if 0.5 < d[2] / d[3] < 1.3]
    rs = rows(comps(sub, 2, 60 * s, 8 * s, 40 * s), 14 * s)
    out = []
    for r in rs:
        xh = np.median([c[3] for c in r])
        if xh < 21 * s:
            continue
        cx, bot = np.mean([c[0] + c[2] / 2 for c in r]), max(c[1] + c[3] for c in r)
        for r2 in rs:
            if r2 is r or not 3 <= len(r2) <= 7 or not -0.3 * xh < min(c[1] for c in r2) - bot < 1.2 * xh:
                continue
            if abs(np.mean([c[0] + c[2] / 2 for c in r2]) - cx) > 2.5 * xh or np.median([c[3] for c in r2]) > xh:
                continue
            dn, dl = over(drops, r, xh), over(drops, r2, xh)
            cost = None
            for d in dn + dl:
                v, sc = dg.number(m['pale'][lo + d[1]:lo + d[1] + d[3], d[0]:d[0] + d[2]], 12 * s, 30 * s)
                if v is not None and sc >= 0.6 and 1 <= v <= 9:
                    cost = v
            lv = None
            if dl:
                x1 = max(d[0] + d[2] for d in dl) + 3 * s
                tail = [c for c in r2 if c[0] > x1][-2:]
                v, sc = dg.read(sub, tail) if len(tail) == 2 else (None, 0)
                lv = int(v) if v and sc >= 0.6 and 10 <= int(v) <= 16 else None
            else:
                v, sc = dg.read(sub, r2[-2:])
                if v is None or sc < 0.6 or not 1 <= int(v) <= 16:
                    v, sc = dg.read(sub, r2[-1:])
                if v is None or sc < 0.6 or not 1 <= int(v) <= 16:
                    continue
                lv = int(v)
            if dn:
                x0, x1 = min(d[0] for d in dn) - 3 * s, max(d[0] + d[2] for d in dn) + 3 * s
                left, right = [c for c in r if c[0] + c[2] < x0], [c for c in r if c[0] > x1]
                name, err, sh, err2 = names.match_gap(left, right, cost, round((x1 - x0 - 6 * s) / (0.9 * xh)))
            else:
                name, err, sh, err2 = names.match(r, cost)
            if err > 1.2:
                continue
            out.append({'card': name, 'err': float(err), 'err2': float(err2), 'shape': sh, 'level': lv, 'u': float(cx), 'v': float(lo + bot),
                        'cost': cost, 'drop': bool(dn or dl)})
    return out
