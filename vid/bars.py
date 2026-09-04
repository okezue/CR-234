import cv2
import numpy as np
from vid.cal import comps, masks, to_tile

# unit bars hide at full HP, the level badge stays, so the badge is the marker and a missing bar means hp 1
FEET = 1.3  # tiles from badge centre down to the unit's feet, measured once on standing units
K = np.ones((5, 5), np.uint8)


def marker(m, mk, team, c, s):
    x, y, w, h = c
    fill = m['red' if team == 'r' else 'blue'][y:y + h, x:x + w]
    dark = m['rdark' if team == 'r' else 'bdark'][y:y + h, x:x + w]
    # the badge ring's right edge is the last tall column, the bar to its right is about half the height
    ext = mk[y:y + h, x:x + min(w, int(1.3 * h))].sum(0) >= 0.6 * h
    bw = int(np.where(ext)[0][-1]) + 1 if ext.any() else min(w, h)
    r0, r1 = int(h * 0.3), int(h * 0.7) + 1
    cols = (fill | dark)[r0:r1, bw:].mean(0) > 0.3
    hp, bar = 1.0, None
    if cols.sum() >= 5 * s:
        hp = float((fill[r0:r1, bw:][:, cols].mean(0) > 0.3).mean())
        bar = (x + bw, y + r0, int(cols.sum()), r1 - r0)
    return {'team': team, 'u': x + bw / 2, 'v': y + h / 2, 'hp': hp, 'badge': (x, y, bw, h), 'bar': bar}


def detect(fr, cal, m=None):
    m = m or masks(fr)
    H, W = fr.shape[:2]
    s = W / 864
    fwd = np.asarray(cal['fwd'])
    vmin, vmax = (fwd @ [0, 33, 1])[1], (fwd @ [0, 1, 1])[1]
    # enemy tower labels move with the opponent's tower skin, so their exclusion zone is taller
    tw = [(a, (40 if k[0] == 'r' else 15) * s) for k, a in cal['anchors'].items() if k != 'lines']
    digit = m['white'] | m['gold']
    out = []
    for team, key in (('r', 'red'), ('b', 'blue')):
        mk = cv2.morphologyEx((m[key] | m[key[0] + 'dark'] | m[key[0] + 'badge']).astype(np.uint8), cv2.MORPH_CLOSE, K)
        # tower bars are wider and flatter (no badge in the mask) than a unit's badge, and sit at the calibrated anchors
        for c in comps(mk, 16 * s, 115 * s, 20 * s, 36 * s):
            x, y, w, h = c
            u, v = x + h / 2, y + h / 2
            if not vmin < v < vmax or w < 0.7 * h or (mk[y:y + h, x:x + h] | digit[y:y + h, x:x + h]).mean() < 0.5:
                continue
            if any(-20 * s < u - a[0] < 130 * s and abs(v - a[1]) < dv for a, dv in tw):
                continue
            d = marker(m, mk, team, c, s)
            x_, y_ = to_tile(cal, d['u'], d['v'])
            d['x'], d['y'] = float(x_), float(y_ - FEET)
            out.append(d)
    return out
