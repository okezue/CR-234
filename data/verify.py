import collections
import json
import re
from pathlib import Path

from data import wiki
from data.build import LEGACY, MIN, gamedata, tables

ROOT = Path(__file__).resolve().parent
TOP40 = """hog_rider knight skeletons ice_spirit the_log fireball musketeer cannon ice_golem valkyrie mega_knight pekka electro_wizard goblin_barrel
princess dart_goblin rocket tesla arrows zap giant wizard witch baby_dragon mini_pekka archers goblins spear_goblins bats minions mega_minion
balloon golem lava_hound royal_giant x_bow mortar miner poison tornado""".split()
PCT, TOWER_PCT = tables(gamedata())


def var(v, *names):
    for n in names:
        if n in v:
            return wiki.num(v[n])
    for k in v:
        if k.endswith(names[0]):
            return wiki.num(v[k])
    return None


def wiki_stats(card):
    w = wiki.page(card["name"])
    v, a = wiki.vars_(w), wiki.attrs(w)
    hp, dmg = var(v, "hp_11", "hp_base"), var(v, "dmg_11", "dmg_base")
    hits = wiki.num(v.get("dmg_hits")) or 1
    rng = a.get("Range", "")
    m = re.fullmatch(r"([\d.]+)\s*-\s*([\d.]+)", rng)
    return {
        "hitpoints[10]": hp, "hitpoints[13]": round(hp * 1.1**3) if hp else None,
        "damage[10]": dmg, "damage[13]": round(dmg * 1.1**3) if dmg else None, "hits": hits,
        "hitSpeed": wiki.num(a.get("Hit Speed")) if "Hit Speed" in a else wiki.num(v.get("atk_speed")),
        "range": float(m.group(2)) if m else wiki.paren(rng) if rng else None,
        "speed": wiki.paren(a.get("Speed")) if "Speed" in a else None,
        "count": wiki.num(a.get("Count")) if "Count" in a else None, "cost": wiki.num(a.get("Cost")),
    }


def ours(card):
    s = card["stats"]
    g = lambda f, i: (s.get(f) or [None] * 16)[i]
    return {"hitpoints[10]": g("hitpoints", 10), "hitpoints[13]": g("hitpoints", 13), "damage[10]": g("damage", 10), "damage[13]": g("damage", 13),
            "hitSpeed": card["hitSpeed"], "range": card["range"], "speed": card["speed"], "count": card["count"], "cost": card["cost"]}


def compare(cards):
    rows = []
    for k in TOP40:
        c = cards[k]
        w, o = wiki_stats(c), ours(c)
        for f, wv in w.items():
            if f == "hits" or wv is None or o[f] is None:
                continue
            ov = o[f]
            cands = [wv] + ([wv * w["hits"]] if f.startswith("damage") and w["hits"] > 1 else [])
            tol = 1 if "[" in f else 0.01
            if min(abs(ov - x) for x in cands) <= tol:
                continue
            note = ""
            if f.endswith("[13]"):
                note = "wiki shows round(L11 * 1.1^3)"
            elif w["hits"] > 1 and f.startswith("damage"):
                note = f"wiki is per hit, x{int(w['hits'])} hits"
            rows.append((k, f, ov, wv, note))
    l14 = sum(f.endswith("[13]") for _, f, *_ in rows)
    print(f"top-40 wiki cross-check: {len(rows)} disagreements over {len(TOP40)} cards, {l14} of them level 14 cells where the wiki "
          f"extrapolates round(L11 * 1.1^3) instead of the game table (3.39 / 2.56)")
    print(f"{'card':<16}{'field':<16}{'ours':>10}{'wiki':>10}  note")
    for k, f, ov, wv, note in rows:
        print(f"{k:<16}{f:<16}{ov:>10g}{wv:>10g}  {note}")
    return rows


def gd_check(cards):
    # ClashStrategic level 11 anchors against floor(base * 2.56) from the game data bases they were derived from
    spells = {s["englishName"].strip(): s for s in gamedata()["items"]["spells"]}
    rows = []
    for k, c in cards.items():
        s = spells.get(c["name"])
        if not s:
            continue
        ch = s.get("summonCharacterData") or s.get("statCharacterData") or {}
        pj = s.get("projectileData") or ch.get("projectileData") or {}
        pct = TOWER_PCT if c["kind"] == "tower" else PCT
        for f, base in (("hitpoints", ch.get("hitpoints")), ("damage", ch.get("damage") if ch.get("damage") is not None else pj.get("damage"))):
            arr = c["stats"].get(f)
            if base is None or not arr or arr[10] is None:
                continue
            exp = base * pct[10] // 100
            if exp != arr[10]:
                rows.append((k, f, arr[10], exp, base, c["src"].get(f"stats.{f}", c["src"]["*"])))
    print(f"\ncards.json level 11 vs game data base x levelMult[10] (towerMult for towers): {len(rows)} differences")
    print(f"{'card':<20}{'field':<11}{'ours':>7}{'game':>7}{'base':>6}  provenance of ours")
    for k, f, ov, exp, base, src in rows:
        print(f"{k:<20}{f:<11}{ov:>7}{exp:>7}{base:>6}  {src}")


def legacy_tables():
    out = []

    def rec(o, path, card):
        if isinstance(o, dict):
            lv = {int(k): v for k, v in o.items() if k.isdigit()}
            if lv and all(isinstance(v, dict) for v in lv.values()):
                stats = collections.defaultdict(dict)
                for L, st in lv.items():
                    for s, x in st.items():
                        if isinstance(x, (int, float)) and not isinstance(x, bool):
                            stats[s][L] = x
                out.extend((card, f"{path}.{s}", m) for s, m in stats.items())
                return
            if lv and all(isinstance(v, (int, float)) for v in lv.values()):
                out.append((card, path, lv))
                return
            for k, v in o.items():
                rec(v, f"{path}.{k}" if path else k, card)
        elif isinstance(o, list) and o and all(isinstance(x, dict) and "level" in x for x in o):
            stats = collections.defaultdict(dict)
            for st in o:
                for s, x in st.items():
                    if s != "level" and isinstance(x, (int, float)) and not isinstance(x, bool):
                        stats[s][int(st["level"])] = x
            out.extend((card, f"{path}.{s}", m) for s, m in stats.items())

    for p in sorted(LEGACY.glob("*.json")):
        rec(json.loads(p.read_text()), "", p.stem)
    return [(c, s, m) for c, s, m in out if len(m) >= 3 and 1 <= min(m) and max(m) <= 16 and max(m.values()) >= 50]


def fit_table(m, idx):
    best = None
    for L, v in m.items():
        if idx(L) is None:
            continue
        b0 = int(v * 100 / PCT[idx(L)])
        for b in range(max(0, b0 - 1), b0 + 2):
            err = [abs(b * PCT[idx(x)] // 100 - y) for x, y in m.items() if idx(x) is not None]
            if err and (best is None or (sum(e == 0 for e in err), -max(err)) > best[0]):
                best = ((sum(e == 0 for e in err), -max(err)), b, err)
    return best


def anchor_report():
    # every ClashStrategic level 11 / level 16 pair should be floor(b * 2.56) / floor(b * 4.09) for one integer b
    raw = json.loads((ROOT / "raw" / "csCards.json").read_text())
    hits = collections.Counter()

    def rec(o, mn):
        if isinstance(o, dict):
            if set(o) == {"level11", "level16"}:
                a, b = o["level11"], o["level16"]
                if isinstance(a, int) and isinstance(b, int) and a > 0:
                    for name, i, j in (("absolute", 10, 15), ("relative", 11 - mn, 16 - mn)):
                        ok = any(x * PCT[i] // 100 == a and x * PCT[j] // 100 == b for x in range(max(0, a * 100 // PCT[i] - 2), a * 100 // PCT[i] + 3))
                        hits[name, ok] += 1
            else:
                for v in o.values():
                    rec(v, mn)
        elif isinstance(o, list):
            for v in o:
                rec(v, mn)

    for c in raw["cards"]:
        if c["rarity"] in MIN:
            rec(c, MIN[c["rarity"]])
    n = sum(v for (name, _), v in hits.items() if name == "absolute")
    print(f"\nClashStrategic level 11/16 pairs reproduced exactly by one integer base ({n} pairs):")
    for name in ("absolute", "relative"):
        print(f"  multiplier index by {name} level: {hits[name, True] / n:6.1%}")


def level_report():
    meta = {p.stem: json.loads(p.read_text()) for p in LEGACY.glob("*.json")}
    rar = {k: d.get("rarity", "").lower() for k, d in meta.items() if "Tower" not in d.get("type", "")}
    tabs = legacy_tables()
    print(f"\nlevelMult fit against {len(tabs)} legacy per-level tables (data/legacy, unverified snapshot, towers excluded)")
    print("levelMult =", [p / 100 for p in PCT], "rule: stat[L] = floor(base * levelMult[L-1]), base integer at absolute level 1")
    worst = []
    for name, idx_of in (("absolute level (used)", lambda mn: lambda L: L - 1 if 0 <= L - 1 < 16 else None),
                         ("relative to rarity minimum", lambda mn: lambda L: L - mn if 0 <= L - mn < 16 else None)):
        cells = exact = within = 0
        for card, stat, m in tabs:
            mn = MIN.get(rar.get(card))
            r = fit_table(m, idx_of(mn)) if mn else None
            if r is None:
                continue
            cells += len(r[2])
            exact += r[0][0]
            within += sum(e <= 1 for e in r[2])
            if name.startswith("absolute"):
                worst.append((max(r[2]), card, stat, r[1]))
        print(f"  {name:<28} cells {cells:5d}  exact {exact / cells:6.1%}  within +-1 {within / cells:6.1%}")
    worst.sort(reverse=True)
    print("  worst legacy tables under the absolute rule (max abs error, card, stat, fitted base):")
    for e, card, stat, b in worst[:8]:
        print(f"    {e:6d}  {card:<20} {stat:<40} base {b}")


def main():
    d = json.loads((ROOT / "cards.json").read_text())
    compare(d["cards"])
    gd_check({**d["cards"], **d["towers"]})
    anchor_report()
    level_report()


if __name__ == "__main__":
    main()
