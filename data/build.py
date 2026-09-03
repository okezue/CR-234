import json
import math
import re
import urllib.request
from datetime import date
from pathlib import Path

from data import overlay, wiki

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
CS = "https://raw.githubusercontent.com/ClashStrategic/stats/main/"
GD = "https://cache.statsroyale.com/gamedata-v5.json"
LEGACY = overlay.LEGACY

MIN = {"common": 1, "rare": 3, "epic": 6, "legendary": 9, "champion": 11}
SPEED = {"slow": 45, "medium": 60, "fast": 90, "very-fast": 120}
PLACE = {"own-side": "own", "anywhere": "any"}
HIT = {"unique": "single", "splash": "splash"}
TARGETS = {"TID_TARGETS_GROUND": ["ground"], "TID_TARGETS_AIR_AND_GROUND": ["air", "ground"], "TID_TARGETS_BUILDINGS": ["buildings"], "TID_TARGETS_NONE": []}
UNIT_CARD = {"Skeleton": "skeletons", "Barbarian": "barbarians", "Bat": "bats", "Goblin": "goblins", "SpearGoblin": "spear_goblins",
             "Spear Goblin": "spear_goblins", "Fire Spirit": "fire_spirit", "Royal Recruit": "royal_recruits"}
UNIT_FIELDS = {"hitpoints": "hitpoints", "damage": "damage", "hitspeed": "hitSpeed", "speed": "speed", "range": "range", "targets": "targets",
               "lifetime": "lifetime", "deployTime": "deployTime", "radius": "radius"}
GD_MS = {"hitSpeed": "hitSpeed", "loadTime": "loadTime", "deployTime": "deployTime", "lifeTime": "lifetime"}
GD_TILES = {"range": "range", "sightRange": "sightRange", "collisionRadius": "collisionRadius"}
GD_LEVELS = {"hitpoints": "hitpoints", "damage": "damage", "deathDamage": "deathDamage"}
# game data character names whose ClashStrategic unit key differs from snake(name)
GD_UNIT = {"Goblinstein_doctor": "doctor"}
# the statsroyale dump stops at level 15 for towers; wiki Tower Princess (4858) and Cannoneer (4164) tables imply 347, ClashStrategic's script uses 346
TOWER16 = 347
# the official battle log reports 7728 king tower hitpoints at level 16 (1,671 untouched towers in data/raw/eval/battles.csv); the wiki table says 7704
KING16 = 7728


def fetch(url, name):
    RAW.mkdir(exist_ok=True)
    p = RAW / name
    if not p.exists():
        req = urllib.request.Request(url, headers={"User-Agent": "cr234-build"})
        p.write_bytes(urllib.request.urlopen(req, timeout=120).read())
    return p


def gamedata():
    return json.loads(fetch(GD, "gamedata.json").read_text())


def tables(gd):
    # multipliers are indexed by absolute level for every rarity; characters are stored at level 1 of the common curve
    r = next(x for x in gd["items"]["rarities"] if x["name"] == "Common")
    return [100, *r["powerLevelMultiplier"]][:16], ([100, *r["supportPowerLevel"]] + [TOWER16])[:16]


def key(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower().replace(".", "")).strip("_")


def snake(name):
    return key(re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name))


def camel(k):
    return re.sub(r"-(\w)", lambda m: m.group(1).upper(), k)


def is_levels(v):
    return isinstance(v, dict) and 0 < len(v) <= 2 and set(v) <= {"level11", "level16"}


def levels(v):
    return {"level11": v.get("level11"), "level16": v.get("level16")} if is_levels(v) else v


def empty(v):
    return v is None or (is_levels(v) and not any(v.values()))


def fit(v11, v16, pct):
    # integer base at absolute level 1 that floors to the level 11 anchor (level 16 breaks ties)
    a, i = (v11, 10) if v11 is not None else (v16, 15)
    lo = max(0, math.floor(a * 100 / pct[i]) - 1)
    return min(range(lo, lo + 4), key=lambda b: (abs(b * pct[i] // 100 - a), abs(b * pct[15] // 100 - v16) if v16 is not None else 0, b))


def curve(b, lo, pct):
    return [None if L < lo else b * pct[L - 1] // 100 for L in range(1, 17)]


def expand(lv, lo, pct, keep16=True):
    # returns the 16 levels plus the level 16 anchor when it was dropped as inconsistent
    v11, v16 = lv["level11"], lv["level16"]
    if v11 is None and v16 is None:
        return None, None
    if not keep16 and v11 is not None:
        v16 = None
    if isinstance(v11, float) or isinstance(v16, float):
        b = (v11 if v11 is not None else v16 * pct[10] / pct[15]) / pct[10] * 100
        out = [None if L < lo else round(b * pct[L - 1] / 100, 3) for L in range(1, 17)]
    else:
        out = curve(fit(v11, v16, pct), lo, pct)
    if v11 is not None:
        out[10] = v11
    if v16 is None or abs(out[15] - v16) <= 1:
        out[15] = v16 if v16 is not None else out[15]
        return out, None
    return out, v16


def walk_levels(o, path, fn):
    if is_levels(o):
        return fn(levels(o), path)
    if isinstance(o, dict):
        return {k: walk_levels(v, f"{path}.{k}" if path else k, fn) for k, v in o.items()}
    if isinstance(o, list):
        return [walk_levels(v, f"{path}[{i}]", fn) for i, v in enumerate(o)]
    return o


def skills(sk):
    out = {}
    for k, v in sk.items():
        k = camel(k)
        if k == "ability":
            out[k] = {"name": v.get("name"), "cost": v.get("elixirCost"), "cooldown": v.get("cooldown"), "skills": skills(v.get("skills") or {})}
            continue
        s = {}
        for f, x in v.items():
            f = {"hitspeed": "hitSpeed", "units": "count"}.get(f, f)
            if f == "speed" and isinstance(x, str):
                x = SPEED[x]
            s[f] = levels(x)
        out[k] = s
    return out


def units(card):
    out = {}

    def scan(sk, path):
        for k, s in sk.items():
            if k == "ability":
                scan(s["skills"], f"{path}.ability.skills")
            elif isinstance(s, dict) and s.get("character"):
                u = out.setdefault(snake(s["character"]), {"name": s["character"], "card": UNIT_CARD.get(s["character"]), "count": None, "from": []})
                u["from"].append(f"{path}.{k}")
                if u["count"] is None:
                    u["count"] = s.get("count")
                for f, g in UNIT_FIELDS.items():
                    # a spawn-on-death damage is the spawned unit's own death damage (Golemite 99), not its attack
                    g = "deathDamage" if k == "spawnOnDeath" and f == "damage" else g
                    # evo and hero spawns (decoy goblins, snowman) keep their own stats on the skill, the shared unit stays the base character
                    if f in ("hitpoints", "damage") and not path.startswith("skills"):
                        continue
                    if not empty(s.get(f)) and u.get(g) is None:
                        u[g] = s[f]

    scan(card["skills"], "skills")
    if card["evo"]:
        scan(card["evo"]["skills"], "evo.skills")
    if card["hero"]:
        scan(card["hero"]["ability"]["skills"], "hero.ability.skills")
    return out


def gd_units(spell, lo, pct, tag, src):
    # spawned characters carried by the game data (Golemite, Lava Pups, Witch skeletons ...) with level 1 bases
    out = {}
    root = spell.get("summonCharacterData") or {}
    proj = spell.get("projectileData") or {}
    # a spell's projectile (Goblin Barrel) or its rolling projectile (Barbarian Barrel) carries the character it spawns
    roots = [(root, f, root.get("deathSpawnCount" if "death" in f else "spawnNumber"))
             for f in ("spawnCharacterData", "deathSpawnCharacterData", "spawnCharacter2Data", "deathSpawnCharacter2Data")]
    roots += [(proj, "spawnCharacterData", proj.get("spawnCharacterCount")), (proj.get("spawnProjectileData") or {}, "spawnCharacterData", 1)]
    # the companions of a group card (Goblin Gang's Spear Goblins, Rascal Girls, Goblinstein's doctor)
    roots += [(spell, "summonCharacterSecondData", spell.get("summonCharacterSecondCount"))]
    for root, f, n in roots:
        ch = root.get(f)
        if not ch or ch.get("hitpoints") is None:
            continue
        k = GD_UNIT.get(ch["name"], snake(ch["name"]))
        pj = ch.get("projectileData") or {}
        u = {"name": ch["name"], "card": UNIT_CARD.get(ch["name"]), "from": [f"gd.{f}"], "count": n, "targets": TARGETS.get(ch.get("tidTarget"))}
        for a, b in GD_LEVELS.items():
            v = ch.get(a) if ch.get(a) is not None else pj.get(a) if a == "damage" else None
            if isinstance(v, int):
                u[b] = curve(v, lo, pct)
        for a, b in GD_MS.items():
            if ch.get(a) is not None:
                u[b] = ch[a] / 1000
        for a, b in GD_TILES.items():
            if ch.get(a) is not None:
                u[b] = ch[a] / 1000
        if ch.get("speed") is not None:
            u["speed"] = ch["speed"]
        if pj:
            u["projectile"] = {"speed": round(pj["speed"] / 60, 3) if pj.get("speed") else None, "count": 1}
        out[k] = u
        src[f"units.{k}"] = tag
    return out


def gd_proj(card, spell, tag):
    # Arrows fall in waves; the wave count and interval are only in the game data
    if card["projectile"] and spell.get("projectileWaves"):
        card["projectile"]["waves"], card["projectile"]["waveInterval"] = spell["projectileWaves"], spell["projectileWaveInterval"] / 1000
        card["src"]["projectile.waves"] = tag
    pj = spell.get("projectileData") or {}
    roll = pj.get("spawnProjectileData") or {}
    if roll.get("speed") and "pierce" in card["skills"]:
        # a rolling spell (The Log, Barbarian Barrel) drops at the cast point and its spawned projectile rolls at this speed; the drop speed is not used
        card["skills"]["pierce"]["speed"], card["src"]["skills.pierce.speed"] = round(roll["speed"] / 60, 3), tag
    elif card["kind"] == "spell" and card["projectile"] and card["projectile"]["speed"] is None and pj.get("speed"):
        # thrown from the king tower (Goblin Barrel, Giant Snowball)
        card["projectile"]["speed"], card["src"]["projectile.speed"] = round(pj["speed"] / 60, 3), tag


def gd_summon(card, spell, tag):
    # deploy formation: the summons stand summonRadius tiles from the point and appear summonDeployDelay apart; the evo may differ (Skeleton Army)
    for f in ("summonRadius", "summonDeployDelay"):
        if spell.get(f) is not None:
            card[f], card["src"][f] = spell[f] / 1000, tag
        ev = (spell.get("evolvedSpellsData") or {}).get(f)
        if card["evo"] and ev is not None and ev != spell.get(f):
            card["evo"][f], card["src"][f"evo.{f}"] = ev / 1000, tag


def gd_death(card, spell, tag):
    # ClashStrategic's areaDamageOnDeath.radius is the dying object's collisionRadius (Golem 750, bomb buildings 450); only a
    # deathAreaEffectData radius (Ice Golem) is a blast radius, the rest is left for the wiki and legacy fills
    s = card["skills"].get("areaDamageOnDeath")
    ch = spell.get("summonCharacterData") or {}
    if not s or not ch:
        return
    ae = ch.get("deathAreaEffectData") or {}
    bomb = ch.get("deathSpawnCharacterData") or {}
    if ae.get("radius"):
        s["radius"], card["src"]["skills.areaDamageOnDeath.radius"] = ae["radius"] / 1000, tag
    elif s.get("radius") is not None and (ch.get("deathDamage") or bomb.get("source") == "buildings") \
            and not card["src"].get("skills.areaDamageOnDeath.radius", "").startswith("patch:"):
        s["radius"], card["src"]["skills.areaDamageOnDeath.radius"] = None, f"{tag} (cs value {s['radius']} is the collision radius, dropped)"
    if bomb.get("source") == "buildings" and bomb.get("deployTime"):
        s["fuse"], card["src"]["skills.areaDamageOnDeath.fuse"] = bomb["deployTime"] / 1000, tag


def gd_area(card, spell, lo, pct, tag):
    # ticking spells: building multiplier (Earthquake) and the per-target-count damage tiers of Void's laser ball
    ae = spell.get("areaEffectObjectData") or {}
    bp = (ae.get("buffData") or {}).get("buildingDamagePercent")
    if bp and card["stats"]["damage"]:
        card["stats"]["buildingDamage"] = [None if x is None else x * bp // 100 for x in card["stats"]["damage"]]
        card["src"]["stats.buildingDamage"] = f"{tag} buildingDamagePercent"
    for a in (ae.get("onStartingAction") or {}).get("subActions") or []:
        if a.get("classType") != "ActionLaserBall":
            continue
        hits = [x["spawnData"] for x in a["onDetectedUnitActionList"]]
        card["skills"]["multiTarget"] = {
            "maxUnits": a["maxUnitPerActionList"],
            "damageTiers": [curve(h["damagePerSecond"] * h["hitFrequency"] // 1000, lo, pct) for h in hits],
            "towerDamageTiers": [curve(h["crownTowerDamagePerHit"], lo, pct) for h in hits],
            "firstDelay": a["firstHitDelay"] / 1000,
        }
        card["src"]["skills.multiTarget"] = f"{tag} ActionLaserBall"


def gd_bases(spell):
    ch = spell.get("summonCharacterData") or spell.get("statCharacterData") or {}
    pj = spell.get("projectileData") or ch.get("projectileData") or {}
    return {"hitpoints": ch.get("hitpoints"), "damage": ch.get("damage") if ch.get("damage") is not None else pj.get("damage")}


def snap(card, spell, pct, tag):
    # ClashStrategic rounds some anchors up; when the game base puts the level 11 value exactly 1 off, take the game value
    for f, b in gd_bases(spell).items():
        lv = card["stats"][f]
        if not isinstance(b, int) or not is_levels(lv) or lv["level11"] is None:
            continue
        v = b * pct[10] // 100
        if 0 < abs(lv["level11"] - v) <= 1:
            card["src"][f"stats.{f}"] = f"{tag} (cs anchor {lv['level11']} rounded off the curve)"
            lv.update({"level11": v, "level16": b * pct[15] // 100})


def normalize(c, tag):
    kind = c["type"]
    src = {"*": tag}
    card = {
        "name": c["name"], "id": c["id"], "rarity": c["rarity"], "kind": kind, "cost": c.get("elixirCost"), "count": c["units"],
        "deployTime": c.get("deployTime"), "placement": PLACE.get(c.get("placement")), "flying": bool(c.get("flying")),
        "targets": sorted(c["targets"]), "speed": SPEED.get(c["speed"]) if kind == "troop" or (kind == "spell" and c["units"]) else None,
        "range": c["range"], "minRange": None, "sightRange": c["sightRange"], "collisionRadius": c["collisionRadius"], "mass": None,
        "summonRadius": None, "summonDeployDelay": None, "hitSpeed": c["hitspeed"], "loadTime": c.get("loadTime"),
        "hitType": HIT.get(c["hitType"], c["hitType"]), "radius": c.get("radius"),
        "duration": c.get("duration") if kind != "building" else None, "lifetime": c.get("duration") if kind == "building" else None,
        "projectile": {"speed": None, "count": c["projectileNumber"]} if c["projectile"] else None,
        "kamikaze": bool(c.get("kamikaze")), "arena": c["unlockArena"], "tribe": c["tribe"],
        "stats": {"hitpoints": levels(c["hitpoints"]), "damage": levels(c["damage"]), "towerDamage": levels(c.get("towerDamage") or {"level11": None})},
        "skills": skills(c["skills"]), "evo": None, "hero": None, "units": {}, "src": src,
    }
    if c.get("evolution"):
        e = c["evoStats"]
        card["evo"] = {"cycles": e["cycles"], "stats": {f: levels(e[f]) for f in ("hitpoints", "damage", "towerDamage")}, "skills": skills(e["skills"])}
    if c.get("hero"):
        card["hero"] = {"ability": skills(c["heroStats"]["skills"])["ability"], "stats": {}}
    return card


def wiki_fill(card):
    # only fields ClashStrategic lacks: projectile speed and the near edge of a min-max range
    a = wiki.attrs(wiki.page(card["name"]))
    tag = f"wiki:{card['name']}"
    ps = wiki.num(a.get("Projectile Speed"))
    if card["projectile"] and card["projectile"]["speed"] is None and ps:
        card["projectile"]["speed"], card["src"]["projectile.speed"] = round(ps / 60, 3), tag
    m = re.fullmatch(r"([\d.]+)\s*-\s*([\d.]+)", a.get("Range", ""))
    if m and card["minRange"] is None:
        card["minRange"], card["src"]["minRange"] = float(m.group(1)), tag
    dr = wiki.num(a.get("Death Damage Splash Radius"))
    if dr and "areaDamageOnDeath" in card["skills"] and card["skills"]["areaDamageOnDeath"].get("radius") is None:
        card["skills"]["areaDamageOnDeath"]["radius"], card["src"]["skills.areaDamageOnDeath.radius"] = dr, tag
    v = wiki.vars_(wiki.page(card["name"]))
    if card["kind"] == "spell" and card["duration"] and (v.get("dmg_hits") or card["hitSpeed"] or overlay.tick_interval(key(card["name"]))):
        # ticking spells: the wiki lists damage per hit (and often the hit count); ClashStrategic mixes per hit, per second and totals
        if v.get("dmg_hits"):
            card["tick"], card["src"]["tick"] = {"count": int(v["dmg_hits"])}, tag
        for f, k in (("damage", v.get("curse_dmg_11") and "curse_dmg_11" or "dmg_11"), ("towerDamage", "crown_dmg_11")):
            w = wiki.num(v.get(k))
            lv = card["stats"][f]
            if w and is_levels(lv) and lv["level11"] and abs(lv["level11"] - w) / w > 0.05:
                card["src"][f"stats.{f}"] = f"{tag} per hit (cs anchor {lv['level11']} is not per hit)"
                lv.update({"level11": int(w), "level16": None})


def king():
    w = wiki.page("King's Tower")
    hp, dmg = [None] * 16, [None] * 16
    for r in wiki.table(w, "unit-statistics-table") or []:
        L = int(wiki.num(r.get("Level")) or 0)
        if 1 <= L <= 16:
            hp[L - 1], dmg[L - 1] = int(wiki.num(r["Hitpoints"])), int(wiki.num(r["Damage"]))
    hp[15] = KING16
    a = wiki.attrs(w)
    return {
        "name": "King's Tower", "id": None, "rarity": None, "kind": "tower", "cost": None, "count": 1, "deployTime": None, "placement": None,
        "flying": False, "targets": ["air", "ground"], "speed": None, "range": wiki.num(a.get("Range")), "minRange": None, "sightRange": None,
        "collisionRadius": None, "mass": None, "summonRadius": None, "summonDeployDelay": None, "hitSpeed": wiki.num(a.get("Hit Speed")),
        "loadTime": None, "hitType": "single", "radius": None,
        "duration": None, "lifetime": None, "projectile": {"speed": None, "count": 1}, "kamikaze": False, "arena": None, "tribe": None,
        "stats": {"hitpoints": hp, "damage": dmg, "towerDamage": None}, "skills": {}, "evo": None, "hero": None, "units": {},
        "src": {"*": "wiki:King's Tower", "stats.hitpoints[15]": "royaleapi battle log (KING16)"},
    }


def set_path(card, path, value, tag):
    if path == "":
        card.clear()
        card.update(value)
        card["src"] = {"*": tag}
        return
    parts = re.findall(r"[^.\[\]]+|\[\d+\]", path)
    o = card
    for p in parts[:-1]:
        if o.get(p) is None:
            o[p] = {}
        o = o[p]
    last = parts[-1]
    if last == "[10]":
        o.clear()
        o.update({"level11": value, "level16": None})
        card["src"][path[:-4]] = tag
    else:
        o[last] = value
        card["src"][path] = tag


def apply_patches(cards, kinds):
    applied = []
    for pf in sorted((ROOT / "patches").glob("*.json")):
        tag = f"patch:{pf.stem}"
        for e in json.loads(pf.read_text()):
            k = e["card"]
            if e["path"] == "":
                # a whole-card patch may predate schema fields: absent ones are null like a normalized card
                cards.setdefault(k, {})
                kinds[k] = e["value"]["kind"]
                e["value"] = {**dict.fromkeys(next(c for c in cards.values() if c)), **e["value"]}
            set_path(cards[k], e["path"], e["value"], tag)
        applied.append(pf.stem)
    return applied


def finish(card, pct, tower, gd_spell, gd_tag):
    lo = MIN[card["rarity"]]
    src = card["src"]

    def fn(lv, path):
        arr, dropped = expand(lv, lo, pct, keep16=not tower)
        if arr is not None:
            src[f"{path}[]"] = "derived:towerMult" if tower else "derived:levelMult"
        if dropped is not None:
            src[f"{path}[15]"] = f"derived:levelMult (source level16 {dropped} is off the level curve)"
        return arr

    gd = gd_units(gd_spell, lo, pct, gd_tag, src) if gd_spell else {}
    cs = {**units(card), **card.get("units", {})}
    card["units"] = {}
    for k in sorted(set(gd) | set(cs)):
        a, b = gd.get(k, {}), cs.get(k, {})
        card["units"][k] = {**a, **{f: v for f, v in b.items() if not empty(v) and f != "from"}, "from": a.get("from", []) + b.get("from", [])}
    out = {k: walk_levels(v, k, fn) for k, v in card.items() if k != "src"}
    out["src"] = dict(sorted(src.items()))
    return out


def dumps(o):
    s = json.dumps(o, indent=1, sort_keys=True, ensure_ascii=False)
    return re.sub(r"\[\s+((?:[^\[\]{}]*?,\s+)*[^\[\]{}]*?)\s+\]", lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]", s) + "\n"


def main():
    version = json.loads(fetch(CS + "package.json", "csPackage.json").read_text())["version"]
    tag = f"cs{version}"
    raw = json.loads(fetch(CS + "data/cards.json", "csCards.json").read_text())
    gd = gamedata()
    gd_tag = f"gd:{gd['meta']['fingerprint'][:8]}"
    # the English name is shared with event variants (Royal Recruits chess boards), so the card id keys the game record
    spells = {s["id"]: s for s in gd["items"]["spells"]}
    spells.update({s["englishName"].strip(): s for s in gd["items"]["spells"] if s["englishName"].strip() not in spells})
    pct, tower_pct = tables(gd)
    cards, kinds = {}, {}
    for c in raw["cards"] + raw["towerCards"]:
        k = key(c["name"])
        cards[k], kinds[k] = normalize(c, tag), c["type"]
        sp = spells.get(c["id"]) or spells.get(c["name"])
        if sp:
            snap(cards[k], sp, tower_pct if c["type"] == "tower" else pct, gd_tag)
    patches = apply_patches(cards, kinds)
    out = {"cards": {}, "towers": {"king_tower": king()}}
    for k, c in cards.items():
        sp = spells.get(c["id"]) or spells.get(c["name"])
        if sp:
            gd_death(c, sp, gd_tag)
            gd_proj(c, sp, gd_tag)
            gd_summon(c, sp, gd_tag)
        wiki_fill(c)
        tower = kinds[k] == "tower"
        p = tower_pct if tower else pct
        rec = finish(c, p, tower, sp, gd_tag)
        if sp:
            gd_area(rec, sp, MIN[c["rarity"]], p, gd_tag)
        # legacy mechanics fill only what is still null; level-11 legacy anchors are expanded through the same curve
        overlay.apply(rec, k, lambda v, lo=MIN[c["rarity"]], p=p: curve(fit(v, None, p), lo, p))
        rec["src"] = dict(sorted(rec["src"].items()))
        out["towers" if tower else "cards"][k] = rec
    out["meta"] = {
        "built": date.today().isoformat(),
        "sources": [
            {"tag": tag, "name": "ClashStrategic/stats", "url": CS + "data/cards.json", "version": version, "license": "Apache-2.0"},
            {"tag": gd_tag, "name": "statsroyale game data dump (ClashStrategic's upstream)", "url": GD, "fingerprint": gd["meta"]["fingerprint"],
             "fields": ["meta.levelMult", "meta.towerMult", "units.* spawned characters", "stats.buildingDamage", "skills.multiTarget",
                        "projectile.waves", "skills.areaDamageOnDeath.radius/fuse", "summonRadius", "summonDeployDelay"]},
            {"tag": "wiki:<page>", "name": "Clash Royale Fandom wiki, MediaWiki API wikitext", "url": wiki.API,
             "fields": ["projectile.speed", "minRange", "towers.king_tower", "towerMult[15]", "skills.areaDamageOnDeath.radius",
                        "tick.count and per-hit damage anchors of ticking spells"]},
            {"tag": "legacy:<field>", "name": "data/legacy (unverified jan2026 snapshot), fills only null mechanics; level-11 anchors expanded by levelMult",
             "fields": ["mass", "collisionRadius", "sightRange", "loadTime", "projectile.speed", "skills.*", "evo.skills.*", "hero.ability.*", "units.*",
                        "hitSpeed (spell tick)", "stats.buildingDamage (Earthquake)"]},
        ] + [{"tag": f"patch:{p}", "name": f"data/patches/{p}.json"} for p in patches],
        "levelMult": [p / 100 for p in pct],
        "towerMult": [p / 100 for p in tower_pct],
        "minLevel": MIN,
        "levelRule": "stat[L] = floor(base * levelMult[L-1]); base is the integer level 1 value of the game data (rarities.powerLevelMultiplier, the same "
        "table for every rarity, indexed by absolute level); rarity only sets minLevel, levels below it are null; towers use towerMult (supportPowerLevel)",
        "src": "src['*'] is the default provenance; 'path[]' covers the levels derived from the anchors; anchors keep the source value verbatim",
        "units": {"speed": "tiles per minute (slow 45, medium 60, fast 90, very fast 120)", "projectile.speed": "tiles per second (game speed / 60)",
                  "range": "tiles", "time": "seconds", "multipliers": "percent",
                  "summonRadius": "tiles from the deploy point to each summon of a multi-unit card (null: they appear touching)",
                  "summonDeployDelay": "seconds between consecutive summons (null: all at once)"},
        "hitType": {"single": "one target per hit (source: unique)", "splash": "area hit"},
        "abilities": "champion abilities live in skills.ability, hero abilities in hero.ability; cooldown null means single use",
        "legacySkills": {
            "group": "characters deployed alongside the card's own unit (units[k].count each); the card stats are the remaining count, "
                     "leader names its unit overrides",
            "secondaryAttack": "independent attached attacker (rocket launcher, rider, backpack goblins); character names the unit",
            "areaDamageOnSpawn": "damage and radius dealt where the unit appears",
            "produceElixir": "interval and amount", "teleport": "distance moved backwards on cast",
            "rampingHitSpeed": "hitSpeedTiers stepped every hitsPerTier hits while standing still", "recoil": "self knockback per shot",
            "transform": "mode switch below hpPercent of hitpoints: speed, range, targets and lifetime of the new form (Goblin Demolisher's rocket); "
                         "building: the new form is a rooted building whose remaining hitpoints drain over the lifetime (Cannon Cart)",
            "charging": "loadFirstHit: the unit comes in unloaded and charges the whole hit speed before its first hit, a stun empties the charge (Sparky); "
                        "every other unit deploys loaded and loads while walking, so its first hit takes hit speed less load time",
            "line": "spacing: the summons stand on a horizontal line this many tiles apart instead of the summonRadius circle (Royal Recruits)",
            "immunity": "knockback: the troop ignores pushback regardless of mass (Prince, Dark Prince)",
            "meleeSwitch": "damage and range of a melee attack used instead of the shot while the target is a ground unit within reach (Elite Musketeers)",
            "spawn.minRadius": "a periodic spell spawn rises between this distance and the spell radius from the centre (Graveyard)",
            "burrow": "underground travel from the own King Tower at speed (tiles/min) for at least the deploy time; "
                      "resurfacePercent/resurfaceCount for the evo drill",
            "params": "charge.range, dash.chargeTime/radius/speed/count/towerDamage, spawnOnDeath.count/hpPercent, "
                      "periodicSpawn.firstDelay/hpPercent/lifetime, spawn.interval/firstDelay/kind, pull.strength/damage/distance, pushback.cycle/damage, "
                      "heal.radius/overHeal/perKillTiers, poison.stackHits, snipe.minRange/maxRange/cooldown/towerDamagePercent, "
                      "slow.count/radius/strikes/damage/everyHits, stun.targets/radius/damage, pierce.bounceDamagePercent/returnTime, "
                      "jump.damage/radius/duration, burrow.speed/resurfacePercent/resurfaceCount, reflect.damageMultiplier/cooldown, "
                      "rampingDamage.retainTime/finalStageTime, stack.interval/healPercent/firstDelay/maxInterval, "
                      "boost.flying/count/hitsPerBonus/hitSpeed/towerDamagePercent, invincible.minHitpoints, areaDamageOnSpawn.towerDamage (0 spares towers), "
                      "ability.uses/castTime, evo.count, projectile.waves, volley.reloadTime/towerDamage; spells tick every hitSpeed for duration",
        },
    }
    (ROOT / "cards.json").write_text(dumps(out))
    aliases = {
        "api": {k.replace("_", "-"): k for k in sorted(cards)},
        "names": {c["name"]: k for k, c in sorted(cards.items())},
        "suffix": {"-ev1": "evo", "-hero": "hero"},
        "prefix": {"ability-": "ability"},
        "ignore": ["_invalid"],
    }
    (ROOT / "aliases.json").write_text(dumps(aliases))
    print(f"{len(out['cards'])} cards, {len(out['towers'])} towers, patches {patches}, sources {tag} {gd_tag}")


if __name__ == "__main__":
    main()
