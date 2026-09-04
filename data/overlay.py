import json
import re
from pathlib import Path

LEGACY = Path(__file__).resolve().parent / "legacy"
# legacy sub-unit names whose ClashStrategic unit key differs from snake(name)
ALIAS = {"Lava Pup": "lava_pups", "Guardienne": "protector", "Elixir Golemite": "elixir_golem2", "Soul Skeleton (cloned)": "skeleton",
         "Converted Goblin": "goblin", "Spear Goblin": "spear_goblin"}
UNIT_CARD = {"skeleton": "skeletons", "barbarian": "barbarians", "bat": "bats", "goblin": "goblins", "spear_goblin": "spear_goblins",
             "fire_spirit": "fire_spirit", "royal_recruit": "royal_recruits"}
STAT_KEYS = {"hitpoints": ("hitpoints", "hp"), "damage": ("damage", "area_damage", "damage_per_bolt", "ram_damage", "damage_per_tick")}
SPEED = {"slow": 45, "medium": 60, "fast": 90, "very fast": 120, "very-fast": 120}


def load(key):
    p = LEGACY / f"{key}.json"
    return json.loads(p.read_text()) if p.exists() else None


def snake(name):
    return re.sub(r"[^a-z0-9]+", "_", re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", name).lower()).strip("_")


def g(o, path, default=None):
    for p in path.split("."):
        if not isinstance(o, dict) or p not in o:
            return default
        o = o[p]
    return default if o is None else o


def num(v):
    # legacy strings like "0.5s", "360%", "1.8 tiles", "Medium (60)"
    if isinstance(v, (int, float)):
        return v
    m = re.search(r"-?\d+(?:\.\d+)?", str(v or ""))
    return float(m.group(0)) if m else None


def speed(v):
    if isinstance(v, dict):
        v = v.get("value", v.get("speed_value"))
    if isinstance(v, str):
        return SPEED.get(v.lower(), num(v))
    return v


def tps(v):
    # legacy dash, jump and projectile speeds are the game values: units per tick, tiles/s = value / 50 (data/build.py GD_TPS)
    return round(v / 50, 3) if isinstance(v, (int, float)) else None


def load_time(hs, first_hit):
    # the legacy first hit speed is hit speed less load time (the wiki's First Hit Speed); the schema keeps the game's load time
    hs, fh = num(hs), num(first_hit)
    return round(hs - fh, 3) if hs is not None and fh is not None and 0 < fh < hs else None


def targets(v):
    if isinstance(v, str):
        s = v.lower().replace("_and_", " & ").replace("air_and_ground", "air & ground")
        return sorted(t for t in ("air", "ground", "buildings") if t in s)
    return sorted(t.lower() for t in v) if isinstance(v, list) else None


def level11(d):
    # the three legacy schemas: stats_by_level{"11"}, levels{"11"}, levels{stats:[{level:11}]}
    sb = d.get("stats_by_level") or d.get("levels") or {}
    if isinstance(sb, dict) and "stats" in sb:
        return next((s for s in sb["stats"] if s.get("level") == 11), {})
    return sb.get("11", {}) if isinstance(sb, dict) else {}


def tick_interval(key):
    d = load(key) or {}
    sa = d.get("spell_attributes", {})
    return sa.get("tick_interval_sec", sa.get("strike_interval_sec", num(d.get("tick_interval", d.get("hit_speed")))))


def empty(v):
    return v is None or v == [] or v == {} or (isinstance(v, list) and all(x is None for x in v))


class Overlay:
    def __init__(self, card, key, scale):
        self.c, self.key, self.scale = card, key, scale
        self.d = load(key) or {}
        self.sb = level11(self.d)
        self.src = card["src"]

    def put(self, path, value, field):
        # a patched null is authoritative (single-use abilities), the legacy fill must not restore it
        if value is None or (isinstance(value, list) and not value) or self.src.get(path, "").startswith("patch:"):
            return False
        parts = path.split(".")
        o = self.c
        for p in parts[:-1]:
            if o.get(p) is None:
                o[p] = {}
            o = o[p]
        if not empty(o.get(parts[-1])):
            return False
        o[parts[-1]] = value
        self.src[path] = f"legacy:{field}"
        return True

    def arr(self, path, v11, field):
        return self.put(path, self.scale(int(v11)) if v11 is not None else None, field)

    def stat(self, *keys):
        return next((self.sb[k] for k in keys if isinstance(self.sb.get(k), (int, float))), None)

    def skill(self, scope, name, field, **params):
        # scope is "skills", "evo.skills", "hero.ability.skills" or "skills.ability.skills"; fills each given param
        for k, v in params.items():
            self.put(f"{scope}.{name}.{k}", v, field)

    def unit(self, name, u, field, card=None, count=None, stats=True):
        k = ALIAS.get(name, snake(name))
        card = card or UNIT_CARD.get(k)
        base = f"units.{k}"
        if self.c["units"].get(k) is None:
            self.c["units"][k] = {"name": name, "card": card, "count": None, "from": [f"legacy:{field}"]}
            self.src[base] = f"legacy:{field}"
        elif f"legacy:{field}" not in self.c["units"][k]["from"]:
            self.c["units"][k]["from"].append(f"legacy:{field}")
        self.put(f"{base}.count", count if count is not None else u.get("count"), field)
        atk = u.get("attack", {})
        self.put(f"{base}.hitSpeed", atk.get("hit_speed_sec", u.get("hit_speed_sec")), field)
        self.put(f"{base}.loadTime", load_time(g(self.c, f"{base}.hitSpeed"), atk.get("first_hit_speed_sec", u.get("first_hit_speed_sec"))), field)
        self.put(f"{base}.range", g(atk, "range.tiles", u.get("range_tiles")), field)
        self.put(f"{base}.speed", speed(g(u, "movement.speed", u.get("movement_speed"))), field)
        self.put(f"{base}.targets", targets(atk.get("targets", u.get("targets"))), field)
        self.put(f"{base}.deployTime", u.get("deploy_time_sec"), field)
        self.put(f"{base}.lifetime", u.get("lifetime_sec"), field)
        if g(u, "movement.transport", u.get("transport")) == "Air":
            self.put(f"{base}.flying", True, field)
        for a, b in (("mass", "mass"), ("sightRange", "sight_range_tiles"), ("collisionRadius", "collision_radius_tiles")):
            self.put(f"{base}.{a}", g(u, f"hidden_stats.{b}"), field)
        if atk.get("projectile_speed"):
            self.put(f"{base}.projectile", {"count": 1, "speed": tps(num(atk["projectile_speed"]))}, field)
        if card is None and stats:
            # a referenced card is a better stat source than the legacy snapshot
            lv = level11(u)
            pfx = [snake(name) + "_", name.lower().split()[-1] + "_", "spawned_" + snake(name) + "_"]
            for f, keys in STAT_KEYS.items():
                v = next((lv[a] for a in keys if isinstance(lv.get(a), (int, float))), None)
                if v is None:
                    v = next((self.sb[p + a] for p in pfx for a in keys if isinstance(self.sb.get(p + a), (int, float))), None)
                self.arr(f"{base}.{f}", v, field)
        ds = g(u, "mechanics.death_spawn")
        if ds:
            self.skill(f"{base}.skills", "spawnOnDeath", field, character=ds.get("unit"), count=ds.get("count"))
        dd = g(u, "mechanics.death_damage")
        if dd and dd.get("enabled", True):
            self.skill(f"{base}.skills", "areaDamageOnDeath", field, radius=dd.get("splash_radius_tiles"))
            if empty(self.c["units"][k].get("deathDamage")):
                self.arr(f"{base}.skills.areaDamageOnDeath.damage", level11(u).get("death_damage"), field)
        return k

    def referenced(self):
        keys = set(self.c["units"])

        def scan(sk):
            for k, s in (sk or {}).items():
                if k == "ability":
                    scan(s.get("skills"))
                elif isinstance(s, dict):
                    for ch in [s.get("character")] + list(s.get("characters") or []):
                        if isinstance(ch, str):
                            keys.add(ALIAS.get(ch, snake(ch)))
        scan(self.c["skills"])
        scan(g(self.c, "evo.skills"))
        scan(g(self.c, "hero.ability.skills"))
        for u in self.c["units"].values():
            scan(u.get("skills"))
        return keys

    def run(self):
        if not self.d:
            return
        c, d = self.c, self.d
        hs = d.get("hidden_stats", {})
        for a, b in (("mass", "mass"), ("collisionRadius", "collision_radius_tiles"), ("sightRange", "sight_range_tiles")):
            if isinstance(hs.get(b), (int, float)):
                self.put(a, hs[b], f"hidden_stats.{b}")
        atk = d.get("attack", {})
        self.put("loadTime", load_time(c["hitSpeed"], atk.get("first_hit_speed_sec", d.get("first_hit_speed_sec"))), "attack.first_hit_speed_sec")
        self.put("loadTime", load_time(c["hitSpeed"], d.get("first_hit")), "first_hit")
        self.put("deployTime", d.get("deploy_time_sec", g(d, "building_attributes.deploy_time_sec")), "deploy_time_sec")
        if c["kind"] == "building":
            self.put("lifetime", g(d, "building_attributes.lifetime_sec", g(d, "building_stats.lifetime_sec", d.get("lifetime_sec"))), "lifetime_sec")
        self.put("minRange", g(atk, "range.tiles_min"), "attack.range.tiles_min")
        ps = atk.get("projectile_speed", d.get("projectile_speed", g(d, "projectile.projectile_speed_units", g(d, "spell_attributes.projectile_speed"))))
        if c.get("projectile") and num(ps):
            self.put("projectile.speed", tps(num(ps)), "projectile_speed")
        self.spells()
        self.troop()
        self.champion()
        self.hero()
        self.evo()
        self.tower()

    def troop(self):
        c, d, sb, atk = self.c, self.d, self.sb, self.d.get("attack", {})
        m = d.get("mechanics", {})
        ch = m.get("charge") or g(d, "composite_attack.ram.charge") or {}
        if ch:
            rng = ch.get("charge_distance_tiles", ch.get("charge_range_tiles", ch.get("trigger_distance_tiles")))
            self.skill("skills", "charge", "mechanics.charge", range=rng)
            v = self.stat("charge_damage", "ram_charge_damage")
            mult = ch.get("charge_damage_multiplier", ch.get("damage_multiplier"))
            dmg = c["stats"]["damage"]
            if v is not None:
                self.arr("skills.charge.damage", v, "charge_damage")
            elif mult and dmg:
                self.put("skills.charge.damage", [None if x is None else int(x * mult) for x in dmg], "mechanics.charge.charge_damage_multiplier")
        da = m.get("dash")
        if da:
            self.skill("skills", "dash", "mechanics.dash", chargeTime=da.get("dash_charge_time_sec"), minRange=g(da, "trigger_target_band_tiles.min"),
                       maxRange=g(da, "trigger_target_band_tiles.max"), speed=tps(da.get("dash_speed_value")))
            self.arr("skills.dash.damage", self.stat("dash_damage"), "dash_damage")
        jp = m.get("jump")
        if jp:
            self.skill("skills", "dash", "mechanics.jump", radius=jp.get("splash_radius_tiles"), minRange=g(jp, "trigger_distance_tiles.min"),
                       maxRange=g(jp, "trigger_distance_tiles.max"), speed=tps(jp.get("jump_speed")))
        ds = m.get("death_spawn") or g(m, "spawns_on_arrival_or_death") or {}
        cnt = ds.get("count", d.get("spawns", {}).get("death_spawn_count", atk.get("death_drop", {}).get("spawns_skeletons")))
        if cnt is None and g(d, "spawning.spawn_on_death.enabled"):
            cnt = g(d, "spawning.spawn_on_death.count")
        unit = ds.get("unit", g(d, "spawns.unit", g(d, "spawning.spawns_unit", "Skeleton" if atk.get("death_drop") else None)))
        if cnt:
            self.skill("skills", "spawnOnDeath", "death_spawn", character=unit, count=cnt)
        dd = m.get("death_damage") or m.get("death_bomb") or atk.get("death_drop") or g(m, "kamikaze_mode.death_damage") or g(d, "death_mechanic.explosion")
        dv = self.stat(g(dd or {}, "damage_key", "death_damage"), "death_damage")
        if dd and dv is not None:
            self.skill("skills", "areaDamageOnDeath", "death_damage", radius=dd.get("splash_radius_tiles", dd.get("radius_tiles")))
            self.arr("skills.areaDamageOnDeath.damage", dv, "death_damage")
        if g(d, "death_mechanic.rage"):
            r = d["death_mechanic"]["rage"]
            self.skill("skills", "boost", "death_mechanic.rage", duration=r.get("duration_sec"), radius=r.get("radius_tiles"),
                       speedMultiplier=r.get("boost_percent"), hitSpeedMultiplier=r.get("boost_percent"))
        sp = d.get("spawns") or m.get("spawn") or m.get("reactive_spawn") or d.get("spawning") or {}
        unit = sp.get("unit", sp.get("spawned_unit", sp.get("spawns_unit")))
        if c["kind"] != "spell" and isinstance(unit, str) and (sp.get("spawn_interval_sec") or sp.get("spawn_speed_sec")):
            self.skill("skills", "periodicSpawn", "spawn", character=unit,
                       count=sp.get("spawn_count_per_interval", sp.get("count_per_wave", sp.get("count_per_spawn", sp.get("spawn_count_per_spawn")))),
                       pauseTime=sp.get("spawn_interval_sec", sp.get("spawn_speed_sec")), firstDelay=sp.get("spawn_first_delay_sec"))
        sz = d.get("spawn_zap") or (m.get("spawn_damage") if g(m, "spawn_damage.enabled") else None)
        if sz:
            self.skill("skills", "areaDamageOnSpawn", "spawn_damage", radius=sz.get("radius_tiles", sz.get("splash_radius_tiles")))
            self.arr("skills.areaDamageOnSpawn.damage", self.stat(sz.get("damage_key", "spawn_damage"), "spawn_zap_damage", "spawn_damage"), "spawn_damage")
            self.skill("skills", "stun", "spawn_zap", duration=sz.get("stun_duration_sec"))
        self.skill("skills", "stun", "attack.stun_duration_sec", duration=atk.get("stun_duration_sec"), targets=atk.get("bolts"))
        sl = m.get("slow_effect")
        if sl:
            self.skill("skills", "slow", "mechanics.slow_effect", duration=sl.get("duration_sec"), speedMultiplier=-sl["speed_reduction_percent"])
        zp = m.get("zap_pack")
        if zp:
            self.skill("skills", "reflect", "mechanics.zap_pack", radius=zp.get("reflect_range_tiles"))
            self.arr("skills.reflect.damage", self.stat("zap_damage"), "zap_damage")
            self.skill("skills", "stun", "mechanics.zap_pack", duration=zp.get("stun_duration_sec"))
        hl = m.get("healing") or m.get("heal_on_attack")
        if hl:
            self.skill("skills", "heal", "mechanics.healing", radius=hl.get("radius_tiles", hl.get("heal_radius_tiles")))
            self.arr("skills.heal.perAttack", self.stat("healing_per_second", "heal_per_pulse"), "healing")
        self.skill("skills", "recoil", "attack.recoil_tiles", distance=atk.get("recoil_tiles"))
        combo = g(atk, "mechanics.combo")
        if combo:
            self.skill("skills", "pushback", "attack.mechanics.combo", distance=g(combo, "combo_knockback.knockback_distance_tiles"),
                       cycle=combo.get("hits_per_cycle"))
        st = atk.get("hit_speed_sec_by_stage")
        if st:
            self.skill("skills", "rampingHitSpeed", "attack.hit_speed_sec_by_stage", hitSpeedTiers=[st[k] for k in sorted(st)],
                       hitsPerTier=g(atk, "ramp_up.attacks_per_stage_increase"))
        pr = d.get("production")
        if pr:
            self.skill("skills", "produceElixir", "production", interval=pr.get("interval_sec"), amount=pr.get("elixir_per_tick"))
        rl = g(d, "components.rocket_launcher")
        if rl:
            a, lv = rl["attack"], g(rl, "stats_by_level.11", {})
            self.skill("skills", "secondaryAttack", "components.rocket_launcher", character="RocketLauncher", hitSpeed=a.get("hit_speed_sec"),
                       loadTime=load_time(a.get("hit_speed_sec"), a.get("first_hit_speed_sec")), minRange=g(a, "range.min_tiles"),
                       range=g(a, "range.max_tiles"),
                       radius=a.get("area_damage_radius_tiles"), targets=targets(rl.get("targets")))
            self.arr("skills.secondaryAttack.damage", lv.get("damage"), "components.rocket_launcher")
            self.arr("skills.secondaryAttack.towerDamage", lv.get("crown_tower_damage"), "components.rocket_launcher")
        rd = g(d, "composite_attack.rider")
        if rd:
            self.skill("skills", "secondaryAttack", "composite_attack.rider", character="RamRider", hitSpeed=rd.get("hit_speed_sec"),
                       range=g(rd, "range.tiles"), targets=targets(rd.get("targets")))
            self.arr("skills.secondaryAttack.damage", self.stat("rider_damage"), "rider_damage")
            self.skill("skills", "slow", "composite_attack.rider.snare", duration=g(rd, "snare.duration_sec"),
                       speedMultiplier=g(rd, "snare.movement_reduction_percent") and -rd["snare"]["movement_reduction_percent"])
        bp = m.get("backpack_spear_goblins")
        if bp:
            self.skill("skills", "secondaryAttack", "mechanics.backpack_spear_goblins", character="SpearGoblinGiant", count=bp.get("count"))
        for name, u in (d.get("sub_units") or {}).items():
            # legacy also lists the card's own character (Skeleton Army) and units no skill spawns; those stay out
            if ALIAS.get(name, snake(name)) in self.referenced():
                self.unit(name, u, f"sub_units.{name}")
        comp = d.get("composition")
        if comp:
            members = [(v["type"], v["count"]) for v in comp.values()]
            # the card's own stats are the first member's (the leader); the others ride along as units
            for name, n in members[1:]:
                self.unit(name, g(d, f"sub_units.{name}", {}), f"composition.{name}", count=n)
            self.unit(members[0][0], g(d, f"sub_units.{members[0][0]}", {}), f"composition.{members[0][0]}", count=members[0][1], stats=False)
            self.skill("skills", "group", "composition", characters=[name for name, _ in members[1:]], leader=members[0][0])
        ents = d.get("entities")
        if ents:
            # the card's own stats are the last entity's (the Monster); the others are extra units
            names = list(ents)
            for name in names:
                e = ents[name]
                u = {"attack": e.get("attack", {}), "hidden_stats": g(d, f"hidden_stats.{name}", {}), "count": 1, "movement": d.get("movement")}
                k = self.unit(name.capitalize(), u, f"entities.{name}", stats=name != names[-1])
                if name != names[-1]:
                    for f in ("hitpoints", "damage"):
                        self.arr(f"units.{k}.{f}", g(sb, f"{name}.{f}"), f"entities.{name}")
                self.skill(f"units.{k}.skills", "stun", f"entities.{name}", duration=g(e, "attack.stun_duration_sec"))
            self.skill("skills", "group", "entities", characters=[n.capitalize() for n in names[:-1]], leader=names[-1].capitalize())
        forms = d.get("forms")
        if forms:
            for name, f in forms.items():
                if name == "empress":
                    continue
                u = {"attack": f.get("attack", {}), "movement": d.get("movement"), "count": 1}
                self.unit(name.capitalize(), u, f"forms.{name}")
                self.skill("skills", "spawnOnDeath", f"forms.{name}", character=name.capitalize(), count=1)
        su = d.get("spawned_unit")
        if su:
            self.unit(su["name"], su, "spawned_unit", count=su.get("count"))
            self.skill("skills", "spawn", "spawned_unit", character=su["name"], count=su.get("count"))
        for k, u in c["units"].items():
            if u.get("card") is None and UNIT_CARD.get(k):
                u["card"] = UNIT_CARD[k]
                self.src[f"units.{k}.card"] = "legacy:unit_card"

    def spells(self):
        c, d, sa, sb = self.c, self.d, self.d.get("spell_attributes", {}), self.sb
        if c["kind"] != "spell":
            return
        ti = sa.get("tick_interval_sec", sa.get("strike_interval_sec", num(d.get("tick_interval", d.get("hit_speed")))))
        self.put("hitSpeed", ti, "tick_interval")
        self.skill("skills", "pull", "pull_strength", strength=num(d.get("pull_strength")))
        self.arr("stats.buildingDamage", sb.get("building_damage_per_tick"), "building_damage_per_tick")
        kb = sa.get("knockback_tiles", num(d.get("knockback")))
        self.skill("skills", "pushback", "knockback", distance=kb or g(d, "mechanics.pushback_distance_tiles"))
        mt = sa.get("max_targets")
        if mt:
            self.skill("skills", "stun", "spell_attributes.max_targets", targets=mt, duration=sa.get("duration_sec") if sa.get("grounds_air") else None)
        if sa.get("volleys"):
            self.put("projectile.waves", sa["volleys"], "spell_attributes.volleys")
        unit = sa.get("spawns_unit", g(sa, "spawns.unit", g(d, "spawning.spawned_unit")))
        if unit:
            self.skill("skills", "spawn", "spawns", character=unit, count=sa.get("spawn_count", g(sa, "spawns.count")),
                       interval=g(d, "spawning.spawn_interval_sec"))
            self.unit(unit, g(d, f"sub_units.{unit}", {}), "spawns", count=sa.get("spawn_count", g(sa, "spawns.count", c["count"] or None)))
        if sa.get("conversion"):
            self.skill("skills", "spawnOnKill", "spell_attributes.conversion", character="Goblin", count=1)
            self.unit("Goblin", {}, "spell_attributes.conversion", count=1)
        sh = d.get("shape")
        if sh:
            self.put("radius", sh.get("width_tiles") and sh["width_tiles"] / 2, "shape.width_tiles")

    def champion(self):
        ab = self.d.get("ability")
        if not ab or not g(self.c, "skills.ability"):
            return
        s = "skills.ability.skills"
        self.put("skills.ability.cooldown", ab.get("cooldown_sec"), "ability.cooldown_sec")
        self.put("skills.ability.uses", ab.get("uses_per_deploy"), "ability.uses_per_deploy")
        da = ab.get("dash")
        if da:
            self.skill(s, "dash", "ability.dash", count=da.get("max_dashes"), maxRange=da.get("chain_search_radius_tiles"),
                       speed=tps(speed(da.get("speed_during_ability"))))
        ef = ab.get("effects") if isinstance(ab.get("effects"), dict) else {}
        if "invisibility" in ef:
            self.skill(s, "invisibility", "ability.effects.invisibility", duration=g(ef, "invisibility.duration_sec", ab.get("duration_sec")))
        if "teleport" in ef:
            self.skill(s, "teleport", "ability.effects.teleport", distance=g(ef, "teleport.distance_tiles"))
        if "movement_speed_override" in ef:
            base = speed(g(self.d, "movement.speed", self.d.get("movement"))) or self.c["speed"]
            pct = round(100 * g(ef, "movement_speed_override.speed_value") / base - 100)
            self.skill(s, "boost", "ability.effects.movement_speed_override", speedMultiplier=pct)
        if "bomb" in ef:
            self.skill(s, "pushback", "ability.effects.bomb", distance=g(ef, "bomb.knockback_distance_tiles"))
        if "link" in ef:
            lk = ef["link"]
            self.skill(s, "poison", "ability.effects.link", radius=lk.get("radius_tiles"), duration=ab.get("duration_sec"),
                       tickInterval=lk.get("tick_interval_sec"))
            td = g(ef, "tick_damage_by_level.11", {})
            self.arr(f"{s}.poison.damage", td.get("damage"), "ability.effects.tick_damage_by_level")
            self.arr(f"{s}.poison.towerDamage", td.get("crown_tower_damage"), "ability.effects.tick_damage_by_level")

    def hero(self):
        h = g(self.d, "hero.ability") or g(self.d, "hero_form.ability") or {}
        hf = self.d.get("hero_form", {})
        if not self.c.get("hero"):
            return
        s = "hero.ability.skills"
        self.put("hero.ability.cooldown", h.get("cooldown_sec", hf.get("ability_cooldown_sec")), "hero.ability.cooldown_sec")
        self.put("hero.ability.uses", h.get("uses_per_deploy"), "hero.ability.uses_per_deploy")
        self.skill(s, "periodicSpawn", "hero.ability.banner_duration_sec", lifetime=h.get("banner_duration_sec"))
        self.skill(s, "stack", "hero.ability", interval=h.get("cook_time_sec"), healPercent=h.get("healing_percent"))
        self.arr(f"{s}.pushback.damage", h.get("impact_damage"), "hero.ability.impact_damage")
        tr = h.get("turret")
        if tr:
            k = next((k for k, u in self.c["units"].items() if "hero.ability.skills.spawn" in u["from"]), None)
            if k:
                self.arr(f"units.{k}.hitpoints", tr.get("hp"), "hero.ability.turret")
                self.arr(f"units.{k}.damage", tr.get("damage"), "hero.ability.turret")
                self.put(f"units.{k}.hitSpeed", tr.get("hit_speed_sec"), "hero.ability.turret")
        if h.get("flight_duration_sec"):
            self.skill(s, "boost", "hero.ability.flight_duration_sec", flying=True)
        self.skill(s, "teleport", "hero.ability.dash_distance_tiles", distance=h.get("dash_distance_tiles"))

    def evo(self):
        ev = self.d.get("evolution") or {}
        ch = ev.get("changes") or {}
        if not self.c.get("evo"):
            return
        s = "evo.skills"
        self.skill(s, "charge", "evolution.changes.charge_activation_tiles", range=ch.get("charge_activation_tiles"))
        if ch.get("drops_at_75_percent_hp"):
            self.skill(s, "spawnOnDeath", "evolution.changes.drops_at_75_percent_hp", hpPercent=75)
        self.skill(s, "slow", "evolution.changes.small_sparks", count=ch.get("small_sparks"))
        self.arr(f"{s}.pull.damage", ch.get("tornado_damage"), "evolution.changes.tornado_damage")
        self.skill(s, "snipe", "evolution.changes", minRange=ch.get("sniper_min_range_tiles"), maxRange=ch.get("power_shot_max_range_tiles"))
        self.skill(s, "poison", "evolution.changes.tier_escalation_darts", stackHits=ch.get("tier_escalation_darts"))
        self.arr(f"{s}.jump.damage", ch.get("landing_damage"), "evolution.changes.landing_damage")
        self.skill(s, "jump", "evolution.changes.landing_radius_tiles", radius=ch.get("landing_radius_tiles"))
        self.skill(s, "pull", "evolution.changes.pull_radius_tiles", radius=ch.get("pull_radius_tiles"))
        oc = g(ch, "heal_on_skeleton_death.overheal_cap_multiplier")
        if oc and self.c["evo"]["stats"].get("hitpoints"):
            self.put(f"{s}.heal.overHeal", [None if x is None else int(x * oc) for x in self.c["evo"]["stats"]["hitpoints"]],
                     "evolution.changes.heal_on_skeleton_death.overheal_cap_multiplier")
        tiers = ev.get("stats_by_level")
        if isinstance(tiers, list) and any("heal_per_kill_stage_1" in t for t in tiers):
            t11 = next((t for t in tiers if t.get("level") == 11), {})
            self.put(f"{s}.heal.perKillTiers", [self.scale(t11[f"heal_per_kill_stage_{i}"]) for i in (1, 2, 3) if f"heal_per_kill_stage_{i}" in t11],
                     "evolution.stats_by_level.heal_per_kill_stage")
        gs = ch.get("goblin_spawner")
        if gs:
            self.skill(s, "periodicSpawn", "evolution.changes.goblin_spawner", character="SpearGoblin", count=1, pauseTime=gs.get("spawn_interval_sec"),
                       hpPercent=gs.get("triggers_at_hp_percent"))
        self.skill(s, "stun", "evolution.changes.net_ability", delayBetweenStrikes=g(ch, "net_ability.net_cooldown_sec"))
        ib = g(ch, "infinite_bounce.post_chain_damage_multiplier")
        if isinstance(ib, (int, float)):
            self.skill(s, "pierce", "evolution.changes.infinite_bounce", bounceDamagePercent=round(ib * 100))
        ds = ch.get("death_spawn")
        if ds:
            self.skill(s, "spawnOnDeath", "evolution.changes.death_spawn", character=ds.get("unit"), count=ds.get("count"))
        for name, u in (ev.get("sub_units") or {}).items():
            if ALIAS.get(name, snake(name)) in self.referenced():
                self.unit(name, u, f"evolution.sub_units.{name}")
        rs = g(ch, "resurface.resurface_thresholds_pct")
        if rs:
            self.skill(s, "burrow", "evolution.changes.resurface", resurfacePercent=rs)
        sd = ch.get("souldiers")
        if sd:
            self.skill(s, "spawn", "evolution.changes.souldiers", character="Souldier", count=sd.get("count"))
            k = self.unit("Souldier", {}, "evolution.changes.souldiers", count=sd.get("count"))
            st = g(sd, "souldier_stats_by_level.11", {})
            self.arr(f"units.{k}.hitpoints", st.get("hitpoints"), "evolution.changes.souldiers")
            self.arr(f"units.{k}.damage", st.get("damage"), "evolution.changes.souldiers")
        self.skill(s, "spawn", "evolution.changes.decoy_barrel", count=g(ch, "decoy_barrel.decoy_goblin_count"))
        gh = ev.get("ghost_on_death")
        if gh:
            self.skill(s, "invisibility", "evolution.ghost_on_death", duration=gh.get("lifetime_sec"))
        if ev.get("roll_distance"):
            self.skill(s, "pull", "evolution.roll_distance", distance=num(ev["roll_distance"]))
            self.skill(s, "slow", "evolution.slow_duration", duration=num(ev.get("slow_duration")))

    def tower(self):
        if self.c["kind"] != "tower":
            return
        db = g(self.d, "attack.mechanics.dagger_burst")
        if db:
            self.skill("skills", "volley", "attack.mechanics.dagger_burst", projectileCount=db.get("max_daggers"), reloadTime=db.get("cooldown_hit_speed_sec"))


def apply(card, key, scale):
    Overlay(card, key, scale).run()
