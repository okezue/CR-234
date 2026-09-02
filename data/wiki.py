import json
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

RAW = Path(__file__).resolve().parent / "raw" / "wiki"
API = "https://clashroyale.fandom.com/api.php?"


def page(title):
    # the HTML site sits behind Cloudflare, the MediaWiki API does not
    RAW.mkdir(parents=True, exist_ok=True)
    p = RAW / (re.sub(r"[^A-Za-z0-9]+", "_", title) + ".json")
    if not p.exists():
        q = urllib.parse.urlencode({"action": "parse", "page": title, "prop": "wikitext", "format": "json", "redirects": 1})
        req = urllib.request.Request(API + q, headers={"User-Agent": "cr234-verify"})
        for attempt in range(4):
            try:
                p.write_bytes(urllib.request.urlopen(req, timeout=60).read())
                break
            except (urllib.error.URLError, OSError):
                if attempt == 3:
                    raise
                time.sleep(2 * (attempt + 1))
        time.sleep(0.5)
    d = json.loads(p.read_text())
    return d["parse"]["wikitext"]["*"] if "parse" in d else ""


def clean(s):
    s = re.sub(r"\{\{Rarity\|([^}]+)\}\}", r"\1", s)
    s = re.sub(r"\[\[[^\]|]*\|([^\]]+)\]\]", r"\1", s)
    s = re.sub(r"\{\{[^{}]*\}\}|<[^>]+>|scope=\"col\"", "", s)
    return s.strip(" |!\n")


def vars_(text):
    return {k.strip(): v.strip() for k, v in re.findall(r"\{\{#vardefine:\s*([^|]+?)\s*\|\s*([^}]*?)\s*\}\}", text)}


def table(text, tid):
    m = re.search(r"\{\|[^\n]*id=\"%s\"[^\n]*\n(.*?)\n\|\}" % re.escape(tid), text, re.S)
    if not m:
        return None
    body = re.sub(r"\{\{Icon[^}]*\}\}", "", m.group(1))
    head, _, rows = body.partition("\n|-")
    cols = [clean(c) for c in re.split(r"\n!|!!", head) if clean(c)]
    out = []
    for row in re.split(r"\n\|-\s*", rows):
        cells = [clean(c) for c in re.split(r"\|\||\n\|", "\n" + row.strip()) if c.strip()]
        if cells:
            out.append(dict(zip(cols, cells)))
    return out


def num(s):
    m = re.search(r"-?\d[\d,]*\.?\d*", s or "")
    return float(m.group(0).replace(",", "")) if m else None


def paren(s):
    m = re.search(r"\(([\d.]+)\)", s or "")
    return float(m.group(1)) if m else num(s)


def attrs(text):
    rows = table(text, "unit-attributes-table")
    return rows[0] if rows else {}
