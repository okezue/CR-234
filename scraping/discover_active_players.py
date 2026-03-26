import json,urllib.request,time,os,sys
from bs4 import BeautifulSoup

CF_URL=os.environ.get("CF_URL","http://localhost:3000/cf-clearance-scraper")
OUT_FILE=sys.argv[1] if len(sys.argv)>1 else "active_players.csv"

SOURCES=[
    "https://royaleapi.com/players/leaderboard",
    "https://royaleapi.com/players/leaderboard/2",
    "https://royaleapi.com/players/leaderboard/3",
    "https://royaleapi.com/players/leaderboard/4",
    "https://royaleapi.com/players/leaderboard/5",
    "https://royaleapi.com/clans/leaderboard",
]

for i in range(1,51):
    SOURCES.append(f"https://royaleapi.com/players/leaderboard/{i}")

def fetch_page(url):
    payload=json.dumps({"url":url,"mode":"source"}).encode()
    req=urllib.request.Request(CF_URL,data=payload,headers={"Content-Type":"application/json"})
    resp=urllib.request.urlopen(req,timeout=120)
    data=json.loads(resp.read())
    return data.get("source","")

def extract_player_tags(html):
    soup=BeautifulSoup(html,"html.parser")
    tags=set()
    for a in soup.select("a[href*='/player/']"):
        href=a.get("href","")
        parts=href.split("/player/")
        if len(parts)>1:
            tag=parts[1].strip("/").split("/")[0].replace("#","")
            if len(tag)>=6:tags.add(tag)
    return tags

def extract_clan_tags(html):
    soup=BeautifulSoup(html,"html.parser")
    tags=set()
    for a in soup.select("a[href*='/clan/']"):
        href=a.get("href","")
        parts=href.split("/clan/")
        if len(parts)>1:
            tag=parts[1].strip("/").split("/")[0].replace("#","")
            if len(tag)>=6:tags.add(tag)
    return tags

all_players=set()
print(f"Discovering active players from RoyaleAPI...")

for url in SOURCES:
    try:
        print(f"  Fetching {url}...")
        html=fetch_page(url)
        if not html or len(html)<500:
            print(f"    Empty page, skipping")
            continue
        players=extract_player_tags(html)
        all_players.update(players)
        print(f"    Found {len(players)} players (total: {len(all_players)})")
        time.sleep(2)
    except Exception as e:
        print(f"    Error: {e}")
        time.sleep(5)

for url in SOURCES:
    if "clan" not in url:continue
    try:
        html=fetch_page(url)
        clans=extract_clan_tags(html)
        for clan_tag in list(clans)[:50]:
            try:
                clan_url=f"https://royaleapi.com/clan/{clan_tag}/members"
                chtml=fetch_page(clan_url)
                members=extract_player_tags(chtml)
                all_players.update(members)
                print(f"  Clan {clan_tag}: {len(members)} members (total: {len(all_players)})")
                time.sleep(1)
            except:pass
    except:pass

with open(OUT_FILE,"w") as f:
    f.write("player_tag\n")
    for tag in sorted(all_players):
        f.write(f"#{tag}\n")

print(f"\nDone: {len(all_players)} active players -> {OUT_FILE}")
