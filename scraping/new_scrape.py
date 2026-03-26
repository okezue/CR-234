import asyncio
import pandas as pd
import os
import shutil
import argparse
import random
import json
import urllib.request
from datetime import datetime
from curl_cffi import requests as cffi_requests
from bs4 import BeautifulSoup

ROOT_DIR=os.path.abspath(".")
PLAYERS_CSV=os.path.join(ROOT_DIR,"data/big_data/all_players_from_clans.csv")
BATTLE_CHUNKS_DIR_BASE="data/scraped_data/battle_chunks"
ERROR_LOG_PATH_BASE="error_log.txt"
SCRAPED_PLAYERS_FILE_BASE="scraping/scraped_players.txt"
BATTLE_META_CSV_BASE="data/scraped_data/battle_meta_data.csv"
PROGRESS_LOG_PATH_BASE="progress_log.txt"

MAX_CONCURRENT_WORKERS=4
DEBUG_MODE=False
CONTINUE_FROM_PREV_SCRAPE=True
DEBUG_PLAYER_ID="G9YV9GR8R"
CLEAN_PAST_DATA=False
SCRAPE_TIMEOUT_SECONDS=360000
RETRY_ATTEMPTS=3
RETRY_BACKOFF_S=2
NAV_DELAY_MIN=500
NAV_DELAY_MAX=2000
INSTANCE_ID=0
TOTAL_INSTANCES=1
CF_SCRAPER_URL="http://localhost:3000/cf-clearance-scraper"
REPLAY_API_URL="https://royaleapi.com/data/replay?tag={tag}"
SESSION_COOKIE="4e39380ddcf04e30ae2a7bd2ab7fb4ea"

nav_sem=None
http_session=None

def init_http():
    global http_session
    payload=json.dumps({"url":"https://royaleapi.com/","mode":"waf-session"}).encode()
    req=urllib.request.Request(CF_SCRAPER_URL,data=payload,headers={"Content-Type":"application/json"})
    resp=urllib.request.urlopen(req,timeout=120)
    data=json.loads(resp.read())
    http_session=cffi_requests.Session(impersonate="chrome")
    for c in data.get("cookies",[]):
        http_session.cookies.set(c["name"],c["value"],domain=c.get("domain",".royaleapi.com"))
    http_session.cookies.set("__royaleapi_session_v2",SESSION_COOKIE,domain="royaleapi.com")
    ua=data.get("headers",{}).get("user-agent","")
    http_session.headers.update({"User-Agent":ua})
    print(f"HTTP client ready with {len(http_session.cookies)} cookies")

def fetch_page(url):
    payload=json.dumps({"url":url,"mode":"source"}).encode()
    req=urllib.request.Request(CF_SCRAPER_URL,data=payload,headers={"Content-Type":"application/json"})
    resp=urllib.request.urlopen(req,timeout=120)
    data=json.loads(resp.read())
    return data.get("source","")

def fetch_replay(tag):
    url=REPLAY_API_URL.format(tag=tag)
    resp=http_session.get(url,timeout=30)
    if resp.status_code!=200:
        raise RuntimeError(f"HTTP {resp.status_code}")
    jdata=resp.json()
    if jdata.get("success") and jdata.get("html"):
        return jdata["html"]
    return None

def update_progress_log(path,battles,players,start):
    elapsed=datetime.now()-start
    with open(path,"w",encoding="utf8") as f:
        f.write(f"Battles scraped: {battles}\n")
        f.write(f"Players scraped: {players}\n")
        f.write(f"Time spent: {elapsed}\n")
        f.write(f"Last updated: {datetime.now()}\n")

def elapsed_str(st:datetime)->str:
    d=datetime.now()-st
    t=int(d.total_seconds())
    h,m,s=t//3600,(t%3600)//60,t%60
    return f"{h:02d}:{m:02d}:{s:02d}"

def split_for_workers(ids:list[str],mw:int)->list[list[str]]:
    if not ids:
        return []
    n=min(mw,len(ids))
    b:list[list[str]]=[[] for _ in range(n)]
    for i,pid in enumerate(ids):
        b[i%n].append(pid)
    return b

def get_fraction(ids:list[str],inst_id:int,total:int)->list[str]:
    n=len(ids)
    chunk=n//total
    start=inst_id*chunk
    end=(inst_id+1)*chunk if inst_id<total-1 else n
    return ids[start:end]

def remaining_player_ids(all_ids:list[str],sf:str)->list[str]:
    if not os.path.exists(sf):
        return all_ids
    last=None
    with open(sf,"r",encoding="utf8") as f:
        for line in f:
            pid=line.strip()
            if pid:
                last=pid
    if last is None:
        return all_ids
    try:
        idx=all_ids.index(last)
    except ValueError:
        return all_ids
    return all_ids[idx+1:]

def validate(rec:dict)->bool:
    for k in ["battle_id","card","time","side","team"]:
        if k not in rec or rec[k] is None:
            return False
    if not isinstance(rec["time"],int):
        return False
    return True

def extract_meta(html,pid)->list[dict]:
    soup=BeautifulSoup(html,"html.parser")
    rows=[]
    for menu in soup.select("div.ui.text.fluid.menu.battle_bottom_menu"):
        rb=menu.select_one("button.replay_button")
        if not rb:
            continue
        rid=rb.get("data-replay","")
        team=rb.get("data-team-tags","")
        opp=rb.get("data-opponent-tags","")
        ts_div=menu.select_one("div.item.i18n_duration_short.battle-timestamp-popup")
        ts=ts_div.get("data-content","") if ts_div else ""
        gm=""
        p=menu.parent
        for _ in range(5):
            if not p:
                break
            el=p.select_one(".battle_type,a[href*='/gamemode/']")
            if el:
                gm=el.get_text(strip=True).split("\n")[0]
                break
            p=p.parent
        rows.append({
            "replayTag":rid,
            "player_id":pid,
            "timestamp":ts,
            "team_tags":team,
            "opponent_tags":opp,
            "gameMode_name":gm,
        })
    return rows

def parse_replay(html,bid)->list[dict]:
    soup=BeautifulSoup(html,"html.parser")
    cards_loc=soup.select("div.replay_team img.replay_card")
    markers_loc=soup.select("div.markers > div")
    ability_map={}
    for img in cards_loc:
        ct=img.get("data-t")
        cs=img.get("data-s")
        ab=img.get("data-ability")
        if ab and ab!="None":
            ability_map[(ct,cs)]=int(ab)
    results=[]
    for marker in markers_loc:
        x=marker.get("data-x")
        y=marker.get("data-y")
        t=marker.get("data-t")
        s=marker.get("data-s")
        c=marker.get("data-c")
        idx=marker.get("data-i")
        span=marker.find("span")
        lvl=span.get_text(strip=True) if span else ""
        classes=" ".join(marker.get("class",[]))
        team="red" if "red" in classes else "blue"
        ab=ability_map.get((t,s),0)
        rec={
            "battle_id":bid,
            "x":int(x) if x and x!="None" else None,
            "y":int(y) if y and y!="None" else None,
            "card":c if c else None,
            "time":int(t) if t and t!="None" else None,
            "side":s,
            "team":team,
            "card_index":int(idx) if idx and idx!="None" else None,
            "level":int(lvl) if lvl.isdigit() else None,
            "ability":ab,
        }
        if validate(rec):
            results.append(rec)
    return results

async def scrape_player(pid,wi,start_time):
    url=f"https://royaleapi.com/player/{pid}/battles/"
    async with nav_sem:
        await asyncio.sleep(random.randint(NAV_DELAY_MIN,NAV_DELAY_MAX)/1000)
        html=await asyncio.get_running_loop().run_in_executor(None,fetch_page,url)
    if not html or len(html)<1000:
        print(f"{elapsed_str(start_time)} [W{wi}] Empty page for {pid}, skip")
        return [],[]
    soup=BeautifulSoup(html,"html.parser")
    btns=soup.select("button.replay_button")
    if not btns:
        print(f"{elapsed_str(start_time)} [W{wi}] No replay buttons for {pid}, skip")
        return [],[]
    print(f"{elapsed_str(start_time)} [W{wi}] Found {len(btns)} battles for {pid}")
    meta=extract_meta(html,pid)
    replay_data=[]
    for btn in btns:
        bid=btn.get("data-replay","")
        if not bid:
            continue
        for attempt in range(RETRY_ATTEMPTS):
            try:
                rhtml=await asyncio.get_running_loop().run_in_executor(None,fetch_replay,bid)
                if rhtml:
                    rows=parse_replay(rhtml,bid)
                    deck_rows=[r for r in rows if r.get("card")!="_invalid" and r.get("ability",0)==0]
                    t_cards=set(r["card"] for r in deck_rows if r.get("side")=="t")
                    o_cards=set(r["card"] for r in deck_rows if r.get("side")=="o")
                    if len(t_cards)>8 or len(o_cards)>8:
                        break
                    for r in rows:
                        r["player_id"]=pid
                    replay_data.extend(rows)
                    break
                else:
                    if attempt<RETRY_ATTEMPTS-1:
                        await asyncio.sleep(RETRY_BACKOFF_S*(attempt+1))
            except Exception as e:
                if attempt<RETRY_ATTEMPTS-1:
                    await asyncio.sleep(RETRY_BACKOFF_S*(attempt+1))
                else:
                    print(f"{elapsed_str(start_time)} [W{wi}] Replay {bid} failed: {e}")
    print(f"{elapsed_str(start_time)} [W{wi}] Scraped {len(replay_data)} events from {len(btns)} battles for {pid}")
    return replay_data,meta

async def worker(player_ids:list[str],wi:int,run_id:str,start_time:datetime):
    fn=os.path.join(BATTLE_CHUNKS_DIR,f"{run_id}_worker_{wi}_results.csv" if run_id else f"worker_{wi}_results.csv")
    fe=os.path.exists(fn)
    plog=os.path.join(ROOT_DIR,f"{run_id}_{PROGRESS_LOG_PATH_BASE}" if run_id else PROGRESS_LOG_PATH_BASE)
    bs=0
    ps=0
    swt=datetime.now()
    for pid in player_ids:
        try:
            replay_data,meta_data=await scrape_player(pid,wi,start_time)
            if replay_data:
                os.makedirs(BATTLE_CHUNKS_DIR,exist_ok=True)
                pd.DataFrame(replay_data).to_csv(fn,mode="a",header=not fe,index=False)
                print(f"{elapsed_str(start_time)} [W{wi}] Saved {len(replay_data)} rows for {pid}")
                fe=True
            if meta_data:
                os.makedirs(os.path.dirname(BATTLE_META_CSV),exist_ok=True)
                mdf=pd.DataFrame(meta_data).drop_duplicates(subset=["replayTag"])
                me=os.path.exists(BATTLE_META_CSV)
                mdf.to_csv(BATTLE_META_CSV,mode="a",header=not me,index=False)
            with open(SCRAPED_PLAYERS_FILE,"a",encoding="utf8") as f:
                f.write(f"{pid}\n")
            bs+=len(replay_data)
            ps+=1
            update_progress_log(plog,bs,ps,swt)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            em=f"Error for {pid}: {e}\n"
            with open(ERROR_LOG_PATH,"a",encoding="utf8") as f:
                f.write(em)
            print(f"{elapsed_str(start_time)} [W{wi}] {em.strip()}")
    update_progress_log(plog,bs,ps,swt)

async def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--id",help="Run ID")
    parser.add_argument("--instance-id",type=int,default=0)
    parser.add_argument("--total-instances",type=int,default=1)
    parser.add_argument("--session-cookie",type=str,default=None)
    parser.add_argument("--cf-url",type=str,default=None)
    parser.add_argument("--players-csv",type=str,default=None)
    args=parser.parse_args()
    run_id=args.id or ""
    global INSTANCE_ID,TOTAL_INSTANCES,nav_sem,SESSION_COOKIE,CF_SCRAPER_URL,PLAYERS_CSV
    INSTANCE_ID=args.instance_id
    TOTAL_INSTANCES=args.total_instances
    if args.session_cookie:
        SESSION_COOKIE=args.session_cookie
    if args.cf_url:
        CF_SCRAPER_URL=args.cf_url
    if args.players_csv:
        PLAYERS_CSV=args.players_csv
    nav_sem=asyncio.Semaphore(2)
    start_time=datetime.now()
    global BATTLE_CHUNKS_DIR,ERROR_LOG_PATH,SCRAPED_PLAYERS_FILE,BATTLE_META_CSV
    BATTLE_CHUNKS_DIR=os.path.join(ROOT_DIR,f"{run_id}_{BATTLE_CHUNKS_DIR_BASE}" if run_id else BATTLE_CHUNKS_DIR_BASE)
    ERROR_LOG_PATH=os.path.join(ROOT_DIR,f"{run_id}_{ERROR_LOG_PATH_BASE}" if run_id else ERROR_LOG_PATH_BASE)
    SCRAPED_PLAYERS_FILE=os.path.join(ROOT_DIR,f"{run_id}_{SCRAPED_PLAYERS_FILE_BASE}" if run_id else SCRAPED_PLAYERS_FILE_BASE)
    BATTLE_META_CSV=os.path.join(ROOT_DIR,f"{run_id}_{BATTLE_META_CSV_BASE}" if run_id else BATTLE_META_CSV_BASE)
    if CLEAN_PAST_DATA:
        if os.path.exists(BATTLE_CHUNKS_DIR):
            shutil.rmtree(BATTLE_CHUNKS_DIR)
        if os.path.exists(BATTLE_META_CSV):
            os.remove(BATTLE_META_CSV)
        if os.path.exists(SCRAPED_PLAYERS_FILE):
            os.remove(SCRAPED_PLAYERS_FILE)
        print("Past data cleaned.")
    os.makedirs(BATTLE_CHUNKS_DIR,exist_ok=True)
    os.makedirs(os.path.dirname(BATTLE_META_CSV),exist_ok=True)
    os.makedirs(os.path.dirname(SCRAPED_PLAYERS_FILE),exist_ok=True)
    df=pd.read_csv(PLAYERS_CSV)
    if "player_tag" not in df.columns:
        raise ValueError("CSV must contain a 'player_tag' column")
    df["player_tag"]=df["player_tag"].astype(str).str.replace("#","",regex=False)
    all_ids=df["player_tag"].tolist()
    print("Initializing HTTP client with CF cookies...")
    init_http()
    if TOTAL_INSTANCES>1:
        all_ids=get_fraction(all_ids,INSTANCE_ID,TOTAL_INSTANCES)
        print(f"Instance {INSTANCE_ID}/{TOTAL_INSTANCES}: {len(all_ids)} players")
    if DEBUG_MODE:
        if not DEBUG_PLAYER_ID:
            print("Debug mode enabled but DEBUG_PLAYER_ID not set")
            return
        all_ids=[DEBUG_PLAYER_ID]
        print(f"Debug mode: scraping only {DEBUG_PLAYER_ID}")
    if CONTINUE_FROM_PREV_SCRAPE:
        all_ids=remaining_player_ids(all_ids,SCRAPED_PLAYERS_FILE)
    print(f"Loaded {len(all_ids)} player IDs to scrape")
    wlists=split_for_workers(all_ids,MAX_CONCURRENT_WORKERS)
    tasks=[
        asyncio.create_task(worker(pl,wi=i,run_id=run_id,start_time=start_time))
        for i,pl in enumerate(wlists) if pl
    ]
    try:
        await asyncio.wait_for(asyncio.gather(*tasks),timeout=SCRAPE_TIMEOUT_SECONDS)
        print("All done.")
    except asyncio.TimeoutError:
        print(f"Timeout after {SCRAPE_TIMEOUT_SECONDS}s.")
    except Exception as e:
        print(f"Crashed: {e}")
    finally:
        for t in tasks:
            if not t.done():
                t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)

if __name__=="__main__":
    asyncio.run(main())
