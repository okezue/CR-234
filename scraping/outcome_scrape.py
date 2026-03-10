import asyncio,pandas as pd,os,re,json,urllib.request,argparse,random,time
from datetime import datetime
from curl_cffi import requests as cr
from bs4 import BeautifulSoup
ROOT_DIR=os.path.abspath(".")
META_CSV=os.path.join(ROOT_DIR,"data/collected/all_battle_meta.csv")
OUT_CSV=os.path.join(ROOT_DIR,"data/collected/battle_outcomes.csv")
PROGRESS_PATH=os.path.join(ROOT_DIR,"outcome_progress.txt")
DONE_FILE=os.path.join(ROOT_DIR,"outcome_done_players.txt")
CF_URL="http://localhost:3000/cf-clearance-scraper"
SESSION_COOKIE="4e39380ddcf04e30ae2a7bd2ab7fb4ea"
MAX_WORKERS=4
MAX_PAGES=200
INSTANCE_ID=0
TOTAL_INSTANCES=1
MIN_GAP=0.15
http_session=None
last_req=0
nav_sem=None
def init_http():
    global http_session
    payload=json.dumps({"url":"https://royaleapi.com/","mode":"waf-session"}).encode()
    req=urllib.request.Request(CF_URL,data=payload,headers={"Content-Type":"application/json"})
    resp=urllib.request.urlopen(req,timeout=180)
    d=json.loads(resp.read())
    http_session=cr.Session(impersonate="chrome")
    for c in d.get("cookies",[]):
        http_session.cookies.set(c["name"],c["value"],domain=c.get("domain",".royaleapi.com"))
    http_session.cookies.set("__royaleapi_session_v2",SESSION_COOKIE,domain="royaleapi.com")
    ua=d.get("headers",{}).get("user-agent","")
    http_session.headers.update({"User-Agent":ua})
    print(f"HTTP ready ({len(http_session.cookies)} cookies)")
def throttle():
    global last_req
    now=time.time()
    gap=now-last_req
    if gap<MIN_GAP:time.sleep(MIN_GAP-gap)
    last_req=time.time()
def fetch_history(pid,before=None):
    throttle()
    url=f"https://royaleapi.com/player/{pid}/battles/history"
    if before:url+=f"?before={before}&&"
    resp=http_session.get(url,timeout=30)
    if resp.status_code==429:
        time.sleep(10);init_http()
        resp=http_session.get(url,timeout=30)
    if resp.status_code!=200:return ""
    return resp.text
def extract_outcomes(html,pid):
    soup=BeautifulSoup(html,"html.parser")
    rows=[]
    headers=soup.select("div.battle_header")
    menus=soup.select("div.ui.text.fluid.menu.battle_bottom_menu")
    tag_map={}
    for menu in menus:
        rb=menu.select_one("button.replay_button")
        if not rb:continue
        rid=rb.get("data-replay","")
        p=menu
        for _ in range(10):
            p=p.parent
            if not p:break
            bh=p.select_one("div.battle_header")
            if bh:
                tag_map[id(bh)]=rid
                break
    for bh in headers:
        rid=tag_map.get(id(bh),"")
        if not rid:
            p=bh
            for _ in range(10):
                p=p.parent
                if not p:break
                rb=p.select_one("button.replay_button")
                if rb:
                    rid=rb.get("data-replay","")
                    break
        if not rid:continue
        rh=bh.select_one("div.result_header")
        tc,oc=0,0
        if rh:
            txt=rh.get_text(strip=True)
            m=re.search(r"(\d+)\s*-\s*(\d+)",txt)
            if m:tc,oc=int(m.group(1)),int(m.group(2))
        if tc>oc:res="W"
        elif tc<oc:res="L"
        else:res="D"
        rows.append({"replayTag":rid,"player_id":pid,"result":res,"team_crowns":tc,"opp_crowns":oc})
    return rows
def extract_timestamps(html):
    return re.findall(r'data-timestamp="([^"]+)"',html)
def extract_replay_tags(html):
    return list(set(re.findall(r'data-replay="([^"]+)"',html)))
def elapsed(st):
    t=int((datetime.now()-st).total_seconds())
    return f"{t//3600:02d}:{(t%3600)//60:02d}:{t%60:02d}"
async def scrape_player(pid,wi,st,want_tags):
    html=await asyncio.get_running_loop().run_in_executor(None,fetch_history,pid,None)
    if not html or len(html)<500:
        print(f"{elapsed(st)} [W{wi}] Empty for {pid}")
        return []
    all_tags=set(extract_replay_tags(html))
    results=extract_outcomes(html,pid)
    found={r["replayTag"] for r in results}
    if want_tags and not want_tags.issubset(found):
        before=None
        befores=re.findall(r'before=(\d+)',html)
        if befores:before=min(befores)
        else:
            ts_list=extract_timestamps(html)
            if ts_list:before=str(int(float(min(ts_list))*1000))
        page=1;stale=0
        while before and page<MAX_PAGES:
            page+=1
            try:
                hh=await asyncio.get_running_loop().run_in_executor(None,fetch_history,pid,before)
            except:break
            if not hh or len(hh)<100:break
            new_tags=set(extract_replay_tags(hh))
            new_u=new_tags-all_tags
            if not new_u:
                stale+=1
                if stale>=2:break
            else:stale=0
            all_tags.update(new_tags)
            nr=extract_outcomes(hh,pid)
            results.extend(nr)
            found.update(r["replayTag"] for r in nr)
            if want_tags.issubset(found):break
            bfs=re.findall(r'before=(\d+)',hh)
            if bfs:
                nb=min(bfs)
                if nb==before:break
                before=nb
            else:
                ts2=extract_timestamps(hh)
                if ts2:before=str(int(float(min(ts2))*1000))
                else:break
    print(f"{elapsed(st)} [W{wi}] {pid}: {len(results)} outcomes ({len(all_tags)} battles scanned)")
    return results
async def worker(pids,tag_map,wi,st):
    fn=OUT_CSV.replace(".csv",f"_w{wi}.csv")
    fe=os.path.exists(fn)
    cnt=0
    for pid in pids:
        try:
            want=tag_map.get(pid,set())
            rows=await scrape_player(pid,wi,st,want)
            if rows:
                pd.DataFrame(rows).to_csv(fn,mode="a",header=not fe,index=False)
                fe=True;cnt+=len(rows)
            with open(DONE_FILE,"a") as f:f.write(f"{pid}\n")
        except asyncio.CancelledError:raise
        except Exception as e:
            print(f"{elapsed(st)} [W{wi}] ERR {pid}: {e}")
    print(f"{elapsed(st)} [W{wi}] Done. {cnt} outcomes saved.")
async def main():
    global INSTANCE_ID,TOTAL_INSTANCES,nav_sem,SESSION_COOKIE,CF_URL,META_CSV,OUT_CSV,DONE_FILE,MAX_WORKERS
    parser=argparse.ArgumentParser()
    parser.add_argument("--meta-csv",type=str,default=None)
    parser.add_argument("--out-csv",type=str,default=None)
    parser.add_argument("--instance-id",type=int,default=0)
    parser.add_argument("--total-instances",type=int,default=1)
    parser.add_argument("--session-cookie",type=str,default=None)
    parser.add_argument("--cf-url",type=str,default=None)
    parser.add_argument("--workers",type=int,default=4)
    args=parser.parse_args()
    INSTANCE_ID=args.instance_id
    TOTAL_INSTANCES=args.total_instances
    if args.session_cookie:SESSION_COOKIE=args.session_cookie
    if args.cf_url:CF_URL=args.cf_url
    if args.meta_csv:META_CSV=args.meta_csv
    if args.out_csv:OUT_CSV=args.out_csv
    MAX_WORKERS=args.workers
    DONE_FILE=OUT_CSV.replace(".csv","_done.txt") if args.out_csv else DONE_FILE
    nav_sem=asyncio.Semaphore(10)
    st=datetime.now()
    print(f"Loading meta from {META_CSV}...")
    df=pd.read_csv(META_CSV)
    print(f"Loaded {len(df)} battles")
    tag_map={}
    for _,r in df.iterrows():
        pid=str(r["player_id"])
        rt=str(r["replayTag"])
        if pid not in tag_map:tag_map[pid]=set()
        tag_map[pid].add(rt)
    pids=sorted(tag_map.keys())
    if TOTAL_INSTANCES>1:
        n=len(pids);ch=n//TOTAL_INSTANCES
        s=INSTANCE_ID*ch
        e=(INSTANCE_ID+1)*ch if INSTANCE_ID<TOTAL_INSTANCES-1 else n
        pids=pids[s:e]
        print(f"Instance {INSTANCE_ID}/{TOTAL_INSTANCES}: {len(pids)} players")
    if os.path.exists(DONE_FILE):
        done=set()
        with open(DONE_FILE) as f:
            for l in f:
                p=l.strip()
                if p:done.add(p)
        pids=[p for p in pids if p not in done]
        print(f"Resuming: {len(done)} already done, {len(pids)} remaining")
    print(f"Scraping outcomes for {len(pids)} players...")
    init_http()
    wn=min(MAX_WORKERS,len(pids))
    buckets=[[] for _ in range(wn)]
    for i,p in enumerate(pids):buckets[i%wn].append(p)
    tasks=[asyncio.create_task(worker(b,tag_map,i,st)) for i,b in enumerate(buckets) if b]
    try:
        await asyncio.gather(*tasks)
        print("All done.")
    except Exception as e:
        print(f"Crashed: {e}")
    finally:
        for t in tasks:
            if not t.done():t.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
if __name__=="__main__":
    asyncio.run(main())
