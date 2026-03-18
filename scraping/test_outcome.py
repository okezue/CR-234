import json,urllib.request,re,time,sys
CF="http://localhost:3000/cf-clearance-scraper"
from curl_cffi import requests as cr
print("Getting WAF session...")
payload=json.dumps({"url":"https://royaleapi.com/","mode":"waf-session"}).encode()
req=urllib.request.Request(CF,data=payload,headers={"Content-Type":"application/json"})
resp=urllib.request.urlopen(req,timeout=180)
d=json.loads(resp.read())
s=cr.Session(impersonate="chrome")
for c in d.get("cookies",[]):
    s.cookies.set(c["name"],c["value"],domain=c.get("domain",".royaleapi.com"))
s.cookies.set("__royaleapi_session_v2","4e39380ddcf04e30ae2a7bd2ab7fb4ea",domain="royaleapi.com")
ua=d.get("headers",{}).get("user-agent","")
s.headers.update({"User-Agent":ua})
print(f"Got {len(s.cookies)} cookies")
time.sleep(1)
print("Fetching battle history...")
r=s.get("https://royaleapi.com/player/PQUV9R82G/battles/history",timeout=30)
html=r.text
print(f"Status={r.status_code} len={len(html)}")
if r.status_code!=200:
    print(html[:500])
    sys.exit(1)
with open("/tmp/battle_sample.html","w") as f:
    f.write(html)
for pat in ["victory","defeat","draw","crown","trophy","score","result","battle_header","data-replay"]:
    ms=re.findall(r".{0,100}"+pat+r".{0,100}",html,re.I)[:3]
    for m in ms:
        print(f"[{pat}]: {m.strip()[:250]}")
    if not ms:
        print(f"[{pat}]: NOT FOUND")
print("---FIRST 3000 CHARS---")
print(html[:3000])
