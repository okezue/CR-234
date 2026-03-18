import asyncio,json,urllib.request
from playwright.async_api import async_playwright

def get_cf():
    payload=json.dumps({"url":"https://royaleapi.com/","mode":"waf-session"}).encode()
    req=urllib.request.Request("http://localhost:3000/cf-clearance-scraper",data=payload,headers={"Content-Type":"application/json"})
    resp=urllib.request.urlopen(req,timeout=120)
    data=json.loads(resp.read())
    cookies=[]
    for c in data.get("cookies",[]):
        cookies.append({"name":c["name"],"value":c["value"],"domain":c.get("domain",".royaleapi.com"),"path":c.get("path","/"),"expires":c.get("expires",-1),"httpOnly":c.get("httpOnly",False),"secure":c.get("secure",False),"sameSite":c.get("sameSite","None")})
    ua=data.get("headers",{}).get("user-agent","")
    return cookies,ua

async def run():
    cookies,ua=get_cf()
    print(f"Got {len(cookies)} cookies")
    async with async_playwright() as p:
        browser=await p.chromium.launch(headless=False)
        context=await browser.new_context(user_agent=ua,viewport={"width":1280,"height":720})
        await context.add_cookies(cookies)
        page=await context.new_page()
        await page.goto("https://royaleapi.com/player/UG9RGJ20P/battles/",wait_until="domcontentloaded",timeout=60000)
        await page.wait_for_timeout(10000)
        title=await page.title()
        print(f"Title: {title}")
        el=page.locator("div.ui.container.sidemargin0.battle_list_container")
        print(f"Container: {await el.count()}")
        btns=page.locator("button.replay_button")
        print(f"Replay buttons: {await btns.count()}")
        menus=page.locator("div.ui.text.fluid.menu.battle_bottom_menu")
        print(f"Battle menus: {await menus.count()}")
        await page.screenshot(path="scraping/debug_screenshot.png")
        print("Screenshot saved")
        await page.wait_for_timeout(30000)
        await browser.close()

asyncio.run(run())
