import asyncio,os
from playwright.async_api import async_playwright

async def run():
    async with async_playwright() as p:
        ctx=await p.chromium.launch_persistent_context(
            os.path.expanduser("~/Library/Application Support/Google/Chrome"),
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
            viewport={"width":1280,"height":720},
        )
        page=ctx.pages[0] if ctx.pages else await ctx.new_page()
        print("Navigating to royaleapi...")
        try:
            await page.goto("https://royaleapi.com/player/UG9RGJ20P/battles/",wait_until="commit",timeout=60000)
            print(f"URL: {page.url}")
            await page.wait_for_timeout(10000)
            title=await page.title()
            print(f"Title: {title}")
            el=page.locator("div.ui.container.sidemargin0.battle_list_container")
            cnt=await el.count()
            print(f"Battle container found: {cnt>0}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Final URL: {page.url}")
        await page.wait_for_timeout(5000)
        await ctx.close()

asyncio.run(run())
