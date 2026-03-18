import asyncio
import os
from playwright.async_api import async_playwright

CHROME_PROFILE=os.path.expanduser("~/Library/Application Support/Google/Chrome")
OUT=os.path.join(os.path.abspath("."),"scraping/myGoogleAuth.json")

async def run():
    async with async_playwright() as p:
        ctx=await p.chromium.launch_persistent_context(
            CHROME_PROFILE,
            headless=False,
            channel="chrome",
            args=["--disable-blink-features=AutomationControlled"],
        )
        print("Browser opened. Saving storage state...")
        await ctx.storage_state(path=OUT)
        print(f"Saved auth to {OUT}")
        await ctx.close()
        print("Done.")

if __name__=="__main__":
    asyncio.run(run())
