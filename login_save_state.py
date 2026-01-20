import asyncio
from playwright.async_api import async_playwright, Browser, BrowserContext, Page
import os
from typing import Dict

STORAGE_STATE_PATH = os.path.join(os.path.abspath("."), "myGoogleAuth.json")

STEALTH_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--disable-infobars",
    "--disable-web-security",
    "--disable-features=IsolateOrigins,site-per-process",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--start-maximized",
]

STEALTH_INIT_SCRIPT = """
// Remove webdriver flag
Object.defineProperty(navigator, 'webdriver', {
  get: () => undefined,
});

// Fake plugins
Object.defineProperty(navigator, 'plugins', {
  get: () => [1, 2, 3, 4],
});

// Fake languages
Object.defineProperty(navigator, 'languages', {
  get: () => ['en-US', 'en'],
});
"""

async def create_stealth_context(browser: Browser) -> BrowserContext:
    context = await browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1280, "height": 720},
    )
    await context.add_init_script(STEALTH_INIT_SCRIPT)
    return context

async def login_and_save():
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False, args=STEALTH_ARGS)
        context = await create_stealth_context(browser)
        page = await context.new_page()
        
        await page.goto("https://royaleapi.com/", wait_until="networkidle")
        
        print("Please log in manually in the browser window.")
        print("Once logged in, press Enter in the terminal to save the session.")
        
        input("Press Enter after logging in...")
        
        # Save the storage state
        await context.storage_state(path=STORAGE_STATE_PATH)
        print(f"Storage state saved to {STORAGE_STATE_PATH}")
        
        await browser.close()

if __name__ == "__main__":
    asyncio.run(login_and_save())