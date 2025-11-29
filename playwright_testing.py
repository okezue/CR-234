import asyncio
import pandas as pd
from playwright.async_api import async_playwright, Page
import random
import os
import re


SCROLL_TIMEOUT_SECONDS = 0.5
SCROLL_UP_AMOUNT = -1000
GIGGLE_TIME = 100

async def clear_vignette(page):
    url = page.url

    # Step 1: detect vignette URL
    if "vignette" in url or "googleads" in url:
        print("Vignette detected, going back...")
        await page.go_back()
        await page.wait_for_timeout(300)
        return

    # Step 2: try ESC
    await page.keyboard.press("Escape")
    await page.wait_for_timeout(100)

    # Step 3: close common buttons inside iframes
    for f in page.frames:
        try:
            btn = f.locator("button:has-text('Continue'), button:has-text('Skip')")
            if await btn.count() > 0:
                await btn.first.click()
                print("Closed vignette popup")
                return
        except:
            pass

async def scrape(page: Page):
    print("Scraping page content...")

    battle_data: list[dict] = []

    menus = page.locator("div.ui.text.fluid.menu.battle_bottom_menu")
    count = await menus.count()
    print("Found menus:", count)

    for i in range(count):
        menu = menus.nth(i)

        # timestamp div
        ts_div = menu.locator("div.item.i18n_duration_short.battle-timestamp-popup")
        timestamp = await ts_div.get_attribute("data-content")

        # orange replay button
        replay_button = menu.locator(
            "button.ui.orange.circular.icon.button.button_popup.replay_button"
        )

        # if no replay button, skip this menu
        if await replay_button.count() == 0:
            print(f"Menu {i}: no replay button, skipping")
            continue

        replay_id = await replay_button.get_attribute("data-replay")
        team_id = await replay_button.get_attribute("data-team-tags")
        op_id = await replay_button.get_attribute("data-opponent-tags")

        battle_data.append(
            {
                "menu_index": i,
                "timestamp_utc": timestamp,
                "replay_id": replay_id,
                "team_id": team_id,
                "opponent_id": op_id,
            }
        )

    print("Finished scraping.")
    print("Collected battles:", len(battle_data))
    return battle_data


async def scroll_until_loader_hidden(page: Page, verbose = False) -> None:
    loader = page.locator("div#scrolling_battle_loader.ui.basic.segment")

    while True:
        try:
            await asyncio.wait_for(
                loader.scroll_into_view_if_needed(),
                timeout=SCROLL_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            if verbose:
                print("Scroll to loader timed out after 0.5 seconds")
        if verbose:
            print("Scrolled to loader, checking visibility...")
        display = await loader.evaluate("el => getComputedStyle(el).display")
        visible = display != "none"

        if not visible:
            if verbose:
                print("Loader is hidden, reached the end of the list.")
            break
        await page.wait_for_timeout(GIGGLE_TIME)
        await page.mouse.wheel(0, random.randint(SCROLL_UP_AMOUNT, 0))
        if verbose:
            print("Scrolled up a bit to trigger more loading.")


async def paginate_battles(page: Page, verbose = False):
    battle_data = []
    while True:
        cur_data = await scrape(page)
        battle_data.extend(cur_data)

        next_icon = page.locator("i.angle.right.icon")
        parent_link = next_icon.locator("..")

        classes = await parent_link.get_attribute("class")
        if classes and "disabled" in classes:
            if verbose:
                print("Next page is disabled, stopping pagination.")
            return battle_data
        if verbose:
            print("Going to next page...")
        await next_icon.scroll_into_view_if_needed()
        await next_icon.click()
        await clear_vignette(page)

def find_biggest_chunk(folder):
    pattern = re.compile(r"battles_chunk_(\d+)\.csv")
    max_num = 0
    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num


CHUNK_SIZE = 10


async def run():
    # 1. Load player IDs from file
    df_players = pd.read_csv("all_players_from_clans.csv")

    # Normalize column names
    if "player_tag" not in df_players.columns:
        raise ValueError("CSV must contain a 'tag' column")

    # Remove "#" if players include it
    df_players["player_tag"] = df_players["player_tag"].str.replace("#", "")

    player_ids = df_players["player_tag"].tolist()
    print(f"Loaded {len(player_ids)} player IDs")

    battle_data = []
    chunk_index = 1
    processed_count = 0

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(storage_state="myGoogleAuth.json")
        page = await context.new_page()
        number_of_players_scanned = find_biggest_chunk("battle_chunks") * 100
        for idx, pid in enumerate(player_ids):
            if idx < number_of_players_scanned:
                continue
            print(f"Scraping player {pid} ...")

            url = f"https://royaleapi.com/player/{pid}/battles/"
            try:
                await page.goto(url)
            except:
                with open("error.txt", "a") as f:
                    f.write(pid)
                continue

            await scroll_until_loader_hidden(page, verbose= True)
            cur_data = await paginate_battles(page, verbose=True)

            # Add player id into each row
            for row in cur_data:
                row["player_id"] = pid

            battle_data.extend(cur_data)
            processed_count += 1

            # If we reached chunk size, save to file
            if processed_count % CHUNK_SIZE == 0:
                filename = f"battle_chunks/battles_chunk_{chunk_index}.csv"
                pd.DataFrame(battle_data).to_csv(filename, index=False)
                print(f"Saved chunk file: {filename}")

                battle_data = []
                chunk_index += 1

        # 6. Save remaining data
        if battle_data:
            filename = f"battles_chunk_{chunk_index}.csv"
            pd.DataFrame(battle_data).to_csv(filename, index=False)
            print(f"Saved final chunk file: {filename}")

        await browser.close()


# Run the script
asyncio.run(run())
