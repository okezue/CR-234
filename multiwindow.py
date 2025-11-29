import asyncio
import pandas as pd
import os
import re
import random
from playwright.async_api import async_playwright, Page

# Change this to wherever you want your root directory
ROOT_DIR = os.path.abspath(".")  # for example: "/home/ubuntu/clash_bot"

PLAYERS_CSV = os.path.join(ROOT_DIR, "all_players_from_clans.csv")
BATTLE_CHUNKS_DIR = os.path.join(ROOT_DIR, "battle_chunks")
ERROR_LOG_PATH = os.path.join(ROOT_DIR, "error_log.txt")
STORAGE_STATE_PATH = os.path.join(ROOT_DIR, "myGoogleAuth.json")

TOTAL_INSTANCES = 10
# zero indexed
INSTANCE_ID = 0

SCROLL_TIMEOUT_SECONDS = 0.5
SCROLL_UP_AMOUNT = -1000
CHUNK_SIZE = 100          # save every 100 players
MAX_CONCURRENT_PAGES = 2  # how many workers / pages at once
GIGGLE_TIME = 100
PLAYERS_PER_PAGE = 10     # after this many players, close tab and open a new one


async def clear_vignette(page: Page) -> None:
    url = page.url

    if "vignette" in url or "googleads" in url:
        print("Vignette detected, going back...")
        await page.go_back()
        await page.wait_for_timeout(300)
        return

    await page.keyboard.press("Escape")
    await page.wait_for_timeout(100)

    for f in page.frames:
        try:
            btn = f.locator("button:has-text('Continue'), button:has-text('Skip')")
            if await btn.count() > 0:
                await btn.first.click()
                print("Closed vignette popup")
                return
        except Exception:
            pass


async def scrape(page: Page) -> list[dict]:
    print("Scraping page content...")

    battle_data: list[dict] = []

    menus = page.locator("div.ui.text.fluid.menu.battle_bottom_menu")
    count = await menus.count()
    print("Found menus:", count)

    for i in range(count):
        menu = menus.nth(i)

        ts_div = menu.locator("div.item.i18n_duration_short.battle-timestamp-popup")
        timestamp = await ts_div.get_attribute("data-content")

        replay_button = menu.locator(
            "button.ui.orange.circular.icon.button.button_popup.replay_button"
        )

        if await replay_button.count() == 0:
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

    print("Finished scraping. Collected battles:", len(battle_data))
    return battle_data


async def scroll_until_loader_hidden(page: Page, verbose: bool = False) -> None:
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

        await page.mouse.wheel(0, random.randint(SCROLL_UP_AMOUNT, 0))
        await page.wait_for_timeout(GIGGLE_TIME)
        if verbose:
            print("Scrolled a bit to trigger more loading.")


async def paginate_battles(page: Page, verbose: bool = False) -> list[dict]:
    battle_data: list[dict] = []

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


async def scrape_player_on_page(
    page: Page,
    pid: str,
    verbose: bool = False,
) -> list[dict]:
    """Scrape all battles for a single player id using an existing page."""
    try:
        print(f"[{pid}] Starting scrape")
        url = f"https://royaleapi.com/player/{pid}/battles/"
        await page.goto(url)
        await clear_vignette(page)

        await scroll_until_loader_hidden(page, verbose=verbose)
        cur_data = await paginate_battles(page, verbose=verbose)

        for row in cur_data:
            row["player_id"] = pid

        print(f"[{pid}] Finished with {len(cur_data)} battles")
        return cur_data
    except Exception as e:
        error_message = f"[{pid}] Error: {e}\n"
        with open(ERROR_LOG_PATH, "a", encoding="utf8") as f:
            f.write(error_message)
        print(error_message)
        return []


def find_biggest_chunk(folder: str) -> int:
    pattern = re.compile(r"battles_chunk_(\d+)\.csv")
    max_num = 0

    if not os.path.isdir(folder):
        return 0

    for filename in os.listdir(folder):
        match = pattern.match(filename)
        if match:
            num = int(match.group(1))
            if num > max_num:
                max_num = num
    return max_num


async def player_worker(
    context,
    player_ids: list[str],
    worker_index: int,
    verbose: bool = False,
) -> list[dict]:
    """Worker that reuses a page for up to PLAYERS_PER_PAGE players, then recreates it."""
    results: list[dict] = []
    if not player_ids:
        return results

    page = await context.new_page()
    processed_on_this_page = 0

    try:
        for pid in player_ids:
            print(f"[Worker {worker_index}] Processing {pid} (on this page: {processed_on_this_page})")

            data = await scrape_player_on_page(page, pid, verbose=verbose)
            if data:
                results.extend(data)

            processed_on_this_page += 1

            if processed_on_this_page >= PLAYERS_PER_PAGE:
                print(f"[Worker {worker_index}] Reached {PLAYERS_PER_PAGE} players, recreating page")
                await page.close()
                page = await context.new_page()
                processed_on_this_page = 0
    finally:
        await page.close()

    return results


def split_for_workers(ids: list[str], max_workers: int) -> list[list[str]]:
    """Split ids into up to max_workers lists in round robin fashion."""
    n_workers = min(max_workers, len(ids))
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    for idx, pid in enumerate(ids):
        buckets[idx % n_workers].append(pid)
    return buckets

def get_fraction(df, i, n):
    """
    Return the i-th fraction (0 indexed) out of n equal parts of a DataFrame.
    """
    total = len(df)
    chunk = total // n
    start = i * chunk
    end = (i + 1) * chunk if i < n - 1 else total
    return df.iloc[start:end]


async def run() -> None:
    # Make sure battle_chunks directory exists
    os.makedirs(BATTLE_CHUNKS_DIR, exist_ok=True)

    df_players = pd.read_csv(PLAYERS_CSV)
    df_players = get_fraction(df_players, INSTANCE_ID, TOTAL_INSTANCES)

    if "player_tag" not in df_players.columns:
        raise ValueError("CSV must contain a 'player_tag' column")

    df_players["player_tag"] = df_players["player_tag"].astype(str).str.replace("#", "")
    player_ids = df_players["player_tag"].tolist()
    print(f"Loaded {len(player_ids)} player IDs")

    number_of_files = find_biggest_chunk(BATTLE_CHUNKS_DIR)
    print(f"Starting from {number_of_files} chunk files")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        # uncomment for login
        # context = await browser.new_context(storage_state=STORAGE_STATE_PATH)
        context = await browser.new_context()

        chunk_index = number_of_files + 1

        for start in range(number_of_files * CHUNK_SIZE, len(player_ids), CHUNK_SIZE):
            chunk_ids = player_ids[start:start + CHUNK_SIZE]
            print(f"Processing players {start} to {start + len(chunk_ids) - 1}")

            # Split this chunk among workers
            worker_id_lists = split_for_workers(chunk_ids, MAX_CONCURRENT_PAGES)

            tasks = [
                player_worker(context, pid_list, worker_index=i, verbose=False)
                for i, pid_list in enumerate(worker_id_lists)
                if pid_list  # skip empty lists
            ]

            worker_results = await asyncio.gather(*tasks)

            battle_rows = [
                row
                for worker_rows in worker_results
                if worker_rows
                for row in worker_rows
            ]

            filename = os.path.join(BATTLE_CHUNKS_DIR, f"battles_chunk_{chunk_index}.csv")
            if battle_rows:
                pd.DataFrame(battle_rows).to_csv(filename, index=False)
                print(f"Saved chunk file: {filename} with {len(battle_rows)} rows")
            else:
                print(f"No data in chunk {chunk_index}, nothing saved")

            chunk_index += 1

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
