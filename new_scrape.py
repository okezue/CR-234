import asyncio
import pandas as pd
import os
import io
import shutil
import argparse
from pathlib import Path
from datetime import datetime

from playwright.async_api import async_playwright

# Change this to wherever you want your root directory
ROOT_DIR = os.path.abspath(".")  # for example: "/home/ubuntu/clash_bot"

PLAYERS_CSV = os.path.join(ROOT_DIR, "big_data/all_players_from_clans.csv")
BATTLE_CHUNKS_DIR_BASE = "scraped_data/battle_chunks"
STORAGE_STATE_PATH = os.path.join(ROOT_DIR, "myGoogleAuth.json")
ERROR_LOG_PATH_BASE = "error_log.txt"
SCRAPED_PLAYERS_FILE_BASE = "scraped_players.txt"
BATTLE_META_CSV_BASE = "scraped_data/battle_meta_data.csv"

MAX_CONCURRENT_WORKERS = 5  # Number of concurrent workers
PLAYERS_PER_PAGE = 7        # Reuse page for this many players before recreating
DEBUG_MODE = False
CONTINUE_FROM_PREV_SCRAPE = True  # True continues where you left off via SCRAPED_PLAYERS_FILE
DEBUG_PLAYER_ID = "UG9RGJ20P"     # Set this to a specific player ID for debug mode
CLEAN_PAST_DATA = False           # Set to True to clean all past scraped data before starting
SCRAPE_TIMEOUT_SECONDS = 360000   # Hard cap for whole program (seconds)
PAGE_TIMEOUT_MS = 10000           # Timeout for individual webpage loads (ms)

# Restart the WHOLE BROWSER WINDOW every ~1 hour
BROWSER_RESTART_SECONDS = 30
BROWSER_RESTART_GRACE_SECONDS = 30


def is_ad(url: str) -> bool:
    bad = ["doubleclick", "googlesyndication", "adsystem", "adservice", "taboola", "outbrain"]
    return any(x in url for x in bad)


def elapsed_str(start_time: datetime) -> str:
    delta = datetime.now() - start_time
    total = int(delta.total_seconds())
    h = total // 3600
    m = (total % 3600) // 60
    s = total % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def split_for_workers(ids: list[str], max_workers: int) -> list[list[str]]:
    """Split ids into up to max_workers lists in round robin fashion."""
    if not ids:
        return []
    n_workers = min(max_workers, len(ids))
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    for idx, pid in enumerate(ids):
        buckets[idx % n_workers].append(pid)
    return buckets


def remaining_player_ids(all_player_ids: list[str], scraped_file: str) -> list[str]:
    """Return players not yet scraped, based on SCRAPED_PLAYERS_FILE."""
    if not os.path.exists(scraped_file):
        return all_player_ids

    scraped = set()
    with open(scraped_file, "r", encoding="utf8") as f:
        for line in f:
            pid = line.strip()
            if pid:
                scraped.add(pid)

    return [pid for pid in all_player_ids if pid not in scraped]


async def scrape_battles(page, pid) -> list[dict]:
    # Check if the page has the expected battle container
    container = page.locator("#scrolling_battle_container .ui.container.sidemargin0.battle_list_container")
    if await container.count() == 0:
        print(f"No battle container found for player {pid}, skipping")
        return []

    # Find all replay buttons
    buttons = page.locator("button.replay_button")
    count = await buttons.count()
    if count == 0:
        print(f"No replay buttons found for player {pid}, skipping")
        return []

    # Collect data-divs and click buttons
    battles = {}
    for i in range(count):
        button = buttons.nth(i)
        data_div = await button.get_attribute("data-div")
        battle_id = await button.get_attribute("data-replay")
        await button.click()
        await page.locator(f"#{data_div}").wait_for(timeout=PAGE_TIMEOUT_MS)
        battles[battle_id] = data_div

    # Extract data from all loaded replays
    results = []
    for battle_id, data_div in battles.items():
        container = page.locator(f"#{data_div}")
        await container.wait_for(timeout=PAGE_TIMEOUT_MS)
        await container.locator(".ui.info.message, .battle_replay").first.wait_for(
            state="visible",
            timeout=PAGE_TIMEOUT_MS,
        )

        error_msg = container.locator(".ui.info.message")
        replay_content = container.locator(".battle_replay")

        if await error_msg.is_visible():
            txt = (await error_msg.inner_text()).lower()
            if "replay not found" in txt:
                continue

        if not await replay_content.is_visible():
            continue

        # Extract markers
        markers_locator = page.locator(f"#{data_div} .markers .marker")
        marker_count = await markers_locator.count()
        marker_list = []
        for j in range(marker_count):
            marker = markers_locator.nth(j)
            x = await marker.get_attribute("data-x")
            y = await marker.get_attribute("data-y")
            t = await marker.get_attribute("data-t")
            s = await marker.get_attribute("data-s")
            span = marker.locator("span")
            number = await span.text_content()
            classes = await marker.get_attribute("class")
            team = "red" if classes and "red" in classes else "blue"
            marker_list.append(
                {
                    "x": int(x) if x and x != "None" else None,
                    "y": int(y) if y and y != "None" else None,
                    "t": int(t) if t and t != "None" else None,
                    "s": s,
                    "number": int(number) if number and number != "None" else None,
                    "team": team,
                }
            )

        # Extract replay_cards
        replay_cards_locator = page.locator(f"#{data_div} .replay_team img.replay_card")
        card_count = await replay_cards_locator.count()
        card_list = []
        for j in range(card_count):
            card = replay_cards_locator.nth(j)
            card_name = await card.get_attribute("src")
            card_name = card_name.split("/")[-1].split(".")[0] if card_name else None
            t = await card.get_attribute("data-t")
            s = await card.get_attribute("data-s")
            ability = await card.get_attribute("data-ability")
            card_list.append(
                {
                    "card": card_name,
                    "t": int(t) if t and t != "None" else None,
                    "s": s,
                    "ability": int(ability) if ability and ability != "None" else None,
                }
            )

        marker_list.sort(key=lambda m: m["t"] or 0)
        card_list.sort(key=lambda c: c["t"] or 0)

        assert len(marker_list) == len(card_list), (
            f"Mismatch in {battle_id}: markers {len(marker_list)}, cards {len(card_list)}"
        )

        for marker, card in zip(marker_list, card_list):
            results.append(
                {
                    "battle_id": battle_id,
                    "x": marker["x"],
                    "y": marker["y"],
                    "card": card["card"],
                    "time": marker["t"],
                    "side": marker["s"],
                    "team": marker["team"],
                }
            )

    # Download battle meta data CSV and filter for scraped battles
    if battles:
        csv_url = f"https://royaleapi.com/player/{pid}/battles/csv"
        resp = await page.request.get(csv_url)
        if not resp.ok:
            body = await resp.text()
            raise RuntimeError(f"CSV request failed: {resp.status} {resp.status_text}\n{body[:300]}")

        csv_text = await resp.text()
        meta_df = pd.read_csv(io.StringIO(csv_text))
        meta_df = meta_df[meta_df["replayTag"].astype(str).isin(["#" + key for key in battles.keys()])]

        if not meta_df.empty:
            meta_exists = os.path.exists(BATTLE_META_CSV)
            meta_df.to_csv(BATTLE_META_CSV, mode="a", header=not meta_exists, index=False)

    return results


async def worker(context, player_ids: list[str], worker_index: int, run_id: str, start_time: datetime):
    """Worker that processes a list of player IDs, reusing pages."""
    processed_on_this_page = 0
    filename = os.path.join(
        BATTLE_CHUNKS_DIR,
        f"{run_id}_worker_{worker_index}_results.csv" if run_id else f"worker_{worker_index}_results.csv",
    )
    file_exists = os.path.exists(filename)

    page = await context.new_page()
    await page.route("**/*", lambda route: route.abort() if is_ad(route.request.url) else route.continue_())

    try:
        for pid in player_ids:
            url = f"https://royaleapi.com/player/{pid}/battles/"
            try:
                await page.goto(url, wait_until="commit", timeout=PAGE_TIMEOUT_MS)

                element_locator = page.locator("div.ui.container.sidemargin0.battle_list_container")
                element_task = asyncio.create_task(element_locator.wait_for(timeout=PAGE_TIMEOUT_MS))
                load_task = asyncio.create_task(page.wait_for_load_state("load"))

                done, pending = await asyncio.wait(
                    [element_task, load_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=PAGE_TIMEOUT_MS,
                )

                for task in pending:
                    task.cancel()

                if element_task in done:
                    load_task.cancel()
                    print(f"{elapsed_str(start_time)} [Worker {worker_index}] Opened page for player {pid}: {url}")
                elif load_task in done:
                    element_task.cancel()
                    count = await element_locator.count()
                    if count == 0:
                        print(
                            f"{elapsed_str(start_time)} [Worker {worker_index}] "
                            f"Page loaded but no battle container for {pid}, skipping"
                        )
                        continue
                    print(f"{elapsed_str(start_time)} [Worker {worker_index}] Opened page for player {pid}: {url}")
                else:
                    print(
                        f"{elapsed_str(start_time)} [Worker {worker_index}] "
                        f"Timeout waiting for page/element for {pid}, skipping"
                    )
                    continue

                data = await scrape_battles(page, pid)
                for row in data:
                    row["player_id"] = pid

                if data:
                    os.makedirs(BATTLE_CHUNKS_DIR, exist_ok=True)
                    df_to_save = pd.DataFrame(data)
                    df_to_save.to_csv(filename, mode="a", header=not file_exists, index=False)
                    print(
                        f"{elapsed_str(start_time)} [Worker {worker_index}] "
                        f"Appended {len(data)} rows for {pid} to {filename}"
                    )
                    file_exists = True

                    with open(SCRAPED_PLAYERS_FILE, "a", encoding="utf8") as f:
                        f.write(f"{pid}\n")

            except asyncio.CancelledError:
                raise
            except Exception as e:
                error_message = f"Timeout for player {pid}: {e}\n"
                with open(ERROR_LOG_PATH, "a", encoding="utf8") as f:
                    f.write(error_message)
                print(f"{elapsed_str(start_time)} [Worker {worker_index}] {error_message.strip()}")

            processed_on_this_page += 1
            if processed_on_this_page >= PLAYERS_PER_PAGE:
                print(f"{elapsed_str(start_time)} [Worker {worker_index}] Reached {PLAYERS_PER_PAGE} players, recreating page")
                await page.close()
                page = await context.new_page()
                await page.route("**/*", lambda route: route.abort() if is_ad(route.request.url) else route.continue_())
                processed_on_this_page = 0
    finally:
        try:
            await page.close()
        except Exception:
            pass


async def main():
    parser = argparse.ArgumentParser(description="Scrape Clash Royale battle data.")
    parser.add_argument("--id", help="Run ID to append to output files (optional)")
    args = parser.parse_args()
    run_id = args.id or ""

    start_time = datetime.now()

    # Set paths with run_id
    global BATTLE_CHUNKS_DIR, ERROR_LOG_PATH, SCRAPED_PLAYERS_FILE, BATTLE_META_CSV
    BATTLE_CHUNKS_DIR = os.path.join(ROOT_DIR, f"{run_id}_{BATTLE_CHUNKS_DIR_BASE}" if run_id else BATTLE_CHUNKS_DIR_BASE)
    ERROR_LOG_PATH = os.path.join(ROOT_DIR, f"{run_id}_{ERROR_LOG_PATH_BASE}" if run_id else ERROR_LOG_PATH_BASE)
    SCRAPED_PLAYERS_FILE = os.path.join(ROOT_DIR, f"{run_id}_{SCRAPED_PLAYERS_FILE_BASE}" if run_id else SCRAPED_PLAYERS_FILE_BASE)
    BATTLE_META_CSV = os.path.join(ROOT_DIR, f"{run_id}_{BATTLE_META_CSV_BASE}" if run_id else BATTLE_META_CSV_BASE)

    if CLEAN_PAST_DATA:
        if os.path.exists(BATTLE_CHUNKS_DIR):
            shutil.rmtree(BATTLE_CHUNKS_DIR)
        if os.path.exists(BATTLE_META_CSV):
            os.remove(BATTLE_META_CSV)
        if os.path.exists(SCRAPED_PLAYERS_FILE):
            os.remove(SCRAPED_PLAYERS_FILE)
        print("Past data cleaned.")

    df_players = pd.read_csv(PLAYERS_CSV)
    if "player_tag" not in df_players.columns:
        raise ValueError("CSV must contain a 'player_tag' column")

    df_players["player_tag"] = df_players["player_tag"].astype(str).str.replace("#", "", regex=False)
    all_player_ids = df_players["player_tag"].tolist()

    if DEBUG_MODE:
        if not DEBUG_PLAYER_ID:
            print("Debug mode enabled but DEBUG_PLAYER_ID not set")
            return
        all_player_ids = [DEBUG_PLAYER_ID]
        print(f"Debug mode: scraping only {DEBUG_PLAYER_ID}")

    # Optional: initial print
    initial_remaining = remaining_player_ids(all_player_ids, SCRAPED_PLAYERS_FILE) if CONTINUE_FROM_PREV_SCRAPE else all_player_ids
    print(f"Loaded {len(initial_remaining)} player IDs to scrape (remaining)")

    async with async_playwright() as p:
        session_idx = 0
        overall_deadline = asyncio.get_running_loop().time() + SCRAPE_TIMEOUT_SECONDS

        while True:
            now = asyncio.get_running_loop().time()
            if now >= overall_deadline:
                print(f"Hard timeout reached after {SCRAPE_TIMEOUT_SECONDS} seconds. Exiting.")
                break

            # Recompute remaining players each session
            if CONTINUE_FROM_PREV_SCRAPE:
                player_ids = remaining_player_ids(all_player_ids, SCRAPED_PLAYERS_FILE)
            else:
                player_ids = all_player_ids

            if not player_ids:
                print("All players scraped (or no players to scrape). Done.")
                break

            session_idx += 1
            print(f"\n=== Session {session_idx}: launching fresh browser window for up to {BROWSER_RESTART_SECONDS}s ===")
            print(f"Session {session_idx}: {len(player_ids)} players remaining")

            browser = await p.chromium.launch(headless=False)
            context = await browser.new_context(storage_state=STORAGE_STATE_PATH, accept_downloads=True)

            worker_id_lists = split_for_workers(player_ids, MAX_CONCURRENT_WORKERS)
            tasks = [
                asyncio.create_task(worker(context, pid_list, worker_index=i, run_id=run_id, start_time=start_time))
                for i, pid_list in enumerate(worker_id_lists)
                if pid_list
            ]

            try:
                remaining_overall = max(1, int(overall_deadline - asyncio.get_running_loop().time()))
                session_timeout = min(BROWSER_RESTART_SECONDS, remaining_overall)

                await asyncio.wait_for(asyncio.gather(*tasks), timeout=session_timeout)
                print(f"Session {session_idx} finished before restart timer.")
            except asyncio.TimeoutError:
                print(f"Session {session_idx} reached {BROWSER_RESTART_SECONDS}s (or overall cap). Restarting browser window...")
            except Exception as e:
                print(f"Session {session_idx} crashed with error: {e}. Restarting browser window...")
            finally:
                # Cancel tasks and wait for them to finish
                for t in tasks:
                    if not t.done():
                        t.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)

                try:
                    await context.close()
                except Exception:
                    pass
                try:
                    await browser.close()
                except Exception:
                    pass

            await asyncio.sleep(BROWSER_RESTART_GRACE_SECONDS)


if __name__ == "__main__":
    asyncio.run(main())
