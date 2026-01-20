import asyncio
import pandas as pd
import os
import io
import shutil
import argparse
from pathlib import Path

from playwright.async_api import async_playwright

# Change this to wherever you want your root directory
ROOT_DIR = os.path.abspath(".")  # for example: "/home/ubuntu/clash_bot"

PLAYERS_CSV = os.path.join(ROOT_DIR, "big_data/all_players_from_clans.csv")
BATTLE_CHUNKS_DIR_BASE = "battle_chunks"
STORAGE_STATE_PATH = os.path.join(ROOT_DIR, "myGoogleAuth.json")
ERROR_LOG_PATH_BASE = "error_log.txt"
SCRAPED_PLAYERS_FILE_BASE = "scraped_players.txt"
BATTLE_META_CSV_BASE = "battle_meta_data.csv"

MAX_CONCURRENT_WORKERS = 5  # Number of concurrent workers
PLAYERS_PER_PAGE = 7     # Reuse page for this many players before recreating
DEBUG_MODE = False
DEBUG_PLAYER_ID = "P82R2QJY8"  # Set this to a specific player ID for debug mode
CLEAN_PAST_DATA = False  # Set to True to clean all past scraped data before starting
SCRAPE_TIMEOUT_SECONDS = 3600  # Timeout for the entire scraping process in seconds (e.g., 3600 = 1 hour)
async def scrape_battles(page, pid) -> list[dict]:
    
    # Check if the page has the expected battle container
    container = page.locator("#scrolling_battle_container .ui.container.sidemargin0.battle_list_container")
    if await container.count() == 0:
        print(f"No battle container found for player {pid}, skipping")
        return []
    
    # Find all replay buttons
    buttons = page.locator('button.replay_button')
    count = await buttons.count()
    if count == 0:
        print(f"No replay buttons found for player {pid}, skipping")
        return []
    
    # Collect data-divs and click buttons
    battles = {}
    for i in range(count):
        button = buttons.nth(i)
        data_div = await button.get_attribute('data-div')
        battle_id = await button.get_attribute('data-replay')
        await button.click()
        # Wait for the replay div to load
        await page.locator(f'#{data_div}').wait_for()
        
        # Check if replay loaded successfully
        replay_div = page.locator(f'#{data_div}')
        if await replay_div.get_by_text("Replay not found").count() > 0:
            continue
        battles[battle_id] = data_div
        
    # Now extract data from all loaded replays
    results = []
    for battle_id, data_div in battles.items():
        # Extract markers
        await page.locator(f'#{data_div} .markers').wait_for()
        markers_locator = page.locator(f'#{data_div} .markers .marker')
        marker_count = await markers_locator.count()
        marker_list = []
        for j in range(marker_count):
            marker = markers_locator.nth(j)
            x = await marker.get_attribute('data-x')
            y = await marker.get_attribute('data-y')
            t = await marker.get_attribute('data-t')
            s = await marker.get_attribute('data-s')
            span = marker.locator('span')
            number = await span.text_content()
            classes = await marker.get_attribute('class')
            team = 'red' if 'red' in classes else 'blue'
            marker_list.append({
                'x': int(x) if x and x != 'None' else None,
                'y': int(y) if y and y != 'None' else None,
                't': int(t) if t and t != 'None' else None,
                's': s,
                'number': int(number) if number and number != 'None' else None,
                'team': team
            })
        
        # Extract replay_cards
        replay_cards_locator = page.locator(f'#{data_div} .replay_team img.replay_card')
        card_count = await replay_cards_locator.count()
        card_list = []
        for j in range(card_count):
            card = replay_cards_locator.nth(j)
            card_name = await card.get_attribute('src')
            card_name = card_name.split("/")[-1].split(".")[0]
            t = await card.get_attribute('data-t')
            s = await card.get_attribute('data-s')
            ability = await card.get_attribute('data-ability')
            card_list.append({
                'card': card_name,
                't': int(t) if t and t != 'None' else None,
                's': s,
                'ability': int(ability) if ability and ability != 'None' else None
            })
        
        # Sort both by t
        marker_list.sort(key=lambda m: m['t'] or 0)
        card_list.sort(key=lambda c: c['t'] or 0)
        
        # Assert counts equal
        assert len(marker_list) == len(card_list), f"Mismatch in {battle_id}: markers {len(marker_list)}, cards {len(card_list)}"
        
        # Create results by zipping
        for marker, card in zip(marker_list, card_list):
            results.append({
                'battle_id': battle_id,
                'x': marker['x'],
                'y': marker['y'],
                'card': card['card'],
                'time': marker['t'],
                'side': marker['s'],
                'team': marker['team']
            })
        
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
            meta_df.to_csv(BATTLE_META_CSV, mode='a', header=not meta_exists, index=False)
    return results

def split_for_workers(ids: list[str], max_workers: int) -> list[list[str]]:
    """Split ids into up to max_workers lists in round robin fashion."""
    n_workers = min(max_workers, len(ids))
    buckets: list[list[str]] = [[] for _ in range(n_workers)]
    for idx, pid in enumerate(ids):
        buckets[idx % n_workers].append(pid)
    return buckets

async def worker(context, player_ids: list[str], worker_index: int, run_id: str):
    """Worker that processes a list of player IDs, reusing pages."""
    processed_on_this_page = 0
    filename = os.path.join(BATTLE_CHUNKS_DIR, f"{run_id}_worker_{worker_index}_results.csv" if run_id else f"worker_{worker_index}_results.csv")
    file_exists = os.path.exists(filename)

    page = await context.new_page()

    try:
        for pid in player_ids:
            url = f"https://royaleapi.com/player/{pid}/battles/"
            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30000)
                await page.locator('div.ui.container.sidemargin0.battle_list_container').wait_for()
                print(f"[Worker {worker_index}] Opened page for player {pid}: {url}")

            # Call the scraping function
                data = await scrape_battles(page, pid)
                for row in data:
                    row["player_id"] = pid

                # Save after each player if data was scraped
                if data:
                    os.makedirs(BATTLE_CHUNKS_DIR, exist_ok=True)
                    df_to_save = pd.DataFrame(data)
                    df_to_save.to_csv(filename, mode='a', header=not file_exists, index=False)
                    print(f"[Worker {worker_index}] Appended {len(data)} rows for player {pid} to {filename}")
                    file_exists = True
                    
                    # Mark player as scraped
                    with open(SCRAPED_PLAYERS_FILE, "a") as f:
                        f.write(f"{pid}\n")

            except Exception as e:
                error_message = f"Timeout for player {pid}: {e}\n"
                with open(ERROR_LOG_PATH, "a", encoding="utf8") as f:
                    f.write(error_message)
                print(f"[Worker {worker_index}] {error_message.strip()}")
                # Skip this player, continue to next

            processed_on_this_page += 1

            if processed_on_this_page >= PLAYERS_PER_PAGE:
                print(f"[Worker {worker_index}] Reached {PLAYERS_PER_PAGE} players, recreating page")
                await page.close()
                page = await context.new_page()
                processed_on_this_page = 0
    finally:
        await page.close()

async def main():
    parser = argparse.ArgumentParser(description='Scrape Clash Royale battle data.')
    parser.add_argument('--id', help='Run ID to append to output files (optional)')
    args = parser.parse_args()
    run_id = args.id or ''

    # Set paths with run_id
    global BATTLE_CHUNKS_DIR, ERROR_LOG_PATH, SCRAPED_PLAYERS_FILE, BATTLE_META_CSV
    BATTLE_CHUNKS_DIR = os.path.join(ROOT_DIR, f"{run_id}_{BATTLE_CHUNKS_DIR_BASE}" if run_id else BATTLE_CHUNKS_DIR_BASE)
    ERROR_LOG_PATH = os.path.join(ROOT_DIR, f"{run_id}_{ERROR_LOG_PATH_BASE}" if run_id else ERROR_LOG_PATH_BASE)
    SCRAPED_PLAYERS_FILE = os.path.join(ROOT_DIR, f"{run_id}_{SCRAPED_PLAYERS_FILE_BASE}" if run_id else SCRAPED_PLAYERS_FILE_BASE)
    BATTLE_META_CSV = os.path.join(ROOT_DIR, f"{run_id}_{BATTLE_META_CSV_BASE}" if run_id else BATTLE_META_CSV_BASE)

    if CLEAN_PAST_DATA:
        # Clean past data
        if os.path.exists(BATTLE_CHUNKS_DIR):
            shutil.rmtree(BATTLE_CHUNKS_DIR)
        if os.path.exists(BATTLE_META_CSV):
            os.remove(BATTLE_META_CSV)
        if os.path.exists(SCRAPED_PLAYERS_FILE):
            os.remove(SCRAPED_PLAYERS_FILE)
        print("Past data cleaned.")
    
    # Read player CSV
    df_players = pd.read_csv(PLAYERS_CSV)
    if "player_tag" not in df_players.columns:
        raise ValueError("CSV must contain a 'player_tag' column")

    df_players["player_tag"] = df_players["player_tag"].astype(str).str.replace("#", "")
    player_ids = df_players["player_tag"].tolist()
    
    # Load scraped players to skip
    if os.path.exists(SCRAPED_PLAYERS_FILE):
        with open(SCRAPED_PLAYERS_FILE, "r") as f:
            scraped = set(line.strip() for line in f)
        player_ids = [pid for pid in player_ids if pid not in scraped]
        print(f"Skipping {len(scraped)} already scraped players")
    
    if DEBUG_MODE:
        if DEBUG_PLAYER_ID:
            player_ids = [DEBUG_PLAYER_ID]
            print(f"Debug mode: scraping only {DEBUG_PLAYER_ID}")
        else:
            print("Debug mode enabled but DEBUG_PLAYER_ID not set")
            return
    
    print(f"Loaded {len(player_ids)} player IDs to scrape")

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)  # Set to True for production
        context = await browser.new_context(storage_state=STORAGE_STATE_PATH, accept_downloads=True)

        # Split players among workers
        worker_id_lists = split_for_workers(player_ids, MAX_CONCURRENT_WORKERS)

        tasks = [
            worker(context, pid_list, worker_index=i, run_id=run_id)
            for i, pid_list in enumerate(worker_id_lists)
            if pid_list  # skip empty lists
        ]

        try:
            await asyncio.wait_for(asyncio.gather(*tasks), timeout=SCRAPE_TIMEOUT_SECONDS)
        except asyncio.TimeoutError:
            print(f"Scraping timed out after {SCRAPE_TIMEOUT_SECONDS} seconds.")

        await browser.close()

if __name__ == "__main__":
    asyncio.run(main())