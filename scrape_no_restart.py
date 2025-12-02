import asyncio
import json
from typing import List, Dict
import os
from bs4 import BeautifulSoup
import pandas as pd
from playwright.async_api import (
    async_playwright,
    Browser,
    BrowserContext,
    Page,
)

WORKER_ID = 0
TOTAL_WORKERS = 10

STORAGE_STATE_PATH = "myGoogleAuth.json"
REPLAY_PATH = "battle_chunks/all_battles_combined_dedup.csv"
MAX_WORKERS = 5
BATCH_SIZE = 100


BASE_URL = "https://royaleapi.com/data/replay?tag={tag}"
OUTPUT_CSV = "replay_full_events_all.csv"

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


# =========================================================
# Parse a single HTML fragment (from JSON["html"])
# =========================================================
def parse_replay_html(html: str, replay_tag: str) -> List[Dict]:
    soup = BeautifulSoup(html, "html.parser")

    card_events = []
    for img in soup.select("div.replay_team img.replay_card"):
        card_events.append(
            {
                "card_name": img.get("data-card"),
                "image_src": img.get("src"),
                "side": img.get("data-s"),
                "time": img.get("data-t"),
                "isAbility": img.get("data-ability"),
            }
        )

    card_events = sorted(card_events, key=lambda x: int(x["time"]))

    placement_events = []
    for div in soup.select("div.markers > div"):
        placement_events.append(
            {
                "x": div.get("data-x"),
                "y": div.get("data-y"),
                "time": div.get("data-t"),
            }
        )

    if len(card_events) != len(placement_events):
        raise ValueError(
            f"Length mismatch for replay {replay_tag} "
            f"(cards {len(card_events)} vs placements {len(placement_events)})"
        )

    events = []
    for i in range(len(card_events)):
        ce = card_events[i]
        pe = placement_events[i]

        if ce["time"] != pe["time"]:
            raise ValueError(
                f"Time mismatch for replay {replay_tag} at index {i}: "
                f"{ce['time']} vs {pe['time']}"
            )

        row = {
            "replay_tag": replay_tag,
            "side": ce["side"],
            "time": ce["time"],
            "isAbility": ce["isAbility"],
        }

        if ce["isAbility"] == "1":
            row["card_name"] = (
                "ability-" + ce["image_src"].split("ability-")[-1].split(".png")[0]
            )
            row["x"] = None
            row["y"] = None
        else:
            row["card_name"] = ce["card_name"]
            row["x"] = pe["x"]
            row["y"] = pe["y"]

        events.append(row)

    return events


# =========================================================
# Stealth context helper
# =========================================================
async def create_stealth_context(
    browser: Browser,
    storage_state: str | None = None,
) -> BrowserContext:
    kwargs: Dict = {
        "user_agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/121.0.0.0 Safari/537.36"
        ),
        "viewport": {"width": 1280, "height": 720},
    }
    if storage_state is not None:
        kwargs["storage_state"] = storage_state

    context = await browser.new_context(**kwargs)
    await context.add_init_script(STEALTH_INIT_SCRIPT)
    return context


# =========================================================
# Manual login step (uses visible browser)
# =========================================================
async def manual_login_and_save_state(browser: Browser) -> None:
    context = await create_stealth_context(browser)
    page = await context.new_page()
    await page.goto("https://royaleapi.com/", wait_until="networkidle")

    print("Please log into RoyaleAPI in the opened browser.")
    input("Press ENTER here in the terminal after you are fully logged in.\n")

    await context.storage_state(path=STORAGE_STATE_PATH)
    await context.close()
    print(f"Saved login state to {STORAGE_STATE_PATH}")


# =========================================================
# Fetch a single replay using an existing page
# =========================================================
async def fetch_replay_with_page(page: Page, tag: str) -> pd.DataFrame:
    url = BASE_URL.format(tag=tag)
    await page.goto(url, wait_until="networkidle", timeout=5000)

    body_text = await page.text_content("body")
    if body_text is None:
        raise ValueError(f"No body text for {tag}")

    try:
        data = json.loads(body_text)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Failed to decode JSON for {tag}: {e}\nBody (truncated): {body_text[:200]}"
        )

    if not data.get("success"):
        raise ValueError(f"Replay API returned success=false for {tag}")

    html_fragment = data.get("html")
    if not html_fragment:
        raise ValueError(f"No 'html' field in JSON for {tag}")

    events = parse_replay_html(html_fragment, tag)
    print(f"[Worker] {tag}: {len(events)} events scraped")

    return pd.DataFrame(events)

    
# =========================================================
# Worker that pulls tags from a queue
# - Reuses one tab
# - After 10 replays, closes tab + context and recreates
# =========================================================
async def worker_task(
    name: int,
    browser: Browser,
    queue: asyncio.Queue,
    results: Dict[str, pd.DataFrame],
    batch_size: int,
    total_tags: int,
    overall_completed_ref: Dict[str, int],
    batch_completed_ref: Dict[str, int],
) -> None:
    print(f"[Worker-{name}] Starting")

    context = await create_stealth_context(browser, storage_state=STORAGE_STATE_PATH)
    page = await context.new_page()
    local_count = 0

    try:
        while True:
            tag = await queue.get()
            if tag is None:
                queue.task_done()
                print(f"[Worker-{name}] Received sentinel, exiting")
                break

            try:
                df = await fetch_replay_with_page(page, tag)
                results[tag] = df
            except Exception as e:
                print(f"[Worker-{name}] [ERROR] Failed {tag}: {e}")

            # Update progress
            batch_completed_ref["count"] += 1
            overall_completed_ref["count"] += 1

            batch_done = batch_completed_ref["count"]
            overall_done = overall_completed_ref["count"]

            print(
                f"[Progress] Worker {name} - "
                f"Overall {overall_done}/{total_tags} "
                f"({overall_done / total_tags:.2%}) | "
                f"Current batch {batch_done}/{batch_size}"
            )

            local_count += 1
            queue.task_done()

            # After 10 replays in this worker, recreate context + page
            if local_count >= 10:
                print(f"[Worker-{name}] Processed 10 replays, recreating context and page")
                await page.close()
                await context.close()
                context = await create_stealth_context(
                    browser,
                    storage_state=STORAGE_STATE_PATH,
                )
                page = await context.new_page()
                local_count = 0

    finally:
        try:
            await page.close()
        except Exception:
            pass
        try:
            await context.close()
        except Exception:
            pass
        print(f"[Worker-{name}] Fully shut down")

def split_tags():
    assert 0 <= WORKER_ID < TOTAL_WORKERS
    replay_tags = pd.read_csv(REPLAY_PATH)["replay_id"].tolist()
    total_tags = len(replay_tags)

    seg_length = total_tags // TOTAL_WORKERS
    start_idx = WORKER_ID * seg_length

    # Last worker should take all remaining tags
    if WORKER_ID == TOTAL_WORKERS - 1:
        worker_tags = replay_tags[start_idx:]
    else:
        worker_tags = replay_tags[start_idx:start_idx + seg_length]
    return worker_tags

def reach_current(replay_tags):
    def tail(path):
        last = None
        with open(path) as f:
            for last in f:
                pass
        return last

    if os.path.exists(OUTPUT_CSV):
        last_line = tail(OUTPUT_CSV)
        if last_line:
            final_id = last_line.split(",")[0].strip()
            try:
                idx = replay_tags.index(final_id)
                replay_tags = replay_tags[idx+1:]
            except ValueError:
                pass  # final_id not found
    return replay_tags

# =========================================================
# Run all workers with batch saving every 500 battles
# =========================================================
async def main():
    # first get the section its suppose to get
    replay_tags = split_tags()
    print("Worker total workload: ", len(replay_tags))
    old_len = len(replay_tags)
    # there is already events scraped, so resuming
    replay_tags = reach_current(replay_tags)
    total_tags = len(replay_tags)
    print("Already finished:", 1-(total_tags/old_len), "%,", total_tags,"out of", old_len)

    async with async_playwright() as pw:
        # # First browser: visible, for manual login
        # login_browser = await pw.chromium.launch(
        #     headless=False,
        #     args=STEALTH_ARGS,
        # )
        # await manual_login_and_save_state(login_browser)
        # await login_browser.close()

        # Second browser: headless, for scraping with saved state
        scrape_browser = await pw.chromium.launch(
            headless=True,
            args=STEALTH_ARGS,
        )

        first_write = True
        total_rows_written = 0

        # Global progress counter across all batches
        overall_completed_ref = {"count": 0}

        for start in range(0, total_tags, BATCH_SIZE):
            batch_tags = replay_tags[start:start + BATCH_SIZE]
            batch_size = len(batch_tags)
            print(
                f"\n=== Processing batch {start} to {start + batch_size - 1} "
                f"({batch_size} replays) ==="
            )

            # Queue and shared structures for this batch
            queue: asyncio.Queue = asyncio.Queue()
            results: Dict[str, pd.DataFrame] = {}
            batch_completed_ref = {"count": 0}

            # Enqueue tags
            for tag in batch_tags:
                await queue.put(tag)

            # Add sentinel items to stop workers
            for _ in range(MAX_WORKERS):
                await queue.put(None)

            # Start workers
            workers = [
                asyncio.create_task(
                    worker_task(
                        name=i,
                        browser=scrape_browser,
                        queue=queue,
                        results=results,
                        batch_size=batch_size,
                        total_tags=total_tags,
                        overall_completed_ref=overall_completed_ref,
                        batch_completed_ref=batch_completed_ref,
                    )
                )
                for i in range(MAX_WORKERS)
            ]

            # Wait for queue to be fully processed
            await queue.join()

            # Ensure workers exit
            for w in workers:
                await w

            if not results:
                print("No successful data in this batch, skipping write.")
                continue

            # Combine batch results in the original order of batch_tags
            batch_frames = [
                results[tag]
                for tag in batch_tags
                if tag in results
            ]

            if not batch_frames:
                print("No frames after filtering by tag order, skipping write.")
                continue

            combined_batch = pd.concat(batch_frames, ignore_index=True)
            mode = "w" if first_write else "a"
            header = first_write

            combined_batch.to_csv(OUTPUT_CSV, index=False, mode=mode, header=header)
            total_rows_written += len(combined_batch)
            first_write = False

            print(
                f"[Batch] Written {len(combined_batch)} rows from this batch "
                f"(total rows so far {total_rows_written})"
            )

        await scrape_browser.close()

        if total_rows_written == 0:
            print("No data scraped at all.")
        else:
            print(f"\nAll done. Total rows written to {OUTPUT_CSV}: {total_rows_written}")


if __name__ == "__main__":
    asyncio.run(main())
