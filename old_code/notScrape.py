import csv
import time
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import quote
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any


API = "eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzUxMiIsImtpZCI6IjI4YTMxOGY3LTAwMDAtYTFlYi03ZmExLTJjNzQzM2M2Y2NhNSJ9.eyJpc3MiOiJzdXBlcmNlbGwiLCJhdWQiOiJzdXBlcmNlbGw6Z2FtZWFwaSIsImp0aSI6ImVhYWYwNzA5LTE4OTMtNGE5Ny1iOTQ5LWNiODMxYzI4YWE1YSIsImlhdCI6MTc2NDIyNTkxOSwic3ViIjoiZGV2ZWxvcGVyLzBjMmE4YmM5LTBiZTgtYTcyZi0xZjIxLWJjNGY5OGUzYmM5NyIsInNjb3BlcyI6WyJyb3lhbGUiXSwibGltaXRzIjpbeyJ0aWVyIjoiZGV2ZWxvcGVyL3NpbHZlciIsInR5cGUiOiJ0aHJvdHRsaW5nIn0seyJjaWRycyI6WyIxMjguMTIuMTIzLjY4Il0sInR5cGUiOiJjbGllbnQifV19.8AUI51jcfmiyaQVxsEJm5xazY7nrtcJNMehi21IAZPViABU6reDrhj0Qs5V8NSdsf3YyYRNKVimBBZ8sWr4cfQ"
BASE_URL = "https://api.clashroyale.com/v1"

HEADERS = {
    "Authorization": f"Bearer {API}"
}


def get_all_location_clans(location_id: int, csv_file: str, sleep: float = 0.2) -> int:
    """
    Fetch all Clash Royale clans belonging to a given location ID
    and save them to a CSV file.

    Parameters
    ----------
    location_id : int
        The numeric locationId from the Clash Royale API (e.g., 57000120 for Italy).
    csv_file : str
        Output filename where all clan rows will be written.
    sleep : float, optional
        Delay between paginated API requests to avoid throttling.

    Returns
    -------
    int
        The number of clans fetched and written to the CSV.
    """

    limit = 5000  # API accepts up to 100; higher is ignored but harmless
    all_clans = []
    after = None

    while True:
        params = {"locationId": location_id, "limit": limit}
        if after:
            params["after"] = after

        response = requests.get(f"{BASE_URL}/clans", headers=HEADERS, params=params)
        response.raise_for_status()
        data = response.json()

        # Add current page results
        items = data.get("items", [])
        all_clans.extend(items)

        print(f"[{location_id}] fetched {len(items)} clans, total: {len(all_clans)}")

        # Pagination cursor
        after = data.get("paging", {}).get("cursors", {}).get("after")
        if not after:
            break

        time.sleep(sleep)

    # --- Write CSV ---
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["tag", "name", "type", "clanScore", "members", "requiredTrophies"])

        for clan in all_clans:
            writer.writerow([
                clan.get("tag"),
                clan.get("name"),
                clan.get("type"),
                clan.get("clanScore"),
                clan.get("members"),
                clan.get("requiredTrophies"),
            ])

    print(f"Saved {len(all_clans)} clans → {csv_file}")
    return len(all_clans)


def combine_clan_csv_folder(folder_path: str, output_csv: str) -> None:
    """
    Combine all clan CSV files inside a folder into one master CSV.
    Each row gains a `country` column based on the filename.

    Parameters
    ----------
    folder_path : str
        The folder containing per-country CSV files.
    output_csv : str
        Path to the output master CSV file.

    Returns
    -------
    None
    """

    folder = Path(folder_path)
    dfs = []

    # Collect all CSV files in folder
    for csv_path in folder.glob("*.csv"):
        df = pd.read_csv(csv_path)

        # Extract country name from "Country_clans.csv"
        country_name = csv_path.stem.split("_clans")[0]
        df["country"] = country_name

        dfs.append(df)

    if not dfs:
        raise ValueError(f"No CSV files found in folder: {folder_path}")

    combined = pd.concat(dfs, ignore_index=True)
    combined.to_csv(output_csv, index=False)

    print(f"Combined {len(dfs)} files → {output_csv}")


# ---------------------------------------------------------------------
# Helper to append a failed clan to debug CSV
# ---------------------------------------------------------------------
def log_failed_clan(debug_csv: Path, clan_tag: str, country: str, error_msg: str) -> None:
    """
    Append one failed clan fetch to a debug CSV.

    Creates the file with a header if it does not exist.
    """
    file_exists = debug_csv.exists()
    with debug_csv.open("a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["clan_tag", "country", "error"])
        writer.writerow([clan_tag, country, error_msg])


# ---------------------------------------------------------------------
# API HELPER: fetch all members in a single clan (with retries)
# ---------------------------------------------------------------------
def fetch_clan_members(clan_tag: str, retries: int = 3, backoff: float = 1.5) -> list[dict]:
    """
    Fetch all members of a given clan using the Clash Royale API.

    Parameters
    ----------
    clan_tag : str
        The clan tag as stored in your CSV (usually with a leading '#').
    retries : int
        Number of times to retry on network / 5xx / 429 errors.
    backoff : float
        Exponential backoff base for retries (sleep = backoff ** attempt).

    Returns
    -------
    list[dict]
        A list of raw member dicts from the API (each with tag, expLevel,
        trophies, arena, etc.).

    Raises
    ------
    Exception
        If all retry attempts fail, the last exception is raised.
    """
    encoded_tag = quote(clan_tag, safe="")  # '#GPP2PUGG' -> '%23GPP2PUGG'
    url = f"{BASE_URL}/clans/{encoded_tag}/members"

    last_error: Exception | None = None

    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, params={"limit": 50}, timeout=10)
            # Explicitly treat 429 and 5xx as retryable
            if resp.status_code in (429,) or 500 <= resp.status_code < 600:
                raise requests.HTTPError(
                    f"HTTP {resp.status_code} from {url}", response=resp
                )
            resp.raise_for_status()
            data = resp.json()
            return data.get("items", [])
        except (requests.RequestException, ValueError) as e:
            last_error = e
            wait = backoff ** attempt
            print(
                f"[WARN] fetch_clan_members failed for {clan_tag} "
                f"(attempt {attempt}/{retries}): {e} – retrying in {wait:.1f}s"
            )
            time.sleep(wait)

    # All attempts failed
    assert last_error is not None
    raise last_error


# ---------------------------------------------------------------------
# MAIN PIPELINE: clans CSV -> giant players CSV (fail-safe)
# ---------------------------------------------------------------------
def build_player_dataset(
    clans_csv: str,
    output_csv: str,
    sleep: float = 0.15,
    max_clans: int | None = None,
    debug_csv: str = "failed_clans_debug.csv",
) -> None:
    """
    Build a giant player dataset by:
    - reading a CSV of clans (with columns including 'tag' and 'country')
    - calling /clans/{clanTag}/members for each clan
    - saving all players into one CSV

    If some clans fail (e.g. 404/403/network issues), they are logged to
    `debug_csv` and processing continues.

    Parameters
    ----------
    clans_csv : str
        Path to the CSV file that contains clan data. This file must have
        at least the columns:
            - 'tag'     : clan tag (e.g. "#GPP2PUGG")
            - 'country' : country name or code
    output_csv : str
        Path to the output CSV file that will contain all players.
    sleep : float, optional
        Delay (in seconds) between clan API calls to reduce rate limiting.
    max_clans : int or None, optional
        If set, limit processing to the first `max_clans` clans. Useful
        for testing on a subset before doing a full run.
    debug_csv : str, optional
        Path to a CSV file where failed clan fetches will be logged.

    Returns
    -------
    None
        Writes the combined player dataset to `output_csv`.
    """
    clans_df = pd.read_csv(clans_csv)

    required_cols = {"tag", "country"}
    missing = required_cols - set(clans_df.columns)
    if missing:
        raise ValueError(f"{clans_csv} is missing required columns: {missing}")

    records: list[dict] = []
    debug_path = Path(debug_csv)

    # Optionally limit number of clans (for debugging)
    if max_clans is not None:
        clans_df = clans_df.head(max_clans)

    total_clans = len(clans_df)
    for idx, row in clans_df.iterrows():
        clan_tag = str(row["tag"])
        country = str(row["country"])

        print(f"[{idx + 1}/{total_clans}] Fetching members for clan {clan_tag} ({country})")

        try:
            members = fetch_clan_members(clan_tag)
        except Exception as e:
            err_msg = str(e)
            print(f"[ERROR] Giving up on clan {clan_tag} ({country}): {err_msg}")
            log_failed_clan(debug_path, clan_tag, country, err_msg)
            # Skip this clan and continue with the next one
            time.sleep(sleep)
            continue

        for m in members:
            arena = m.get("arena", {}) or {}
            records.append(
                {
                    "player_tag": m.get("tag"),
                    "clan_tag": clan_tag,
                    "country": country,  # from the clan row
                    "expLevel": m.get("expLevel"),
                    "trophies": m.get("trophies"),
                    "arena_id": arena.get("id"),
                    "arena_name": arena.get("name"),
                }
            )

        # Be nice to the API between clans
        time.sleep(sleep)

    # Convert to DataFrame and save
    if not records:
        print("No player records collected. Check that your clan CSV and API token are correct.")
        return

    players_df = pd.DataFrame.from_records(records)
    players_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(players_df)} player rows to {output_csv}")
    if debug_path.exists():
        print(f"Some clans failed. See debug log: {debug_path}")

def flatten_json(obj: Any, parent_key: str = "", sep: str = ".") -> dict[str, Any]:
    """
    Recursively flattens a nested dict/list into a single-level dict.

    Examples:
      {"arena": {"id": 54000013, "name": "Executioner's Kitchen"}}
        -> {"arena.id": 54000013, "arena.name": "Executioner's Kitchen"}

      {"team": [{"tag": "#AAA"}, {"tag": "#BBB"}]}
        -> {"team[0].tag": "#AAA", "team[1].tag": "#BBB"}
    """
    items: dict[str, Any] = {}

    if isinstance(obj, dict):
        for k, v in obj.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            items.update(flatten_json(v, new_key, sep=sep))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            new_key = f"{parent_key}[{i}]"  # e.g. "team[0]"
            items.update(flatten_json(v, new_key, sep=sep))

    else:
        # base value
        if parent_key:  # ignore completely empty key
            items[parent_key] = obj

    return items

def build_battle_dataset(
    players_csv: str,
    output_csv: str,
    sleep: float = 0.15,
    max_players: int | None = None,
    debug_csv: str = "failed_players_debug.csv",
    max_workers: int = 8,   # NEW: number of concurrent player requests
) -> None:
    """
    Build a giant battle dataset by:
    - reading a CSV of players (at least 'player_tag'; optionally 'clan_tag' and 'country')
    - calling /players/{playerTag}/battlelog for each player (in parallel)
    - saving all battle logs into one CSV

    Parameters
    ----------
    players_csv : str
        Path to the CSV file that contains player data. This file must have
        at least the column:
            - 'player_tag': player tag (e.g. "#QC0RR20QC")
        If present, 'clan_tag' and 'country' will be propagated to the
        battle records.
    output_csv : str
        Path to the output CSV file that will contain all battles.
    sleep : float, optional
        Delay (in seconds) between retries / gentle backoff inside workers.
    max_players : int or None, optional
        If set, limit processing to the first `max_players` rows of players.
    debug_csv : str, optional
        Path to a CSV file where failed player fetches will be logged.
    max_workers : int, optional
        Maximum number of concurrent API calls (threads).

    Returns
    -------
    None
        Writes the combined battle dataset to `output_csv`.
    """
    players_df = pd.read_csv(players_csv)

    if "player_tag" not in players_df.columns:
        raise ValueError(f"{players_csv} must contain a 'player_tag' column")

    if max_players is not None:
        players_df = players_df.head(max_players)

    # Build a list of simple dicts so we don't have to pass full Series around
    tasks: list[dict[str, Any]] = []
    for _, row in players_df.iterrows():
        tasks.append(
            {
                "player_tag": str(row["player_tag"]).strip(),
                "clan_tag": row["clan_tag"] if "clan_tag" in players_df.columns else None,
                "country": row["country"] if "country" in players_df.columns else None,
            }
        )

    debug_path = Path(debug_csv)
    records: list[dict[str, Any]] = []
    failed_players: list[dict[str, Any]] = []

    total_players = len(tasks)

    def _process_player(task: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any] | None]:
        """
        Worker function for a single player.
        Returns (list_of_battle_records, failure_info_or_None).
        """
        player_tag = task["player_tag"]
        clan_tag = task["clan_tag"]
        country = task["country"]

        # URL-encode the player tag (handles leading '#')
        encoded_tag = quote(player_tag, safe="")

        try:
            url = f"{BASE_URL}/players/{encoded_tag}/battlelog"
            resp = requests.get(url, headers=HEADERS, timeout=10)

            if resp.status_code != 200:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:200]}")

            battles = resp.json()

        except Exception as e:
            err_msg = str(e)
            failure = {
                "entity_type": "player",
                "player_tag": player_tag,
                "clan_tag": clan_tag,
                "country": country,
                "error": err_msg,
            }
            time.sleep(sleep)
            return [], failure

        player_records: list[dict[str, Any]] = []

        for b in battles:
            # 1) flatten the entire battle JSON so every field becomes a column
            flat = flatten_json(b)

            # 2) still compute a handy "result" column based on crowns
            team = b.get("team") or []
            opponent = b.get("opponent") or []

            player_side = None
            for t in team:
                if t.get("tag") == player_tag:
                    player_side = t
                    break
            if player_side is None and team:
                player_side = team[0]

            opponent_side = opponent[0] if opponent else None

            crowns_for = player_side.get("crowns") if player_side else None
            crowns_against = opponent_side.get("crowns") if opponent_side else None

            if crowns_for is not None and crowns_against is not None:
                if crowns_for > crowns_against:
                    result = "win"
                elif crowns_for < crowns_against:
                    result = "loss"
                else:
                    result = "draw"
            else:
                result = None

            # 3) add your own metadata and derived fields
            flat["player_tag"] = player_tag
            flat["clan_tag"] = clan_tag
            flat["country"] = country
            flat["result"] = result

            player_records.append(flat)

        time.sleep(sleep)
        return player_records, None


    # --- Thread pool over players ---
    print(f"Fetching battles for {total_players} players with max_workers={max_workers}...")
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(_process_player, task): i
            for i, task in enumerate(tasks, start=1)
        }

        for future in as_completed(future_to_idx):
            idx = future_to_idx[future]
            try:
                player_records, failure = future.result()
            except Exception as e:
                # This should be rare, since _process_player catches most things.
                print(f"[UNEXPECTED ERROR] Worker crashed on player #{idx}: {e}")
                continue

            if player_records:
                records.extend(player_records)

            if failure:
                failed_players.append(failure)

            if idx % 50 == 0 or idx == total_players:
                print(f"[{idx}/{total_players}] players processed")

    if not records:
        print("No battle records collected. Check that your player CSV and API token are correct.")
        # still write debug CSV if we have failures
        if failed_players:
            debug_path.parent.mkdir(parents=True, exist_ok=True)
            with debug_path.open("w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["entity_type", "player_tag", "clan_tag", "country", "error"])
                for fp in failed_players:
                    writer.writerow(
                        [
                            fp["entity_type"],
                            fp["player_tag"],
                            fp["clan_tag"],
                            fp["country"],
                            fp["error"],
                        ]
                    )
            print(f"Some players failed. See debug log: {debug_path}")
        return

    # Write main battles CSV
    battles_df = pd.DataFrame.from_records(records)
    battles_df.to_csv(output_csv, index=False)
    print(f"\nSaved {len(battles_df)} battle rows to {output_csv}")

    # Write debug CSV, if any
    if failed_players:
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["entity_type", "player_tag", "clan_tag", "country", "error"])
            for fp in failed_players:
                writer.writerow(
                    [
                        fp["entity_type"],
                        fp["player_tag"],
                        fp["clan_tag"],
                        fp["country"],
                        fp["error"],
                    ]
                )
        print(f"Some players failed. See debug log: {debug_path}")

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parent

    # Example: your previously built file
    clans_csv_path = base_dir / "all_countries_clans_with_country.csv"

    # Output file with all players
    players_csv_path = base_dir / "all_players_from_clans.csv"

    # build_player_dataset(
    #     clans_csv=str(clans_csv_path),
    #     output_csv=str(players_csv_path),
    #     sleep=0.05,               # can bump to 0.2 if worried about rate limits
    #     # max_clans=100,         # use this first to test on a subset
    #     debug_csv=str(base_dir / "failed_clans_debug.csv"),
    # )

    # You can still use these separately if needed:
    # get_all_location_clans(57000120, "Italy_clans.csv")
    # combine_clan_csv_folder("clans", "all_countries_clans_with_country.csv")

    build_battle_dataset(
    players_csv="all_players_from_clans.csv",
    output_csv="battles.csv",
    sleep=0.1,             # gentle per-player pause
    max_players=10000,        # or None for full run
    debug_csv="failed_players.csv",
    max_workers=20,         # <= tune based on API rate limit
    )

