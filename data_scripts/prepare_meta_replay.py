#!/usr/bin/env python3
"""
Prepare meta and replay data: find intersection, save paired CSVs, split by game mode.
Expects all_battle_meta_data.csv and all_worker_rows.csv in the root folder.
"""

import argparse
from pathlib import Path

import pandas as pd


def main(root_folder: str | Path) -> None:
    root = Path(root_folder)

    meta_path = root / "all_battle_meta_data.csv"
    replay_path = root / "all_worker_rows.csv"

    if not meta_path.exists():
        raise FileNotFoundError(f"Missing {meta_path}")
    if not replay_path.exists():
        raise FileNotFoundError(f"Missing {replay_path}")

    print("Loading data...")
    meta = pd.read_csv(meta_path, low_memory=False)
    replay = pd.read_csv(replay_path)

    # Normalize meta replayTag (strip leading #)
    meta["replayTag"] = meta["replayTag"].apply(lambda x: x.lstrip("#"))

    # Find intersection
    meta_tags = set(meta["replayTag"])
    replay_tags = set(replay["battle_id"])
    common_tags = meta_tags & replay_tags
    print(f"Intersection: {len(common_tags)} battles")

    # Filter to intersection
    meta_paired = meta[meta["replayTag"].isin(common_tags)]
    replay_paired = replay[replay["battle_id"].isin(common_tags)]

    # Save paired files
    meta_paired.to_csv(root / "paired_meta_data.csv", index=False)
    replay_paired.to_csv(root / "paired_replay_data.csv", index=False)
    print(f"Saved paired_meta_data.csv ({len(meta_paired)} rows)")
    print(f"Saved paired_replay_data.csv ({len(replay_paired)} rows)")

    # Split paired meta by game mode
    out_dir = root / "by_modes"
    out_dir.mkdir(exist_ok=True)

    for mode_name, group in meta_paired.groupby("gameMode_name"):
        safe_name = (
            str(mode_name).replace("/", "_").replace(" ", "_").replace("\\", "_")
            if pd.notna(mode_name)
            else "unknown"
        )
        path = out_dir / f"{safe_name}.csv"
        group.to_csv(path, index=False)
        print(f"  by_modes/{safe_name}.csv ({len(group)} rows)")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare meta/replay data: intersection, paired CSVs, split by mode"
    )
    parser.add_argument(
        "root_folder",
        type=str,
        help="Folder containing all_battle_meta_data.csv and all_worker_rows.csv",
    )
    args = parser.parse_args()
    main(args.root_folder)
