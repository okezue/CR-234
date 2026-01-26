import os
from collections import defaultdict

INPUT_FILE = "battle_meta_data.csv"
OUTPUT_DIR = "by_first_field"

os.makedirs(OUTPUT_DIR, exist_ok=True)

files = {}

with open(INPUT_FILE, "r", encoding="utf-8", errors="replace") as f:
    for line_num, line in enumerate(f, start=1):
        line = line.rstrip("\n")
        if not line:
            continue

        # split only on the first comma
        first_field = line.split(",", 1)[0].strip()

        if not first_field:
            first_field = "EMPTY"

        # sanitize filename
        safe_name = "".join(c if c.isalnum() or c in "-_." else "_" for c in first_field)
        out_path = os.path.join(OUTPUT_DIR, f"{safe_name}.csv")

        if safe_name not in files:
            files[safe_name] = open(out_path, "a", encoding="utf-8")

        files[safe_name].write(line + "\n")

for f in files.values():
    f.close()

print(f"Done. Wrote {len(files)} files to {OUTPUT_DIR}/")
