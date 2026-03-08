#!/bin/bash
set -euo pipefail

DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$DEPLOY_DIR")"
INSTANCES_FILE="$DEPLOY_DIR/instances.txt"
KEY_FILE="$DEPLOY_DIR/cr234-key.pem"
COLLECT_DIR="$PROJECT_DIR/data/collected"

if [ ! -f "$INSTANCES_FILE" ]; then
    echo "ERROR: $INSTANCES_FILE not found. Run launch.sh first."
    exit 1
fi

mkdir -p "$COLLECT_DIR"

IPS=($(cat "$INSTANCES_FILE"))
NUM=${#IPS[@]}
echo "Collecting from $NUM instances..."

for IDX in $(seq 0 $((NUM-1))); do
    IP="${IPS[$IDX]}"
    DEST="$COLLECT_DIR/inst$IDX"
    mkdir -p "$DEST"
    echo "=== Instance $IDX ($IP) ==="
    echo "  Checking progress..."
    ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" "ubuntu@$IP" \
        "cat /home/ubuntu/scraper/run1_progress_log.txt 2>/dev/null || echo 'No progress log yet'" || true
    echo "  Downloading data..."
    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" -r \
        "ubuntu@$IP:/home/ubuntu/scraper/data/scraped_data/" "$DEST/" 2>/dev/null || true
    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
        "ubuntu@$IP:/home/ubuntu/scraper/run1_progress_log.txt" "$DEST/" 2>/dev/null || true
    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" \
        "ubuntu@$IP:/home/ubuntu/scraper/run1_error_log.txt" "$DEST/" 2>/dev/null || true
    echo "  Done."
done

echo ""
echo "Merging CSVs..."

MERGED_ROWS="$COLLECT_DIR/all_worker_rows.csv"
MERGED_META="$COLLECT_DIR/all_battle_meta_data.csv"
> "$MERGED_ROWS"
> "$MERGED_META"

HEADER_ROWS=0
HEADER_META=0
for IDX in $(seq 0 $((NUM-1))); do
    DEST="$COLLECT_DIR/inst$IDX/scraped_data/battle_chunks"
    if [ -d "$DEST" ]; then
        for f in "$DEST"/*.csv; do
            [ -f "$f" ] || continue
            if [ $HEADER_ROWS -eq 0 ]; then
                cat "$f" >> "$MERGED_ROWS"
                HEADER_ROWS=1
            else
                tail -n +2 "$f" >> "$MERGED_ROWS"
            fi
        done
    fi
    META="$COLLECT_DIR/inst$IDX/scraped_data/battle_meta_data.csv"
    if [ -f "$META" ]; then
        if [ $HEADER_META -eq 0 ]; then
            cat "$META" >> "$MERGED_META"
            HEADER_META=1
        else
            tail -n +2 "$META" >> "$MERGED_META"
        fi
    fi
done

echo ""
echo "========================================="
echo "Collection complete!"
echo "========================================="
echo "Merged replay data: $MERGED_ROWS"
echo "Merged meta data:   $MERGED_META"
if [ -f "$MERGED_ROWS" ]; then
    TOTAL=$(wc -l < "$MERGED_ROWS")
    echo "Total replay rows:  $((TOTAL-1))"
fi
if [ -f "$MERGED_META" ]; then
    TOTAL=$(wc -l < "$MERGED_META")
    echo "Total meta rows:    $((TOTAL-1))"
fi
