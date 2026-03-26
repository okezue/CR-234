#!/bin/bash
# Full scraping pipeline: discover active players -> scrape battles
# Usage: bash run_scrape_pipeline.sh <instance_id> <total_instances>
INST_ID=${1:-0}
TOTAL=${2:-4}

export DISPLAY=:99
pgrep Xvfb || Xvfb :99 -screen 0 1920x1080x24 &>/dev/null &
sleep 2

echo "=== Starting CF scraper ==="
cd /home/ubuntu/cf-clearance-scraper
nohup bash -c 'while true; do node src/index.js; echo "CF crashed, restarting..." >> /home/ubuntu/cf_restart.log; sleep 5; done' > /home/ubuntu/cf.log 2>&1 &
CF_PID=$!
sleep 15

# Test CF
if curl -s -o /dev/null -w '%{http_code}' http://localhost:3000/ | grep -q '404'; then
    echo "CF scraper ready"
else
    echo "CF scraper failed to start, waiting more..."
    sleep 30
fi

echo "=== Discovering active players ==="
cd /home/ubuntu/cr234
python3 scraping/discover_active_players.py /home/ubuntu/cr234/data/big_data/active_players.csv
PLAYER_COUNT=$(wc -l < /home/ubuntu/cr234/data/big_data/active_players.csv)
echo "Found $PLAYER_COUNT active players"

if [ "$PLAYER_COUNT" -lt 100 ]; then
    echo "Too few players found, using fallback list..."
    # Fall back to original list but skip first 100K (already scraped)
    tail -n +100000 /home/ubuntu/cr234/data/big_data/all_players_from_clans.csv > /home/ubuntu/cr234/data/big_data/active_players.csv 2>/dev/null
fi

echo "=== Starting scraper (instance $INST_ID/$TOTAL) ==="
SESSION=$(curl -s -X POST http://localhost:3000/cf-clearance-scraper \
    -H "Content-Type: application/json" \
    -d '{"url":"https://royaleapi.com/","mode":"waf-session"}' | \
    python3 -c "import sys,json;d=json.load(sys.stdin);cs={c['name']:c['value'] for c in d.get('cookies',[])};print(cs.get('__royaleapi_session_v2',''))")

if [ -z "$SESSION" ]; then
    SESSION="07e90893efda4047bb5071f44d7dbc19"
    echo "Using fallback session cookie"
fi
echo "Session cookie: ${SESSION:0:10}..."

python3 scraping/new_scrape.py \
    --session-cookie "$SESSION" \
    --cf-url http://localhost:3000/cf-clearance-scraper \
    --players-csv data/big_data/active_players.csv \
    --instance-id "$INST_ID" \
    --total-instances "$TOTAL"

echo "=== Scraper finished ==="
