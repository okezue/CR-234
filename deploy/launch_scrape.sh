#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash deploy/launch_scrape.sh <session-cookie> [num-instances] [instance-type]"
    echo "  session-cookie: RoyaleAPI __royaleapi_session_v2 cookie"
    echo "  num-instances: default 8"
    echo "  instance-type: default t3.xlarge"
    exit 1
fi

SESSION_COOKIE="$1"
NUM_INSTANCES="${2:-8}"
INST_TYPE="${3:-t3.xlarge}"
REGION="us-east-1"
KEY_NAME="okezue"
SG_NAME="cr234-sg"
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$DEPLOY_DIR")"
KEY_FILE="$HOME/Downloads/okezue.pem"
PLAYERS_CSV="$HOME/Downloads/CR234/okezue_players.csv"

echo "=== CR-234 Rust Scraper Deployment ==="
echo "Instances: $NUM_INSTANCES x $INST_TYPE"
echo "Players CSV: $(wc -l < "$PLAYERS_CSV") players"

# Cross-compile Rust binary for Linux
echo "Building Rust scraper for Linux..."
cd "$PROJECT_DIR/scraping/cr-scraper"
if ! command -v cross &>/dev/null; then
    cargo install cross --git https://github.com/cross-rs/cross
fi
cross build --release --target x86_64-unknown-linux-gnu
BINARY="target/x86_64-unknown-linux-gnu/release/cr-scraper"
echo "Binary built: $(ls -lh $BINARY | awk '{print $5}')"

# Get SG ID
SG_ID=$(aws ec2 describe-security-groups --group-names "$SG_NAME" --region "$REGION" \
    --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "")
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group --group-name "$SG_NAME" --description "CR234 scraping" \
        --region "$REGION" --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0 --region "$REGION"
fi
echo "Security group: $SG_ID"

# Launch instances
AMI_ID=$(aws ec2 describe-images --region "$REGION" --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' --output text)
echo "AMI: $AMI_ID"

INSTANCE_IDS=$(aws ec2 run-instances \
    --image-id "$AMI_ID" \
    --instance-type "$INST_TYPE" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --count "$NUM_INSTANCES" \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cr234-scraper}]" \
    --query 'Instances[*].InstanceId' --output text)
echo "Launched: $INSTANCE_IDS"

sleep 30

# Get IPs
declare -a IPS
i=0
for IID in $INSTANCE_IDS; do
    IP=$(aws ec2 describe-instances --instance-ids "$IID" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    IPS[$i]="$IP"
    echo "  $IID -> $IP"
    i=$((i+1))
done

echo "Waiting for SSH..."
sleep 30

# Setup each instance
for i in $(seq 0 $((NUM_INSTANCES-1))); do
    IP="${IPS[$i]}"
    echo "Setting up $IP (instance $i/$NUM_INSTANCES)..."

    # Wait for SSH
    for attempt in $(seq 1 10); do
        if ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$KEY_FILE" ubuntu@"$IP" "echo ready" 2>/dev/null; then
            break
        fi
        sleep 10
    done

    # Upload binary, players CSV, and cf-clearance-scraper
    ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$IP" "mkdir -p ~/scraper/output"
    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" "$BINARY" ubuntu@"$IP":~/scraper/cr-scraper
    scp -o StrictHostKeyChecking=no -i "$KEY_FILE" "$PLAYERS_CSV" ubuntu@"$IP":~/scraper/players.csv
    ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$IP" "chmod +x ~/scraper/cr-scraper"

    # Install and start cf-clearance-scraper
    scp -r -o StrictHostKeyChecking=no -i "$KEY_FILE" "$PROJECT_DIR/cf-clearance-scraper" ubuntu@"$IP":~/
    ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$IP" "bash -s" <<'SETUP'
sudo apt-get update -qq && sudo apt-get install -y -qq xvfb google-chrome-stable nodejs npm >/dev/null 2>&1 || true
curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash - >/dev/null 2>&1
sudo apt-get install -y -qq nodejs >/dev/null 2>&1 || true
cd ~/cf-clearance-scraper && npm install --silent 2>/dev/null
# Patch context.close crashes
find . -name "*.js" -exec sed -i 's/await context.close()/await context.close().catch(()=>{})/g' {} \;
# Start xvfb + cf-clearance-scraper
export DISPLAY=:99
Xvfb :99 -screen 0 1920x1080x24 &
sleep 2
nohup node index.js > ~/cf.log 2>&1 &
sleep 5
echo "CF scraper started"
SETUP

    # Start Rust scraper in tmux
    ssh -o StrictHostKeyChecking=no -i "$KEY_FILE" ubuntu@"$IP" \
        "tmux new-session -d -s scrape 'cd ~/scraper && ./cr-scraper \
            --players-csv players.csv \
            --session-cookie $SESSION_COOKIE \
            --cf-url http://localhost:3000/cf-clearance-scraper \
            --workers 32 \
            --instance-id $i \
            --total-instances $NUM_INSTANCES \
            --out-dir output \
            2>&1 | tee scrape.log'"

    echo "  Scraper started on $IP"
done

echo ""
echo "=== All $NUM_INSTANCES scrapers running ==="
echo "Instance IDs: $INSTANCE_IDS"
echo "Monitor: ssh -i $KEY_FILE ubuntu@<IP> 'tail -f ~/scraper/scrape.log'"
echo "Collect: bash deploy/collect_scrape.sh"

# Save instance info
echo "$INSTANCE_IDS" > "$DEPLOY_DIR/scraper_instances.txt"
printf '%s\n' "${IPS[@]}" > "$DEPLOY_DIR/scraper_ips.txt"
