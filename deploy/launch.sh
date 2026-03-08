#!/bin/bash
set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: bash deploy/launch.sh <session-cookie> [num-instances]"
    exit 1
fi

SESSION_COOKIE="$1"
NUM_INSTANCES="${2:-10}"
REGION="us-east-1"
INSTANCE_TYPE="t3.xlarge"
KEY_NAME="cr234-key"
SG_NAME="cr234-sg"
DEPLOY_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$DEPLOY_DIR")"
INSTANCES_FILE="$DEPLOY_DIR/instances.txt"
PLAYERS_CSV="/Users/okezuebell/Downloads/CR234/okezue_players.csv"

AMI_ID=$(aws ec2 describe-images \
    --region "$REGION" \
    --owners 099720109477 \
    --filters "Name=name,Values=ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*" \
              "Name=state,Values=available" \
    --query 'Images | sort_by(@, &CreationDate) | [-1].ImageId' \
    --output text)
echo "Using AMI: $AMI_ID"

if ! aws ec2 describe-key-pairs --key-names "$KEY_NAME" --region "$REGION" >/dev/null 2>&1; then
    echo "Creating key pair $KEY_NAME..."
    aws ec2 create-key-pair \
        --key-name "$KEY_NAME" \
        --region "$REGION" \
        --query 'KeyMaterial' \
        --output text > "$DEPLOY_DIR/$KEY_NAME.pem"
    chmod 600 "$DEPLOY_DIR/$KEY_NAME.pem"
else
    echo "Key pair $KEY_NAME already exists"
fi
KEY_FILE="$DEPLOY_DIR/$KEY_NAME.pem"
if [ ! -f "$KEY_FILE" ]; then
    echo "ERROR: $KEY_FILE not found. Delete the key pair and re-run, or place the .pem file at $KEY_FILE"
    exit 1
fi

VPC_ID=$(aws ec2 describe-vpcs --region "$REGION" --filters "Name=isDefault,Values=true" --query 'Vpcs[0].VpcId' --output text)
SG_ID=$(aws ec2 describe-security-groups --region "$REGION" --filters "Name=group-name,Values=$SG_NAME" "Name=vpc-id,Values=$VPC_ID" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || true)
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "Creating security group $SG_NAME..."
    SG_ID=$(aws ec2 create-security-group \
        --group-name "$SG_NAME" \
        --description "CR234 scraper SSH access" \
        --vpc-id "$VPC_ID" \
        --region "$REGION" \
        --query 'GroupId' --output text)
    aws ec2 authorize-security-group-ingress \
        --group-id "$SG_ID" \
        --protocol tcp --port 22 --cidr 0.0.0.0/0 \
        --region "$REGION"
else
    echo "Security group $SG_NAME already exists: $SG_ID"
fi

USERDATA=$(base64 < "$DEPLOY_DIR/setup_ec2.sh")

echo "Launching $NUM_INSTANCES instances..."
INSTANCE_IDS=$(aws ec2 run-instances \
    --region "$REGION" \
    --image-id "$AMI_ID" \
    --instance-type "$INSTANCE_TYPE" \
    --count "$NUM_INSTANCES" \
    --key-name "$KEY_NAME" \
    --security-group-ids "$SG_ID" \
    --user-data "$USERDATA" \
    --block-device-mappings '[{"DeviceName":"/dev/sda1","Ebs":{"VolumeSize":30,"VolumeType":"gp3"}}]' \
    --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cr234-scraper},{Key=Project,Value=CR234}]" \
    --query 'Instances[*].InstanceId' --output text)

echo "Instances: $INSTANCE_IDS"
echo "Waiting for instances to be running..."
aws ec2 wait instance-running --instance-ids $INSTANCE_IDS --region "$REGION"
echo "Waiting for status checks..."
aws ec2 wait instance-status-ok --instance-ids $INSTANCE_IDS --region "$REGION"

IPS=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_IDS \
    --region "$REGION" \
    --query 'Reservations[*].Instances[*].PublicIpAddress' --output text)

> "$INSTANCES_FILE"
IDX=0
for IP in $IPS; do
    echo "$IP" >> "$INSTANCES_FILE"
    IDX=$((IDX+1))
done
echo "Saved $IDX IPs to $INSTANCES_FILE"

ssh_cmd() {
    ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$KEY_FILE" "ubuntu@$1" "$2"
}
scp_cmd() {
    scp -o StrictHostKeyChecking=no -o ConnectTimeout=10 -i "$KEY_FILE" -r "$2" "ubuntu@$1:$3"
}

wait_for_setup() {
    local ip=$1
    local max=60
    for i in $(seq 1 $max); do
        if ssh_cmd "$ip" "test -f /home/ubuntu/.setup_done && echo ok" 2>/dev/null | grep -q ok; then
            return 0
        fi
        echo "  Waiting for setup on $ip ($i/$max)..."
        sleep 30
    done
    echo "ERROR: Setup timed out on $ip"
    return 1
}

echo ""
echo "Waiting for setup to complete on all instances..."
IDX=0
for IP in $IPS; do
    echo "Checking instance $IDX ($IP)..."
    wait_for_setup "$IP" &
    IDX=$((IDX+1))
done
wait
echo "All instances setup complete."

echo ""
echo "Uploading code and starting scrapers..."
IDX=0
for IP in $IPS; do
    echo "=== Instance $IDX ($IP) ==="
    scp_cmd "$IP" "$PROJECT_DIR/scraping/new_scrape.py" "/home/ubuntu/scraper/scraping/"
    scp_cmd "$IP" "$PROJECT_DIR/requirements.txt" "/home/ubuntu/scraper/"
    scp_cmd "$IP" "$PLAYERS_CSV" "/home/ubuntu/scraper/data/big_data/okezue_players.csv"
    ssh_cmd "$IP" "cd /home/ubuntu/scraper && tmux new-session -d -s scrape '/home/ubuntu/venv/bin/python3.11 scraping/new_scrape.py --id run1 --instance-id $IDX --total-instances $NUM_INSTANCES --session-cookie $SESSION_COOKIE --players-csv data/big_data/okezue_players.csv 2>&1 | tee scrape.log'"
    echo "  Started scraper on instance $IDX"
    IDX=$((IDX+1))
done

echo ""
echo "========================================="
echo "All $NUM_INSTANCES instances launched and scraping!"
echo "========================================="
echo ""
echo "Instance IPs saved to: $INSTANCES_FILE"
echo ""
echo "Monitor commands:"
IDX=0
for IP in $IPS; do
    echo "  Instance $IDX: ssh -i $KEY_FILE ubuntu@$IP 'cat /home/ubuntu/scraper/run1_progress_log.txt'"
    IDX=$((IDX+1))
done
echo ""
echo "Attach to scraper: ssh -i $KEY_FILE ubuntu@<IP> -t 'tmux attach -t scrape'"
echo "Collect results:   bash deploy/collect.sh"
echo "Terminate all:     aws ec2 terminate-instances --instance-ids $INSTANCE_IDS --region $REGION"
