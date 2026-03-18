#!/bin/bash
set -e

KEY="$HOME/Downloads/okezue.pem"
AWS_KEY="${AWS_ACCESS_KEY_ID:?Set AWS_ACCESS_KEY_ID}"
AWS_SEC="${AWS_SECRET_ACCESS_KEY:?Set AWS_SECRET_ACCESS_KEY}"
REGION="us-east-1"
AMI="ami-0c02fb55956c7d316"
SG="sg-0a1b2c3d4e5f67890"
PROJ="/Users/okezuebell/Documents/GitHub/CR-234"
INST_TYPE="g4dn.xlarge"

export AWS_ACCESS_KEY_ID="$AWS_KEY"
export AWS_SECRET_ACCESS_KEY="$AWS_SEC"
export AWS_DEFAULT_REGION="$REGION"

chmod 600 "$KEY"

echo "=== CR-234 GPU Training Launch ==="
echo "Instance type: $INST_TYPE"

SG_ID=$(aws ec2 describe-security-groups --group-names "cr234-gpu" --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "")
if [ -z "$SG_ID" ] || [ "$SG_ID" = "None" ]; then
    echo "Creating security group..."
    SG_ID=$(aws ec2 create-security-group --group-name cr234-gpu --description "CR234 GPU training" --output text --query GroupId)
    aws ec2 authorize-security-group-ingress --group-id "$SG_ID" --protocol tcp --port 22 --cidr 0.0.0.0/0
fi
echo "SG: $SG_ID"

AMIS="ami-0de4ae9106f688338"
echo "AMI: $AMIS"

PY="/opt/pytorch/bin/python3"
RUNS=(
    "horde_v2:cd cr234/training && $PY -m horde.train_horde_v2 --episodes 2000 --max-battles 2000 --save checkpoints/horde_v2.pt --log-interval 50"
    "cql_planner:cd cr234/training && $PY cql_lstm.py --name cql_planner --mode planner --epochs 6 --bs 48 --hs 192 --nl 3 --ed 24 --cql-alpha 0.5"
    "cql_reacter:cd cr234/training && $PY cql_lstm.py --name cql_reacter --mode reacter --epochs 6 --bs 48 --hs 192 --nl 3 --ed 24 --cql-alpha 0.3"
    "three_lstm:cd cr234/training && $PY three_lstm.py --task all --name three_run --epochs 8 --bs 32 --lr 3e-4"
)

INSTANCE_IDS=()
INSTANCE_IPS=()

for run_spec in "${RUNS[@]}"; do
    RUN_NAME="${run_spec%%:*}"
    RUN_CMD="${run_spec#*:}"
    echo ""
    echo "=== Launching $RUN_NAME ==="
    IID=$(aws ec2 run-instances \
        --image-id "$AMIS" \
        --instance-type "$INST_TYPE" \
        --key-name okezue \
        --security-group-ids "$SG_ID" \
        --count 1 \
        --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=cr234-$RUN_NAME}]" \
        --block-device-mappings '[{"DeviceName":"/dev/xvda","Ebs":{"VolumeSize":100,"VolumeType":"gp3"}}]' \
        --query 'Instances[0].InstanceId' --output text)
    echo "  Instance: $IID"
    INSTANCE_IDS+=("$IID")

    aws ec2 wait instance-running --instance-ids "$IID"
    IP=$(aws ec2 describe-instances --instance-ids "$IID" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
    echo "  IP: $IP"
    INSTANCE_IPS+=("$IP")

    echo "  Waiting for SSH..."
    for i in $(seq 1 30); do
        ssh -o StrictHostKeyChecking=no -o ConnectTimeout=5 -i "$KEY" ec2-user@"$IP" "echo ok" 2>/dev/null && break
        sleep 10
    done

    echo "  Uploading code..."
    rsync -az --exclude '__pycache__' --exclude '.git' --exclude 'data/collected' --exclude 'data/processed' --exclude '*.pt' \
        -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
        "$PROJ/" ec2-user@"$IP":cr234/

    echo "  Installing deps..."
    ssh -i "$KEY" -o StrictHostKeyChecking=no ec2-user@"$IP" "bash cr234/deploy/gpu_setup.sh" 2>&1 | tail -3

    echo "  Starting: $RUN_CMD"
    ssh -i "$KEY" -o StrictHostKeyChecking=no ec2-user@"$IP" \
        "tmux new-session -d -s train 'nohup bash -c \"$RUN_CMD\" > /home/ec2-user/train.log 2>&1'"

    echo "  $RUN_NAME running on $IP"
    echo "$IID $IP $RUN_NAME" >> "$PROJ/deploy/gpu_instances.txt"
done

echo ""
echo "=== All ${#RUNS[@]} runs launched ==="
echo "Monitor: ssh -i $KEY ec2-user@<IP> 'tail -f train.log'"
echo "Instances saved to deploy/gpu_instances.txt"
cat "$PROJ/deploy/gpu_instances.txt"
