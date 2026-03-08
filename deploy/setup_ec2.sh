#!/bin/bash
set -euo pipefail
exec > /var/log/setup_ec2.log 2>&1
export DEBIAN_FRONTEND=noninteractive

apt-get update && apt-get install -y python3.11 python3.11-venv python3-pip tmux xvfb \
    libnss3 libatk-bridge2.0-0 libdrm2 libxkbcommon0 libgbm1 libgtk-3-0 \
    libasound2 libxshmfence1 libx11-xcb1 fonts-liberation xdg-utils wget

wget -q https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
dpkg -i google-chrome-stable_current_amd64.deb || apt-get install -fy
rm google-chrome-stable_current_amd64.deb

curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
apt-get install -y nodejs

cd /home/ubuntu
git clone https://github.com/zfcsoftware/cf-clearance-scraper.git
cd cf-clearance-scraper
npm install
sed -i "s/await context.close();/await context.close().catch(()=>{});/g" src/endpoints/getSource.js src/endpoints/wafSession.js
cat > uncaught.js << "EOF"
process.on("uncaughtException",(e)=>{console.error("Uncaught:",e.message)});
process.on("unhandledRejection",(e)=>{console.error("Unhandled:",e?.message||e)});
require("./src/index.js");
EOF

cat > /home/ubuntu/cf_wrapper.sh << "SCRIPT"
#!/bin/bash
export DISPLAY=:99
cd /home/ubuntu/cf-clearance-scraper
while true; do
    echo "[$(date)] Starting cf-clearance-scraper..." >> /home/ubuntu/cf_scraper.log
    node uncaught.js >> /home/ubuntu/cf_scraper.log 2>&1
    echo "[$(date)] Exited, restarting in 3s..." >> /home/ubuntu/cf_scraper.log
    sleep 3
done
SCRIPT
chmod +x /home/ubuntu/cf_wrapper.sh

Xvfb :99 -screen 0 1280x720x24 -ac &>/dev/null &
sleep 2
tmux new-session -d -s cfscraper /home/ubuntu/cf_wrapper.sh
sleep 5
cd /home/ubuntu

python3.11 -m venv /home/ubuntu/venv
/home/ubuntu/venv/bin/pip install pandas beautifulsoup4 curl_cffi

mkdir -p /home/ubuntu/scraper/data/scraped_data/battle_chunks
mkdir -p /home/ubuntu/scraper/data/big_data
mkdir -p /home/ubuntu/scraper/scraping

chown -R ubuntu:ubuntu /home/ubuntu
echo "SETUP_COMPLETE" > /home/ubuntu/.setup_done
