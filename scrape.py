from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time
from pathlib import Path

# ----------------------------
# 1. Setup Selenium
# ----------------------------
options = webdriver.ChromeOptions()
options.add_experimental_option("detach", True)
options.add_argument("--disable-blink-features=AutomationControlled")
options.add_experimental_option("excludeSwitches", ["enable-automation"])
options.add_experimental_option("useAutomationExtension", False)

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)

# ----------------------------
# 2. Login manually
# ----------------------------
driver.get("https://royaleapi.com/")
print("Please log into RoyaleAPI with Google...")
input("Press ENTER AFTER you are fully logged in.\n")

# ----------------------------
# 3. Choose replay
# ----------------------------
replay_url = "https://royaleapi.com/replay?tag=028PU89CL99C"
driver.get(replay_url)
time.sleep(3)


# ----------------------------
# 4. Parse HTML
# ----------------------------
html = driver.page_source
soup = BeautifulSoup(html, "html.parser")


# ----------------------------
# 5. Extract CARDS
# ----------------------------
card_events = []
for img in soup.select("div.replay_team img.replay_card"):
    card_events.append({
        "card_name": img.get("data-card"),
        "image_src": img.get("src"),
        "side": img.get("data-s"),
        "time": img.get("data-t"),
        "isAbility": img.get("data-ability"),
    })
card_events = sorted(card_events, key=lambda x: int(x["time"]))

placement_events = []
for div in soup.select("div.markers > div"):
    placement_events.append({
        "x": div.get("data-x"),
        "y": div.get("data-y"),
        "time": div.get("data-t"),
    })
df1 = pd.DataFrame(card_events)
df2 = pd.DataFrame(placement_events)
df1.to_csv("cardevents.csv")
df2.to_csv("placement_events.csv")
assert(len(card_events) == len(placement_events))
events = []
for i in range(len(placement_events)):
    assert(card_events[i]["time"] == placement_events[i]["time"])
    cur_event = dict()
    cur_event["side"] = card_events[i]['side']
    cur_event['time'] = card_events[i]["time"]
    cur_event['isAbilily'] = card_events[i]["isAbility"]
    if card_events[i]["isAbility"] == "1":
        cur_event['card_name'] = "ability-" + card_events[i]["image_src"].split("ability-")[-1].split(".png")[0]
        cur_event['x'] = None
        cur_event['y'] = None
    else:
        cur_event['card_name'] = card_events[i]["card_name"]
        cur_event['x'] = placement_events[i]["x"]
        cur_event['y'] = placement_events[i]["y"]
    events.append(cur_event)

# ----------------------------
# 7. SAVE EVERYTHING INTO ONE CSV
# ----------------------------

output_csv = "replay_full_events.csv"
df = pd.DataFrame(events)
df.to_csv(output_csv, index=False)
print(f"Saved all {len(card_events)} events to {output_csv}")
