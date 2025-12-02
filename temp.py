import pandas as pd

df = pd.read_csv("replay_full_events_all.csv")
df = df.drop_duplicates(subset=["replay_tag"])
print(len(df))