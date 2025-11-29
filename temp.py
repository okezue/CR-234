import pandas as pd

df = pd.read_csv("all_players_from_clans.csv")
df = df.head(10000)
df.to_csv("small_players.csv", index=False)