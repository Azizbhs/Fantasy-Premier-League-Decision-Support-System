import requests
import pandas as pd
from tqdm import tqdm

BASE_URL = "https://fantasy.premierleague.com/api/"

def get_bootstrap():
    r = requests.get(BASE_URL + "bootstrap-static/")
    r.raise_for_status()
    return r.json()

def get_player_history(player_id):
    r = requests.get(BASE_URL + f"element-summary/{player_id}/")
    r.raise_for_status()
    return r.json()["history"]

# --- Load metadata ---
bootstrap = get_bootstrap()

players_df = pd.DataFrame(bootstrap["elements"])[
    ["id", "web_name", "element_type", "team"]
]
teams_df = pd.DataFrame(bootstrap["teams"])[["id", "name"]].rename(
    columns={"id": "team_id", "name": "team_name"}
)
positions = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}

players_df["position"] = players_df["element_type"].map(positions)
players_df = players_df.merge(teams_df, left_on="team", right_on="team_id")

# --- Fetch per-GW history for every player ---
all_rows = []
for _, player in tqdm(players_df.iterrows(), total=len(players_df)):
    history = get_player_history(player["id"])
    for gw in history:
        gw["name"] = player["web_name"]
        gw["position"] = player["position"]
        gw["team"] = player["team_name"]
    all_rows.extend(history)

df = pd.DataFrame(all_rows)

# Rename to match your schema
df = df.rename(columns={
    "round": "GW",
    "total_points": "total_points",
    # add any further renames here
})

df["value"] = df["value"] / 10  # convert to £m

df.to_csv("fpl_2025_26_current.csv", index=False)
print(f"Saved {len(df)} rows")