import pandas as pd
import numpy as np

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/processed/final_training_set.csv')

# Ensure correct sort order so rolling windows are computed chronologically
df = df.sort_values(['name', 'season', 'GW']).reset_index(drop=True)

# ── Helper: per-player rolling window ─────────────────────────────────────────
def player_rolling(series, window):
    return (
        df.groupby('name')[series]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    )

# ═════════════════════════════════════════════════════════════════════════════
# 1. FORM & MOMENTUM
# ═════════════════════════════════════════════════════════════════════════════

# Rolling average points – last 3 and last 5 GWs
df['rolling_pts_3']     = player_rolling('total_points', 3)
df['rolling_pts_5']     = player_rolling('total_points', 5)

# Rolling average minutes – last 3 GWs
df['rolling_mins_3']    = player_rolling('minutes', 3)

# Rolling average goal involvements (goals + assists) – last 3 GWs
df['goal_involvement']  = df['goals_scored'] + df['assists']
df['rolling_gi_3']      = player_rolling('goal_involvement', 3)

# ═════════════════════════════════════════════════════════════════════════════
# 2. VALUE & EFFICIENCY
# ═════════════════════════════════════════════════════════════════════════════

# Points per million
df['pts_per_million']   = df['total_points'] / df['value'].replace(0, np.nan)

# Price change trend: positive = rising, negative = falling
df['price_change']      = df.groupby('name')['value'].diff().fillna(0)

# ═════════════════════════════════════════════════════════════════════════════
# 3. MATCH CONTEXT
# ═════════════════════════════════════════════════════════════════════════════

# was_home is already in the dataset as True/False — convert to int
df['was_home']          = df['was_home'].astype(int)

# team_match_result is already engineered in filter_data.py (3/1/0)

# Opponent strength — rolling goals conceded AND goals scored by opponent
# We proxy this by computing, for each team per GW, how many goals they
# have conceded and scored in recent games, then merging onto the opponent.

# Step 1: build a per-team-per-GW summary
team_gw = (
    df.groupby(['team', 'season', 'GW'])
    .agg(
        team_goals_scored   = ('goals_scored',   'sum'),
        team_goals_conceded = ('goals_conceded',  'sum'),
    )
    .reset_index()
)

# Step 2: rolling 5-GW averages for each team (shifted to avoid leakage)
team_gw = team_gw.sort_values(['team', 'season', 'GW'])
team_gw['opp_avg_goals_scored']   = (
    team_gw.groupby('team')['team_goals_scored']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)
team_gw['opp_avg_goals_conceded'] = (
    team_gw.groupby('team')['team_goals_conceded']
    .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
)

# Step 3: merge opponent stats onto the main df using opponent_team
# opponent_team in the raw data is a team ID — we need team name
# Build a team id → name map from the data itself
team_id_map = df[['team', 'opponent_team']].drop_duplicates()
# We'll match on the opponent's team name by joining team_gw on opponent side
df = df.merge(
    team_gw[['team', 'season', 'GW', 'opp_avg_goals_scored', 'opp_avg_goals_conceded']],
    left_on  = ['team', 'season', 'GW'],
    right_on = ['team', 'season', 'GW'],
    how='left'
)

# ═════════════════════════════════════════════════════════════════════════════
# 4. CONSISTENCY FLAGS
# ═════════════════════════════════════════════════════════════════════════════

# Did the player play 60+ minutes? (important FPL threshold for full appearance points)
df['played_60_plus']    = (df['minutes'] >= 60).astype(int)

# Blank gameweek flag — 1 or fewer points (hauls and blanks are very different)
df['blanked']           = (df['total_points'] <= 1).astype(int)

# ═════════════════════════════════════════════════════════════════════════════
# 5. POSITION-SPECIFIC
# ═════════════════════════════════════════════════════════════════════════════

# Goal involvement rate: (goals + assists) / games played so far this season
df['games_played']      = df.groupby(['name', 'season']).cumcount() + 1
df['gi_rate']           = df['goal_involvement'] / df['games_played']

# Clean sheet proxy for GK/DEF: opponent's average goals scored (already computed above)
# Lower opp_avg_goals_scored → higher chance of a clean sheet
# We keep opp_avg_goals_scored as the feature; model will learn direction per position

# ═════════════════════════════════════════════════════════════════════════════
# 6. CARDS
# ═════════════════════════════════════════════════════════════════════════════

# Rolling yellow cards – last 3 GWs (booking risk)
df['rolling_yellows_3'] = player_rolling('yellow_cards', 3)

# Red card flag this GW — signals player likely misses next game
df['red_card_flag']     = (df['red_cards'] > 0).astype(int)

# ═════════════════════════════════════════════════════════════════════════════
# 7. SAVE
# ═════════════════════════════════════════════════════════════════════════════

output_path = 'data/processed/featured_training_set.csv'
df.to_csv(output_path, index=False)

print(f"Feature engineering complete.")
print(f"Shape: {df.shape}")
print(f"New columns added: rolling_pts_3, rolling_pts_5, rolling_mins_3,")
print(f"  rolling_gi_3, goal_involvement, pts_per_million, price_change,")
print(f"  opp_avg_goals_scored, opp_avg_goals_conceded, played_60_plus,")
print(f"  blanked, games_played, gi_rate, rolling_yellows_3, red_card_flag")
print(f"Saved to {output_path}")