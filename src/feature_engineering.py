import pandas as pd
import numpy as np
import os

# ── Paths ──────────────────────────────────────────────────────────────────────
RAW_PATHS = [
    ('data/raw_historical/raw_23_24.csv', 2324),
    ('data/raw_historical/raw_24_25.csv', 2425),
    ('data/raw_live/raw_current.csv',     2526),
]
OUTPUT_PATH = 'data/processed/featured_training_set.csv'

# ── Load and concatenate seasons ───────────────────────────────────────────────
def load_data(paths):
    frames = []
    for path, season in paths:
        df = pd.read_csv(path)
        df['season'] = season
        frames.append(df)
    df = pd.concat(frames, ignore_index=True)
    print(f"Loaded {len(df)} rows across {df['season'].nunique()} seasons")
    return df

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(df):
    # Normalise price
    df['value'] = df['value'] / 10.0

    # Match result: win=3, draw=1, loss=0
    def get_result(row):
        if row['was_home']:
            gs, gc = row['team_h_score'], row['team_a_score']
        else:
            gs, gc = row['team_a_score'], row['team_h_score']
        if gs > gc:    return 3
        elif gs == gc: return 1
        else:          return 0

    df['team_match_result'] = df.apply(get_result, axis=1)

    # Filter to current season players only
    current_players = df[df['season'] == 2526]['element'].unique()
    df = df[df['element'].isin(current_players)].copy()
    print(f"After filtering: {len(df)} rows, {df['element'].nunique()} players")

    # Sort for rolling features
    df = df.sort_values(['element', 'season', 'GW']).reset_index(drop=True)
    return df

# ── Rolling helper ─────────────────────────────────────────────────────────────
def player_rolling(df, col, window, func='mean', new_col=None):
    """Shift-1 rolling to prevent data leakage."""
    if new_col is None:
        new_col = f'rolling_{col}_{window}'
    df[new_col] = (
        df.groupby('element')[col]
        .transform(lambda x: x.shift(1).rolling(window, min_periods=1).agg(func))
    )
    return df

# ── Feature Engineering ────────────────────────────────────────────────────────
def engineer_features(df):

    # ── 1. Player form and momentum ──────────────────────────────────────────
    df = player_rolling(df, 'total_points', 3, 'mean', 'rolling_pts_3')
    df = player_rolling(df, 'total_points', 5, 'mean', 'rolling_pts_5')
    df = player_rolling(df, 'total_points', 3, 'std',  'rolling_pts_std_3')
    df = player_rolling(df, 'total_points', 5, 'std',  'rolling_pts_std_5')
    df = player_rolling(df, 'goals_scored', 3, 'mean', 'rolling_goals_3')
    df = player_rolling(df, 'assists',      3, 'mean', 'rolling_assists_3')
    df = player_rolling(df, 'minutes',      3, 'mean', 'rolling_mins_3')

    df['goal_involvements'] = df['goals_scored'] + df['assists']
    df = player_rolling(df, 'goal_involvements', 3, 'mean', 'rolling_gi_3')

    # ── 2. Trend / slope ────────────────────────────────────────────────────
    # Positive = improving form, negative = declining
    df['form_trend'] = df['rolling_pts_3'] - df['rolling_pts_5']

    # ── 3. Value and efficiency ──────────────────────────────────────────────
    df['pts_per_million'] = (
        df.groupby(['element', 'season'])['total_points']
        .transform('cumsum') / df['value'].replace(0, np.nan)
    )
    df['price_change'] = (
        df.groupby('element')['value']
        .transform(lambda x: x.diff().fillna(0))
    )

    # ── 4. Participation ─────────────────────────────────────────────────────
    df['games_played'] = df.groupby(['element', 'season']).cumcount()
    df['pct_matches_played'] = df['games_played'] / df['GW'].clip(lower=1)

    # ── 5. Match context flags ───────────────────────────────────────────────
    df['was_home']      = df['was_home'].astype(int)
    df['played_60_plus'] = (df['minutes'] >= 60).astype(int)
    df['blanked']        = (df['total_points'] <= 1).astype(int)
    df['red_card_flag']  = (df['red_cards'] > 0).astype(int)

    # ── 6. Opponent strength ─────────────────────────────────────────────────
    # Build team-level stats per GW using team name
    team_stats = (
        df.groupby(['team', 'season', 'GW'], as_index=False)
        .agg(avg_gs=('goals_scored', 'mean'),
             avg_gc=('goals_conceded', 'mean'))
    )
    team_stats['opp_avg_goals_scored'] = (
        team_stats.groupby('team')['avg_gs']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )
    team_stats['opp_avg_goals_conceded'] = (
        team_stats.groupby('team')['avg_gc']
        .transform(lambda x: x.shift(1).rolling(5, min_periods=1).mean())
    )

    # Build integer opponent_team ID → team name map
    # Each row has both 'team' (name) and 'opponent_team' (int ID)
    # So: for team A playing opponent B, B's name appears as 'team' in other rows
    # We can map opponent_team int → team name using fixture structure
    id_to_name = (
        df[['opponent_team', 'team', 'season', 'GW']]
        .rename(columns={'opponent_team': 'opp_id', 'team': 'opp_name'})
        .drop_duplicates(subset=['opp_id', 'season', 'GW'])
    )
    # The opponent of team A in GW X is the team that has A as their opponent
    # Simpler: map opponent_team int to the team name that appears as 'team'
    # when that team is playing. We find this by matching opponent_team to
    # the fixture partner.
    fixture_map = (
        df[['team', 'opponent_team', 'season', 'GW']]
        .drop_duplicates()
    )
    # For each row, opponent's name = team name where opponent_team matches
    opp_name = (
        fixture_map
        .merge(fixture_map.rename(columns={'team': 'opp_team_name',
                                           'opponent_team': 'my_team_id'}),
               left_on=['opponent_team', 'season', 'GW'],
               right_on=['my_team_id', 'season', 'GW'],
               how='left')
        [['team', 'season', 'GW', 'opp_team_name']]
        .drop_duplicates(subset=['team', 'season', 'GW'])
    )

    df = df.merge(opp_name, on=['team', 'season', 'GW'], how='left')

    # Now merge opponent stats using opponent team name
    df = df.merge(
        team_stats[['team', 'season', 'GW',
                    'opp_avg_goals_scored', 'opp_avg_goals_conceded']],
        left_on=['opp_team_name', 'season', 'GW'],
        right_on=['team', 'season', 'GW'],
        how='left', suffixes=('', '_drop')
    )
    df = df.drop(columns=[c for c in df.columns if c.endswith('_drop')])
    df = df.drop(columns=['opp_team_name'], errors='ignore')

    # ── 7. Own team form ─────────────────────────────────────────────────────
    team_result = (
        df.groupby(['team', 'season', 'GW'])['team_match_result']
        .first().reset_index()
    )
    team_result['team_rolling_pts_3'] = (
        team_result.groupby('team')['team_match_result']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    team_scored = (
        df.groupby(['team', 'season', 'GW'])['goals_scored']
        .mean().reset_index()
    )
    team_scored['team_rolling_goals_3'] = (
        team_scored.groupby('team')['goals_scored']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )

    df = df.merge(
        team_result[['team', 'season', 'GW', 'team_rolling_pts_3']],
        on=['team', 'season', 'GW'], how='left'
    )
    df = df.merge(
        team_scored[['team', 'season', 'GW', 'team_rolling_goals_3']],
        on=['team', 'season', 'GW'], how='left'
    )

    # ── 8. Fantasy-specific variables ────────────────────────────────────────
    df['gi_rate'] = (
        df['goal_involvements'] / df['games_played'].replace(0, np.nan)
    )
    df['rolling_yellows_3'] = (
        df.groupby('element')['yellow_cards']
        .transform(lambda x: x.shift(1).rolling(3, min_periods=1).mean())
    )
    df['transfers_per_selected'] = (
        df['transfers_in'] / df['selected'].replace(0, np.nan)
    )

    # ── 9. Position encoding ─────────────────────────────────────────────────
    df['position_enc'] = df['position'].astype('category').cat.codes

    print(f"Feature engineering complete. Shape: {df.shape}")
    return df

# ── Feature columns for nb2 (featured only, no raw stats) ────────────────────
FEATURED_COLS = [
    # Metadata first
    'name', 'team', 'position', 'element', 'season', 'GW',
    # Target
    'total_points',
    # Value
    'value', 'price_change', 'pts_per_million',
    # Position encoding
    'position_enc',
    # Player form
    'rolling_pts_3', 'rolling_pts_5',
    'rolling_pts_std_3', 'rolling_pts_std_5',
    'rolling_goals_3', 'rolling_assists_3',
    'rolling_mins_3', 'rolling_gi_3',
    # Trend
    'form_trend',
    # Participation
    'pct_matches_played', 'games_played',
    # Match context
    'was_home', 'team_match_result',
    'played_60_plus', 'blanked', 'red_card_flag',
    # Opponent strength
    'opp_avg_goals_scored', 'opp_avg_goals_conceded',
    # Own team form
    'team_rolling_pts_3', 'team_rolling_goals_3',
    # Fantasy vars
    'gi_rate', 'rolling_yellows_3', 'transfers_per_selected',
]

# ── Main ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    df = load_data(RAW_PATHS)
    df = preprocess(df)
    df = engineer_features(df)

    # Keep only the featured columns
    available = [c for c in FEATURED_COLS if c in df.columns]
    missing   = [c for c in FEATURED_COLS if c not in df.columns]
    if missing:
        print(f"WARNING: missing columns: {missing}")

    df_out = df[available]

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    df_out.to_csv(OUTPUT_PATH, index=False)

    print(f"\nSaved to: {OUTPUT_PATH}")
    print(f"Total rows: {len(df_out)}")
    print(f"Total columns: {len(df_out.columns)}")
    print(f"Columns: {list(df_out.columns)}")