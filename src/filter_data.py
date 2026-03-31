import pandas as pd
import numpy as np

# 1. Load all three
df1 = pd.read_csv('data/raw_historical/raw_23_24.csv')
df2 = pd.read_csv('data/raw_historical/raw_24_25.csv')
df3 = pd.read_csv('data/raw_live/raw_current.csv')

# 2. Combine them
# We add the season label just so the AI can distinguish trends over time
df1['season'] = '2324'
df2['season'] = '2425'
df3['season'] = '2526'

master_df = pd.concat([df1, df2, df3], ignore_index=True)

# 3. Apply the "Engineering" fixes
# Fix the Value (Price)
master_df['value'] = master_df['value'] / 10

# Create the Win/Loss/Draw (Match Points)
is_draw = master_df['team_h_score'] == master_df['team_a_score']
is_win = ((master_df['team_h_score'] > master_df['team_a_score']) & (master_df['was_home'] == True)) | \
         ((master_df['team_a_score'] > master_df['team_h_score']) & (master_df['was_home'] == False))

master_df['team_match_result'] = np.select([is_win, is_draw], [3, 1], default=0)

# 4. Filter for ACTIVE players
# We only want to train on players who are currently in the 25/26 season 
# to avoid predicting for players who have retired or moved to Saudi/MLS.
current_players = df3['name'].unique()
master_df = master_df[master_df['name'].isin(current_players)]

# 5. Final Save
master_df.to_csv('data/processed/final_training_set.csv', index=False)

print(f"Master file created with {len(master_df)} rows and {len(master_df['name'].unique())} unique players.")