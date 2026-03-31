import pandas as pd

def check_names():
    # Load the datasets
    df_hist = pd.read_csv('data/raw_historical/merged_gw_2024-2025.csv')
    df_live = pd.read_csv('data/raw_live/fpl_2024_25_current.csv')
    
    print("--- HISTORICAL DATA NAMES (First 5) ---")
    if 'name' in df_hist.columns:
        print(df_hist['name'].head(5).tolist())
    else:
        print(f"Columns available: {df_hist.columns.tolist()}")

    print("\n--- LIVE DATA NAMES (First 5) ---")
    if 'name' in df_live.columns:
        print(df_live['name'].head(5).tolist())
    else:
        print("No 'name' column found. Columns available:")
        print(df_live.columns.tolist())

if __name__ == "__main__":
    check_names()