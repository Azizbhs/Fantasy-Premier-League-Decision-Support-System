import streamlit as st
import pandas as pd
import numpy as np
import requests
import pickle
import os
import tensorflow as tf
import autokeras as ak
from sklearn.impute import SimpleImputer
from pulp import *
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FPL Decision Support System",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
    .main-title {
        font-family: 'Bebas Neue', sans-serif;
        font-size: 4rem;
        letter-spacing: 4px;
        background: linear-gradient(135deg, #00d2ff, #7b2ff7);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .subtitle { color: #888; font-size: 0.95rem; letter-spacing: 2px; text-transform: uppercase; margin-top: 0; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #0f3460;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
    }
    .metric-value { font-family: 'Bebas Neue', sans-serif; font-size: 2.5rem; color: #00d2ff; letter-spacing: 2px; }
    .metric-label { font-size: 0.75rem; color: #888; text-transform: uppercase; letter-spacing: 1px; }
    .player-card { background: #1a1a2e; border-left: 4px solid #00d2ff; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0; }
    .captain-card { background: #1a1a2e; border-left: 4px solid #ffd700; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0; }
    .bench-card { background: #111; border-left: 4px solid #444; border-radius: 8px; padding: 0.8rem 1rem; margin: 0.4rem 0; opacity: 0.7; }
    .fixture-card { background: #1a1a2e; border-radius: 10px; padding: 1rem 1.5rem; margin: 0.4rem 0; border: 1px solid #0f3460; }
    .pos-badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.7rem; font-weight: 600; letter-spacing: 1px; }
    .conf-high { color: #00ff88; }
    .conf-medium { color: #ffd700; }
    .conf-low { color: #ff6b6b; }
    .section-header { font-family: 'Bebas Neue', sans-serif; font-size: 1.8rem; letter-spacing: 3px; color: #fff; border-bottom: 2px solid #0f3460; padding-bottom: 0.5rem; margin: 1.5rem 0 1rem 0; }
    div[data-testid="stSidebar"] { background: #0d0d1a; }
    .transfer-in { color: #00ff88; font-weight: 600; }
    .transfer-out { color: #ff6b6b; font-weight: 600; }
    input[type=number]::-webkit-inner-spin-button,
    input[type=number]::-webkit-outer-spin-button { -webkit-appearance: none; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────────
FPL_BASE     = "https://fantasy.premierleague.com/api/"
NEXT_GW      = 32
POSITION_MAP = {0: 'DEF', 1: 'FWD', 2: 'GK', 3: 'MID'}
POS_COLORS   = {'GK': '#f0a500', 'DEF': '#00b4d8', 'MID': '#7b2ff7', 'FWD': '#ff6b6b'}
MODEL_DIR    = 'models'
DATA_PATH    = 'data/processed/featured_training_set.csv'

# ── FPL API ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=3600)
def fetch_bootstrap():
    try:
        r = requests.get(FPL_BASE + "bootstrap-static/", timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return None

@st.cache_data(ttl=3600)
def fetch_fixtures(gw):
    try:
        r = requests.get(FPL_BASE + f"fixtures/?event={gw}", timeout=10)
        r.raise_for_status()
        return r.json()
    except:
        return []

@st.cache_data(ttl=300)
def fetch_user_squad(fpl_id, gw):
    try:
        r = requests.get(FPL_BASE + f"entry/{fpl_id}/event/{gw}/picks/", timeout=10)
        r.raise_for_status()
        data  = r.json()
        picks = [p['element'] for p in data['picks']]
        itb   = data['entry_history']['bank'] / 10
        return picks, itb
    except:
        return None, None

# ── Load Models ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_models():
    models = {}

    # Huber Regressor (PyCaret pipeline)
    try:
        with open(os.path.join(MODEL_DIR, 'pycaret_best_model.pkl'), 'rb') as f:
            models['huber'] = pickle.load(f)
        st.sidebar.success("✅ Huber loaded")
    except Exception as e:
        models['huber'] = None
        st.sidebar.warning(f"⚠️ Huber: {str(e)[:40]}")

    # LightGBM (PyCaret pipeline)
    try:
        with open(os.path.join(MODEL_DIR, 'best_models.pkl'), 'rb') as f:
            best = pickle.load(f)
            models['lgbm'] = best[1] if len(best) > 1 else None
        st.sidebar.success("✅ LightGBM loaded")
    except Exception as e:
        models['lgbm'] = None
        st.sidebar.warning(f"⚠️ LightGBM: {str(e)[:40]}")

    # AutoKeras
    try:
        models['autokeras'] = tf.keras.models.load_model(
            os.path.join(MODEL_DIR, 'autokeras_best_model.keras'),
            custom_objects=ak.CUSTOM_OBJECTS
        )
        st.sidebar.success("✅ AutoKeras loaded")
    except Exception as e:
        models['autokeras'] = None
        st.sidebar.warning(f"⚠️ AutoKeras: {str(e)[:40]}")

    return models

# ── Generate Predictions ───────────────────────────────────────────────────────
@st.cache_data
def generate_predictions(_models):
    if not os.path.exists(DATA_PATH):
        return None

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df = df.sort_values(['name', 'season', 'GW'])

    # Most recent GW row per player = current form snapshot
    latest = df.groupby('name').last().reset_index()

    # Keep metadata
    meta = latest[['name', 'team', 'position', 'value']].copy()
    if meta['position'].dtype == object:
        # Already strings — map directly
        str_map = {'GK': 'GK', 'DEF': 'DEF', 'MID': 'MID', 'FWD': 'FWD'}
        meta['pos'] = meta['position'].map(str_map)
    else:
        # Encoded integers
        meta['pos'] = meta['position'].map(POSITION_MAP)

    # Encode same as training
    latest['position'] = latest['position'].astype('category').cat.codes
    latest['was_home'] = latest['was_home'].astype(int)

    DROP_COLS = ['name', 'team', 'season', 'GW', 'kickoff_time', 'total_points',
                 'element', 'fixture', 'modified']
    DROP_COLS   = [c for c in DROP_COLS if c in latest.columns]
    feature_df  = latest.drop(columns=DROP_COLS)
    non_numeric = feature_df.select_dtypes(exclude=[np.number, 'bool']).columns.tolist()
    feature_df  = feature_df.drop(columns=non_numeric)

    imputer   = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(feature_df).astype(np.float32)
    feat_cols = feature_df.columns.tolist()
    X_df      = pd.DataFrame(X_imputed, columns=feat_cols)

    # Huber predictions — try DataFrame then numpy array
    if _models.get('huber') is not None:
        try:
            huber_p = np.clip(_models['huber'].predict(X_df), 0, None)
        except:
            try:
                huber_p = np.clip(_models['huber'].predict(X_imputed), 0, None)
            except:
                huber_p = np.zeros(len(X_df))
    else:
        huber_p = np.zeros(len(X_df))

    # LightGBM predictions — try DataFrame then numpy array
    if _models.get('lgbm') is not None:
        try:
            lgbm_p = np.clip(_models['lgbm'].predict(X_df), 0, None)
        except:
            try:
                lgbm_p = np.clip(_models['lgbm'].predict(X_imputed), 0, None)
            except:
                lgbm_p = np.zeros(len(X_df))
    else:
        lgbm_p = np.zeros(len(X_df))

    # AutoKeras predictions
    if _models.get('autokeras') is not None:
        try:
            ak_p = np.clip(_models['autokeras'].predict(X_imputed).flatten(), 0, None)
        except:
            ak_p = np.zeros(len(X_df))
    else:
        ak_p = np.zeros(len(X_df))

    # Weighted ensemble: 20% Huber + 30% LightGBM + 50% AutoKeras
    mid = np.clip(0.2 * huber_p + 0.3 * lgbm_p + 0.5 * ak_p, 0, None)
    std = np.std(np.array([huber_p, lgbm_p, ak_p]), axis=0)
    low  = np.clip(mid - std, 0, None)
    high = mid + std

    interval_width = high - low
    q33, q66 = np.percentile(interval_width, [33, 66])
    def get_conf(w):
        if w <= q33:   return 'High'
        elif w <= q66: return 'Medium'
        else:          return 'Low'
    confidence = [get_conf(w) for w in interval_width]

    scaled = np.clip(
        mid * 0.4 + (meta['value'].values / meta['value'].max()) * mid.max() * 0.6,
        0, None
    )

    preds = meta.copy().reset_index(drop=True)
    preds['predicted_pts_mid']    = np.round(mid,    2)
    preds['predicted_pts_low']    = np.round(low,    2)
    preds['predicted_pts_high']   = np.round(high,   2)
    preds['predicted_pts_scaled'] = np.round(scaled, 4)
    preds['confidence']           = confidence
    return preds

# ── PuLP Optimiser ─────────────────────────────────────────────────────────────
def optimise_squad(df, budget=100.0, current_squad=None, n_transfers=15):
    players = list(df.index)
    starter  = LpVariable.dicts('starter',  players, cat='Binary')
    benched  = LpVariable.dicts('benched',  players, cat='Binary')
    captain  = LpVariable.dicts('captain',  players, cat='Binary')
    tin      = LpVariable.dicts('tin',      players, cat='Binary')
    tout     = LpVariable.dicts('tout',     players, cat='Binary')
    keep     = LpVariable.dicts('keep',     players, cat='Binary')

    prob = LpProblem('FPL', LpMaximize)
    prob += (
        lpSum(starter[i] * df.loc[i, 'predicted_pts_scaled'] for i in players) +
        lpSum(captain[i] * df.loc[i, 'predicted_pts_scaled'] for i in players)
    )

    gk_idx  = [i for i in players if df.loc[i, 'pos'] == 'GK']
    def_idx = [i for i in players if df.loc[i, 'pos'] == 'DEF']
    mid_idx = [i for i in players if df.loc[i, 'pos'] == 'MID']
    fwd_idx = [i for i in players if df.loc[i, 'pos'] == 'FWD']

    if current_squad is None:
        # Fresh squad
        for i in players:
            prob += keep[i] == 1
            prob += tin[i]  == 0
            prob += tout[i] == 0
            prob += starter[i] + benched[i] <= 1
        prob += lpSum(starter[i] + benched[i] for i in players) == 15
        prob += lpSum((starter[i] + benched[i]) * df.loc[i, 'value'] for i in players) <= budget
        prob += lpSum(starter[i] + benched[i] for i in gk_idx)  == 2
        prob += lpSum(starter[i] + benched[i] for i in def_idx) == 5
        prob += lpSum(starter[i] + benched[i] for i in mid_idx) == 5
        prob += lpSum(starter[i] + benched[i] for i in fwd_idx) == 3
        for club in df['team'].unique():
            ci = [i for i in players if df.loc[i, 'team'] == club]
            prob += lpSum(starter[i] + benched[i] for i in ci) <= 3
    else:
        # Transfer optimisation
        current_names   = set(current_squad)
        current_idx     = [i for i in players if df.loc[i, 'name'] in current_names]
        not_current_idx = [i for i in players if df.loc[i, 'name'] not in current_names]

        for i in current_idx:
            prob += keep[i] + tout[i] == 1
        for i in not_current_idx:
            prob += keep[i] == 0
        for i in players:
            prob += tin[i] <= 1 - keep[i]

        prob += lpSum(tin[i]  for i in players) <= n_transfers
        prob += lpSum(tout[i] for i in players) <= n_transfers
        prob += lpSum(tin[i]  for i in players) == lpSum(tout[i] for i in players)
        prob += lpSum(keep[i] + tin[i] for i in players) == 15
        prob += lpSum((keep[i] + tin[i]) * df.loc[i, 'value'] for i in players) <= budget
        prob += lpSum(keep[i] + tin[i] for i in gk_idx)  == 2
        prob += lpSum(keep[i] + tin[i] for i in def_idx) == 5
        prob += lpSum(keep[i] + tin[i] for i in mid_idx) == 5
        prob += lpSum(keep[i] + tin[i] for i in fwd_idx) == 3
        for i in players:
            prob += starter[i] + benched[i] <= keep[i] + tin[i]
        for club in df['team'].unique():
            ci = [i for i in players if df.loc[i, 'team'] == club]
            prob += lpSum(keep[i] + tin[i] for i in ci) <= 3

    # Common constraints
    prob += lpSum(starter[i] for i in players) == 11
    prob += lpSum(benched[i] for i in players) == 4
    prob += lpSum(starter[i] for i in gk_idx)  == 1
    prob += lpSum(starter[i] for i in def_idx) >= 3
    prob += lpSum(starter[i] for i in mid_idx) >= 2
    prob += lpSum(starter[i] for i in fwd_idx) >= 1
    for i in players:
        prob += captain[i] <= starter[i]
    prob += lpSum(captain[i] for i in players) == 1

    PULP_CBC_CMD(msg=0).solve(prob)

    if LpStatus[prob.status] != 'Optimal':
        return None, None, None, None

    s_ids  = [i for i in players if starter[i].value()  == 1]
    b_ids  = [i for i in players if benched[i].value()  == 1]
    cap_id = [i for i in players if captain[i].value()  == 1][0]
    t_in   = [df.loc[i, 'name'] for i in players if tin[i].value()  == 1]
    t_out  = [df.loc[i, 'name'] for i in players if tout[i].value() == 1]

    return s_ids, b_ids, cap_id, (t_in, t_out)

# ── UI Helpers ─────────────────────────────────────────────────────────────────
def pos_badge(pos):
    color = POS_COLORS.get(pos, '#888')
    return f'<span class="pos-badge" style="background:{color}22;color:{color};">{pos}</span>'

def display_squad(df, s_ids, b_ids, cap_id):
    s11   = df.loc[s_ids].copy()
    bench = df.loc[b_ids].copy()
    s11['role']   = 'Starter'
    s11.loc[cap_id, 'role'] = 'Captain'
    bench['role'] = 'Bench'

    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        for _, row in s11[s11['pos'] == pos].sort_values('predicted_pts_scaled', ascending=False).iterrows():
            is_cap   = row['role'] == 'Captain'
            card     = 'captain-card' if is_cap else 'player-card'
            cap_tag  = ' 🅒' if is_cap else ''
            conf_cls = {'High': 'conf-high', 'Medium': 'conf-medium', 'Low': 'conf-low'}.get(row['confidence'], 'conf-low')
            st.markdown(f"""
            <div class="{card}">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>{pos_badge(pos)} <span style="color:#fff;font-weight:600;margin-left:8px;">{row['name']}{cap_tag}</span> <span style="color:#666;font-size:0.8rem;margin-left:8px;">{row['team']}</span></div>
                    <div style="text-align:right;">
                        <span style="color:#00d2ff;font-weight:600;">{row['predicted_pts_mid']:.2f} pts</span>
                        <span style="color:#555;font-size:0.8rem;margin:0 8px;">[{row['predicted_pts_low']:.2f}–{row['predicted_pts_high']:.2f}]</span>
                        <span class="{conf_cls}" style="font-size:0.8rem;">● {row['confidence']}</span>
                        <span style="color:#888;font-size:0.8rem;margin-left:12px;">£{row['value']}m</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

    st.markdown('<p class="section-header" style="font-size:1.2rem;">BENCH</p>', unsafe_allow_html=True)
    for _, row in bench.sort_values('pos').iterrows():
        st.markdown(f"""
        <div class="bench-card">
            <div style="display:flex;justify-content:space-between;">
                <div>{pos_badge(row['pos'])} <span style="color:#aaa;margin-left:8px;">{row['name']}</span> <span style="color:#555;font-size:0.8rem;">{row['team']}</span></div>
                <div><span style="color:#666;">{row['predicted_pts_mid']:.2f} pts</span> <span style="color:#555;font-size:0.8rem;margin-left:8px;">£{row['value']}m</span></div>
            </div>
        </div>""", unsafe_allow_html=True)

    total_cost = pd.concat([s11, bench])['value'].sum()
    total_pred = s11['predicted_pts_scaled'].sum() + s11.loc[cap_id, 'predicted_pts_scaled']
    cap_name   = df.loc[cap_id, 'name']
    st.markdown("")
    c1, c2, c3 = st.columns(3)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">£{total_cost:.1f}m</div><div class="metric-label">Total Cost</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{total_pred:.1f}</div><div class="metric-label">Predicted Pts</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">{cap_name.split()[-1]}</div><div class="metric-label">Captain</div></div>', unsafe_allow_html=True)

# ── Load everything ────────────────────────────────────────────────────────────
models      = load_models()
predictions = generate_predictions(models)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title" style="font-size:2rem;">FPL DSS</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Decision Support System</p>', unsafe_allow_html=True)
    st.markdown("---")
    page = st.radio("Nav", [
        "🏠 Overview",
        f"🏆 Best Squad — GW{NEXT_GW}",
        "👤 My Squad",
        "📊 Player Predictions",
        "🎯 Captain Picks",
        f"📅 GW{NEXT_GW} Fixtures"
    ], label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f'<p style="color:#555;font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;">Predicting GW{NEXT_GW}</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#00d2ff;font-size:0.85rem;">Huber MAE: 0.797</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#00d2ff;font-size:0.85rem;">LightGBM R²: 0.328</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#00d2ff;font-size:0.85rem;">AutoKeras R²: 0.349</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#888;font-size:0.75rem;">20% Huber · 30% LGBM · 50% AK</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<h1 class="main-title">FPL DECISION SUPPORT</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">GW{NEXT_GW} Predictions · AutoML · Mathematical Optimisation · 2023–2026</p>', unsafe_allow_html=True)
    st.markdown("")

    n_players = len(predictions) if predictions is not None else 0
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{n_players}</div><div class="metric-label">Players Tracked</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><div class="metric-value">3</div><div class="metric-label">Seasons of Data</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><div class="metric-value">0.797</div><div class="metric-label">Best MAE (Huber)</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><div class="metric-value">45%</div><div class="metric-label">vs Naive Baseline</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-header">MODEL COMPARISON</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    mnames = ['Huber\nRegressor', 'LightGBM', 'AutoKeras\nNeural Net', 'Naive\nBaseline']
    maes   = [0.797, 0.930, 0.904, 1.448]
    r2s    = [0.239, 0.328, 0.349]
    cm     = ['#00d2ff', '#00ff88', '#f0a500', '#444']

    with c1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#1a1a2e')
        bars = ax.bar(mnames, maes, color=cm, width=0.5, edgecolor='none')
        ax.set_title('MAE — Lower is Better', color='white', fontsize=11, pad=10)
        ax.tick_params(colors='#888', labelsize=8); ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, color='#222', linewidth=0.5); ax.set_axisbelow(True)
        for bar, val in zip(bars, maes):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    with c2:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        fig.patch.set_facecolor('#1a1a2e'); ax.set_facecolor('#1a1a2e')
        bars = ax.bar(mnames[:3], r2s, color=cm[:3], width=0.5, edgecolor='none')
        ax.set_title('R² — Higher is Better', color='white', fontsize=11, pad=10)
        ax.tick_params(colors='#888', labelsize=8); ax.spines[:].set_visible(False)
        ax.yaxis.grid(True, color='#222', linewidth=0.5); ax.set_axisbelow(True)
        for bar, val in zip(bars, r2s):
            ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.003, f'{val:.3f}', ha='center', va='bottom', color='white', fontsize=8)
        plt.tight_layout(); st.pyplot(fig); plt.close()

    st.markdown('<p class="section-header">PIPELINE</p>', unsafe_allow_html=True)
    steps = [("📥","Data Engineering","3 seasons · vaastav"),("⚙️","Feature Engineering","15+ features"),("🤖","AutoML","PyCaret + AutoKeras"),("🧮","Optimisation","PuLP · Knapsack"),("📊","Dashboard","Streamlit · Live FPL")]
    cols  = st.columns(5)
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="metric-card" style="padding:1rem;"><div style="font-size:1.8rem;">{icon}</div><div style="color:#fff;font-weight:600;font-size:0.85rem;margin:0.3rem 0;">{title}</div><div style="color:#666;font-size:0.75rem;">{desc}</div></div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — BEST SQUAD
# ══════════════════════════════════════════════════════════════════════════════
elif page == f"🏆 Best Squad — GW{NEXT_GW}":
    st.markdown(f'<h1 class="main-title">BEST SQUAD — GW{NEXT_GW}</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Mathematically optimal squad · PuLP knapsack · £100m budget</p>', unsafe_allow_html=True)

    if predictions is None:
        st.error("Could not load predictions. Ensure featured_training_set.csv and model files exist.")
    else:
        with st.spinner(f"Optimising for GW{NEXT_GW}..."):
            s_ids, b_ids, cap_id, _ = optimise_squad(predictions)
        if s_ids is None:
            st.error("Optimisation failed.")
        else:
            st.markdown('<p class="section-header">STARTING XI</p>', unsafe_allow_html=True)
            display_squad(predictions, s_ids, b_ids, cap_id)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — MY SQUAD
# ══════════════════════════════════════════════════════════════════════════════
elif page == "👤 My Squad":
    st.markdown('<h1 class="main-title">MY SQUAD</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Import your FPL squad and get optimal transfer suggestions</p>', unsafe_allow_html=True)

    c1, c2 = st.columns([1, 2])
    with c1:
        st.markdown('<p class="section-header" style="font-size:1.2rem;">IMPORT</p>', unsafe_allow_html=True)
        # Text input — no annoying +/- buttons
        fpl_id_str     = st.text_input("Your FPL Team ID", value="", placeholder="e.g. 3187370",
                                        help="Find in FPL app under Points — it's in the URL")
        free_transfers = st.selectbox("Free Transfers", [1, 2, "Wildcard"])
        fetch_btn      = st.button("🔍 Import & Optimise", type="primary")

    with c2:
        if fetch_btn:
            if not fpl_id_str.strip().isdigit():
                st.error("Please enter a valid numeric FPL Team ID.")
            else:
                fpl_id  = int(fpl_id_str.strip())
                prev_gw = NEXT_GW - 1

                with st.spinner("Fetching squad from FPL API..."):
                    picks, itb = fetch_user_squad(fpl_id, prev_gw)

                if picks is None:
                    st.error("Could not fetch squad. Check your FPL ID or internet connection.")
                elif predictions is None:
                    st.error("Predictions not loaded. Check model files.")
                else:
                    bootstrap = fetch_bootstrap()
                    if bootstrap is None:
                        st.error("Could not reach FPL API.")
                    else:
                        id_to_name   = {p['id']: p['web_name'] for p in bootstrap['elements']}
                        squad_names  = [id_to_name.get(pid, '') for pid in picks if id_to_name.get(pid)]
                        current_df   = predictions[predictions['name'].isin(squad_names)]
                        total_budget = current_df['value'].sum() + itb
                        n_transfers  = 15 if free_transfers == "Wildcard" else int(free_transfers)

                        st.markdown(f'<p style="color:#888;">✅ {len(squad_names)} players imported · Budget: £{total_budget:.1f}m · {free_transfers} transfer(s)</p>', unsafe_allow_html=True)

                        with st.spinner("Running transfer optimisation..."):
                            result = optimise_squad(
                                predictions, budget=total_budget,
                                current_squad=squad_names, n_transfers=n_transfers
                            )

                        if result[0] is None:
                            st.error("Optimisation failed. Try more transfers or check your squad.")
                        else:
                            s_ids, b_ids, cap_id, (t_in, t_out) = result

                            if t_in:
                                st.markdown('<p class="section-header" style="font-size:1.2rem;">RECOMMENDED TRANSFERS</p>', unsafe_allow_html=True)
                                for p_in, p_out in zip(t_in, t_out):
                                    ri = predictions[predictions['name'] == p_in]
                                    ro = predictions[predictions['name'] == p_out]
                                    if ri.empty or ro.empty:
                                        continue
                                    ri = ri.iloc[0]; ro = ro.iloc[0]
                                    diff     = ri['value'] - ro['value']
                                    diff_str = f"+£{diff:.1f}m" if diff >= 0 else f"-£{abs(diff):.1f}m"
                                    st.markdown(f"""
                                    <div style="background:#1a1a2e;border-radius:10px;padding:1rem;margin:0.5rem 0;border:1px solid #0f3460;">
                                        <div style="display:flex;justify-content:space-between;align-items:center;">
                                            <div><span class="transfer-out">▼ OUT: {p_out}</span> <span style="color:#555;font-size:0.8rem;">£{ro['value']}m · {ro['predicted_pts_mid']:.2f} pts</span></div>
                                            <div><span class="transfer-in">▲ IN: {p_in}</span> <span style="color:#555;font-size:0.8rem;">£{ri['value']}m · {ri['predicted_pts_mid']:.2f} pts</span></div>
                                            <div style="color:#888;">{diff_str}</div>
                                        </div>
                                    </div>""", unsafe_allow_html=True)
                            else:
                                st.success("✅ Your squad is already optimal — no transfers needed!")

                            cap_name = predictions.loc[cap_id, 'name']
                            cap_pts  = predictions.loc[cap_id, 'predicted_pts_mid']
                            st.markdown(f'<div class="captain-card" style="margin-top:1rem;"><span style="color:#ffd700;font-weight:600;">🅒 Captain: {cap_name}</span> <span style="color:#888;margin-left:8px;">{cap_pts:.2f} pts → {cap_pts*2:.2f} as captain</span></div>', unsafe_allow_html=True)

                            st.markdown('<p class="section-header" style="font-size:1.2rem;margin-top:1.5rem;">OPTIMISED STARTING XI</p>', unsafe_allow_html=True)
                            display_squad(predictions, s_ids, b_ids, cap_id)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — PLAYER PREDICTIONS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Player Predictions":
    st.markdown('<h1 class="main-title">PLAYER PREDICTIONS</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">GW{NEXT_GW} ensemble predictions · Huber + LightGBM + AutoKeras</p>', unsafe_allow_html=True)

    if predictions is None:
        st.error("Predictions not loaded.")
    else:
        c1, c2, c3, c4 = st.columns(4)
        with c1: pos_f  = st.selectbox("Position",   ["All", "GK", "DEF", "MID", "FWD"])
        with c2: conf_f = st.selectbox("Confidence", ["All", "High", "Medium", "Low"])
        with c3: max_p  = st.slider("Max Price (£m)", 4.0, 15.0, 15.0, 0.5)
        with c4: search = st.text_input("Search Player", "")

        f = predictions.copy()
        if pos_f  != "All": f = f[f['pos'] == pos_f]
        if conf_f != "All": f = f[f['confidence'] == conf_f]
        f = f[f['value'] <= max_p]
        if search: f = f[f['name'].str.contains(search, case=False, na=False)]
        f = f.sort_values('predicted_pts_mid', ascending=False).reset_index(drop=True)

        st.markdown(f'<p style="color:#888;font-size:0.85rem;">{len(f)} players shown</p>', unsafe_allow_html=True)
        for _, row in f.head(50).iterrows():
            conf_cls = {'High': 'conf-high', 'Medium': 'conf-medium', 'Low': 'conf-low'}.get(row['confidence'], 'conf-low')
            st.markdown(f"""
            <div class="player-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div>{pos_badge(row['pos'])} <span style="color:#fff;font-weight:500;margin-left:8px;">{row['name']}</span> <span style="color:#555;font-size:0.8rem;margin-left:8px;">{row['team']}</span></div>
                    <div style="text-align:right;">
                        <span style="color:#00d2ff;font-weight:600;">{row['predicted_pts_mid']:.2f} pts</span>
                        <span style="color:#444;font-size:0.8rem;margin:0 8px;">[{row['predicted_pts_low']:.2f}–{row['predicted_pts_high']:.2f}]</span>
                        <span class="{conf_cls}" style="font-size:0.8rem;">● {row['confidence']}</span>
                        <span style="color:#888;font-size:0.8rem;margin-left:12px;">£{row['value']}m</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — CAPTAIN PICKS
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯 Captain Picks":
    st.markdown('<h1 class="main-title">CAPTAIN PICKS</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="subtitle">GW{NEXT_GW} top captaincy recommendations</p>', unsafe_allow_html=True)

    if predictions is None:
        st.error("Predictions not loaded.")
    else:
        top = predictions.nlargest(10, 'predicted_pts_mid').reset_index(drop=True)
        st.markdown('<p class="section-header">TOP 10 CAPTAINCY OPTIONS</p>', unsafe_allow_html=True)
        for rank, (_, row) in enumerate(top.iterrows(), 1):
            conf_cls   = {'High': 'conf-high', 'Medium': 'conf-medium', 'Low': 'conf-low'}.get(row['confidence'], 'conf-low')
            rank_color = "#ffd700" if rank == 1 else "#888"
            border     = "#ffd700" if rank == 1 else "#00d2ff"
            st.markdown(f"""
            <div style="background:#1a1a2e;border-left:4px solid {border};border-radius:8px;padding:1rem;margin:0.4rem 0;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="display:flex;align-items:center;gap:12px;">
                        <span style="font-family:'Bebas Neue',sans-serif;font-size:1.5rem;color:{rank_color};min-width:30px;">#{rank}</span>
                        {pos_badge(row['pos'])}
                        <div>
                            <div style="color:#fff;font-weight:600;">{row['name']}</div>
                            <div style="color:#555;font-size:0.8rem;">{row['team']} · £{row['value']}m</div>
                        </div>
                    </div>
                    <div style="text-align:right;">
                        <div style="color:#00d2ff;font-size:1.2rem;font-weight:600;">{row['predicted_pts_mid']:.2f} pts</div>
                        <div style="color:#888;font-size:0.8rem;">×2 = {row['predicted_pts_mid']*2:.2f} as captain</div>
                        <div style="color:#555;font-size:0.75rem;">[{row['predicted_pts_low']:.2f}–{row['predicted_pts_high']:.2f}]</div>
                        <span class="{conf_cls}" style="font-size:0.75rem;">● {row['confidence']}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — FIXTURES
# ══════════════════════════════════════════════════════════════════════════════
elif page == f"📅 GW{NEXT_GW} Fixtures":
    st.markdown(f'<h1 class="main-title">GW{NEXT_GW} FIXTURES</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Upcoming fixtures — home team on the left</p>', unsafe_allow_html=True)

    bootstrap = fetch_bootstrap()
    fixtures  = fetch_fixtures(NEXT_GW)

    if not bootstrap or not fixtures:
        st.error("Could not load fixtures from FPL API.")
    else:
        team_map = {t['id']: t['name'] for t in bootstrap['teams']}

        st.markdown(f'<p class="section-header">GAMEWEEK {NEXT_GW} FIXTURES</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#888;font-size:0.85rem;">{len(fixtures)} fixtures</p>', unsafe_allow_html=True)
        st.markdown("")

        for fix in fixtures:
            home = team_map.get(fix['team_h'], 'Unknown')
            away = team_map.get(fix['team_a'], 'Unknown')

            # Difficulty colors
            h_diff = fix.get('team_h_difficulty', 3)
            a_diff = fix.get('team_a_difficulty', 3)
            def diff_color(d):
                return {1:'#00ff88', 2:'#00d2ff', 3:'#888', 4:'#ffd700', 5:'#ff6b6b'}.get(d, '#888')

            st.markdown(f"""
            <div class="fixture-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="flex:1;text-align:right;">
                        <span style="color:#fff;font-weight:600;font-size:1rem;">{home}</span>
                        <span style="color:{diff_color(h_diff)};font-size:0.75rem;margin-left:8px;">FDR {h_diff}</span>
                    </div>
                    <div style="margin:0 1.5rem;color:#555;font-family:'Bebas Neue',sans-serif;font-size:1.2rem;letter-spacing:2px;">VS</div>
                    <div style="flex:1;text-align:left;">
                        <span style="color:{diff_color(a_diff)};font-size:0.75rem;margin-right:8px;">FDR {a_diff}</span>
                        <span style="color:#fff;font-weight:600;font-size:1rem;">{away}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)
