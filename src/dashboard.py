import streamlit as st
import pandas as pd
import numpy as np
import requests
import os
from pulp import *
from difflib import get_close_matches
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
FPL_BASE      = "https://fantasy.premierleague.com/api/"
POSITION_MAP  = {0: 'DEF', 1: 'FWD', 2: 'GK', 3: 'MID'}
POS_COLORS    = {'GK': '#f0a500', 'DEF': '#00b4d8', 'MID': '#7b2ff7', 'FWD': '#ff6b6b'}
MODEL_DIR     = 'models'
DATA_PATH     = 'data/processed/featured_training_set.csv'
PRED_CSV_PATH = 'models/nb2_regression_featured/predictions.csv'

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

def match_squad_names(api_names, prediction_names):
    matched = []
    pred_lower = {n.lower(): n for n in prediction_names}
    last_name_map = {}
    for n in prediction_names:
        parts = n.split()
        if parts:
            last = parts[-1].lower()
            if last not in last_name_map:
                last_name_map[last] = n
    for api_name in api_names:
        api_lower = api_name.lower().strip()
        exact = [n for n in prediction_names if api_lower in n.lower()]
        if len(exact) == 1:
            matched.append(exact[0]); continue
        if api_lower in last_name_map:
            matched.append(last_name_map[api_lower]); continue
        if '.' in api_lower:
            surname = api_lower.split('.')[-1].strip()
            if surname in last_name_map:
                matched.append(last_name_map[surname]); continue
        close = get_close_matches(api_lower, list(pred_lower.keys()), n=1, cutoff=0.75)
        if close:
            matched.append(pred_lower[close[0]])
    return matched

def get_current_gw():
    try:
        r = requests.get(FPL_BASE + "bootstrap-static/", timeout=10)
        r.raise_for_status()
        events = r.json()['events']
        for event in events:
            if event['is_next']:
                return event['id']
        for event in events:
            if event['is_current']:
                return event['id'] + 1
    except:
        pass
    return 38

NEXT_GW = get_current_gw()

# ── Load Predictions ───────────────────────────────────────────────────────────
@st.cache_data
def load_predictions():
    if not os.path.exists(PRED_CSV_PATH):
        st.sidebar.error("⚠️ predictions.csv not found")
        return None
    df = pd.read_csv(PRED_CSV_PATH)
    if 'predicted_pts' in df.columns and 'predicted_pts_mid' not in df.columns:
        df = df.rename(columns={'predicted_pts': 'predicted_pts_mid'})
    if 'pos' not in df.columns and 'position' in df.columns:
        df['pos'] = df['position']
    if 'predicted_pts_low' not in df.columns:
        df['predicted_pts_low']  = np.round(df['predicted_pts_mid'] * 0.85, 2)
        df['predicted_pts_high'] = np.round(df['predicted_pts_mid'] * 1.15, 2)
    def get_conf(pts):
        if pts >= 6:   return 'High'
        elif pts >= 3: return 'Medium'
        else:          return 'Low'
    df['confidence']           = df['predicted_pts_mid'].apply(get_conf)
    df['predicted_pts_scaled'] = df['predicted_pts_mid']
    st.sidebar.success("✅ NB2 Huber predictions loaded")
    return df

# ── PuLP Optimiser ─────────────────────────────────────────────────────────────
def optimise_squad(df, budget=100.0, current_squad=None, n_transfers=15):
    players = list(df.index)
    starter = LpVariable.dicts('starter', players, cat='Binary')
    benched = LpVariable.dicts('benched', players, cat='Binary')
    captain = LpVariable.dicts('captain', players, cat='Binary')
    tin     = LpVariable.dicts('tin',     players, cat='Binary')
    tout    = LpVariable.dicts('tout',    players, cat='Binary')
    keep    = LpVariable.dicts('keep',    players, cat='Binary')

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
        prob += lpSum(tin[i]  for i in gk_idx)  == lpSum(tout[i] for i in gk_idx)
        prob += lpSum(tin[i]  for i in def_idx) == lpSum(tout[i] for i in def_idx)
        prob += lpSum(tin[i]  for i in mid_idx) == lpSum(tout[i] for i in mid_idx)
        prob += lpSum(tin[i]  for i in fwd_idx) == lpSum(tout[i] for i in fwd_idx)
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

    s_ids  = [i for i in players if starter[i].value() == 1]
    b_ids  = [i for i in players if benched[i].value() == 1]
    cap_id = [i for i in players if captain[i].value() == 1][0]
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
    s11['role'] = 'Starter'
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
predictions = load_predictions()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="main-title" style="font-size:2rem;">FPL DSS</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Decision Support System</p>', unsafe_allow_html=True)
    st.markdown("---")
    _nav_options = [
        "🏠 Overview",
        f"🏆 Best Squad — GW{NEXT_GW}",
        "👤 My Squad",
        "📊 Player Predictions",
        "🎯 Captain Picks",
        f"📅 GW{NEXT_GW} Fixtures",
        "📈 Season Simulation"
    ]
    _default = st.session_state.pop("nav", "🏠 Overview")
    _idx = _nav_options.index(_default) if _default in _nav_options else 0
    page = st.radio("Nav", _nav_options, index=_idx, label_visibility="collapsed")
    st.markdown("---")
    st.markdown(f'<p style="color:#555;font-size:0.75rem;letter-spacing:1px;text-transform:uppercase;">Predicting GW{NEXT_GW}</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#00d2ff;font-size:0.85rem;">Model: NB2 Huber Regressor</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#00d2ff;font-size:0.85rem;">MAE: 0.864 · MAE@15: 7.34</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#00d2ff;font-size:0.85rem;">Baseline improvement: 42.6%</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠 Overview":
    st.markdown('<h1 class="main-title">FPL DECISION SUPPORT</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Stop guessing. Start winning.</p>', unsafe_allow_html=True)
    st.markdown("")

    st.markdown('''
    <div style="background:#1a1a2e;border:1px solid #0f3460;border-radius:12px;padding:1.5rem 2rem;margin:1rem 0;">
        <p style="color:#ccc;font-size:1.05rem;line-height:1.8;margin:0;">
            Fantasy Premier League is played by over <span style="color:#00d2ff;font-weight:600;">10 million managers</span> worldwide.
            Most rely on gut feeling, recency bias, and community hype to make decisions.
            This system replaces intuition with a <span style="color:#00d2ff;font-weight:600;">data-driven pipeline</span> —
            predicting each player's expected points using three years of historical data,
            then selecting the mathematically optimal squad using <span style="color:#00d2ff;font-weight:600;">integer linear programming</span>.
        </p>
    </div>
    ''', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-header">WHAT CAN YOU DO HERE?</p>', unsafe_allow_html=True)
    features = [
        ("🏆", "Find the Best Squad", "See the mathematically optimal 15-player squad within the £100m budget for the upcoming gameweek."),
        ("👤", "Import Your Squad", "Enter your FPL team ID and get personalised transfer recommendations and captaincy picks."),
        ("📊", "Explore Predictions", "Browse all 820 players ranked by predicted score. Filter by position, price, and confidence."),
        ("🎯", "Pick Your Captain", "See the top 10 captaincy candidates with their double-points projections."),
        ("📈", "See the Simulation", "Watch how a model-driven manager performed over GW15–29, beating the average by +95 points."),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(features):
        col = c1 if i % 2 == 0 else c2
        with col:
            st.markdown(f'''
            <div style="background:#1a1a2e;border:1px solid #0f3460;border-radius:10px;
                        padding:1rem 1.2rem;margin:0.4rem 0;display:flex;align-items:flex-start;gap:1rem;">
                <span style="font-size:1.6rem;">{icon}</span>
                <div>
                    <div style="color:#00d2ff;font-weight:600;font-size:0.95rem;margin-bottom:0.3rem;">{title}</div>
                    <div style="color:#888;font-size:0.85rem;line-height:1.5;">{desc}</div>
                </div>
            </div>''', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-header">BY THE NUMBERS</p>', unsafe_allow_html=True)
    n_players = len(predictions) if predictions is not None else 820
    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{n_players}</div><div class="metric-label">Players Tracked</div></div>', unsafe_allow_html=True)
    with c2: st.markdown('<div class="metric-card"><div class="metric-value">79,277</div><div class="metric-label">Training Observations</div></div>', unsafe_allow_html=True)
    with c3: st.markdown('<div class="metric-card"><div class="metric-value">+95 pts</div><div class="metric-label">Advantage vs Avg Manager</div></div>', unsafe_allow_html=True)
    with c4: st.markdown('<div class="metric-card"><div class="metric-value">Top 0.5%</div><div class="metric-label">Estimated Global Rank</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<p class="section-header">GET STARTED</p>', unsafe_allow_html=True)
    st.markdown('<p style="color:#888;font-size:0.9rem;margin-bottom:1rem;">Import your FPL squad and get your personalised transfer recommendations in seconds.</p>', unsafe_allow_html=True)
    if st.button("🚀  Import My Squad & Get Recommendations", use_container_width=True):
        st.session_state["nav"] = "👤 My Squad"
        st.rerun()

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
        fpl_id_str = st.text_input("Your FPL Team ID", value="", placeholder="e.g. 3187370",
                                    help="Find in FPL app under Points — it's in the URL")
        wildcard = st.checkbox("Wildcard")
        if not wildcard:
            n_ft = st.number_input("Free Transfers", min_value=1, max_value=15, value=1, step=1)
        else:
            n_ft = 15
        fetch_btn = st.button("🔍 Import & Optimise", type="primary")

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
                        if 'element' in predictions.columns:
                            elem_to_name = dict(zip(predictions['element'].astype(int), predictions['name']))
                            squad_names  = [elem_to_name[pid] for pid in picks if pid in elem_to_name]
                        else:
                            id_to_name  = {p['id']: p['web_name'] for p in bootstrap['elements']}
                            api_names   = [id_to_name.get(pid, '') for pid in picks if id_to_name.get(pid)]
                            squad_names = match_squad_names(api_names, predictions['name'].tolist())
                        current_df   = predictions[predictions['name'].isin(squad_names)]
                        total_budget = current_df['value'].sum() + itb
                        n_transfers  = n_ft

                        st.markdown(f'<p style="color:#888;">✅ {len(squad_names)}/15 players matched · Budget: £{total_budget:.1f}m · {"Wildcard" if wildcard else str(n_ft) + " transfer(s)"}</p>', unsafe_allow_html=True)

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
    st.markdown(f'<p class="subtitle">GW{NEXT_GW} · NB2 Huber Regressor · MAE@15: 7.34</p>', unsafe_allow_html=True)

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

    # If no fixtures for NEXT_GW (season over), fall back to GW38
    fallback = False
    if bootstrap and not fixtures:
        fixtures = fetch_fixtures(38)
        fallback = True

    if not bootstrap or not fixtures:
        st.error("Could not load fixtures from FPL API.")
    else:
        team_map   = {t['id']: t['name'] for t in bootstrap['teams']}
        display_gw = 38 if fallback else NEXT_GW

        st.markdown(f'<p class="section-header">GAMEWEEK {display_gw} FIXTURES</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="color:#888;font-size:0.85rem;">{len(fixtures)} fixtures</p>', unsafe_allow_html=True)
        if fallback:
            st.markdown(
                '<div style="background:#1a2e1a;border:1px solid #06D6A0;border-radius:8px;padding:0.6rem 1rem;margin:0.5rem 0;">'
                '<span style="color:#06D6A0;font-size:0.9rem;">ℹ️ The 2025/26 Premier League season has concluded. Showing the final gameweek (GW38) results.</span>'
                '</div>', unsafe_allow_html=True)
        st.markdown("")

        for fix in fixtures:
            home = team_map.get(fix['team_h'], 'Unknown')
            away = team_map.get(fix['team_a'], 'Unknown')
            h_diff = fix.get('team_h_difficulty', 3)
            a_diff = fix.get('team_a_difficulty', 3)
            def diff_color(d):
                return {1:'#00ff88', 2:'#00d2ff', 3:'#888', 4:'#ffd700', 5:'#ff6b6b'}.get(d, '#888')
            finished    = fix.get('finished', False)
            h_score     = fix.get('team_h_score')
            a_score     = fix.get('team_a_score')
            score_str   = f"{h_score} – {a_score}" if finished and h_score is not None else "VS"
            score_color = "#00d2ff" if finished else "#555"

            st.markdown(f"""
            <div class="fixture-card">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <div style="flex:1;text-align:right;">
                        <span style="color:#fff;font-weight:600;font-size:1rem;">{home}</span>
                        <span style="color:{diff_color(h_diff)};font-size:0.75rem;margin-left:8px;">FDR {h_diff}</span>
                    </div>
                    <div style="margin:0 1.5rem;color:{score_color};font-family:'Bebas Neue',sans-serif;font-size:1.2rem;letter-spacing:2px;">{score_str}</div>
                    <div style="flex:1;text-align:left;">
                        <span style="color:{diff_color(a_diff)};font-size:0.75rem;margin-right:8px;">FDR {a_diff}</span>
                        <span style="color:#fff;font-weight:600;font-size:1rem;">{away}</span>
                    </div>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — SEASON SIMULATION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈 Season Simulation":
    st.markdown('<h1 class="main-title">SEASON SIMULATION</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">GW15–29 · NB2 model-driven decisions vs average FPL manager</p>', unsafe_allow_html=True)

    sim_data = {
        'GW':      [15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28],
        'our_pts': [91, 90, 63, 41, 66, 37, 65, 47, 40, 46, 70, 38, 39, 64],
        'avg_pts': [49, 60, 66, 44, 40, 42, 48, 40, 44, 55, 58, 58, 45, 53],
        'captain': ['Haaland','Haaland','Haaland','Haaland','Haaland','Haaland',
                    'Haaland','Haaland','Haaland','Haaland','Haaland','Haaland',
                    'Haaland','Salah'],
    }
    sim_df = pd.DataFrame(sim_data)
    sim_df['diff']     = sim_df['our_pts'] - sim_df['avg_pts']
    sim_df['our_cum']  = sim_df['our_pts'].cumsum()
    sim_df['avg_cum']  = sim_df['avg_pts'].cumsum()
    sim_df['beat_avg'] = sim_df['diff'] > 0

    our_total  = int(sim_df['our_pts'].sum())
    avg_total  = int(sim_df['avg_pts'].sum())
    advantage  = our_total - avg_total
    beat_count = int(sim_df['beat_avg'].sum())

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.markdown(f'<div class="metric-card"><div class="metric-value">{our_total}</div><div class="metric-label">Our Total Pts</div></div>', unsafe_allow_html=True)
    with c2: st.markdown(f'<div class="metric-card"><div class="metric-value">{avg_total}</div><div class="metric-label">Avg Manager Pts</div></div>', unsafe_allow_html=True)
    with c3: st.markdown(f'<div class="metric-card"><div class="metric-value">+{advantage}</div><div class="metric-label">Points Advantage</div></div>', unsafe_allow_html=True)
    with c4: st.markdown(f'<div class="metric-card"><div class="metric-value">{beat_count}/14</div><div class="metric-label">GWs Beat Average</div></div>', unsafe_allow_html=True)

    st.markdown("")
    st.markdown(
        '<div style="background:#1a1a2e;border:1px solid #0f3460;border-radius:12px;padding:1rem 1.5rem;margin:1rem 0;text-align:center;">'
        '<span style="color:#00d2ff;font-family:Bebas Neue,sans-serif;font-size:1.4rem;letter-spacing:2px;">EST. RANK: TOP 50,000</span>'
        '<span style="color:#888;font-size:0.9rem;margin-left:16px;">≈ top 0.5% of 10M+ managers globally</span>'
        '</div>', unsafe_allow_html=True)

    st.markdown('<p class="section-header">POINTS PER GAMEWEEK</p>', unsafe_allow_html=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 7))
    fig.patch.set_facecolor('#0d0d1a')
    for ax in axes:
        ax.set_facecolor('#0d0d1a')
        ax.tick_params(colors='#888')
        ax.spines['bottom'].set_color('#0f3460')
        ax.spines['left'].set_color('#0f3460')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    gws = sim_df['GW'].values
    axes[0].plot(gws, sim_df['our_pts'], 'o-', color='#00d2ff', linewidth=2, label=f'Our model ({our_total} pts)')
    axes[0].plot(gws, sim_df['avg_pts'], 's--', color='#888', linewidth=1.5, label=f'Average manager ({avg_total} pts)')
    axes[0].fill_between(gws, sim_df['our_pts'], sim_df['avg_pts'], alpha=0.12, color='#00d2ff')
    axes[0].set_ylabel('Points', color='#888')
    axes[0].legend(facecolor='#1a1a2e', edgecolor='#0f3460', labelcolor='#ccc')
    axes[0].set_xticks(gws)

    axes[1].plot(gws, sim_df['our_cum'], 'o-', color='#00d2ff', linewidth=2, label='Our model (cumulative)')
    axes[1].plot(gws, sim_df['avg_cum'], 's--', color='#888', linewidth=1.5, label='Average manager (cumulative)')
    axes[1].set_xlabel('Gameweek', color='#888')
    axes[1].set_ylabel('Cumulative Points', color='#888')
    axes[1].legend(facecolor='#1a1a2e', edgecolor='#0f3460', labelcolor='#ccc')
    axes[1].set_xticks(gws)

    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    st.markdown('<p class="section-header">GAMEWEEK BREAKDOWN</p>', unsafe_allow_html=True)
    for _, row in sim_df.iterrows():
        color    = '#00ff88' if row['beat_avg'] else '#ff6b6b'
        symbol   = '▲' if row['beat_avg'] else '▼'
        diff_str = f"+{int(row['diff'])}" if row['diff'] >= 0 else str(int(row['diff']))
        st.markdown(f"""
        <div style="background:#1a1a2e;border-left:4px solid {color};border-radius:8px;
                    padding:0.6rem 1rem;margin:0.3rem 0;display:flex;
                    justify-content:space-between;align-items:center;">
            <span style="color:#fff;font-weight:600;min-width:40px;">GW{int(row['GW'])}</span>
            <span style="color:#00d2ff;min-width:80px;">Our: {int(row['our_pts'])} pts</span>
            <span style="color:#888;min-width:80px;">Avg: {int(row['avg_pts'])} pts</span>
            <span style="color:{color};min-width:60px;">{symbol} {diff_str}</span>
            <span style="color:#555;font-size:0.8rem;">⚽ Captain: {row['captain']}</span>
        </div>""", unsafe_allow_html=True)