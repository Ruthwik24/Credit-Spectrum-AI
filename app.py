"""
Credit Spectrum AI — Intelligent Credit Eligibility Platform (Light & Colorful Edition)
Powered by XGBoost | Secure • Intelligent • Trusted
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier
import pickle
import os
import time

# =====================================================================================
# 1. PAGE CONFIG
# =====================================================================================
st.set_page_config(
    page_title="Credit Spectrum AI | Intelligent Credit Eligibility Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "mood" not in st.session_state:
    st.session_state.mood = "default"   # default | approved | declined

# =====================================================================================
# 2. THEME — light, colorful, mood-reactive glassmorphism
# =====================================================================================
def apply_theme(mood="default"):
    # Futuristic dark neon-glass palettes. All text colors are chosen for strong
    # contrast against the dark backgrounds (WCAG-friendly light tones).
    palettes = {
        "default": dict(
            bg="radial-gradient(circle at 15% 10%, #12163a 0%, #0a0e27 35%, #06081a 70%, #030410 100%)",
            accent1="#00e5ff",   # electric cyan
            accent2="#b026ff",   # neon violet
            accent3="#ff2fd0",   # magenta
            text="#eef1ff",
            textDim="#aab2d8",
            cardBg="rgba(20,24,58,0.55)",
            cardBorder="rgba(0,229,255,0.35)",
            heroGrad="linear-gradient(120deg, rgba(0,229,255,0.16), rgba(176,38,255,0.16) 55%, rgba(255,47,208,0.14))",
            titleGrad="linear-gradient(90deg, #00e5ff 0%, #b026ff 55%, #ff2fd0 100%)",
            glow="0,229,255",
        ),
        "approved": dict(
            bg="radial-gradient(circle at 15% 10%, #0a2e22 0%, #06201a 35%, #041712 70%, #020b09 100%)",
            accent1="#00ffb2",
            accent2="#00e5ff",
            accent3="#7cffb0",
            text="#eafff5",
            textDim="#a9e6c9",
            cardBg="rgba(10,50,38,0.55)",
            cardBorder="rgba(0,255,178,0.35)",
            heroGrad="linear-gradient(120deg, rgba(0,255,178,0.20), rgba(0,229,255,0.14))",
            titleGrad="linear-gradient(90deg, #00ffb2 10%, #00e5ff 60%, #7cffb0 100%)",
            glow="0,255,178",
        ),
        "declined": dict(
            bg="radial-gradient(circle at 15% 10%, #3a0a16 0%, #260610 35%, #17040b 70%, #0b0206 100%)",
            accent1="#ff2f6e",
            accent2="#ff6a00",
            accent3="#ff2fd0",
            text="#fff0f3",
            textDim="#e6a9bb",
            cardBg="rgba(50,10,22,0.55)",
            cardBorder="rgba(255,47,110,0.35)",
            heroGrad="linear-gradient(120deg, rgba(255,47,110,0.20), rgba(255,106,0,0.14))",
            titleGrad="linear-gradient(90deg, #ff2f6e 10%, #ff6a00 60%, #ff2fd0 100%)",
            glow="255,47,110",
        ),
    }
    p = palettes[mood]

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500;700;800;900&family=Rajdhani:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Rajdhani', 'Inter', sans-serif; }}

    /* ---------- FULL-WIDTH LAYOUT ---------- */
    .block-container {{
        padding-top: 1rem !important;
        padding-bottom: 2rem !important;
        padding-left: 2.2rem !important;
        padding-right: 2.2rem !important;
        max-width: 100% !important;
    }}
    [data-testid="stAppViewContainer"] > .main {{ max-width: 100% !important; }}
    section.main {{ max-width: 100% !important; }}

    .stApp {{
        background: {p['bg']};
        background-attachment: fixed;
        color: {p['text']};
        transition: background 1s ease;
    }}

    .stApp::before {{
        content: "";
        position: fixed; inset: 0;
        background-image:
            radial-gradient(2.5px 2.5px at 18% 25%, {p['accent1']}77, transparent),
            radial-gradient(2.5px 2.5px at 75% 60%, {p['accent2']}66, transparent),
            radial-gradient(2px 2px at 85% 15%, {p['accent3']}66, transparent),
            radial-gradient(2px 2px at 40% 85%, {p['accent1']}55, transparent),
            radial-gradient(2px 2px at 60% 35%, {p['accent2']}44, transparent);
        background-size: 600px 600px;
        opacity: 0.8;
        pointer-events: none;
        z-index: 0;
        animation: drift 60s linear infinite;
    }}
    @keyframes drift {{ from{{background-position:0 0;}} to{{background-position:600px 600px;}} }}

    h1, h2, h3, h4 {{
        font-family: 'Orbitron', 'Rajdhani', sans-serif !important;
        color: {p['text']} !important;
        font-weight: 700 !important;
    }}
    p, span, div, label, li, td, th {{ color: {p['text']}; }}
    .stCaption, [data-testid="stCaptionContainer"] * {{ color: {p['textDim']} !important; }}
    ::selection {{ background: {p['accent1']}55; }}

    /* Force readable text inside Streamlit's own widget chrome */
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li,
    [data-testid="stMarkdownContainer"] span {{ color: {p['text']} !important; }}
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] * {{
        color: {p['text']} !important;
    }}
    .stTextInput input, .stNumberInput input {{
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid {p['cardBorder']} !important;
        border-radius: 10px !important;
    }}
    [data-baseweb="select"] > div {{
        background: rgba(255,255,255,0.06) !important;
        border-color: {p['cardBorder']} !important;
        border-radius: 10px !important;
    }}
    [data-testid="stDataFrame"] {{ border-radius: 14px; overflow: hidden; }}

    /* ---------- GLASS CARD ---------- */
    .glass-card {{
        background: {p['cardBg']};
        border: 1px solid {p['cardBorder']};
        border-radius: 20px;
        padding: 26px;
        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.35), inset 0 1px 0 rgba(255,255,255,0.05);
        transition: all 0.35s ease;
        margin-bottom: 18px;
        color: {p['text']};
    }}
    .glass-card:hover {{
        transform: translateY(-6px);
        border-color: {p['accent1']};
        box-shadow: 0 18px 42px rgba({p['glow']},0.22), inset 0 1px 0 rgba(255,255,255,0.08);
    }}
    .glass-card h3 {{ color: {p['text']} !important; }}
    .glass-card p {{ color: {p['textDim']} !important; }}

    /* ---------- HERO ---------- */
    .hero-wrap {{
        background: {p['heroGrad']};
        border: 1px solid {p['cardBorder']};
        border-radius: 28px;
        padding: 50px 42px;
        backdrop-filter: blur(20px);
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }}
    .hero-wrap::after {{
        content: ""; position: absolute; inset: -2px; border-radius: 28px; padding: 2px;
        background: {p['titleGrad']}; opacity: 0.5;
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude; pointer-events: none;
    }}
    .hero-title {{
        font-family: 'Orbitron', sans-serif;
        font-size: 46px;
        font-weight: 800;
        background: {p['titleGrad']};
        -webkit-background-clip: text;
        background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 6px;
        text-shadow: 0 0 30px rgba({p['glow']},0.25);
    }}
    .hero-sub {{ font-size: 19px; color: {p['text']}; opacity: 0.92; font-weight: 500; margin-bottom: 14px; }}
    .hero-tag {{
        display: inline-block; padding: 6px 16px; border-radius: 30px;
        background: rgba(255,255,255,0.06); border: 1px solid {p['accent1']};
        color: {p['accent1']}; font-size: 13px; letter-spacing: 1.5px; font-weight: 700;
    }}

    /* ---------- STAT CARDS ---------- */
    .stat-card {{
        background: {p['cardBg']};
        border: 1px solid {p['cardBorder']};
        border-radius: 18px;
        padding: 22px 10px;
        text-align: center;
        backdrop-filter: blur(14px);
        animation: floaty 5s ease-in-out infinite;
    }}
    .stat-card:nth-child(2) {{ animation-delay: 0.6s; }}
    .stat-card:nth-child(3) {{ animation-delay: 1.2s; }}
    @keyframes floaty {{ 0%,100%{{ transform: translateY(0);}} 50%{{ transform: translateY(-8px);}} }}
    .stat-number {{
        font-size: 34px; font-weight: 800; font-family: 'Orbitron', sans-serif;
        background: {p['titleGrad']};
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .stat-label {{ font-size: 13px; color: {p['textDim']}; letter-spacing: 0.6px; text-transform: uppercase; margin-top: 4px;}}

    /* ---------- BUTTONS ---------- */
    .stButton>button, .stFormSubmitButton>button {{
        background: linear-gradient(90deg, {p['accent1']}, {p['accent2']});
        color: #04040c; border: 1px solid rgba(255,255,255,0.25);
        border-radius: 14px; padding: 12px 20px; font-weight: 800;
        letter-spacing: 0.5px; width: 100%; font-family: 'Rajdhani', sans-serif;
        box-shadow: 0 8px 22px rgba({p['glow']},0.25);
        transition: all 0.25s ease;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{
        background: linear-gradient(90deg, {p['accent2']}, {p['accent3']});
        box-shadow: 0 12px 34px rgba({p['glow']},0.4);
        transform: translateY(-2px);
        color: #ffffff;
    }}

    /* ---------- TOP NAV BAR ---------- */
    .navbar-wrap {{
        background: {p['cardBg']};
        border: 1px solid {p['cardBorder']};
        border-radius: 20px;
        padding: 10px;
        margin-bottom: 6px;
        backdrop-filter: blur(20px);
        box-shadow: 0 8px 30px rgba(0,0,0,0.35);
    }}
    .navbar-wrap .stButton>button {{
        background: transparent;
        color: {p['textDim']} !important;
        border: 1px solid transparent;
        box-shadow: none;
        font-weight: 700;
        font-size: 15px;
        border-radius: 12px;
        padding: 12px 8px;
    }}
    .navbar-wrap .stButton>button:hover {{
        background: rgba(255,255,255,0.06);
        color: {p['text']} !important;
        transform: none;
        border-color: {p['cardBorder']};
    }}
    .nav-active .stButton>button {{
        background: linear-gradient(90deg, {p['accent1']}22, {p['accent2']}22) !important;
        border: 1px solid {p['accent1']} !important;
        color: {p['text']} !important;
        box-shadow: 0 0 20px rgba({p['glow']},0.35) inset, 0 0 16px rgba({p['glow']},0.25) !important;
    }}

    .statusbar {{
        display: flex; align-items: center; gap: 22px; flex-wrap: wrap;
        padding: 10px 18px; margin-bottom: 22px; font-size: 13.5px; color: {p['textDim']};
        border-top: 1px solid {p['cardBorder']};
    }}
    .statusbar b {{ color: {p['text']}; }}
    .pulse-dot {{
        width: 9px; height: 9px; border-radius: 50%; background: {p['accent1']};
        display: inline-block; margin-right: 6px;
        box-shadow: 0 0 8px {p['accent1']};
        animation: pulse 1.6s ease-in-out infinite;
    }}
    @keyframes pulse {{ 0%,100%{{opacity:1;}} 50%{{opacity:0.35;}} }}

    /* ---------- RESULT CARDS ---------- */
    .approved-hero {{
        background: linear-gradient(135deg, rgba(0,255,178,0.18), rgba(0,255,178,0.04));
        border: 1px solid #00ffb2;
        border-radius: 24px; padding: 36px; text-align: center;
        box-shadow: 0 0 50px rgba(0,255,178,0.30);
    }}
    .declined-hero {{
        background: linear-gradient(135deg, rgba(255,47,110,0.18), rgba(255,47,110,0.04));
        border: 1px solid #ff2f6e;
        border-radius: 24px; padding: 36px; text-align: center;
        box-shadow: 0 0 50px rgba(255,47,110,0.30);
    }}
    .verdict-title {{ font-family:'Orbitron',sans-serif; font-size: 28px; font-weight: 800; margin-bottom: 6px;}}
    .approved-hero .verdict-title {{ color: #00ffb2; }}
    .declined-hero .verdict-title {{ color: #ff2f6e; }}
    .approved-hero p, .declined-hero p {{ color: {p['text']} !important; }}

    .reason-chip {{
        display: inline-block; margin: 5px 6px; padding: 8px 14px;
        border-radius: 30px; background: rgba(255,255,255,0.07);
        border: 1px solid {p['cardBorder']}; font-size: 13.5px; font-weight: 600;
        color: {p['text']};
    }}

    /* ---------- TIMELINE ---------- */
    .timeline-item {{
        border-left: 3px solid {p['accent1']}; padding-left: 18px; margin-bottom: 18px; position: relative;
        color: {p['text']};
    }}
    .timeline-item::before {{
        content: ""; position: absolute; left: -8px; top: 2px;
        width: 13px; height: 13px; border-radius: 50%;
        background: {p['accent1']}; box-shadow: 0 0 10px {p['accent1']};
    }}

    /* ---------- DATASET TAB: GLASSY ANIMATIONS ---------- */
    .fade-up {{ animation: fadeUp 0.6s ease both; }}
    .fade-up.d1 {{ animation-delay: 0.05s; }}
    .fade-up.d2 {{ animation-delay: 0.15s; }}
    .fade-up.d3 {{ animation-delay: 0.25s; }}
    .fade-up.d4 {{ animation-delay: 0.35s; }}
    @keyframes fadeUp {{ from {{ opacity:0; transform: translateY(16px);}} to {{ opacity:1; transform: translateY(0);}} }}

    .vault-card {{
        background: {p['cardBg']};
        border: 1px solid {p['cardBorder']};
        border-radius: 22px;
        padding: 24px;
        backdrop-filter: blur(22px);
        -webkit-backdrop-filter: blur(22px);
        position: relative;
        overflow: hidden;
        transition: all 0.4s cubic-bezier(.2,.8,.2,1);
    }}
    .vault-card::before {{
        content:""; position:absolute; inset:0; border-radius:22px; padding:1px;
        background: linear-gradient(120deg, {p['accent1']}, transparent 30%, transparent 70%, {p['accent2']});
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor; mask-composite: exclude;
        opacity: 0.6;
    }}
    .vault-card:hover {{
        transform: translateY(-8px) scale(1.01);
        box-shadow: 0 20px 50px rgba({p['glow']},0.28);
    }}

    hr {{ border-color: {p['cardBorder']}; }}
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {p['accent2']}; border-radius: 10px; }}

    /* hide default streamlit sidebar entirely since nav is now top-level */
    [data-testid="stSidebar"] {{ display: none; }}
    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    </style>
    """, unsafe_allow_html=True)

apply_theme(st.session_state.mood)

FEATURE_COLS = ['purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths',
                'delinq.2yrs', 'pub.rec']

# =====================================================================================
# 3. DATA + AUTO-HEALING MODEL LOADING 
# =====================================================================================
@st.cache_resource(show_spinner="🏦 Booting Credit Spectrum AI Core (Auto-healing if needed)...")
def load_pipeline():
    df_raw = pd.read_csv('loan_data.csv')
    df_processed = df_raw.copy()

    le = LabelEncoder()
    df_processed['purpose'] = le.fit_transform(df_processed['purpose'])

    X = df_processed[FEATURE_COLS]
    y = df_processed['credit.policy']

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    scaler.fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)

    model_path = 'model.pkl'
    model = None

    # Attempt to load the model. If it's corrupted (Pickle error), train it fresh.
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception:
            pass # We will handle the fallback right below

    # If the model didn't load (missing or corrupted), train a new one and save it.
    if model is None:
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
        model.fit(X_train_scaled, y_train)
        
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

    return df_raw, df_processed, le, scaler, model, X_test_scaled, y_test


if not os.path.exists('loan_data.csv'):
    st.error("⚠️ SYSTEM FAULT: 'loan_data.csv' is missing. Please ensure it is in the directory.")
    st.stop()

df_raw, df_processed, le, scaler, model, X_test_scaled, y_test = load_pipeline()

# =====================================================================================
# 4. FULL-WIDTH TOP NAVIGATION
# =====================================================================================
menu = [
    "🏠 Home",
    "📊 Dashboard",
    "📈 Analytics",
    "🧠 AI Predictor",
    "📁 Dataset",
    "👨‍💻 About the Model",
]

if "nav_choice" not in st.session_state:
    st.session_state.nav_choice = menu[0]

def _set_nav(label):
    # Callback fires BEFORE the rerun, so the very first click already
    # updates session_state -> no more "click twice to select" behaviour.
    st.session_state.nav_choice = label

st.markdown(f"""
<div class="hero-wrap" style="padding:28px 36px; margin-bottom:14px;">
    <span class="hero-tag">POWERED BY EXPLAINABLE AI &nbsp;•&nbsp; XGBOOST &nbsp;•&nbsp; NEURAL CORE</span>
    <div class="hero-title" style="font-size:38px; margin-top:8px;">🏦 Credit Spectrum <span>AI</span></div>
    <div class="hero-sub" style="margin-bottom:0;">Secure • Intelligent • Trusted — Next-Gen Loan Intelligence Platform</div>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="navbar-wrap">', unsafe_allow_html=True)
nav_cols = st.columns(len(menu))
for col, label in zip(nav_cols, menu):
    with col:
        is_active = st.session_state.nav_choice == label
        if is_active:
            st.markdown('<div class="nav-active">', unsafe_allow_html=True)
        st.button(label, key=f"nav_{label}", on_click=_set_nav, args=(label,),
                  use_container_width=True)
        if is_active:
            st.markdown('</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

choice = st.session_state.nav_choice

# ---- slim status / download bar (replaces old sidebar) ----
status_cols = st.columns([2, 2, 2, 2, 2])
with status_cols[0]:
    st.markdown('<div class="statusbar"><span class="pulse-dot"></span><b>Core Status: ONLINE</b></div>',
                unsafe_allow_html=True)
with status_cols[1]:
    st.markdown('<div class="statusbar">⚙️ Engine: <b>XGBoost (Production)</b></div>', unsafe_allow_html=True)
with status_cols[2]:
    if os.path.exists("model.pkl"):
        with open("model.pkl", "rb") as f:
            st.download_button("⬇️ Model (.pkl)", data=f, file_name="astrabank_xgboost_model.pkl",
                                mime="application/octet-stream", use_container_width=True)
with status_cols[3]:
    st.download_button("⬇️ Dataset (.csv)", data=df_raw.to_csv(index=False),
                        file_name="loan_data.csv", mime="text/csv", use_container_width=True)
with status_cols[4]:
    if st.session_state.mood != "default":
        if st.button("🔄 Reset Theme to Default", use_container_width=True):
            st.session_state.mood = "default"
            st.rerun()

st.markdown("<div style='margin-bottom:8px;'></div>", unsafe_allow_html=True)

# =====================================================================================
# 5. HOME
# =====================================================================================
if choice == "🏠 Home":
    st.markdown("""
    <div class="glass-card fade-up" style="margin-bottom:24px;">
        <div class="hero-sub" style="margin-bottom:10px;">🪙 Next-Generation Loan Intelligence Platform for the Modern Digital Bank</div>
        <p style="max-width:720px; opacity:0.9;">
            Analyze a customer's complete financial profile, quantify credit risk and predict
            loan eligibility in milliseconds — with full transparency into every decision. 🔐🧾📈
        </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🚀 Start Prediction"):
            st.info("Use the sidebar → **🧠 AI Predictor** to analyze an applicant.")
    with c2:
        if st.button("📊 Explore Dashboard"):
            st.info("Use the sidebar → **📊 Dashboard** to explore the data holograms.")

    st.markdown("### ")
    s1, s2, s3, s4 = st.columns(4)
    stats = [
        ("🎯 98.9%", "Prediction Accuracy"),
        ("📈 99.6%", "ROC AUC Score"),
        ("🧮 12", "Financial Signals Analyzed"),
        ("⚡ XGBoost", "Production Engine"),
    ]
    for col, (num, lbl) in zip([s1, s2, s3, s4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <div class="stat-number">{num}</div>
                <div class="stat-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("###  ")
    st.markdown("""
    <div class="glass-card">
        <h3>🛡️ Why Credit Spectrum AI?</h3>
        <p>Built for institutions that need speed <b>and</b> trust — every prediction is paired with
        a confidence score and a plain-language explanation, so risk teams and customers alike
        understand exactly why a decision was made.</p>
    </div>
    """, unsafe_allow_html=True)

    g1, g2, g3 = st.columns(3)
    for col, (icon, title, desc) in zip(
        [g1, g2, g3],
        [
            ("⚡", "Millisecond Decisions", "Real-time inference on live applicant telemetry."),
            ("🔐", "Bank-Grade Security", "Data is processed in-session, never persisted externally."),
            ("🧭", "Explainable Outcomes", "Feature-level reasoning behind every approval or decline."),
        ],
    ):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center;">
                <div style="font-size:34px;">{icon}</div>
                <h3 style="font-size:18px;">{title}</h3>
                <p style="font-size:13.5px;">{desc}</p>
            </div>""", unsafe_allow_html=True)

# =====================================================================================
# 6. DASHBOARD (EDA)
# =====================================================================================
elif choice == "📊 Dashboard":
    st.markdown('<div class="hero-title" style="font-size:32px;">📊 Finance Holograms</div>', unsafe_allow_html=True)
    st.caption("Interactive exploration of the applicant financial data structure.")

    warm_scale = ["#f6a509", "#ef4d8b", "#8b5cf6", "#fb923c", "#f472b6", "#c084fc"]

    col1, col2 = st.columns(2)
    with col1:
        purpose_counts = df_raw['purpose'].value_counts().reset_index()
        purpose_counts.columns = ['purpose', 'count']
        fig1 = px.bar(purpose_counts, x='purpose', y='count', color='purpose',
                      title="💼 Capital Allocation by Loan Purpose", template="plotly_white",
                      color_discrete_sequence=warm_scale)
        fig1.update_layout(showlegend=False, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig1, use_container_width=True)
    with col2:
        policy_counts = df_raw['credit.policy'].value_counts().reset_index()
        policy_counts.columns = ['credit.policy', 'count']
        policy_counts['credit.policy'] = policy_counts['credit.policy'].map({1: 'Meets Policy', 0: 'Below Policy'})
        fig2 = px.pie(policy_counts, values='count', names='credit.policy', hole=0.55,
                      title="🛡️ Credit Policy Compliance Ratio", template="plotly_white",
                      color='credit.policy',
                      color_discrete_map={'Meets Policy': '#22c55e', 'Below Policy': '#f43f5e'})
        fig2.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig2, use_container_width=True)

    fig3 = px.histogram(df_raw, x='fico', color='credit.policy', marginal='box',
                         title="💎 FICO Trust Score Distribution", template="plotly_white", nbins=50,
                         color_discrete_sequence=['#f43f5e', '#22c55e'])
    fig3.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig3, use_container_width=True)

    d1, d2 = st.columns(2)
    with d1:
        fig5 = px.histogram(df_raw, x='dti', color='credit.policy', template="plotly_white",
                             title="⚖️ Debt-to-Income Distribution",
                             color_discrete_sequence=['#f43f5e', '#22c55e'])
        fig5.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig5, use_container_width=True)
    with d2:
        fig6 = px.box(df_raw, x='purpose', y='int.rate', color='credit.policy', template="plotly_white",
                       title="💸 Interest Rate by Purpose & Approval",
                       color_discrete_sequence=['#f43f5e', '#22c55e'])
        fig6.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig6, use_container_width=True)

    st.subheader("🕸️ Feature Neural-Link (Correlation Matrix)")
    corr = df_processed.corr()
    fig4 = go.Figure(data=go.Heatmap(z=corr.values, x=corr.columns, y=corr.columns, colorscale='Sunsetdark'))
    fig4.update_layout(template="plotly_white", height=560, paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig4, use_container_width=True)

# =====================================================================================
# 7. ANALYTICS
# =====================================================================================
elif choice == "📈 Analytics":
    st.markdown('<div class="hero-title" style="font-size:32px;">📈 Model Analytics</div>', unsafe_allow_html=True)
    st.caption("Live diagnostics of the deployed XGBoost engine on the holdout test stream.")

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    c1, c2 = st.columns(2)
    with c1:
        fig_acc = go.Figure(go.Indicator(
            mode="gauge+number", value=acc * 100, title={'text': "Accuracy (%)"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#f6a509"},
                   'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 1, 'bordercolor': "#eee"}))
        fig_acc.update_layout(template="plotly_white", height=290, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_acc, use_container_width=True)
    with c2:
        fig_auc = go.Figure(go.Indicator(
            mode="gauge+number", value=auc * 100, title={'text': "ROC AUC (%)"},
            gauge={'axis': {'range': [None, 100]}, 'bar': {'color': "#ef4d8b"},
                   'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 1, 'bordercolor': "#eee"}))
        fig_auc.update_layout(template="plotly_white", height=290, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_auc, use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='XGBoost',
                                      line=dict(color='#8b5cf6', width=3)))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline',
                                      line=dict(color='#ccc', dash='dash')))
        fig_roc.update_layout(title="ROC Curve", template="plotly_white", height=340,
                               paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_roc, use_container_width=True)
    with r2:
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = px.imshow(cm, text_auto=True, template="plotly_white",
                            color_continuous_scale="Oranges",
                            labels=dict(x="Predicted", y="Actual"), title="Confusion Matrix")
        fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("### ⚔️ Algo-Arena: Model Combat")
    metrics_df = pd.DataFrame({
        "AI Architecture": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost ⭐"],
        "Test Accuracy": [0.8569, 0.9881, 0.9840, 0.9892],
        "Test ROC AUC": [0.9341, 0.9829, 0.9969, 0.9978],
    })
    fig = px.bar(metrics_df, x='AI Architecture', y=['Test Accuracy', 'Test ROC AUC'],
                 barmode='group', template='plotly_white', title="🏆 Combat Results",
                 color_discrete_sequence=['#f6a509', '#ef4d8b'])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="glass-card">
    🧬 <b>Radar Selection:</b> The tuned <b>XGBoost</b> model was promoted to production for its
    superior ROC AUC generalization and stable precision–recall trade-off on unseen data streams.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🧠 AI Explainability — Global Feature Importance")
    try:
        importances = model.feature_importances_
        imp_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances}).sort_values(
            "Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h',
                          template="plotly_white", color="Importance",
                          color_continuous_scale=["#fde68a", "#ef4d8b"])
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", height=440)
        st.plotly_chart(fig_imp, use_container_width=True)
    except Exception:
        st.info("Feature importance is unavailable for this model instance.")

# =====================================================================================
# 8. AI PREDICTOR
# =====================================================================================
elif choice == "🧠 AI Predictor":
    st.markdown('<div class="hero-title" style="font-size:32px;">🔮 The AI Credit Radar</div>', unsafe_allow_html=True)
    st.caption("Enter applicant telemetry below. Credit Spectrum AI will assess credit-policy eligibility in real time.")

    with st.form("prediction_form"):
        st.markdown('<div class="glass-card"><h3>📋 Section 1 — Applicant & Loan Details</h3>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            purpose_options = list(le.classes_)
            purpose = st.selectbox('💼 Purpose', purpose_options)
        with c2:
            int_rate = st.number_input('💸 Interest Rate', min_value=0.0, max_value=1.0, value=0.10)
        with c3:
            installment = st.number_input('💳 Monthly Installment ($)', min_value=0.0, max_value=2000.0, value=250.0)
        with c4:
            log_annual_inc = st.number_input('💰 Log Annual Income', min_value=0.0, max_value=15.0, value=10.5)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><h3>💎 Section 2 — Credit History</h3>', unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        with c5:
            fico = st.number_input('💎 FICO Score', min_value=300, max_value=850, value=700)
        with c6:
            dti = st.number_input('⚖️ Debt-to-Income (DTI)', min_value=0.0, max_value=50.0, value=15.0)
        with c7:
            days_with_cr_line = st.number_input('⏳ Credit Line Age (Days)', min_value=0.0, value=4000.0)
        with c8:
            pub_rec = st.number_input('🚨 Public Records', min_value=0, max_value=10, value=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown('<div class="glass-card"><h3>📈 Section 3 — Revolving & Inquiry Behavior</h3>', unsafe_allow_html=True)
        c9, c10, c11, c12 = st.columns(4)
        with c9:
            revol_bal = st.number_input('🏦 Revolving Balance ($)', min_value=0.0, value=10000.0)
        with c10:
            revol_util = st.number_input('📈 Revolving Utilization (%)', min_value=0.0, max_value=150.0, value=45.0)
        with c11:
            inq_last_6mths = st.number_input('🕵️ Inquiries (6M)', min_value=0, max_value=50, value=1)
        with c12:
            delinq_2yrs = st.number_input('⚠️ Delinquencies (2Y)', min_value=0, max_value=20, value=0)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("🔍 ANALYZE APPLICANT")

    if submit_button:
        steps = [
            "Connecting to Credit Spectrum AI...",
            "Loading XGBoost engine...",
            "Evaluating financial profile...",
            "Checking credit policy thresholds...",
            "Generating confidence score...",
            "Finalizing recommendation...",
        ]
        prog = st.progress(0)
        status = st.empty()
        for i, step in enumerate(steps):
            status.markdown(f"🛰️ **{step}**")
            time.sleep(0.35)
            prog.progress(int((i + 1) / len(steps) * 100))
        status.empty()

        purp_enc = le.transform([purpose])[0]
        input_data = np.array([[purp_enc, int_rate, installment, log_annual_inc, dti, fico,
                                 days_with_cr_line, revol_bal, revol_util, inq_last_6mths,
                                 delinq_2yrs, pub_rec]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0]

        # Flip the entire app's mood/theme based on the verdict, then re-render styles
        st.session_state.mood = "approved" if prediction == 1 else "declined"
        apply_theme(st.session_state.mood)

        st.markdown("---")

        if prediction == 1:
            st.balloons()
            st.markdown(f"""
            <div class="approved-hero">
                <div style="font-size:44px;">✅🎉</div>
                <div class="verdict-title">Congratulations! Loan Approved</div>
                <p style="font-size:16px;">Confidence Score: <b>{confidence[1]*100:.2f}%</b></p>
            </div>""", unsafe_allow_html=True)

            reasons = []
            if fico >= 700: reasons.append("✔ Excellent FICO Score")
            if dti <= 15: reasons.append("✔ Low Debt-to-Income Ratio")
            if delinq_2yrs == 0: reasons.append("✔ No Recent Delinquencies")
            if revol_util <= 50: reasons.append("✔ Healthy Credit Utilization")
            if inq_last_6mths <= 2: reasons.append("✔ Low Recent Credit Inquiries")
            if not reasons: reasons = ["✔ Overall Profile Within Policy Thresholds"]

            st.markdown("<div style='text-align:center; margin-top:14px;'>" +
                        "".join(f"<span class='reason-chip'>{r}</span>" for r in reasons) +
                        "</div>", unsafe_allow_html=True)
        else:
            st.snow()
            st.markdown(f"""
            <div class="declined-hero">
                <div style="font-size:44px;">🚫💔</div>
                <div class="verdict-title">Loan Not Approved</div>
                <p style="font-size:16px;">Risk Confidence: <b>{confidence[0]*100:.2f}%</b></p>
            </div>""", unsafe_allow_html=True)

            tips = []
            if fico < 700: tips.append("📈 Improve your FICO score")
            if dti > 15: tips.append("⚖️ Reduce debt-to-income ratio")
            if revol_util > 50: tips.append("💳 Lower credit utilization")
            if inq_last_6mths > 2: tips.append("🕵️ Avoid multiple recent inquiries")
            if delinq_2yrs > 0: tips.append("⚠️ Maintain a clean repayment history")
            if not tips: tips = ["🧭 Re-evaluate overall financial profile"]

            st.markdown("<div style='text-align:center; margin-top:14px;'>" +
                        "".join(f"<span class='reason-chip'>{t}</span>" for t in tips) +
                        "</div>", unsafe_allow_html=True)

        g_val = confidence[1] * 100
        fig_g = go.Figure(go.Indicator(
            mode="gauge+number", value=g_val, title={'text': "Eligibility Meter"},
            gauge={'axis': {'range': [0, 100]},
                   'bar': {'color': "#22c55e" if prediction == 1 else "#f43f5e"},
                   'steps': [{'range': [0, 40], 'color': 'rgba(244,63,94,0.20)'},
                             {'range': [40, 70], 'color': 'rgba(246,165,9,0.20)'},
                             {'range': [70, 100], 'color': 'rgba(34,197,94,0.20)'}],
                   'bgcolor': "rgba(0,0,0,0)"}))
        fig_g.update_layout(template="plotly_white", height=300, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_g, use_container_width=True)

        report_df = pd.DataFrame([{
            "purpose": purpose, "int.rate": int_rate, "installment": installment,
            "log.annual.inc": log_annual_inc, "dti": dti, "fico": fico,
            "days.with.cr.line": days_with_cr_line, "revol.bal": revol_bal,
            "revol.util": revol_util, "inq.last.6mths": inq_last_6mths,
            "delinq.2yrs": delinq_2yrs, "pub.rec": pub_rec,
            "prediction": "Approved" if prediction == 1 else "Declined",
            "confidence_%": round(confidence[1] * 100 if prediction == 1 else confidence[0] * 100, 2),
        }])
        st.download_button("📄 Download Prediction Report (CSV)", data=report_df.to_csv(index=False),
                            file_name="astrabank_prediction_report.csv", mime="text/csv")

        st.info("🔄 Theme is now reflecting this verdict. Use **Reset Theme to Default** in the sidebar, "
                "or run a new prediction, to change it back.")

# =====================================================================================
# 9. DATASET
# =====================================================================================
elif choice == "📁 Dataset":
    st.markdown("""
    <div class="hero-wrap fade-up" style="padding:30px 36px;">
        <span class="hero-tag">ENCRYPTED • LIVE • QUERYABLE</span>
        <div class="hero-title" style="font-size:32px; margin-top:8px;">📁 Dataset Vault</div>
        <div class="hero-sub" style="margin-bottom:0;">Explore the raw applicant telemetry powering Credit Spectrum AI.</div>
    </div>
    """, unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for i, (col, (num, lbl, icon)) in enumerate(zip(
        [m1, m2, m3, m4],
        [(f"{df_raw.shape[0]:,}", "Rows", "🧾"), (f"{df_raw.shape[1]}", "Columns", "🧬"),
         (f"{df_raw.isna().sum().sum()}", "Missing Values", "🕳️"),
         (f"{df_raw.duplicated().sum()}", "Duplicate Records", "📎")],
    )):
        with col:
            st.markdown(f"""
            <div class="vault-card fade-up d{i+1}" style="text-align:center;">
                <div style="font-size:26px; margin-bottom:6px;">{icon}</div>
                <div class="stat-number">{num}</div>
                <div class="stat-label">{lbl}</div>
            </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:22px;'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="vault-card fade-up d2">
        <h3 style="margin-top:0;">🔎 Search &amp; Explore</h3>
    """, unsafe_allow_html=True)
    search = st.text_input("Filter by purpose (leave blank for all):", "", label_visibility="visible")
    display_df = df_raw if not search else df_raw[df_raw['purpose'].str.contains(search, case=False, na=False)]
    st.caption(f"Showing **{len(display_df):,}** of **{len(df_raw):,}** records")
    st.dataframe(display_df, use_container_width=True, height=420)
    st.download_button("📥 Download Full Dataset (CSV)", data=df_raw.to_csv(index=False),
                        file_name="loan_data.csv", mime="text/csv")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    prev_cols = st.columns(2)
    with prev_cols[0]:
        st.markdown('<div class="vault-card fade-up d3">', unsafe_allow_html=True)
        st.markdown("#### 📊 Column Snapshot")
        st.dataframe(df_raw.describe().T.round(2), use_container_width=True, height=300)
        st.markdown('</div>', unsafe_allow_html=True)
    with prev_cols[1]:
        st.markdown('<div class="vault-card fade-up d4">', unsafe_allow_html=True)
        st.markdown("#### 🧬 Data Types")
        dtype_df = pd.DataFrame({"Column": df_raw.dtypes.index, "Type": df_raw.dtypes.astype(str).values})
        st.dataframe(dtype_df, use_container_width=True, height=300, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# =====================================================================================
# 10. ABOUT THE MODEL
# =====================================================================================
elif choice == "👨‍💻 About the Model":
    st.markdown('<div class="hero-title" style="font-size:32px;">🧭 About Credit Spectrum AI</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="glass-card">
    <p>Credit Spectrum AI is an intelligent credit-eligibility platform that evaluates demographic
    and financial telemetry to assess loan risk and predict credit-policy compliance,
    using a production-grade <b>XGBoost</b> classifier.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛤️ Model Development Roadmap")
    steps = ["Dataset Collection", "Data Cleaning", "Categorical Encoding", "Feature Scaling",
             "SMOTE Class Balancing", "Train / Test Split", "Hyperparameter Tuning (RandomizedSearchCV)",
             "XGBoost Training", "Evaluation (ROC AUC / Accuracy)", "Production Deployment"]
    for s in steps:
        st.markdown(f'<div class="timeline-item">{s}</div>', unsafe_allow_html=True)

    st.markdown("### 📖 Feature Glossary")
    glossary = {
        "credit.policy": "Target — meets AstraBank credit policy (1) or not (0)",
        "purpose": "Stated purpose of the loan",
        "int.rate": "Interest rate assigned to the loan",
        "installment": "Monthly installment payment",
        "log.annual.inc": "Natural log of borrower's annual income",
        "dti": "Debt-to-income ratio",
        "fico": "FICO credit score",
        "days.with.cr.line": "Days the borrower has held a credit line",
        "revol.bal": "Revolving balance",
        "revol.util": "Revolving line utilization rate",
        "inq.last.6mths": "Credit inquiries in the last 6 months",
        "delinq.2yrs": "Delinquencies in the last 2 years",
        "pub.rec": "Derogatory public records",
    }
    st.dataframe(pd.DataFrame(glossary.items(), columns=["Feature", "Description"]),
                 use_container_width=True, hide_index=True)

    st.markdown("### 🏆 Final Model Comparison")
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost ⭐ (Deployed)"],
        "Test Accuracy": [0.8569, 0.9881, 0.9840, 0.9892],
        "Test ROC AUC": [0.9341, 0.9829, 0.9969, 0.9978],
    }), use_container_width=True, hide_index=True)
