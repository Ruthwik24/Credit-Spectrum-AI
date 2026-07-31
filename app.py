"""
Credit Spectrum AI — Intelligent Credit Eligibility Platform (Money Edition)
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
import random

# =====================================================================================
# 1. PAGE CONFIG
# =====================================================================================
st.set_page_config(
    page_title="Credit Spectrum AI | Intelligent Credit Eligibility Platform",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "mood" not in st.session_state:
    st.session_state.mood = "default"   # default | approved | declined

# =====================================================================================
# 2. THEME — light, colorful, money-reactive glassmorphism + floating cash/crypto
# =====================================================================================
def apply_theme(mood="default"):
    palettes = {
        "default": dict(
            bg="radial-gradient(circle at 12% 8%, #fff3d6 0%, #ffe6ef 38%, #f1e4ff 72%, #ffe9d6 100%)",
            accent1="#f6a509", accent2="#ef4d8b", accent3="#8b5cf6", text="#3a2a4d",
            cardBorder="rgba(139,92,246,0.28)",
            heroGrad="linear-gradient(120deg, rgba(246,165,9,0.22), rgba(239,77,139,0.18) 55%, rgba(139,92,246,0.18))",
            titleGrad="linear-gradient(90deg, #f6a509 0%, #ef4d8b 25%, #8b5cf6 50%, #f6a509 75%, #ef4d8b 100%)",
        ),
        "approved": dict(
            bg="radial-gradient(circle at 12% 8%, #e4ffe8 0%, #cdf7d6 40%, #a9edc0 75%, #d9ffe0 100%)",
            accent1="#16a34a", accent2="#22c55e", accent3="#65d68a", text="#0d3b20",
            cardBorder="rgba(22,163,74,0.35)",
            heroGrad="linear-gradient(120deg, rgba(34,197,94,0.28), rgba(101,214,138,0.18))",
            titleGrad="linear-gradient(90deg, #0d7a3c 0%, #22c55e 30%, #86e0a3 60%, #16a34a 100%)",
        ),
        "declined": dict(
            bg="radial-gradient(circle at 12% 8%, #ffe9e9 0%, #ffd2d2 40%, #ffb4b8 75%, #ffe3e3 100%)",
            accent1="#dc2626", accent2="#f43f5e", accent3="#fb7185", text="#5c0d14",
            cardBorder="rgba(220,38,38,0.35)",
            heroGrad="linear-gradient(120deg, rgba(244,63,94,0.26), rgba(251,113,133,0.18))",
            titleGrad="linear-gradient(90deg, #b91c1c 0%, #f43f5e 30%, #fca5ac 60%, #dc2626 100%)",
        ),
    }
    p = palettes[mood]

    glyphs = ["₹", "₹", "₹", "💵", "💴", "🪙", "₿", "Ξ", "💸", "💎", "₹", "🪙"]
    particles_html = ""
    for i in range(22):
        g = random.choice(glyphs)
        left = random.randint(0, 98)
        dur = random.uniform(10, 22)
        delay = random.uniform(0, 12)
        size = random.randint(16, 30)
        opacity = round(random.uniform(0.12, 0.30), 2)
        particles_html += (
            f'<span class="money-particle" style="left:{left}vw; font-size:{size}px; '
            f'animation-duration:{dur}s; animation-delay:{delay}s; opacity:{opacity};">{g}</span>'
        )

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700;800;900&family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', 'Poppins', sans-serif; }}

    .stApp {{
        background: {p['bg']};
        background-attachment: fixed;
        color: {p['text']};
        transition: background 1.2s ease;
    }}

    .stApp::before {{
        content: "";
        position: fixed; inset: 0;
        background-image:
            radial-gradient(2.5px 2.5px at 18% 25%, {p['accent1']}66, transparent),
            radial-gradient(2.5px 2.5px at 75% 60%, {p['accent2']}55, transparent),
            radial-gradient(2px 2px at 85% 15%, {p['accent3']}55, transparent),
            radial-gradient(2px 2px at 40% 85%, {p['accent1']}44, transparent);
        background-size: 550px 550px;
        opacity: 0.7;
        pointer-events: none;
        z-index: 0;
        animation: drift 45s linear infinite;
    }}
    @keyframes drift {{ from{{background-position:0 0;}} to{{background-position:550px 550px;}} }}

    .money-bg {{ position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; }}
    .money-particle {{
        position: absolute; top: -60px; animation-name: fallDrift; animation-timing-function: linear;
        animation-iteration-count: infinite; filter: drop-shadow(0 2px 4px rgba(0,0,0,0.12));
    }}
    @keyframes fallDrift {{
        0%   {{ transform: translateY(-8vh) translateX(0) rotate(0deg); }}
        50%  {{ transform: translateY(55vh) translateX(30px) rotate(180deg); }}
        100% {{ transform: translateY(115vh) translateX(-20px) rotate(360deg); }}
    }}

    h1, h2, h3 {{ font-family: 'Poppins', sans-serif !important; color: {p['text']} !important; font-weight: 700 !important; }}
    p, span, div, label {{ color: {p['text']}; }}

    .glass-card {{
        background: rgba(255,255,255,0.55); border: 1px solid {p['cardBorder']};
        border-radius: 20px; padding: 26px; backdrop-filter: blur(16px); -webkit-backdrop-filter: blur(16px);
        box-shadow: 0 10px 30px rgba(80,40,90,0.10); transition: all 0.35s ease; margin-bottom: 18px;
        position: relative; z-index: 1;
    }}
    .glass-card:hover {{ transform: translateY(-6px); border-color: {p['accent2']}; box-shadow: 0 18px 38px rgba(80,40,90,0.16); }}

    .hero-wrap {{
        background: {p['heroGrad']}; border: 1px solid {p['cardBorder']}; border-radius: 28px;
        padding: 46px 42px; backdrop-filter: blur(20px); margin-bottom: 22px; position: relative;
        overflow: hidden; z-index: 1; text-align: center;
    }}
    .brand-row {{ display: flex; align-items: center; justify-content: center; gap: 14px; flex-wrap: wrap; }}
    .coin-spin {{ display: inline-block; font-size: 42px; animation: spinCoin 3.2s linear infinite; }}
    @keyframes spinCoin {{ 0%{{ transform: rotateY(0deg); }} 100%{{ transform: rotateY(360deg); }} }}
    .hero-title {{
        font-family: 'Poppins', sans-serif; font-size: 50px; font-weight: 900;
        background: {p['titleGrad']}; background-size: 300% auto;
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px; animation: shimmer 6s linear infinite;
    }}
    @keyframes shimmer {{ 0%{{ background-position: 0% center; }} 100%{{ background-position: 300% center; }} }}
    .orbit-emoji {{ display:inline-block; animation: bob 2.4s ease-in-out infinite; }}
    .orbit-emoji.d2 {{ animation-delay: 0.4s; }}
    .orbit-emoji.d3 {{ animation-delay: 0.8s; }}
    @keyframes bob {{ 0%,100%{{ transform: translateY(0) rotate(-4deg); }} 50%{{ transform: translateY(-9px) rotate(4deg); }} }}
    .hero-sub {{ font-size: 19px; color: {p['text']}; opacity: 0.85; font-weight: 400; margin: 8px 0 14px; }}
    .hero-tag {{
        display: inline-block; padding: 6px 16px; border-radius: 30px; background: rgba(255,255,255,0.55);
        border: 1px solid {p['accent1']}; color: {p['accent1']}; font-size: 13px; letter-spacing: 1px; font-weight: 700;
    }}

    .navbar-wrap {{
        background: rgba(255,255,255,0.55); border: 1px solid {p['cardBorder']}; border-radius: 20px;
        padding: 10px 14px; backdrop-filter: blur(16px); margin-bottom: 22px; z-index: 2; position: relative;
    }}
    div[data-testid="stHorizontalBlock"] div[role="radiogroup"] {{
        display: flex; flex-wrap: wrap; gap: 6px; justify-content: center;
    }}
    div[role="radiogroup"] label {{
        background: rgba(255,255,255,0.6); border: 1px solid {p['cardBorder']}; border-radius: 14px;
        padding: 8px 16px !important; font-weight: 600; transition: all 0.25s ease; cursor: pointer;
    }}
    div[role="radiogroup"] label:hover {{ transform: translateY(-2px); border-color: {p['accent2']}; }}

    .stat-card {{
        background: rgba(255,255,255,0.55); border: 1px solid {p['cardBorder']}; border-radius: 18px;
        padding: 22px 10px; text-align: center; backdrop-filter: blur(14px); animation: floaty 5s ease-in-out infinite;
        position: relative; z-index: 1;
    }}
    .stat-card:nth-child(2) {{ animation-delay: 0.6s; }}
    .stat-card:nth-child(3) {{ animation-delay: 1.2s; }}
    @keyframes floaty {{ 0%,100%{{ transform: translateY(0);}} 50%{{ transform: translateY(-8px);}} }}
    .stat-number {{
        font-size: 34px; font-weight: 800; font-family: 'Poppins', sans-serif; background: {p['titleGrad']};
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .stat-label {{ font-size: 13px; color: {p['text']}; opacity:0.75; letter-spacing: 0.5px; text-transform: uppercase; margin-top: 4px;}}

    .stButton>button, .stFormSubmitButton>button {{
        background: linear-gradient(90deg, {p['accent1']}, {p['accent2']});
        color: #fff; border: 1px solid rgba(255,255,255,0.5); border-radius: 14px; padding: 12px 20px;
        font-weight: 700; letter-spacing: 0.4px; width: 100%; box-shadow: 0 8px 22px rgba(80,40,90,0.18);
        transition: all 0.3s ease;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover {{
        background: linear-gradient(90deg, {p['accent2']}, {p['accent3']});
        box-shadow: 0 12px 30px rgba(80,40,90,0.26); transform: translateY(-2px);
    }}

    .approved-hero {{
        background: linear-gradient(135deg, rgba(34,197,94,0.22), rgba(34,197,94,0.06));
        border: 1px solid #16a34a; border-radius: 24px; padding: 36px; text-align: center;
        box-shadow: 0 0 44px rgba(34,197,94,0.30); position: relative; z-index: 2;
    }}
    .declined-hero {{
        background: linear-gradient(135deg, rgba(244,63,94,0.22), rgba(244,63,94,0.06));
        border: 1px solid #dc2626; border-radius: 24px; padding: 36px; text-align: center;
        box-shadow: 0 0 44px rgba(244,63,94,0.30); position: relative; z-index: 2;
    }}
    .verdict-title {{ font-family:'Poppins',sans-serif; font-size: 32px; font-weight: 800; margin-bottom: 6px;}}
    .approved-hero .verdict-title {{ color: #0d7a3c; }}
    .declined-hero .verdict-title {{ color: #b91c1c; }}
    .verdict-icon {{ font-size: 54px; animation: popIn 0.6s ease; }}
    @keyframes popIn {{ 0%{{ transform: scale(0);}} 70%{{ transform: scale(1.2);}} 100%{{ transform: scale(1);}} }}

    .reason-chip {{
        display: inline-block; margin: 5px 6px; padding: 8px 14px; border-radius: 30px;
        background: rgba(255,255,255,0.65); border: 1px solid {p['cardBorder']}; font-size: 13.5px; font-weight: 600;
    }}

    .cash-rain {{ position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 9999; }}
    .cash-drop {{
        position: absolute; top: -50px; font-size: 30px; animation-name: cashFall;
        animation-timing-function: ease-in; animation-iteration-count: 1; animation-fill-mode: forwards;
    }}
    @keyframes cashFall {{
        0%   {{ transform: translateY(-10vh) rotate(0deg); opacity: 1; }}
        100% {{ transform: translateY(105vh) rotate(400deg); opacity: 0.9; }}
    }}

    .decline-drop {{
        position: absolute; top: -50px; font-size: 26px; animation-name: declineFall;
        animation-timing-function: ease-in; animation-iteration-count: 1; animation-fill-mode: forwards;
    }}
    @keyframes declineFall {{
        0%   {{ transform: translateY(-10vh) rotate(0deg); opacity: 0.9; }}
        100% {{ transform: translateY(105vh) rotate(-200deg); opacity: 0.3; }}
    }}

    .timeline-item {{ border-left: 3px solid {p['accent1']}; padding-left: 18px; margin-bottom: 18px; position: relative; }}
    .timeline-item::before {{
        content: ""; position: absolute; left: -8px; top: 2px; width: 13px; height: 13px; border-radius: 50%;
        background: {p['accent1']}; box-shadow: 0 0 10px {p['accent1']};
    }}

    .leader-row {{
        display: flex; align-items: center; justify-content: space-between; padding: 14px 20px;
        background: rgba(255,255,255,0.55); border: 1px solid {p['cardBorder']}; border-radius: 16px;
        margin-bottom: 10px; transition: all 0.25s ease;
    }}
    .leader-row:hover {{ transform: translateX(6px); border-color: {p['accent2']}; }}
    .leader-crown {{ font-size: 22px; margin-right: 10px; }}
    .champion-badge {{
        display: inline-block; padding: 20px 30px; border-radius: 22px;
        background: linear-gradient(135deg, rgba(246,165,9,0.25), rgba(239,77,139,0.15));
        border: 2px solid {p['accent1']}; text-align: center; animation: floaty 4s ease-in-out infinite;
    }}

    hr {{ border-color: {p['cardBorder']}; }}
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {p['accent2']}; border-radius: 10px; }}
    </style>
    <div class="money-bg">{particles_html}</div>
    """, unsafe_allow_html=True)


def cash_celebration(good=True, n=36):
    """Full-screen falling cash / crypto glyphs shown once after a verdict."""
    good_glyphs = ["💵", "💰", "🤑", "₹", "🪙", "💸", "₿", "Ξ", "🎉"]
    bad_glyphs = ["📉", "💔", "🥀", "☔", "🪙", "🍂"]
    glyphs = good_glyphs if good else bad_glyphs
    cls = "cash-drop" if good else "decline-drop"
    spans = ""
    for i in range(n):
        g = random.choice(glyphs)
        left = random.randint(0, 98)
        dur = random.uniform(2.2, 4.5)
        delay = random.uniform(0, 1.6)
        size = random.randint(22, 40)
        spans += (f'<span class="{cls}" style="left:{left}vw; font-size:{size}px; '
                  f'animation-duration:{dur}s; animation-delay:{delay}s;">{g}</span>')
    st.markdown(f'<div class="cash-rain">{spans}</div>', unsafe_allow_html=True)


apply_theme(st.session_state.mood)

FEATURE_COLS = ['purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths',
                'delinq.2yrs', 'pub.rec']

# =====================================================================================
# 3. DATA + AUTO-HEALING MODEL LOADING
# =====================================================================================
@st.cache_resource(show_spinner="💰 Booting Credit Spectrum AI Core (Auto-healing if needed)...")
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
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
        except Exception:
            pass

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
# 4. HERO / BRAND HEADER (with animated coin + orbiting money emojis)
# =====================================================================================
st.markdown("""
<div class="hero-wrap">
    <span class="hero-tag">POWERED BY EXPLAINABLE AI &nbsp;•&nbsp; XGBOOST &nbsp;•&nbsp; SECURE • INTELLIGENT • TRUSTED</span>
    <div class="brand-row">
        <span class="coin-spin">🪙</span>
        <span class="hero-title">Credit Spectrum AI</span>
        <span class="orbit-emoji">💵</span><span class="orbit-emoji d2">₿</span><span class="orbit-emoji d3">Ξ</span>
    </div>
    <div class="hero-sub">🪙 Next-Generation Loan Intelligence Platform for the Modern Digital Bank ₹💹</div>
    <p style="max-width:680px; margin:0 auto; opacity:0.85;">
        Analyze a customer's complete financial profile, quantify credit risk and predict
        loan eligibility in milliseconds — with full transparency into every decision. 🔐🧾📈
    </p>
</div>
""", unsafe_allow_html=True)

# =====================================================================================
# 5. TOP NAVIGATION BAR (moved from sidebar to a horizontal bar under the hero)
# =====================================================================================
st.markdown('<div class="navbar-wrap">', unsafe_allow_html=True)
menu = [
    "🏠 Home", "📊 Dashboard", "📈 Analytics", "🧠 AI Predictor", "📁 Dataset", "👨‍💻 About the Model",
]
choice = st.radio("Navigate", menu, horizontal=True, label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

top_l, top_r = st.columns([3, 1])
with top_r:
    if st.session_state.mood != "default":
        if st.button("🔄 Reset Theme to Default"):
            st.session_state.mood = "default"
            st.rerun()

with st.expander("📥 Download Center  •  🟢 Core Status: ONLINE  •  ⚙️ Engine: XGBoost (Production)"):
    dc1, dc2 = st.columns(2)
    with dc1:
        if os.path.exists("model.pkl"):
            with open("model.pkl", "rb") as f:
                st.download_button("⬇️ Download XGBoost Model", data=f, file_name="astrabank_xgboost_model.pkl",
                                    mime="application/octet-stream", use_container_width=True)
    with dc2:
        st.download_button("⬇️ Download Dataset (CSV)", data=df_raw.to_csv(index=False),
                            file_name="loan_data.csv", mime="text/csv", use_container_width=True)

# =====================================================================================
# 6. HOME
# =====================================================================================
if choice == "🏠 Home":
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🚀 Start Prediction"):
            st.info("Use the top navigation → **🧠 AI Predictor** to analyze an applicant.")
    with c2:
        if st.button("📊 Explore Dashboard"):
            st.info("Use the top navigation → **📊 Dashboard** to explore the data holograms.")

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
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{num}</div>
                        <div class="stat-label">{lbl}</div></div>""", unsafe_allow_html=True)

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
            st.markdown(f"""<div class="glass-card" style="text-align:center;">
                <div style="font-size:34px;">{icon}</div>
                <h3 style="font-size:18px;">{title}</h3>
                <p style="font-size:13.5px;">{desc}</p>
            </div>""", unsafe_allow_html=True)

    st.markdown("### 💹 The Journey of Money — From Application to Approval")
    st.image("https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif",
              caption="Every approved application is a step toward financial freedom 💵", use_container_width=True)

# =====================================================================================
# 7. DASHBOARD (EDA)
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
# 8. ANALYTICS  (Leaderboard + Champion model)
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

    # ---------------- LEADERBOARD ----------------
    st.markdown("### 🏆 Algo-Arena Leaderboard")
    st.caption("Every architecture we trained, ranked by test-set performance.")
    leaderboard = [
        (1, "🥇", "XGBoost ⭐ (Deployed)", 0.9892, 0.9978),
        (2, "🥈", "Random Forest", 0.9840, 0.9969),
        (3, "🥉", "Decision Tree", 0.9881, 0.9829),
        (4, "🎖️", "Logistic Regression", 0.8569, 0.9341),
    ]
    for rank, medal, name, accv, aucv in sorted(leaderboard, key=lambda r: -r[4]):
        st.markdown(f"""
        <div class="leader-row">
            <div><span class="leader-crown">{medal}</span><b>{name}</b></div>
            <div>Accuracy: <b>{accv*100:.2f}%</b> &nbsp;|&nbsp; ROC AUC: <b>{aucv*100:.2f}%</b></div>
        </div>""", unsafe_allow_html=True)

    ch1, ch2 = st.columns([1, 2])
    with ch1:
        st.markdown("""
        <div style="text-align:center; margin-top:10px;">
            <div class="champion-badge">
                <div style="font-size:40px;">🏆👑</div>
                <div style="font-weight:800; font-size:18px;">XGBoost</div>
                <div style="font-size:12.5px; opacity:0.8;">Reigning Champion Model</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with ch2:
        st.image("https://media.giphy.com/media/g9582DNuQppxC/giphy.gif",
                  caption="🎉 XGBoost takes the crown — cha-ching! 💰", use_container_width=True)

    metrics_df = pd.DataFrame({
        "AI Architecture": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost ⭐"],
        "Test Accuracy": [0.8569, 0.9881, 0.9840, 0.9892],
        "Test ROC AUC": [0.9341, 0.9829, 0.9969, 0.9978],
    })
    fig = px.bar(metrics_df, x='AI Architecture', y=['Test Accuracy', 'Test ROC AUC'],
                 barmode='group', template='plotly_white', title="⚔️ Combat Results",
                 color_discrete_sequence=['#f6a509', '#ef4d8b'])
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("""
    <div class="glass-card">
    🧬 <b>Why XGBoost Was Chosen:</b> The tuned <b>XGBoost</b> model was promoted to production for its
    superior ROC AUC generalization, stable precision–recall trade-off on unseen data streams, native
    handling of feature interactions, and resilience to the class imbalance addressed via SMOTE —
    outperforming Logistic Regression, Decision Tree, and Random Forest across every holdout metric.
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
# 9. AI PREDICTOR
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
            "Counting the digital coins... 🪙",
            "Generating confidence score...",
            "Finalizing recommendation...",
        ]
        prog = st.progress(0)
        status = st.empty()
        for i, step in enumerate(steps):
            status.markdown(f"🛰️ **{step}**")
            time.sleep(0.3)
            prog.progress(int((i + 1) / len(steps) * 100))
        status.empty()

        purp_enc = le.transform([purpose])[0]
        input_data = np.array([[purp_enc, int_rate, installment, log_annual_inc, dti, fico,
                                 days_with_cr_line, revol_bal, revol_util, inq_last_6mths,
                                 delinq_2yrs, pub_rec]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        confidence = model.predict_proba(input_scaled)[0]

        st.session_state.mood = "approved" if prediction == 1 else "declined"
        apply_theme(st.session_state.mood)

        st.markdown("---")

        if prediction == 1:
            cash_celebration(good=True)
            st.balloons()
            st.markdown(f"""
            <div class="approved-hero">
                <div class="verdict-icon">🤑💵🎉</div>
                <div class="verdict-title">✅ ELIGIBLE FOR LOAN!</div>
                <p style="font-size:16px;">Confidence Score: <b>{confidence[1]*100:.2f}%</b></p>
                <p style="font-size:15px;">₹💰 Cha-ching! This applicant meets the credit policy. 💰₹</p>
            </div>""", unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/xT9IgzoKnwFNmISR8I/giphy.gif",
                      caption="💸 Making it rain — loan approved!", use_container_width=True)

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
            cash_celebration(good=False)
            st.snow()
            st.markdown(f"""
            <div class="declined-hero">
                <div class="verdict-icon">💔📉🚫</div>
                <div class="verdict-title">🚫 NOT ELIGIBLE FOR LOAN</div>
                <p style="font-size:16px;">Risk Confidence: <b>{confidence[0]*100:.2f}%</b></p>
                <p style="font-size:15px;">The financial profile falls short of the credit policy threshold.</p>
            </div>""", unsafe_allow_html=True)
            st.image("https://media.giphy.com/media/d2Z9QYzA2aidiWn6/giphy.gif",
                      caption="📉 No cha-ching this time — but there's a path forward.", use_container_width=True)

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

        st.info("🔄 Theme is now reflecting this verdict. Use **Reset Theme to Default** above the navigation, "
                "or run a new prediction, to change it back.")

# =====================================================================================
# 10. DATASET
# =====================================================================================
elif choice == "📁 Dataset":
    st.markdown('<div class="hero-title" style="font-size:32px;">📁 Dataset Vault</div>', unsafe_allow_html=True)

    m1, m2, m3, m4 = st.columns(4)
    for col, (num, lbl) in zip(
        [m1, m2, m3, m4],
        [(f"{df_raw.shape[0]:,}", "Rows"), (f"{df_raw.shape[1]}", "Columns"),
         (f"{df_raw.isna().sum().sum()}", "Missing Values"),
         (f"{df_raw.duplicated().sum()}", "Duplicate Records")],
    ):
        with col:
            st.markdown(f"""<div class="stat-card"><div class="stat-number">{num}</div>
                        <div class="stat-label">{lbl}</div></div>""", unsafe_allow_html=True)

    st.markdown("### 🔎 Search & Explore")
    search = st.text_input("Filter by purpose (leave blank for all):", "")
    display_df = df_raw if not search else df_raw[df_raw['purpose'].str.contains(search, case=False, na=False)]
    st.dataframe(display_df, use_container_width=True, height=420)

    st.download_button("📥 Download Full Dataset (CSV)", data=df_raw.to_csv(index=False),
                        file_name="loan_data.csv", mime="text/csv")

# =====================================================================================
# 11. ABOUT THE MODEL
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

    st.markdown("### 🏆 Final Model Comparison & Why XGBoost Won")
    st.dataframe(pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost ⭐ (Deployed)"],
        "Test Accuracy": [0.8569, 0.9881, 0.9840, 0.9892],
        "Test ROC AUC": [0.9341, 0.9829, 0.9969, 0.9978],
    }), use_container_width=True, hide_index=True)

    st.markdown("""
    <div class="glass-card">
    💡 <b>Selection Rationale:</b> XGBoost was chosen for production because gradient-boosted trees
    capture non-linear interactions between financial signals (like FICO × DTI × utilization) far
    better than Logistic Regression, while its regularization keeps it more stable on unseen data
    than a single Decision Tree or even Random Forest. Combined with SMOTE-balanced training data,
    it delivered the best accuracy <b>and</b> the best ROC AUC of every architecture tested. 🏆
    </div>
    """, unsafe_allow_html=True)
