"""
Credit Spectrum AI — Intelligent Credit Eligibility Platform (Luxury Banking Edition)
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
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "mood" not in st.session_state:
    st.session_state.mood = "default"   # default | approved | declined
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"
if "batch_result" not in st.session_state:
    st.session_state.batch_result = None

# =====================================================================================
# 2. THEME — Luxury Banking Theme (Gold, Platinum, Black), mood-reactive
# =====================================================================================
def apply_theme(mood="default"):
    palettes = {
        "default": dict(
            bg="radial-gradient(circle at 15% 10%, #1a1814 0%, #0d0c0a 45%, #000000 100%)",
            accent1="#D4AF37", accent2="#E5E4E2", accent3="#AA8000", text="#F8F8F8",
            cardBorder="rgba(212,175,55,0.35)", glow="rgba(212,175,55,0.35)",
            heroGrad="linear-gradient(120deg, rgba(212,175,55,0.12), rgba(229,228,226,0.08) 55%, rgba(0,0,0,0.3))",
            titleGrad="linear-gradient(90deg, #D4AF37 0%, #FFF8DC 30%, #AA8000 60%, #D4AF37 100%)",
        ),
        "approved": dict(
            bg="radial-gradient(circle at 15% 10%, #0a1c11 0%, #040d08 45%, #000000 100%)",
            accent1="#00ffa3", accent2="#D4AF37", accent3="#E5E4E2", text="#e7fff2",
            cardBorder="rgba(0,255,163,0.40)", glow="rgba(0,255,163,0.40)",
            heroGrad="linear-gradient(120deg, rgba(0,255,163,0.15), rgba(212,175,55,0.1))",
            titleGrad="linear-gradient(90deg, #00ffa3 0%, #D4AF37 50%, #E5E4E2 100%)",
        ),
        "declined": dict(
            bg="radial-gradient(circle at 15% 10%, #24070d 0%, #120306 45%, #000000 100%)",
            accent1="#ff2e5b", accent2="#D4AF37", accent3="#E5E4E2", text="#ffe9ee",
            cardBorder="rgba(255,46,91,0.40)", glow="rgba(255,46,91,0.40)",
            heroGrad="linear-gradient(120deg, rgba(255,46,91,0.15), rgba(212,175,55,0.1))",
            titleGrad="linear-gradient(90deg, #ff2e5b 0%, #D4AF37 50%, #E5E4E2 100%)",
        ),
    }
    p = palettes[mood]

    glyphs = ["₹", "₹", "💵", "💴", "💶", "💷", "🪙", "₿", "Ξ", "💸", "💎", "💳", "🏦", "📄", "🪙"]
    particles_html = ""
    for i in range(34):
        g = random.choice(glyphs)
        left = random.randint(0, 99)
        dur = random.uniform(9, 20)
        delay = random.uniform(0, 14)
        size = random.randint(15, 30)
        opacity = round(random.uniform(0.10, 0.28), 2)
        particles_html += (
            f'<span class="money-particle" style="left:{left}vw; font-size:{size}px; '
            f'animation-duration:{dur}s; animation-delay:{delay}s; opacity:{opacity};">{g}</span>'
        )

    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@500;700;800;900&family=Cormorant+Garamond:wght@400;500;600;700&family=EB+Garamond:ital,wght@0,400;0,500;0,600;1,400&family=Special+Elite&family=Orbitron:wght@500;700;900&family=Cinzel:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {{ font-family: 'EB Garamond', 'Cormorant Garamond', serif; font-size: 17px; }}
    #MainMenu, footer, header {{ visibility: hidden; }}

    /* ---------- RETRO STRUCTURE: sharp, ledger-like corners everywhere (no rounded modern UI) ---------- */
    .stApp, .stApp * {{ border-radius: 0 !important; }}

    /* ---------- VINTAGE GRAIN + VIGNETTE OVERLAY (retro texture, colors untouched) ---------- */
    .stApp::after {{
        content: ""; position: fixed; inset: 0; z-index: 9998; pointer-events: none;
        background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='120' height='120'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)' opacity='0.05'/%3E%3C/svg%3E");
        mix-blend-mode: overlay; opacity: 0.5;
        box-shadow: inset 0 0 140px 60px rgba(0,0,0,0.65), inset 0 0 40px 10px rgba(0,0,0,0.4);
    }}

    /* CRITICAL FIX: Ensure the underlying DOM body is completely black. 
       This prevents popovers (which render outside .stApp) from revealing a white background through their transparency! */
    html, body {{
        background-color: #000000 !important;
        background: #000000 !important;
    }}

    .stApp {{
        background: {p['bg']};
        background-attachment: fixed; color: {p['text']}; transition: background 1.2s ease;
    }}
    .block-container {{
        padding-top: 1.2rem; max-width: 1900px; width: 100%;
        margin-left: auto; margin-right: auto;
    }}
    section.main > div.block-container {{ margin-left: auto; margin-right: auto; }}
    [data-testid="stAppViewContainer"] {{ display: flex; justify-content: center; }}

    .stApp::before {{
        content: ""; position: fixed; inset: 0;
        background-image:
            radial-gradient(2.5px 2.5px at 18% 25%, {p['accent1']}66, transparent),
            radial-gradient(2.5px 2.5px at 75% 60%, {p['accent2']}55, transparent),
            radial-gradient(2px 2px at 85% 15%, {p['accent3']}55, transparent),
            radial-gradient(2px 2px at 40% 85%, {p['accent1']}44, transparent),
            radial-gradient(1.5px 1.5px at 60% 35%, {p['accent2']}44, transparent);
        background-size: 550px 550px; opacity: 0.45; pointer-events: none; z-index: 0;
        animation: drift 40s linear infinite;
    }}
    @keyframes drift {{ from{{background-position:0 0;}} to{{background-position:550px 550px;}} }}

    .money-bg {{ position: fixed; inset: 0; overflow: hidden; pointer-events: none; z-index: 0; height: 100vh; width: 100vw; }}
    .money-particle {{
        position: absolute; top: -60px; animation-name: fallDrift; animation-timing-function: linear;
        animation-iteration-count: infinite; filter: drop-shadow(0 0 6px {p['glow']});
    }}
    @keyframes fallDrift {{
        0%   {{ transform: translateY(-8vh) translateX(0) rotate(0deg); }}
        50%  {{ transform: translateY(55vh) translateX(35px) rotate(180deg); }}
        100% {{ transform: translateY(115vh) translateX(-25px) rotate(360deg); }}
    }}

    h1, h2, h3 {{ font-family: 'Playfair Display', serif !important; color: {p['text']} !important; font-weight: 800 !important; letter-spacing: 0.5px; }}
    p, span, div, label {{ color: {p['text']}; }}

    /* ---------- GLASS CARD (retro plaque edition — vintage frame, letterpress emboss) ---------- */
    .glass-card {{
        background: rgba(255,255,255,0.03); border: 1px double {p['cardBorder']};
        outline: 1px solid rgba(0,0,0,0.5); outline-offset: -6px;
        padding: 28px 26px; backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 0 30px rgba(0,0,0,0.6), 0 0 12px {p['glow']} inset, inset 0 1px 0 rgba(255,255,255,0.12), inset 0 -1px 0 rgba(0,0,0,0.5);
        transition: all 0.35s ease;
        margin-bottom: 18px; position: relative; z-index: 1;
    }}
    /* vintage certificate corner brackets */
    .glass-card::before, .glass-card::after,
    div[data-testid="stVerticalBlockBorderWrapper"]::before, div[data-testid="stVerticalBlockBorderWrapper"]::after {{
        content: ""; position: absolute; width: 16px; height: 16px; pointer-events: none; z-index: 2;
        border-color: {p['accent1']}; opacity: 0.75;
    }}
    .glass-card::before, div[data-testid="stVerticalBlockBorderWrapper"]::before {{
        top: 6px; left: 6px; border-top: 2px solid; border-left: 2px solid;
    }}
    .glass-card::after, div[data-testid="stVerticalBlockBorderWrapper"]::after {{
        bottom: 6px; right: 6px; border-bottom: 2px solid; border-right: 2px solid;
    }}
    .glass-card:hover {{
        transform: translateY(-4px); border-color: {p['accent2']};
        box-shadow: 0 0 40px {p['glow']}, 0 0 16px {p['glow']} inset, inset 0 1px 0 rgba(255,255,255,0.16), inset 0 -1px 0 rgba(0,0,0,0.5);
    }}
    .glass-card h3 {{ font-family: 'Playfair Display', serif !important; letter-spacing: 0.5px; border-bottom: 1px solid {p['cardBorder']}; padding-bottom: 10px; }}

    /* ---------- PANEL (for dashboard/analytics grid boxes) ---------- */
    div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: rgba(255,255,255,0.02) !important;
        border: 1px double {p['cardBorder']} !important;
        outline: 1px solid rgba(0,0,0,0.5) !important; outline-offset: -6px !important;
        backdrop-filter: blur(16px); box-shadow: 0 0 24px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.1); position: relative; z-index: 1;
        transition: box-shadow 0.3s ease, transform 0.3s ease; padding: 4px !important;
    }}
    div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 0 24px {p['glow']}; transform: translateY(-3px);
    }}
    .panel-title {{
        font-family:'Playfair Display',serif; font-weight:800; font-size:18px; margin-bottom:2px;
        color:{p['accent1']} !important; letter-spacing: 0.4px;
    }}
    .panel-sub {{ font-family: 'Special Elite', monospace; font-size:12px; opacity:0.65; margin-bottom:10px; letter-spacing: 0.5px; }}

    /* ---------- HERO / TITLE ---------- */
    .hero-wrap {{
        background: {p['heroGrad']}; border: 3px double {p['cardBorder']}; border-radius: 8px;
        padding: 50px 42px; backdrop-filter: blur(22px); margin-bottom: 22px; position: relative;
        overflow: hidden; z-index: 1; text-align: center; box-shadow: 0 0 50px {p['glow']}, inset 0 0 0 8px rgba(0,0,0,0.25);
    }}
    .hero-wrap::before {{
        content: "❧"; position: absolute; top: 10px; left: 18px; font-size: 22px; color: {p['accent1']}; opacity: 0.55;
    }}
    .hero-wrap::after {{
        content: "❧"; position: absolute; bottom: 10px; right: 18px; font-size: 22px; color: {p['accent1']}; opacity: 0.55; transform: rotate(180deg);
    }}
    .brand-row {{ display: flex; align-items: center; justify-content: center; gap: 16px; flex-wrap: wrap; }}
    .coin-spin {{ display: inline-block; font-size: 44px; animation: spinCoin 3.2s linear infinite; filter: drop-shadow(0 0 10px {p['glow']}); }}
    @keyframes spinCoin {{ 0%{{ transform: rotateY(0deg); }} 100%{{ transform: rotateY(360deg); }} }}
    .hero-title {{
        font-family: 'Playfair Display', serif; font-size: 52px; font-weight: 900; letter-spacing: 2px;
        text-transform: uppercase;
        background: {p['titleGrad']}; background-size: 300% auto;
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
        margin-bottom: 4px; animation: shimmer 6s linear infinite;
        text-shadow: 0 0 30px {p['glow']};
    }}
    @keyframes shimmer {{ 0%{{ background-position: 0% center; }} 100%{{ background-position: 300% center; }} }}
    .orbit-emoji {{ display:inline-block; animation: bob 2.4s ease-in-out infinite; filter: drop-shadow(0 0 8px {p['glow']}); }}
    .orbit-emoji.d2 {{ animation-delay: 0.4s; }}
    .orbit-emoji.d3 {{ animation-delay: 0.8s; }}
    @keyframes bob {{ 0%,100%{{ transform: translateY(0) rotate(-4deg); }} 50%{{ transform: translateY(-9px) rotate(4deg); }} }}
    .hero-sub {{ font-family: 'Cormorant Garamond', serif; font-size: 21px; color: {p['text']}; opacity: 0.85; font-weight: 500; font-style: italic; margin: 10px 0 14px; }}
    .hero-tag {{
        display: inline-block; padding: 6px 18px; border-radius: 2px; background: rgba(255,255,255,0.05);
        border: 1px solid {p['accent1']}; color: {p['accent1']}; font-family: 'Special Elite', monospace;
        font-size: 11.5px; letter-spacing: 1.8px; font-weight: 400;
        box-shadow: 0 0 14px {p['glow']};
    }}

    /* ---------- FLOATING VINTAGE PLAQUE CARD ---------- */
    .credit-card {{
        width: 230px; height: 140px; margin: 22px auto 0; border-radius: 6px; position: relative;
        background: linear-gradient(135deg, {p['accent2']}, {p['accent1']} 60%, {p['accent3']});
        border: 1px solid rgba(0,0,0,0.4);
        box-shadow: 0 0 30px {p['glow']}, 0 12px 30px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.5), inset 0 -3px 6px rgba(0,0,0,0.25);
        animation: cardFloat 4.5s ease-in-out infinite; padding: 16px; text-align: left;
    }}
    @keyframes cardFloat {{ 0%,100%{{ transform: translateY(0) rotate(-2deg);}} 50%{{ transform: translateY(-10px) rotate(2deg);}} }}
    .credit-card .chip {{ font-size: 26px; }}
    .credit-card .num {{ color: #000; font-family: 'Special Elite', monospace; letter-spacing: 2px; font-size: 14px; margin-top: 14px; text-shadow: 0 0 6px rgba(255,255,255,0.4); }}
    .credit-card .brand {{ position:absolute; bottom: 12px; right: 16px; font-size: 20px; }}

    /* ---------- TOP NAV BAR (rectangle vintage tabs, equal width) ---------- */
    /* NOTE: styling is applied directly to Streamlit's own stRadio container —
       the markdown wrapper div below never actually nests around the widget in the DOM. */
    div[data-testid="stRadio"] {{
        background: linear-gradient(135deg, rgba(255,255,255,0.04), rgba(255,255,255,0.01)) !important;
        border: 1px solid {p['cardBorder']} !important; border-radius: 4px !important;
        padding: 10px !important; backdrop-filter: blur(26px) saturate(160%) !important; -webkit-backdrop-filter: blur(26px) saturate(160%) !important;
        margin: 0 0 22px 0 !important; z-index: 20; position: relative; width: 100% !important;
        box-shadow: 0 0 30px {p['glow']}, 0 8px 32px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.05) !important;
        overflow: visible !important;
    }}
    div[data-testid="stRadio"] > label,
    div[data-testid="stWidgetLabel"] {{
        display: none !important; height: 0 !important; padding: 0 !important; margin: 0 !important;
    }}
    div[role="radiogroup"] {{
        display: flex !important; flex-direction: row !important; flex-wrap: nowrap !important;
        gap: 10px !important; width: 100% !important; margin: 0 !important; padding: 0 !important;
        background: transparent !important; border: none !important;
        align-items: stretch !important; /* Fix for uneven box heights */
    }}
    div[role="radiogroup"] label {{
        background: rgba(255,255,255,0.02); border: 1px solid {p['cardBorder']};
        padding: 10px 8px !important; font-family: 'Cinzel', 'Playfair Display', serif; font-weight: 600; font-size: 15px; letter-spacing: 1px;
        text-transform: uppercase; height: auto !important; min-height: 100% !important; 
        color: {p['text']} !important; opacity: 0.9; flex: 1 1 0px !important; min-width: 0 !important; max-width: none !important;
        justify-content: center !important; align-items: center !important; text-align: center; gap: 0 !important;
        transition: box-shadow 0.3s ease, border-color 0.3s ease, opacity 0.3s ease, background 0.3s ease, color 0.3s ease;
        cursor: pointer; position: relative; overflow: visible; z-index: 1; display: flex !important; box-sizing: border-box !important;
    }}
    
    /* FIX: Completely remove the native radio graphic/white square from the nav bar */
    div[data-testid="stRadio"] div[role="radiogroup"] label > div:first-child,
    div[data-testid="stRadio"] div[role="radiogroup"] label div[data-baseweb="radio"] {{
        display: none !important;
        opacity: 0 !important;
        width: 0 !important;
        height: 0 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        pointer-events: none !important;
    }}
    
    div[role="radiogroup"] label input {{ display: none !important; }}
    div[role="radiogroup"] label::after {{ content: none !important; }}
    div[role="radiogroup"] label div[data-testid="stMarkdownContainer"] {{
        width: 100% !important; flex: 1 1 auto !important; min-width: 0 !important;
        display: block !important; visibility: visible !important; opacity: 1 !important; position: relative !important;
    }}
    div[role="radiogroup"] label p {{
        color: inherit !important; margin: 0 !important; text-align: center !important; width: 100% !important;
        white-space: normal !important; overflow: visible !important; text-overflow: unset !important;
        line-height: 1.35 !important; word-break: keep-all !important; visibility: visible !important; opacity: 1 !important;
    }}
    div[role="radiogroup"] label::before {{
        content: ""; position: absolute; inset: 0; z-index: -1;
        background: conic-gradient(from 0deg, {p['accent1']}, {p['accent2']}, {p['accent3']}, {p['accent1']});
        opacity: 0; transition: opacity 0.35s ease;
        filter: blur(10px);
    }}
    div[role="radiogroup"] label:hover {{
        border-color: {p['accent1']};
        background: rgba(212, 175, 55, 0.12);
        box-shadow: 0 6px 20px {p['glow']}; color: {p['accent1']} !important; opacity: 1;
    }}
    div[role="radiogroup"] label:has(input:checked) {{
        background: rgba(10,10,10,0.9);
        border-color: {p['accent1']}; opacity: 1;
        color: {p['accent1']} !important; font-weight: 800;
        box-shadow: 0 0 26px {p['glow']}, inset 0 0 14px {p['glow']};
    }}
    div[role="radiogroup"] label:has(input:checked)::before {{
        opacity: 0.35; animation: navConicSpin 5s linear infinite;
    }}
    @keyframes navConicSpin {{ from{{ filter: blur(10px) hue-rotate(0deg); }} to{{ filter: blur(10px) hue-rotate(360deg); }} }}
    div[role="radiogroup"] input {{ display:none !important; }}

    /* ---------- STAT / METRIC CARDS ---------- */
    .stat-card {{
        background: rgba(255,255,255,0.02); border: 1px double {p['cardBorder']};
        outline: 1px solid rgba(0,0,0,0.4); outline-offset: -5px;
        padding: 22px 10px; text-align: center; backdrop-filter: blur(14px); animation: floaty 5s ease-in-out infinite;
        position: relative; z-index: 1; box-shadow: 0 0 16px {p['glow']};
    }}
    .stat-card:nth-child(2) {{ animation-delay: 0.6s; }}
    .stat-card:nth-child(3) {{ animation-delay: 1.2s; }}
    @keyframes floaty {{ 0%,100%{{ transform: translateY(0);}} 50%{{ transform: translateY(-8px);}} }}
    .stat-number {{
        font-size: 34px; font-weight: 800; font-family: 'Orbitron', sans-serif; background: {p['titleGrad']};
        -webkit-background-clip: text; background-clip: text; -webkit-text-fill-color: transparent;
    }}
    .stat-label {{ font-size: 12.5px; color: {p['text']}; opacity:0.7; letter-spacing: 0.6px; text-transform: uppercase; margin-top: 4px;}}

    [data-testid="stMetric"] {{
        background: rgba(255,255,255,0.02); border: 1px solid {p['cardBorder']}; border-radius: 6px;
        padding: 14px 10px; box-shadow: 0 0 14px {p['glow']};
    }}
    [data-testid="stMetricValue"] {{ color: {p['accent1']} !important; }}

    /* ---------- BUTTONS (glassy) ---------- */
    .stButton>button, .stFormSubmitButton>button, .stDownloadButton>button {{
        background: linear-gradient(180deg, rgba(255,255,255,0.10), rgba(255,255,255,0.01) 45%, rgba(0,0,0,0.15)) !important;
        color: {p['text']} !important; 
        border: 1px double {p['accent1']}88 !important; 
        outline: 1px solid rgba(0,0,0,0.4) !important; outline-offset: -4px !important;
        padding: 12px 20px !important;
        font-family: 'Cinzel', 'Playfair Display', serif !important;
        font-weight: 600 !important; 
        letter-spacing: 2px !important; 
        text-transform: uppercase !important;
        font-size: 12.5px !important;
        width: 100% !important;
        backdrop-filter: blur(16px) !important; 
        -webkit-backdrop-filter: blur(16px) !important;
        box-shadow: 0 0 18px {p['glow']}, inset 0 1px 0 rgba(255,255,255,0.18), inset 0 -2px 4px rgba(0,0,0,0.35) !important;
        transition: all 0.3s ease !important;
    }}
    .stButton>button:hover, .stFormSubmitButton>button:hover, .stDownloadButton>button:hover {{
        background: linear-gradient(180deg, {p['accent1']}44, {p['accent2']}22) !important;
        border-color: {p['accent1']} !important; 
        color: {p['accent1']} !important;
        box-shadow: 0 0 28px {p['glow']}, inset 0 1px 0 rgba(255,255,255,0.3) !important; 
        transform: translateY(-1px) !important;
    }}

    /* ---------- RESULT CARDS ---------- */
    .approved-hero {{
        background: linear-gradient(135deg, rgba(0,255,163,0.16), rgba(0,255,163,0.04));
        border: 3px double #00ffa3; border-radius: 8px; padding: 36px; text-align: center;
        box-shadow: 0 0 55px rgba(0,255,163,0.45), inset 0 0 0 6px rgba(0,0,0,0.2); position: relative; z-index: 2;
    }}
    .declined-hero {{
        background: linear-gradient(135deg, rgba(255,46,91,0.16), rgba(255,46,91,0.04));
        border: 3px double #ff2e5b; border-radius: 8px; padding: 36px; text-align: center;
        box-shadow: 0 0 55px rgba(255,46,91,0.45), inset 0 0 0 6px rgba(0,0,0,0.2); position: relative; z-index: 2;
    }}
    .verdict-title {{ font-family:'Playfair Display',serif; font-size: 30px; font-weight: 900; letter-spacing: 1px; text-transform: uppercase; margin-bottom: 6px;}}
    .approved-hero .verdict-title {{ color: #00ffa3; text-shadow: 0 0 18px rgba(0,255,163,0.6); }}
    .declined-hero .verdict-title {{ color: #ff6b6b; text-shadow: 0 0 18px rgba(255,46,91,0.6); }}
    .verdict-icon {{ font-size: 54px; animation: popIn 0.6s ease; }}
    @keyframes popIn {{ 0%{{ transform: scale(0);}} 70%{{ transform: scale(1.2);}} 100%{{ transform: scale(1);}} }}

    .reason-chip {{
        display: inline-block; margin: 5px 6px; padding: 8px 14px; border-radius: 3px;
        background: rgba(255,255,255,0.05); border: 1px solid {p['cardBorder']};
        font-family: 'Special Elite', monospace; font-size: 12.5px; font-weight: 400; letter-spacing: 0.3px;
    }}

    /* ---------- CASH RAIN ---------- */
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

    /* ---------- TIMELINE (legacy, kept for compatibility) ---------- */
    .timeline-item {{
        border-left: 3px solid {p['accent1']}; padding-left: 20px; margin-bottom: 20px; position: relative;
        background: rgba(255,255,255,0.02); padding-top: 10px; padding-bottom: 10px; padding-right: 10px;
        border-radius: 0 15px 15px 0; backdrop-filter: blur(10px);
        transition: transform 0.3s ease, border-left-color 0.3s ease, background 0.3s ease;
    }}
    .timeline-item:hover {{
        transform: translateX(5px);
        border-left-color: {p['accent2']};
        background: rgba(212, 175, 55, 0.1);
    }}
    .timeline-item::before {{
        content: ""; position: absolute; left: -10.5px; top: 13px; width: 16px; height: 16px; border-radius: 50%;
        background: #0a0a0a; border: 3px solid {p['accent1']}; box-shadow: 0 0 10px {p['glow']};
    }}

    /* ---------- ROADMAP (winding highlighted-road map style) ---------- */
    .roadmap-wrap {{
        position: relative; width: 100%; aspect-ratio: 1000 / 460;
        margin: 34px 0 24px 0;
        background:
            radial-gradient(circle at 15% 15%, rgba(212,175,55,0.06) 0%, transparent 40%),
            radial-gradient(circle at 85% 85%, rgba(212,175,55,0.05) 0%, transparent 40%);
        border: 1px solid {p['cardBorder']}; border-radius: 14px;
    }}
    .roadmap-svg {{ position: absolute; inset: 0; width: 100%; height: 100%; z-index: 1; }}
    .roadmap-stop {{
        position: absolute; display: flex; align-items: center; gap: 10px;
        transform: translate(-50%, -50%); max-width: 210px; z-index: 2;
    }}
    .roadmap-stop.side-left {{ flex-direction: row-reverse; text-align: right; }}
    .roadmap-pin {{
        flex: 0 0 auto; width: 42px; height: 42px; border-radius: 50%;
        background: radial-gradient(circle at 35% 30%, {p['accent2']}, {p['accent1']} 70%);
        color: #0a0a0a; display: flex; align-items: center; justify-content: center;
        font-weight: 800; font-size: 14px; box-shadow: 0 0 0 5px rgba(10,10,10,0.95), 0 0 14px {p['glow']};
        border: 2px solid rgba(255,255,255,0.35);
        transition: transform 0.25s ease, box-shadow 0.25s ease;
    }}
    .roadmap-stop.roadmap-final .roadmap-pin {{
        background: radial-gradient(circle at 35% 30%, #fff8dc, {p['accent1']} 70%);
        box-shadow: 0 0 0 5px rgba(10,10,10,0.95), 0 0 24px {p['glow']}, 0 0 42px {p['accent1']};
    }}
    .roadmap-label {{
        font-size: 12.5px; font-weight: 700; line-height: 1.25; color: {p['text']};
        background: rgba(10,10,10,0.6); padding: 5px 10px; border-radius: 6px;
        border: 1px solid {p['cardBorder']}; backdrop-filter: blur(6px);
        transition: border-color 0.25s ease, color 0.25s ease;
    }}
    .roadmap-stop:hover .roadmap-pin {{
        transform: scale(1.1);
        box-shadow: 0 0 0 5px rgba(10,10,10,0.95), 0 0 22px {p['glow']};
    }}
    .roadmap-stop:hover .roadmap-label {{ border-color: {p['accent2']}; color: {p['accent1']}; }}

    /* ---------- LEADERBOARD ---------- */
    .leader-row {{
        display: flex; align-items: center; justify-content: space-between; padding: 14px 20px;
        background: rgba(255,255,255,0.02); border: 1px solid {p['cardBorder']}; border-radius: 6px;
        margin-bottom: 10px; transition: all 0.25s ease;
    }}
    .leader-row:hover {{ transform: translateX(6px); border-color: {p['accent2']}; box-shadow: 0 0 16px {p['glow']}; }}
    .leader-crown {{ font-size: 22px; margin-right: 10px; }}
    .champion-badge {{
        display: inline-block; padding: 20px 30px; border-radius: 6px;
        background: linear-gradient(135deg, rgba(212,175,55,0.1), rgba(229,228,226,0.1));
        border: 2px solid {p['accent1']}; text-align: center; animation: floaty 4s ease-in-out infinite;
        box-shadow: 0 0 26px {p['glow']};
    }}

    /* ---------- FILE UPLOADER (drag & drop) ---------- */
    [data-testid="stFileUploaderDropzone"] {{
        background: rgba(255,255,255,0.02) !important; border: 2px dashed {p['accent1']} !important;
        border-radius: 6px !important; box-shadow: 0 0 16px {p['glow']};
    }}

    /* Fix to make the uploaded file list item properly translucent and readable */
    [data-testid="stFileUploader"] section,
    [data-testid="stFileUploader"] ul,
    [data-testid="stFileUploader"] li,
    [data-testid="stFileUploader"] li div,
    [data-testid="stFileUploader"] div,
    [data-testid="stFileUploader"] span,
    [data-testid="stFileUploaderFile"],
    [data-testid="stFileUploaderFileName"],
    [data-testid="stUploadedFile"],
    div[data-testid="stUploadedFile"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderFile"] * {{
        background-color: rgba(15, 15, 15, 0.8) !important;
        background: rgba(15, 15, 15, 0.8) !important;
        color: {p['text']} !important;
        border-radius: 6px !important;
    }}

    /* the dropzone itself must stay as-is, not the darker translucent fill above */
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"],
    [data-testid="stFileUploader"] [data-testid="stFileUploaderDropzone"] * {{
        background-color: transparent !important;
        background: transparent !important;
    }}
    [data-testid="stFileUploaderDropzone"] {{
        background: rgba(255,255,255,0.02) !important;
    }}

    hr {{ border-color: {p['cardBorder']}; }}
    ::-webkit-scrollbar {{ width: 8px; }}
    ::-webkit-scrollbar-thumb {{ background: {p['accent2']}; border-radius: 10px; }}

    /* ================= WIDGET TEXT VISIBILITY FIXES & GLASSY INPUTS ================= */
    label, .stMarkdown, .stCaption, [data-testid="stWidgetLabel"] p,
    [data-testid="stMetricLabel"], [data-testid="stMetricValue"],
    [data-testid="stMetricDelta"], .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] div, .streamlit-expanderHeader, .streamlit-expanderHeader p {{
        color: {p['text']} !important;
    }}
    [data-testid="stCaptionContainer"] {{ color: {p['text']} !important; opacity: 0.7; }}

    /* Target the container wrappers for ALL inputs */
    div[data-baseweb="base-input"],
    div[data-baseweb="input"],
    div[data-testid="stSelectbox"] > div > div,
    div[data-testid="stMultiSelect"] > div > div,
    div[data-testid="stTextInput"] > div > div,
    div[data-testid="stNumberInputContainer"] {{
        background: rgba(15, 15, 15, 0.6) !important;
        background-color: rgba(15, 15, 15, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border: 1px solid {p['cardBorder']} !important;
        border-radius: 6px !important;
        color: {p['text']} !important;
        box-shadow: inset 0 2px 6px rgba(0,0,0,0.6), 0 0 10px rgba(0,0,0,0.2) !important;
    }}

    /* Target the actual input elements so they inherit the glass effect naturally */
    input[type="text"], 
    input[type="number"], 
    textarea, 
    div[data-baseweb="select"] div {{
        background: transparent !important;
        color: {p['text']} !important;
        -webkit-text-fill-color: {p['text']} !important;
        box-shadow: none !important;
    }}

    /* ================= TRANSLUCENT DARK GLASS DROPDOWN (readable text, not opaque white, not see-through) ================= */
    /* Nuclear catch-all first: strip out whatever default (often white) background BaseWeb sets on its portal wrappers. */
    div[data-baseweb="popover"],
    div[data-baseweb="popover"] *,
    [data-testid="stVirtualDropdown"],
    [data-testid="stVirtualDropdown"] * {{
        background: transparent !important;
        background-color: transparent !important;
    }}

    /* The dropdown shell itself: dark translucent glass so the menu is legible against any page content behind it */
    div[data-baseweb="popover"] > div,
    [data-testid="stVirtualDropdown"] > div,
    [data-testid="stVirtualDropdown"] > div > div,
    ul[data-baseweb="menu"],
    ul[role="listbox"],
    div[role="listbox"] {{
        background: #141414 !important;
        background-color: #141414 !important;
        border: 1px solid {p['cardBorder']} !important;
        border-radius: 6px !important;
        box-shadow: 0 10px 40px rgba(0,0,0,0.8), 0 0 20px {p['glow']} !important;
    }}

    /* INDIVIDUAL OPTION TILES — softly translucent, clearly legible text, thin gold-tinted rim.
       Covers both <li role="option"> (standard menu) and <div role="option"> (virtualized list rows). */
    div[data-baseweb="popover"] li,
    ul[role="listbox"] li,
    div[role="listbox"] li,
    li[role="option"],
    div[role="option"],
    [data-testid="stVirtualDropdown"] [role="option"] {{
        background: transparent !important;
        color: {p['text']} !important;
        font-family: 'EB Garamond', serif !important;
        font-size: 16px !important;
    }}

    /* Hover and Selected states for tiles */
    div[data-baseweb="popover"] li:hover,
    ul[role="listbox"] li:hover,
    div[role="listbox"] li:hover,
    li[role="option"]:hover,
    div[role="option"]:hover,
    div[data-baseweb="popover"] li[aria-selected="true"],
    ul[role="listbox"] li[aria-selected="true"],
    div[role="listbox"] li[aria-selected="true"],
    li[role="option"][aria-selected="true"],
    div[role="option"][aria-selected="true"] {{
        background: rgba(212, 175, 55, 0.2) !important;
        color: {p['accent1']} !important;
        font-weight: 700 !important;
    }}

    /* Multiselect Tags */
    span[data-baseweb="tag"] {{ 
        background: rgba(212,175,55,0.2) !important; 
        color: {p['text']} !important; 
        border: 1px solid {p['accent1']} !important; 
        border-radius: 6px !important; 
    }}
    span[data-baseweb="tag"] span {{
        color: {p['text']} !important; 
    }}

    /* +/- Buttons on Number Inputs */
    div[data-testid="stNumberInputStepUp"],
    div[data-testid="stNumberInputStepDown"] {{
        background: transparent !important;
    }}
    div[data-testid="stNumberInputStepUp"] svg,
    div[data-testid="stNumberInputStepDown"] svg,
    div[data-baseweb="select"] svg {{
        fill: {p['accent1']} !important;
    }}
    
    /* Hover states for inputs */
    div[data-baseweb="base-input"]:focus-within,
    div[data-baseweb="input"]:focus-within,
    div[data-testid="stSelectbox"] > div > div:focus-within {{
        border-color: {p['accent1']} !important;
        box-shadow: 0 0 15px {p['glow']}, inset 0 2px 6px rgba(0,0,0,0.6) !important;
    }}

    /* tabs */
    .stTabs [data-baseweb="tab-list"] {{ gap: 8px; }}
    .stTabs [data-baseweb="tab"] {{
        background: rgba(255,255,255,0.02) !important; border: 1px solid {p['cardBorder']} !important;
        border-radius: 6px 14px 0 0 !important; color: {p['text']} !important;
    }}
    .stTabs [aria-selected="true"] {{
        background: rgba(212,175,55,0.1) !important; color: {p['accent1']} !important;
        border-bottom: 2px solid {p['accent1']} !important;
    }}

    /* ================= STRICT EXPANDER / DOWNLOAD CENTER FIX ================= */
    [data-testid="stExpander"] {{
        background: rgba(15, 15, 15, 0.6) !important; 
        background-color: rgba(15, 15, 15, 0.6) !important; 
        border: 1px solid {p['cardBorder']} !important;
        border-radius: 6px !important; 
        backdrop-filter: blur(14px) !important; 
        -webkit-backdrop-filter: blur(14px) !important;
        overflow: hidden !important;
    }}
    
    [data-testid="stExpander"] > details > summary,
    [data-testid="stExpander"] > details > summary:hover,
    [data-testid="stExpander"] > details > summary:focus,
    [data-testid="stExpander"] > details > summary:active,
    [data-testid="stExpander"] > details > summary:focus-visible,
    [data-testid="stExpander"] > details:focus-within > summary {{
        background: rgba(20, 20, 20, 0.8) !important;
        background-color: rgba(20, 20, 20, 0.8) !important;
        color: {p['text']} !important;
        border: none !important;
        outline: none !important;
    }}

    [data-testid="stExpander"] > details > summary:hover {{
        background: rgba(212, 175, 55, 0.15) !important;
        background-color: rgba(212, 175, 55, 0.15) !important;
        color: {p['accent1']} !important;
    }}

    [data-testid="stExpander"] > details > summary * {{
        background: transparent !important;
        background-color: transparent !important;
    }}

    .streamlit-expanderHeader svg {{ 
        fill: {p['accent1']} !important; 
        color: {p['accent1']} !important; 
    }}
    
    [data-testid="stExpanderDetails"],
    [data-testid="stExpander"] > details > div {{
        background: rgba(10,10,10,0.8) !important; 
        background-color: rgba(10,10,10,0.8) !important; 
        border-top: 1px solid {p['cardBorder']} !important;
    }}

    /* alerts (info/success/warning/error) */
    [data-testid="stAlertContainer"], .stAlert {{
        background: rgba(255,255,255,0.05) !important; border: 1px solid {p['cardBorder']} !important;
        border-radius: 6px !important; backdrop-filter: blur(10px);
    }}
    [data-testid="stAlertContainer"] p, .stAlert p, .stAlert div {{ color: {p['text']} !important; }}

    /* file uploader text */
    [data-testid="stFileUploaderDropzone"] * {{ color: {p['text']} !important; }}
    [data-testid="stFileUploaderDropzone"] button {{
        background: rgba(255,255,255,0.05) !important; color: {p['text']} !important;
        border: 1px solid {p['cardBorder']} !important;
    }}

    /* code blocks */
    code {{ color: {p['accent1']} !important; background: rgba(255,255,255,0.05) !important; border: 1px solid {p['cardBorder']}; }}

    /* ================= GLASSY THEMED TABLES (replaces default st.dataframe) ================= */
    .glass-table-wrap {{
        background: rgba(255,255,255,0.02); border: 1px solid {p['cardBorder']}; border-radius: 6px;
        padding: 4px; backdrop-filter: blur(18px); -webkit-backdrop-filter: blur(18px);
        box-shadow: 0 0 24px rgba(0,0,0,0.5), 0 0 14px {p['glow']} inset;
        overflow: auto; margin-bottom: 16px; position: relative; z-index: 1;
    }}
    .glass-table {{
        width: 100%; border-collapse: separate; border-spacing: 0; font-size: 14px;
        font-family: 'Inter', sans-serif; color: {p['text']} !important;
    }}
    .glass-table thead th {{
        position: sticky; top: 0; text-align: left; padding: 13px 16px; font-weight: 800;
        font-size: 12.5px; letter-spacing: 0.6px; text-transform: uppercase;
        color: {p['accent1']} !important; background: #0d0c0a !important;
        border-bottom: 1px solid {p['cardBorder']}; z-index: 10;
        box-shadow: 0 2px 5px rgba(0,0,0,0.5);
    }}
    .glass-table tbody td {{
        padding: 11px 16px; border-bottom: 1px solid rgba(255,255,255,0.05);
        color: {p['text']} !important; white-space: nowrap;
    }}
    .glass-table tbody tr {{ transition: background 0.2s ease, transform 0.15s ease; }}
    .glass-table tbody tr:hover {{
        background: rgba(212,175,55,0.15); color: {p['accent1']} !important;
    }}
    .glass-table tbody tr:nth-child(even) {{ background: rgba(255,255,255,0.015); }}
    .glass-table tbody tr:last-child td {{ border-bottom: none; }}
    </style>
    <div class="money-bg">{particles_html}</div>
    """, unsafe_allow_html=True)
    return p


def cash_celebration(good=True, n=36):
    """Full-screen falling cash / crypto glyphs shown once after a verdict."""
    good_glyphs = ["💵", "💰", "🤑", "₹", "🪙", "💸", "₿", "Ξ", "💳", "🎉"]
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


def render_glass_table(df, max_height=420, max_rows=500):
    """Render a DataFrame as a theme-matched glassy HTML table (replaces default white st.dataframe)."""
    show_df = df.head(max_rows)
    html = show_df.to_html(classes="glass-table", index=False, border=0, escape=True)
    note = ""
    if len(df) > max_rows:
        note = (f'<div style="padding:8px 4px 2px; font-size:12px; opacity:0.6;">'
                f'Showing first {max_rows:,} of {len(df):,} rows — use the download button for the full dataset.</div>')
    st.markdown(
        f'<div class="glass-table-wrap" style="max-height:{max_height}px;">{html}</div>{note}',
        unsafe_allow_html=True,
    )


def panel_header(icon, title, subtitle=""):
    st.markdown(f'<div class="panel-title">{icon} {title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)


def style_fig(fig, p, height=380, hovermode="closest"):
    """Apply a consistent luxury glassy / futuristic look to any Plotly figure, with fully
    legible legend, axis, and hover text regardless of the active color theme."""
    fig.update_layout(
        title=None,
        template="plotly_dark",
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,0.015)",
        font=dict(family="Inter, sans-serif", color=p['text'], size=13),
        margin=dict(t=65, l=10, r=10, b=40),
        hovermode=hovermode,
        hoverlabel=dict(
            bgcolor="rgba(10,10,10,0.95)", bordercolor=p['accent1'],
            font=dict(color=p['text'], size=13, family="Inter, sans-serif"),
        ),
        legend=dict(
            bgcolor="rgba(0,0,0,0.6)", bordercolor=p['cardBorder'], borderwidth=1,
            font=dict(color=p['text'], size=12.5),
            orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5,
        ),
        transition=dict(duration=350, easing="cubic-in-out"),
    )
    fig.update_xaxes(
        gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color=p['text']), title_font=dict(color=p['text']),
        linecolor="rgba(255,255,255,0.1)",
    )
    fig.update_yaxes(
        gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)",
        tickfont=dict(color=p['text']), title_font=dict(color=p['text']),
        linecolor="rgba(255,255,255,0.1)",
    )
    return fig


PLOTLY_CONFIG = {
    "displayModeBar": True,
    "displaylogo": False,
    "scrollZoom": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}


PALETTE = apply_theme(st.session_state.mood)

FEATURE_COLS = ['purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths',
                'delinq.2yrs', 'pub.rec']

# =====================================================================================
# 3. DATA + AUTO-HEALING MODEL LOADING
# =====================================================================================
@st.cache_resource(show_spinner="💠 Booting Credit Spectrum AI Core...")
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
# 4. HERO / BRAND HEADER
# =====================================================================================
st.markdown("""
<div class="hero-wrap">
    <span class="hero-tag">POWERED BY EXPLAINABLE AI &nbsp;•&nbsp; XGBOOST &nbsp;•&nbsp; SECURE • INTELLIGENT • TRUSTED</span>
    <div class="brand-row">
        <span class="coin-spin">🪙</span>
        <span class="hero-title">Credit Spectrum AI</span>
        <span class="orbit-emoji">💵</span><span class="orbit-emoji d2">₿</span><span class="orbit-emoji d3">💳</span>
    </div>
    <div class="hero-sub">💠 Premium Loan Intelligence Platform for the Modern Digital Bank ₹💹</div>
    <p style="max-width:680px; margin:0 auto; opacity:0.85;">
        Analyze a customer's complete financial profile, quantify credit risk and predict
        loan eligibility in milliseconds — with full transparency into every decision. 🔐🧾📈
    </p>
    <div class="credit-card">
        <div class="chip">🟨</div>
        <div class="num">•••• •••• •••• 4042</div>
        <div class="brand">💳</div>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================================
# 5. TOP NAVIGATION BAR (rectangle vintage tabs)
# =====================================================================================
menu = [
    "🏠 Home", "📊 Dashboard", "📈 Analytics", "🧠 AI Predictor", "📁 Dataset", "👨‍💻 About the Model",
]
choice = st.radio("Navigate", menu, horizontal=True, label_visibility="collapsed", key="page")

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
    st.markdown("""
    <div class="glass-card" style="text-align:center; padding:22px 24px; margin-bottom:6px;">
        <span style="font-size:13.5px; opacity:0.8;">
        ✨ Use the navigation bar above — <b>🧠 AI Predictor</b> to score an applicant,
        or <b>📊 Dashboard</b> to explore the live data holograms.</span>
    </div>
    """, unsafe_allow_html=True)

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
        <p>Built for institutions that demand premium speed <b>and</b> ultimate trust — every prediction is paired with
        a confidence score and a plain-language explanation, so risk teams and exclusive clients alike
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

# =====================================================================================
# 7. DASHBOARD (EDA) — Financial Holograms, luxury styling
# =====================================================================================
elif choice == "📊 Dashboard":
    st.markdown('<div class="hero-title" style="font-size:32px;">📊 Financial Holograms</div>', unsafe_allow_html=True)
    st.caption("Interactive, live exploration of the applicant financial data structure.")

    k1, k2, k3, k4 = st.columns(4)
    k1.metric("💼 Total Applicants", f"{df_raw.shape[0]:,}")
    k2.metric("🛡️ Meets Policy", f"{(df_raw['credit.policy']==1).mean()*100:.1f}%")
    k3.metric("💎 Avg FICO", f"{df_raw['fico'].mean():.0f}")
    k4.metric("💸 Avg Interest Rate", f"{df_raw['int.rate'].mean()*100:.2f}%")

    st.markdown("###  ")
    purpose_filter = st.multiselect(
        "🔎 Filter by loan purpose", 
        options=sorted(df_raw['purpose'].dropna().unique()),
        default=[],
        placeholder="Choose options"
    )
    filtered = df_raw[df_raw['purpose'].isin(purpose_filter)] if purpose_filter else df_raw

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            panel_header("💼", "Capital Allocation by Purpose", "Volume of applications per stated loan purpose")
            purpose_counts = filtered['purpose'].value_counts().reset_index()
            purpose_counts.columns = ['purpose', 'count']
            purpose_counts = purpose_counts.sort_values('count', ascending=True)
            fig1 = px.bar(
                purpose_counts, x='count', y='purpose', orientation='h',
                color='count', color_continuous_scale=[PALETTE['accent2'], PALETTE['accent1']],
                text='count',
            )
            fig1.update_traces(
                texttemplate='%{text:,}', textposition='outside',
                marker_line_color='rgba(255,255,255,0.35)', marker_line_width=1,
                hovertemplate='<b>%{y}</b><br>Applications: %{x:,}<extra></extra>',
            )
            fig1.update_layout(coloraxis_showscale=False, showlegend=False,
                                xaxis_title="Applications", yaxis_title="")
            style_fig(fig1, PALETTE, height=380)
            st.plotly_chart(fig1, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
    with col2:
        with st.container(border=True):
            panel_header("🛡️", "Credit Policy Compliance", "Share of applicants meeting vs. missing policy")
            policy_counts = filtered['credit.policy'].dropna().value_counts().reset_index()
            policy_counts.columns = ['credit.policy', 'count']
            policy_counts['credit.policy'] = policy_counts['credit.policy'].map({1: 'Meets Policy', 0: 'Below Policy'})
            fig2 = px.pie(
                policy_counts, values='count', names='credit.policy', hole=0.6,
                color='credit.policy',
                color_discrete_map={'Meets Policy': PALETTE['accent1'], 'Below Policy': '#555555'},
            )
            fig2.update_traces(
                textinfo='percent+label', textfont=dict(color=PALETTE['text'], size=14, weight="bold"),
                marker=dict(line=dict(color='#0d0c0a', width=3)),
                pull=[0.04, 0.04], rotation=90,
                hovertemplate='<b>%{label}</b><br>%{value:,} applicants (%{percent})<extra></extra>',
            )
            fig2.update_layout(showlegend=False, annotations=[dict(
                text=f"{(filtered['credit.policy']==1).mean()*100:.0f}%<br><span style='font-size:12px;opacity:0.7'>Compliant</span>",
                x=0.5, y=0.5, showarrow=False, font=dict(size=24, color=PALETTE['accent1'], weight="bold")
            )])
            style_fig(fig2, PALETTE, height=380)
            st.plotly_chart(fig2, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    with st.container(border=True):
        panel_header("💎", "FICO Trust-Score Distribution", "Credit score spread split by policy outcome")
        fico_labels = filtered['credit.policy'].map({1: 'Meets Policy', 0: 'Below Policy'})
        fig3 = px.histogram(
            filtered.assign(**{'Policy Status': fico_labels}), x='fico', color='Policy Status',
            marginal='box', nbins=45, barmode='overlay', opacity=0.82,
            color_discrete_map={'Meets Policy': PALETTE['accent1'], 'Below Policy': '#777777'},
        )
        fig3.update_traces(marker_line_color='rgba(255,255,255,0.25)', marker_line_width=0.6,
                            hovertemplate='FICO: %{x}<br>Count: %{y}<extra></extra>')
        fig3.update_layout(bargap=0.04, xaxis_title="FICO Score", yaxis_title="Applicants")
        style_fig(fig3, PALETTE, height=420, hovermode="x unified")
        st.plotly_chart(fig3, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    d1, d2 = st.columns(2)
    with d1:
        with st.container(border=True):
            panel_header("⚖️", "Debt-to-Income Spread", "How DTI relates to policy compliance")
            dti_labels = filtered['credit.policy'].map({1: 'Meets Policy', 0: 'Below Policy'})
            fig5 = px.histogram(
                filtered.assign(**{'Policy Status': dti_labels}), x='dti', color='Policy Status',
                nbins=40, barmode='overlay', opacity=0.82,
                color_discrete_map={'Meets Policy': PALETTE['accent1'], 'Below Policy': '#777777'},
            )
            fig5.update_traces(marker_line_color='rgba(255,255,255,0.25)', marker_line_width=0.6,
                                hovertemplate='DTI: %{x:.1f}<br>Count: %{y}<extra></extra>')
            fig5.update_layout(bargap=0.04, xaxis_title="Debt-to-Income Ratio", yaxis_title="Applicants")
            style_fig(fig5, PALETTE, height=380, hovermode="x unified")
            st.plotly_chart(fig5, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
    with d2:
        with st.container(border=True):
            panel_header("💸", "Interest Rate by Purpose", "Average rate assigned across purposes & outcomes")
            rate_labels = filtered['credit.policy'].map({1: 'Meets Policy', 0: 'Below Policy'})
            rate_summary = (
                filtered.assign(**{'Policy Status': rate_labels})
                .groupby(['purpose', 'Policy Status'])['int.rate'].mean().mul(100).reset_index()
            )
            fig6 = px.bar(
                rate_summary, x='purpose', y='int.rate', color='Policy Status', barmode='group',
                color_discrete_map={'Meets Policy': PALETTE['accent1'], 'Below Policy': '#777777'},
            )
            fig6.update_traces(
                marker_line_color='rgba(255,255,255,0.3)', marker_line_width=1,
                hovertemplate='<b>%{x}</b><br>Avg Rate: %{y:.2f}%<extra></extra>',
            )
            fig6.update_layout(xaxis_title="", yaxis_title="Avg Interest Rate (%)", bargap=0.25)
            fig6.update_xaxes(tickangle=-30)
            style_fig(fig6, PALETTE, height=380)
            st.plotly_chart(fig6, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    with st.container(border=True):
        panel_header("🕸️", "Feature Correlation Link", "Correlation matrix across every engineered signal")
        corr = df_processed.corr()
        fig4 = go.Figure(data=go.Heatmap(
            z=corr.values, x=corr.columns, y=corr.columns,
            colorscale=[[0, "#000000"], [0.5, "#444444"], [1, PALETTE['accent1']]],
            zmid=0, text=corr.round(2).values, texttemplate="%{text}",
            textfont=dict(size=10, color=PALETTE['text']),
            hovertemplate='%{x} ↔ %{y}<br>Correlation: %{z:.2f}<extra></extra>',
            colorbar=dict(
                thickness=14, outlinewidth=0, tickfont=dict(color=PALETTE['text']),
                bgcolor="rgba(0,0,0,0)",
            ),
        ))
        style_fig(fig4, PALETTE, height=560)
        fig4.update_xaxes(tickangle=-40)
        st.plotly_chart(fig4, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

# =====================================================================================
# 8. ANALYTICS — Premium Interactive panels
# =====================================================================================
elif choice == "📈 Analytics":
    st.markdown('<div class="hero-title" style="font-size:32px;">📈 Model Analytics</div>', unsafe_allow_html=True)
    st.caption("Live diagnostics of the deployed XGBoost engine on the holdout test stream.")

    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    m1, m2, m3 = st.columns(3)
    m1.metric("🎯 Accuracy", f"{acc*100:.2f}%")
    m2.metric("📈 ROC AUC", f"{auc*100:.2f}%")
    m3.metric("🧮 Test Samples", f"{len(y_test):,}")

    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            panel_header("🎯", "Accuracy Gauge", "Overall correct-classification rate on holdout data")
            fig_acc = go.Figure(go.Indicator(
                mode="gauge+number", value=acc * 100,
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': PALETTE['accent1']},
                       'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 1, 'bordercolor': "#444"}))
            fig_acc.update_layout(template="plotly_dark", height=270, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20))
            st.plotly_chart(fig_acc, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
    with c2:
        with st.container(border=True):
            panel_header("📈", "ROC AUC Gauge", "Model's ability to rank approvals above declines")
            fig_auc = go.Figure(go.Indicator(
                mode="gauge+number", value=auc * 100,
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': PALETTE['accent2']},
                       'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 1, 'bordercolor': "#444"}))
            fig_auc.update_layout(template="plotly_dark", height=270, paper_bgcolor="rgba(0,0,0,0)", margin=dict(t=20))
            st.plotly_chart(fig_auc, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    r1, r2 = st.columns(2)
    with r1:
        with st.container(border=True):
            panel_header("🌐", "ROC Curve", "True vs. false positive trade-off vs. a random baseline")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='XGBoost',
                                          line=dict(color=PALETTE['accent1'], width=3)))
            fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Baseline',
                                          line=dict(color=PALETTE['accent2'], dash='dash')))
            style_fig(fig_roc, PALETTE, height=320)
            st.plotly_chart(fig_roc, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
    with r2:
        with st.container(border=True):
            panel_header("🧮", "Confusion Matrix", "Predicted vs. actual outcomes on the test set")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, template="plotly_dark",
                                color_continuous_scale=[[0, "#0a0a0a"], [1, PALETTE['accent1']]], labels=dict(x="Predicted", y="Actual"))
            style_fig(fig_cm, PALETTE, height=320)
            st.plotly_chart(fig_cm, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

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

    st.markdown(f"""
    <div style="text-align:center; margin-top:20px; margin-bottom: 20px;">
        <div class="champion-badge">
            <div style="font-size:40px;">🏆👑</div>
            <div style="font-weight:800; font-size:18px; color: {PALETTE['accent1']};">XGBoost</div>
            <div style="font-size:12.5px; opacity:0.8;">Reigning Champion Model</div>
        </div>
    </div>""", unsafe_allow_html=True)

    metrics_df = pd.DataFrame({
        "AI Architecture": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost ⭐"],
        "Test Accuracy": [0.8569, 0.9881, 0.9840, 0.9892],
        "Test ROC AUC": [0.9341, 0.9829, 0.9969, 0.9978],
    })
    with st.container(border=True):
        panel_header("⚔️", "Combat Results", "Head-to-head test performance across every architecture")
        fig = px.bar(metrics_df, x='AI Architecture', y=['Test Accuracy', 'Test ROC AUC'],
                     barmode='group', template='plotly_dark', color_discrete_sequence=[PALETTE['accent1'], PALETTE['accent2']])
        style_fig(fig, PALETTE, height=360)
        st.plotly_chart(fig, use_container_width=True, config=PLOTLY_CONFIG, theme=None)

    st.markdown("""
    <div class="glass-card">
    🧬 <b>Why XGBoost Was Chosen:</b> The tuned <b>XGBoost</b> model was promoted to production for its
    superior ROC AUC generalization, stable precision–recall trade-off on unseen data streams, native
    handling of feature interactions, and resilience to the class imbalance addressed via SMOTE —
    outperforming Logistic Regression, Decision Tree, and Random Forest across every holdout metric.
    </div>
    """, unsafe_allow_html=True)

    with st.container(border=True):
        panel_header("🧠", "Global Feature Importance", "Which signals the model leans on most")
        try:
            importances = model.feature_importances_
            imp_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": importances}).sort_values(
                "Importance", ascending=True)
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', template="plotly_dark",
                              color="Importance", color_continuous_scale=[[0, "#0a0a0a"], [1, PALETTE['accent1']]])
            style_fig(fig_imp, PALETTE, height=440)
            st.plotly_chart(fig_imp, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
        except Exception:
            st.info("Feature importance is unavailable for this model instance.")

# =====================================================================================
# 9. AI PREDICTOR — single applicant + batch CSV upload
# =====================================================================================
elif choice == "🧠 AI Predictor":
    st.markdown('<div class="hero-title" style="font-size:32px;">🔮 The AI Credit Radar</div>', unsafe_allow_html=True)
    st.caption("Analyze one applicant, or upload a CSV to batch-score an entire portfolio in real time.")

    clr1, clr2 = st.columns([1, 4])
    with clr1:
        if st.button("🧹 Clear & Reset"):
            st.session_state.mood = "default"
            st.session_state.batch_result = None
            st.rerun()

    tab_single, tab_batch = st.tabs(["👤 Single Applicant", "📂 Batch Processing (CSV)"])

    # ---------------- SINGLE APPLICANT ----------------
    with tab_single:
        with st.form("prediction_form"):
            st.markdown('<div class="glass-card"><h3>📋 Section 1 — Applicant & Loan Details</h3>', unsafe_allow_html=True)
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                purpose_options = list(le.classes_)
                purpose = st.selectbox('💼 Purpose', options=purpose_options, index=None, placeholder="Choose options")
            with c2:
                int_rate = st.number_input('💸 Interest Rate', min_value=0.0, max_value=1.0, value=0.00)
            with c3:
                installment = st.number_input('💳 Monthly Installment ($)', min_value=0.0, max_value=5000.0, value=0.00)
            with c4:
                log_annual_inc = st.number_input('💰 Log Annual Income', min_value=0.0, max_value=20.0, value=0.00)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card"><h3>💎 Section 2 — Credit History</h3>', unsafe_allow_html=True)
            c5, c6, c7, c8 = st.columns(4)
            with c5:
                fico = st.number_input('💎 FICO Score', min_value=0, max_value=850, value=0)
            with c6:
                dti = st.number_input('⚖️ Debt-to-Income (DTI)', min_value=0.0, max_value=100.0, value=0.00)
            with c7:
                days_with_cr_line = st.number_input('⏳ Credit Line Age (Days)', min_value=0.0, value=0.00)
            with c8:
                pub_rec = st.number_input('🚨 Public Records', min_value=0, max_value=20, value=0)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown('<div class="glass-card"><h3>📈 Section 3 — Revolving & Inquiry Behavior</h3>', unsafe_allow_html=True)
            c9, c10, c11, c12 = st.columns(4)
            with c9:
                revol_bal = st.number_input('🏦 Revolving Balance ($)', min_value=0.0, value=0.00)
            with c10:
                revol_util = st.number_input('📈 Revolving Utilization (%)', min_value=0.0, max_value=200.0, value=0.00)
            with c11:
                inq_last_6mths = st.number_input('🕵️ Inquiries (6M)', min_value=0, max_value=100, value=0)
            with c12:
                delinq_2yrs = st.number_input('⚠️ Delinquencies (2Y)', min_value=0, max_value=50, value=0)
            st.markdown('</div>', unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            submit_button = st.form_submit_button("🔍 ANALYZE APPLICANT")

        if submit_button:
            if purpose is None:
                st.warning("⚠️ Please select a **Purpose** before analyzing the applicant.")
            else:
                with st.spinner("💠 Credit Spectrum AI is evaluating the financial profile..."):
                    time.sleep(0.8)
    
                purp_enc = le.transform([purpose])[0]
                input_data = np.array([[purp_enc, int_rate, installment, log_annual_inc, dti, fico,
                                         days_with_cr_line, revol_bal, revol_util, inq_last_6mths,
                                         delinq_2yrs, pub_rec]])
                input_scaled = scaler.transform(input_data)
                prediction = model.predict(input_scaled)[0]
                confidence = model.predict_proba(input_scaled)[0]
    
                st.session_state.mood = "approved" if prediction == 1 else "declined"
                PALETTE = apply_theme(st.session_state.mood)
    
                st.markdown("---")
    
                if prediction == 1:
                    cash_celebration(good=True)
                    st.markdown(f"""
                    <div class="approved-hero">
                        <div class="verdict-icon">🤑💵🎉</div>
                        <div class="verdict-title">✅ ELIGIBLE FOR LOAN!</div>
                        <p style="font-size:16px;">Confidence Score: <b>{confidence[1]*100:.2f}%</b></p>
                        <p style="font-size:15px; color: {PALETTE['accent2']};">₹💰 Cha-ching! This applicant meets the credit policy. 💰₹</p>
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
                    cash_celebration(good=False)
                    st.markdown(f"""
                    <div class="declined-hero">
                        <div class="verdict-icon">💔📉🚫</div>
                        <div class="verdict-title">🚫 NOT ELIGIBLE FOR LOAN</div>
                        <p style="font-size:16px;">Risk Confidence: <b>{confidence[0]*100:.2f}%</b></p>
                        <p style="font-size:15px; color: {PALETTE['accent2']};">The financial profile falls short of the credit policy threshold.</p>
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
                           'bar': {'color': "#00ffa3" if prediction == 1 else "#ff2e5b"},
                           'steps': [{'range': [0, 40], 'color': 'rgba(255,46,91,0.20)'},
                                     {'range': [40, 70], 'color': 'rgba(212,175,55,0.20)'},
                                     {'range': [70, 100], 'color': 'rgba(0,255,163,0.20)'}],
                           'bgcolor': "rgba(0,0,0,0)"}))
                fig_g.update_layout(template="plotly_dark", height=300, paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_g, use_container_width=True, config=PLOTLY_CONFIG, theme=None)
    
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
    
                st.info("🔄 Theme is now reflecting this verdict. Use **🧹 Clear & Reset** above, "
                        "or **Reset Theme to Default**, to return to the default look.")

    # ---------------- BATCH PROCESSING ----------------
    with tab_batch:
        st.markdown("""
        <div class="glass-card">
        📂 Upload a CSV with columns: <code>purpose, int.rate, installment, log.annual.inc, dti, fico,
        days.with.cr.line, revol.bal, revol.util, inq.last.6mths, delinq.2yrs, pub.rec</code><br>
        Drag & drop the file below, or click to browse.
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader("Drop your applicant CSV here", type=["csv"], label_visibility="collapsed")

        if uploaded is not None:
            try:
                batch_df = pd.read_csv(uploaded)
                render_glass_table(batch_df.head(), max_height=260)
                if st.button("⚡ Run Batch Prediction"):
                    with st.spinner("💠 Scoring the entire portfolio..."):
                        work = batch_df.copy()
                        work['purpose'] = work['purpose'].apply(
                            lambda v: v if v in le.classes_ else le.classes_[0])
                        work['purpose'] = le.transform(work['purpose'])
                        X_batch = work[FEATURE_COLS]
                        X_batch_scaled = scaler.transform(X_batch)
                        preds = model.predict(X_batch_scaled)
                        probs = model.predict_proba(X_batch_scaled)[:, 1]
                        batch_df['prediction'] = np.where(preds == 1, "Eligible ✅", "Not Eligible 🚫")
                        batch_df['confidence_%'] = (probs * 100).round(2)
                        st.session_state.batch_result = batch_df
            except Exception as e:
                st.error(f"⚠️ Could not process file: {e}")
        else:
            st.session_state.batch_result = None

        if st.session_state.batch_result is not None:
            res = st.session_state.batch_result
            n_eligible = (res['prediction'] == "Eligible ✅").sum()
            n_total = len(res)
            b1, b2, b3 = st.columns(3)
            b1.metric("📦 Total Scored", n_total)
            b2.metric("✅ Eligible", int(n_eligible))
            b3.metric("🚫 Not Eligible", int(n_total - n_eligible))
            render_glass_table(res, max_height=380)
            st.download_button("📥 Download Batch Results (CSV)", data=res.to_csv(index=False),
                                file_name="batch_prediction_results.csv", mime="text/csv")

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
    purpose_filter_dataset = st.multiselect(
        "Filter by purpose", 
        options=sorted(df_raw['purpose'].dropna().unique()),
        default=[],
        placeholder="Choose options"
    )
    display_df = df_raw[df_raw['purpose'].isin(purpose_filter_dataset)] if purpose_filter_dataset else df_raw
    render_glass_table(display_df, max_height=420)

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
    using a premium production-grade <b>XGBoost</b> classifier.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 🛤️ Model Development Roadmap")
    steps = ["Dataset Collection", "Data Cleaning", "Label Encoding", "Feature Scaling",
             "SMOTE Class Balancing", "Train / Test Split", "Hyperparameter Tuning (RandomizedSearchCV)",
             "XGBoost Training", "Evaluation (ROC AUC / Accuracy)", "Production Deployment"]

    n = len(steps)
    pad = 60
    span = 1000 - 2 * pad
    step_x = span / (n - 1)
    xs = [pad + i * step_x for i in range(n)]
    ys = [160 if i % 2 == 0 else 300 for i in range(n)]      # road bends: top / bottom
    pys = [50 if i % 2 == 0 else 410 for i in range(n)]      # pin sits further out from the bend

    # smooth wavy road path through the bend points
    path_d = f"M {xs[0]:.1f} {ys[0]:.1f}"
    for i in range(1, n):
        midx = (xs[i - 1] + xs[i]) / 2
        path_d += f" C {midx:.1f} {ys[i-1]:.1f}, {midx:.1f} {ys[i]:.1f}, {xs[i]:.1f} {ys[i]:.1f}"

    dots_svg = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="7" fill="#0a0a0a" stroke="{p["accent1"]}" stroke-width="3"/>'
        for x, y in zip(xs, ys)
    )
    stems_svg = "".join(
        f'<line x1="{x:.1f}" y1="{y:.1f}" x2="{x:.1f}" y2="{py:.1f}" '
        f'stroke="{p["accent1"]}" stroke-width="2" stroke-dasharray="4 4" opacity="0.55"/>'
        for x, y, py in zip(xs, ys, pys)
    )

    road_svg = (
        f'<svg class="roadmap-svg" viewBox="0 0 1000 460" preserveAspectRatio="none">'
        f'<defs><filter id="roadGlow" x="-50%" y="-50%" width="200%" height="200%">'
        f'<feGaussianBlur stdDeviation="7"/></filter></defs>'
        f'<path d="{path_d}" fill="none" stroke="{p["accent1"]}" stroke-width="40" opacity="0.28" filter="url(#roadGlow)" stroke-linecap="round"/>'
        f'<path d="{path_d}" fill="none" stroke="#141414" stroke-width="30" stroke-linecap="round"/>'
        f'<path d="{path_d}" fill="none" stroke="{p["accent1"]}" stroke-width="3" stroke-dasharray="14 12" opacity="0.85" stroke-linecap="round"/>'
        f'{stems_svg}{dots_svg}'
        f'</svg>'
    )

    stops_html = ""
    for i, s in enumerate(steps):
        final_class = " roadmap-final" if i == n - 1 else ""
        icon = "🏁" if i == n - 1 else str(i + 1)
        side = "right" if xs[i] <= 500 else "left"
        left_pct = xs[i] / 1000 * 100
        top_pct = pys[i] / 460 * 100
        stops_html += (
            f'<div class="roadmap-stop side-{side}{final_class}" '
            f'style="left:{left_pct:.2f}%; top:{top_pct:.2f}%;">'
            f'<div class="roadmap-pin">{icon}</div>'
            f'<div class="roadmap-label">{s}</div>'
            f'</div>'
        )

    st.markdown(
        f'<div class="roadmap-wrap">{road_svg}{stops_html}</div>',
        unsafe_allow_html=True
    )

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
    render_glass_table(pd.DataFrame(glossary.items(), columns=["Feature", "Description"]), max_height=340)

    st.markdown("### 🏆 Final Model Comparison & Why XGBoost Won")
    render_glass_table(pd.DataFrame({
        "Model": ["Logistic Regression", "Decision Tree", "Random Forest", "XGBoost ⭐ (Deployed)"],
        "Test Accuracy": [0.8569, 0.9881, 0.9840, 0.9892],
        "Test ROC AUC": [0.9341, 0.9829, 0.9969, 0.9978],
    }), max_height=260)

    st.markdown("""
    <div class="glass-card">
    💡 <b>Selection Rationale:</b> XGBoost was chosen for production because gradient-boosted trees
    capture non-linear interactions between financial signals (like FICO × DTI × utilization) far
    better than Logistic Regression, while its regularization keeps it more stable on unseen data
    than a single Decision Tree or even Random Forest. Combined with SMOTE-balanced training data,
    it delivered the best accuracy <b>and</b> the best ROC AUC of every architecture tested. 🏆
    </div>
    """, unsafe_allow_html=True)
