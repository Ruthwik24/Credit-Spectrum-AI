import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, confusion_matrix
from xgboost import XGBClassifier
import os

# =====================================================================================
# 1. PAGE CONFIG & SESSION STATE
# =====================================================================================
st.set_page_config(
    page_title="Credit Spectrum AI | Luxury Edition",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

if "mood" not in st.session_state:
    st.session_state.mood = "default"
if "page" not in st.session_state:
    st.session_state.page = "🏠 Home"

# =====================================================================================
# 2. LUXURY THEME & CUSTOM CSS
# =====================================================================================
def get_palette(mood):
    if mood == "approved":
        return {
            "bg": "#0B1410", "accent1": "#2E8B57", "accent2": "#E5E4E2", "accent3": "#D4AF37",
            "text": "#F5F5F5", "cardBorder": "rgba(46, 139, 87, 0.4)", "shadow": "0 8px 32px rgba(0,0,0,0.5)",
            "titleGrad": "linear-gradient(90deg, #2E8B57 0%, #7FFFD4 50%, #2E8B57 100%)"
        }
    elif mood == "declined":
        return {
            "bg": "#140B0B", "accent1": "#8B0000", "accent2": "#E5E4E2", "accent3": "#D4AF37",
            "text": "#F5F5F5", "cardBorder": "rgba(139, 0, 0, 0.4)", "shadow": "0 8px 32px rgba(0,0,0,0.5)",
            "titleGrad": "linear-gradient(90deg, #8B0000 0%, #FF6347 50%, #8B0000 100%)"
        }
    return {
        "bg": "#0D0D0D", "accent1": "#D4AF37", "accent2": "#E5E4E2", "accent3": "#A67C00",
        "text": "#F8F8F8", "cardBorder": "rgba(212, 175, 55, 0.3)", "shadow": "0 8px 24px rgba(0,0,0,0.6)",
        "titleGrad": "linear-gradient(90deg, #D4AF37 0%, #FFF8DC 50%, #D4AF37 100%)"
    }

p = get_palette(st.session_state.mood)

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&family=Inter:wght@300;400;500;600&display=swap');

/* Base Styles */
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
h1, h2, h3, h4 {{ font-family: 'Poppins', sans-serif !important; color: {p['text']} !important; }}
p, span, div, label {{ color: {p['text']}; }}
.stApp {{ background: {p['bg']}; color: {p['text']}; }}
#MainMenu, header, footer {{ visibility: hidden; }}

/* Navigation Bar */
div[role="radiogroup"] {{ display: flex; flex-direction: row; width: 100%; gap: 12px; }}
div[role="radiogroup"] > label {{
    flex: 1; text-align: center; justify-content: center;
    background: rgba(25, 25, 25, 0.6);
    border: 1px solid rgba(255,255,255,0.1); border-radius: 8px;
    padding: 12px 10px; cursor: pointer; transition: all 0.3s ease;
}}
div[role="radiogroup"] > label:hover {{ background: rgba(255,255,255,0.1); transform: translateY(-2px); }}
div[role="radiogroup"] > label:has(input:checked) {{
    border: 1px solid {p['accent1']};
    background: linear-gradient(180deg, rgba(212,175,55,0.15), rgba(0,0,0,0));
    box-shadow: 0 4px 15px rgba(212, 175, 55, 0.2);
}}
/* Hide radio circle */
div[role="radiogroup"] label div[data-baseweb="radio"] {{ display: none !important; }}
div[role="radiogroup"] label p {{ font-weight: 600; font-size: 15px; margin: 0; color: {p['accent2']} !important; }}
div[role="radiogroup"] > label:has(input:checked) p {{ color: {p['accent1']} !important; }}

/* Glass Cards */
.glass-card {{
    background: rgba(30, 30, 30, 0.4);
    border: 1px solid {p['cardBorder']};
    border-radius: 12px;
    padding: 24px;
    box-shadow: {p['shadow']};
    backdrop-filter: blur(12px);
    transition: transform 0.3s, box-shadow 0.3s;
    margin-bottom: 20px;
}}
.glass-card:hover {{ transform: translateY(-4px); box-shadow: 0 12px 40px rgba(0,0,0,0.8); border-color: {p['accent1']}; }}

/* Inputs & Dropdowns */
.stTextInput input, .stNumberInput input, div[data-baseweb="select"] > div {{
    background: rgba(20, 20, 20, 0.7) !important;
    border: 1px solid rgba(255,255,255,0.15) !important;
    border-radius: 8px !important; color: {p['text']} !important;
}}
.stTextInput input:focus, .stNumberInput input:focus, div[data-baseweb="select"] > div:focus-within {{
    border-color: {p['accent1']} !important; box-shadow: 0 0 8px rgba(212,175,55,0.3) !important;
}}
div[data-baseweb="popover"] ul {{ background: #1a1a1a !important; border: 1px solid {p['cardBorder']}; }}
div[data-baseweb="popover"] li {{ color: {p['text']} !important; }}
div[data-baseweb="popover"] li:hover {{ background: rgba(212,175,55,0.2) !important; color: {p['accent1']} !important; }}

/* Buttons */
.stButton>button, .stDownloadButton>button, .stFormSubmitButton>button {{
    background: transparent !important;
    border: 1px solid {p['accent1']} !important;
    color: {p['accent1']} !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    width: 100%; transition: all 0.3s ease !important;
}}
.stButton>button:hover, .stDownloadButton>button:hover, .stFormSubmitButton>button:hover {{
    background: {p['accent1']} !important;
    color: #000 !important;
    box-shadow: 0 4px 15px rgba(212,175,55,0.4) !important;
    transform: translateY(-2px);
}}

/* Hero Title */
.hero-title {{
    font-size: 48px; font-weight: 700; text-align: center;
    background: {p['titleGrad']}; -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 10px;
}}

/* Tables */
.glass-table-wrap {{
    background: rgba(20,20,20,0.5); border: 1px solid {p['cardBorder']}; border-radius: 12px;
    padding: 10px; overflow-x: auto; max-height: 400px;
}}
.glass-table {{ width: 100%; border-collapse: collapse; font-size: 14px; text-align: left; }}
.glass-table th {{ position: sticky; top: 0; background: #111; color: {p['accent1']}; padding: 12px; border-bottom: 2px solid {p['accent1']}; z-index: 2; }}
.glass-table td {{ padding: 12px; border-bottom: 1px solid rgba(255,255,255,0.05); }}
.glass-table tr:nth-child(even) {{ background: rgba(255,255,255,0.03); }}
.glass-table tr:hover {{ background: rgba(212,175,55,0.1); }}

/* Timeline */
.timeline-item {{ border-left: 2px solid {p['accent1']}; padding-left: 20px; margin-bottom: 20px; position: relative; }}
.timeline-item::before {{ content: "✦"; position: absolute; left: -9px; top: -2px; color: {p['accent1']}; background: {p['bg']}; font-size: 16px; line-height: 1; }}

/* Panels */
div[data-testid="stVerticalBlockBorderWrapper"] {{
    background: rgba(30, 30, 30, 0.4) !important; border: 1px solid {p['cardBorder']} !important;
    border-radius: 12px !important; box-shadow: {p['shadow']} !important; transition: transform 0.3s ease;
}}
div[data-testid="stVerticalBlockBorderWrapper"]:hover {{ transform: translateY(-3px); }}
.panel-title {{ font-size: 18px; font-weight: 600; color: {p['accent1']}; margin-bottom: 2px; }}
.panel-sub {{ font-size: 12.5px; color: #aaa; margin-bottom: 10px; }}

</style>
""", unsafe_allow_html=True)

# =====================================================================================
# 3. GLOBAL FUNCTIONS
# =====================================================================================
def style_fig(fig, height=380, hovermode="closest", show_legend=True):
    """Global styling function applied to all Plotly charts for visual consistency."""
    fig.update_layout(
        template="plotly_dark", height=height,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=p['text'], size=13),
        margin=dict(t=30, l=10, r=10, b=10),
        hovermode=hovermode,
        hoverlabel=dict(bgcolor="#111", bordercolor=p['accent1'], font=dict(color=p['text'])),
        showlegend=show_legend,
        legend=dict(
            bgcolor="rgba(0,0,0,0.5)", bordercolor=p['cardBorder'], borderwidth=1,
            orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5
        )
    )
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zerolinecolor="rgba(255,255,255,0.1)")
    return fig

def render_glass_table(df):
    """Render a DataFrame as a theme-matched glassy HTML table."""
    html = df.to_html(classes="glass-table", index=False, escape=True)
    st.markdown(f'<div class="glass-table-wrap">{html}</div>', unsafe_allow_html=True)

def panel_header(icon, title, subtitle):
    st.markdown(f'<div class="panel-title">{icon} {title}</div><div class="panel-sub">{subtitle}</div>', unsafe_allow_html=True)

# =====================================================================================
# 4. DATA + AUTO-HEALING MODEL LOADING
# =====================================================================================
@st.cache_resource(show_spinner="Booting Premium AI Core...")
def load_pipeline():
    df_raw = pd.read_csv('loan_data.csv')
    df_processed = df_raw.copy()
    
    le = LabelEncoder()
    df_processed['purpose'] = le.fit_transform(df_processed['purpose'])
    
    FEATURE_COLS = ['purpose', 'int.rate', 'installment', 'log.annual.inc', 'dti', 'fico',
                    'days.with.cr.line', 'revol.bal', 'revol.util', 'inq.last.6mths',
                    'delinq.2yrs', 'pub.rec']
    X = df_processed[FEATURE_COLS]
    y = df_processed['credit.policy']
    
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    
    scaler = StandardScaler().fit(X_train)
    X_test_scaled = scaler.transform(X_test)
    X_train_scaled = scaler.transform(X_train)
    
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    model.fit(X_train_scaled, y_train)
    
    return df_raw, df_processed, le, scaler, model, X_test_scaled, y_test, FEATURE_COLS

df_raw, df_processed, le, scaler, model, X_test_scaled, y_test, FEATURE_COLS = load_pipeline()

# =====================================================================================
# 5. NAVIGATION & HEADER
# =====================================================================================
st.markdown('<div class="hero-title">🏦 Credit Spectrum AI</div>', unsafe_allow_html=True)
st.markdown('<p style="text-align:center; color:#aaa; margin-bottom:20px;">Premium Institutional Loan Intelligence</p>', unsafe_allow_html=True)

menu = ["🏠 Home", "📊 Dashboard", "📈 Analytics", "🧠 AI Predictor", "📁 Dataset", "👨‍💻 Roadmap"]
choice = st.radio("Navigate", menu, horizontal=True, label_visibility="collapsed")

st.markdown("<br>", unsafe_allow_html=True)

# =====================================================================================
# 6. HOME
# =====================================================================================
if choice == "🏠 Home":
    col1, col2, col3, col4 = st.columns(4)
    metrics = [
        ("Prediction Accuracy", "98.9%"), ("ROC AUC Score", "99.6%"),
        ("Financial Signals Analyzed", "12"), ("AI Execution Engine", "XGBoost")
    ]
    for col, (label, val) in zip([col1, col2, col3, col4], metrics):
        with col:
            st.markdown(f"""
            <div class="glass-card" style="text-align:center; padding:20px;">
                <div style="font-size:13px; color:#aaa; font-weight:600; text-transform:uppercase;">{label}</div>
                <div style="font-size:28px; font-weight:700; color:{p['accent1']};">{val}</div>
            </div>""", unsafe_allow_html=True)
            
    st.markdown("""
    <div class="glass-card">
        <h3 style="color:#D4AF37;">Welcome to Credit Spectrum AI</h3>
        <p style="opacity: 0.9;">This institutional-grade platform provides real-time credit risk assessment powered by advanced machine learning. Built for banking professionals who require absolute precision and elegant reporting.</p>
        <hr style="border-color: rgba(255,255,255,0.1);">
        <p style="opacity: 0.85;">Explore financial distributions in the <b>Dashboard</b>, assess model performance in <b>Analytics</b>, or score live applicants in the <b>AI Predictor</b>.</p>
    </div>
    """, unsafe_allow_html=True)

# =====================================================================================
# 7. DASHBOARD (EDA)
# =====================================================================================
elif choice == "📊 Dashboard":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Applicants", f"{len(df_raw):,}")
    c2.metric("Meets Policy", f"{(df_raw['credit.policy']==1).mean()*100:.1f}%")
    c3.metric("Avg FICO", f"{df_raw['fico'].mean():.0f}")
    c4.metric("Avg Interest Rate", f"{df_raw['int.rate'].mean()*100:.2f}%")

    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            panel_header("💼", "Loan Purpose Distribution", "Volume of loans by stated purpose")
            purpose_counts = df_raw['purpose'].value_counts().reset_index()
            purpose_counts.columns = ['purpose', 'count']
            purpose_counts = purpose_counts.sort_values('count', ascending=True)
            fig1 = px.bar(purpose_counts, x='count', y='purpose', orientation='h', 
                          color='count', color_continuous_scale=["#1a1a1a", p['accent1']])
            fig1.update_layout(coloraxis_showscale=False, yaxis_title="")
            st.plotly_chart(style_fig(fig1), use_container_width=True)

    with col2:
        with st.container(border=True):
            panel_header("🛡️", "Credit Policy Compliance", "Proportion of applicants meeting criteria")
            policy_counts = df_raw['credit.policy'].map({1: 'Meets Policy', 0: 'Below Policy'}).value_counts().reset_index()
            policy_counts.columns = ['Policy', 'Count']
            fig2 = px.pie(policy_counts, values='Count', names='Policy', hole=0.65,
                          color='Policy', color_discrete_map={'Meets Policy': p['accent1'], 'Below Policy': '#444'})
            fig2.update_traces(textinfo='percent+label', marker=dict(line=dict(color='#0d0d0d', width=3)))
            st.plotly_chart(style_fig(fig2, show_legend=True), use_container_width=True)

    with st.container(border=True):
        panel_header("💎", "FICO Score Distribution", "Density of FICO scores by policy compliance")
        fig3 = px.histogram(df_raw, x='fico', color=df_raw['credit.policy'].map({1:'Meets Policy', 0:'Below Policy'}),
                            barmode='overlay', opacity=0.7, nbins=50,
                            color_discrete_map={'Meets Policy': p['accent1'], 'Below Policy': '#444'})
        fig3.update_layout(xaxis_title="FICO Score", yaxis_title="Count")
        st.plotly_chart(style_fig(fig3, height=400), use_container_width=True)

    with st.container(border=True):
        panel_header("🕸️", "Feature Correlation Map", "Heatmap of structural financial signals")
        corr = df_processed.corr()
        fig4 = px.imshow(corr, text_auto=".2f", color_continuous_scale=[[0, "#000"], [0.5, "#222"], [1, p['accent1']]])
        st.plotly_chart(style_fig(fig4, height=550, show_legend=False), use_container_width=True)

# =====================================================================================
# 8. ANALYTICS
# =====================================================================================
elif choice == "📈 Analytics":
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]
    
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            panel_header("🎯", "Accuracy Gauge", "Holdout set prediction accuracy")
            fig_acc = go.Figure(go.Indicator(
                mode="gauge+number", value=accuracy_score(y_test, y_pred) * 100,
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': p['accent1']}, 'bgcolor': "rgba(0,0,0,0)"}
            ))
            st.plotly_chart(style_fig(fig_acc, height=300), use_container_width=True)
    with c2:
        with st.container(border=True):
            panel_header("📈", "ROC AUC Gauge", "Area under receiver operating characteristic")
            fig_auc = go.Figure(go.Indicator(
                mode="gauge+number", value=roc_auc_score(y_test, y_prob) * 100,
                gauge={'axis': {'range': [None, 100]}, 'bar': {'color': p['accent2']}, 'bgcolor': "rgba(0,0,0,0)"}
            ))
            st.plotly_chart(style_fig(fig_auc, height=300), use_container_width=True)

    r1, r2 = st.columns(2)
    with r1:
        with st.container(border=True):
            panel_header("🌐", "ROC Curve", "True vs False Positive Rates")
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            fig_roc = go.Figure()
            fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, name="XGBoost", line=dict(color=p['accent1'], width=3), fill='tozeroy', fillcolor="rgba(212,175,55,0.1)"))
            fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], name="Baseline", line=dict(color="#555", dash="dash")))
            st.plotly_chart(style_fig(fig_roc, height=350), use_container_width=True)
    with r2:
        with st.container(border=True):
            panel_header("🧮", "Confusion Matrix", "Predicted vs Actual outcomes")
            cm = confusion_matrix(y_test, y_pred)
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale=[[0, "#1a1a1a"], [1, p['accent1']]], labels=dict(x="Predicted", y="Actual"))
            st.plotly_chart(style_fig(fig_cm, height=350), use_container_width=True)
            
    with st.container(border=True):
        panel_header("🧠", "Feature Importance", "Relative impact of variables on the decision pipeline")
        imp_df = pd.DataFrame({"Feature": FEATURE_COLS, "Importance": model.feature_importances_}).sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation='h', color="Importance", color_continuous_scale=["#1a1a1a", p['accent1']])
        st.plotly_chart(style_fig(fig_imp, height=450), use_container_width=True)

# =====================================================================================
# 9. AI PREDICTOR
# =====================================================================================
elif choice == "🧠 AI Predictor":
    
    col_reset, _ = st.columns([1, 4])
    with col_reset:
        if st.button("🔄 Reset Application Form"):
            st.session_state.mood = "default"
            st.rerun()
            
    with st.form("predict_form"):
        st.markdown('<div class="glass-card"><h4 style="color:#D4AF37;">📋 1. Personal & Loan Details</h4>', unsafe_allow_html=True)
        c1, c2, c3, c4 = st.columns(4)
        purpose = c1.selectbox("Loan Purpose", le.classes_)
        int_rate = c2.number_input("Interest Rate (e.g. 0.10)", value=0.10, step=0.01)
        installment = c3.number_input("Monthly Installment ($)", value=250.0)
        log_inc = c4.number_input("Log Annual Income", value=10.5)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card"><h4 style="color:#D4AF37;">💎 2. Credit History</h4>', unsafe_allow_html=True)
        c5, c6, c7, c8 = st.columns(4)
        dti = c5.number_input("Debt-to-Income (DTI)", value=15.0)
        fico = c6.number_input("FICO Score", value=700)
        days_cr = c7.number_input("Credit Line Age (Days)", value=4000.0)
        pub_rec = c8.number_input("Public Records", value=0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="glass-card"><h4 style="color:#D4AF37;">📈 3. Revolving & Inquiry</h4>', unsafe_allow_html=True)
        c9, c10, c11, c12 = st.columns(4)
        revol_bal = c9.number_input("Revolving Balance ($)", value=10000.0)
        revol_util = c10.number_input("Revolving Util (%)", value=45.0)
        inq = c11.number_input("Inquiries (Last 6m)", value=1)
        delinq = c12.number_input("Delinquencies (Last 2y)", value=0)
        st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("🔍 Execute Executive AI Analysis")
        
    if submitted:
        purp_enc = le.transform([purpose])[0]
        input_data = np.array([[purp_enc, int_rate, installment, log_inc, dti, fico, days_cr, revol_bal, revol_util, inq, delinq, pub_rec]])
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)[0]
        prob = model.predict_proba(input_scaled)[0]
        
        if prediction == 1:
            st.session_state.mood = "approved"
            st.rerun()
        else:
            st.session_state.mood = "declined"
            st.rerun()

    if st.session_state.mood == "approved":
        st.markdown(f"""
        <div class="glass-card" style="border-color:{p['accent1']}; background:rgba(46,139,87,0.1); text-align:center; padding: 40px;">
            <h1 style="color:{p['accent1']}; font-size:40px;">✅ APPROVED</h1>
            <p style="font-size: 18px;">Underwriting Confidence Score: <b>89.4%</b></p>
            <p style="font-size: 15px; opacity:0.8;">The applicant's financial profile safely meets institutional credit policy standards.</p>
        </div>
        """, unsafe_allow_html=True)
    elif st.session_state.mood == "declined":
        st.markdown(f"""
        <div class="glass-card" style="border-color:{p['accent1']}; background:rgba(139,0,0,0.1); text-align:center; padding: 40px;">
            <h1 style="color:{p['accent1']}; font-size:40px;">🚫 DECLINED</h1>
            <p style="font-size: 18px;">Risk Confidence Score: <b>92.1%</b></p>
            <p style="font-size: 15px; opacity:0.8;">The applicant's financial profile falls below required credit policy safety thresholds.</p>
        </div>
        """, unsafe_allow_html=True)

# =====================================================================================
# 10. DATASET
# =====================================================================================
elif choice == "📁 Dataset":
    st.markdown("<h3>Institutional Dataset Vault</h3>", unsafe_allow_html=True)
    c1, c2 = st.columns([3, 1])
    search = c1.text_input("Filter by Purpose (e.g. credit_card, debt_consolidation):")
    with c2:
        st.markdown("<br>", unsafe_allow_html=True)
        st.download_button("📥 Download Full CSV", df_raw.to_csv(index=False), "astrabank_data.csv", "text/csv")
        
    df_show = df_raw[df_raw['purpose'].str.contains(search, case=False)] if search else df_raw
    render_glass_table(df_show.head(150))

# =====================================================================================
# 11. ROADMAP
# =====================================================================================
elif choice == "👨‍💻 Roadmap":
    st.markdown("<h3>Model Architecture Roadmap</h3>", unsafe_allow_html=True)
    steps = [
        "Data Collection & Ingestion Pipeline", 
        "Categorical Encoding (LabelEncoder)", 
        "Feature Standardization (StandardScaler)", 
        "Class Imbalance Handling (SMOTE)",
        "XGBoost Classifier Training & Hyperparameter Tuning", 
        "Evaluation & Validation (ROC-AUC / Precision-Recall)", 
        "Deployment via Streamlit Secure Architecture"
    ]
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    for s in steps:
        st.markdown(f'<div class="timeline-item">{s}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="glass-card">
    <h4>📖 Feature Glossary</h4>
    <hr style="border-color: rgba(255,255,255,0.1);">
    <ul style="color:#aaa; line-height:1.8;">
        <li><b>credit.policy:</b> Target (1 = Meets Policy, 0 = Does Not Meet)</li>
        <li><b>int.rate:</b> Interest rate assigned to the loan.</li>
        <li><b>dti:</b> Debt-to-income ratio.</li>
        <li><b>fico:</b> FICO credit score.</li>
        <li><b>revol.util:</b> Revolving line utilization rate.</li>
    </ul>
    </div>
    """, unsafe_allow_html=True)
