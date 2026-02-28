# ABOUTME: Dark fintech dashboard UI for the Intelligent Credit Risk Scorer.
# ABOUTME: Features navbar, tabbed form, KPI grid, donut gauge, risk indicators, and recommendation panel.

import os
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.predict import make_prediction

MODEL_PATH = os.path.join("models", "credit_risk_model_v2.pkl")

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CreditRisk Dashboard",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;500;600;700;800&family=DM+Mono:wght@400;500&display=swap');

* { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif !important;
}

/* ── Global background ── */
[data-testid="stAppViewContainer"] { background-color: #06060f !important; }
[data-testid="stHeader"] { background-color: #06060f !important; display: none; }
.main .block-container {
    padding: 1.5rem 2.5rem 3rem 2.5rem !important;
    max-width: 100% !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #06060f; }
::-webkit-scrollbar-thumb { background: #1e1e38; border-radius: 2px; }

/* ─────────────────── NAVBAR ─────────────────── */
.navbar {
    display: flex; align-items: center; justify-content: space-between;
    background: #0a0a18; border: 1px solid #181830;
    border-radius: 16px; padding: 14px 24px; margin-bottom: 28px;
}
.navbar-brand { display: flex; align-items: center; gap: 16px; }
.logo-box {
    width: 42px; height: 42px;
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
    border-radius: 11px;
    display: flex; align-items: center; justify-content: center;
    font-family: 'DM Mono', monospace; font-weight: 500;
    font-size: 13px; color: white; letter-spacing: -0.5px;
    box-shadow: 0 4px 16px rgba(37,99,235,0.35);
}
.brand-text .name { color: white; font-size: 17px; font-weight: 700; letter-spacing: -0.5px; }
.brand-text .name em { color: #ef4444; font-style: normal; }
.brand-text .sub { color: #2e2e50; font-size: 11px; font-weight: 400; margin-top: 1px; }
.navbar-right { display: flex; align-items: center; gap: 10px; }
.badge-online {
    display: flex; align-items: center; gap: 7px;
    background: rgba(34,197,94,0.08); border: 1px solid rgba(34,197,94,0.25);
    border-radius: 20px; padding: 6px 14px;
    color: #22c55e; font-size: 11px; font-weight: 600;
    font-family: 'DM Mono', monospace; letter-spacing: 0.5px;
}
.badge-dot {
    width: 6px; height: 6px; background: #22c55e; border-radius: 50%;
    box-shadow: 0 0 8px #22c55e;
    animation: glow-pulse 2.4s ease-in-out infinite;
}
@keyframes glow-pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 8px #22c55e; }
    50% { opacity: 0.5; box-shadow: 0 0 3px #22c55e; }
}
.badge-version {
    background: #0d0d20; border: 1px solid #1e1e38;
    border-radius: 7px; padding: 6px 11px;
    color: #2e2e50; font-size: 11px; font-family: 'DM Mono', monospace;
}

/* ─────────────────── SECTION HEADERS ─────────────────── */
.sec-head {
    display: flex; align-items: center; gap: 13px; margin-bottom: 20px;
}
.sec-ico {
    width: 36px; height: 36px; border-radius: 10px;
    background: #0d0d20; border: 1px solid #1e1e38;
    display: flex; align-items: center; justify-content: center; font-size: 16px;
}
.sec-title { color: white; font-size: 15px; font-weight: 700; }
.sec-sub { color: #2e2e50; font-size: 11px; font-weight: 400; margin-top: 2px; }

/* ─────────────────── KPI CARDS ─────────────────── */
.kpi-row {
    display: grid; grid-template-columns: repeat(4, 1fr);
    gap: 10px; margin-bottom: 14px;
}
.kpi {
    background: #0a0a18; border: 1px solid #181830;
    border-radius: 13px; padding: 15px 16px;
    position: relative; overflow: hidden;
}
.kpi::after {
    content: ''; position: absolute; top: 0; left: 0; right: 0;
    height: 2px; border-radius: 2px 2px 0 0;
}
.kpi.c-blue::after  { background: #2563eb; }
.kpi.c-green::after { background: #22c55e; }
.kpi.c-amber::after { background: #f59e0b; }
.kpi.c-red::after   { background: #ef4444; }
.kpi.c-violet::after { background: #8b5cf6; }
.kpi-lbl {
    color: #2e2e50; font-size: 9px; font-weight: 600;
    letter-spacing: 1.8px; text-transform: uppercase;
    font-family: 'DM Mono', monospace; margin-bottom: 8px;
}
.kpi-val {
    color: white; font-size: 21px; font-weight: 700;
    letter-spacing: -0.5px; margin-bottom: 4px;
}
.kpi-tag { font-size: 10px; font-weight: 600; }
.kpi-bg-icon {
    position: absolute; bottom: 10px; right: 12px;
    font-size: 22px; opacity: 0.12;
}

/* ─────────────────── PANEL ─────────────────── */
.panel {
    background: #0a0a18; border: 1px solid #181830;
    border-radius: 14px; padding: 18px; margin-bottom: 12px;
}
.panel-hdr {
    display: flex; justify-content: space-between;
    align-items: flex-start; margin-bottom: 14px;
}
.p-title { color: white; font-size: 13px; font-weight: 700; }
.p-sub { color: #2e2e50; font-size: 10px; font-weight: 400; margin-top: 2px; }

/* ─────────────────── PROFILE GRID ─────────────────── */
.profile-g {
    display: grid; grid-template-columns: 1fr 1fr; gap: 9px;
}
.p-cell {
    background: #0d0d20; border-radius: 10px;
    padding: 12px 10px; text-align: center;
}
.p-cell-v {
    color: white; font-size: 19px; font-weight: 700;
    font-family: 'DM Mono', monospace; margin-bottom: 3px;
}
.p-cell-k {
    color: #2e2e50; font-size: 8px; font-weight: 600;
    letter-spacing: 1.5px; text-transform: uppercase;
}

/* ─────────────────── RISK INDICATORS ─────────────────── */
.ri {
    display: flex; align-items: center; justify-content: space-between;
    padding: 11px 0; border-bottom: 1px solid #0f0f1e;
}
.ri:last-child { border-bottom: none; }
.ri-l { display: flex; align-items: center; gap: 10px; }
.ri-ico {
    width: 28px; height: 28px; border-radius: 7px;
    display: flex; align-items: center; justify-content: center; font-size: 12px;
}
.ri-ico.ok  { background: rgba(34,197,94,0.12); }
.ri-ico.bad { background: rgba(239,68,68,0.12); }
.ri-name { color: #c0c0e0; font-size: 12px; font-weight: 600; }
.ri-hint { color: #2e2e50; font-size: 10px; margin-top: 1px; }
.ri-score { font-family: 'DM Mono', monospace; font-size: 12px; font-weight: 500; }

/* ─────────────────── PROBABILITY SCALE ─────────────────── */
.prob-hdr { display: flex; justify-content: space-between; margin-bottom: 12px; }
.prob-label { color: white; font-size: 13px; font-weight: 700; }
.prob-num { font-family: 'DM Mono', monospace; font-size: 13px; font-weight: 500; }
.prob-track { height: 8px; border-radius: 4px; background: #0d0d20; overflow: hidden; }
.prob-fill {
    height: 100%; border-radius: 4px;
    background: linear-gradient(90deg, #22c55e 0%, #f59e0b 50%, #ef4444 100%);
    background-size: 300% 100%;
    background-position-x: calc(100% - var(--pct, 50%) * 2);
}
.prob-ticks {
    display: flex; justify-content: space-between;
    margin-top: 6px; color: #1e1e38;
    font-size: 9px; font-family: 'DM Mono', monospace;
}

/* ─────────────────── RECOMMENDATION ─────────────────── */
.rec { border-radius: 14px; padding: 18px; }
.rec.decline { background: rgba(239,68,68,0.05); border: 1px solid rgba(239,68,68,0.2); }
.rec.approve { background: rgba(34,197,94,0.05); border: 1px solid rgba(34,197,94,0.2); }
.rec-h { display: flex; align-items: center; gap: 12px; margin-bottom: 10px; }
.rec-ico { font-size: 22px; }
.rec-t { color: white; font-size: 14px; font-weight: 700; }
.rec-s { color: #2e2e50; font-size: 10px; margin-top: 2px; }
.rec-body { color: #6060a0; font-size: 12px; line-height: 1.65; }
.rec-body strong { color: #9090c0; }

/* ─────────────────── WAITING ─────────────────── */
.waiting {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; padding: 90px 40px; text-align: center;
}
.w-ico { font-size: 52px; opacity: 0.18; margin-bottom: 20px; }
.w-title { color: #2e2e50; font-size: 15px; font-weight: 700; margin-bottom: 8px; }
.w-hint { color: #1e1e38; font-size: 12px; line-height: 1.7; }
.w-hint strong { color: #2e2e50; }

/* ─────────────────── TABS ─────────────────── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #181830 !important;
    gap: 0 !important; padding-bottom: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #2e2e50 !important;
    border-bottom: 2px solid transparent !important;
    border-radius: 0 !important;
    padding: 9px 18px !important;
    font-size: 12px !important; font-weight: 600 !important;
    letter-spacing: 0.3px !important;
    font-family: 'Syne', sans-serif !important;
    margin-right: 2px !important;
}
.stTabs [aria-selected="true"] {
    color: white !important;
    border-bottom-color: #2563eb !important;
}
.stTabs [data-baseweb="tab-panel"] { padding: 18px 0 0 0 !important; }
.stTabs [data-baseweb="tab-highlight"] { display: none !important; }

/* ─────────────────── FORM WIDGETS ─────────────────── */
[data-testid="stWidgetLabel"] p {
    color: #3a3a60 !important; font-size: 10px !important;
    font-weight: 600 !important; text-transform: uppercase !important;
    letter-spacing: 1px !important; font-family: 'DM Mono', monospace !important;
}
[data-testid="stNumberInput"] input {
    background: #0d0d20 !important; border: 1px solid #181830 !important;
    color: white !important; border-radius: 9px !important;
    font-family: 'DM Mono', monospace !important; font-size: 14px !important;
}
[data-testid="stNumberInput"] button {
    background: #0d0d20 !important; border-color: #181830 !important;
    color: #3a3a60 !important;
}
[data-testid="stSelectbox"] > div > div {
    background: #0d0d20 !important; border: 1px solid #181830 !important;
    color: white !important; border-radius: 9px !important;
}
[data-testid="stSlider"] { padding-top: 6px !important; }
[data-baseweb="slider"] [role="slider"] {
    background: #2563eb !important;
    box-shadow: 0 0 12px rgba(37,99,235,0.6) !important;
}
[data-baseweb="slider"] div[data-testid] {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
}

/* ─────────────────── BUTTON ─────────────────── */
[data-testid="baseButton-primary"] {
    background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%) !important;
    border: none !important; color: white !important;
    font-weight: 700 !important; font-size: 11px !important;
    letter-spacing: 2px !important; border-radius: 11px !important;
    padding: 15px 0 !important; width: 100% !important;
    font-family: 'DM Mono', monospace !important; text-transform: uppercase !important;
    box-shadow: 0 4px 20px rgba(37,99,235,0.25) !important;
    transition: all 0.2s ease !important;
}
[data-testid="baseButton-primary"]:hover {
    box-shadow: 0 8px 32px rgba(37,99,235,0.45) !important;
    transform: translateY(-1px) !important;
}

/* ─────────────────── HIDE CHROME ─────────────────── */
[data-testid="stToolbar"], footer, #MainMenu { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
</style>
""",
    unsafe_allow_html=True,
)


# ── Model ──────────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


try:
    model = load_model()
except FileNotFoundError:
    st.error(f"Model not found at '{MODEL_PATH}'. Run `python3 run_training.py` first.")
    st.stop()


# ── Helpers ────────────────────────────────────────────────────────────────────
def create_donut_gauge(default_prob_pct: float) -> go.Figure:
    pct = default_prob_pct
    rem = 100.0 - pct

    if pct >= 70:
        color, label = "#ef4444", "HIGH RISK"
    elif pct >= 40:
        color, label = "#f59e0b", "MEDIUM RISK"
    else:
        color, label = "#22c55e", "LOW RISK"

    fig = go.Figure(
        go.Pie(
            values=[pct, rem],
            hole=0.62,
            marker=dict(
                colors=[color, "#111125"],
                line=dict(width=0),
            ),
            textinfo="none",
            hoverinfo="none",
            showlegend=False,
            sort=False,
            direction="clockwise",
            rotation=90,
        )
    )

    fig.add_annotation(
        x=0.5, y=0.60,
        text=f"<b>{pct:.1f}</b>",
        font=dict(size=40, color="white", family="Syne"),
        showarrow=False,
    )
    fig.add_annotation(
        x=0.5, y=0.42,
        text="percent",
        font=dict(size=12, color="#2e2e50", family="Syne"),
        showarrow=False,
    )
    fig.add_annotation(
        x=0.5, y=0.19,
        text=f"● {label}",
        font=dict(size=11, color=color, family="DM Mono"),
        showarrow=False,
    )

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=240,
        margin=dict(l=0, r=0, t=10, b=0),
        showlegend=False,
    )
    return fig


def savings_signal(val: str):
    good = val in {"quite rich", "rich", "moderate"}
    desc = "Strong savings buffer" if good else "Insufficient savings"
    return good, desc


def checking_signal(val: str):
    good = val in {"rich", "moderate"}
    desc = "Adequate liquidity" if good else "Low account balance"
    return good, desc


def duration_signal(months: int):
    if months <= 12:
        return True, "Short-term loan", f"{months} mo"
    elif months <= 24:
        return True, "Medium-term loan", f"{months} mo"
    else:
        return False, "Long-term exposure", f"{months} mo"


def ri_html(label: str, value: str, desc: str, good: bool) -> str:
    ico = "✓" if good else "✕"
    cls = "ok" if good else "bad"
    color = "#22c55e" if good else "#ef4444"
    return f"""
    <div class="ri">
        <div class="ri-l">
            <div class="ri-ico {cls}">{ico}</div>
            <div>
                <div class="ri-name">{label}</div>
                <div class="ri-hint">{desc}</div>
            </div>
        </div>
        <div class="ri-score" style="color:{color}">{value}</div>
    </div>"""


# ── Navbar ─────────────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="navbar">
    <div class="navbar-brand">
        <div class="logo-box">CR</div>
        <div class="brand-text">
            <div class="name">Credit<em>Risk</em> Dashboard</div>
            <div class="sub">AI-Powered Risk Intelligence Platform</div>
        </div>
    </div>
    <div class="navbar-right">
        <div class="badge-online"><div class="badge-dot"></div> System Online</div>
        <div class="badge-version">v2.0</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ── Layout ─────────────────────────────────────────────────────────────────────
col_left, col_right = st.columns([4, 6], gap="large")

# ════════════════════════════════════════
# LEFT COLUMN — Input Form
# ════════════════════════════════════════
with col_left:
    st.markdown(
        """
    <div class="sec-head">
        <div class="sec-ico">📋</div>
        <div>
            <div class="sec-title">Client Assessment</div>
            <div class="sec-sub">Enter applicant details for risk evaluation</div>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    tab_demo, tab_credit, tab_fin = st.tabs(
        ["👤  Demographics", "💳  Credit Profile", "📊  Financials"]
    )

    with tab_demo:
        age = st.slider("Age", min_value=18, max_value=80, value=32)
        sex = st.selectbox("Gender", options=["male", "female"])
        job = st.selectbox(
            "Job Skill Level",
            options=[0, 1, 2, 3],
            format_func=lambda x: {
                0: "0 – Unskilled / Non-resident",
                1: "1 – Unskilled / Resident",
                2: "2 – Skilled",
                3: "3 – Highly Skilled",
            }[x],
        )
        housing = st.selectbox("Housing Status", options=["own", "free", "rent"])

    with tab_credit:
        saving_accounts = st.selectbox(
            "Saving Accounts",
            options=["unknown", "little", "moderate", "quite rich", "rich"],
        )
        checking_account = st.selectbox(
            "Checking Account",
            options=["unknown", "little", "moderate", "rich"],
        )

    with tab_fin:
        credit_amount = st.number_input(
            "Credit Amount (DM)", min_value=100, max_value=20000, value=2500, step=100
        )
        duration = st.number_input(
            "Loan Duration (months)", min_value=1, max_value=72, value=18
        )
        purpose = st.selectbox(
            "Purpose of Loan",
            options=[
                "radio/TV",
                "education",
                "furniture/equipment",
                "car",
                "business",
                "domestic appliances",
                "repairs",
                "vacation/others",
            ],
        )

    st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)
    analyze = st.button(
        "🔍  ANALYZE RISK PROFILE", type="primary", use_container_width=True
    )

    if analyze:
        input_df = pd.DataFrame(
            {
                "Age": [age],
                "Sex": [sex],
                "Job": [job],
                "Housing": [housing],
                "Saving accounts": [saving_accounts],
                "Checking account": [checking_account],
                "Credit amount": [credit_amount],
                "Duration": [duration],
                "Purpose": [purpose],
            }
        )
        result = make_prediction(model, input_df)
        st.session_state["result"] = {
            **result,
            "age": age,
            "sex": sex,
            "job": job,
            "housing": housing,
            "saving_accounts": saving_accounts,
            "checking_account": checking_account,
            "credit_amount": credit_amount,
            "duration": duration,
            "purpose": purpose,
        }

# ════════════════════════════════════════
# RIGHT COLUMN — Risk Assessment Panel
# ════════════════════════════════════════
with col_right:
    st.markdown(
        """
    <div class="sec-head">
        <div class="sec-ico">📊</div>
        <div>
            <div class="sec-title">Risk Assessment</div>
            <div class="sec-sub">AI-generated analysis &amp; insights</div>
        </div>
    </div>""",
        unsafe_allow_html=True,
    )

    if "result" not in st.session_state:
        # ── Waiting State ──
        st.markdown(
            """
        <div class="waiting">
            <div class="w-ico">💳</div>
            <div class="w-title">No Assessment Yet</div>
            <div class="w-hint">
                Complete the client profile on the left<br>
                and click <strong>Analyze Risk Profile</strong><br>
                to generate a full risk assessment.
            </div>
        </div>""",
            unsafe_allow_html=True,
        )
    else:
        d = st.session_state["result"]
        is_bad = d["prediction"] == 1
        default_prob = d["default_probability"]
        risk_color = "#ef4444" if is_bad else "#22c55e"

        # ── KPI Cards ──────────────────────────────────────────────────────────
        amount_k = f"DM {d['credit_amount']:,}"
        if d["credit_amount"] > 10000:
            amt_cls, amt_tag, amt_color = "c-red", "High Value", "#ef4444"
        elif d["credit_amount"] > 5000:
            amt_cls, amt_tag, amt_color = "c-amber", "Moderate", "#f59e0b"
        else:
            amt_cls, amt_tag, amt_color = "c-green", "Standard", "#22c55e"

        job_labels_short = {0: "Unskilled", 1: "Resident", 2: "Skilled", 3: "Expert"}
        job_color = "#2563eb" if d["job"] >= 2 else "#f59e0b"

        sav_color_map = {
            "unknown": "#ef4444", "little": "#f59e0b",
            "moderate": "#2563eb", "quite rich": "#22c55e", "rich": "#22c55e",
        }
        sav_color = sav_color_map.get(d["saving_accounts"], "#6060a0")

        prob_color = "#ef4444" if default_prob >= 70 else ("#f59e0b" if default_prob >= 40 else "#22c55e")

        st.markdown(
            f"""
        <div class="kpi-row">
            <div class="kpi c-blue">
                <div class="kpi-lbl">Credit Amount</div>
                <div class="kpi-val" style="font-size:17px">{amount_k}</div>
                <div class="kpi-tag" style="color:{amt_color}">{amt_tag}</div>
                <div class="kpi-bg-icon">💰</div>
            </div>
            <div class="kpi {amt_cls}">
                <div class="kpi-lbl">Duration</div>
                <div class="kpi-val">{d['duration']}</div>
                <div class="kpi-tag" style="color:#2563eb">months</div>
                <div class="kpi-bg-icon">⏱</div>
            </div>
            <div class="kpi c-violet">
                <div class="kpi-lbl">Job Level</div>
                <div class="kpi-val">{d['job']}<span style="font-size:13px;color:#3a3a60"> / 3</span></div>
                <div class="kpi-tag" style="color:{job_color}">{job_labels_short[d['job']]}</div>
                <div class="kpi-bg-icon">💼</div>
            </div>
            <div class="kpi c-green">
                <div class="kpi-lbl">Savings</div>
                <div class="kpi-val" style="font-size:15px">{d['saving_accounts'].title()}</div>
                <div class="kpi-tag" style="color:{sav_color}">Account Level</div>
                <div class="kpi-bg-icon">🏦</div>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        # ── Gauge + Profile ────────────────────────────────────────────────────
        g_col, p_col = st.columns([1, 1], gap="small")

        with g_col:
            st.markdown(
                """<div class="panel" style="padding-bottom:6px">
                <div class="panel-hdr" style="margin-bottom:4px">
                    <div>
                        <div class="p-title">Risk Score</div>
                        <div class="p-sub">Default Probability</div>
                    </div>
                </div>""",
                unsafe_allow_html=True,
            )
            st.plotly_chart(
                create_donut_gauge(default_prob),
                use_container_width=True,
                config={"displayModeBar": False},
            )
            st.markdown("</div>", unsafe_allow_html=True)

        with p_col:
            housing_icon = {"own": "🏠", "free": "🏢", "rent": "🏗️"}.get(d["housing"], "🏠")
            chk_color = "#22c55e" if d["checking_account"] in {"rich", "moderate"} else "#ef4444"

            st.markdown(
                f"""
            <div class="panel" style="height:100%">
                <div class="panel-hdr">
                    <div>
                        <div class="p-title">Client Profile</div>
                        <div class="p-sub">Anonymous Client</div>
                    </div>
                </div>
                <div class="profile-g">
                    <div class="p-cell">
                        <div class="p-cell-v">{d['age']}</div>
                        <div class="p-cell-k">Age</div>
                    </div>
                    <div class="p-cell">
                        <div class="p-cell-v" style="color:#2563eb">{d['job']}</div>
                        <div class="p-cell-k">Job Lvl</div>
                    </div>
                    <div class="p-cell">
                        <div class="p-cell-v" style="font-size:15px">{housing_icon} {d['housing'].title()}</div>
                        <div class="p-cell-k">Housing</div>
                    </div>
                    <div class="p-cell">
                        <div class="p-cell-v" style="color:{chk_color};font-size:13px">{d['checking_account'].title()}</div>
                        <div class="p-cell-k">Checking</div>
                    </div>
                </div>
            </div>""",
                unsafe_allow_html=True,
            )

        # ── Risk Analysis Indicators ───────────────────────────────────────────
        sav_good, sav_desc = savings_signal(d["saving_accounts"])
        chk_good, chk_desc = checking_signal(d["checking_account"])
        dur_good, dur_desc, dur_val = duration_signal(d["duration"])

        st.markdown(
            f"""
        <div class="panel">
            <div class="panel-hdr">
                <div>
                    <div class="p-title">Risk Analysis</div>
                    <div class="p-sub">Key Indicators</div>
                </div>
            </div>
            {ri_html("Savings Account", d['saving_accounts'].title(), sav_desc, sav_good)}
            {ri_html("Checking Account", d['checking_account'].title(), chk_desc, chk_good)}
            {ri_html("Loan Duration", dur_val, dur_desc, dur_good)}
        </div>""",
            unsafe_allow_html=True,
        )

        # ── Default Probability Scale ──────────────────────────────────────────
        st.markdown(
            f"""
        <div class="panel">
            <div class="prob-hdr">
                <div class="prob-label">Default Probability Scale</div>
                <div class="prob-num" style="color:{prob_color}">{default_prob}%</div>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{default_prob}%"></div>
            </div>
            <div class="prob-ticks">
                <span>0%</span><span>25%</span><span>50%</span><span>75%</span><span>100% chance</span>
            </div>
        </div>""",
            unsafe_allow_html=True,
        )

        # ── Recommendation ─────────────────────────────────────────────────────
        if is_bad:
            rec_cls = "decline"
            rec_ico = "✕"
            rec_title = "Decline Recommended"
            rec_sub = "AI Engine Assessment"
            rec_body = (
                f"High probability of default detected. Significant repayment risk with "
                f"a <strong>{default_prob}%</strong> default probability score. "
                f"<strong>Decline recommended.</strong>"
            )
        else:
            rec_cls = "approve"
            rec_ico = "✓"
            rec_title = "Approval Recommended"
            rec_sub = "AI Engine Assessment"
            rec_body = (
                f"Low probability of default. Strong repayment capacity with a "
                f"<strong>{default_prob}%</strong> default probability score. "
                f"<strong>Approval recommended.</strong>"
            )

        st.markdown(
            f"""
        <div class="rec {rec_cls}">
            <div class="rec-h">
                <div class="rec-ico">{rec_ico}</div>
                <div>
                    <div class="rec-t">{rec_title}</div>
                    <div class="rec-s">{rec_sub}</div>
                </div>
            </div>
            <div class="rec-body">{rec_body}</div>
        </div>""",
            unsafe_allow_html=True,
        )

        # ── Footer ─────────────────────────────────────────────────────────────
        st.markdown(
            f"<div style='text-align:center;color:#1e1e38;font-size:10px;"
            f"font-family:DM Mono,monospace;margin-top:16px'>"
            f"Analysis by CreditRisk AI v2.0 · German Credit Dataset · "
            f"{d['purpose'].title()} Loan Assessment</div>",
            unsafe_allow_html=True,
        )
