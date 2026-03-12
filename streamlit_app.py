import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

# ── Optional DB import ─────────────────────────────────────────────────────────
try:
    from db import init_db, insert_prediction, fetch_latest, delete_prediction, clear_all
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_FILE    = "rf_model.pkl"
GENDER_MAP    = {"Male": 0, "Female": 1, "Other": 2, "Unknown": 3}
QUARTER_MAP   = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
GENDER_LABEL  = {0: "Male", 1: "Female", 2: "Other", 3: "Unknown"}
QUARTER_LABEL = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"}


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        return joblib.load(MODEL_FILE)
    return None

model = load_model()


# ── Feature engineering ────────────────────────────────────────────────────────
def compute_engineered(f: dict) -> dict:
    login   = f.get("Login_Frequency", 0)
    days    = f.get("Days_Since_Last_Purchase", 999)
    ltv     = f.get("Lifetime_Value", 0)
    purch   = f.get("Total_Purchases", 1) or 1
    session = f.get("Session_Duration_Avg", 0)
    pages   = f.get("Pages_Per_Session", 0)
    return {
        "engagement_recency_ratio": round(login / (days + 1), 4),
        "purchase_value_score":     round(ltv / purch, 4),
        "activity_score":           round((session * pages) / (days + 1), 4),
    }


def heuristic(f: dict) -> float:
    prob = (
        min(f.get("Days_Since_Last_Purchase", 999) / 365, 1) * 0.22 +
        max(0, 1 - min(f.get("Lifetime_Value", 0) / 2000, 1)) * 0.15 +
        max(0, 1 - min(f.get("Total_Purchases", 0) / 20, 1)) * 0.10 +
        (f.get("Cart_Abandonment_Rate", 50) / 100) * 0.09 +
        max(0, 1 - min(f.get("Average_Order_Value", 0) / 150, 1)) * 0.08 +
        (f.get("Discount_Usage_Rate", 0) / 100) * 0.07 +
        max(0, 1 - min(f.get("Membership_Years", 0) / 5, 1)) * 0.06 +
        (f.get("Returns_Rate", 0) / 100) * 0.05 +
        max(0, 1 - min(f.get("Login_Frequency", 0) / 20, 1)) * 0.05 +
        max(0, 1 - min(f.get("Email_Open_Rate", 0) / 100, 1)) * 0.05 +
        (f.get("Customer_Service_Calls", 0) / 10) * 0.04 +
        max(0, 1 - min(f.get("Mobile_App_Usage", 0) / 30, 1)) * 0.04
    )
    return max(0.01, min(0.98, prob))


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Churn Predictor", layout="wide", page_icon="◈")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.stApp { 
    background: #f8f9fc; 
}

.block-container { 
    padding: 2.5rem 3rem 4rem; 
    max-width: 1100px; 
}

/* Header */
.app-header { 
    display: flex; 
    align-items: baseline; 
    gap: 14px; 
    margin-bottom: 6px; 
}
.app-title  { 
    font-size: 26px; 
    font-weight: 600; 
    color: #1a1a2e; 
    letter-spacing: -0.4px; 
}
.app-badge  { 
    font-family: 'DM Mono', monospace; 
    font-size: 10px; 
    color: #6b7280; 
    letter-spacing: 0.15em;
    text-transform: uppercase; 
    border: 1px solid #d1d5db; 
    padding: 2px 8px; 
    border-radius: 3px; 
    background: #f3f4f6;
}
.app-sub    { 
    font-size: 13px; 
    color: #6b7280; 
    margin-bottom: 32px; 
}

/* Status */
.status-ok   { 
    font-family: 'DM Mono', monospace; 
    font-size: 11px; 
    color: #166534;
    background: rgba(22,101,52,0.08); 
    border: 1px solid rgba(22,101,52,0.20);
    border-radius: 5px; 
    padding: 7px 13px; 
    margin-bottom: 28px; 
    display: inline-block; 
}
.status-warn { 
    font-family: 'DM Mono', monospace; 
    font-size: 11px; 
    color: #92400e;
    background: rgba(146,64,14,0.08); 
    border: 1px solid rgba(146,64,14,0.20);
    border-radius: 5px; 
    padding: 7px 13px; 
    margin-bottom: 28px; 
    display: inline-block; 
}

/* Section labels */
.section-label { 
    font-family: 'DM Mono', monospace; 
    font-size: 9px; 
    font-weight: 500;
    letter-spacing: 0.18em; 
    text-transform: uppercase; 
    color: #6b7280;
    border-bottom: 1px solid #e5e7eb; 
    padding-bottom: 8px; 
    margin: 28px 0 16px; 
}

/* Inputs */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    background: #ffffff !important;
    border: 1px solid #d1d5db !important;
    border-radius: 6px !important;
}
div[data-baseweb="input"] > div:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: #9ca3af !important;
    box-shadow: 0 0 0 3px rgba(209,213,219,0.4) !important;
}
input { 
    color: #1f2937 !important; 
    font-family: 'DM Mono', monospace !important; 
    font-size: 13px !important; 
}
label[data-testid="stWidgetLabel"] p {
    font-size: 11px !important; 
    color: #6b7280 !important; 
    font-weight: 500 !important;
    letter-spacing: 0.05em; 
    text-transform: uppercase;
}

/* Number input arrows */
button[data-testid="stNumberInputStepDown"],
button[data-testid="stNumberInputStepUp"] {
    background: transparent !important;
    border: none !important;
    font-size: 14px !important;
    line-height: 1 !important;
    width: 20px !important;
}
button[data-testid="stNumberInputStepDown"] svg,
button[data-testid="stNumberInputStepUp"] svg { display: none !important; }

button[data-testid="stNumberInputStepUp"]::after   { content: '↑'; font-size:13px; color:#16a34a; }
button[data-testid="stNumberInputStepDown"]::after { content: '↓'; font-size:13px; color:#dc2626; }
button[data-testid="stNumberInputStepUp"]:hover::after   { opacity: .7; }
button[data-testid="stNumberInputStepDown"]:hover::after { opacity: .7; }

/* Submit button */
div[data-testid="stFormSubmitButton"] button {
    background: #1e293b !important; 
    color: #f1f5f9 !important;
    font-family: 'DM Sans', sans-serif !important; 
    font-size: 13px !important;
    font-weight: 600 !important; 
    letter-spacing: 0.05em;
    border: none !important; 
    border-radius: 6px !important;
    padding: 12px 0 !important; 
    width: 100%; 
    margin-top: 10px;
    transition: opacity .15s !important;
}
div[data-testid="stFormSubmitButton"] button:hover { opacity: .92 !important; }

/* Result card */
.result-card { 
    background: #ffffff; 
    border: 1px solid #e5e7eb; 
    border-radius: 10px;
    padding: 28px 32px; 
    margin: 28px 0 0; 
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.result-row  { 
    display: flex; 
    align-items: center; 
    justify-content: space-between; 
    margin-bottom: 16px; 
}
.result-pct  { 
    font-family: 'DM Mono', monospace; 
    font-size: 52px; 
    font-weight: 300;
    letter-spacing: -3px; 
    line-height: 1; 
}
.result-label{ 
    font-family: 'DM Mono', monospace; 
    font-size: 9px; 
    color: #6b7280;
    letter-spacing: 0.18em; 
    text-transform: uppercase; 
    margin-bottom: 6px; 
}
.risk-pill   { 
    font-family: 'DM Mono', monospace; 
    font-size: 11px; 
    font-weight: 500;
    letter-spacing: 0.1em; 
    text-transform: uppercase; 
    padding: 5px 14px; 
    border-radius: 20px; 
}
.risk-high   { color:#dc2626; background:rgba(220,38,38,0.08);  border:1px solid rgba(220,38,38,0.25); }
.risk-medium { color:#d97706; background:rgba(217,119,6,0.08);  border:1px solid rgba(217,119,6,0.25);  }
.risk-low    { color:#15803d; background:rgba(21,128,61,0.08);  border:1px solid rgba(21,128,61,0.25);  }

.bar-track   { 
    height: 6px; 
    background: #e5e7eb; 
    border-radius: 99px; 
    overflow: hidden; 
    margin-bottom: 16px; 
}
.result-rec  { 
    font-size: 13px; 
    color: #4b5563; 
    line-height: 1.75; 
}

/* History */
.hist-section { margin-top: 36px; }
.hist-title   { 
    font-family: 'DM Mono', monospace; 
    font-size: 9px; 
    font-weight: 500; 
    color: #6b7280;
    letter-spacing: 0.18em; 
    text-transform: uppercase; 
    margin-bottom: 16px; 
}
.hist-table   { 
    border: 1px solid #e5e7eb; 
    border-radius: 8px; 
    overflow: hidden; 
    background: white;
}
.hist-hdr, .hist-row-inner {
    display: grid;
    grid-template-columns: 44px 60px 44px 80px 72px 96px 32px;
    gap: 8px; 
    padding: 10px 14px; 
    align-items: center;
}
.hist-hdr     { 
    background: #f3f4f6; 
    border-bottom: 1px solid #e5e7eb; 
}
.hist-hd      { 
    font-family: 'DM Mono', monospace; 
    font-size: 9px; 
    color: #6b7280; 
    letter-spacing: 0.12em; 
    text-transform: uppercase; 
}
.hist-row-inner { border-bottom: 1px solid #f3f4f6; }
.hist-row-inner:last-child { border-bottom: none; }
.hist-id   { font-family: 'DM Mono', monospace; font-size: 11px; color: #4b5563; }
.hist-time { font-family: 'DM Mono', monospace; font-size: 11px; color: #9ca3af; }
.hist-cell { font-size: 12px; color: #6b7280; }
.hist-prob { font-family: 'DM Mono', monospace; font-size: 12px; color: #374151; font-weight: 500; }
.no-hist   { 
    font-family: 'DM Mono', monospace; 
    font-size: 12px; 
    color: #9ca3af;
    padding: 20px 14px; 
}

/* Clear button override */
button[kind="secondary"] { background: transparent !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #f1f5f9; }
::-webkit-scrollbar-thumb { background: #9ca3af; border-radius: 3px; }

#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
  <span class="app-title">Churn Predictor</span>
  <span class="app-badge">RF Model</span>
</div>
<div class="app-sub">Enter customer details to estimate churn probability.</div>
""", unsafe_allow_html=True)

if model is None:
    st.markdown('<div class="status-warn">⚠ &nbsp;rf_model.pkl not found — heuristic fallback active</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-ok">✓ &nbsp;Model loaded</div>', unsafe_allow_html=True)


# ── Form ──────────────────────────────────────────────────────────────────────
with st.form("churn_form"):

    st.markdown('<div class="section-label">Demographics</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    age              = c1.number_input("Age", 18, 100, 35)
    membership_years = c2.number_input("Membership Years", 0.0, 20.0, 2.0, step=0.1)
    gender           = c3.selectbox("Gender", ["Unknown", "Male", "Female", "Other"])
    signup_quarter   = c4.selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])

    st.markdown('<div class="section-label">Purchase Behaviour</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    days_since      = c1.number_input("Days Since Last Purchase", 0, 1000, 30)
    lifetime_value  = c2.number_input("Lifetime Value ($)", 0.0, 10000.0, 1200.0)
    avg_order       = c3.number_input("Avg Order Value ($)", 0.0, 1000.0, 85.0)
    total_purchases = c4.number_input("Total Purchases", 0, 200, 14)

    c1, c2, c3, c4 = st.columns(4)
    discount_rate   = c1.number_input("Discount Usage (%)", 0.0, 100.0, 40.0)
    returns_rate    = c2.number_input("Returns Rate (%)", 0.0, 100.0, 10.0)
    cart_abandon    = c3.number_input("Cart Abandonment (%)", 0.0, 100.0, 50.0)
    wishlist_items  = c4.number_input("Wishlist Items", 0, 100, 5)

    st.markdown('<div class="section-label">Engagement</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    login_freq        = c1.number_input("Login Frequency / mo", 0, 100, 8)
    session_duration  = c2.number_input("Session Duration (min)", 0.0, 120.0, 12.0)
    pages_per_session = c3.number_input("Pages Per Session", 0.0, 50.0, 5.0)
    email_open_rate   = c4.number_input("Email Open Rate (%)", 0.0, 100.0, 30.0)

    c1, c2, c3 = st.columns(3)
    mobile_app_usage = c1.number_input("Mobile App Usage (days/mo)", 0, 31, 15)
    social_media     = c2.number_input("Social Media Engagement", 0.0, 100.0, 3.0)
    reviews_written  = c3.number_input("Product Reviews Written", 0, 100, 2)

    st.markdown('<div class="section-label">Financial & Support</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    credit_balance    = c1.number_input("Credit Balance ($)", 0.0, 5000.0, 200.0)
    service_calls     = c2.number_input("Customer Service Calls", 0, 50, 1)
    payment_diversity = c3.number_input("Payment Method Diversity", 0, 10, 2)

    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Run Prediction", use_container_width=True)


# ── Prediction result ─────────────────────────────────────────────────────────
if submitted:
    raw = {
        "Age":                           age,
        "Gender":                        GENDER_MAP[gender],
        "Signup_Quarter":                QUARTER_MAP[signup_quarter],
        "Membership_Years":              membership_years,
        "Days_Since_Last_Purchase":      days_since,
        "Lifetime_Value":                lifetime_value,
        "Average_Order_Value":           avg_order,
        "Total_Purchases":               total_purchases,
        "Discount_Usage_Rate":           discount_rate,
        "Returns_Rate":                  returns_rate,
        "Cart_Abandonment_Rate":         cart_abandon,
        "Wishlist_Items":                wishlist_items,
        "Login_Frequency":               login_freq,
        "Session_Duration_Avg":          session_duration,
        "Pages_Per_Session":             pages_per_session,
        "Email_Open_Rate":               email_open_rate,
        "Mobile_App_Usage":              mobile_app_usage,
        "Social_Media_Engagement_Score": social_media,
        "Product_Reviews_Written":       reviews_written,
        "Credit_Balance":                credit_balance,
        "Customer_Service_Calls":        service_calls,
        "Payment_Method_Diversity":      payment_diversity,
    }

    eng = compute_engineered(raw)

    if model is not None:
        row = {**raw, **eng}
        df  = pd.DataFrame([row])
        for col in model.feature_names_in_:
            if col not in df.columns:
                df[col] = 0
        df   = df[model.feature_names_in_]
        prob = float(model.predict_proba(df)[0][1])
    else:
        prob = heuristic(raw)

    prediction = 1 if prob > 0.50 else 0
    risk_level = "High" if prob > 0.65 else "Medium" if prob > 0.35 else "Low"
    pct        = round(prob * 100, 1)

    bar_color  = {"High": "#dc2626", "Medium": "#d97706", "Low": "#15803d"}[risk_level]
    risk_class = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}[risk_level]
    rec_text   = {
        "High":   "Strong churn signals detected. Consider a targeted retention offer or personal outreach within 48 hours.",
        "Medium": "Moderate churn risk. A re-engagement email or check-in campaign could help.",
        "Low":    "Healthy engagement signals. Maintain current strategy and consider upsell opportunities.",
    }[risk_level]

    st.markdown(f"""
    <div class="result-card">
      <div class="result-row">
        <div>
          <div class="result-label">Churn Probability</div>
          <div class="result-pct" style="color:{bar_color}">{pct}%</div>
        </div>
        <span class="risk-pill {risk_class}">{risk_level}&nbsp;Risk</span>
      </div>
      <div class="bar-track">
        <div style="height:100%;width:{pct}%;background:{bar_color};border-radius:99px"></div>
      </div>
      <div class="result-rec">{rec_text}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("Engineered features", expanded=False):
        st.json(eng)

    if DB_AVAILABLE:
        try:
            customer_record = {**raw, "Gender": gender, "Signup_Quarter": signup_quarter}
            insert_prediction(
                created_at  = datetime.now().isoformat(sep=" ", timespec="seconds"),
                customer    = customer_record,
                engineered  = eng,
                prediction  = prediction,
                probability = prob,
                risk_level  = risk_level,
            )
        except Exception as e:
            st.warning(f"Could not save to DB: {e}")


# ── History ───────────────────────────────────────────────────────────────────
if DB_AVAILABLE:
    try:
        rows = fetch_latest(20)

        st.markdown('<div class="hist-section">', unsafe_allow_html=True)
        h1, h2 = st.columns([5, 1])
        h1.markdown('<div class="hist-title" style="margin-top:4px">Recent Predictions</div>', unsafe_allow_html=True)
        if h2.button("Clear all", use_container_width=True):
            clear_all()
            st.rerun()

        if not rows:
            st.markdown('<div class="no-hist">— no predictions yet —</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="hist-table">
              <div class="hist-hdr">
                <span class="hist-hd">#</span>
                <span class="hist-hd">Time</span>
                <span class="hist-hd">Age</span>
                <span class="hist-hd">Gender</span>
                <span class="hist-hd">Prob</span>
                <span class="hist-hd">Risk</span>
                <span></span>
              </div>
            """, unsafe_allow_html=True)

            for r in rows:
                rid, created_at, age_r, gender_r, _pred, prob_r, risk_r = r
                risk_cls = {"High": "risk-high", "Medium": "risk-medium", "Low": "risk-low"}.get(risk_r, "")
                time_str = str(created_at)[11:16]
                pct_str  = f"{round(prob_r * 100, 1)}%"

                c_id, c_time, c_age, c_gen, c_prob, c_risk, c_del = st.columns([0.9, 1.3, 0.9, 1.7, 1.5, 2, 0.7])
                c_id.markdown(f"<span class='hist-id'>#{rid}</span>", unsafe_allow_html=True)
                c_time.markdown(f"<span class='hist-time'>{time_str}</span>", unsafe_allow_html=True)
                c_age.markdown(f"<span class='hist-cell'>{age_r}</span>", unsafe_allow_html=True)
                c_gen.markdown(f"<span class='hist-cell'>{gender_r}</span>", unsafe_allow_html=True)
                c_prob.markdown(f"<span class='hist-prob'>{pct_str}</span>", unsafe_allow_html=True)
                c_risk.markdown(f"<span class='risk-pill {risk_cls}' style='font-size:10px;padding:3px 10px'>{risk_r}</span>", unsafe_allow_html=True)
                if c_del.button("✕", key=f"del_{rid}"):
                    delete_prediction(rid)
                    st.rerun()

            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

    except Exception as e:
        st.warning(f"Could not load history: {e}")