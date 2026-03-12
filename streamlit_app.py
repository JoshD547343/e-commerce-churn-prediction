import os
import joblib
import pandas as pd
import streamlit as st
from datetime import datetime

# ── Optional DB import (graceful fallback if db.py not present) ────────────────
try:
    from db import init_db, insert_prediction, fetch_latest, delete_prediction, clear_all
    DB_AVAILABLE = True
except ImportError:
    DB_AVAILABLE = False

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_FILE = "rf_model.pkl"

GENDER_MAP    = {"Male": 0, "Female": 1, "Other": 2, "Unknown": 3}
QUARTER_MAP   = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
QUARTER_LABEL = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"}
GENDER_LABEL  = {0: "Male", 1: "Female", 2: "Other", 3: "Unknown"}


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if os.path.exists(MODEL_FILE):
        m = joblib.load(MODEL_FILE)
        return m
    return None

model = load_model()


# ── Feature engineering (mirrors app.py exactly) ──────────────────────────────
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


# ── Heuristic fallback (mirrors app.py exactly) ───────────────────────────────
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
st.set_page_config(page_title="Churn Predictor", layout="wide")
st.title("Churn Predictor")
st.caption("Enter customer details to estimate churn risk.")

if model is None:
    st.warning("⚠ `rf_model.pkl` not found — using heuristic fallback for predictions.")
else:
    st.success("✓ Model loaded successfully.")

# ── Input form ─────────────────────────────────────────────────────────────────
with st.form("churn_form"):

    st.subheader("Demographics")
    col1, col2, col3, col4 = st.columns(4)
    age              = col1.number_input("Age", 18, 100, 35)
    membership_years = col2.number_input("Membership Years", 0.0, 20.0, 2.0, step=0.1)
    gender           = col3.selectbox("Gender", ["Unknown", "Male", "Female", "Other"])
    signup_quarter   = col4.selectbox("Signup Quarter", ["Q1", "Q2", "Q3", "Q4"])

    st.subheader("Purchase Behaviour")
    col1, col2, col3, col4 = st.columns(4)
    days_since       = col1.number_input("Days Since Last Purchase", 0, 1000, 30)
    lifetime_value   = col2.number_input("Lifetime Value ($)", 0.0, 10000.0, 1200.0)
    avg_order        = col3.number_input("Avg. Order Value ($)", 0.0, 1000.0, 85.0)
    total_purchases  = col4.number_input("Total Purchases", 0, 200, 14)

    col1, col2, col3, col4 = st.columns(4)
    discount_rate    = col1.number_input("Discount Usage (%)", 0.0, 100.0, 40.0)
    returns_rate     = col2.number_input("Returns Rate (%)", 0.0, 100.0, 10.0)
    cart_abandon     = col3.number_input("Cart Abandonment (%)", 0.0, 100.0, 50.0)
    wishlist_items   = col4.number_input("Wishlist Items", 0, 100, 5)

    st.subheader("Engagement")
    col1, col2, col3, col4 = st.columns(4)
    login_freq       = col1.number_input("Login Frequency / mo", 0, 100, 8)
    session_duration = col2.number_input("Session Duration Avg (min)", 0.0, 120.0, 12.0)
    pages_per_session= col3.number_input("Pages Per Session", 0.0, 50.0, 5.0)
    email_open_rate  = col4.number_input("Email Open Rate (%)", 0.0, 100.0, 30.0)

    col1, col2, col3 = st.columns(3)
    mobile_app_usage = col1.number_input("Mobile App Usage (days/mo)", 0, 31, 15)
    social_media     = col2.number_input("Social Media Engagement", 0.0, 100.0, 3.0)
    reviews_written  = col3.number_input("Product Reviews Written", 0, 100, 2)

    st.subheader("Financial & Support")
    col1, col2, col3 = st.columns(3)
    credit_balance    = col1.number_input("Credit Balance ($)", 0.0, 5000.0, 200.0)
    service_calls     = col2.number_input("Customer Service Calls", 0, 50, 1)
    payment_diversity = col3.number_input("Payment Method Diversity", 0, 10, 2)

    submitted = st.form_submit_button("Predict Churn", use_container_width=True)


# ── Prediction ─────────────────────────────────────────────────────────────────
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

    # Result display
    st.divider()
    m_col, b_col = st.columns([1, 2])
    m_col.metric("Churn Probability", f"{prob * 100:.1f}%")
    b_col.progress(prob)

    if risk_level == "High":
        st.error("⚠ High Risk — Strong churn signals detected. Consider a targeted retention offer or personal outreach within 48 hours.")
    elif risk_level == "Medium":
        st.warning("◈ Medium Risk — Moderate churn risk. A re-engagement email or check-in campaign could help.")
    else:
        st.success("✓ Low Risk — Healthy engagement signals. Maintain current strategy and consider upsell opportunities.")

    # Engineered feature debug expander
    with st.expander("Engineered features"):
        st.json(eng)

    # Persist to DB
    if DB_AVAILABLE:
        try:
            customer_record = {**raw}
            customer_record["Gender"]         = gender
            customer_record["Signup_Quarter"] = signup_quarter
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


# ── History ────────────────────────────────────────────────────────────────────
if DB_AVAILABLE:
    st.divider()
    h_col, c_col = st.columns([3, 1])
    h_col.subheader("Recent Predictions")

    if c_col.button("Clear All", use_container_width=True):
        clear_all()
        st.rerun()

    try:
        rows = fetch_latest(20)
        if rows:
            df_hist = pd.DataFrame(rows, columns=["ID", "Time", "Age", "Gender",
                                                   "Prediction", "Probability", "Risk"])
            df_hist["Probability"] = (df_hist["Probability"] * 100).round(1).astype(str) + "%"
            df_hist["Time"]        = df_hist["Time"].str[11:16]

            # Per-row delete via buttons outside the table
            for _, row in df_hist.iterrows():
                cols = st.columns([1, 2, 1, 1, 2, 2, 1])
                cols[0].write(f"#{int(row['ID'])}")
                cols[1].write(row["Time"])
                cols[2].write(row["Age"])
                cols[3].write(row["Gender"])
                cols[4].write(row["Probability"])
                risk = row["Risk"]
                if risk == "High":
                    cols[5].error(risk)
                elif risk == "Medium":
                    cols[5].warning(risk)
                else:
                    cols[5].success(risk)
                if cols[6].button("✕", key=f"del_{row['ID']}"):
                    delete_prediction(int(row["ID"]))
                    st.rerun()
        else:
            st.caption("No predictions yet.")
    except Exception as e:
        st.warning(f"Could not load history: {e}")