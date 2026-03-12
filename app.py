import os
import joblib
import pandas as pd
from datetime import datetime
from flask import Flask, request, jsonify
from db import init_db, insert_prediction, fetch_latest, delete_prediction, clear_all

app = Flask(__name__)

MODEL_FILE = "rf_model.pkl"
model = None
EXPECTED_FEATURES = None

GENDER_MAP    = {"Male": 0, "Female": 1, "Other": 2, "Unknown": 3}
QUARTER_MAP   = {"Q1": 0, "Q2": 1, "Q3": 2, "Q4": 3}
QUARTER_LABEL = {0: "Q1", 1: "Q2", 2: "Q3", 3: "Q4"}
GENDER_LABEL  = {0: "Male", 1: "Female", 2: "Other", 3: "Unknown"}


def load_model():
    global model, EXPECTED_FEATURES
    if os.path.exists(MODEL_FILE):
        model = joblib.load(MODEL_FILE)
        EXPECTED_FEATURES = model.feature_names_in_
        print("Model loaded successfully.")
    else:
        print(f"WARNING: '{MODEL_FILE}' not found — using heuristic fallback.")


def compute_engineered(f: dict) -> dict:
    login   = f.get("Login_Frequency", 0)
    days    = f.get("Days_Since_Last_Purchase", 999)
    ltv     = f.get("Lifetime_Value", 0)
    purch   = f.get("Total_Purchases", 1) or 1
    aov     = f.get("Average_Order_Value", 0)
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


HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Churn Predictor</title>
<link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600&display=swap" rel="stylesheet">
<style>
  * { margin:0; padding:0; box-sizing:border-box; }
  body { font-family:'Outfit',sans-serif; background:#f5f5f2; color:#1a1a1a; min-height:100vh; display:flex; justify-content:center; padding:36px 16px; }
  .wrap { width:100%; max-width:600px; display:flex; flex-direction:column; gap:20px; }

  .card { background:#fff; border-radius:14px; padding:28px; box-shadow:0 2px 20px rgba(0,0,0,0.06); }

  h1 { font-size:20px; font-weight:600; margin-bottom:3px; }
  .desc { font-size:13px; color:#888; font-weight:300; margin-bottom:20px; }

  .section-title { font-size:10px; font-weight:600; letter-spacing:.12em; text-transform:uppercase; color:#bbb; border-bottom:1px solid #ebebeb; padding-bottom:6px; margin:18px 0 12px; }
  .section-title:first-of-type { margin-top:0; }

  .grid { display:grid; grid-template-columns:1fr 1fr; gap:10px; }
  .grid-3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; }

  .field label { display:block; font-size:11px; font-weight:500; color:#555; margin-bottom:4px; }
  .field input, .field select {
    width:100%; padding:8px 10px; border:1.5px solid #e8e8e8; border-radius:7px;
    background:#fafafa; font-family:'Outfit',sans-serif; font-size:13px; color:#1a1a1a;
    outline:none; transition:border-color .15s; appearance:none;
  }
  .field input:focus, .field select:focus { border-color:#1a1a1a; background:#fff; }

  button {
    width:100%; padding:12px; background:#1a1a1a; color:#fff; border:none; border-radius:8px;
    font-family:'Outfit',sans-serif; font-size:14px; font-weight:500; cursor:pointer;
    margin-top:18px; transition:opacity .15s;
  }
  button:hover { opacity:.85; }
  button:disabled { opacity:.5; cursor:not-allowed; }

  /* Result */
  #result { display:none; }
  #result.show { display:block; }
  .result-top { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
  .result-pct { font-size:36px; font-weight:600; }
  .result-sub { font-size:12px; color:#888; margin-bottom:2px; }
  .badge { font-size:12px; font-weight:500; padding:4px 12px; border-radius:20px; }
  .bar-track { height:6px; background:#ebebeb; border-radius:99px; overflow:hidden; margin-bottom:12px; }
  .bar-fill { height:100%; border-radius:99px; width:0; transition:width .7s cubic-bezier(.16,1,.3,1); }
  .rec { font-size:13px; color:#555; line-height:1.6; }
  .err { font-size:12px; color:#d93025; margin-top:10px; display:none; }

  /* History table */
  .tbl-wrap { overflow-x:auto; }
  table { width:100%; border-collapse:collapse; font-size:12px; }
  th { text-align:left; font-weight:600; font-size:10px; letter-spacing:.08em; text-transform:uppercase; color:#aaa; padding:6px 8px; border-bottom:1px solid #ebebeb; }
  td { padding:8px; border-bottom:1px solid #f3f3f3; color:#333; vertical-align:middle; }
  tr:last-child td { border-bottom:none; }
  .risk-chip { display:inline-block; font-size:11px; font-weight:500; padding:2px 8px; border-radius:20px; }
  .risk-high   { color:#d93025; background:#fff0ef; }
  .risk-medium { color:#e37400; background:#fff8ec; }
  .risk-low    { color:#1e8e3e; background:#f0faf3; }
  .del-btn { background:none; border:none; color:#ccc; cursor:pointer; font-size:14px; width:auto; padding:0; margin:0; transition:color .15s; }
  .del-btn:hover { color:#d93025; opacity:1; }
  .clear-btn { background:none; border:1.5px solid #e8e8e8; color:#aaa; font-size:12px; width:auto; padding:6px 14px; border-radius:7px; cursor:pointer; margin:0; }
  .clear-btn:hover { border-color:#d93025; color:#d93025; opacity:1; }
  .tbl-header { display:flex; align-items:center; justify-content:space-between; margin-bottom:12px; }
  .tbl-title { font-size:14px; font-weight:600; }
  #no-history { font-size:13px; color:#bbb; text-align:center; padding:20px 0; display:none; }
</style>
</head>
<body>
<div class="wrap">

  <!-- Form card -->
  <div class="card">
    <h1>Churn Predictor</h1>
    <p class="desc">Enter customer details to estimate churn risk.</p>

    <form id="form">

      <div class="section-title">Demographics</div>
      <div class="grid">
        <div class="field"><label>Age</label><input type="number" id="age" placeholder="e.g. 35" min="18" max="100"></div>
        <div class="field"><label>Membership Years</label><input type="number" id="membership_years" placeholder="e.g. 2" min="0" step="0.1"></div>
        <div class="field"><label>Gender</label>
          <select id="gender">
            <option value="3">Unknown</option><option value="0">Male</option>
            <option value="1">Female</option><option value="2">Other</option>
          </select>
        </div>
        <div class="field"><label>Signup Quarter</label>
          <select id="signup_quarter">
            <option value="0">Q1 · Jan–Mar</option><option value="1">Q2 · Apr–Jun</option>
            <option value="2">Q3 · Jul–Sep</option><option value="3">Q4 · Oct–Dec</option>
          </select>
        </div>
      </div>

      <div class="section-title">Purchase Behaviour</div>
      <div class="grid">
        <div class="field"><label>Days Since Last Purchase</label><input type="number" id="days_since" placeholder="e.g. 30" min="0"></div>
        <div class="field"><label>Lifetime Value ($)</label><input type="number" id="lifetime_value" placeholder="e.g. 1200" min="0"></div>
        <div class="field"><label>Avg. Order Value ($)</label><input type="number" id="avg_order" placeholder="e.g. 85" min="0"></div>
        <div class="field"><label>Total Purchases</label><input type="number" id="total_purchases" placeholder="e.g. 14" min="0"></div>
        <div class="field"><label>Discount Usage (%)</label><input type="number" id="discount_rate" placeholder="e.g. 40" min="0" max="100"></div>
        <div class="field"><label>Returns Rate (%)</label><input type="number" id="returns_rate" placeholder="e.g. 10" min="0" max="100"></div>
        <div class="field"><label>Cart Abandonment (%)</label><input type="number" id="cart_abandon" placeholder="e.g. 50" min="0" max="100"></div>
        <div class="field"><label>Wishlist Items</label><input type="number" id="wishlist_items" placeholder="e.g. 5" min="0"></div>
      </div>

      <div class="section-title">Engagement</div>
      <div class="grid">
        <div class="field"><label>Login Frequency / mo</label><input type="number" id="login_freq" placeholder="e.g. 8" min="0"></div>
        <div class="field"><label>Session Duration Avg (min)</label><input type="number" id="session_duration" placeholder="e.g. 12" min="0"></div>
        <div class="field"><label>Pages Per Session</label><input type="number" id="pages_per_session" placeholder="e.g. 5" min="0"></div>
        <div class="field"><label>Email Open Rate (%)</label><input type="number" id="email_open_rate" placeholder="e.g. 30" min="0" max="100"></div>
        <div class="field"><label>Mobile App Usage (days/mo)</label><input type="number" id="mobile_app_usage" placeholder="e.g. 15" min="0"></div>
        <div class="field"><label>Social Media Engagement</label><input type="number" id="social_media" placeholder="e.g. 3" min="0"></div>
        <div class="field"><label>Product Reviews Written</label><input type="number" id="reviews_written" placeholder="e.g. 2" min="0"></div>
      </div>

      <div class="section-title">Financial & Support</div>
      <div class="grid">
        <div class="field"><label>Credit Balance ($)</label><input type="number" id="credit_balance" placeholder="e.g. 200" min="0"></div>
        <div class="field"><label>Customer Service Calls</label><input type="number" id="service_calls" placeholder="e.g. 1" min="0"></div>
        <div class="field"><label>Payment Method Diversity</label><input type="number" id="payment_diversity" placeholder="e.g. 2" min="0"></div>
      </div>

      <button type="submit" id="btn">Predict Churn</button>
      <div class="err" id="err"></div>
    </form>

    <!-- Result -->
    <div id="result" style="margin-top:20px;">
      <div class="result-top">
        <div>
          <div class="result-sub">Churn Probability</div>
          <div class="result-pct" id="pct">—</div>
        </div>
        <div id="badge" class="badge"></div>
      </div>
      <div class="bar-track"><div class="bar-fill" id="bar"></div></div>
      <div class="rec" id="rec"></div>
    </div>
  </div>

  <!-- History card -->
  <div class="card" id="history-card">
    <div class="tbl-header">
      <span class="tbl-title">Recent Predictions</span>
      <button class="clear-btn" onclick="clearAll()">Clear All</button>
    </div>
    <div id="no-history">No predictions yet.</div>
    <div class="tbl-wrap">
      <table id="hist-table">
        <thead>
          <tr>
            <th>#</th><th>Time</th><th>Age</th><th>Gender</th>
            <th>Probability</th><th>Risk</th><th></th>
          </tr>
        </thead>
        <tbody id="hist-body"></tbody>
      </table>
    </div>
  </div>

</div>

<script>
  function g(id, fb) {
    const v = document.getElementById(id).value.trim();
    return v === '' ? fb : parseFloat(v);
  }

  document.getElementById('form').addEventListener('submit', async function(e) {
    e.preventDefault();
    const btn = document.getElementById('btn');
    const errEl = document.getElementById('err');
    btn.disabled = true; btn.textContent = 'Predicting...';
    errEl.style.display = 'none';

    const payload = {
      Age:                            g('age', 35),
      Gender:                         parseFloat(document.getElementById('gender').value),
      Signup_Quarter:                 parseFloat(document.getElementById('signup_quarter').value),
      Membership_Years:               g('membership_years', 0),
      Days_Since_Last_Purchase:       g('days_since', 999),
      Lifetime_Value:                 g('lifetime_value', 0),
      Average_Order_Value:            g('avg_order', 0),
      Total_Purchases:                g('total_purchases', 0),
      Discount_Usage_Rate:            g('discount_rate', 0),
      Returns_Rate:                   g('returns_rate', 0),
      Cart_Abandonment_Rate:          g('cart_abandon', 50),
      Wishlist_Items:                 g('wishlist_items', 0),
      Login_Frequency:                g('login_freq', 0),
      Session_Duration_Avg:           g('session_duration', 0),
      Pages_Per_Session:              g('pages_per_session', 0),
      Email_Open_Rate:                g('email_open_rate', 0),
      Mobile_App_Usage:               g('mobile_app_usage', 0),
      Social_Media_Engagement_Score:  g('social_media', 0),
      Product_Reviews_Written:        g('reviews_written', 0),
      Credit_Balance:                 g('credit_balance', 0),
      Customer_Service_Calls:         g('service_calls', 0),
      Payment_Method_Diversity:       g('payment_diversity', 1),
    };

    try {
      const res  = await fetch('/predict', { method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload) });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Server error');
      showResult(data.probability);
      loadHistory();
    } catch(err) {
      errEl.textContent = 'Error: ' + err.message;
      errEl.style.display = 'block';
    } finally {
      btn.disabled = false; btn.textContent = 'Predict Churn';
    }
  });

  function showResult(prob) {
    const pct = Math.round(prob * 100);
    let color, bg, label, rec;
    if (prob > 0.65)      { color='#d93025'; bg='#fff0ef'; label='⚠ High Risk';   rec='Strong churn signals detected. Consider a targeted retention offer or personal outreach within 48 hours.'; }
    else if (prob > 0.35) { color='#e37400'; bg='#fff8ec'; label='◈ Medium Risk'; rec='Moderate churn risk. A re-engagement email or check-in campaign could help.'; }
    else                  { color='#1e8e3e'; bg='#f0faf3'; label='✓ Low Risk';    rec='Healthy engagement signals. Maintain current strategy and consider upsell opportunities.'; }

    document.getElementById('pct').textContent = pct + '%';
    document.getElementById('pct').style.color = color;
    const badge = document.getElementById('badge');
    badge.textContent = label; badge.style.color = color; badge.style.background = bg;
    document.getElementById('bar').style.background = color;
    document.getElementById('rec').textContent = rec;
    const r = document.getElementById('result');
    r.classList.remove('show'); void r.offsetWidth; r.classList.add('show');
    setTimeout(() => { document.getElementById('bar').style.width = pct + '%'; }, 50);
  }

  async function loadHistory() {
    const res  = await fetch('/history');
    const rows = await res.json();
    const tbody = document.getElementById('hist-body');
    const noH   = document.getElementById('no-history');
    const tbl   = document.getElementById('hist-table');
    if (!rows.length) { noH.style.display='block'; tbl.style.display='none'; return; }
    noH.style.display = 'none'; tbl.style.display = '';
    tbody.innerHTML = rows.map(r => {
      const riskClass = r.risk_level === 'High' ? 'risk-high' : r.risk_level === 'Medium' ? 'risk-medium' : 'risk-low';
      const pct = Math.round(r.churn_probability * 100);
      const time = r.created_at.slice(11, 16);
      return `<tr>
        <td style="color:#bbb">#${r.id}</td>
        <td>${time}</td>
        <td>${r.age}</td>
        <td>${r.gender}</td>
        <td style="font-weight:600">${pct}%</td>
        <td><span class="risk-chip ${riskClass}">${r.risk_level}</span></td>
        <td><button class="del-btn" onclick="deletePred(${r.id})">✕</button></td>
      </tr>`;
    }).join('');
  }

  async function deletePred(id) {
    await fetch('/history/' + id, { method:'DELETE' });
    loadHistory();
  }

  async function clearAll() {
    if (!confirm('Clear all prediction history?')) return;
    await fetch('/history', { method:'DELETE' });
    loadHistory();
  }

  loadHistory();
</script>
</body>
</html>"""


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return HTML


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        # Encode categoricals if sent as strings
        if isinstance(data.get("Gender"), str):
            data["Gender"] = GENDER_MAP.get(data["Gender"], 3)
        if isinstance(data.get("Signup_Quarter"), str):
            data["Signup_Quarter"] = QUARTER_MAP.get(data["Signup_Quarter"], 0)

        # Engineered features
        eng = compute_engineered(data)

        # Prediction
        if model is not None:
            row = {**data, **eng}
            df  = pd.DataFrame([row])
            for col in EXPECTED_FEATURES:
                if col not in df.columns:
                    df[col] = 0
            df   = df[EXPECTED_FEATURES]
            prob = float(model.predict_proba(df)[0][1])
        else:
            prob = heuristic(data)

        prediction = 1 if prob > 0.50 else 0
        risk_level = "High" if prob > 0.65 else "Medium" if prob > 0.35 else "Low"

        # Save to DB — store Gender/Quarter as readable labels
        customer_record = {**data}
        customer_record["Gender"]         = GENDER_LABEL.get(int(data.get("Gender", 3)), "Unknown")
        customer_record["Signup_Quarter"] = QUARTER_LABEL.get(int(data.get("Signup_Quarter", 0)), "Q1")

        insert_prediction(
            created_at  = datetime.now().isoformat(sep=" ", timespec="seconds"),
            customer    = customer_record,
            engineered  = eng,
            prediction  = prediction,
            probability = prob,
            risk_level  = risk_level,
        )

        return jsonify({"probability": prob, "verdict": "CHURN" if prediction else "RETAIN", "risk_level": risk_level})

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/history", methods=["GET"])
def history():
    rows = fetch_latest(20)
    return jsonify([
        {"id": r[0], "created_at": r[1], "age": r[2], "gender": r[3],
         "churn_prediction": r[4], "churn_probability": r[5], "risk_level": r[6]}
        for r in rows
    ])


@app.route("/history/<int:pred_id>", methods=["DELETE"])
def delete_row(pred_id):
    delete_prediction(pred_id)
    return jsonify({"deleted": pred_id})


@app.route("/history", methods=["DELETE"])
def clear_history():
    clear_all()
    return jsonify({"cleared": True})

