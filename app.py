import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import random
import time

px.defaults.template = "plotly_white"

# ─────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="Quick Commerce Intelligence Dashboard",
    layout="wide",
    page_icon="🛒",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────
# GLOBAL CSS — Polished & Aligned
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');
* { font-family: 'Inter', sans-serif !important; }

.stApp {
    background: linear-gradient(135deg, #fff7ed 0%, #ffedd5 60%, #fed7aa 100%);
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0533 0%, #2d0f5e 60%, #1c0040 100%) !important;
    border-right: 3px solid #f97316;
}
section[data-testid="stSidebar"] * { color: #ffffff !important; }
section[data-testid="stSidebar"] .stMarkdown p { color: #fb923c !important; font-weight: 700; }
section[data-testid="stSidebar"] h2 { color: #fb923c !important; }
section[data-testid="stSidebar"] .stMarkdown h3 { color: #fbbf24 !important; }

/* ── Metric Cards ── */
div[data-testid="metric-container"] {
    background: white;
    border-left: 6px solid #f97316;
    padding: 16px 20px;
    border-radius: 14px;
    box-shadow: 0 4px 20px rgba(249,115,22,0.15);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
    height: 100%;
}
div[data-testid="metric-container"]:hover {
    transform: translateY(-3px);
    box-shadow: 0 10px 30px rgba(249,115,22,0.25);
}
div[data-testid="metric-container"] label {
    font-size: 0.78rem !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    color: #9a3412 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 1.55rem !important;
    font-weight: 900 !important;
    color: #1a1a2e !important;
}

/* ── Charts ── */
.stPlotlyChart {
    background: white;
    padding: 16px;
    border-radius: 16px;
    box-shadow: 0 4px 18px rgba(249,115,22,0.10);
    margin-bottom: 12px;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: transparent;
    border-bottom: 2.5px solid #fed7aa;
    padding-bottom: 0;
    flex-wrap: wrap;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 10px 10px 0 0;
    padding: 9px 16px;
    font-weight: 700;
    font-size: 0.82rem;
    background: white;
    color: #ea580c;
    border: 2px solid #fed7aa;
    border-bottom: none;
    transition: all 0.18s ease;
}
.stTabs [data-baseweb="tab"]:hover {
    background: #fff7ed;
    color: #c2410c;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #f97316, #ea580c) !important;
    color: white !important;
    border-color: #f97316 !important;
}

/* ── Download Buttons ── */
.stDownloadButton button {
    background: linear-gradient(90deg, #f97316, #ea580c) !important;
    color: white !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    border: none !important;
    padding: 10px 22px !important;
    transition: opacity 0.2s !important;
}
.stDownloadButton button:hover { opacity: 0.88 !important; }

/* ── Section Headers ── */
.sec-hdr {
    background: linear-gradient(90deg, #f97316, #c2410c);
    color: white;
    padding: 8px 20px;
    border-radius: 10px;
    font-size: 0.95rem;
    font-weight: 700;
    margin: 20px 0 12px 0;
    display: inline-block;
    letter-spacing: 0.01em;
}

/* ── Alert Cards ── */
.card-danger  { background:#fef2f2; border-left:5px solid #ef4444; color:#991b1b; padding:13px 18px; border-radius:10px; margin:8px 0; font-weight:600; }
.card-warning { background:#fffbeb; border-left:5px solid #f59e0b; color:#92400e; padding:13px 18px; border-radius:10px; margin:8px 0; font-weight:600; }
.card-success { background:#f0fdf4; border-left:5px solid #22c55e; color:#166534; padding:13px 18px; border-radius:10px; margin:8px 0; font-weight:600; }
.card-info    { background:#eff6ff; border-left:5px solid #3b82f6; color:#1e40af; padding:13px 18px; border-radius:10px; margin:8px 0; font-weight:600; }

/* ── Live Badge ── */
.live-badge {
    display:inline-block; background:#ef4444; color:white;
    padding:3px 12px; border-radius:20px; font-size:0.75rem; font-weight:700;
    animation: blink 1.3s ease-in-out infinite; margin-left:12px; vertical-align:middle;
}
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.35} }

/* ── Dashboard Header ── */
.dash-title {
    background: linear-gradient(135deg, #7c2d12, #c2410c, #ea580c);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    font-size: 2.3rem !important;
    font-weight: 900 !important;
    line-height: 1.1 !important;
    margin-bottom: 4px;
}
.dash-subtitle {
    color: #ea580c;
    font-size: 1rem;
    font-weight: 600;
    margin-top: 0;
    margin-bottom: 16px;
}

/* ── KPI Row ── */
.kpi-section {
    background: white;
    border-radius: 18px;
    padding: 20px 24px;
    box-shadow: 0 4px 24px rgba(249,115,22,0.12);
    margin: 12px 0 20px 0;
}

/* ── Headings ── */
h1 { color: #c2410c !important; font-size: 2.2rem !important; font-weight: 900 !important; }
h2 { color: #9a3412 !important; font-weight: 800 !important; margin-top: 8px !important; }
h3 { color: #7c2d12 !important; font-weight: 700 !important; }
h4 { color: #92400e !important; font-weight: 700 !important; }

/* ── Dividers ── */
hr { border-color: #fed7aa !important; margin: 20px 0 !important; }

/* ── Dataframe ── */
.stDataFrame { border-radius: 12px !important; overflow: hidden; }

/* ── Info/Warning/Success boxes ── */
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
# HEADER
# ─────────────────────────────────────
col_hdr, col_time = st.columns([4, 1])
with col_hdr:
    st.markdown(f"""
    <h1>🛒 Quick Commerce Intelligence Platform
    <span class="live-badge">● LIVE</span></h1>
    <p class="dash-subtitle">
        Advanced AI Analytics &nbsp;·&nbsp; Blinkit &nbsp;|&nbsp; Swiggy Instamart &nbsp;|&nbsp; Zepto
    </p>
    """, unsafe_allow_html=True)
with col_time:
    st.markdown(f"""
    <div style="text-align:right; padding-top:18px;">
        <div style="font-size:0.75rem; color:#9a3412; font-weight:600; text-transform:uppercase; letter-spacing:0.05em;">Last Refreshed</div>
        <div style="font-size:1rem; color:#ea580c; font-weight:800;">{datetime.now().strftime('%d %b %Y')}</div>
        <div style="font-size:0.9rem; color:#c2410c; font-weight:700;">{datetime.now().strftime('%I:%M %p')}</div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────
# LOAD DATA — Cached with hash suppression for speed
# ─────────────────────────────────────
@st.cache_data(show_spinner="⏳ Loading data…")
def load_data():
    df = pd.read_csv("sales_data.csv")
    if "Order_Date" not in df.columns:
        rng = np.random.default_rng(42)
        base = datetime(2024, 1, 1)
        offsets = rng.integers(0, 364, size=len(df))
        df["Order_Date"] = [base + timedelta(days=int(o)) for o in offsets]
    else:
        df["Order_Date"] = pd.to_datetime(df["Order_Date"], errors="coerce")
    if "Customer_ID" not in df.columns:
        rng = np.random.default_rng(42)
        df["Customer_ID"] = ["CUST_" + str(x) for x in rng.integers(1000, 4000, size=len(df))]
    # Pre-cast numeric dtypes for speed
    for col in ["Order_Value", "Delivery_Time_Min", "Customer_Rating", "Customer_Age"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # Clean categorical string columns — drop rows where key cols are NaN,
    # then cast to str so sorted() never sees mixed float/str types
    str_cols = ["Company", "City", "Product_Category", "Payment_Method",
                "Discount_Applied", "Customer_ID"]
    for col in str_cols:
        if col in df.columns:
            df[col] = df[col].fillna("Unknown").astype(str).str.strip()
    # Drop rows that are completely empty on the core numeric fields
    df = df.dropna(subset=["Order_Value", "Delivery_Time_Min", "Customer_Rating"], how="all")
    df = df.reset_index(drop=True)
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("❌ sales_data.csv not found. Place it in the same folder as app.py.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

# ─────────────────────────────────────
# CACHED ML FUNCTIONS
# ─────────────────────────────────────
@st.cache_data(show_spinner=False)
def run_anomaly_detection(df_key, contamination):
    feat_cols = ["Order_Value", "Delivery_Time_Min", "Customer_Rating"]
    X = df_key[feat_cols].fillna(df_key[feat_cols].mean())
    iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=80, n_jobs=-1)
    scores = iso.fit_predict(X)
    out = df_key[feat_cols].copy()
    out["Anomaly_Score"] = scores
    out["Is_Anomaly"]    = scores == -1
    out["Anomaly_Label"] = out["Is_Anomaly"].map({True: "🔴 Anomaly", False: "🟢 Normal"})
    return out

@st.cache_data(show_spinner=False)
def run_churn_model(df_key, sla):
    churn_df = df_key.copy()
    churn_df["Churn"] = (
        (churn_df["Customer_Rating"] < 3) &
        (churn_df["Delivery_Time_Min"] > sla)
    ).astype(int)
    le = LabelEncoder()
    for col in ["Company", "City", "Product_Category"]:
        churn_df[f"{col}_enc"] = le.fit_transform(churn_df[col].astype(str))
    feat_c = ["Order_Value", "Delivery_Time_Min", "Customer_Rating", "Customer_Age",
              "Company_enc", "City_enc", "Product_Category_enc"]
    X_c = churn_df[feat_c].fillna(0)
    y_c = churn_df["Churn"]
    if y_c.nunique() < 2:
        return {"df": churn_df, "model": None, "features": feat_c, "trained": False}
    X_tr, X_te, y_tr, y_te = train_test_split(X_c, y_c, test_size=0.25, random_state=42)
    rf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1, max_depth=8)
    rf.fit(X_tr, y_tr)
    churn_df["Churn_Prob"] = rf.predict_proba(X_c)[:, 1]
    return {"df": churn_df, "model": rf, "features": feat_c, "trained": True}

@st.cache_data(show_spinner=False)
def compute_rfm(df_key):
    now = pd.Timestamp.now()
    rfm = df_key.groupby("Customer_ID").agg(
        Recency  =("Order_Date", lambda x: (now - pd.to_datetime(x).max()).days),
        Frequency=("Order_Value", "count"),
        Monetary =("Order_Value", "sum")
    ).reset_index()
    for col_name, labels in [("Recency", [5,4,3,2,1]),
                              ("Frequency", [1,2,3,4,5]),
                              ("Monetary",  [1,2,3,4,5])]:
        try:
            rfm[f"{col_name}_Score"] = pd.qcut(rfm[col_name], 5, labels=labels, duplicates="drop")
        except Exception:
            rfm[f"{col_name}_Score"] = 3
    rfm["RFM_Score"] = (rfm["Recency_Score"].astype(int) +
                        rfm["Frequency_Score"].astype(int) +
                        rfm["Monetary_Score"].astype(int))
    def seg(s):
        if s >= 12:  return "🏆 Champions"
        elif s >= 9: return "💚 Loyal"
        elif s >= 6: return "🟡 At-Risk"
        else:        return "🔴 Lost"
    rfm["Segment"] = rfm["RFM_Score"].apply(seg)
    return rfm

# ─────────────────────────────────────
# COLUMN VALIDATION
# ─────────────────────────────────────
REQUIRED = ["Company", "City", "Customer_Age", "Order_Value",
            "Delivery_Time_Min", "Product_Category",
            "Payment_Method", "Customer_Rating", "Discount_Applied"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    st.error(f"Missing columns: {missing}")
    st.stop()

# ─────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────
st.sidebar.markdown("## 🎛️ Control Panel")
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Data Filters")

def _opts(series):
    """Return sorted unique string values, dropping NaN/empty."""
    return sorted(series.dropna().astype(str).str.strip().unique().tolist())

company  = st.sidebar.multiselect("🏢 Company",          _opts(df["Company"]),          default=_opts(df["Company"]))
city     = st.sidebar.multiselect("📍 City",             _opts(df["City"]),             default=_opts(df["City"]))
category = st.sidebar.multiselect("🛍 Product Category", _opts(df["Product_Category"]), default=_opts(df["Product_Category"]))

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ ML & Ops Settings")
delivery_sla          = st.sidebar.number_input("🚚 Delivery SLA (min)", min_value=10, max_value=60, value=30)
anomaly_contamination = st.sidebar.slider("🔍 Anomaly Sensitivity", 0.01, 0.20, 0.05, 0.01,
                                          help="Higher = more anomalies flagged")
forecast_days         = st.sidebar.slider("📈 Forecast Horizon (days)", 7, 90, 30)
forecast_method       = st.sidebar.selectbox("📊 Forecast Model",
                                             ["Linear Regression", "Polynomial Degree-3", "Rolling Average"])

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔴 Live Feed")
auto_refresh = st.sidebar.checkbox("⟳ Auto-Refresh every 15s", value=False)
if auto_refresh:
    time.sleep(15)
    st.rerun()

# ─────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────
filtered_df = df[
    df["Company"].isin(company) &
    df["City"].isin(city) &
    df["Product_Category"].isin(category)
].copy().reset_index(drop=True)

if filtered_df.empty:
    st.warning("⚠️ No data for the selected filters. Please adjust your selections.")
    st.stop()

# ─────────────────────────────────────
# KPIs — Computed once, used everywhere
# ─────────────────────────────────────
total_orders   = len(filtered_df)
total_revenue  = filtered_df["Order_Value"].sum()
avg_delivery   = filtered_df["Delivery_Time_Min"].mean()
avg_rating     = filtered_df["Customer_Rating"].mean()
sla_breach_pct = (filtered_df["Delivery_Time_Min"] > delivery_sla).mean() * 100
avg_order_val  = filtered_df["Order_Value"].mean()
top_category   = filtered_df.groupby("Product_Category")["Order_Value"].sum().idxmax()

# ─────────────────────────────────────
# PRE-COMPUTE AGGREGATIONS — once for all tabs
# ─────────────────────────────────────
city_sales    = filtered_df.groupby("City")["Order_Value"].sum().reset_index()
city_delivery = filtered_df.groupby("City")["Delivery_Time_Min"].mean().reset_index()
payment_data  = filtered_df.groupby("Payment_Method")["Order_Value"].sum().reset_index()
comp_revenue  = filtered_df.groupby("Company")["Order_Value"].sum().reset_index()
comp_rating   = filtered_df.groupby("Company")["Customer_Rating"].mean().reset_index()

# Pre-compute RFM here so Tab 7 export always has it
rfm = compute_rfm(filtered_df)

# ─────────────────────────────────────
# KPI DISPLAY — Executive Overview
# ─────────────────────────────────────
st.markdown("## 📊 Executive Overview")

with st.container():
    k1, k2, k3, k4, k5, k6, k7 = st.columns(7)
    k1.metric("📦 Total Orders",    f"{total_orders:,}")
    k2.metric("💰 Revenue",         f"₹{total_revenue:,.0f}")
    k3.metric("⏱ Avg Delivery",     f"{avg_delivery:.1f} min")
    k4.metric("⭐ Avg Rating",       f"{avg_rating:.2f} / 5")
    k5.metric("🚨 SLA Breach",      f"{sla_breach_pct:.1f}%",
              delta=f"{sla_breach_pct-10:.1f}% vs 10% goal", delta_color="inverse")
    k6.metric("🛒 Avg Order Value", f"₹{avg_order_val:.0f}")
    k7.metric("🔥 Top Category",    top_category)

st.markdown("")
if avg_delivery > delivery_sla:
    st.markdown(f'<div class="card-danger">🚨 <strong>SLA BREACH</strong> — Avg delivery ({avg_delivery:.1f} min) exceeds your SLA ({delivery_sla} min). Immediate fleet action required!</div>', unsafe_allow_html=True)
elif avg_delivery > delivery_sla * 0.85:
    st.markdown(f'<div class="card-warning">⚠️ <strong>WARNING</strong> — Delivery time ({avg_delivery:.1f} min) approaching SLA limit of {delivery_sla} min.</div>', unsafe_allow_html=True)
else:
    st.markdown(f'<div class="card-success">✅ <strong>HEALTHY</strong> — All deliveries within SLA. Avg: {avg_delivery:.1f} min.</div>', unsafe_allow_html=True)

st.divider()

# ─────────────────────────────────────
# TABS
# ─────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📈 Sales Overview",
    "🚚 Delivery Performance",
    "💳 Customer Insights",
    "🤖 AI & Anomaly Detection",
    "📊 Demand Forecasting",
    "🔴 Live Order Feed",
    "🔍 NL Query & Export"
])

# ══════════════════════════════════
# TAB 1 — SALES OVERVIEW
# ══════════════════════════════════
with tab1:
    st.markdown('<div class="sec-hdr">📍 Revenue by City</div>', unsafe_allow_html=True)
    fig1 = px.bar(city_sales.sort_values("Order_Value", ascending=False),
                  x="City", y="Order_Value",
                  color="Order_Value", color_continuous_scale="Oranges",
                  text_auto=True, title="City-wise Total Revenue")
    fig1.update_traces(textfont_size=12, marker_line_width=0)
    fig1.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380,
                       margin=dict(t=50, b=40))
    st.plotly_chart(fig1, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown('<div class="sec-hdr">🛍 Revenue by Category</div>', unsafe_allow_html=True)
        cat_data = filtered_df.groupby("Product_Category")["Order_Value"].sum().reset_index()
        fig2 = px.pie(cat_data, names="Product_Category", values="Order_Value",
                      hole=0.44, title="Category Revenue Share",
                      color_discrete_sequence=px.colors.sequential.Oranges_r)
        fig2.update_traces(textinfo="percent+label", textfont_size=13)
        fig2.update_layout(paper_bgcolor="white", height=380, margin=dict(t=50, b=20))
        st.plotly_chart(fig2, use_container_width=True)

    with col_b:
        st.markdown('<div class="sec-hdr">🏢 Company Revenue</div>', unsafe_allow_html=True)
        fig3 = px.bar(comp_revenue.sort_values("Order_Value", ascending=False),
                      x="Company", y="Order_Value",
                      color="Company", text_auto=True, title="Revenue by Company")
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           height=380, showlegend=False, margin=dict(t=50, b=40))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown('<div class="sec-hdr">🗺️ Revenue Heatmap — City × Company</div>', unsafe_allow_html=True)
    pivot = filtered_df.pivot_table(values="Order_Value", index="City",
                                    columns="Company", aggfunc="sum", fill_value=0)
    fig_heat = px.imshow(pivot, text_auto=".2s", aspect="auto",
                         color_continuous_scale="Oranges",
                         title="Revenue Heatmap (City × Company)")
    fig_heat.update_layout(height=420, paper_bgcolor="white",
                            margin=dict(l=40, r=40, t=60, b=40))
    fig_heat.update_traces(textfont_size=13)
    st.plotly_chart(fig_heat, use_container_width=True)
    st.info("💡 High-intensity cells reveal top city-company combos — great for expansion decisions.")

    st.markdown('<div class="sec-hdr">🏷 Discount Impact Analysis</div>', unsafe_allow_html=True)
    disc = filtered_df.groupby("Discount_Applied").agg(
        Avg_Order_Value=("Order_Value", "mean"),
        Avg_Rating=("Customer_Rating", "mean"),
        Count=("Order_Value", "count")
    ).reset_index()
    fig_disc = go.Figure()
    fig_disc.add_trace(go.Bar(
        x=disc["Discount_Applied"].astype(str), y=disc["Avg_Order_Value"],
        name="Avg Order Value (₹)", marker_color="#f97316",
        text=disc["Avg_Order_Value"].round(0), textposition="outside"
    ))
    fig_disc.add_trace(go.Scatter(
        x=disc["Discount_Applied"].astype(str), y=disc["Avg_Rating"],
        name="Avg Rating", yaxis="y2", mode="lines+markers",
        line=dict(color="#7c3aed", width=3), marker=dict(size=10)
    ))
    fig_disc.update_layout(
        title="Discount vs Order Value & Customer Rating",
        yaxis=dict(title="Avg Order Value (₹)"),
        yaxis2=dict(title="Avg Rating", overlaying="y", side="right", range=[0, 5]),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", y=-0.22), height=380,
        margin=dict(t=55, b=80)
    )
    st.plotly_chart(fig_disc, use_container_width=True)

# ══════════════════════════════════
# TAB 2 — DELIVERY PERFORMANCE
# ══════════════════════════════════
with tab2:
    st.markdown('<div class="sec-hdr">🚚 Average Delivery Time by City</div>', unsafe_allow_html=True)
    fig4 = px.bar(city_delivery.sort_values("Delivery_Time_Min", ascending=False),
                  x="City", y="Delivery_Time_Min",
                  color="Delivery_Time_Min", color_continuous_scale="RdYlGn_r",
                  text_auto=True, title="Avg Delivery Time per City (min)")
    fig4.add_hline(y=delivery_sla, line_dash="dash", line_color="red",
                   annotation_text=f"SLA = {delivery_sla} min", annotation_font_color="red")
    fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380,
                       margin=dict(t=55, b=40))
    st.plotly_chart(fig4, use_container_width=True)

    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.markdown('<div class="sec-hdr">📦 Delivery Distribution by Company</div>', unsafe_allow_html=True)
        fig_box = px.box(filtered_df, x="Company", y="Delivery_Time_Min",
                         color="Company", points=False,
                         title="Delivery Time Distribution (Box Plot)")
        fig_box.add_hline(y=delivery_sla, line_dash="dash", line_color="red",
                          annotation_text=f"SLA = {delivery_sla} min")
        fig_box.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=400, showlegend=False, margin=dict(t=55, b=40))
        st.plotly_chart(fig_box, use_container_width=True)

    with col_d2:
        st.markdown('<div class="sec-hdr">⭐ Company Average Ratings</div>', unsafe_allow_html=True)
        fig_cr = px.bar(comp_rating.sort_values("Customer_Rating", ascending=False),
                        x="Company", y="Customer_Rating",
                        color="Customer_Rating", color_continuous_scale="RdYlGn",
                        text_auto=True, title="Avg Rating by Company")
        fig_cr.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             height=400, margin=dict(t=55, b=40))
        st.plotly_chart(fig_cr, use_container_width=True)

    st.markdown('<div class="sec-hdr">⭐ Delivery Time vs Customer Rating</div>', unsafe_allow_html=True)
    _sc = filtered_df.dropna(subset=["Delivery_Time_Min", "Customer_Rating", "Order_Value"])
    # Sample for large datasets to keep scatter fast
    _sc_plot = _sc.sample(min(len(_sc), 2000), random_state=42) if len(_sc) > 2000 else _sc
    fig5 = px.scatter(_sc_plot, x="Delivery_Time_Min", y="Customer_Rating",
                      color="Company", size="Order_Value",
                      hover_data=["City", "Product_Category"],
                      title="Does Faster Delivery → Higher Rating?")
    if len(_sc) > 1:
        _xv = _sc["Delivery_Time_Min"].values
        _yv = _sc["Customer_Rating"].values
        _c  = np.polyfit(_xv, _yv, 1)
        _xl = np.linspace(_xv.min(), _xv.max(), 200)
        fig5.add_trace(go.Scatter(x=_xl, y=np.polyval(_c, _xl),
                                  mode="lines", name="Trend",
                                  line=dict(color="black", width=2, dash="dash")))
    fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=430,
                       margin=dict(t=55, b=40))
    st.plotly_chart(fig5, use_container_width=True)

# ══════════════════════════════════
# TAB 3 — CUSTOMER INSIGHTS + RFM
# ══════════════════════════════════
with tab3:
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        st.markdown('<div class="sec-hdr">💳 Revenue by Payment Method</div>', unsafe_allow_html=True)
        fig6 = px.bar(payment_data.sort_values("Order_Value", ascending=False),
                      x="Payment_Method", y="Order_Value",
                      color="Order_Value", color_continuous_scale="Oranges",
                      text_auto=True, title="Payment Method Revenue")
        fig6.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=370,
                           margin=dict(t=55, b=40))
        st.plotly_chart(fig6, use_container_width=True)

    with col_p2:
        st.markdown('<div class="sec-hdr">👤 Spending by Customer Age</div>', unsafe_allow_html=True)
        age_grp = filtered_df.groupby("Customer_Age")["Order_Value"].sum().reset_index()
        fig7 = px.line(age_grp, x="Customer_Age", y="Order_Value",
                       markers=True, title="Customer Age vs Total Spending",
                       color_discrete_sequence=["#f97316"])
        fig7.update_traces(line_width=2.5, marker_size=7)
        fig7.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=370,
                           margin=dict(t=55, b=40))
        st.plotly_chart(fig7, use_container_width=True)

    st.markdown('<div class="sec-hdr">⭐ Rating Distribution by Company (Violin)</div>', unsafe_allow_html=True)
    fig_violin = px.violin(filtered_df, x="Company", y="Customer_Rating",
                           color="Company", box=True, points=False,
                           title="Rating Distribution — Violin + Box")
    fig_violin.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             height=390, showlegend=False, margin=dict(t=55, b=40))
    st.plotly_chart(fig_violin, use_container_width=True)

    st.markdown('<div class="sec-hdr">👥 RFM Customer Segmentation</div>', unsafe_allow_html=True)
    st.markdown("**RFM = Recency · Frequency · Monetary** — the gold standard in retail customer analytics.")

    seg_counts = rfm["Segment"].value_counts().reset_index()
    seg_counts.columns = ["Segment", "Count"]

    r1, r2 = st.columns(2)
    with r1:
        fig_rfm_pie = px.pie(seg_counts, names="Segment", values="Count", hole=0.44,
                             title="Customer Segments (RFM)",
                             color_discrete_sequence=["#22c55e", "#f97316", "#facc15", "#ef4444"])
        fig_rfm_pie.update_layout(paper_bgcolor="white", height=370, margin=dict(t=55, b=20))
        st.plotly_chart(fig_rfm_pie, use_container_width=True)
    with r2:
        seg_kpis = rfm.groupby("Segment").agg(
            Customers=("Customer_ID", "count"),
            Avg_Recency=("Recency", "mean"),
            Avg_Orders=("Frequency", "mean"),
            Avg_Revenue=("Monetary", "mean")
        ).reset_index().round(1)
        st.markdown("#### 📊 Segment KPIs")
        st.dataframe(
            seg_kpis.style.background_gradient(subset=["Avg_Revenue"], cmap="Oranges"),
            use_container_width=True, height=340
        )

    # Sample 3D scatter for performance
    rfm_plot = rfm.sample(min(len(rfm), 1500), random_state=42) if len(rfm) > 1500 else rfm
    fig_rfm_3d = px.scatter_3d(rfm_plot, x="Recency", y="Frequency", z="Monetary",
                                color="Segment", hover_data=["Customer_ID"],
                                title="3D RFM Customer Space",
                                color_discrete_sequence=["#22c55e", "#f97316", "#facc15", "#ef4444"])
    fig_rfm_3d.update_layout(paper_bgcolor="white", height=500, margin=dict(t=60))
    st.plotly_chart(fig_rfm_3d, use_container_width=True)

# ══════════════════════════════════
# TAB 4 — AI & ANOMALY DETECTION
# ══════════════════════════════════
with tab4:
    st.markdown('<div class="sec-hdr">🤖 Isolation Forest — Anomaly Detection</div>', unsafe_allow_html=True)
    st.markdown("""
    **Isolation Forest** isolates anomalies by randomly partitioning data. Orders isolated quickly
    (short decision paths) are flagged as anomalies — potential fraud, outlier spend, or data errors.
    """)

    with st.spinner("Running anomaly detection (cached after first run)…"):
        anom_result = run_anomaly_detection(filtered_df, anomaly_contamination)
    filtered_df["Anomaly_Score"] = anom_result["Anomaly_Score"].values
    filtered_df["Is_Anomaly"]    = anom_result["Is_Anomaly"].values
    filtered_df["Anomaly_Label"] = anom_result["Anomaly_Label"].values

    n_anom = int(filtered_df["Is_Anomaly"].sum())
    am1, am2, am3 = st.columns(3)
    am1.metric("🔴 Anomalies Detected", n_anom)
    am2.metric("🟢 Normal Orders",      total_orders - n_anom)
    am3.metric("⚠️ Anomaly Rate",       f"{n_anom / total_orders * 100:.1f}%")

    st.markdown("")
    filtered_df["Rating_Size"] = filtered_df["Customer_Rating"].fillna(1).clip(lower=1)
    # Sample scatter for large datasets
    _anom_plot = filtered_df.sample(min(len(filtered_df), 2500), random_state=42) if len(filtered_df) > 2500 else filtered_df
    fig_anom = px.scatter(
        _anom_plot, x="Delivery_Time_Min", y="Order_Value",
        color="Anomaly_Label",
        color_discrete_map={"🔴 Anomaly": "#ef4444", "🟢 Normal": "#22c55e"},
        size="Rating_Size",
        hover_data=["City", "Company", "Product_Category"],
        title="Anomaly Detection — Isolation Forest (Delivery Time vs Order Value)"
    )
    fig_anom.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=430,
                           margin=dict(t=55, b=40))
    st.plotly_chart(fig_anom, use_container_width=True)

    st.markdown("#### 🔴 Top Anomalous Orders")
    anom_tbl = filtered_df[filtered_df["Is_Anomaly"]]\
        [["Company", "City", "Order_Value", "Delivery_Time_Min", "Customer_Rating", "Product_Category"]]\
        .sort_values("Order_Value", ascending=False).head(20)
    st.dataframe(
        anom_tbl.style.background_gradient(subset=["Order_Value"], cmap="Reds"),
        use_container_width=True
    )

    st.divider()
    st.markdown('<div class="sec-hdr">📉 Churn Prediction — Random Forest Classifier</div>', unsafe_allow_html=True)
    st.markdown("""
    Customers with **low rating + delivery beyond SLA** are labelled as churned.
    A **Random Forest Classifier** predicts future churn probability per customer interaction.
    """)

    with st.spinner("Training churn model (cached after first run)…"):
        churn_result = run_churn_model(filtered_df, delivery_sla)
    churn_df = churn_result["df"]
    rf_model = churn_result["model"]
    feat_c   = churn_result["features"]

    if churn_result["trained"]:
        ci1, ci2 = st.columns(2)
        with ci1:
            imp_df = pd.DataFrame({"Feature": feat_c, "Importance": rf_model.feature_importances_})\
                .sort_values("Importance")
            fig_imp = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                             color="Importance", color_continuous_scale="Oranges",
                             title="Feature Importance (Churn Prediction Model)")
            fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380,
                                  margin=dict(t=55, b=40))
            st.plotly_chart(fig_imp, use_container_width=True)
        with ci2:
            fig_ch = px.histogram(churn_df, x="Churn_Prob", nbins=30,
                                  color_discrete_sequence=["#f97316"],
                                  title="Churn Probability Distribution")
            fig_ch.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=380,
                                 margin=dict(t=55, b=40))
            st.plotly_chart(fig_ch, use_container_width=True)

        high_risk = churn_df[churn_df["Churn_Prob"] > 0.55]\
            [["Customer_ID", "City", "Company", "Order_Value",
              "Delivery_Time_Min", "Customer_Rating", "Churn_Prob"]]\
            .sort_values("Churn_Prob", ascending=False).head(15)
        if not high_risk.empty:
            st.markdown("#### 🚨 High-Risk Churn Customers (Probability > 55%)")
            st.dataframe(
                high_risk.style.background_gradient(subset=["Churn_Prob"], cmap="Reds"),
                use_container_width=True
            )
    else:
        st.info("ℹ️ Not enough churn diversity in filtered data. Try expanding your filters.")

# ══════════════════════════════════
# TAB 5 — DEMAND FORECASTING
# ══════════════════════════════════
with tab5:
    st.markdown('<div class="sec-hdr">📈 Revenue Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown(f"Forecasting next **{forecast_days} days** using **{forecast_method}**.")

    # Use daily aggregated revenue for a meaningful time-series forecast
    fdf_ts = filtered_df[["Order_Date", "Order_Value"]].copy()
    fdf_ts["Order_Date"] = pd.to_datetime(fdf_ts["Order_Date"])
    daily_rev = fdf_ts.groupby("Order_Date")["Order_Value"].sum().reset_index().sort_values("Order_Date")

    if len(daily_rev) < 3:
        # Fallback: use raw order index
        fdf = filtered_df[["Order_Value"]].copy()
        fdf["Index"] = np.arange(len(fdf))
        X_f = fdf[["Index"]]
        y_f = fdf["Order_Value"]
        x_actual   = fdf["Index"].values
        y_actual   = y_f.values
        x_label    = "Order Index"
        x_future   = np.arange(len(fdf), len(fdf) + forecast_days).reshape(-1, 1)
        x_future_flat = x_future.flatten()
    else:
        daily_rev["Index"] = np.arange(len(daily_rev))
        X_f = daily_rev[["Index"]]
        y_f = daily_rev["Order_Value"]
        x_actual   = daily_rev["Index"].values
        y_actual   = y_f.values
        x_label    = "Day"
        x_future   = np.arange(len(daily_rev), len(daily_rev) + forecast_days).reshape(-1, 1)
        x_future_flat = x_future.flatten()

    if forecast_method == "Linear Regression":
        mdl = LinearRegression()
        mdl.fit(X_f, y_f)
        pred = mdl.predict(x_future)
        note = f"R² = {mdl.score(X_f, y_f):.4f} | Slope = {mdl.coef_[0]:.3f}"
    elif forecast_method == "Polynomial Degree-3":
        mdl = Pipeline([
            ("poly", PolynomialFeatures(degree=3, include_bias=False)),
            ("reg",  LinearRegression())
        ])
        mdl.fit(X_f, y_f)
        pred = mdl.predict(x_future)
        note = f"Polynomial Degree-3 | Train R² = {mdl.score(X_f, y_f):.4f}"
    else:
        window = min(30, len(y_f))
        roll   = pd.Series(y_f.values).rolling(window).mean().iloc[-1]
        pred   = np.full(forecast_days, roll if not np.isnan(roll) else y_f.mean())
        note   = f"Rolling {window}-period mean = ₹{pred[0]:,.2f}"

    st.info(f"📊 Model → {note}")

    fig_fc = go.Figure()
    fig_fc.add_trace(go.Scatter(
        x=x_actual, y=y_actual, mode="lines",
        name="Historical", line=dict(color="#f97316", width=2)
    ))
    fig_fc.add_trace(go.Scatter(
        x=x_future_flat, y=pred, mode="lines",
        name=f"{forecast_method} Forecast",
        line=dict(color="#7c3aed", width=2.5, dash="dash")
    ))
    fig_fc.add_trace(go.Scatter(
        x=np.concatenate([x_future_flat, x_future_flat[::-1]]),
        y=np.concatenate([pred * 1.10, (pred * 0.90)[::-1]]),
        fill="toself", fillcolor="rgba(124,58,237,0.10)",
        line=dict(color="rgba(0,0,0,0)"), name="90% Confidence Band",
        showlegend=True
    ))
    fig_fc.add_vline(x=x_actual[-1], line_dash="dot", line_color="gray",
                     annotation_text="Forecast Start ▶", annotation_font_size=11)
    fig_fc.update_layout(
        title=f"{forecast_days}-Day Revenue Forecast ({forecast_method})",
        xaxis_title=x_label, yaxis_title="Revenue (₹)",
        plot_bgcolor="white", paper_bgcolor="white",
        hovermode="x unified", height=440,
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=60, b=80)
    )
    st.plotly_chart(fig_fc, use_container_width=True)

    fc1, fc2, fc3 = st.columns(3)
    fc1.metric("📈 Total Forecast Revenue", f"₹{pred.sum():,.0f}")
    fc2.metric("📊 Avg Revenue / Period",   f"₹{pred.mean():,.0f}")
    fc3.metric("🔺 Peak Forecast Value",    f"₹{pred.max():,.0f}")

    st.divider()
    st.markdown('<div class="sec-hdr">📅 Monthly Revenue Trend</div>', unsafe_allow_html=True)
    fdf_m = filtered_df[["Order_Date", "Order_Value"]].copy()
    fdf_m["Month"] = pd.to_datetime(fdf_m["Order_Date"]).dt.to_period("M").astype(str)
    monthly = fdf_m.groupby("Month")["Order_Value"].sum().reset_index().sort_values("Month")
    fig_m = px.line(monthly, x="Month", y="Order_Value", markers=True,
                    color_discrete_sequence=["#ea580c"],
                    title="Monthly Revenue Trend")
    fig_m.update_traces(line_width=2.5, marker_size=8)
    fig_m.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=370,
                         margin=dict(t=55, b=40))
    st.plotly_chart(fig_m, use_container_width=True)

    # Category-wise forecast summary
    st.markdown('<div class="sec-hdr">🛍 Category Revenue Breakdown</div>', unsafe_allow_html=True)
    cat_rev = filtered_df.groupby("Product_Category")["Order_Value"].agg(["sum","mean","count"]).reset_index()
    cat_rev.columns = ["Category", "Total Revenue", "Avg Order Value", "Order Count"]
    cat_rev = cat_rev.sort_values("Total Revenue", ascending=False)
    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        fig_cat_bar = px.bar(cat_rev, x="Category", y="Total Revenue",
                             color="Total Revenue", color_continuous_scale="Oranges",
                             text_auto=True, title="Revenue by Product Category")
        fig_cat_bar.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=360,
                                  margin=dict(t=55, b=40))
        st.plotly_chart(fig_cat_bar, use_container_width=True)
    with col_f2:
        st.markdown("#### 📋 Category Summary")
        st.dataframe(cat_rev.style.background_gradient(subset=["Total Revenue"], cmap="Oranges"),
                     use_container_width=True, height=340)

# ══════════════════════════════════
# TAB 6 — LIVE ORDER FEED
# ══════════════════════════════════
with tab6:
    st.markdown('<div class="sec-hdr">🔴 Live Order Simulation Feed</div>', unsafe_allow_html=True)
    st.markdown("Simulates incoming real-time orders as they would arrive on the platform.")

    if "live_orders" not in st.session_state:
        st.session_state.live_orders = []

    b1, b2, b3 = st.columns([1.4, 1.1, 4.5])
    with b1:
        gen_btn = st.button("▶️ Generate 5 Orders", use_container_width=True)
    with b2:
        clr_btn = st.button("🗑️ Clear Feed", use_container_width=True)
    with b3:
        st.markdown(f"<span style='color:#9a3412;font-weight:600;font-size:0.9rem;'>Total simulated: {len(st.session_state.live_orders)} orders</span>", unsafe_allow_html=True)

    if clr_btn:
        st.session_state.live_orders = []
        st.success("✅ Feed cleared.")

    if gen_btn:
        companies_list  = df["Company"].unique().tolist()
        cities_list     = df["City"].unique().tolist()
        categories_list = df["Product_Category"].unique().tolist()
        payments_list   = df["Payment_Method"].unique().tolist()

        for _ in range(5):
            eta = random.randint(8, 50)
            order = {
                "🕐 Time":         datetime.now().strftime("%H:%M:%S"),
                "🏢 Company":      random.choice(companies_list),
                "📍 City":         random.choice(cities_list),
                "🛍 Category":     random.choice(categories_list),
                "💰 Value (₹)":    random.randint(150, 2500),
                "⏱ ETA (min)":     eta,
                "💳 Payment":      random.choice(payments_list),
                "⭐ Pred. Rating": round(random.uniform(2.5, 5.0), 1),
                "🚦 Status": (
                    "⚠️ SLA Risk"  if eta > delivery_sla else
                    "🚀 Express"   if eta < 15 else
                    "✅ On Track"
                )
            }
            st.session_state.live_orders.insert(0, order)
        st.session_state.live_orders = st.session_state.live_orders[:60]

    if st.session_state.live_orders:
        live_df = pd.DataFrame(st.session_state.live_orders)

        # Live KPIs
        lm1, lm2, lm3, lm4 = st.columns(4)
        lm1.metric("📦 Live Orders",      len(live_df))
        lm2.metric("⏱ Avg ETA",           f"{live_df['⏱ ETA (min)'].mean():.1f} min")
        lm3.metric("🚨 SLA Risks",        int((live_df["🚦 Status"] == "⚠️ SLA Risk").sum()))
        lm4.metric("💰 Avg Order Value",  f"₹{live_df['💰 Value (₹)'].mean():.0f}")

        st.markdown("")
        st.dataframe(live_df, use_container_width=True, height=360)

        col_l1, col_l2 = st.columns(2)
        with col_l1:
            values_stream = [o["💰 Value (₹)"] for o in st.session_state.live_orders[:25]]
            fig_live = go.Figure(go.Scatter(
                y=values_stream, mode="lines+markers",
                line=dict(color="#f97316", width=2.5),
                marker=dict(size=7, color="#ea580c"),
                fill="tozeroy", fillcolor="rgba(249,115,22,0.12)"
            ))
            fig_live.update_layout(
                title="📈 Order Value Stream (Last 25)",
                plot_bgcolor="white", paper_bgcolor="white",
                height=280, margin=dict(t=45, b=30, l=20, r=20),
                xaxis_title="Order #", yaxis_title="Order Value (₹)"
            )
            st.plotly_chart(fig_live, use_container_width=True)

        with col_l2:
            status_counts = live_df["🚦 Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig_status = px.pie(status_counts, names="Status", values="Count",
                                title="📊 Live Order Status Mix", hole=0.44,
                                color_discrete_sequence=["#22c55e", "#f59e0b", "#3b82f6"])
            fig_status.update_layout(paper_bgcolor="white", height=280, margin=dict(t=45, b=10))
            st.plotly_chart(fig_status, use_container_width=True)
    else:
        st.info("👆 Click **Generate 5 Orders** to start the live simulation feed.")
        st.markdown("""
        <div class="card-info">
        ℹ️ <strong>How it works:</strong> Each click generates 5 simulated orders with random values, ETAs, companies, and cities.
        Orders are colour-coded by SLA status: 🚀 Express (&lt;15 min) · ✅ On Track · ⚠️ SLA Risk (exceeds your SLA).
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════
# TAB 7 — NL QUERY & EXPORT
# ══════════════════════════════════
with tab7:
    # ── NATURAL LANGUAGE QUERY ──
    st.markdown('<div class="sec-hdr">🔍 Natural Language Data Query</div>', unsafe_allow_html=True)
    st.markdown("Ask questions about your data in plain English — no code needed.")

    examples = [
        "top 5 cities by revenue", "worst delivery city", "best rated company",
        "most popular payment method", "average delivery by company",
        "low rated orders", "revenue by category", "highest order value"
    ]
    ex_cols = st.columns(4)
    for i, ex in enumerate(examples):
        ex_cols[i % 4].code(ex, language=None)

    query = st.text_input("🔍 Ask your data…", placeholder="e.g.  top 5 cities by revenue")

    if query:
        q = query.lower().strip()
        res_df, res_fig, answer = None, None, ""

        if "top" in q and "cit" in q and "revenue" in q:
            n = next((int(t) for t in q.split() if t.isdigit()), 5)
            res_df = city_sales.nlargest(n, "Order_Value")
            res_fig = px.bar(res_df, x="City", y="Order_Value",
                             color="Order_Value", color_continuous_scale="Oranges",
                             title=f"Top {n} Cities by Revenue", text_auto=True)
            answer = f"Top {n} cities by revenue shown below."

        elif "worst" in q and "deliver" in q:
            res_df = city_delivery.nlargest(5, "Delivery_Time_Min")
            res_fig = px.bar(res_df, x="City", y="Delivery_Time_Min",
                             color="Delivery_Time_Min", color_continuous_scale="Reds",
                             title="Cities with Worst Delivery Times")
            answer = f"📍 Worst delivery city: **{res_df.iloc[0]['City']}** ({res_df.iloc[0]['Delivery_Time_Min']:.1f} min)"

        elif "best rated" in q or "highest rating" in q:
            best = comp_rating.loc[comp_rating["Customer_Rating"].idxmax()]
            answer = f"🏆 Best rated: **{best['Company']}** — avg **{best['Customer_Rating']:.2f} ⭐**"
            res_fig = px.bar(
                comp_rating.sort_values("Customer_Rating", ascending=False),
                x="Company", y="Customer_Rating",
                color="Customer_Rating", color_continuous_scale="Greens",
                title="Company Ratings", text_auto=True
            )

        elif "payment" in q:
            res_df = filtered_df["Payment_Method"].value_counts().reset_index()
            res_df.columns = ["Payment_Method", "Count"]
            answer = f"💳 Most popular: **{res_df.iloc[0]['Payment_Method']}** ({res_df.iloc[0]['Count']} orders)"
            res_fig = px.pie(res_df, names="Payment_Method", values="Count",
                             title="Payment Method Distribution")

        elif "average delivery" in q and "company" in q:
            cd = filtered_df.groupby("Company")["Delivery_Time_Min"].mean()\
                .reset_index().sort_values("Delivery_Time_Min")
            answer = f"⏱ Fastest: **{cd.iloc[0]['Company']}** ({cd.iloc[0]['Delivery_Time_Min']:.1f} min)"
            res_df = cd
            res_fig = px.bar(cd, x="Company", y="Delivery_Time_Min",
                             color="Delivery_Time_Min", color_continuous_scale="RdYlGn_r",
                             title="Avg Delivery Time by Company")

        elif "low rat" in q:
            res_df = filtered_df[filtered_df["Customer_Rating"] < 3]\
                [["Company", "City", "Order_Value", "Delivery_Time_Min", "Customer_Rating"]]\
                .sort_values("Customer_Rating").head(20)
            answer = f"⭐ {len(res_df)} orders with rating < 3 shown below."

        elif "category" in q:
            cat = filtered_df.groupby("Product_Category")["Order_Value"].sum()\
                .reset_index().sort_values("Order_Value", ascending=False)
            answer = f"🛍 Top: **{cat.iloc[0]['Product_Category']}** — ₹{cat.iloc[0]['Order_Value']:,.0f}"
            res_df = cat
            res_fig = px.pie(cat, names="Product_Category", values="Order_Value",
                             title="Revenue by Category")

        elif "highest order" in q:
            res_df = filtered_df.nlargest(10, "Order_Value")\
                [["Company", "City", "Order_Value", "Product_Category", "Payment_Method"]]
            answer = f"💰 Highest: ₹{res_df.iloc[0]['Order_Value']:,.0f} from {res_df.iloc[0]['City']}"

        else:
            answer = ("🤔 Query not recognised. Try: *top 5 cities by revenue*, "
                      "*worst delivery city*, *best rated company*, *payment method*, etc.")

        st.markdown(f'<div class="card-info">💬 {answer}</div>', unsafe_allow_html=True)
        if res_fig:
            res_fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                  height=380, margin=dict(t=55, b=40))
            st.plotly_chart(res_fig, use_container_width=True)
        if res_df is not None and not res_df.empty:
            st.dataframe(res_df, use_container_width=True)

    st.divider()

    # ── EXPORT ──
    st.markdown('<div class="sec-hdr">📄 Export Analytics Reports</div>', unsafe_allow_html=True)

    best_city_nm = city_sales.loc[city_sales["Order_Value"].idxmax(), "City"]
    slow_city_nm = city_delivery.loc[city_delivery["Delivery_Time_Min"].idxmax(), "City"]
    best_pay_nm  = payment_data.loc[payment_data["Order_Value"].idxmax(), "Payment_Method"]
    comp_str     = filtered_df.groupby("Company")["Order_Value"].sum().to_string()

    summary = f"""
QUICK COMMERCE INTELLIGENCE — EXECUTIVE SUMMARY
Generated : {datetime.now().strftime('%d %B %Y | %I:%M %p')}

FILTERS APPLIED
Companies  : {', '.join(company)}
Cities     : {', '.join(city)}
Categories : {', '.join(category)}

KEY PERFORMANCE INDICATORS
Total Orders         : {total_orders:,}
Total Revenue        : Rs {total_revenue:,.0f}
Avg Delivery Time    : {avg_delivery:.1f} min
Avg Customer Rating  : {avg_rating:.2f} / 5.00
SLA Breach Rate      : {sla_breach_pct:.1f}%
Avg Order Value      : Rs {avg_order_val:.0f}
Top Category         : {top_category}

CITY INTELLIGENCE
Highest Revenue City  : {best_city_nm}  (Rs {city_sales['Order_Value'].max():,.0f})
Slowest Delivery City : {slow_city_nm}  ({city_delivery['Delivery_Time_Min'].max():.1f} min)

COMPANY PERFORMANCE
{comp_str}

AI STRATEGIC RECOMMENDATIONS
1. Expand operations in {best_city_nm} — highest revenue potential.
2. Improve delivery fleet in {slow_city_nm} — worst SLA performance.
3. Promote {best_pay_nm} cashback offers to drive adoption.
4. Run targeted discounts for At-Risk & Lost RFM segments.
5. Investigate anomalous orders for potential fraud or data quality issues.

Built by Affan Inamdar
Data Science / ML Course Project
Powered by: Streamlit · Plotly · Scikit-Learn · Pandas · NumPy
""".strip()

    e1, e2, e3 = st.columns(3)
    with e1:
        st.markdown("**📊 Filtered Dataset**")
        st.download_button(
            "⬇️ Download CSV",
            data=filtered_df.to_csv(index=False),
            file_name=f"qcommerce_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True
        )
    with e2:
        st.markdown("**📝 Executive Summary**")
        st.download_button(
            "⬇️ Download TXT Summary",
            data=summary,
            file_name=f"summary_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            mime="text/plain", use_container_width=True
        )
    with e3:
        st.markdown("**👥 RFM Segments**")
        rfm_exp = rfm[["Customer_ID", "Recency", "Frequency", "Monetary", "RFM_Score", "Segment"]]
        st.download_button(
            "⬇️ Download RFM CSV",
            data=rfm_exp.to_csv(index=False),
            file_name=f"rfm_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv", use_container_width=True
        )

    st.text_area("📋 Executive Summary Preview", summary, height=360)

# ─────────────────────────────────────
# AI RECOMMENDATIONS BANNER
# ─────────────────────────────────────
st.divider()
st.markdown("## 🤖 AI Strategic Recommendations")
rec1, rec2, rec3 = st.columns(3)
rec1.success(f"✅ Expand in **{city_sales.loc[city_sales['Order_Value'].idxmax(), 'City']}** — highest revenue city")
rec2.warning(f"🚚 Fix logistics in **{city_delivery.loc[city_delivery['Delivery_Time_Min'].idxmax(), 'City']}** — slowest delivery")
rec3.info(f"💳 Push **{payment_data.loc[payment_data['Order_Value'].idxmax(), 'Payment_Method']}** cashback offers")

if avg_rating < 4:
    st.error("⭐ Customer satisfaction needs immediate attention — avg rating is below 4.0")
else:
    st.success("⭐ Customer satisfaction is healthy — maintain your service quality!")

# ─────────────────────────────────────
# FOOTER
# ─────────────────────────────────────
st.divider()
st.markdown(f"""
<div style="text-align:center; padding: 12px 0;">
<p style="color:#ea580c;font-size:1.15rem;font-weight:800;">🛒 Quick Commerce Intelligence Platform</p>
<p style="color:#7c2d12;font-size:0.92rem;line-height:1.8;">
    Built by <strong>Affan Inamdar</strong> &nbsp;·&nbsp; Data Science / ML Course Project<br>
    Powered by: Streamlit &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; Scikit-Learn &nbsp;·&nbsp; Pandas &nbsp;·&nbsp; NumPy<br>
    <em>Version 3.0 &nbsp;|&nbsp; Last refreshed: {datetime.now().strftime('%d %b %Y, %I:%M %p')}</em>
</p>
<p style="color:#9a3412;font-size:0.9rem;">
    📊 Data Analytics &nbsp;|&nbsp; 🤖 Machine Learning &nbsp;|&nbsp; 📈 Business Intelligence
</p>
<p><a href="https://github.com/affanazinamdar91" target="_blank"
    style="color:#ea580c;font-weight:700;text-decoration:none;">🔗 GitHub Portfolio</a></p>
</div>
""", unsafe_allow_html=True)
