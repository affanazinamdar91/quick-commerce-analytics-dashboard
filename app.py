import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np
from sklearn.linear_model import LinearRegression
from datetime import datetime

px.defaults.template = "plotly_white"

# -----------------------------------
# PAGE CONFIG
# -----------------------------------
st.set_page_config(
    page_title="Quick Commerce Intelligence Dashboard",
    layout="wide",
    page_icon="🛒"
)

# -----------------------------------
# PREMIUM ORANGE UI
# -----------------------------------
st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #fff7ed, #ffedd5, #fed7aa);
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f97316, #ea580c);
    border-right: 4px solid #fb923c;
}

section[data-testid="stSidebar"] * {
    color: white !important;
    font-weight: 600;
}

/* Metric Cards */
div[data-testid="metric-container"] {
    background: white;
    border-left: 8px solid #f97316;
    padding: 18px;
    border-radius: 18px;
    box-shadow: 0px 8px 20px rgba(249,115,22,0.18);
}

/* Charts */
.stPlotlyChart {
    background: white;
    padding: 15px;
    border-radius: 18px;
    box-shadow: 0px 8px 18px rgba(249,115,22,0.12);
}

/* Buttons */
.stDownloadButton button {
    background: linear-gradient(90deg, #f97316, #ea580c);
    color: white !important;
    border-radius: 12px;
    font-weight: bold;
    border: none;
    padding: 12px 20px;
}

h1 {
    color: #c2410c !important;
    font-size: 54px !important;
}

h2, h3 {
    color: #9a3412 !important;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------------
# HEADER
# -----------------------------------
st.markdown("""
# 🛒 Quick Commerce Intelligence Platform
### Advanced AI Analytics for Blinkit / Swiggy / Zepto Operations
""")

st.caption(f"Last Updated: {datetime.now().strftime('%d %b %Y | %I:%M %p')}")

# -----------------------------------
# LOAD DATA
# -----------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("sales_data.csv")

try:
    df = load_data()
except FileNotFoundError:
    st.error("sales_data.csv file not found.")
    st.stop()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# -----------------------------------
# VALIDATION
# -----------------------------------
required_columns = [
    "Company", "City", "Customer_Age", "Order_Value",
    "Delivery_Time_Min", "Product_Category",
    "Payment_Method", "Customer_Rating",
    "Discount_Applied"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    st.error(f"Missing Columns: {missing}")
    st.stop()

# -----------------------------------
# SIDEBAR FILTERS
# -----------------------------------
st.sidebar.markdown("## 🔍 Live Filters")

company = st.sidebar.multiselect(
    "🏢 Company",
    df["Company"].unique(),
    default=df["Company"].unique()
)

city = st.sidebar.multiselect(
    "📍 City",
    df["City"].unique(),
    default=df["City"].unique()
)

category = st.sidebar.multiselect(
    "🛍 Product Category",
    df["Product_Category"].unique(),
    default=df["Product_Category"].unique()
)

filtered_df = df[
    (df["Company"].isin(company)) &
    (df["City"].isin(city)) &
    (df["Product_Category"].isin(category))
].copy()

if filtered_df.empty:
    st.warning("No data available for selected filters.")
    st.stop()

# -----------------------------------
# KPI SECTION
# -----------------------------------
total_orders = len(filtered_df)
total_revenue = filtered_df["Order_Value"].sum()
avg_delivery = filtered_df["Delivery_Time_Min"].mean()
avg_rating = filtered_df["Customer_Rating"].mean()

top_category = filtered_df.groupby("Product_Category")["Order_Value"].sum().idxmax()
efficiency = max(0, 100 - avg_delivery)

st.markdown("##  Executive Overview")

c1, c2, c3, c4, c5, c6 = st.columns(6)

c1.metric("📦 Orders", f"{total_orders:,}")
c2.metric("💰 Revenue", f"₹{total_revenue:,.0f}")
c3.metric("⏱ Delivery", f"{avg_delivery:.1f} min")
c4.metric("⭐ Rating", f"{avg_rating:.2f}")
c5.metric("⚡ Efficiency", f"{efficiency:.0f}%")
c6.metric("🔥 Top Category", top_category)

st.divider()

# -----------------------------------
# TABS
# -----------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Sales",
    "🚚 Delivery",
    "💳 Customer Insights",
    "🤖 AI Forecast"
])

# -----------------------------------
# SALES TAB
# -----------------------------------
with tab1:
    city_sales = filtered_df.groupby("City")["Order_Value"].sum().reset_index()

    fig1 = px.bar(
        city_sales,
        x="City",
        y="Order_Value",
        color="Order_Value",
        title="Revenue by City",
        text_auto=True
    )
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.pie(
        filtered_df,
        names="Product_Category",
        values="Order_Value",
        title="Revenue by Category"
    )
    st.plotly_chart(fig2, use_container_width=True)

    company_compare = filtered_df.groupby("Company")["Order_Value"].sum().reset_index()

    fig3 = px.bar(
        company_compare,
        x="Company",
        y="Order_Value",
        color="Company",
        title="Company Revenue Comparison"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Heatmap
    pivot = filtered_df.pivot_table(
    values="Order_Value",
    index="City",
    columns="Company",
    aggfunc="sum",
    fill_value=0
)

fig_heat = px.imshow(
    pivot,
    text_auto=".2s",
    title="Revenue Heatmap",
    aspect="auto",
    color_continuous_scale="Oranges"
)

fig_heat.update_layout(
    height=700,
    title_font_size=24,
    font=dict(size=16),
    margin=dict(l=50, r=50, t=80, b=50)
)

fig_heat.update_traces(
    textfont_size=14
)

st.plotly_chart(fig_heat, use_container_width=True)
st.info("The heatmap highlights high-revenue city-company combinations, helping identify expansion opportunities and operational focus areas.")
# -----------------------------------
# DELIVERY TAB
# -----------------------------------
with tab2:
    city_delivery = filtered_df.groupby("City")["Delivery_Time_Min"].mean().reset_index()

    fig4 = px.bar(
        city_delivery,
        x="City",
        y="Delivery_Time_Min",
        color="Delivery_Time_Min",
        title="Average Delivery Time"
    )
    st.plotly_chart(fig4, use_container_width=True)

    fig5 = px.scatter(
        filtered_df,
        x="Delivery_Time_Min",
        y="Customer_Rating",
        color="Company",
        title="Delivery Time vs Customer Rating"
    )
    st.plotly_chart(fig5, use_container_width=True)

# -----------------------------------
# CUSTOMER TAB
# -----------------------------------
with tab3:
    payment_data = filtered_df.groupby("Payment_Method")["Order_Value"].sum().reset_index()

    fig6 = px.bar(
        payment_data,
        x="Payment_Method",
        y="Order_Value",
        title="Revenue by Payment Method"
    )
    st.plotly_chart(fig6, use_container_width=True)

    customer_age = filtered_df.groupby("Customer_Age")["Order_Value"].sum().reset_index()

    fig7 = px.line(
        customer_age,
        x="Customer_Age",
        y="Order_Value",
        title="Customer Spending by Age"
    )
    st.plotly_chart(fig7, use_container_width=True)

# -----------------------------------
# FORECAST TAB
# -----------------------------------
# -----------------------------------
# FORECAST TAB
# -----------------------------------
with tab4:

    # Limit rows for forecasting
    forecast_data = filtered_df.head(365).copy()

    # Create daily dates safely
    forecast_data["Date"] = pd.date_range(
        start="2024-01-01",
        periods=len(forecast_data),
        freq="D"
    )

    monthly = forecast_data.groupby(
        forecast_data["Date"].dt.to_period("M")
    )["Order_Value"].sum().reset_index()

    monthly["Month"] = np.arange(len(monthly))

    X = monthly[["Month"]]
    y = monthly["Order_Value"]

    model = LinearRegression()
    model.fit(X, y)

    future = np.arange(len(monthly), len(monthly)+6).reshape(-1,1)
    pred = model.predict(future)

    forecast_df = pd.DataFrame({
        "Month": future.flatten(),
        "Predicted Revenue": pred
    })

    fig8 = px.line(
        forecast_df,
        x="Month",
        y="Predicted Revenue",
        title="6-Month Revenue Forecast",
        markers=True
    )

    st.plotly_chart(fig8, use_container_width=True)

# -----------------------------------
# AI RECOMMENDATIONS
# -----------------------------------
st.markdown("##  AI Strategic Recommendations")

best_city = city_sales.loc[city_sales["Order_Value"].idxmax(), "City"]
slow_city = city_delivery.loc[city_delivery["Delivery_Time_Min"].idxmax(), "City"]
best_payment = payment_data.loc[payment_data["Order_Value"].idxmax(), "Payment_Method"]

col1, col2, col3 = st.columns(3)

col1.success(f" Expand in **{best_city}**")
col2.warning(f"⚠ Improve logistics in **{slow_city}**")
col3.info(f"💳 Promote **{best_payment}** cashback offers")

if avg_rating < 4:
    st.error("Customer satisfaction needs improvement")
else:
    st.success("Customer satisfaction is healthy")

# -----------------------------------
# REPORT DOWNLOAD
# -----------------------------------
st.download_button(
    " Download Analytics Report",
    data=filtered_df.to_csv(index=False),
    file_name="quick_commerce_dashboard_report.csv",
    mime="text/csv"
)

# -----------------------------------
# FOOTER
# -----------------------------------
st.divider()

st.markdown("""
<center>
<h4>Built by Affan Inamdar</h4>
<p>AI-Powered Quick Commerce Intelligence Dashboard</p>
<p>
📊 Data Analytics | 🤖 Machine Learning | 📈 Business Intelligence
</p>
<p>
🔗 <a href="https://github.com/affanazinamdar91" target="_blank">
GitHub Portfolio
</a>
</p>
<p><b>Version 1.0 | Real-Time Analytics Dashboard</b></p>
</center>
""", unsafe_allow_html=True)