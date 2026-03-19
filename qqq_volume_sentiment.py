"""
QQQ Volume Sentiment Indicator
Inspired by McClellan Financial Publications' weekly chart analysis.

For large ETFs like QQQ, volume acts as an inverse sentiment indicator:
  - High volume spikes → bottoming conditions (fear/capitulation)
  - Low volume periods → topping conditions (complacency)

This tool normalizes QQQ daily volume by shares outstanding to allow
apples-to-apples comparison across different time periods.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(page_title="QQQ Volume Sentiment Indicator", layout="wide")

st.title("📊 QQQ Volume as an Inverse Sentiment Indicator")
st.caption(
    "High volume days tend to mark bottoms (fear/capitulation), "
    "while low volume days often signal tops (complacency). "
    "Inspired by McClellan Financial Publications."
)

# ── Sidebar Controls ─────────────────────────────────────────────────────────
st.sidebar.header("Settings")

lookback_months = st.sidebar.slider(
    "Lookback (months)", min_value=6, max_value=60, value=24, step=3
)

ma_period = st.sidebar.slider(
    "Volume Moving Average Period", min_value=5, max_value=50, value=10, step=1
)

normalize_volume = st.sidebar.checkbox(
    "Normalize Volume by Shares Outstanding", value=True,
    help="Divides daily volume by total shares outstanding to allow comparison across time periods."
)

st.sidebar.markdown("---")
st.sidebar.subheader("Extreme Detection")

bb_period = st.sidebar.slider("Bollinger Period (hidden, powers z-scores)", 20, 100, 50, 5)
bb_std = 1.0

highlight_extremes = st.sidebar.checkbox("Highlight Extreme Volume Days", value=True)
extreme_high_z = st.sidebar.slider(
    "High Volume Threshold (z-score)", 1.0, 4.0, 2.0, 0.25,
    help="Volume z-score above this → bottom signal (red). Higher = fewer, more reliable signals."
)
extreme_low_z = st.sidebar.slider(
    "Low Volume Threshold (z-score)", -4.0, -0.5, -1.0, 0.25,
    help="Volume z-score below this → top signal (orange). Less reliable than bottoms — filter out holidays."
)

vol_bar_opacity = st.sidebar.slider(
    "Volume Bar Opacity", 0.1, 0.8, 0.25, 0.05,
    help="Lower values keep the volume bars from obscuring the price action."
)

# ── Data Fetch ───────────────────────────────────────────────────────────────
end_date = datetime.today()
start_date = end_date - timedelta(days=lookback_months * 30 + 90)  # extra for BB warmup

@st.cache_data(ttl=3600, show_spinner="Fetching QQQ data…")
def fetch_data(start, end):
    qqq = yf.Ticker("QQQ")
    hist = qqq.history(start=start, end=end, auto_adjust=True)
    info = qqq.info
    shares_outstanding = info.get("sharesOutstanding", None)
    return hist, shares_outstanding

try:
    df, shares_outstanding = fetch_data(start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d"))
except Exception as e:
    st.error(f"Failed to fetch data: {e}")
    st.stop()

if df.empty:
    st.warning("No data returned. Try a different lookback period.")
    st.stop()

# ── Volume Processing ────────────────────────────────────────────────────────
if normalize_volume and shares_outstanding:
    df["Vol_Display"] = df["Volume"] / shares_outstanding * 100  # as percentage
    vol_label = "Volume (% of Shares Outstanding)"
elif normalize_volume and not shares_outstanding:
    st.sidebar.warning("Shares outstanding not available — showing raw volume.")
    df["Vol_Display"] = df["Volume"]
    vol_label = "Volume (shares)"
else:
    df["Vol_Display"] = df["Volume"]
    vol_label = "Volume (shares)"

# Moving average
df["Vol_MA"] = df["Vol_Display"].rolling(window=ma_period).mean()

# Bollinger Bands on volume (hidden but powers the z-score math)
df["Vol_BB_Mid"] = df["Vol_Display"].rolling(window=bb_period).mean()
df["Vol_BB_Std"] = df["Vol_Display"].rolling(window=bb_period).std()
df["Vol_BB_Upper"] = df["Vol_BB_Mid"] + bb_std * df["Vol_BB_Std"]
df["Vol_BB_Lower"] = (df["Vol_BB_Mid"] - bb_std * df["Vol_BB_Std"]).clip(lower=0)

# Z-score for extreme detection
df["Vol_Z"] = (df["Vol_Display"] - df["Vol_BB_Mid"]) / df["Vol_BB_Std"]

# Trim warmup
trim_date = end_date - timedelta(days=lookback_months * 30)
df = df.loc[df.index >= pd.Timestamp(trim_date)]

# ── Identify Extremes ────────────────────────────────────────────────────────
df["Is_High"] = df["Vol_Z"] >= extreme_high_z
df["Is_Low"] = df["Vol_Z"] <= extreme_low_z

# Color volume bars
colors = []
for _, row in df.iterrows():
    if highlight_extremes and row["Is_High"]:
        colors.append("rgba(220, 50, 50, 0.90)")   # red – high vol / bottom
    elif highlight_extremes and row["Is_Low"]:
        colors.append("rgba(255, 165, 0, 0.90)")    # orange – low vol / top
    else:
        colors.append(f"rgba(100, 140, 200, {vol_bar_opacity})")

# ── Build Overlaid Chart ─────────────────────────────────────────────────────
fig = go.Figure()

# Volume bars on secondary y-axis (drawn first so they sit behind price)
fig.add_trace(
    go.Bar(
        x=df.index, y=df["Vol_Display"], name="Volume",
        marker_color=colors, yaxis="y2", opacity=0.9,
    )
)

# Volume MA on secondary y-axis
fig.add_trace(
    go.Scatter(
        x=df.index, y=df["Vol_MA"],
        mode="lines", name=f"Vol {ma_period}MA",
        line=dict(color="rgba(30, 136, 229, 0.5)", width=1.5),
        yaxis="y2",
    )
)

# Candlestick price on primary y-axis
fig.add_trace(
    go.Candlestick(
        x=df.index, open=df["Open"], high=df["High"],
        low=df["Low"], close=df["Close"], name="QQQ",
        increasing_line_color="#26a69a", decreasing_line_color="#ef5350",
    )
)

# Signal markers on price
if highlight_extremes:
    highs = df[df["Is_High"]]
    lows = df[df["Is_Low"]]
    if not highs.empty:
        fig.add_trace(
            go.Scatter(
                x=highs.index, y=highs["Low"] * 0.993,
                mode="markers", name="High Vol (Bottom Signal)",
                marker=dict(symbol="triangle-up", size=11, color="red",
                            line=dict(width=1, color="white")),
            )
        )
    if not lows.empty:
        fig.add_trace(
            go.Scatter(
                x=lows.index, y=lows["High"] * 1.007,
                mode="markers", name="Low Vol (Top Signal)",
                marker=dict(symbol="triangle-down", size=11, color="orange",
                            line=dict(width=1, color="white")),
            )
        )

# Layout with dual y-axes
fig.update_layout(
    height=700,
    template="plotly_dark",
    title=dict(text="QQQ Price with Volume Sentiment Overlay", font=dict(size=14)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5, font=dict(size=11)),
    xaxis=dict(title="Date", rangeslider=dict(visible=False)),
    yaxis=dict(
        title="Price ($)",
        side="left",
        showgrid=True,
        gridcolor="rgba(80,103,132,0.3)",
    ),
    yaxis2=dict(
        title=vol_label,
        side="right",
        overlaying="y",
        showgrid=False,
        range=[0, df["Vol_Display"].max() * 4],  # scale down bars to ~25% of chart height
    ),
    margin=dict(l=60, r=60, t=50, b=30),
    hovermode="x unified",
)

st.plotly_chart(fig, use_container_width=True)

# ── Summary Table of Recent Extremes ─────────────────────────────────────────
st.subheader("Recent Extreme Volume Days")

col1, col2 = st.columns(2)
with col1:
    st.markdown("🔴 **High Volume Days (Potential Bottoms)**")
    recent_highs = df[df["Is_High"]][["Close", "Vol_Display", "Vol_Z"]].tail(10).sort_index(ascending=False)
    recent_highs.columns = ["Close", vol_label, "Z-Score"]
    if not recent_highs.empty:
        st.dataframe(recent_highs.style.format({"Close": "${:.2f}", vol_label: "{:.4f}" if normalize_volume else "{:,.0f}", "Z-Score": "{:.2f}"}))
    else:
        st.info("No high-volume extremes in this window.")

with col2:
    st.markdown("🟠 **Low Volume Days (Potential Tops)**")
    st.caption("⚠️ Less reliable than bottom signals — filter out holidays and half-days.")
    recent_lows = df[df["Is_Low"]][["Close", "Vol_Display", "Vol_Z"]].tail(10).sort_index(ascending=False)
    recent_lows.columns = ["Close", vol_label, "Z-Score"]
    if not recent_lows.empty:
        st.dataframe(recent_lows.style.format({"Close": "${:.2f}", vol_label: "{:.4f}" if normalize_volume else "{:,.0f}", "Z-Score": "{:.2f}"}))
    else:
        st.info("No low-volume extremes in this window.")

# ── Methodology ──────────────────────────────────────────────────────────────
with st.expander("📖 How This Works"):
    st.markdown("""
**QQQ Volume as an Inverse Sentiment Indicator**

For large, highly liquid ETFs like QQQ, trading volume functions as a surprisingly reliable
inverse sentiment gauge:

- **High volume spikes** typically occur during sharp selloffs driven by fear and capitulation.
  These spikes often mark significant market bottoms because they reflect traders rushing to
  hedge portfolio risk or exit positions via the ETF rather than selling individual stocks.

- **Low volume periods** tend to coincide with market tops. When sentiment is positive and
  complacent, traders have less need for the hedging and liquidity functions that ETFs provide,
  so volume dries up. **These signals are less reliable** — volume can be low due to holidays,
  summer trading lulls, or simply quiet news days, so always apply a mental filter.

**Why Normalize?**

Raw QQQ volume is not directly comparable over long periods because the number of shares
outstanding changes over time (Invesco issues or redeems shares to keep the price near NAV).
Expressing daily volume as a percentage of shares outstanding creates a more consistent measure.

**Key Settings:**
- The **10-day moving average** (adjustable) smooths daily noise and reveals the underlying volume trend.
- **Bollinger Bands** on volume (hidden) power the z-score math that identifies statistically extreme days.
- **Z-score thresholds** let you tune how aggressive the extreme-detection is.
- **Volume Bar Opacity** controls how prominent the bars are behind the price — lower keeps it cleaner.

**Caveats:**
- Low volume on holidays is *not* a bearish signal — apply a mental filter on known half-days.
- SPY volume can be skewed by quarterly options/futures expiration; QQQ is less affected.
- This is one tool among many — always use it in conjunction with other analysis.
""")
