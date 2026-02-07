# create an interface that matches what is shown in UI.png
# use Dash to create the interface
# Dash Stocks Explorer — UI Skeleton (Fixed)
# ------------------------------------------------------------
# This file provides a polished Dash UI with callbacks that
# call placeholder functions you can implement later.
# Replace the stub functions at the bottom with your own
# data logic (prices, company info, trending lists).
#
# FIX: Removed html.Style (not a valid dash.html component).
#      Using inline styles via `style={...}` on components instead.
# ------------------------------------------------------------

from __future__ import annotations
import pandas as pd
from typing import List, Dict, Optional

from dash import (
    Dash, dcc, html, Input, Output, State
)
import plotly.graph_objects as go
import os
import datetime
from datetime import datetime, timedelta
import yfinance as yf

# ------------------------------------------------------------
# App setup
# ------------------------------------------------------------
APP_TITLE = "Stocks Explorer"
DEFAULT_TICKERS = [
    {"label": "Apple (AAPL)", "value": "AAPL"},
    {"label": "NVIDIA (NVDA)", "value": "NVDA"},
    {"label": "Tesla (TSLA)", "value": "TSLA"},
    {"label": "Microsoft (MSFT)", "value": "MSFT"},
    {"label": "Advanced Micro Devices (AMD)", "value": "AMD"},
]

tickers = [f.name.replace('.parquet', '') for f in os.scandir("price_history") if f.is_file()]


TIMEFRAMES = [
    ("1D", "1d"), ("5D", "5d"), ("1M", "1mo"), ("6M", "6mo"), ("1Y", "1y"), ("5Y", "5y"), ("Max", "max")
]

# Inline styles (replacement for the removed <style> block)
CARD_STYLE = {
    "border": "1px solid #e6e6e6",
    "borderRadius": 12,
    "padding": 12,
    "boxShadow": "0 1px 2px rgba(0,0,0,.03)",
}


app = Dash(__name__)
app.title = APP_TITLE

# ------------------------------------------------------------
# Helper: figure from price dataframe
# Expected df index is datetime; columns include at least "Close".
# Optionally: Open/High/Low/Volume for candlesticks.
# ------------------------------------------------------------

def extra_info_in_title(df_price: pd.DataFrame) -> Dict[str, str]:
    """
    Extract key statistics from price data for display in chart title.
    Returns a dictionary with keys like 'Max', 'Min', 'Current', etc.
    """
    if df_price is None or df_price.empty:
        return {}
    info = {}
    # Get the main price column (Close or Adj Close)
    price_col = None
    if "Close" in df_price.columns:
        price_col = "Close"
    elif "Adj Close" in df_price.columns:
        price_col = "Adj Close"
    elif len(df_price.columns) > 0:
        # Fallback to first numeric column
        price_col = df_price.columns[0]
    if price_col is None:
        print("No price column found")
        return {}
    
    try:
        price_data = df_price[price_col].dropna()
        if len(price_data) == 0:
            return {}

        print(f"Current price: {price_data.iloc[-1].values[0]}")
        # Current price (most recent)
        current_price = price_data.iloc[-1].values[0]
        info["Current"] = f"${current_price:.2f}"
        
        # Maximum price
        max_price = price_data.max().values[0]
        print(f"Max price: {max_price}")
        info["Max"] = f"${max_price:.2f}"
        
        # Minimum price
        min_price = price_data.min().values[0]
        info["Min"] = f"${min_price:.2f}"
        # Price change (current vs first)
        first_price = price_data.iloc[0].values[0]
        price_change = current_price - first_price
        price_change_pct = (price_change / first_price) * 100
        
        if price_change >= 0:
            info["Change"] = f"+${price_change:.2f} (+{price_change_pct:.1f}%)"
        else:
            info["Change"] = f"${price_change:.2f} ({price_change_pct:.1f}%)"
        print(f"info: {info}")
        # Volume info if available
        if "Volume" in df_price.columns:
            volume_data = df_price["Volume"].dropna()
            if len(volume_data) > 0:
                avg_volume = volume_data.mean().values[0]
                if avg_volume >= 1e9:
                    info["Avg Vol"] = f"{avg_volume/1e9:.1f}B"
                elif avg_volume >= 1e6:
                    info["Avg Vol"] = f"{avg_volume/1e6:.1f}M"
                else:
                    info["Avg Vol"] = f"{avg_volume/1e3:.1f}K"
        
        # Date range info
        if len(df_price.index) > 1:
            start_date = df_price.index[0].strftime("%b %Y")
            end_date = df_price.index[-1].strftime("%b %Y")
            if start_date != end_date:
                info["Period"] = f"{start_date} - {end_date}"
            else:
                info["Period"] = start_date
                
    except Exception as e:
        print(f"Error calculating extra info: {e}")
        return {}
    print(f"info: {info}")
    return info



def make_price_figure(df: Optional[pd.DataFrame], symbol: str, timeframe_label: str) -> go.Figure:
    fig = go.Figure()
    print("We are making the price figure")
    try:
        print(df.head())
    except Exception as e:
        print(f"Error: {e}")
    if df is None or df.empty:
        fig.update_layout(
            annotations=[dict(
                text="Connect your price data in get_price_df(...)",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#888")
            )],
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_white",
            title=f"{symbol} — {timeframe_label}"
        )
        return fig

    # If we have OHLC, draw candlestick; else line of Close
    # Handle both regular columns and MultiIndex columns from yfinance
    print(f"DataFrame columns type: {type(df.columns)}")
    print(f"DataFrame columns: {df.columns}")
    
    if isinstance(df.columns, pd.MultiIndex):
        # For MultiIndex columns, flatten to get the actual column names
        cols = {c[0].lower(): c for c in df.columns}
        print(f"MultiIndex detected, flattened columns: {cols}")
        print(f"df.columns: {df.columns}")
    else:
        # For regular string columns
        cols = {c.lower(): c for c in df.columns}
        print(f"Regular columns detected: {cols}")
    
    has_ohlc = all(k in cols for k in ("open", "high", "low", "close"))
    print(f"Has OHLC data: {has_ohlc}")

    if has_ohlc:
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df[cols["open"]], high=df[cols["high"]],
            low=df[cols["low"]], close=df[cols["close"]],
            name=symbol
        ))
    else:
        try:
            # Fallback to line on Close/Adj Close
            ycol = cols.get("close") or cols.get("adj close") or list(df.columns)[0]
            fig.add_trace(go.Scatter(x=df.index, y=df[ycol], name=symbol, mode="lines"))
        except Exception as e:
            print(f"Error in fallback plotting: {e}")
            # Try to plot the first available column as a last resort
            try:
                first_col = list(df.columns)[0]
                fig.add_trace(go.Scatter(x=df.index, y=df[first_col], name=symbol, mode="lines"))
            except Exception as e2:
                print(f"Final fallback failed: {e2}")

    # Volume as secondary (if present)
    if "volume" in cols:
        fig.add_trace(go.Bar(x=df.index, y=df[cols["volume"]], name="Volume", yaxis="y2", opacity=0.2))
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", showgrid=False))

    # Get extra info for title
    extra_info = extra_info_in_title(df)
    # Remove legend from the figure
    fig.update_layout(showlegend=False)
    
    # Build title with extra info
    title_parts = [f"{symbol} — {timeframe_label}"]

    if extra_info:
        # Add key statistics to title in smaller font
        info_text = " | ".join([f"{k}: {v}" for k, v in extra_info.items()])
        title_parts.append(f"<span style='font-size: 10px;'>{info_text}</span>")
    title = "<br>".join(title_parts)
    
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None, yaxis_title=None,
        title=title
    )
    return fig

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
app.layout = html.Div(
    [
        # Headera
        html.Div([
            html.H1(APP_TITLE, style={"margin": 0}),
            html.Div("Explore unfamiliar tickers with price history and quick context.",
                    style={"color": "#666"}),
        ], style={"display": "flex", "flexDirection": "column", "gap": 4, "marginBottom": 12}),

        # Search row
        html.Div([
            dcc.Dropdown(id="ticker", options=tickers, value="AAPL",
                         placeholder="Search tickers or companies…", style={"flex": 1}),
            html.Div(dcc.RadioItems(
                id="timeframe",
                options=[{"label": lab, "value": val} for lab, val in TIMEFRAMES],
                value="6mo",
                labelStyle={"display": "inline-block", "marginRight": 10}
            ), style={"flex": 1, "textAlign": "right"}),
        ], style={"display": "flex", "gap": 12, "alignItems": "center", "marginBottom": 8}),

        # Main grid
        html.Div([
            # Left: chart + company snapshot
            html.Div([
                html.Div([
                    dcc.Loading(dcc.Graph(id="price-chart", figure=make_price_figure(None, "—", "—"),
                                           config={"displaylogo": False}), type="default")
                ], style=CARD_STYLE),

                html.Div([
                    html.H3("Company Snapshot", style={"marginTop": 0}),
                    html.Div(id="company-name", style={"fontWeight": 600}),
                    html.Div(id="company-meta", style={"color": "#666", "marginBottom": 6}),
                    html.Div(id="company-cap", style={"marginBottom": 10}),
                    html.Div("About:", style={"fontWeight": 600, "marginTop": 6}),
                    html.Div(id="company-about", style={"whiteSpace": "pre-wrap"}),
                ], style=CARD_STYLE),
            ], style={"display": "grid", "gap": 12}),

            # Right: trending panels
            html.Div([
                html.Div([
                    html.H3("Trending on X (last 24h)", style={"marginTop": 0}),
                    html.Div(id="trending-x"),
                ], style=CARD_STYLE),
                html.Div([
                    html.H3("Trending on r/wallstreetbets (24–48h)", style={"marginTop": 0}),
                    html.Div(id="trending-wsb"),
                ], style=CARD_STYLE),
            ], style={"minWidth": 340, "display": "grid", "gap": 12}),
        ], style={"display": "grid", "gridTemplateColumns": "1fr 360px", "gap": 12}),
    ], style={"maxWidth": 1200, "margin": "0 auto", "padding": 14})

# ------------------------------------------------------------
# Callbacks — these call your functions (stubs below)
# ------------------------------------------------------------

@app.callback(
    Output("price-chart", "figure"),
    Input("ticker", "value"),
    Input("timeframe", "value"),
    prevent_initial_call=False,
)
def update_chart(symbol: str, timeframe: str):
    if not symbol:
        return make_price_figure(None, "—", "—")
    
    label = next((lab for lab, val in TIMEFRAMES if val == timeframe), timeframe)
    try:
        df = get_price_df(symbol, timeframe)  # <-- implement
    except Exception as e:
        print(f"Error loading data for {symbol}: {e}")
        df = None
    return make_price_figure(df, symbol, label)


@app.callback(
    Output("company-name", "children"),
    Output("company-meta", "children"),
    Output("company-cap", "children"),
    Output("company-about", "children"),
    Input("ticker", "value"),  # triggers when ticker changes
)
def update_company_info(symbol: str):
    if not symbol:
        return ("—", "Sector: —  •  Industry: —", "Market Cap: —", "Select a ticker to view company information.")
    
    try:
        info = get_company_snapshot(symbol)  # <-- implement
    except Exception as e:
        print(f"Error loading company info for {symbol}: {e}")
        info = None

    if not info:
        return (
            "—",
            "Sector: —  •  Industry: —",
            "Market Cap: —",
            "Connect your company data in get_company_snapshot(...).",
        )

    name = info.get("name") or symbol
    sector = info.get("sector", "—")
    industry = info.get("industry", "—")
    cap = info.get("marketCap")
    about = info.get("longBusinessSummary") or "—"

    def fmt_money(n):
        try:
            n = float(n)
            if n >= 1e12: return f"${n/1e12:.2f}T"
            if n >= 1e9:  return f"${n/1e9:.2f}B"
            if n >= 1e6:  return f"${n/1e6:.2f}M"
            return f"${n:,.0f}"
        except Exception:
            return "—"

    return (
        f"{name} ({symbol})",
        f"Sector: {sector}  •  Industry: {industry}",
        f"Market Cap: {fmt_money(cap)}",
        about,
    )


@app.callback(
    Output("trending-x", "children"),
    Input("price-chart", "figure"),
)
def update_trending_x(_):
    try:
        items = get_trending_x()  # <-- implement
    except Exception:
        items = None

    if not items:
        return html.Div(
            "Connect your X data in get_trending_x(...).",
            style={"color": "#777"}
        )

    # items: List[{symbol, name?, mentions?, sentiment?, changePct?}]
    rows = []
    for i, it in enumerate(items[:10], start=1):
        sym = it.get("symbol")
        name = it.get("name") or sym
        m = it.get("mentions")
        s = it.get("sentiment")
        ch = it.get("changePct")
        meta_bits = []
        if m is not None: meta_bits.append(f"mentions: {m}")
        if s is not None: meta_bits.append(f"sentiment: {s:+.2f}")
        if ch is not None: meta_bits.append(f"Δ {ch:+.1f}%")
        rows.append(html.Div(f"#{i} {sym} — {name}  {'  |  '.join(meta_bits)}"))
    return html.Div(rows, style={"display": "grid", "gap": 6})


@app.callback(
    Output("trending-wsb", "children"),
    Input("price-chart", "figure"),
)
def update_trending_wsb(_):
    try:
        items = get_trending_wsb()  # <-- implement
    except Exception:
        items = None

    if not items:
        return html.Div(
            "Connect your Reddit data in get_trending_wsb(...).",
            style={"color": "#777"}
        )

    rows = []
    for it in items[:10]:
        sym = it.get("symbol")
        name = it.get("name") or sym
        m = it.get("mentions")
        s = it.get("sentiment")
        ch = it.get("changePct")
        bits = [name]
        if m is not None: bits.append(f"mentions: {m}")
        if s is not None: bits.append(f"sentiment: {s:+.2f}")
        if ch is not None: bits.append(f"Δ {ch:+.1f}%")
        rows.append(html.Div(f"{sym} — {'  |  '.join(bits)}"))
    return html.Div(rows, style={"display": "grid", "gap": 6})

# ------------------------------------------------------------
# YOUR DATA FUNCTIONS — replace these stubs
# ------------------------------------------------------------

PRICE_HISTORY_DIR = "price_history"


def update_parquet_if_stale(symbol: str) -> bool:
    """
    Check the latest date in the parquet file and append any missing days from yfinance.
    Returns True if successful, False on error.
    """
    parquet_path = f"{PRICE_HISTORY_DIR}/{symbol}.parquet"
    today = datetime.now().date()
    
    try:
        df_existing = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        return False  # No file to update
    except Exception as e:
        print(f"[{symbol}] Error reading parquet: {e}")
        return False
    
    # Get the latest date in existing data
    latest_date = pd.Timestamp(df_existing.index[-1]).date()
    
    # If already up to date, skip
    if latest_date >= today:
        return True
    
    # Fetch missing days (start from day after latest)
    start_date = latest_date + timedelta(days=1)
    
    try:
        df_new = yf.download(
            tickers=symbol,
            start=start_date.isoformat(),
            end=today.isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        
        if df_new.empty:
            return True  # No new data (e.g., weekend/holiday)
        
        # Handle MultiIndex columns from yfinance
        if isinstance(df_new.columns, pd.MultiIndex):
            df_new.columns = df_new.columns.droplevel(1)
        
        # Normalize timezones
        if df_new.index.tz is not None:
            df_new.index = df_new.index.tz_convert(None)
        if df_existing.index.tz is not None:
            df_existing.index = df_existing.index.tz_convert(None)
        
        # Append new data and save
        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep='last')]
        df_combined.sort_index(inplace=True)
        df_combined.to_parquet(parquet_path)
        
        print(f"[{symbol}] Added {len(df_new)} new rows")
        return True
        
    except Exception as e:
        print(f"[{symbol}] Error fetching updates: {e}")
        return False


def get_price_df(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    Return a DataFrame indexed by datetime with columns like
    [Open, High, Low, Close, Volume] (case-insensitive).
    Reads from parquet files in price_history/ directory.
    Automatically fetches missing data from yfinance if stale.
    """
    # Check and update parquet if data is stale
    update_parquet_if_stale(symbol)
    
    try:
        df = pd.read_parquet(f"{PRICE_HISTORY_DIR}/{symbol}.parquet")
    except FileNotFoundError:
        print(f"Parquet file not found for {symbol}")
        return None
    except Exception as e:
        print(f"Error reading parquet for {symbol}: {e}")
        return None
    
    # Filter by timeframe
    today = datetime.now().date()
    try:
        if timeframe == "1d":
            df = df[df.index.date >= today - timedelta(days=1)]
        elif timeframe == "5d":
            df = df[df.index.date >= today - timedelta(days=5)]
        elif timeframe == "1mo":
            df = df[df.index.date >= today - timedelta(days=30)]
        elif timeframe == "6mo":
            df = df[df.index.date >= today - timedelta(days=180)]
        elif timeframe == "1y":
            df = df[df.index.date >= today - timedelta(days=365)]
        elif timeframe == "5y":
            df = df[df.index.date >= today - timedelta(days=1825)]
        elif timeframe == "max":
            pass  # Return all data
        else:
            raise ValueError(f"Invalid timeframe: {timeframe}")
    except Exception as e:
        print(f"Error filtering by timeframe: {e}")
        return None
    
    # Handle timezone conversion if needed
    if df.index.tz is not None:
        df.index = df.index.tz_convert(None)
    
    return df


def get_company_snapshot(symbol: str) -> Optional[Dict]:
    """
    Return a dict with at least: {name, sector, industry, marketCap, description}.
    """
    try:
        info = yf.Ticker(symbol)
    except Exception as e:
        print(f"Error: {e}")
        return None
    return info.info


def get_trending_x() -> Optional[List[Dict]]:
    """Return a list of dicts: {symbol, name?, mentions?, sentiment?, changePct?}."""
    # TODO: implement.
    return None


def get_trending_wsb() -> Optional[List[Dict]]:
    """Return a list of dicts: {symbol, name?, mentions?, sentiment?, changePct?}."""
    # TODO: implement.
    return None

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
