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
import json
import os
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional

import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL
import dash

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

def _load_ticker_options():
    parquet_dir = "price_history"
    parquet_tickers = []
    if os.path.isdir(parquet_dir):
        parquet_tickers = [f.name.replace(".parquet", "") for f in os.scandir(parquet_dir) if f.is_file()]
    promising = []
    csv_path = os.path.join(os.path.dirname(__file__) or ".", "promising_stocks.csv")
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            promising = df["ticker"].dropna().astype(str).str.strip().tolist()
        except Exception:
            pass
    return sorted(set(parquet_tickers) | set(promising))

tickers = _load_ticker_options()


def _load_similarity_map() -> Dict[str, List[Dict[str, object]]]:
    similarity_path = os.path.join(os.path.dirname(__file__) or ".", "autoencoder_similar_stocks.json")
    if not os.path.isfile(similarity_path):
        return {}
    try:
        with open(similarity_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
        return {
            str(symbol).upper(): neighbors
            for symbol, neighbors in raw.items()
            if isinstance(neighbors, list)
        }
    except Exception as e:
        print(f"Error loading similarity map: {e}")
        return {}


SIMILARITY_MAP = _load_similarity_map()


def _normalize_selected_symbols(symbol_value) -> List[str]:
    if symbol_value is None:
        return []
    if isinstance(symbol_value, (list, tuple)):
        return [str(symbol).strip().upper() for symbol in symbol_value if str(symbol).strip()]
    symbol = str(symbol_value).strip()
    return [symbol.upper()] if symbol else []


TIMEFRAMES = [
    ("1D", "1d"), ("5D", "5d"), ("1M", "1mo"), ("6M", "6mo"), ("1Y", "1y"), ("5Y", "5y"), ("Max", "max")
]

# Moving average options: (label, value, rolling window in trading days)
MA_OPTIONS = [
    ("50-day MA", "ma_50d", 50),
    ("200-day MA", "ma_200d", 200),
    ("50-week MA", "ma_50w", 250),   # ~50 * 5 trading days
    ("200-week MA", "ma_200w", 1000), # ~200 * 5
]
MA_MAP = {v: (label, window) for label, v, window in MA_OPTIONS}

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



def make_price_figure(
    df: Optional[pd.DataFrame],
    symbol: str,
    timeframe_label: str,
    selected_ma: Optional[List[str]] = None,
) -> go.Figure:
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

    # Moving averages (daily data only; need enough bars)
    close_col = cols.get("close") or cols.get("adj close")
    if close_col and selected_ma:
        close = df[close_col]
        for ma_key in selected_ma:
            if ma_key not in MA_MAP:
                continue
            label, window = MA_MAP[ma_key]
            if len(close) < window:
                continue
            ma_series = close.rolling(window, min_periods=window).mean()
            fig.add_trace(go.Scatter(x=df.index, y=ma_series, name=label, mode="lines"))

    # Get extra info for title
    extra_info = extra_info_in_title(df)
    fig.update_layout(showlegend=bool(selected_ma))
    
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

    # Hide non-trading gaps for intraday views (1D / 5D)
    if timeframe_label in ("1D", "5D"):
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),           # weekends
                dict(bounds=[16, 9.5], pattern="hour"), # overnight
            ]
        )

    return fig


def make_comparison_figure(
    price_frames: Dict[str, Optional[pd.DataFrame]],
    timeframe_label: str,
    normalize: bool = False,
) -> go.Figure:
    fig = go.Figure()
    added_trace = False

    for symbol, df in price_frames.items():
        if df is None or df.empty:
            continue
        close_col = "Close" if "Close" in df.columns else "Adj Close" if "Adj Close" in df.columns else None
        if close_col is None:
            continue
        series = pd.to_numeric(df[close_col], errors="coerce")
        if normalize:
            first_valid_index = series.first_valid_index()
            if first_valid_index is None:
                continue
            base_price = series.loc[first_valid_index]
            if not pd.notna(base_price) or abs(float(base_price)) < 1e-8:
                continue
            series = (series / float(base_price)) * 100.0
        fig.add_trace(go.Scatter(
            x=df.index,
            y=series,
            name=symbol,
            mode="lines",
        ))
        added_trace = True

    if not added_trace:
        fig.update_layout(
            annotations=[dict(
                text="No closing-price data available for the selected comparison.",
                x=0.5, y=0.5, xref="paper", yref="paper",
                showarrow=False, font=dict(size=14, color="#888")
            )],
            margin=dict(l=10, r=10, t=30, b=10),
            template="plotly_white",
            title=f"{'Normalized ' if normalize else ''}Closing Price Comparison - {timeframe_label}"
        )
        return fig

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="Normalized Close (Start = 100)" if normalize else "Close",
        title=f"{'Normalized ' if normalize else ''}Closing Price Comparison - {timeframe_label}",
    )

    if timeframe_label in ("1D", "5D"):
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.5], pattern="hour"),
            ]
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
            dcc.Dropdown(id="ticker", options=tickers, value=["AAPL"], multi=True,
                         placeholder="Search tickers or companies…", style={"flex": 1}),
            html.Div(dcc.RadioItems(
                id="timeframe",
                options=[{"label": lab, "value": val} for lab, val in TIMEFRAMES],
                value="6mo",
                inline=True,
                labelStyle={"display": "inline-block", "marginRight": 10}
            ), style={"width": "100%"}),
        ], style={"display": "flex", "flexDirection": "column", "gap": 8, "marginBottom": 8}),

        # Moving averages checklist
        html.Div([
            html.Span("Moving averages: ", style={"marginRight": 8, "color": "#666"}),
            dcc.Checklist(
                id="ma-checklist",
                options=[{"label": lab, "value": val} for lab, val, _ in MA_OPTIONS],
                value=[],
                inline=True,
                labelStyle={"display": "inline-block", "marginRight": 16},
            ),
        ], style={"marginBottom": 8}),

        html.Div([
            dcc.Checklist(
                id="normalize-comparison",
                options=[{"label": "Normalize comparison (start = 100)", "value": "normalize"}],
                value=[],
                inline=True,
                labelStyle={"display": "inline-block", "marginRight": 16, "color": "#666"},
            ),
        ], style={"marginBottom": 8}),

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

            # Right: promising stocks + near 200w MA
            html.Div([
                html.Div([
                    html.H3("Promising stocks", style={"marginTop": 0}),
                    html.Div(id="promising-stocks"),
                ], style=CARD_STYLE),
                dcc.Store(id="promising-store", data=None),
                html.Div([
                    html.H3("Most Similar Stocks", style={"marginTop": 0}),
                    html.Div(id="similar-stocks"),
                ], id="similar-stocks-card", style=CARD_STYLE),
                html.Div([
                    html.H3("Within 10% of 200-week MA", style={"marginTop": 0}),
                    html.Div(id="near-200w-ma"),
                ], style=CARD_STYLE),
                html.Div([
                    html.H3("Below 50-week MA", style={"marginTop": 0}),
                    html.Div(id="below-50w-ma"),
                ], style=CARD_STYLE),
                dcc.Store(id="near-200w-store", data=None),
                dcc.Store(id="below-50w-store", data=None),
                dcc.Interval(id="interval-near200w", interval=5_000, n_intervals=0),
                dcc.Interval(id="interval-below50w", interval=5_000, n_intervals=0),
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
    Input("ma-checklist", "value"),
    Input("normalize-comparison", "value"),
    prevent_initial_call=False,
)
def update_chart(symbol: str, timeframe: str, selected_ma: Optional[List[str]], normalize_comparison: Optional[List[str]]):
    symbols = _normalize_selected_symbols(symbol)
    if not symbols:
        return make_price_figure(None, "—", "—", selected_ma=selected_ma or [])
    
    label = next((lab for lab, val in TIMEFRAMES if val == timeframe), timeframe)
    if len(symbols) > 1:
        normalize = "normalize" in (normalize_comparison or [])
        price_frames = {}
        for selected_symbol in symbols:
            try:
                price_frames[selected_symbol] = get_price_df(selected_symbol, timeframe)
            except Exception as e:
                print(f"Error loading data for {selected_symbol}: {e}")
                price_frames[selected_symbol] = None
        return make_comparison_figure(price_frames, label, normalize=normalize)

    selected_symbol = symbols[0]
    try:
        df = get_price_df(selected_symbol, timeframe)
    except Exception as e:
        print(f"Error loading data for {selected_symbol}: {e}")
        df = None
    return make_price_figure(df, selected_symbol, label, selected_ma=selected_ma or [])


@app.callback(
    Output("company-name", "children"),
    Output("company-meta", "children"),
    Output("company-cap", "children"),
    Output("company-about", "children"),
    Input("ticker", "value"),  # triggers when ticker changes
)
def update_company_info(symbol: str):
    symbols = _normalize_selected_symbols(symbol)
    if not symbols:
        return ("—", "Sector: —  •  Industry: —", "Market Cap: —", "Select a ticker to view company information.")
    
    if len(symbols) > 1:
        preview = ", ".join(symbols[:5])
        if len(symbols) > 5:
            preview += ", ..."
        return (
            "Comparison mode",
            f"Selected tickers: {preview}",
            "Market Cap: —",
            "Company snapshot is only shown when a single ticker is selected.",
        )

    selected_symbol = symbols[0]
    try:
        info = get_company_snapshot(selected_symbol)  # <-- implement
    except Exception as e:
        print(f"Error loading company info for {selected_symbol}: {e}")
        info = None

    if not info:
        return (
            "—",
            "Sector: —  •  Industry: —",
            "Market Cap: —",
            "Connect your company data in get_company_snapshot(...).",
        )

    name = info.get("name") or selected_symbol
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
        f"{name} ({selected_symbol})",
        f"Sector: {sector}  •  Industry: {industry}",
        f"Market Cap: {fmt_money(cap)}",
        about,
    )


# Load promising tickers into Store once (so panel doesn't re-render on every ticker change)
@app.callback(
    Output("promising-store", "data"),
    Input("ticker", "value"),
    State("promising-store", "data"),
    prevent_initial_call=False,
)
def fill_promising_store(_ticker, current):
    if current is not None:
        return dash.no_update
    return get_promising_tickers()


# Build promising-stocks panel from Store (renders once when Store is filled)
@app.callback(
    Output("promising-stocks", "children"),
    Input("promising-store", "data"),
    prevent_initial_call=False,
)
def build_promising_panel(ticker_list):
    if not ticker_list:
        return html.Div("No promising_stocks.csv or empty.", style={"color": "#777"})
    btn_style = {"display": "block", "width": "100%", "textAlign": "left", "marginBottom": 4, "cursor": "pointer", "padding": "6px 8px", "border": "1px solid #e0e0e0", "borderRadius": 4, "background": "#fafafa"}
    return html.Div([
        html.Button(t, id={"type": "ticker-select-promising", "index": t}, style=btn_style)
        for t in ticker_list
    ], style={"display": "grid", "gap": 4, "maxHeight": 280, "overflowY": "auto"})


@app.callback(
    Output("similar-stocks", "children"),
    Input("ticker", "value"),
    prevent_initial_call=False,
)
def build_similar_stocks_panel(symbol: str):
    symbols = _normalize_selected_symbols(symbol)
    if not symbols:
        return html.Div("Select a ticker to view similar stocks.", style={"color": "#777"})
    if len(symbols) != 1:
        return html.Div()

    neighbors = SIMILARITY_MAP.get(symbols[0], [])[:10]
    if not neighbors:
        return html.Div("No autoencoder similarity data available for this ticker.", style={"color": "#777"})

    btn_style = {
        "display": "flex",
        "justifyContent": "space-between",
        "alignItems": "center",
        "width": "100%",
        "textAlign": "left",
        "marginBottom": 4,
        "cursor": "pointer",
        "padding": "6px 8px",
        "border": "1px solid #e0e0e0",
        "borderRadius": 4,
        "background": "#fafafa",
        "gap": 8,
    }
    return html.Div([
        html.Button(
            [
                html.Span(neighbor.get("ticker", "—"), style={"fontWeight": 600}),
                html.Span(
                    f"L2 distance: {float(neighbor.get('distance', 0.0)):.4f}",
                    style={"color": "#666", "fontSize": 12},
                ),
            ],
            id={"type": "ticker-select-similar", "index": neighbor.get("ticker", "")},
            style=btn_style,
        )
        for neighbor in neighbors
        if neighbor.get("ticker")
    ], style={"display": "grid", "gap": 4, "maxHeight": 320, "overflowY": "auto"})


def _has_real_click(value) -> bool:
    if isinstance(value, (list, tuple)):
        return any(_has_real_click(item) for item in value)
    if value is None:
        return False
    try:
        return float(value) > 0
    except (TypeError, ValueError):
        return False


@app.callback(
    Output("similar-stocks-card", "style"),
    Input("ticker", "value"),
    prevent_initial_call=False,
)
def toggle_similar_stocks_card(symbol):
    symbols = _normalize_selected_symbols(symbol)
    if len(symbols) == 1:
        return CARD_STYLE
    return {**CARD_STYLE, "display": "none"}

# Poll cache file; only update Store when content actually changes (avoids re-mounting buttons every 5s)
@app.callback(
    Output("near-200w-store", "data"),
    Input("ticker", "value"),
    Input("interval-near200w", "n_intervals"),
    State("near-200w-store", "data"),
    prevent_initial_call=False,
)
def update_near_200w_store(_ticker, _n_intervals, current_list):
    if not os.path.isfile(_NEAR_200W_CACHE_FILE):
        if not os.path.isfile(_NEAR_200W_LOCK_FILE):
            try:
                open(_NEAR_200W_LOCK_FILE, "w").close()
                t = threading.Thread(target=_compute_near_200w_ma_to_cache, daemon=True)
                t.start()
            except Exception:
                pass
        return dash.no_update
    try:
        with open(_NEAR_200W_CACHE_FILE) as f:
            new_list = json.load(f)
        if new_list != current_list:
            return new_list
    except Exception:
        pass
    return dash.no_update


# Build near-200w MA panel from Store (only re-renders when Store changes)
@app.callback(
    Output("near-200w-ma", "children"),
    Input("near-200w-store", "data"),
    prevent_initial_call=False,
)
def build_near_200w_panel(ticker_list):
    if not ticker_list:
        return html.Div("Computing in background…", style={"color": "#777"})
    btn_style = {"display": "block", "width": "100%", "textAlign": "left", "marginBottom": 4, "cursor": "pointer", "padding": "6px 8px", "border": "1px solid #e0e0e0", "borderRadius": 4, "background": "#fafafa"}
    return html.Div([
        html.Button(t, id={"type": "ticker-select-near200w", "index": t}, style=btn_style)
        for t in ticker_list
    ], style={"display": "grid", "gap": 4, "maxHeight": 280, "overflowY": "auto"})


# Poll cache for below-50w MA list; update Store only when content changes
@app.callback(
    Output("below-50w-store", "data"),
    Input("ticker", "value"),
    Input("interval-below50w", "n_intervals"),
    State("below-50w-store", "data"),
    prevent_initial_call=False,
)
def update_below_50w_store(_ticker, _n_intervals, current_list):
    if not os.path.isfile(_BELOW_50W_CACHE_FILE):
        if not os.path.isfile(_BELOW_50W_LOCK_FILE):
            try:
                open(_BELOW_50W_LOCK_FILE, "w").close()
                t = threading.Thread(target=_compute_below_50w_ma_to_cache, daemon=True)
                t.start()
            except Exception:
                pass
        return dash.no_update
    try:
        with open(_BELOW_50W_CACHE_FILE) as f:
            new_list = json.load(f)
        if new_list != current_list:
            return new_list
    except Exception:
        pass
    return dash.no_update


@app.callback(
    Output("below-50w-ma", "children"),
    Input("below-50w-store", "data"),
    prevent_initial_call=False,
)
def build_below_50w_panel(ticker_list):
    if not ticker_list:
        return html.Div("Computing in background…", style={"color": "#777"})
    btn_style = {"display": "block", "width": "100%", "textAlign": "left", "marginBottom": 4, "cursor": "pointer", "padding": "6px 8px", "border": "1px solid #e0e0e0", "borderRadius": 4, "background": "#fafafa"}
    return html.Div([
        html.Button(t, id={"type": "ticker-select-below50w", "index": t}, style=btn_style)
        for t in ticker_list
    ], style={"display": "grid", "gap": 4, "maxHeight": 280, "overflowY": "auto"})


# Clicking a ticker in any panel updates the dropdown (and thus the chart)
@app.callback(
    Output("ticker", "value", allow_duplicate=True),
    Input({"type": "ticker-select-promising", "index": ALL}, "n_clicks"),
    Input({"type": "ticker-select-similar", "index": ALL}, "n_clicks"),
    Input({"type": "ticker-select-near200w", "index": ALL}, "n_clicks"),
    Input({"type": "ticker-select-below50w", "index": ALL}, "n_clicks"),
    prevent_initial_call=True,
)
def set_ticker_from_panel(_promising_clicks, _similar_clicks, _near200w_clicks, _below50w_clicks):
    if not ctx.triggered_id or not ctx.triggered:
        return dash.no_update
    if not _has_real_click(ctx.triggered[0].get("value")):
        return dash.no_update
    return [ctx.triggered_id["index"]]

# ------------------------------------------------------------
# YOUR DATA FUNCTIONS — replace these stubs
# ------------------------------------------------------------

PRICE_HISTORY_DIR = "price_history"
_NEAR_200W_CACHE_FILE = os.path.join(os.path.dirname(__file__) or ".", ".near_200w_ma_cache.json")
_NEAR_200W_LOCK_FILE = os.path.join(os.path.dirname(__file__) or ".", ".near_200w_ma_computing.lock")
_BELOW_50W_CACHE_FILE = os.path.join(os.path.dirname(__file__) or ".", ".below_50w_ma_cache.json")
_BELOW_50W_LOCK_FILE = os.path.join(os.path.dirname(__file__) or ".", ".below_50w_ma_computing.lock")

# Timeframe cutoffs in days for parquet-based lookups
_TF_DAYS = {"1mo": 30, "6mo": 180, "1y": 365, "5y": 1825}


def _compute_near_200w_ma_to_cache():
    """Run in background: compute get_tickers_near_200w_ma() and write list to cache file."""
    try:
        tickers = get_tickers_near_200w_ma()
        with open(_NEAR_200W_CACHE_FILE, "w") as f:
            json.dump(tickers, f)
    except Exception:
        pass
    finally:
        if os.path.isfile(_NEAR_200W_LOCK_FILE):
            try:
                os.remove(_NEAR_200W_LOCK_FILE)
            except Exception:
                pass


def _compute_below_50w_ma_to_cache():
    """Run in background: compute get_tickers_below_50w_ma() and write list to cache file."""
    try:
        tickers = get_tickers_below_50w_ma()
        with open(_BELOW_50W_CACHE_FILE, "w") as f:
            json.dump(tickers, f)
    except Exception:
        pass
    finally:
        if os.path.isfile(_BELOW_50W_LOCK_FILE):
            try:
                os.remove(_BELOW_50W_LOCK_FILE)
            except Exception:
                pass


def _normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Strip MultiIndex columns and convert index to naive ET datetimes."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    if df.index.tz is not None:
        # Convert to ET then strip tz — keeps wall-clock hours aligned
        # so rangebreaks at [16, 9.5] work correctly
        df.index = df.index.tz_convert("America/New_York").tz_localize(None)
    return df


def update_parquet_if_stale(symbol: str) -> bool:
    """
    Check the latest date in the parquet file and append any missing
    daily bars from yfinance.  Returns True on success / no-op.
    """
    parquet_path = os.path.join(PRICE_HISTORY_DIR, f"{symbol}.parquet")
    today = datetime.now().date()

    try:
        df_existing = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        return False
    except Exception as e:
        print(f"[{symbol}] Error reading parquet: {e}")
        return False

    df_existing = _normalize_df(df_existing)
    latest_date = pd.Timestamp(df_existing.index[-1]).date()

    if latest_date >= today:
        return True  # already current

    start_date = latest_date + timedelta(days=1)
    try:
        df_new = yf.download(
            tickers=symbol,
            start=start_date.isoformat(),
            end=(today + timedelta(days=1)).isoformat(),
            interval="1d",
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df_new.empty:
            return True  # no new trading days

        df_new = _normalize_df(df_new)

        df_combined = pd.concat([df_existing, df_new])
        df_combined = df_combined[~df_combined.index.duplicated(keep="last")]
        df_combined.sort_index(inplace=True)
        df_combined.to_parquet(parquet_path)
        print(f"[{symbol}] Appended {len(df_new)} new row(s)")
        return True
    except Exception as e:
        print(f"[{symbol}] Error fetching updates: {e}")
        return False


def _fetch_live_intraday(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    For 1D / 5D: pull intraday bars directly from yfinance.
    Uses 1-minute bars for 1D, 5-minute bars for 5D.
    """
    interval = "1m" if timeframe == "1d" else "5m"
    try:
        df = yf.download(
            tickers=symbol,
            period=timeframe,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
        if df.empty:
            return None
        return _normalize_df(df)
    except Exception as e:
        print(f"[{symbol}] Intraday fetch error: {e}")
        return None


def get_price_df(symbol: str, timeframe: str) -> Optional[pd.DataFrame]:
    """
    1D / 5D  → live intraday data from yfinance (minute/5-min bars).
    1M+ / Max → daily bars from local parquet (auto-backfilled).
    """
    # --- short-horizon: live intraday ---
    if timeframe in ("1d", "5d"):
        return _fetch_live_intraday(symbol, timeframe)

    # --- longer horizon: parquet + backfill ---
    update_parquet_if_stale(symbol)

    parquet_path = os.path.join(PRICE_HISTORY_DIR, f"{symbol}.parquet")
    try:
        df = pd.read_parquet(parquet_path)
    except FileNotFoundError:
        print(f"Parquet file not found for {symbol}")
        return None
    except Exception as e:
        print(f"Error reading parquet for {symbol}: {e}")
        return None

    df = _normalize_df(df)

    days = _TF_DAYS.get(timeframe)
    if days:
        cutoff = datetime.now().date() - timedelta(days=days)
        df = df[df.index.date >= cutoff]
    # timeframe == "max" → return everything

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


def get_promising_tickers() -> List[str]:
    """Tickers from promising_stocks.csv."""
    csv_path = os.path.join(os.path.dirname(__file__) or ".", "promising_stocks.csv")
    if not os.path.isfile(csv_path):
        return []
    try:
        df = pd.read_csv(csv_path)
        return df["ticker"].dropna().astype(str).str.strip().tolist()
    except Exception:
        return []


def get_tickers_near_200w_ma(pct_band: float = 0.10) -> List[str]:
    """Tickers whose latest close is within pct_band of their 200-week (1000-day) MA.
    Only includes stocks with at least 10 years of data (~2520 trading days).
    """
    out = []
    ma_days = 1000
    min_trading_days = 2520  # ~10 years
    if not os.path.isdir(PRICE_HISTORY_DIR):
        return out
    for f in os.scandir(PRICE_HISTORY_DIR):
        if not f.is_file() or not f.name.endswith(".parquet"):
            continue
        symbol = f.name.replace(".parquet", "")
        try:
            df = pd.read_parquet(f.path)
            df = _normalize_df(df)
            close_col = "Close" if "Close" in df.columns else "Adj Close"
            if close_col not in df.columns or len(df) < ma_days:
                continue
            if len(df) < min_trading_days:
                continue
            close = df[close_col]
            ma = close.rolling(ma_days, min_periods=ma_days).mean()
            last_close = close.iloc[-1]
            last_ma = ma.iloc[-1]
            if pd.isna(last_ma) or last_ma <= 0:
                continue
            ratio = last_close / last_ma
            if 1 - pct_band <= ratio <= 1 + pct_band:
                out.append(symbol)
        except Exception:
            continue
    return sorted(out)


def get_tickers_below_50w_ma() -> List[str]:
    """Tickers whose latest close is below their 50-week (250-day) MA.
    Only includes stocks with at least 10 years of data (~2520 trading days).
    """
    out = []
    ma_days = 250  # 50 weeks
    min_trading_days = 2520
    if not os.path.isdir(PRICE_HISTORY_DIR):
        return out
    for f in os.scandir(PRICE_HISTORY_DIR):
        if not f.is_file() or not f.name.endswith(".parquet"):
            continue
        symbol = f.name.replace(".parquet", "")
        try:
            df = pd.read_parquet(f.path)
            df = _normalize_df(df)
            close_col = "Close" if "Close" in df.columns else "Adj Close"
            if close_col not in df.columns or len(df) < ma_days or len(df) < min_trading_days:
                continue
            close = df[close_col]
            ma = close.rolling(ma_days, min_periods=ma_days).mean()
            last_close = close.iloc[-1]
            last_ma = ma.iloc[-1]
            if pd.isna(last_ma):
                continue
            if last_close < last_ma:
                out.append(symbol)
        except Exception:
            continue
    return sorted(out)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
