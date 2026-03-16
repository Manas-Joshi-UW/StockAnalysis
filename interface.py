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
DEFAULT_TICKER_NAMES = {
    item["value"]: item["label"].rsplit("(", 1)[0].strip()
    for item in DEFAULT_TICKERS
}


def _load_listing_name_map() -> Dict[str, str]:
    csv_path = os.path.join(os.path.dirname(__file__) or ".", "nyse_nasdaq_listings.csv")
    if not os.path.isfile(csv_path):
        return {}
    try:
        df = pd.read_csv(csv_path, usecols=["Symbol", "Security Name"])
        df["Symbol"] = df["Symbol"].astype(str).str.strip().str.upper()
        df["Security Name"] = df["Security Name"].fillna("").astype(str).str.strip()
        df = df[df["Symbol"] != ""].drop_duplicates(subset=["Symbol"], keep="first")
        return {
            row["Symbol"]: row["Security Name"]
            for _, row in df.iterrows()
            if row["Security Name"]
        }
    except Exception as e:
        print(f"Error loading ticker name map: {e}")
        return {}


LISTING_NAME_MAP = _load_listing_name_map()


def _ticker_option(symbol: str) -> Dict[str, str]:
    name = LISTING_NAME_MAP.get(symbol) or DEFAULT_TICKER_NAMES.get(symbol) or ""
    label = f"{symbol} - {name}" if name else symbol
    return {"label": label, "value": symbol}


def _load_ticker_options():
    parquet_dir = "price_history"
    parquet_tickers = []
    if os.path.isdir(parquet_dir):
        parquet_tickers = [
            f.name.replace(".parquet", "").upper()
            for f in os.scandir(parquet_dir)
            if f.is_file()
        ]
    promising = []
    csv_path = os.path.join(os.path.dirname(__file__) or ".", "promising_stocks.csv")
    if os.path.isfile(csv_path):
        try:
            df = pd.read_csv(csv_path)
            promising = df["ticker"].dropna().astype(str).str.strip().str.upper().tolist()
        except Exception:
            pass
    ordered_symbols = list(
        dict.fromkeys(
            list(DEFAULT_TICKER_NAMES)
            + sorted(set(parquet_tickers) | set(promising))
        )
    )
    return [_ticker_option(symbol) for symbol in ordered_symbols]

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

def _column_lookup(df: pd.DataFrame) -> Dict[str, object]:
    if isinstance(df.columns, pd.MultiIndex):
        return {str(column[0]).lower(): column for column in df.columns}
    return {str(column).lower(): column for column in df.columns}


def _extract_price_series(df_price: Optional[pd.DataFrame]) -> Optional[pd.Series]:
    if df_price is None or df_price.empty:
        return None

    cols = _column_lookup(df_price)
    price_col = cols.get("close") or cols.get("adj close") or next(iter(cols.values()), None)
    if price_col is None:
        return None

    series = df_price[price_col]
    if isinstance(series, pd.DataFrame):
        series = series.iloc[:, 0]
    series = pd.to_numeric(series, errors="coerce").dropna()
    if series.empty:
        return None
    return series


def _normalize_symbol_selection(value) -> List[str]:
    if value is None:
        return []
    raw_values = value if isinstance(value, (list, tuple)) else [value]
    selected_symbols: List[str] = []
    for raw_value in raw_values:
        symbol = str(raw_value or "").strip().upper()
        if symbol and symbol not in selected_symbols:
            selected_symbols.append(symbol)
    return selected_symbols


def _primary_selected_symbol(value) -> Optional[str]:
    selected_symbols = _normalize_symbol_selection(value)
    return selected_symbols[0] if selected_symbols else None


def extra_info_in_title(df_price: pd.DataFrame) -> Dict[str, str]:
    """
    Extract key statistics from price data for display in chart title.
    Returns a dictionary with keys like 'Max', 'Min', 'Current', etc.
    """
    if df_price is None or df_price.empty:
        return {}
    info = {}
    cols = _column_lookup(df_price)
    price_data = _extract_price_series(df_price)
    if price_data is None:
        return {}

    try:
        current_price = float(price_data.iloc[-1])
        info["Current"] = f"${current_price:.2f}"

        max_price = float(price_data.max())
        info["Max"] = f"${max_price:.2f}"

        min_price = float(price_data.min())
        info["Min"] = f"${min_price:.2f}"

        first_price = float(price_data.iloc[0])
        price_change = current_price - first_price
        price_change_pct = (price_change / first_price) * 100 if first_price else 0.0

        if price_change > 0:
            info["Change"] = f"+${price_change:.2f} (+{price_change_pct:.1f}%)"
        elif price_change < 0:
            info["Change"] = f"-${abs(price_change):.2f} ({price_change_pct:.1f}%)"
        else:
            info["Change"] = "$0.00 (0.0%)"

        volume_col = cols.get("volume")
        if volume_col is not None:
            volume_data = df_price[volume_col]
            if isinstance(volume_data, pd.DataFrame):
                volume_data = volume_data.iloc[:, 0]
            volume_data = pd.to_numeric(volume_data, errors="coerce").dropna()
            if not volume_data.empty:
                avg_volume = float(volume_data.mean())
                if avg_volume >= 1e9:
                    info["Avg Vol"] = f"{avg_volume/1e9:.1f}B"
                elif avg_volume >= 1e6:
                    info["Avg Vol"] = f"{avg_volume/1e6:.1f}M"
                else:
                    info["Avg Vol"] = f"{avg_volume/1e3:.1f}K"

        if len(df_price.index) > 1 and hasattr(df_price.index[0], "strftime"):
            start_date = df_price.index[0].strftime("%b %Y")
            end_date = df_price.index[-1].strftime("%b %Y")
            if start_date != end_date:
                info["Period"] = f"{start_date} - {end_date}"
            else:
                info["Period"] = start_date

    except Exception as e:
        print(f"Error calculating extra info: {e}")
        return {}
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
            title=f"{symbol} - {timeframe_label}"
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

    fig.update_layout(showlegend=bool(selected_ma))

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=30, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None, yaxis_title=None,
        title=f"{symbol} - {timeframe_label}"
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


def make_normalized_returns_figure(series_map: Dict[str, pd.Series], timeframe_label: str) -> go.Figure:
    fig = go.Figure()

    if not series_map:
        fig.update_layout(
            annotations=[
                dict(
                    text="Add comparison tickers to view normalized returns.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="#888"),
                )
            ],
            margin=dict(l=10, r=10, t=40, b=10),
            template="plotly_white",
            title=f"Normalized Returns - {timeframe_label}",
            yaxis_title="Return (%)",
        )
        return fig

    for index, (symbol, series) in enumerate(series_map.items()):
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=symbol,
                mode="lines",
                line=dict(width=3 if index == 0 else 2),
            )
        )

    fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#98a2b3")
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="Return (%)",
        title=f"Normalized Returns - {timeframe_label}",
    )
    return fig


def make_multi_symbol_figure(
    series_map: Dict[str, pd.Series],
    timeframe_label: str,
    *,
    normalized: bool,
) -> go.Figure:
    fig = go.Figure()

    if not series_map:
        fig.update_layout(
            annotations=[
                dict(
                    text="Select one or more stocks to view the chart.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="#888"),
                )
            ],
            margin=dict(l=10, r=10, t=40, b=10),
            template="plotly_white",
            title=f"Selected Stocks - {timeframe_label}",
            yaxis_title="Return (%)" if normalized else "Price ($)",
        )
        return fig

    symbols = list(series_map.keys())
    title_symbols = ", ".join(symbols[:3])
    if len(symbols) > 3:
        title_symbols = f"{title_symbols} +{len(symbols) - 3} more"

    title = f"{title_symbols} - {timeframe_label}"
    if normalized:
        title = f"{title} (Normalized)"

    show_period_return = len(series_map) > 1
    for index, (symbol, series) in enumerate(series_map.items()):
        legend_name = symbol
        if show_period_return:
            try:
                if normalized:
                    period_return_pct = float(series.iloc[-1])
                else:
                    first_value = float(series.iloc[0])
                    last_value = float(series.iloc[-1])
                    period_return_pct = ((last_value / first_value) - 1.0) * 100.0 if first_value else 0.0
                legend_name = f"{symbol} ({period_return_pct:+.1f}%)"
            except Exception:
                legend_name = symbol
        fig.add_trace(
            go.Scatter(
                x=series.index,
                y=series,
                name=legend_name,
                mode="lines",
                line=dict(width=3 if index == 0 else 2),
            )
        )

    if normalized:
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#98a2b3")

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title="Return (%)" if normalized else "Price ($)",
        title=title,
    )

    if timeframe_label in ("1D", "5D"):
        fig.update_xaxes(
            rangebreaks=[
                dict(bounds=["sat", "mon"]),
                dict(bounds=[16, 9.5], pattern="hour"),
            ]
        )

    return fig


SUMMARY_METRICS = ("Current", "Change", "Max", "Min", "Avg Vol")
TICKER_BUTTON_STYLE = {
    "display": "block",
    "width": "100%",
    "textAlign": "left",
    "marginBottom": 4,
    "cursor": "pointer",
    "padding": "6px 8px",
    "border": "1px solid #e0e0e0",
    "borderRadius": 4,
    "background": "#fafafa",
}
SIMILAR_BUTTON_STYLE = {
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
TICKER_LIST_STYLE = {"display": "grid", "gap": 4, "maxHeight": 280, "overflowY": "auto"}
SIMILAR_LIST_STYLE = {"display": "grid", "gap": 4, "maxHeight": 320, "overflowY": "auto"}
PANEL_MESSAGE_STYLE = {"color": "#667085", "fontSize": 13, "lineHeight": 1.5}
PANEL_ERROR_STYLE = {
    "border": "1px solid #f5c2c7",
    "borderRadius": 8,
    "padding": "10px 12px",
    "background": "#fff5f5",
    "display": "grid",
    "gap": 6,
}
PANEL_TIMESTAMP_STYLE = {"color": "#666", "fontSize": 12}
PRICE_SUMMARY_ROW_STYLE = {
    "display": "grid",
    "gridTemplateColumns": "repeat(auto-fit, minmax(130px, 1fr))",
    "gap": 8,
    "marginBottom": 12,
}
SUMMARY_CARD_STYLE = {
    "border": "1px solid #e6e6e6",
    "borderRadius": 10,
    "padding": "8px 10px",
    "background": "#fafafa",
}
SUMMARY_LABEL_STYLE = {"color": "#667085", "fontSize": 12, "marginBottom": 4}
SUMMARY_VALUE_STYLE = {"fontWeight": 700, "fontSize": 15, "color": "#101828"}


def _timestamp_now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _file_timestamp(path: str) -> Optional[str]:
    if not path or not os.path.isfile(path):
        return None
    try:
        return datetime.fromtimestamp(os.path.getmtime(path)).isoformat(timespec="seconds")
    except OSError:
        return None


def _format_updated_label(updated_at: Optional[str]) -> str:
    if not updated_at:
        return "Updated recently"
    try:
        dt = datetime.fromisoformat(updated_at)
        return f"Updated {dt.strftime('%b %d, %I:%M %p')}"
    except ValueError:
        return "Updated recently"


def _build_panel_state(
    items: Optional[List[str]] = None,
    *,
    status: Optional[str] = None,
    message: Optional[str] = None,
    updated_at: Optional[str] = None,
) -> Dict[str, object]:
    normalized_items = [item for item in (items or []) if item]
    resolved_status = status or ("ready" if normalized_items else "empty")
    payload: Dict[str, object] = {"status": resolved_status, "items": normalized_items}
    if message:
        payload["message"] = message
    if updated_at:
        payload["updated_at"] = updated_at
    return payload


def _normalize_cached_panel_payload(payload, cache_file: Optional[str] = None) -> Dict[str, object]:
    if isinstance(payload, dict) and payload.get("status"):
        normalized = dict(payload)
        normalized.setdefault("items", [])
        if normalized["status"] in {"ready", "empty"}:
            normalized["items"] = [item for item in normalized.get("items", []) if item]
            normalized.setdefault("updated_at", _file_timestamp(cache_file) or _timestamp_now())
            if normalized["status"] == "ready" and not normalized["items"]:
                normalized["status"] = "empty"
        return normalized
    if isinstance(payload, list):
        return _build_panel_state(payload, updated_at=_file_timestamp(cache_file) or _timestamp_now())
    if payload is None:
        return _build_panel_state(status="loading")
    return _build_panel_state(
        status="error",
        message="Unexpected panel data format. Refresh the page to retry.",
        updated_at=_file_timestamp(cache_file) or _timestamp_now(),
    )


def _build_price_summary_cards(info: Optional[Dict[str, str]]):
    info = info or {}
    cards = []
    for label in SUMMARY_METRICS:
        value = info.get(label) or "--"
        value_style = dict(SUMMARY_VALUE_STYLE)
        if label == "Change":
            normalized_value = value.strip() if isinstance(value, str) else ""
            has_numeric_change = any(char.isdigit() for char in normalized_value)
            if has_numeric_change and normalized_value.startswith("+"):
                value_style["color"] = "#1a7f37"
            elif has_numeric_change and normalized_value.startswith("-"):
                value_style["color"] = "#b42318"
        cards.append(
            html.Div(
                [
                    html.Div(label, style=SUMMARY_LABEL_STYLE),
                    html.Div(value, style=value_style),
                ],
                style=SUMMARY_CARD_STYLE,
            )
        )
    return cards


def _render_ticker_buttons(items: List[str], button_type: str):
    return html.Div(
        [
            html.Button(item, id={"type": button_type, "index": item}, style=TICKER_BUTTON_STYLE)
            for item in items
        ],
        style=TICKER_LIST_STYLE,
    )


def _render_async_ticker_panel(
    panel_data,
    *,
    button_type: str,
    loading_message: str,
    empty_message: str,
    error_message: str,
):
    payload = _normalize_cached_panel_payload(panel_data)
    status = payload.get("status")

    if status == "loading":
        return html.Div(payload.get("message") or loading_message, style=PANEL_MESSAGE_STYLE)

    if status == "error":
        return html.Div(
            [
                html.Div(payload.get("message") or error_message, style={"fontWeight": 600, "color": "#b42318"}),
                html.Div("Refresh the page or rerun the background scan to retry this panel.", style=PANEL_MESSAGE_STYLE),
            ],
            style=PANEL_ERROR_STYLE,
        )

    updated_label = html.Div(_format_updated_label(payload.get("updated_at")), style=PANEL_TIMESTAMP_STYLE)
    items = [item for item in payload.get("items", []) if item]
    if not items:
        return html.Div(
            [updated_label, html.Div(empty_message, style=PANEL_MESSAGE_STYLE)],
            style={"display": "grid", "gap": 6},
        )

    return html.Div(
        [updated_label, _render_ticker_buttons(items, button_type)],
        style={"display": "grid", "gap": 8},
    )

# ------------------------------------------------------------
# Layout
# ------------------------------------------------------------
PAGE_CONTAINER_STYLE = {
    "maxWidth": 1200,
    "margin": "0 auto",
    "padding": 14,
    "width": "100%",
    "boxSizing": "border-box",
    "display": "grid",
    "gap": 12,
}
HEADER_STYLE = {"display": "flex", "flexDirection": "column", "gap": 4}
CONTROL_STACK_STYLE = {"display": "grid", "gap": 12}
SEARCH_BLOCK_STYLE = {"display": "grid", "gap": 6}
TOP_CONTROL_ROW_STYLE = {
    "display": "flex",
    "flexWrap": "wrap",
    "gap": 12,
    "alignItems": "flex-start",
}
CONTROL_BLOCK_STYLE = {"display": "grid", "gap": 6, "flex": "1 1 220px", "minWidth": 0}
WIDE_CONTROL_BLOCK_STYLE = {"display": "grid", "gap": 6, "flex": "2 1 360px", "minWidth": 0}
DISABLED_CONTROL_BLOCK_STYLE = {
    "opacity": 0.45,
    "background": "#f5f5f5",
    "border": "1px solid #e4e7ec",
    "borderRadius": 10,
    "padding": 10,
}
CONTROL_LABEL_STYLE = {"fontSize": 12, "fontWeight": 600, "color": "#667085"}
CONTROL_HINT_STYLE = {"fontSize": 12, "color": "#667085"}
DROPDOWN_STYLE = {"width": "100%", "minWidth": 0}
RADIO_ITEMS_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": "8px 12px"}
RADIO_LABEL_STYLE = {"display": "flex", "alignItems": "center", "marginRight": 0}
CHECKLIST_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": "8px 16px"}
CHECKLIST_LABEL_STYLE = {"display": "flex", "alignItems": "center", "marginRight": 0}
MAIN_CONTENT_GRID_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": 12, "alignItems": "flex-start"}
LEFT_CONTENT_COLUMN_STYLE = {"display": "grid", "gap": 12, "flex": "999 1 680px", "minWidth": 0}
RIGHT_SIDEBAR_STYLE = {"display": "grid", "gap": 12, "flex": "1 1 320px", "minWidth": 0}

"""
_LEGACY_LAYOUT = html.Div(
    [
        html.Div(
            [
                html.H1(APP_TITLE, style={"margin": 0}),
                html.Div(
                    "Explore unfamiliar tickers with price history and quick context.",
                    style={"color": "#666"},
                ),
            ],
            style=HEADER_STYLE,
        ),

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
                ], style=CARD_STYLE),
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
"""

app.layout = html.Div(
    [
        html.Div(
            [
                html.H1(APP_TITLE, style={"margin": 0}),
                html.Div(
                    "Explore unfamiliar tickers with price history and quick context.",
                    style={"color": "#666"},
                ),
            ],
            style=HEADER_STYLE,
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div("Search", style=CONTROL_LABEL_STYLE),
                        html.Div("Select one or more tickers or company names.", style=CONTROL_HINT_STYLE),
                        dcc.Dropdown(
                            id="ticker",
                            options=tickers,
                            value=["AAPL"],
                            multi=True,
                            placeholder="Start typing a ticker or company name...",
                            maxHeight=420,
                            style=DROPDOWN_STYLE,
                        ),
                    ],
                    style=SEARCH_BLOCK_STYLE,
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    [
                                        html.Div("Timeframe", style=CONTROL_LABEL_STYLE),
                                        dcc.RadioItems(
                                            id="timeframe",
                                            options=[{"label": lab, "value": val} for lab, val in TIMEFRAMES],
                                            value="6mo",
                                            style=RADIO_ITEMS_STYLE,
                                            labelStyle=RADIO_LABEL_STYLE,
                                        ),
                                    ],
                                    style=CONTROL_BLOCK_STYLE,
                                ),
                                html.Div(
                                    [
                                        html.Div("Moving averages", style=CONTROL_LABEL_STYLE),
                                        dcc.Checklist(
                                            id="ma-checklist",
                                            options=[{"label": lab, "value": val} for lab, val, _ in MA_OPTIONS],
                                            value=[],
                                            style=CHECKLIST_STYLE,
                                            labelStyle=CHECKLIST_LABEL_STYLE,
                                        ),
                                    ],
                                    id="ma-control-block",
                                    style=WIDE_CONTROL_BLOCK_STYLE,
                                ),
                                html.Div(
                                    [
                                        html.Div("Chart mode", style=CONTROL_LABEL_STYLE),
                                        dcc.Checklist(
                                            id="normalized-toggle",
                                            options=[{"label": "Show normalized returns", "value": "normalized"}],
                                            value=[],
                                            style=CHECKLIST_STYLE,
                                            labelStyle=CHECKLIST_LABEL_STYLE,
                                        ),
                                    ],
                                    style=CONTROL_BLOCK_STYLE,
                                ),
                            ],
                            style=TOP_CONTROL_ROW_STYLE,
                        ),
                    ],
                    style=SEARCH_BLOCK_STYLE,
                ),
            ],
            style=CONTROL_STACK_STYLE,
        ),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(
                            [
                                html.Div(
                                    id="price-summary",
                                    children=_build_price_summary_cards({}),
                                    style=PRICE_SUMMARY_ROW_STYLE,
                                ),
                                dcc.Loading(
                                    dcc.Graph(
                                        id="price-chart",
                                        figure=make_price_figure(None, "--", "--"),
                                        config={"displaylogo": False, "responsive": True},
                                        style={"width": "100%"},
                                    ),
                                    type="default",
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        html.Div(
                            [
                                html.H3("Company Snapshot", style={"marginTop": 0}),
                                html.Div(id="company-name", style={"fontWeight": 600}),
                                html.Div(id="company-meta", style={"color": "#666", "marginBottom": 6}),
                                html.Div(id="company-cap", style={"marginBottom": 10}),
                                html.Div("About:", style={"fontWeight": 600, "marginTop": 6}),
                                html.Div(id="company-about", style={"whiteSpace": "pre-wrap"}),
                            ],
                            style=CARD_STYLE,
                        ),
                    ],
                    style=LEFT_CONTENT_COLUMN_STYLE,
                ),
                html.Div(
                    [
                        dcc.Store(
                            id="promising-store",
                            data={"status": "loading", "message": "Loading promising stocks...", "items": []},
                        ),
                        dcc.Store(
                            id="near-200w-store",
                            data={"status": "loading", "message": "Scanning for tickers near the 200-week moving average...", "items": []},
                        ),
                        dcc.Store(
                            id="below-50w-store",
                            data={"status": "loading", "message": "Scanning for tickers below the 50-week moving average...", "items": []},
                        ),
                        html.Div(
                            [
                                html.H3("Promising stocks", style={"marginTop": 0}),
                                html.Div(
                                    id="promising-stocks",
                                    children=html.Div("Loading promising stocks...", style=PANEL_MESSAGE_STYLE),
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        html.Div(
                            [
                                html.H3("Most Similar Stocks", style={"marginTop": 0}),
                                html.Div(
                                    id="similar-stocks",
                                    children=html.Div("Loading similar stocks...", style=PANEL_MESSAGE_STYLE),
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        html.Div(
                            [
                                html.H3("Within 10% of 200-week MA", style={"marginTop": 0}),
                                html.Div(
                                    id="near-200w-ma",
                                    children=html.Div(
                                        "Scanning for tickers near the 200-week moving average...",
                                        style=PANEL_MESSAGE_STYLE,
                                    ),
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        html.Div(
                            [
                                html.H3("Below 50-week MA", style={"marginTop": 0}),
                                html.Div(
                                    id="below-50w-ma",
                                    children=html.Div(
                                        "Scanning for tickers below the 50-week moving average...",
                                        style=PANEL_MESSAGE_STYLE,
                                    ),
                                ),
                            ],
                            style=CARD_STYLE,
                        ),
                        dcc.Interval(id="interval-near200w", interval=5_000, n_intervals=0),
                        dcc.Interval(id="interval-below50w", interval=5_000, n_intervals=0),
                    ],
                    style=RIGHT_SIDEBAR_STYLE,
                ),
            ],
            style=MAIN_CONTENT_GRID_STYLE,
        ),
    ],
    style=PAGE_CONTAINER_STYLE,
)

# ------------------------------------------------------------
# Callbacks — these call your functions (stubs below)
# ------------------------------------------------------------

@app.callback(
    Output("price-chart", "figure"),
    Input("ticker", "value"),
    Input("timeframe", "value"),
    Input("ma-checklist", "value"),
    Input("normalized-toggle", "value"),
    prevent_initial_call=False,
)
def update_chart(selected_symbols, timeframe: str, selected_ma: Optional[List[str]], normalized_toggle: Optional[List[str]]):
    symbols = _normalize_symbol_selection(selected_symbols)
    normalized = "normalized" in (normalized_toggle or [])
    if not symbols:
        return make_multi_symbol_figure({}, "--", normalized=normalized)

    label = next((lab for lab, val in TIMEFRAMES if val == timeframe), timeframe)
    if len(symbols) == 1 and not normalized:
        primary_symbol = symbols[0]
        try:
            df = get_price_df(primary_symbol, timeframe)
        except Exception as e:
            print(f"Error loading data for {primary_symbol}: {e}")
            df = None
        return make_price_figure(df, primary_symbol, label, selected_ma=selected_ma or [])

    series_map: Dict[str, pd.Series] = {}
    for symbol in symbols:
        try:
            df = get_price_df(symbol, timeframe)
            series = _extract_price_series(df)
            if series is None:
                continue
            if normalized:
                if len(series) < 2:
                    continue
                first_price = float(series.iloc[0])
                if first_price == 0:
                    continue
                series = ((series / first_price) - 1.0) * 100.0
            series_map[symbol] = series
        except Exception as e:
            print(f"Error loading comparison data for {symbol}: {e}")

    return make_multi_symbol_figure(series_map, label, normalized=normalized)


@app.callback(
    Output("ma-checklist", "options"),
    Output("ma-checklist", "value"),
    Output("ma-control-block", "style"),
    Input("ticker", "value"),
    State("ma-checklist", "value"),
    prevent_initial_call=False,
)
def sync_ma_controls(selected_symbols, selected_ma):
    symbols = _normalize_symbol_selection(selected_symbols)
    multi_select_active = len(symbols) > 1
    options = [
        {"label": label, "value": value, "disabled": multi_select_active}
        for label, value, _ in MA_OPTIONS
    ]
    control_style = dict(WIDE_CONTROL_BLOCK_STYLE)
    if multi_select_active:
        control_style.update(DISABLED_CONTROL_BLOCK_STYLE)
        return options, [], control_style
    return options, selected_ma or [], control_style


@app.callback(
    Output("price-summary", "children"),
    Input("ticker", "value"),
    Input("timeframe", "value"),
    prevent_initial_call=False,
)
def update_price_summary(symbol: str, timeframe: str):
    selected_symbols = _normalize_symbol_selection(symbol)
    if len(selected_symbols) > 1:
        return _build_price_summary_cards({label: "-" for label in SUMMARY_METRICS})

    primary_symbol = _primary_selected_symbol(symbol)
    if not primary_symbol:
        return _build_price_summary_cards({})

    try:
        df = get_price_df(primary_symbol, timeframe)
        info = extra_info_in_title(df)
    except Exception as e:
        print(f"Error loading summary data for {primary_symbol}: {e}")
        info = {}

    return _build_price_summary_cards(info)


@app.callback(
    Output("company-name", "children"),
    Output("company-meta", "children"),
    Output("company-cap", "children"),
    Output("company-about", "children"),
    Input("ticker", "value"),  # triggers when ticker changes
)
def update_company_info(symbol: str):
    primary_symbol = _primary_selected_symbol(symbol)
    if not primary_symbol:
        return ("--", "Sector: --  |  Industry: --", "Market Cap: --", "Select a ticker to view company information.")

    try:
        info = get_company_snapshot(primary_symbol)
    except Exception as e:
        print(f"Error loading company info for {primary_symbol}: {e}")
        info = None

    if not info:
        return (
            "--",
            "Sector: --  |  Industry: --",
            "Market Cap: --",
            "Connect your company data in get_company_snapshot(...).",
        )

    name = info.get("name") or primary_symbol
    sector = info.get("sector", "--")
    industry = info.get("industry", "--")
    cap = info.get("marketCap")
    about = info.get("longBusinessSummary") or "--"

    def fmt_money(n):
        try:
            n = float(n)
            if n >= 1e12:
                return f"${n/1e12:.2f}T"
            if n >= 1e9:
                return f"${n/1e9:.2f}B"
            if n >= 1e6:
                return f"${n/1e6:.2f}M"
            return f"${n:,.0f}"
        except Exception:
            return "--"

    return (
        f"{name} ({primary_symbol})",
        f"Sector: {sector}  |  Industry: {industry}",
        f"Market Cap: {fmt_money(cap)}",
        about,
    )
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


# Load promising tickers into Store once (so panel doesn't re-render on every ticker change)
@app.callback(
    Output("promising-store", "data"),
    Input("ticker", "value"),
    State("promising-store", "data"),
    prevent_initial_call=False,
)
def fill_promising_store(_ticker, current):
    current_state = _normalize_cached_panel_payload(current)
    if current_state.get("status") in {"ready", "empty"}:
        return dash.no_update
    try:
        return _build_panel_state(get_promising_tickers(), updated_at=_timestamp_now())
    except Exception as e:
        print(f"Error loading promising tickers: {e}")
        return _build_panel_state(
            status="error",
            message="Could not load promising stocks from promising_stocks.csv.",
        )


# Build promising-stocks panel from Store (renders once when Store is filled)
@app.callback(
    Output("promising-stocks", "children"),
    Input("promising-store", "data"),
    prevent_initial_call=False,
)
def build_promising_panel(panel_data):
    return _render_async_ticker_panel(
        panel_data,
        button_type="ticker-select-promising",
        loading_message="Loading promising stocks...",
        empty_message="No promising stocks were found.",
        error_message="Promising stocks could not be loaded.",
    )


@app.callback(
    Output("similar-stocks", "children"),
    Input("ticker", "value"),
    prevent_initial_call=False,
)
def build_similar_stocks_panel(symbol: str):
    primary_symbol = _primary_selected_symbol(symbol)
    if not primary_symbol:
        return html.Div("Select a ticker to view similar stocks.", style=PANEL_MESSAGE_STYLE)

    try:
        neighbors = SIMILARITY_MAP.get(primary_symbol, [])[:10]
    except Exception as e:
        print(f"Error reading similarity data for {primary_symbol}: {e}")
        return html.Div(
            [
                html.Div("Similarity data could not be loaded.", style={"fontWeight": 600, "color": "#b42318"}),
                html.Div("Refresh the page or rebuild autoencoder_similar_stocks.json to retry.", style=PANEL_MESSAGE_STYLE),
            ],
            style=PANEL_ERROR_STYLE,
        )

    filtered_neighbors = [neighbor for neighbor in neighbors if neighbor.get("ticker")]
    if not filtered_neighbors:
        return html.Div(
            "No autoencoder similarity data is available for this ticker.",
            style=PANEL_MESSAGE_STYLE,
        )

    return html.Div(
        [
            html.Div("Updated from the local similarity map", style=PANEL_TIMESTAMP_STYLE),
            html.Div(
                [
                    html.Button(
                        [
                            html.Span(neighbor.get("ticker", "--"), style={"fontWeight": 600}),
                            html.Span(
                                f"L2 distance: {float(neighbor.get('distance', 0.0)):.4f}",
                                style={"color": "#666", "fontSize": 12},
                            ),
                        ],
                        id={"type": "ticker-select-similar", "index": neighbor.get("ticker", "")},
                        style=SIMILAR_BUTTON_STYLE,
                    )
                    for neighbor in filtered_neighbors
                ],
                style=SIMILAR_LIST_STYLE,
            ),
        ],
        style={"display": "grid", "gap": 8},
    )
    if not symbol:
        return html.Div("Select a ticker to view similar stocks.", style=PANEL_MESSAGE_STYLE)

    try:
        neighbors = SIMILARITY_MAP.get(str(symbol).upper(), [])[:10]
    except Exception as e:
        print(f"Error reading similarity data for {symbol}: {e}")
        return html.Div(
            [
                html.Div("Similarity data could not be loaded.", style={"fontWeight": 600, "color": "#b42318"}),
                html.Div("Refresh the page or rebuild autoencoder_similar_stocks.json to retry.", style=PANEL_MESSAGE_STYLE),
            ],
            style=PANEL_ERROR_STYLE,
        )

    filtered_neighbors = [neighbor for neighbor in neighbors if neighbor.get("ticker")]
    if not filtered_neighbors:
        return html.Div(
            "No autoencoder similarity data is available for this ticker.",
            style=PANEL_MESSAGE_STYLE,
        )

    return html.Div(
        [
            html.Div("Updated from the local similarity map", style=PANEL_TIMESTAMP_STYLE),
            html.Div(
                [
                    html.Button(
                        [
                            html.Span(neighbor.get("ticker", "--"), style={"fontWeight": 600}),
                            html.Span(
                                f"L2 distance: {float(neighbor.get('distance', 0.0)):.4f}",
                                style={"color": "#666", "fontSize": 12},
                            ),
                        ],
                        id={"type": "ticker-select-similar", "index": neighbor.get("ticker", "")},
                        style=SIMILAR_BUTTON_STYLE,
                    )
                    for neighbor in filtered_neighbors
                ],
                style=SIMILAR_LIST_STYLE,
            ),
        ],
        style={"display": "grid", "gap": 8},
    )
    if not symbol:
        return html.Div("Select a ticker to view similar stocks.", style={"color": "#777"})

    neighbors = SIMILARITY_MAP.get(str(symbol).upper(), [])[:10]
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

# Poll cache file; only update Store when content actually changes (avoids re-mounting buttons every 5s)
@app.callback(
    Output("near-200w-store", "data"),
    Input("ticker", "value"),
    Input("interval-near200w", "n_intervals"),
    State("near-200w-store", "data"),
    prevent_initial_call=False,
)
def update_near_200w_store(_ticker, _n_intervals, current_list):
    loading_state = _build_panel_state(
        status="loading",
        message="Scanning for tickers near the 200-week moving average...",
    )
    if not os.path.isfile(_NEAR_200W_CACHE_FILE):
        if not os.path.isfile(_NEAR_200W_LOCK_FILE):
            try:
                with open(_NEAR_200W_LOCK_FILE, "w", encoding="utf-8") as handle:
                    handle.write(_timestamp_now())
                t = threading.Thread(target=_compute_near_200w_ma_to_cache, daemon=True)
                t.start()
            except Exception as e:
                print(f"Error starting 200-week MA scan: {e}")
                return _build_panel_state(
                    status="error",
                    message="Could not start the 200-week MA scan.",
                )
        return current_list if current_list else loading_state
    try:
        with open(_NEAR_200W_CACHE_FILE, encoding="utf-8") as f:
            payload = _normalize_cached_panel_payload(json.load(f), _NEAR_200W_CACHE_FILE)
        if payload != current_list:
            return payload
    except Exception as e:
        print(f"Error reading 200-week MA cache: {e}")
        return _build_panel_state(
            status="error",
            message="Could not read the 200-week MA results.",
        )
    return dash.no_update


# Build near-200w MA panel from Store (only re-renders when Store changes)
@app.callback(
    Output("near-200w-ma", "children"),
    Input("near-200w-store", "data"),
    prevent_initial_call=False,
)
def build_near_200w_panel(ticker_list):
    return _render_async_ticker_panel(
        ticker_list,
        button_type="ticker-select-near200w",
        loading_message="Scanning for tickers near the 200-week moving average...",
        empty_message="No tickers are currently within 10% of the 200-week moving average.",
        error_message="The 200-week moving average scan failed.",
    )
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
    loading_state = _build_panel_state(
        status="loading",
        message="Scanning for tickers below the 50-week moving average...",
    )
    if not os.path.isfile(_BELOW_50W_CACHE_FILE):
        if not os.path.isfile(_BELOW_50W_LOCK_FILE):
            try:
                with open(_BELOW_50W_LOCK_FILE, "w", encoding="utf-8") as handle:
                    handle.write(_timestamp_now())
                t = threading.Thread(target=_compute_below_50w_ma_to_cache, daemon=True)
                t.start()
            except Exception as e:
                print(f"Error starting 50-week MA scan: {e}")
                return _build_panel_state(
                    status="error",
                    message="Could not start the 50-week MA scan.",
                )
        return current_list if current_list else loading_state
    try:
        with open(_BELOW_50W_CACHE_FILE, encoding="utf-8") as f:
            payload = _normalize_cached_panel_payload(json.load(f), _BELOW_50W_CACHE_FILE)
        if payload != current_list:
            return payload
    except Exception as e:
        print(f"Error reading 50-week MA cache: {e}")
        return _build_panel_state(
            status="error",
            message="Could not read the 50-week MA results.",
        )
    return dash.no_update


@app.callback(
    Output("below-50w-ma", "children"),
    Input("below-50w-store", "data"),
    prevent_initial_call=False,
)
def build_below_50w_panel(ticker_list):
    return _render_async_ticker_panel(
        ticker_list,
        button_type="ticker-select-below50w",
        loading_message="Scanning for tickers below the 50-week moving average...",
        empty_message="No tickers are currently below the 50-week moving average.",
        error_message="The 50-week moving average scan failed.",
    )
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


def _write_panel_cache(cache_file: str, payload: Dict[str, object]) -> None:
    with open(cache_file, "w", encoding="utf-8") as handle:
        json.dump(payload, handle)


def _compute_near_200w_ma_to_cache():
    """Run in background: compute get_tickers_near_200w_ma() and write list to cache file."""
    try:
        tickers = get_tickers_near_200w_ma()
        _write_panel_cache(_NEAR_200W_CACHE_FILE, _build_panel_state(tickers, updated_at=_timestamp_now()))
    except Exception as e:
        print(f"Error computing 200-week MA tickers: {e}")
        try:
            _write_panel_cache(
                _NEAR_200W_CACHE_FILE,
                _build_panel_state(
                    status="error",
                    message="The 200-week MA scan failed. Refresh the page to retry.",
                    updated_at=_timestamp_now(),
                ),
            )
        except Exception as write_error:
            print(f"Error writing 200-week MA failure state: {write_error}")
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
        _write_panel_cache(_BELOW_50W_CACHE_FILE, _build_panel_state(tickers, updated_at=_timestamp_now()))
    except Exception as e:
        print(f"Error computing 50-week MA tickers: {e}")
        try:
            _write_panel_cache(
                _BELOW_50W_CACHE_FILE,
                _build_panel_state(
                    status="error",
                    message="The 50-week MA scan failed. Refresh the page to retry.",
                    updated_at=_timestamp_now(),
                ),
            )
        except Exception as write_error:
            print(f"Error writing 50-week MA failure state: {write_error}")
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
    df = pd.read_csv(csv_path)
    return df["ticker"].dropna().astype(str).str.strip().tolist()


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
