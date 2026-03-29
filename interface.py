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
import csv
import json
import os
import threading
from datetime import datetime, timedelta
from typing import List, Dict, Optional
from urllib.parse import quote
import requests as _requests

import numpy as np
import pandas as pd
import yfinance as yf
import plotly.graph_objects as go
import umap
from authlib.integrations.flask_client import OAuth
from dash import Dash, dcc, html, Input, Output, State, ctx, ALL
import dash
from flask import Response, redirect, request, session
from werkzeug.middleware.proxy_fix import ProxyFix

from FormData.Get10k import get_company_facts

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


def _strip_env_value(value: str) -> str:
    cleaned = str(value or "").strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] and cleaned[0] in {"'", '"'}:
        return cleaned[1:-1]
    return cleaned


def _load_local_env_file():
    env_path = os.path.join(os.path.dirname(__file__) or ".", ".env")
    if not os.path.isfile(env_path):
        return

    try:
        with open(env_path, "r", encoding="utf-8") as handle:
            for raw_line in handle:
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue

                key, value = line.split("=", 1)
                key = key.strip()
                if key.lower().startswith("export "):
                    key = key[7:].strip()
                if not key or key in os.environ:
                    continue
                os.environ[key] = _strip_env_value(value)
    except OSError as e:
        print(f"Error loading .env file: {e}")


def _env_value(*names: str, default: Optional[str] = None) -> Optional[str]:
    for name in names:
        value = str(os.environ.get(name) or "").strip()
        if value:
            return value
    return default


_load_local_env_file()

CHAT_SERVICE_URL: str = _env_value("CHAT_SERVICE_URL", default="http://127.0.0.1:8001")
_CHAT_TIMEOUT: int = 60  # seconds before we give up waiting for the model


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


def _load_stock_embedding_map() -> Dict[str, Dict[str, object]]:
    embeddings_path = os.path.join(os.path.dirname(__file__) or ".", "stock_embeddings.json")
    if not os.path.isfile(embeddings_path):
        return {}

    try:
        with open(embeddings_path, "r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception as e:
        print(f"Error loading stock embeddings: {e}")
        return {}

    normalized: Dict[str, Dict[str, object]] = {}
    for raw_symbol, payload in raw.items():
        if not isinstance(payload, dict):
            continue

        symbol = str(raw_symbol or "").strip().upper()
        embedding = payload.get("embedding")
        if not symbol or not isinstance(embedding, list) or not embedding:
            continue

        try:
            vector = np.asarray([float(value) for value in embedding], dtype=float)
        except (TypeError, ValueError):
            continue

        if vector.ndim != 1 or not vector.size or not np.isfinite(vector).all():
            continue

        try:
            reconstruction_score = float(
                payload.get("reconstruction_loss", payload.get("reconstruction_l2", np.nan))
            )
        except (TypeError, ValueError):
            reconstruction_score = float("nan")

        normalized[symbol] = {
            "embedding": vector,
            "num_windows": int(payload.get("num_windows") or 0),
            "reconstruction_score": reconstruction_score,
        }

    return normalized


STOCK_EMBEDDING_MAP = _load_stock_embedding_map()


def _build_embedding_projection() -> tuple[pd.DataFrame, List[str], int]:
    empty_df = pd.DataFrame(
        columns=["ticker", "company_name", "num_windows", "reconstruction_score", "x", "y", "z"]
    )

    # Load from precomputed cache if available (run precompute_umap.py to generate it)
    cache_path = os.path.join(os.path.dirname(__file__) or ".", "umap_projection.json")
    if os.path.isfile(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as handle:
                cached = json.load(handle)
            rows = cached.get("rows", [])
            axis_titles = cached.get("axis_titles", ["UMAP 1", "UMAP 2", "UMAP 3"])
            vector_size = int(cached.get("vector_size", 0))
            if rows:
                return pd.DataFrame(rows), axis_titles, vector_size
        except Exception as e:
            print(f"Warning: could not load umap_projection.json, falling back to live computation: {e}")

    if not STOCK_EMBEDDING_MAP:
        return empty_df, ["UMAP 1", "UMAP 2", "UMAP 3"], 0

    expected_vector_size = 0
    rows = []
    vectors = []
    for symbol in sorted(STOCK_EMBEDDING_MAP):
        payload = STOCK_EMBEDDING_MAP[symbol]
        vector = payload.get("embedding")
        if not isinstance(vector, np.ndarray) or vector.ndim != 1 or not vector.size:
            continue
        if expected_vector_size == 0:
            expected_vector_size = int(vector.size)
        if int(vector.size) != expected_vector_size:
            continue

        rows.append(
            {
                "ticker": symbol,
                "company_name": LISTING_NAME_MAP.get(symbol, ""),
                "num_windows": int(payload.get("num_windows") or 0),
                "reconstruction_score": float(payload.get("reconstruction_score", np.nan)),
            }
        )
        vectors.append(vector)

    if not vectors:
        return empty_df, ["UMAP 1", "UMAP 2", "UMAP 3"], 0

    matrix = np.vstack(vectors).astype(float, copy=False)
    projected = np.zeros((matrix.shape[0], 3), dtype=float)
    if matrix.shape[0] > 3:
        reducer = umap.UMAP(
            n_components=3,
            n_neighbors=max(2, min(15, matrix.shape[0] - 1)),
            min_dist=0.1,
            metric="cosine",
            n_jobs=1,
            random_state=42,
        )
        projected = reducer.fit_transform(matrix)
    elif matrix.shape[0] > 1:
        centered = matrix - matrix.mean(axis=0, keepdims=True)
        component_count = min(3, centered.shape[0], centered.shape[1])
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        projected[:, :component_count] = centered @ vt[:component_count].T

    projection_df = pd.DataFrame(rows)
    projection_df["x"] = projected[:, 0]
    projection_df["y"] = projected[:, 1]
    projection_df["z"] = projected[:, 2]
    return projection_df, ["UMAP 1", "UMAP 2", "UMAP 3"], expected_vector_size


EMBEDDING_PROJECTION_DF, EMBEDDING_AXIS_TITLES, EMBEDDING_VECTOR_SIZE = _build_embedding_projection()
EMBEDDING_TICKER_OPTIONS = [
    _ticker_option(symbol)
    for symbol in EMBEDDING_PROJECTION_DF["ticker"].tolist()
]


def _rank_ticker_options(query: str, options: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Return options ranked by relevance to query.

    Priority (lowest score = shown first):
      0 — exact ticker match
      1 — ticker starts with query
      2 — company name starts with query
      3 — ticker contains query
      4 — company name contains query
    Options that don't match at all are excluded.
    """
    q = query.strip().upper()
    if not q:
        return options

    scored: List[tuple] = []
    for opt in options:
        value = str(opt.get("value", "")).upper()
        label = str(opt.get("label", "")).upper()
        if value == q:
            score = 0
        elif value.startswith(q):
            score = 1
        elif label.startswith(q):
            score = 2
        elif q in value:
            score = 3
        elif q in label:
            score = 4
        else:
            continue
        scored.append((score, value, opt))

    scored.sort(key=lambda x: (x[0], x[1]))
    return [opt for _, _, opt in scored]


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
CHART_MODE_OPTIONS = [
    ("Show normalized returns", "normalized"),
    ("Show log returns", "log"),
]
FINANCIAL_PERIOD_OPTIONS = [
    ("Quarterly", "quarterly"),
    ("Annual", "annual"),
]
SEC_TAXONOMY_REVENUE_FACT_KEYS = {
    "us-gaap": (
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "SalesRevenueNet",
        "Revenues",
        "SalesRevenueServicesNet",
    ),
    "ifrs-full": (
        "Revenue",
        "RevenueAndOperatingIncome",
        "RevenueFromSaleOfElectricity",
        "RevenueFromConstructionContracts",
    ),
}
YFINANCE_REVENUE_ROW_KEYS = (
    "Total Revenue",
    "Operating Revenue",
    "Revenue",
    "Net Sales",
    "Sales",
)
QUARTERLY_REVENUE_LIMIT = 20
ANNUAL_REVENUE_LIMIT = 10

# Inline styles (replacement for the removed <style> block)
CARD_STYLE = {
    "border": "1px solid #e6e6e6",
    "borderRadius": 12,
    "padding": 12,
    "boxShadow": "0 1px 2px rgba(0,0,0,.03)",
}


app = Dash(__name__, suppress_callback_exceptions=True)
app.title = APP_TITLE
server = app.server
server.secret_key = (
    _env_value("FLASK_SECRET_KEY", "SECRET_KEY", "flask_secret_key", "secret_key")
    or "dev-only-secret-key-change-me"
)
server.wsgi_app = ProxyFix(server.wsgi_app, x_for=1, x_proto=1, x_host=1)
server.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SAMESITE="Lax",
)
if str(os.environ.get("FORCE_SECURE_SESSION_COOKIE") or "").strip().lower() in {"1", "true", "yes"}:
    server.config["SESSION_COOKIE_SECURE"] = True

BASE_DIR = os.path.dirname(__file__) or "."
BETA_ALLOWLIST_PATH = os.environ.get(
    "BETA_ALLOWLIST_PATH",
    os.path.join(BASE_DIR, "beta_allowlist.txt"),
)
BETA_WAITLIST_PATH = os.environ.get(
    "BETA_WAITLIST_PATH",
    os.path.join(BASE_DIR, "beta_waitlist.csv"),
)
BETA_ALLOWLIST_ENV = "BETA_ALLOWLIST_EMAILS"
GOOGLE_CLIENT_ID = _env_value(
    "GOOGLE_OAUTH_CLIENT_ID",
    "GOOGLE_CLIENT_ID",
    "google_oauth_clientid",
    "google_client_id",
)
GOOGLE_CLIENT_SECRET = _env_value(
    "GOOGLE_OAUTH_CLIENT_SECRET",
    "GOOGLE_CLIENT_SECRET",
    "google_oauth_clientsecret",
    "google_client_secret",
)
EXTERNAL_BASE_URL = _env_value(
    "EXTERNAL_BASE_URL",
    "PUBLIC_BASE_URL",
    "APP_BASE_URL",
    "external_base_url",
    "public_base_url",
    "app_base_url",
)
GOOGLE_DISCOVERY_URL = "https://accounts.google.com/.well-known/openid-configuration"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SCOPE = "openid email profile"
WAITLIST_LOCK = threading.Lock()

oauth = OAuth(server)
if GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET:
    oauth.register(
        name="google",
        client_id=GOOGLE_CLIENT_ID,
        client_secret=GOOGLE_CLIENT_SECRET,
        server_metadata_url=GOOGLE_DISCOVERY_URL,
        client_kwargs={"scope": GOOGLE_SCOPE},
    )

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


def _transform_chart_series(
    series: pd.Series,
    *,
    normalized: bool,
    log_returns: bool,
) -> Optional[pd.Series]:
    transformed = pd.to_numeric(series, errors="coerce").dropna()
    if transformed.empty:
        return None

    transformed = transformed.astype(float)
    if log_returns:
        transformed = transformed[transformed > 0]
        if len(transformed) < 2:
            return None
        transformed = pd.Series(
            np.log(transformed.to_numpy(dtype=float)),
            index=transformed.index,
        )
        if normalized:
            # Rebase after the log transform when both modes are enabled.
            transformed = (transformed - float(transformed.iloc[0])) * 100.0
        else:
            transformed = transformed.diff().fillna(0.0).cumsum() * 100.0
        return transformed

    if normalized:
        if len(transformed) < 2:
            return None
        first_value = float(transformed.iloc[0])
        if first_value == 0:
            return None
        transformed = ((transformed / first_value) - 1.0) * 100.0

    return transformed


def _chart_mode_title_suffix(*, normalized: bool, log_returns: bool) -> str:
    if normalized and log_returns:
        return " (Normalized Log Returns)"
    if log_returns:
        return " (Log Returns)"
    if normalized:
        return " (Normalized)"
    return ""


def _chart_mode_yaxis_title(*, normalized: bool, log_returns: bool) -> str:
    if normalized or log_returns:
        return "Return (%)" if normalized and not log_returns else "Log Return (%)"
    return "Price ($)"


def _format_money_compact(value) -> str:
    try:
        amount = float(value)
    except Exception:
        return "--"

    sign = "-" if amount < 0 else ""
    amount = abs(amount)
    if amount >= 1e12:
        return f"{sign}${amount/1e12:.2f}T"
    if amount >= 1e9:
        return f"{sign}${amount/1e9:.2f}B"
    if amount >= 1e6:
        return f"{sign}${amount/1e6:.2f}M"
    if amount >= 1e3:
        return f"{sign}${amount/1e3:.1f}K"
    return f"{sign}${amount:,.0f}"


def make_financials_figure(financial_df: Optional[pd.DataFrame], symbol: str, period_label: str) -> go.Figure:
    fig = go.Figure()
    title = f"{symbol} Revenue - {period_label}"

    if financial_df is None or financial_df.empty:
        fig.update_layout(
            annotations=[
                dict(
                    text="Select a ticker to load revenue data." if symbol == "--" else "No revenue data was found for this ticker.",
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
            title=title,
            yaxis_title="Revenue ($)",
        )
        return fig

    customdata = np.column_stack(
        [
            financial_df["period_end"].dt.strftime("%Y-%m-%d"),
            financial_df["filed"].dt.strftime("%Y-%m-%d"),
            financial_df["source"],
        ]
    )
    fig.add_trace(
        go.Bar(
            x=financial_df["label"],
            y=financial_df["value"],
            customdata=customdata,
            marker_color="#0f766e",
            hovertemplate=(
                "%{x}<br>"
                "Period End: %{customdata[0]}<br>"
                "Filed: %{customdata[1]}<br>"
                "Source: %{customdata[2]}<br>"
                "Revenue: %{y:$,.0f}<extra></extra>"
            ),
        )
    )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        title=title,
        xaxis_title=None,
        yaxis_title="Revenue ($)",
        bargap=0.2,
    )
    fig.update_xaxes(tickangle=-35)
    fig.update_yaxes(tickprefix="$", tickformat="~s")
    return fig


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
    log_returns: bool,
) -> go.Figure:
    fig = go.Figure()
    title_suffix = _chart_mode_title_suffix(normalized=normalized, log_returns=log_returns)
    yaxis_title = _chart_mode_yaxis_title(normalized=normalized, log_returns=log_returns)

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
            title=f"Selected Stocks - {timeframe_label}{title_suffix}",
            yaxis_title=yaxis_title,
        )
        return fig

    symbols = list(series_map.keys())
    title_symbols = ", ".join(symbols[:3])
    if len(symbols) > 3:
        title_symbols = f"{title_symbols} +{len(symbols) - 3} more"

    title = f"{title_symbols} - {timeframe_label}"
    if title_suffix:
        title = f"{title}{title_suffix}"

    show_period_return = len(series_map) > 1
    for index, (symbol, series) in enumerate(series_map.items()):
        legend_name = symbol
        if show_period_return:
            try:
                if normalized or log_returns:
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

    if normalized or log_returns:
        fig.add_hline(y=0, line_width=1, line_dash="dot", line_color="#98a2b3")

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=40, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title=None,
        yaxis_title=yaxis_title,
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


def make_stock_clusters_figure(selected_symbols: Optional[List[str]] = None) -> go.Figure:
    fig = go.Figure()

    if EMBEDDING_PROJECTION_DF.empty:
        fig.update_layout(
            annotations=[
                dict(
                    text="No stock embeddings were found in stock_embeddings.json.",
                    x=0.5,
                    y=0.5,
                    xref="paper",
                    yref="paper",
                    showarrow=False,
                    font=dict(size=14, color="#888"),
                )
            ],
            margin=dict(l=0, r=0, t=40, b=0),
            template="plotly_white",
            title="Stock Clusters",
        )
        return fig

    highlighted_symbols = set(_normalize_symbol_selection(selected_symbols))
    base_df = EMBEDDING_PROJECTION_DF[~EMBEDDING_PROJECTION_DF["ticker"].isin(highlighted_symbols)]
    highlighted_df = EMBEDDING_PROJECTION_DF[EMBEDDING_PROJECTION_DF["ticker"].isin(highlighted_symbols)]

    def _customdata(df: pd.DataFrame) -> np.ndarray:
        reconstruction_labels = df["reconstruction_score"].map(
            lambda value: f"{float(value):.4f}" if pd.notna(value) else "--"
        )
        return np.column_stack(
            [
                df["ticker"],
                df["company_name"].replace("", "--"),
                df["num_windows"].map(lambda value: f"{int(value):,}"),
                reconstruction_labels,
            ]
        )

    hovertemplate = (
        "<b>%{customdata[0]}</b><br>"
        "Company: %{customdata[1]}<br>"
        "Training windows: %{customdata[2]}<br>"
        "Reconstruction Score: %{customdata[3]}<br>"
        "X: %{x:.2f}<br>"
        "Y: %{y:.2f}<br>"
        "Z: %{z:.2f}<extra></extra>"
    )

    if not base_df.empty:
        fig.add_trace(
            go.Scatter3d(
                x=base_df["x"],
                y=base_df["y"],
                z=base_df["z"],
                mode="markers",
                name="Embedded stocks",
                customdata=_customdata(base_df),
                hovertemplate=hovertemplate,
                marker=dict(size=3, color="#2563eb", opacity=0.5),
            )
        )

    if not highlighted_df.empty:
        show_labels = len(highlighted_df) <= 12
        fig.add_trace(
            go.Scatter3d(
                x=highlighted_df["x"],
                y=highlighted_df["y"],
                z=highlighted_df["z"],
                mode="markers+text" if show_labels else "markers",
                text=highlighted_df["ticker"] if show_labels else None,
                textposition="top center",
                name="Highlighted tickers",
                customdata=_customdata(highlighted_df),
                hovertemplate=hovertemplate,
                marker=dict(
                    size=7,
                    color="#f97316",
                    opacity=0.95,
                    line=dict(color="#ffffff", width=1),
                ),
            )
        )

    fig.update_layout(
        template="plotly_white",
        margin=dict(l=0, r=0, t=40, b=0),
        title="Stock Clusters",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        scene=dict(
            aspectmode="data",
            dragmode="turntable",
            xaxis=dict(title=EMBEDDING_AXIS_TITLES[0], backgroundcolor="#f8fafc"),
            yaxis=dict(title=EMBEDDING_AXIS_TITLES[1], backgroundcolor="#f8fafc"),
            zaxis=dict(title=EMBEDDING_AXIS_TITLES[2], backgroundcolor="#f8fafc"),
        ),
        uirevision="stock-clusters-view",
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
HEADER_BAR_STYLE = {
    "display": "flex",
    "flexWrap": "wrap",
    "justifyContent": "space-between",
    "alignItems": "flex-start",
    "gap": 12,
}
PAGE_NAV_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": 8, "alignItems": "center"}
NAV_LINK_STYLE = {
    "padding": "8px 12px",
    "border": "1px solid #d0d5dd",
    "borderRadius": 999,
    "color": "#344054",
    "textDecoration": "none",
    "fontWeight": 600,
    "background": "#ffffff",
}
NAV_LINK_ACTIVE_STYLE = {"background": "#101828", "borderColor": "#101828", "color": "#ffffff"}
AUTH_PILL_STYLE = {
    "display": "flex",
    "alignItems": "center",
    "gap": 10,
    "padding": "8px 12px",
    "border": "1px solid #d0d5dd",
    "borderRadius": 999,
    "background": "#ffffff",
    "color": "#344054",
}
AUTH_EMAIL_STYLE = {"fontWeight": 600, "fontSize": 13}
AUTH_ACTION_LINK_STYLE = {
    "color": "#344054",
    "fontSize": 13,
    "fontWeight": 600,
    "textDecoration": "none",
}
GATE_PAGE_STYLE = {
    "minHeight": "100vh",
    "display": "flex",
    "alignItems": "center",
    "justifyContent": "center",
    "padding": 24,
    "boxSizing": "border-box",
    "background": "#f8fafc",
}
GATE_CARD_STYLE = {
    "maxWidth": 560,
    "width": "100%",
    "background": "#ffffff",
    "border": "1px solid #e4e7ec",
    "borderRadius": 20,
    "boxShadow": "0 10px 30px rgba(16,24,40,.08)",
    "padding": 28,
    "display": "grid",
    "gap": 18,
}
GATE_BADGE_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "gap": 8,
    "padding": "6px 10px",
    "borderRadius": 999,
    "background": "#ecfdf3",
    "border": "1px solid #d1fadf",
    "color": "#027a48",
    "fontSize": 12,
    "fontWeight": 700,
    "letterSpacing": "0.04em",
    "textTransform": "uppercase",
}
GATE_BUTTON_ROW_STYLE = {"display": "flex", "flexWrap": "wrap", "gap": 10}
PRIMARY_BUTTON_LINK_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "padding": "12px 18px",
    "borderRadius": 12,
    "background": "#101828",
    "border": "1px solid #101828",
    "color": "#ffffff",
    "fontWeight": 700,
    "textDecoration": "none",
}
SECONDARY_BUTTON_LINK_STYLE = {
    "display": "inline-flex",
    "alignItems": "center",
    "justifyContent": "center",
    "padding": "12px 18px",
    "borderRadius": 12,
    "background": "#ffffff",
    "border": "1px solid #d0d5dd",
    "color": "#344054",
    "fontWeight": 700,
    "textDecoration": "none",
}
GATE_STATUS_BOX_STYLE = {
    "display": "grid",
    "gap": 8,
    "padding": "14px 16px",
    "borderRadius": 14,
    "background": "#f8fafc",
    "border": "1px solid #e4e7ec",
}
GATE_HELP_TEXT_STYLE = {"fontSize": 13, "color": "#667085", "lineHeight": 1.6}
CLUSTERS_GRAPH_STYLE = {"width": "100%", "height": "70vh", "minHeight": 420}
CLUSTERS_NOTE_STYLE = {
    "fontSize": 13,
    "color": "#475467",
    "lineHeight": 1.6,
    "padding": "10px 12px",
    "borderRadius": 10,
    "background": "#f8fafc",
    "border": "1px solid #e4e7ec",
}


def _normalize_email(value: Optional[str]) -> str:
    return str(value or "").strip().lower()


def _is_gmail_address(email: Optional[str]) -> bool:
    normalized = _normalize_email(email)
    return normalized.endswith("@gmail.com") or normalized.endswith("@googlemail.com")


def _safe_internal_path(path: Optional[str]) -> str:
    normalized = str(path or "/").strip() or "/"
    if not normalized.startswith("/") or normalized.startswith("//"):
        return "/"
    if normalized.startswith("/_dash") or normalized.startswith("/auth/"):
        return "/"
    return normalized


def _set_post_login_redirect(path: Optional[str]):
    session["post_login_redirect"] = _safe_internal_path(path)


def _consume_post_login_redirect(default: str = "/") -> str:
    return _safe_internal_path(session.pop("post_login_redirect", default))


def _google_oauth_enabled() -> bool:
    return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET and hasattr(oauth, "google"))


def _load_allowlist_emails() -> set[str]:
    emails: set[str] = set()

    env_value = str(os.environ.get(BETA_ALLOWLIST_ENV) or "").replace(";", ",")
    for raw_email in env_value.split(","):
        normalized = _normalize_email(raw_email)
        if normalized:
            emails.add(normalized)

    if not os.path.isfile(BETA_ALLOWLIST_PATH):
        return emails

    try:
        with open(BETA_ALLOWLIST_PATH, "r", encoding="utf-8") as handle:
            for line in handle:
                normalized = _normalize_email(line.split("#", 1)[0])
                if normalized:
                    emails.add(normalized)
    except OSError as e:
        print(f"Error loading beta allowlist: {e}")

    return emails


def _waitlist_entry_for_email(email: Optional[str]) -> Optional[Dict[str, str]]:
    normalized_email = _normalize_email(email)
    if not normalized_email or not os.path.isfile(BETA_WAITLIST_PATH):
        return None

    try:
        with open(BETA_WAITLIST_PATH, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if _normalize_email(row.get("email")) == normalized_email:
                    return {key: str(value or "") for key, value in row.items()}
    except OSError as e:
        print(f"Error loading beta waitlist: {e}")

    return None


def _append_waitlist_entry(email: str, name: Optional[str] = None):
    normalized_email = _normalize_email(email)
    if not normalized_email:
        return

    with WAITLIST_LOCK:
        if _waitlist_entry_for_email(normalized_email):
            return

        parent_dir = os.path.dirname(BETA_WAITLIST_PATH)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

        fieldnames = ["email", "name", "requested_at_utc", "status"]
        file_exists = os.path.isfile(BETA_WAITLIST_PATH)
        try:
            with open(BETA_WAITLIST_PATH, "a", encoding="utf-8", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                if not file_exists:
                    writer.writeheader()
                writer.writerow(
                    {
                        "email": normalized_email,
                        "name": str(name or "").strip(),
                        "requested_at_utc": datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
                        "status": "pending",
                    }
                )
        except OSError as e:
            print(f"Error writing beta waitlist: {e}")


def _current_user_email() -> str:
    return _normalize_email(session.get("user_email"))


def _is_email_allowlisted(email: Optional[str]) -> bool:
    normalized_email = _normalize_email(email)
    return bool(normalized_email and normalized_email in _load_allowlist_emails())


def _current_user_is_allowed() -> bool:
    return _is_email_allowlisted(_current_user_email())


def _sign_in_href() -> str:
    next_path = quote(_safe_internal_path(session.get("post_login_redirect", "/")), safe="/")
    return f"/auth/google/login?next={next_path}"


def _build_external_url(path: str) -> str:
    normalized_path = "/" + str(path or "/").lstrip("/")
    if EXTERNAL_BASE_URL:
        return EXTERNAL_BASE_URL.rstrip("/") + normalized_path
    return request.url_root.rstrip("/") + normalized_path


def _normalize_pathname(pathname: Optional[str]) -> str:
    normalized = str(pathname or "/").strip() or "/"
    if normalized != "/":
        normalized = normalized.rstrip("/")
    return normalized or "/"


def _build_nav_link(label: str, href: str, pathname: Optional[str]):
    style = dict(NAV_LINK_STYLE)
    if _normalize_pathname(pathname) == href:
        style.update(NAV_LINK_ACTIVE_STYLE)
    return dcc.Link(label, href=href, style=style)


def _build_auth_controls():
    email = _current_user_email()
    if not email:
        return html.A("Sign In", href=_sign_in_href(), style=SECONDARY_BUTTON_LINK_STYLE)

    return html.Div(
        [
            html.Div(email, style=AUTH_EMAIL_STYLE),
            html.A("Log Out", href="/logout", style=AUTH_ACTION_LINK_STYLE),
        ],
        style=AUTH_PILL_STYLE,
    )


def _build_page_header(title: str, subtitle: str, pathname: Optional[str]):
    return html.Div(
        [
            html.Div(
                [
                    html.H1(title, style={"margin": 0}),
                    html.Div(subtitle, style={"color": "#666"}),
                ],
                style=HEADER_STYLE,
            ),
            html.Div(
                [
                    _build_nav_link("Stock Explorer", "/", pathname),
                    _build_nav_link("Stock Clusters", "/stock_clusters", pathname),
                    _build_auth_controls(),
                ],
                style=PAGE_NAV_STYLE,
            ),
        ],
        style=HEADER_BAR_STYLE,
    )


def build_beta_gate_page():
    email = _current_user_email()
    waitlist_entry = _waitlist_entry_for_email(email) if email else None
    requested_at = str((waitlist_entry or {}).get("requested_at_utc") or "").strip()
    auth_error = str(session.pop("auth_error", "") or "").strip()
    oauth_ready = _google_oauth_enabled()

    status_children = []
    if auth_error:
        status_children.append(
            html.Div(
                auth_error,
                style={
                    **GATE_STATUS_BOX_STYLE,
                    "background": "#fef3f2",
                    "border": "1px solid #fecdca",
                    "color": "#b42318",
                },
            )
        )

    if email and not _is_gmail_address(email):
        status_children.append(
            html.Div(
                [
                    html.Div("Use a Gmail account", style={"fontWeight": 700}),
                    html.Div(
                        f"{email} signed in successfully, but beta access is limited to Gmail accounts.",
                        style=GATE_HELP_TEXT_STYLE,
                    ),
                ],
                style=GATE_STATUS_BOX_STYLE,
            )
        )
    elif email:
        request_label = f"Requested on {requested_at}" if requested_at else "Your request has been recorded."
        status_children.append(
            html.Div(
                [
                    html.Div("Waitlist request received", style={"fontWeight": 700}),
                    html.Div(
                        f"{email} is not approved yet. Once you add it to the beta allowlist, a refresh will unlock the app.",
                        style=GATE_HELP_TEXT_STYLE,
                    ),
                    html.Div(request_label, style=GATE_HELP_TEXT_STYLE),
                ],
                style=GATE_STATUS_BOX_STYLE,
            )
        )
    elif not oauth_ready:
        status_children.append(
            html.Div(
                [
                    html.Div("Google sign-in is not configured", style={"fontWeight": 700}),
                    html.Div(
                        "Set GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET before sharing the beta publicly.",
                        style=GATE_HELP_TEXT_STYLE,
                    ),
                ],
                style=GATE_STATUS_BOX_STYLE,
            )
        )
    else:
        status_children.append(
            html.Div(
                [
                    html.Div("Private beta access", style={"fontWeight": 700}),
                    html.Div(
                        "Sign in with Google. Approved Gmail addresses get immediate access, and everyone else is added to the waitlist automatically.",
                        style=GATE_HELP_TEXT_STYLE,
                    ),
                ],
                style=GATE_STATUS_BOX_STYLE,
            )
        )

    action_links = []
    if oauth_ready and not email:
        action_links.append(html.A("Continue With Google", href=_sign_in_href(), style=PRIMARY_BUTTON_LINK_STYLE))
    if email and _is_gmail_address(email):
        action_links.append(html.A("Refresh Access", href=_safe_internal_path(session.get("post_login_redirect", "/")), style=PRIMARY_BUTTON_LINK_STYLE))
    if email:
        action_links.append(html.A("Use Another Account", href="/logout", style=SECONDARY_BUTTON_LINK_STYLE))

    return html.Div(
        [
            html.Div(
                [
                    html.Div("Beta Access", style=GATE_BADGE_STYLE),
                    html.Div(
                        [
                            html.H1("Stocks Explorer Private Beta", style={"margin": 0}),
                            html.Div(
                                "Collect Gmail signups first, then approve testers by adding their addresses to the allowlist.",
                                style={"color": "#475467", "fontSize": 15, "lineHeight": 1.6},
                            ),
                        ],
                        style={"display": "grid", "gap": 8},
                    ),
                    *status_children,
                    html.Div(action_links, style=GATE_BUTTON_ROW_STYLE),
                    html.Div(
                        [
                            html.Div("How it works", style={"fontWeight": 700}),
                            html.Div(
                                "Anyone who signs in with an unapproved Gmail account is written to beta_waitlist.csv. Approve access by adding that email to beta_allowlist.txt or the BETA_ALLOWLIST_EMAILS environment variable.",
                                style=GATE_HELP_TEXT_STYLE,
                            ),
                        ],
                        style=GATE_STATUS_BOX_STYLE,
                    ),
                ],
                style=GATE_CARD_STYLE,
            )
        ],
        style=GATE_PAGE_STYLE,
    )


def _build_stock_clusters_summary(selected_symbols: Optional[List[str]] = None):
    if EMBEDDING_PROJECTION_DF.empty:
        return html.Div("No stock embeddings are available to project.", style=PANEL_MESSAGE_STYLE)

    available_symbols = set(EMBEDDING_PROJECTION_DF["ticker"].tolist())
    normalized_symbols = _normalize_symbol_selection(selected_symbols)
    highlighted_symbols = [symbol for symbol in normalized_symbols if symbol in available_symbols]
    missing_symbols = [symbol for symbol in normalized_symbols if symbol not in available_symbols]

    if not highlighted_symbols:
        return html.Div(
            f"Showing {len(EMBEDDING_PROJECTION_DF):,} embedded stocks. Rotate, zoom, and hover over points to inspect clusters. Use the selector above or click points in the plot to highlight specific tickers.",
            style=PANEL_MESSAGE_STYLE,
        )

    summary = f"Highlighted {len(highlighted_symbols)} ticker(s): {', '.join(highlighted_symbols[:12])}"
    if len(highlighted_symbols) > 12:
        summary = f"{summary}, +{len(highlighted_symbols) - 12} more. Labels are hidden when many tickers are selected to keep the plot readable."
    if missing_symbols:
        summary = f"{summary} Missing from the embedding set: {', '.join(missing_symbols)}."
    return html.Div(summary, style=PANEL_MESSAGE_STYLE)

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

def build_stock_explorer_page(pathname: Optional[str]):
    return html.Div(
        [
            _build_page_header(
                APP_TITLE,
                "Explore unfamiliar tickers with price history and quick context.",
                pathname,
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
                                            id="chart-mode-toggle",
                                            options=[
                                                {"label": label, "value": value}
                                                for label, value in CHART_MODE_OPTIONS
                                            ],
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
                        html.Div(
                            [
                                html.H3("Company Financials", style={"marginTop": 0, "marginBottom": 10}),
                                html.Div(
                                    [
                                        html.Div("Revenue from SEC filings with a Yahoo Finance fallback.", style=CONTROL_HINT_STYLE),
                                        dcc.RadioItems(
                                            id="financial-period",
                                            options=[{"label": label, "value": value} for label, value in FINANCIAL_PERIOD_OPTIONS],
                                            value="quarterly",
                                            style=RADIO_ITEMS_STYLE,
                                            labelStyle=RADIO_LABEL_STYLE,
                                        ),
                                        html.Div(id="financials-caption", style=CONTROL_HINT_STYLE),
                                    ],
                                    style={"display": "grid", "gap": 8, "marginBottom": 10},
                                ),
                                dcc.Loading(
                                    dcc.Graph(
                                        id="financials-chart",
                                        figure=make_financials_figure(None, "--", "Quarterly"),
                                        config={"displaylogo": False, "responsive": True},
                                        style={"width": "100%"},
                                    ),
                                    type="default",
                                ),
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


def build_stock_clusters_page(pathname: Optional[str]):
    return html.Div(
        [
            _build_page_header(
                "Stock Clusters",
                "Inspect the local stock embeddings in a rotatable 3D UMAP projection and identify clusters visually.",
                pathname,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Stocks projected", style=SUMMARY_LABEL_STYLE),
                            html.Div(f"{len(EMBEDDING_PROJECTION_DF):,}", style=SUMMARY_VALUE_STYLE),
                        ],
                        style=SUMMARY_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Div("Embedding dimensions", style=SUMMARY_LABEL_STYLE),
                            html.Div(str(EMBEDDING_VECTOR_SIZE or "--"), style=SUMMARY_VALUE_STYLE),
                        ],
                        style=SUMMARY_CARD_STYLE,
                    ),
                    html.Div(
                        [
                            html.Div("Projection", style=SUMMARY_LABEL_STYLE),
                            html.Div("3D UMAP", style=SUMMARY_VALUE_STYLE),
                        ],
                        style=SUMMARY_CARD_STYLE,
                    ),
                ],
                style=PRICE_SUMMARY_ROW_STYLE,
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Highlight tickers", style=CONTROL_LABEL_STYLE),
                            html.Div("Search for stocks to label on the 3D map.", style=CONTROL_HINT_STYLE),
                            dcc.Dropdown(
                                id="stock-clusters-ticker",
                                options=EMBEDDING_TICKER_OPTIONS,
                                value=[],
                                multi=True,
                                placeholder="Start typing a ticker or company name...",
                                maxHeight=420,
                                style=DROPDOWN_STYLE,
                            ),
                        ],
                        style=SEARCH_BLOCK_STYLE,
                    ),
                    html.Div(
                        "The plot uses the local autoencoder embeddings from stock_embeddings.json and projects them into three UMAP dimensions. Hover points for details, rotate the scene to inspect neighborhood structure, and click a point to add it to the highlighted set.",
                        style=CLUSTERS_NOTE_STYLE,
                    ),
                ],
                style=CONTROL_STACK_STYLE,
            ),
            html.Div(
                [
                    dcc.Loading(
                        dcc.Graph(
                            id="stock-clusters-graph",
                            figure=make_stock_clusters_figure([]),
                            config={"displaylogo": False, "responsive": True},
                            style=CLUSTERS_GRAPH_STYLE,
                        ),
                        type="default",
                    ),
                    html.Div(
                        id="stock-clusters-selection-summary",
                        children=_build_stock_clusters_summary([]),
                    ),
                ],
                style=CARD_STYLE,
            ),
        ],
        style=PAGE_CONTAINER_STYLE,
    )


@server.before_request
def _protect_beta_endpoints():
    path = request.path or "/"
    if (
        request.method == "GET"
        and not path.startswith("/_dash")
        and not path.startswith("/auth/")
        and not path.startswith("/assets/")
        and not path.startswith("/_favicon")
        and path not in {"/favicon.ico", "/logout"}
    ):
        _set_post_login_redirect(path)

    if path == "/_dash-update-component" and not _current_user_is_allowed():
        return Response("Forbidden", status=403)

    return None


@server.route("/auth/google/login")
def google_login():
    next_path = request.args.get("next")
    if next_path:
        _set_post_login_redirect(next_path)

    if not _google_oauth_enabled():
        session["auth_error"] = "Google sign-in is not configured yet."
        return redirect(_consume_post_login_redirect("/"))

    redirect_uri = _build_external_url("/auth/google/callback")
    return oauth.google.authorize_redirect(redirect_uri)


@server.route("/auth/google/callback")
def google_callback():
    if not _google_oauth_enabled():
        session["auth_error"] = "Google sign-in is not configured yet."
        return redirect(_consume_post_login_redirect("/"))

    try:
        token = oauth.google.authorize_access_token()
        userinfo = token.get("userinfo")
        if not isinstance(userinfo, dict):
            userinfo = oauth.google.get(GOOGLE_USERINFO_URL).json()
    except Exception as e:
        print(f"Google OAuth error: {e}")
        session.pop("user_email", None)
        session.pop("user_name", None)
        session["auth_error"] = "Google sign-in failed. Please try again."
        return redirect(_consume_post_login_redirect("/"))

    email = _normalize_email(userinfo.get("email"))
    display_name = str(userinfo.get("name") or "").strip()
    email_verified = bool(userinfo.get("email_verified"))

    session["user_email"] = email
    session["user_name"] = display_name
    session.pop("auth_error", None)

    if not email or not email_verified:
        session["auth_error"] = "Google did not return a verified email address."
        return redirect(_consume_post_login_redirect("/"))

    if not _is_gmail_address(email):
        session["auth_error"] = "Please sign in with a Gmail account."
        return redirect(_consume_post_login_redirect("/"))

    if not _is_email_allowlisted(email):
        _append_waitlist_entry(email, display_name)

    return redirect(_consume_post_login_redirect("/"))


@server.route("/logout")
def logout():
    next_path = _safe_internal_path(request.args.get("next") or session.get("post_login_redirect") or "/")
    session.pop("user_email", None)
    session.pop("user_name", None)
    session.pop("auth_error", None)
    _set_post_login_redirect(next_path)
    return redirect(next_path)


def _build_chat_shell():
    """Fixed chat bubble + slide-up panel, rendered once at the top level."""
    return html.Div(
        [
            # ── Stores ──────────────────────────────────────────────────────
            dcc.Store(id="chat-open", data=False),
            dcc.Store(id="chat-history", data=[]),

            # ── Floating bubble button ───────────────────────────────────
            html.Button(
                "💬",
                id="chat-bubble-btn",
                title="Open finance assistant",
                n_clicks=0,
            ),

            # ── Slide-up panel ───────────────────────────────────────────
            html.Div(
                [
                    # Header
                    html.Div(
                        [
                            html.P("Finance Assistant", id="chat-panel-title"),
                            html.Button("✕", id="chat-close-btn", n_clicks=0),
                        ],
                        id="chat-panel-header",
                    ),
                    # Message area (wrapped in dcc.Loading for spinner)
                    dcc.Loading(
                        html.Div([], id="chat-messages"),
                        type="circle",
                        color="#2563eb",
                    ),
                    # Input row
                    html.Div(
                        [
                            dcc.Textarea(
                                id="chat-input",
                                placeholder="Ask about a stock or company…",
                                rows=1,
                            ),
                            html.Button("➤", id="chat-send-btn", n_clicks=0),
                        ],
                        id="chat-input-row",
                    ),
                ],
                id="chat-panel",
                className="",
            ),
        ],
        id="chat-shell",
        style={"position": "fixed", "zIndex": 1100},
    )


def serve_layout():
    if _current_user_is_allowed():
        return html.Div(
            [
                dcc.Location(id="url", refresh=False),
                html.Div(id="page-content"),
                _build_chat_shell(),
            ]
        )
    return build_beta_gate_page()


app.layout = serve_layout


@app.callback(
    Output("page-content", "children"),
    Input("url", "pathname"),
    prevent_initial_call=False,
)
def render_page(pathname: Optional[str]):
    normalized_path = _normalize_pathname(pathname)
    if normalized_path == "/stock_clusters":
        return build_stock_clusters_page(normalized_path)
    return build_stock_explorer_page(normalized_path)


# ------------------------------------------------------------
# Callbacks — these call your functions (stubs below)
# ------------------------------------------------------------

@app.callback(
    Output("ticker", "options"),
    Input("ticker", "search_value"),
    State("ticker", "value"),
    prevent_initial_call=True,
)
def rank_ticker_search(search_value: Optional[str], current_value):
    q = (search_value or "").strip()
    if not q:
        return tickers
    ranked = _rank_ticker_options(q, tickers)
    # Always include currently selected values so they aren't lost
    if current_value:
        selected = [current_value] if isinstance(current_value, str) else list(current_value)
        ranked_values = {opt["value"] for opt in ranked}
        extras = [opt for opt in tickers if opt["value"] in selected and opt["value"] not in ranked_values]
        ranked = extras + ranked
    return ranked


@app.callback(
    Output("stock-clusters-ticker", "options"),
    Input("stock-clusters-ticker", "search_value"),
    State("stock-clusters-ticker", "value"),
    prevent_initial_call=True,
)
def rank_clusters_ticker_search(search_value: Optional[str], current_value):
    q = (search_value or "").strip()
    if not q:
        return EMBEDDING_TICKER_OPTIONS
    ranked = _rank_ticker_options(q, EMBEDDING_TICKER_OPTIONS)
    if current_value:
        selected = [current_value] if isinstance(current_value, str) else list(current_value)
        ranked_values = {opt["value"] for opt in ranked}
        extras = [opt for opt in EMBEDDING_TICKER_OPTIONS if opt["value"] in selected and opt["value"] not in ranked_values]
        ranked = extras + ranked
    return ranked


@app.callback(
    Output("stock-clusters-graph", "figure"),
    Output("stock-clusters-selection-summary", "children"),
    Input("stock-clusters-ticker", "value"),
    prevent_initial_call=False,
)
def update_stock_clusters(selected_symbols):
    normalized_symbols = _normalize_symbol_selection(selected_symbols)
    return (
        make_stock_clusters_figure(normalized_symbols),
        _build_stock_clusters_summary(normalized_symbols),
    )


@app.callback(
    Output("stock-clusters-ticker", "value"),
    Input("stock-clusters-graph", "clickData"),
    State("stock-clusters-ticker", "value"),
    prevent_initial_call=True,
)
def add_clicked_stock_cluster_ticker(click_data, current_selection):
    if not click_data or not click_data.get("points"):
        return dash.no_update

    point = click_data["points"][0]
    customdata = point.get("customdata") or []
    ticker = str(customdata[0] if len(customdata) > 0 else "").strip().upper()
    if not ticker:
        return dash.no_update

    selected_symbols = _normalize_symbol_selection(current_selection)
    if ticker in selected_symbols:
        return dash.no_update
    return selected_symbols + [ticker]


@app.callback(
    Output("price-chart", "figure"),
    Input("ticker", "value"),
    Input("timeframe", "value"),
    Input("ma-checklist", "value"),
    Input("chart-mode-toggle", "value"),
    prevent_initial_call=False,
)
def update_chart(selected_symbols, timeframe: str, selected_ma: Optional[List[str]], chart_mode_toggle: Optional[List[str]]):
    symbols = _normalize_symbol_selection(selected_symbols)
    chart_modes = set(chart_mode_toggle or [])
    normalized = "normalized" in chart_modes
    log_returns = "log" in chart_modes
    if not symbols:
        return make_multi_symbol_figure({}, "--", normalized=normalized, log_returns=log_returns)

    label = next((lab for lab, val in TIMEFRAMES if val == timeframe), timeframe)
    if len(symbols) == 1 and not normalized and not log_returns:
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
            transformed_series = _transform_chart_series(
                series,
                normalized=normalized,
                log_returns=log_returns,
            )
            if transformed_series is None:
                continue
            series_map[symbol] = transformed_series
        except Exception as e:
            print(f"Error loading comparison data for {symbol}: {e}")

    return make_multi_symbol_figure(
        series_map,
        label,
        normalized=normalized,
        log_returns=log_returns,
    )


@app.callback(
    Output("ma-checklist", "options"),
    Output("ma-checklist", "value"),
    Output("ma-control-block", "style"),
    Input("ticker", "value"),
    Input("chart-mode-toggle", "value"),
    State("ma-checklist", "value"),
    prevent_initial_call=False,
)
def sync_ma_controls(selected_symbols, chart_mode_toggle, selected_ma):
    symbols = _normalize_symbol_selection(selected_symbols)
    multi_select_active = len(symbols) > 1
    derived_chart_mode_active = bool(chart_mode_toggle)
    disable_ma = multi_select_active or derived_chart_mode_active
    options = [
        {"label": label, "value": value, "disabled": disable_ma}
        for label, value, _ in MA_OPTIONS
    ]
    control_style = dict(WIDE_CONTROL_BLOCK_STYLE)
    if disable_ma:
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
        f"Market Cap: {_format_money_compact(cap)}",
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


@app.callback(
    Output("financials-chart", "figure"),
    Output("financials-caption", "children"),
    Input("ticker", "value"),
    Input("financial-period", "value"),
    prevent_initial_call=False,
)
def update_financials_chart(symbol, period: str):
    primary_symbol = _primary_selected_symbol(symbol)
    period_label = "Quarterly" if period == "quarterly" else "Annual"
    if not primary_symbol:
        return make_financials_figure(None, "--", period_label), "Select a ticker to view company revenue."

    try:
        financial_df, concept_label, source_label = get_company_revenue(primary_symbol, period)
    except Exception as e:
        print(f"Error loading SEC revenue for {primary_symbol}: {e}")
        return (
            make_financials_figure(None, primary_symbol, period_label),
            "The revenue request failed for this ticker.",
        )

    if financial_df.empty:
        return (
            make_financials_figure(None, primary_symbol, period_label),
            "No revenue series was available for this ticker.",
        )

    latest_row = financial_df.iloc[-1]
    caption = (
        f"Source: {source_label} ({concept_label})  |  "
        f"Latest: {_format_money_compact(latest_row['value'])} for {latest_row['label']}  |  "
        f"Filed {latest_row['filed'].strftime('%Y-%m-%d')}"
    )
    return make_financials_figure(financial_df, primary_symbol, period_label), caption


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


def _coerce_sec_date(value) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        timestamp = pd.Timestamp(value)
    except Exception:
        return None
    if pd.isna(timestamp):
        return None
    return timestamp.normalize()


def _latest_sec_record(records: List[Dict[str, object]]) -> Optional[Dict[str, object]]:
    if not records:
        return None

    def sort_key(record: Dict[str, object]):
        filed = record.get("filed") or pd.Timestamp.min
        period_end = record.get("end") or pd.Timestamp.min
        duration_days = record.get("duration_days")
        return (filed, period_end, duration_days if duration_days is not None else -1)

    return max(records, key=sort_key)


def _select_sec_records(
    records: List[Dict[str, object]],
    *,
    fp_values: Optional[set] = None,
    min_days: Optional[int] = None,
    max_days: Optional[int] = None,
    forms: Optional[set] = None,
) -> List[Dict[str, object]]:
    selected: List[Dict[str, object]] = []
    for record in records:
        if forms and record["form"] not in forms:
            continue
        if fp_values and record["fp"] not in fp_values:
            continue
        duration_days = record.get("duration_days")
        if min_days is not None and (duration_days is None or duration_days < min_days):
            continue
        if max_days is not None and (duration_days is None or duration_days > max_days):
            continue
        selected.append(record)
    return selected


def _parse_revenue_fact_records(unit_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in unit_rows or []:
        try:
            value = float(row.get("val"))
        except Exception:
            continue

        period_end = _coerce_sec_date(row.get("end"))
        if period_end is None:
            continue

        period_start = _coerce_sec_date(row.get("start"))
        filed = _coerce_sec_date(row.get("filed")) or period_end
        duration_days = None
        if period_start is not None and period_end > period_start:
            duration_days = int((period_end - period_start).days)

        records.append(
            {
                "value": value,
                "start": period_start,
                "end": period_end,
                "filed": filed,
                "duration_days": duration_days,
                "form": str(row.get("form") or "").strip().upper(),
                "fy": str(row.get("fy") or period_end.year),
                "fp": str(row.get("fp") or "").strip().upper(),
            }
        )
    return records


def _build_annual_revenue_df(records: List[Dict[str, object]]) -> pd.DataFrame:
    annual_forms = {"10-K", "10-K/A", "20-F", "20-F/A", "40-F", "40-F/A"}
    annual_candidates = _select_sec_records(records, min_days=340, max_days=380, forms=annual_forms)
    by_fiscal_year: Dict[str, List[Dict[str, object]]] = {}
    for record in annual_candidates:
        by_fiscal_year.setdefault(record["fy"], []).append(record)

    rows = []
    for fiscal_year, year_records in sorted(by_fiscal_year.items()):
        chosen = _latest_sec_record(year_records)
        if not chosen:
            continue
        rows.append(
            {
                "label": f"FY {fiscal_year}",
                "value": chosen["value"],
                "period_end": chosen["end"],
                "filed": chosen["filed"],
                "source": "reported annual",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["label", "value", "period_end", "filed", "source"])

    df = pd.DataFrame(rows).sort_values("period_end").tail(10).reset_index(drop=True)
    return df


def _build_quarterly_revenue_df(records: List[Dict[str, object]]) -> pd.DataFrame:
    quarterly_forms = {"10-Q", "10-Q/A", "10-K", "10-K/A", "6-K", "6-K/A"}
    fiscal_years = sorted({record["fy"] for record in records if record["form"] in quarterly_forms})
    rows: List[Dict[str, object]] = []

    def add_row(quarter_name: str, fiscal_year: str, value, period_end, filed, source: str) -> None:
        if value is None or period_end is None:
            return
        rows.append(
            {
                "label": f"{quarter_name} {fiscal_year}",
                "value": float(value),
                "period_end": period_end,
                "filed": filed or period_end,
                "source": source,
            }
        )

    for fiscal_year in fiscal_years:
        year_records = [record for record in records if record["fy"] == fiscal_year and record["form"] in quarterly_forms]

        q1 = _latest_sec_record(_select_sec_records(year_records, fp_values={"Q1"}, min_days=80, max_days=110))
        q2 = _latest_sec_record(_select_sec_records(year_records, fp_values={"Q2"}, min_days=80, max_days=110))
        q2_ytd = _latest_sec_record(_select_sec_records(year_records, fp_values={"Q2"}, min_days=170, max_days=210))
        q3 = _latest_sec_record(_select_sec_records(year_records, fp_values={"Q3"}, min_days=80, max_days=110))
        q3_ytd = _latest_sec_record(_select_sec_records(year_records, fp_values={"Q3"}, min_days=260, max_days=310))
        q4 = _latest_sec_record(_select_sec_records(year_records, fp_values={"Q4", "FY"}, min_days=80, max_days=110))
        fy_total = _latest_sec_record(_select_sec_records(year_records, fp_values={"FY", "Q4"}, min_days=340, max_days=380))

        q1_value = q1["value"] if q1 else None
        q2_value = q2["value"] if q2 else (q2_ytd["value"] - q1_value if q2_ytd and q1_value is not None else None)
        if q3:
            q3_value = q3["value"]
        elif q3_ytd and q2_ytd:
            q3_value = q3_ytd["value"] - q2_ytd["value"]
        elif q3_ytd and q1_value is not None and q2_value is not None:
            q3_value = q3_ytd["value"] - q1_value - q2_value
        else:
            q3_value = None

        if q4:
            q4_value = q4["value"]
        elif fy_total and q3_ytd:
            q4_value = fy_total["value"] - q3_ytd["value"]
        elif fy_total and q1_value is not None and q2_value is not None and q3_value is not None:
            q4_value = fy_total["value"] - q1_value - q2_value - q3_value
        else:
            q4_value = None

        add_row("Q1", fiscal_year, q1_value, q1["end"] if q1 else None, q1["filed"] if q1 else None, "reported quarter")
        add_row(
            "Q2",
            fiscal_year,
            q2_value,
            (q2 or q2_ytd or {}).get("end"),
            (q2 or q2_ytd or {}).get("filed"),
            "reported quarter" if q2 else "derived from YTD",
        )
        add_row(
            "Q3",
            fiscal_year,
            q3_value,
            (q3 or q3_ytd or {}).get("end"),
            (q3 or q3_ytd or {}).get("filed"),
            "reported quarter" if q3 else "derived from YTD",
        )
        add_row(
            "Q4",
            fiscal_year,
            q4_value,
            (q4 or fy_total or {}).get("end"),
            (q4 or fy_total or {}).get("filed"),
            "reported quarter" if q4 else "derived from annual",
        )

    if not rows:
        return pd.DataFrame(columns=["label", "value", "period_end", "filed", "source"])

    df = (
        pd.DataFrame(rows)
        .sort_values(["period_end", "filed"])
        .drop_duplicates(subset=["label", "period_end"], keep="last")
        .tail(QUARTERLY_REVENUE_LIMIT)
        .reset_index(drop=True)
    )
    return df


def _select_monetary_unit_rows(fact_data: Dict[str, object]) -> tuple[List[Dict[str, object]], str]:
    units = fact_data.get("units", {}) if isinstance(fact_data, dict) else {}
    if not isinstance(units, dict) or not units:
        return [], ""

    if "USD" in units:
        return units["USD"], "USD"

    currency_units = [
        (unit_name, rows)
        for unit_name, rows in units.items()
        if isinstance(unit_name, str) and len(unit_name) == 3 and unit_name.isalpha() and unit_name.isupper()
    ]
    if currency_units:
        best_unit, best_rows = max(currency_units, key=lambda item: len(item[1] or []))
        return best_rows, best_unit

    unit_name, rows = next(iter(units.items()))
    return rows or [], str(unit_name)


def _normalize_metric_key(value: str) -> str:
    return "".join(char.lower() for char in str(value or "") if char.isalnum())


def _extract_yfinance_revenue_df(symbol: str, period: str) -> tuple[pd.DataFrame, str, str]:
    attribute_candidates = (
        ("quarterly_income_stmt", "quarterly_financials")
        if period == "quarterly"
        else ("income_stmt", "financials")
    )
    try:
        ticker = yf.Ticker(symbol)
    except Exception as e:
        print(f"Error creating yfinance ticker for {symbol}: {e}")
        return pd.DataFrame(columns=["label", "value", "period_end", "filed", "source"]), "Revenue", "Yahoo Finance"

    raw_df = pd.DataFrame()
    for attribute_name in attribute_candidates:
        try:
            candidate_df = getattr(ticker, attribute_name)
        except Exception as e:
            print(f"Error loading {attribute_name} for {symbol}: {e}")
            continue
        if isinstance(candidate_df, pd.DataFrame) and not candidate_df.empty:
            raw_df = candidate_df
            break

    if raw_df.empty:
        return pd.DataFrame(columns=["label", "value", "period_end", "filed", "source"]), "Revenue", "Yahoo Finance"

    normalized_index = {
        _normalize_metric_key(metric_name): metric_name
        for metric_name in raw_df.index
    }
    selected_metric = None
    for metric_name in YFINANCE_REVENUE_ROW_KEYS:
        selected_metric = normalized_index.get(_normalize_metric_key(metric_name))
        if selected_metric:
            break

    if not selected_metric:
        return pd.DataFrame(columns=["label", "value", "period_end", "filed", "source"]), "Revenue", "Yahoo Finance"

    revenue_series = pd.to_numeric(raw_df.loc[selected_metric], errors="coerce").dropna()
    if revenue_series.empty:
        return pd.DataFrame(columns=["label", "value", "period_end", "filed", "source"]), str(selected_metric), "Yahoo Finance"

    revenue_series.index = pd.to_datetime(revenue_series.index)
    revenue_series = revenue_series.sort_index()

    rows = []
    for period_end, value in revenue_series.items():
        if pd.isna(period_end) or pd.isna(value):
            continue
        period_end_ts = pd.Timestamp(period_end).normalize()
        if period == "annual":
            label = f"FY {period_end_ts.year}"
        else:
            label = period_end_ts.strftime("%Y-%m-%d")
        rows.append(
            {
                "label": label,
                "value": float(value),
                "period_end": period_end_ts,
                "filed": period_end_ts,
                "source": "Yahoo Finance",
            }
        )

    if not rows:
        return pd.DataFrame(columns=["label", "value", "period_end", "filed", "source"]), str(selected_metric), "Yahoo Finance"

    revenue_df = (
        pd.DataFrame(rows)
        .sort_values("period_end")
        .tail(QUARTERLY_REVENUE_LIMIT if period == "quarterly" else ANNUAL_REVENUE_LIMIT)
        .reset_index(drop=True)
    )
    return revenue_df, str(selected_metric), "Yahoo Finance"


def get_company_revenue(symbol: str, period: str) -> tuple[pd.DataFrame, str, str]:
    try:
        company_facts = get_company_facts(symbol)
        taxonomy_facts = company_facts.get("facts", {})
    except Exception as e:
        print(f"Error loading SEC company facts for {symbol}: {e}")
        taxonomy_facts = {}
    best_sec_result: Optional[tuple[pd.DataFrame, str, str]] = None
    best_sec_score: Optional[tuple[pd.Timestamp, int, int]] = None

    for taxonomy_name, fact_keys in SEC_TAXONOMY_REVENUE_FACT_KEYS.items():
        scoped_facts = taxonomy_facts.get(taxonomy_name, {})
        for fact_key in fact_keys:
            fact_data = scoped_facts.get(fact_key)
            if not fact_data:
                continue

            unit_rows, unit_label = _select_monetary_unit_rows(fact_data)
            parsed_records = _parse_revenue_fact_records(unit_rows)
            if not parsed_records:
                continue

            if period == "annual":
                revenue_df = _build_annual_revenue_df(parsed_records)
            else:
                revenue_df = _build_quarterly_revenue_df(parsed_records)

            if not revenue_df.empty:
                revenue_df["period_end"] = pd.to_datetime(revenue_df["period_end"])
                revenue_df["filed"] = pd.to_datetime(revenue_df["filed"])
                latest_period_end = revenue_df["period_end"].max()
                score = (latest_period_end, len(revenue_df), 1)
                if best_sec_score is None or score > best_sec_score:
                    best_sec_score = score
                    concept_label = str(fact_data.get("label") or fact_key)
                    source_label = f"SEC company facts ({taxonomy_name}, {unit_label or 'reported currency'})"
                    best_sec_result = (revenue_df, concept_label, source_label)

    revenue_df, concept_label, source_label = _extract_yfinance_revenue_df(symbol, period)
    best_yahoo_result: Optional[tuple[pd.DataFrame, str, str]] = None
    if not revenue_df.empty:
        revenue_df["period_end"] = pd.to_datetime(revenue_df["period_end"])
        revenue_df["filed"] = pd.to_datetime(revenue_df["filed"])
        best_yahoo_result = (revenue_df, concept_label, source_label)

    if best_sec_result is None and best_yahoo_result is not None:
        return best_yahoo_result
    if best_sec_result is not None and best_yahoo_result is None:
        return best_sec_result
    if best_sec_result is not None and best_yahoo_result is not None:
        sec_df = best_sec_result[0]
        yahoo_df = best_yahoo_result[0]
        sec_latest = sec_df["period_end"].max()
        yahoo_latest = yahoo_df["period_end"].max()
        staleness_threshold_days = 180 if period == "quarterly" else 400
        if (yahoo_latest - sec_latest).days > staleness_threshold_days:
            return best_yahoo_result
        return best_sec_result

    return pd.DataFrame(columns=["label", "value", "period_end", "filed", "source"]), "Revenue", "No source"


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
# Chat callbacks
# ------------------------------------------------------------

@app.callback(
    Output("chat-panel", "className"),
    Output("chat-open", "data"),
    Input("chat-bubble-btn", "n_clicks"),
    Input("chat-close-btn", "n_clicks"),
    State("chat-open", "data"),
    prevent_initial_call=True,
)
def toggle_chat_panel(bubble_clicks, close_clicks, is_open):
    triggered = ctx.triggered_id
    if triggered == "chat-close-btn":
        return "", False
    # bubble button toggles
    new_open = not is_open
    return ("chat-panel--open" if new_open else ""), new_open


@app.callback(
    Output("chat-messages", "children"),
    Output("chat-history", "data"),
    Output("chat-input", "value"),
    Input("chat-send-btn", "n_clicks"),
    State("chat-input", "value"),
    State("chat-history", "data"),
    prevent_initial_call=True,
)
def send_chat_message(n_clicks, user_text, history):
    if not user_text or not user_text.strip():
        raise dash.exceptions.PreventUpdate

    user_text = user_text.strip()

    # Append user message to history
    history = list(history or [])
    history.append({"role": "user", "content": user_text})

    # Call the chat microservice
    assistant_text: Optional[str] = None
    error_text: Optional[str] = None
    try:
        resp = _requests.post(
            f"{CHAT_SERVICE_URL}/chat",
            json={"messages": history},
            timeout=_CHAT_TIMEOUT,
        )
        if resp.ok:
            data = resp.json()
            assistant_text = data.get("response") or data.get("error") or "No response."
        else:
            error_text = f"Service error {resp.status_code}: {resp.text[:200]}"
    except _requests.exceptions.ConnectionError:
        error_text = (
            "Could not reach the finance assistant. "
            "Make sure the chat service is running (start_chat_service.ps1)."
        )
    except _requests.exceptions.Timeout:
        error_text = "The model took too long to respond. Try a shorter question."
    except Exception as exc:  # noqa: BLE001
        error_text = f"Unexpected error: {exc}"

    if assistant_text:
        history.append({"role": "assistant", "content": assistant_text})

    # Build message bubbles
    bubbles = []
    for msg in history:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system":
            continue
        css_class = "chat-msg chat-msg--user" if role == "user" else "chat-msg chat-msg--assistant"
        bubbles.append(html.Div(content, className=css_class))

    if error_text:
        bubbles.append(html.Div(error_text, className="chat-msg chat-msg--error"))

    return bubbles, history, ""


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
