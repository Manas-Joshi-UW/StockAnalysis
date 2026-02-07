# pip install yfinance pandas pyarrow
import os
import math
from datetime import date
from typing import Iterable, List, Dict

import pandas as pd
import yfinance as yf

# --------- CONFIG ---------
START = "2010-01-01"         # change as needed
END   = date.today().isoformat()
OUT_DIR = "price_history"     # directory to store per-ticker files
ACTIONS = True                # include splits & dividends
INTERVAL = "1d"               # "1d", "1wk", "1mo"
# --------------------------

os.makedirs(OUT_DIR, exist_ok=True)

def clean_symbols(symbols: pd.Series) -> List[str]:
    # Basic cleanup: drop blanks/dupes and symbols with spaces (rare cases on otherlisted)
    syms = symbols.dropna().astype(str).str.strip()
    syms = syms[syms != ""].drop_duplicates()
    # Yahoo typically supports raw NASDAQ/NYSE symbols as-is.
    return syms.tolist()

def download_prices_for_universe(df_listings: pd.DataFrame,
                                 start=START, end=END,
                                 interval=INTERVAL,
                                 actions=ACTIONS) -> Dict[str, str]:
    """
    Downloads OHLCV (+ actions) for all tickers in df_listings['Symbol'].
    Saves per-ticker Parquet files to OUT_DIR.
    Returns a dict: {symbol: "ok"/"empty"/"error"}.
    """
    symbols = clean_symbols(df_listings["Symbol"])
    print(f"Symbols: {symbols}")
    status = {}

    # Process symbols one by one to avoid threading issues
    for symbol in symbols:
        print(f"Downloading {symbol}...")
        try:
            data = yf.download(
                tickers=symbol,
                start=start, end=end,
                interval=interval,
                auto_adjust=False,
                actions=actions,
                threads=False,  # Disable threading
                progress=False,
            )
            print(f"Data: {data}")
            # Check if data is empty
            if data.dropna(how="all").empty:
                print(f"Data is empty for {symbol}")
                status[symbol] = "empty"
                continue
                
            # Save per-ticker parquet
            out_path = os.path.join(OUT_DIR, f"{symbol}.parquet")
            data.to_parquet(out_path)
            status[symbol] = "ok"
            
        except Exception as e:
            status[symbol] = f"error: {e}"
            continue

    return status

# ---------- Example usage ----------
if __name__ == "__main__":
    # If you have the DataFrame from the previous step:
    # from listings script: df = get_nyse_nasdaq_listings(include_etfs=False)
    # For demonstration, re-load from CSV if you saved it earlier:
    df = pd.read_csv("nyse_nasdaq_listings.csv")
    # Placeholder: raise if df isn't defined
    try:
        df
    except NameError:
        raise SystemExit("Please define `df` (your listings DataFrame) or load nyse_nasdaq_listings.csv first.")

    # Optionally filter to common stocks only (yfinance handles ETFs too, but you might have excluded them upstream)
    # Keep as-is if you want everything currently in `df`.
    symbols_count = df["Symbol"].nunique()
    print(f"Tickers to download: {symbols_count:,}")

    status_map = download_prices_for_universe(df, start=START, end=END, interval=INTERVAL, actions=ACTIONS)
    # Quick summary
    summary = pd.Series(status_map).value_counts()
    print("Download status summary:\n", summary)

    # Optional: create a single wide CSV of Adj Close for convenience
    # (reads each parquet and merges on Date index)
    print("Building wide Adj Close CSV (optional)...")
    frames = []
    for s, st in status_map.items():
        if st == "ok":
            p = os.path.join(OUT_DIR, f"{s}.parquet")
            try:
                sub = pd.read_parquet(p)
                if "Adj Close" in sub.columns:
                    frames.append(sub[["Adj Close"]].rename(columns={"Adj Close": s}))
            except Exception:
                pass
    if frames:
        wide = pd.concat(frames, axis=1)
        wide.sort_index(inplace=True)
        wide.to_csv("adj_close_wide.csv")
        print(f"Saved wide matrix with shape {wide.shape} to adj_close_wide.csv")
    else:
        print("No data frames available to build wide CSV.")
