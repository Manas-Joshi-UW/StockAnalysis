# pull all stocks in the stock market
import io
import requests
import pandas as pd

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/symdir/nasdaqlisted.txt"
OTHER_URL  = "https://www.nasdaqtrader.com/dynamic/symdir/otherlisted.txt"

def _read_symdir(url: str) -> pd.DataFrame:
    """
    Reads NASDAQ Trader symdir files (pipe-delimited) into a DataFrame,
    dropping the trailing 'File Creation Time' footer row.
    """
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), sep="|", dtype=str, engine="python")
    # Drop footer rows (they don't have the same columns)
    df = df[df.columns[:-1]] if df.columns[-1].startswith("File Creation Time") else df
    # Also drop any fully-empty rows just in case
    df = df.dropna(how="all")
    return df

def get_nyse_nasdaq_listings(include_etfs: bool = False) -> pd.DataFrame:
    """
    Returns a DataFrame of current NASDAQ + NYSE listings from NASDAQ Trader.
    - Excludes test issues (Test Issue == 'N' kept).
    - include_etfs=False excludes ETFs.
    Columns returned: Symbol, Security Name, Exchange, ETF, Market Category (if available)
    """
    # --- NASDAQ ---
    nasdaq = _read_symdir(NASDAQ_URL)
    # Columns in nasdaqlisted.txt: Symbol | Security Name | Market Category | Test Issue | Financial Status | ...
    nasdaq = nasdaq.rename(columns={"Symbol": "Symbol", "Security Name": "Security Name"})
    nasdaq = nasdaq[nasdaq["Test Issue"] == "N"].copy()
    if not include_etfs and "ETF" in nasdaq.columns:
        nasdaq = nasdaq[nasdaq["ETF"] != "Y"]
    nasdaq["Exchange"] = "NASDAQ"
    # Keep a consistent set of columns
    nasdaq = nasdaq[["Symbol", "Security Name", "Exchange", "ETF", "Market Category"]]

    # --- NYSE (from otherlisted.txt) ---
    other = _read_symdir(OTHER_URL)
    # Columns in otherlisted.txt typically include:
    # 'ACT Symbol' | 'Security Name' | 'Exchange' | 'CQS Symbol' | 'ETF' | 'Round Lot Size' | 'Test Issue' | 'NASDAQ Symbol'
    other = other.rename(columns={"ACT Symbol": "Symbol"})
    # NYSE rows in 'Exchange' are marked 'N'; others include 'A' (NYSE MKT/AMEX), 'P' (NYSE Arca), 'Z' (BATS)
    nyse = other[other["Exchange"] == "N"].copy()
    nyse = nyse[nyse["Test Issue"] == "N"]
    if not include_etfs:
        nyse = nyse[nyse["ETF"] != "Y"]
    nyse["Exchange"] = "NYSE"
    nyse["Market Category"] = pd.NA  # not provided in this file
    nyse = nyse[["Symbol", "Security Name", "Exchange", "ETF", "Market Category"]]

    # Combine + clean
    combined = pd.concat([nasdaq, nyse], ignore_index=True)
    combined = combined.drop_duplicates(subset=["Symbol"]).reset_index(drop=True)

    # (Optional) simple sorting: by Exchange then Symbol
    combined = combined.sort_values(["Exchange", "Symbol"], kind="stable").reset_index(drop=True)
    return combined

if __name__ == "__main__":
    df = get_nyse_nasdaq_listings(include_etfs=False)  # set True to keep ETFs
    print(df.head(20))
    # Save to CSV if you want:
    df.to_csv("nyse_nasdaq_listings.csv", index=False)
    print(f"Saved {len(df):,} rows to nyse_nasdaq_listings.csv")
