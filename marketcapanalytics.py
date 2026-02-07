import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from datetime import datetime, timedelta
from pathlib import Path

thresholds = [1, 5, 10, 20, 50, 100, 200]
ANALYSIS_CACHE_FILE = "analyzed_stocks_cache.csv"
PROMISING_STOCKS_FILE = "promising_stocks.csv"


def load_analysis_cache() -> pd.DataFrame:
    """Load the analysis cache from CSV, returning empty DataFrame if not exists."""
    if Path(ANALYSIS_CACHE_FILE).exists():
        df = pd.read_csv(ANALYSIS_CACHE_FILE, parse_dates=["analyzed_date"])
        return df
    return pd.DataFrame(columns=["ticker", "analyzed_date"])


def save_to_cache(ticker: str, cache_df: pd.DataFrame) -> pd.DataFrame:
    """Append a ticker with today's date to cache and save to CSV."""
    new_row = pd.DataFrame({"ticker": [ticker], "analyzed_date": [datetime.now()]})
    cache_df = pd.concat([cache_df, new_row], ignore_index=True)
    cache_df.to_csv(ANALYSIS_CACHE_FILE, index=False)
    return cache_df


def get_recently_analyzed(cache_df: pd.DataFrame, days: int = 21) -> set:
    """Return set of tickers analyzed within the last N days."""
    if cache_df.empty:
        return set()
    cutoff = datetime.now() - timedelta(days=days)
    recent = cache_df[cache_df["analyzed_date"] >= cutoff]
    return set(recent["ticker"].unique())


def load_promising_stocks() -> pd.DataFrame:
    """Load the promising stocks CSV, returning empty DataFrame if missing."""
    if Path(PROMISING_STOCKS_FILE).exists():
        return pd.read_csv(PROMISING_STOCKS_FILE, parse_dates=["analyzed_date"])
    return pd.DataFrame(columns=["ticker", "analyzed_date"])


def save_promising_stock(ticker: str, promising_df: pd.DataFrame) -> pd.DataFrame:
    """Append a promising ticker with today's date and save to CSV."""
    new_row = pd.DataFrame({"ticker": [ticker], "analyzed_date": [datetime.now()]})
    promising_df = pd.concat([promising_df, new_row], ignore_index=True)
    promising_df.to_csv(PROMISING_STOCKS_FILE, index=False)
    return promising_df

def plot_historical_market_cap(
    ticker: str,
    start: str = "2000-01-01",
    end: str = None,
    log_scale: bool = False
):
    """
    Plots historical market capitalization for any public company.

    Args:
        ticker (str): Stock ticker (e.g. 'NVDA', 'AAPL', 'MSFT')
        start (str): Start date (YYYY-MM-DD)
        end (str): End date (YYYY-MM-DD or None)
        log_scale (bool): Use log scale for y-axis
    """

    stock = yf.Ticker(ticker)

    # 1. Get historical price data (adjusted for splits)
    hist = stock.history(start=start, end=end, auto_adjust=False)

    if hist.empty:
        raise ValueError("No price data found for ticker")

    # 2. Get current shares outstanding
    shares_outstanding = stock.info.get("sharesOutstanding")

    if shares_outstanding is None:
        raise ValueError("Shares outstanding not available")

    # 3. Compute market cap
    hist["market_cap"] = hist["Adj Close"] * shares_outstanding

    # 4. Plot
    plt.figure(figsize=(12, 6))
    plt.plot(hist.index, hist["market_cap"] / 1e9)

    # if the market cap is greater than the threshold, plot a line for the threshold
    for threshold in thresholds:
        plt.axhline(y=threshold, color="red", linestyle="--", label=f"{threshold}B")

    plt.title(f"{ticker} Historical Market Capitalization")
    plt.xlabel("Date")
    plt.ylabel("Market Cap (USD Billions)")
    plt.grid(True)

    if log_scale:
        plt.yscale("log")

    plt.tight_layout()
    plt.show()

    return hist[["market_cap"]]


# I want a list of stocks that were under the 1 billion market cap for 1 year and then crossed a 5 billion market cap at any point
def get_stocks_crossed_threshold(below_threshold: int, above_threshold: int, years: int, cache_days: int = 21) -> List[str]:
    """
    Get a list of stocks that were under a threshold for a year and then crossed another threshold at any point.
    
    Args:
        cache_days: Skip stocks analyzed within this many days (default 21)
    """
    stocks = pd.read_csv("nyse_nasdaq_listings.csv")
    stocks = stocks["Symbol"].unique()
    print(f"Total stocks in list: {len(stocks)}")
    
    # Load cache and filter out recently analyzed stocks
    cache_df = load_analysis_cache()
    recently_analyzed = get_recently_analyzed(cache_df, days=cache_days)
    stocks_to_analyze = [s for s in stocks if s not in recently_analyzed]
    print(f"Skipping {len(recently_analyzed)} stocks analyzed in last {cache_days} days")
    print(f"Stocks to analyze: {len(stocks_to_analyze)}")
    
    promising_stocks = []
    promising_df = load_promising_stocks()
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=5 * 365)
    # convert start_date and end_date to yyyy-mm-dd format
    start_date = start_date.strftime("%Y-%m-%d")
    end_date = end_date.strftime("%Y-%m-%d")
    print(f"start_date: {start_date}, end_date: {end_date}")
    cter = 0
    for ticker in stocks_to_analyze:
        print(f"ticker: {ticker}")
        stock = yf.Ticker(ticker)
        hist = stock.history(start=start_date, end=end_date, auto_adjust=False)
        shares_outstanding = stock.info.get("sharesOutstanding")

        try:
            hist["market_cap"] = hist["Adj Close"] * shares_outstanding
        except Exception as e:
            print(f"Error calculating market cap for {ticker}: {e}")
            cache_df = save_to_cache(ticker, cache_df)  # Cache even on error to avoid re-hitting
            continue
        
        if hist.empty or len(hist) < 126:  # Need at least 6 months of data
            cache_df = save_to_cache(ticker, cache_df)
            continue
        
        mcap = hist["market_cap"].values
        min_days_above = int(years * 252)  # ~252 trading days per year
        
        # 1. Check if stock was ever below below_threshold (started small)
        was_below = mcap < below_threshold
        if not was_below.any():
            cache_df = save_to_cache(ticker, cache_df)
            continue
        
        # 2. Find first index where it crossed above below_threshold after being below
        first_below_idx = was_below.argmax()  # First True
        above_lower = mcap >= below_threshold
        
        # Find where it first goes above below_threshold after being below
        crossed_lower_idx = None
        for i in range(first_below_idx, len(mcap)):
            if above_lower[i]:
                crossed_lower_idx = i
                break
        
        if crossed_lower_idx is None:
            cache_df = save_to_cache(ticker, cache_df)
            continue
        
        # 3. Check if it stayed above below_threshold for at least 'years' and crossed above_threshold
        post_cross = mcap[crossed_lower_idx:]
        if len(post_cross) < min_days_above:
            cache_df = save_to_cache(ticker, cache_df)
            continue
        
        # Find consecutive days above below_threshold, then check if crossed above_threshold
        above_mask = post_cross >= below_threshold
        crossed_upper = post_cross >= above_threshold
        
        # Find runs where stock stayed above below_threshold
        run_length = 0
        for i, (is_above, hit_upper) in enumerate(zip(above_mask, crossed_upper)):
            if is_above:
                run_length += 1
                if run_length >= min_days_above and hit_upper:
                    promising_stocks.append(ticker)
                    promising_df = save_promising_stock(ticker, promising_df)
                    print(f"Found promising stock: {ticker}")
                    break
            else:
                run_length = 0  # Reset if it drops below lower threshold
        
        # Save to cache after full analysis
        cache_df = save_to_cache(ticker, cache_df)
        
        if cter % 100 == 0:
            print(f"Processed {cter} stocks, found {len(promising_stocks)} promising")
        cter += 1
        
    return promising_stocks

if __name__ == "__main__":
    stocks = get_stocks_crossed_threshold(1e9, 5e9, 1)
    print(f"Number of stocks: {len(stocks)}")
    print(stocks)
    # plot_historical_market_cap("AACB")