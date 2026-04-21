"""
Daily job: incrementally update all per-ticker parquet files in price_history/.

For each existing <symbol>.parquet, reads the last bar's date, downloads bars
from the day after (through today), and appends. Idempotent — safe to re-run.

Schedule via Windows Task Scheduler using update_price_action.ps1.
"""
import logging
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import yfinance as yf

PROJECT_ROOT = Path(__file__).resolve().parent
PRICE_DIR = PROJECT_ROOT / "price_history"
LOG_DIR = PROJECT_ROOT / "logs"
INTERVAL = "1d"
ACTIONS = True


def setup_logging() -> Path:
    LOG_DIR.mkdir(exist_ok=True)
    log_file = LOG_DIR / f"update_price_action_{date.today().isoformat()}.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    return log_file


def last_bar_date(path: Path):
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        logging.warning("Unreadable parquet %s: %s", path.name, e)
        return None
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return None
    return df.index.max().date()


def update_one(path: Path, today: date) -> str:
    symbol = path.stem
    last = last_bar_date(path)
    if last is None:
        return "skip-empty"

    start = last + timedelta(days=1)
    if start > today:
        return "up-to-date"

    try:
        new = yf.download(
            tickers=symbol,
            start=start.isoformat(),
            end=(today + timedelta(days=1)).isoformat(),  # yfinance end is exclusive
            interval=INTERVAL,
            auto_adjust=False,
            actions=ACTIONS,
            threads=False,
            progress=False,
        )
    except Exception as e:
        return f"error: {e}"

    if new is None or new.dropna(how="all").empty:
        return "no-new-data"

    # yf.download may return MultiIndex columns even for a single ticker; flatten
    # to match the shape already on disk.
    if isinstance(new.columns, pd.MultiIndex):
        new.columns = new.columns.get_level_values(0)

    existing = pd.read_parquet(path)
    combined = pd.concat([existing, new])
    combined = combined[~combined.index.duplicated(keep="last")]
    combined.sort_index(inplace=True)
    combined.to_parquet(path)
    return f"ok:+{len(new)}"


def bucket(status: str) -> str:
    if status.startswith("ok:"):
        return "ok"
    if status.startswith("error:"):
        return "error"
    return status


def main() -> int:
    log_file = setup_logging()
    if not PRICE_DIR.exists():
        logging.error("price_history not found at %s", PRICE_DIR)
        return 1

    files = sorted(PRICE_DIR.glob("*.parquet"))
    total = len(files)
    logging.info("Starting update for %d tickers. Log: %s", total, log_file)

    counts: dict[str, int] = {}
    today = date.today()
    for i, f in enumerate(files, 1):
        status = update_one(f, today)
        counts[bucket(status)] = counts.get(bucket(status), 0) + 1
        if status.startswith("error:"):
            logging.warning("%s -> %s", f.stem, status)
        if i % 200 == 0:
            logging.info("[%d/%d] running totals: %s", i, total, counts)

    logging.info("Done. Summary: %s", counts)
    return 0


if __name__ == "__main__":
    sys.exit(main())
