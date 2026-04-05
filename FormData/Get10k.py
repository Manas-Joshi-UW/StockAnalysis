import os
import re
import time
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List

import pandas as pd
import requests

UA = {"User-Agent": "Manas Joshi joshi.manas01@gmail.com"}  # SEC requires a descriptive UA

def _sec_get_json(url: str) -> Dict[str, Any]:
    response = requests.get(url, headers=UA, timeout=30)
    response.raise_for_status()
    return response.json()


@lru_cache(maxsize=1)
def _company_ticker_map() -> Dict[str, str]:
    payload = _sec_get_json("https://www.sec.gov/files/company_tickers.json")
    ticker_map: Dict[str, str] = {}
    for entry in payload.values():
        ticker = str(entry.get("ticker") or "").strip().upper()
        cik_value = entry.get("cik_str")
        if not ticker or cik_value in (None, ""):
            continue
        ticker_map[ticker] = str(cik_value).zfill(10)
    return ticker_map


def _ticker_to_cik(ticker: str) -> str:
    normalized_ticker = str(ticker or "").strip().upper()
    cik = _company_ticker_map().get(normalized_ticker)
    if cik:
        return cik
    raise ValueError(f"Ticker not found: {ticker}")


@lru_cache(maxsize=512)
def get_company_facts_by_cik(cik: str) -> Dict[str, Any]:
    normalized_cik = str(cik or "").strip().zfill(10)
    return _sec_get_json(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{normalized_cik}.json")


def get_company_facts(ticker: str) -> Dict[str, Any]:
    return get_company_facts_by_cik(_ticker_to_cik(ticker))

def _latest_10k_url_for_cik(cik: str) -> str:
    sub = _sec_get_json(f"https://data.sec.gov/submissions/CIK{cik}.json")
    recent = sub["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]  # usually *.htm
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    raise ValueError("No 10-K found")

def _clean_html_to_text(html: str) -> str:
    # 1) Parse
    try:
        from bs4 import BeautifulSoup
    except ImportError as exc:
        raise RuntimeError("beautifulsoup4 is required to parse 10-K HTML") from exc
    # Prefer lxml for speed; fall back to the stdlib html.parser (always available)
    for parser in ("lxml", "html.parser"):
        try:
            soup = BeautifulSoup(html, features=parser)
            break
        except Exception:
            continue
    else:
        raise RuntimeError("No suitable HTML parser found. Install lxml: pip install lxml")

    # 2) Drop non-content elements
    for tag in soup(["script", "style", "noscript", "header", "footer"]):
        tag.decompose()

    # EDGAR often uses tables for navigation/indexes; remove obvious ones
    for t in soup.find_all("table"):
        classes = " ".join(t.get("class", [])).lower()
        # keep financial statements (often needed), skip index/nav tables
        if any(k in classes for k in ["nav", "menu", "toc", "tablefile", "index"]):
            t.decompose()

    # 3) Normalize line breaks for readability
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for p in soup.find_all("p"):
        # ensure paragraph separation
        if p.text and not p.text.endswith("\n"):
            p.append("\n\n")

    # 4) Extract text
    text = soup.get_text(separator="\n")

    # 5) Unescape & normalize whitespace
    text = text.replace("\xa0", " ")  # non-breaking space → space
    # collapse 3+ newlines to 2; collapse multi spaces
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # 6) Trim leading/trailing whitespace on lines
    text = "\n".join(line.strip() for line in text.splitlines())

    # 7) Remove empty lines that repeat
    # (already reduced, but one more light pass keeps it neat)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text

def get_latest_10k_text(ticker: str) -> str:
    cik = _ticker_to_cik(ticker)
    url = _latest_10k_url_for_cik(cik)
    html = requests.get(url, headers=UA).text
    return _clean_html_to_text(html)

# --- Optional: extract common 10-K sections by “ITEM” headings ---
_ITEM_PATTERNS = {
    "business": r"(?is)\bitem\s*1\.\s*business\b",
    "risk_factors": r"(?is)\bitem\s*1a\.\s*risk\s*factors\b",
    "mda": r"(?is)\bitem\s*7\.\s*management(?:’|'|’)?s\s*discussion.*?analysis\b",
    "financials": r"(?is)\bitem\s*8\.\s*financial\s*statements\b",
    "controls": r"(?is)\bitem\s*9a\.\s*controls\s*and\s*procedures\b",
}

def extract_10k_section(full_text: str, key: str) -> str:
    """
    key in {"business","risk_factors","mda","financials","controls"}
    Returns the text from the requested ITEM heading up to the next ITEM heading.
    """
    # Build a generic "next ITEM" lookahead
    next_item = re.compile(r"(?is)\n\s*item\s*\d+[a]?\.", re.IGNORECASE)
    start_pat = re.compile(_ITEM_PATTERNS[key], re.IGNORECASE)

    m = start_pat.search(full_text)
    if not m:
        raise ValueError(f"Section not found: {key}")

    start = m.start()
    nxt = next_item.search(full_text, m.end())
    end = nxt.start() if nxt else len(full_text)
    section = full_text[start:end].strip()
    return section

RETRIEVAL_LOG = "10k_retrieval_log.csv"


def _load_retrieval_log(log_path: str) -> pd.DataFrame:
    if os.path.exists(log_path):
        return pd.read_csv(log_path, parse_dates=["last_date_of_retrieval"])
    return pd.DataFrame(columns=["ticker", "last_date_of_retrieval", "status"])


def _save_retrieval_log(df: pd.DataFrame, log_path: str) -> None:
    df.to_csv(log_path, index=False)


def get_tickers_from_price_history(price_history_dir: str = "price_history") -> List[str]:
    """Return sorted list of tickers that have a parquet file in price_history_dir."""
    if not os.path.isdir(price_history_dir):
        raise FileNotFoundError(f"Directory not found: {price_history_dir}")
    return sorted(
        f[:-len(".parquet")].upper()
        for f in os.listdir(price_history_dir)
        if f.endswith(".parquet")
    )


def get_10k_for_tickers(
    tickers: List[str],
    output_dir: str = "10k_text",
    log_path: str = RETRIEVAL_LOG,
    delay: float = 0.5,
    skip_already_logged: bool = True,
) -> pd.DataFrame:
    """
    Fetch and save 10-K filings for a list of tickers, tracking retrieval in a CSV log.

    Args:
        tickers:            List of ticker symbols.
        output_dir:         Directory to save <TICKER>.txt files.
        log_path:           Path to the CSV tracking file.
        delay:              Seconds to wait between SEC requests.
        skip_already_logged: Skip tickers that already have a successful entry in the log.

    Returns:
        Updated retrieval log as a DataFrame (also saved to log_path).
    """
    os.makedirs(output_dir, exist_ok=True)
    log = _load_retrieval_log(log_path)

    already_done: set = set()
    if skip_already_logged:
        already_done = set(
            log.loc[log["status"] == "ok", "ticker"].str.upper().tolist()
        )

    new_rows = []
    for ticker in tickers:
        ticker = str(ticker).strip().upper()

        if ticker in already_done:
            print(f"[skip] {ticker} — already in log")
            continue

        out_path = os.path.join(output_dir, f"{ticker}.txt")
        try:
            text = get_latest_10k_text(ticker)
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(text)
            status = "ok"
            print(f"[ok]   {ticker}")
        except Exception as exc:
            status = str(exc)
            print(f"[err]  {ticker}: {status}")

        new_rows.append({
            "ticker": ticker,
            "last_date_of_retrieval": datetime.now(timezone.utc).isoformat(),
            "status": status,
        })
        _save_retrieval_log(
            pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True),
            log_path,
        )
        time.sleep(delay)

    updated_log = pd.concat([log, pd.DataFrame(new_rows)], ignore_index=True)
    _save_retrieval_log(updated_log, log_path)
    return updated_log


def fetch_all_10ks_from_price_history(
    price_history_dir: str = "price_history",
    output_dir: str = "10k_text",
    log_path: str = RETRIEVAL_LOG,
    delay: float = 0.5,
) -> pd.DataFrame:
    """
    Convenience wrapper: discovers tickers from price_history_dir and fetches
    10-K filings for any not already in the retrieval log.

    Returns the updated retrieval log DataFrame.
    """
    tickers = get_tickers_from_price_history(price_history_dir)
    print(f"Found {len(tickers)} tickers in {price_history_dir}/")
    return get_10k_for_tickers(
        tickers,
        output_dir=output_dir,
        log_path=log_path,
        delay=delay,
        skip_already_logged=True,
    )


# ------------------ Example usage ------------------
if __name__ == "__main__":
    log = fetch_all_10ks_from_price_history()
    print(f"\nDone. Log saved to {RETRIEVAL_LOG}")
    print(log["status"].value_counts().to_string())
