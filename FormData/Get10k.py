import re
from functools import lru_cache
from typing import Any, Dict

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
    soup = BeautifulSoup(html, features = 'xml')

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

# ------------------ Example usage ------------------
if __name__ == "__main__":
    import os
    ticker = "NVDA" 
    text = get_latest_10k_text(ticker)
    # save the text to a folder directory called 10k_text
    os.makedirs("FormData/10k_text", exist_ok=True)
    with open(f"FormData/10k_text/{ticker}.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print(text[:2000])
