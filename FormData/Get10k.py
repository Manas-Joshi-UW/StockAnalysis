import re
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Manas Joshi joshi.manas01@gmail.com"}  # SEC requires a descriptive UA

def _ticker_to_cik(ticker: str) -> str:
    m = requests.get("https://www.sec.gov/files/company_tickers.json", headers=UA).json()
    for entry in m.values():
        if entry["ticker"].lower() == ticker.lower():
            return str(entry["cik_str"]).zfill(10)
    raise ValueError(f"Ticker not found: {ticker}")

def _latest_10k_url_for_cik(cik: str) -> str:
    sub = requests.get(f"https://data.sec.gov/submissions/CIK{cik}.json", headers=UA).json()
    recent = sub["filings"]["recent"]
    for i, form in enumerate(recent["form"]):
        if form == "10-K":
            accession = recent["accessionNumber"][i].replace("-", "")
            doc = recent["primaryDocument"][i]  # usually *.htm
            return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession}/{doc}"
    raise ValueError("No 10-K found")

def _clean_html_to_text(html: str) -> str:
    # 1) Parse
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
    text = get_latest_10k_text("ABCL")
    print(text[:2000])
