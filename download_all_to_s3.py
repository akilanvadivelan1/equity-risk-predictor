#!/usr/bin/env python3
"""
S&P 500 Risk Radar - Bulk 10-K Risk Factor Downloader (S3)
===========================================================
Downloads the two most recent 10-K filings for every S&P 500 company,
extracts Item 1A (Risk Factors), and uploads each one to S3 as a text file.

Design goals for this long running job:
  - CONSERVATIVE toward SEC EDGAR. One request every few seconds, well under
    SEC's 10 requests/second limit, with retries and backoff on errors.
  - RESUMABLE. A local progress file (progress.json) records every ticker/year
    that is already done or has permanently failed. Re-running the script skips
    all of those, so you can stop and restart safely at any time.
  - ROBUST. One bad filing never stops the whole run. Failures are recorded
    with a reason so you can review the handful of problem cases later.

Storage layout in S3:
    s3://snp500-risk-radar-10k-data/risk-factors/<filing_year>/<TICKER>.txt
    s3://snp500-risk-radar-10k-data/manifest.json     (all successful filings)

Local files created next to this script:
    progress.json    (resume state: done + failed, ticker/year keyed)
    run.log          (human readable progress log)

Usage:
    pip install boto3
    aws sso login --profile NAME   (or aws configure)   # active AWS creds
    python3 download_all_to_s3.py

    If it stops for any reason, just run it again. It resumes.

AWS credentials use standard boto3 resolution. No keys are hardcoded.
"""

import os
import re
import sys
import json
import time
import gzip
import urllib.request
import urllib.error
from datetime import datetime
from html import unescape

# ============================================================
# Configuration
# ============================================================
BUCKET = "snp500-risk-radar-10k-data"
REGION = "us-east-1"
PREFIX = "risk-factors"
CONTACT_EMAIL = "velnraj@gmail.com"

# How many of the most recent 10-K filings to keep per company.
RECENT_YEARS_TO_KEEP = 2

# Conservative delay (seconds) BEFORE every SEC request. A full pause per the
# request to be gentle on a government server. 2 to 5 seconds as requested.
SEC_SLEEP_SECONDS = 3.0

# Extra pause between companies, on top of the per request sleep.
BETWEEN_COMPANY_SLEEP = 1.0

# Retry policy for transient SEC errors (429 too many requests, 503, timeouts).
MAX_RETRIES = 4
RETRY_BACKOFF_SECONDS = 15.0   # grows each retry: 15, 30, 45, 60

# Minimum characters for an extraction to count as a real risk section.
MIN_VALID_CHARS = 500

# Local state files (next to this script).
HERE = os.path.dirname(os.path.abspath(__file__))
PROGRESS_FILE = os.path.join(HERE, "progress.json")
LOG_FILE = os.path.join(HERE, "run.log")

SEC_HEADERS_WWW = {
    "User-Agent": f"S&P500 Risk Radar Research Tool {CONTACT_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
SEC_HEADERS_DATA = {
    "User-Agent": f"S&P500 Risk Radar Research Tool {CONTACT_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}


# ============================================================
# Small logging helper
# ============================================================
def log(msg: str):
    stamp = datetime.now().strftime("%H:%M:%S")
    line = f"[{stamp}] {msg}"
    print(line, flush=True)
    try:
        with open(LOG_FILE, "a") as f:
            f.write(line + "\n")
    except Exception:
        pass


# ============================================================
# Polite SEC fetch with rate limiting and retries
# ============================================================
def sec_get(url: str, host_headers: dict, timeout: int = 60) -> str:
    """
    Fetch a URL from SEC with a conservative pre-sleep, gzip handling, and
    retry with backoff on transient errors (429 / 503 / timeouts).
    """
    last_err = None
    for attempt in range(MAX_RETRIES):
        # Conservative pause BEFORE every request.
        time.sleep(SEC_SLEEP_SECONDS)
        try:
            req = urllib.request.Request(url, headers=host_headers)
            resp = urllib.request.urlopen(req, timeout=timeout)
            raw = resp.read()
            if resp.headers.get("Content-Encoding") == "gzip":
                raw = gzip.decompress(raw)
            return raw.decode("utf-8", errors="ignore")
        except urllib.error.HTTPError as e:
            last_err = e
            if e.code in (429, 503):
                wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
                log(f"    SEC returned {e.code}. Backing off {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES}).")
                time.sleep(wait)
                continue
            # Other HTTP errors (404 etc.) are not worth retrying.
            raise
        except (urllib.error.URLError, TimeoutError) as e:
            last_err = e
            wait = RETRY_BACKOFF_SECONDS * (attempt + 1)
            log(f"    Network error: {e}. Retrying in {wait:.0f}s (attempt {attempt + 1}/{MAX_RETRIES}).")
            time.sleep(wait)
            continue
    raise RuntimeError(f"Failed after {MAX_RETRIES} attempts: {url} ({last_err})")


# ============================================================
# S&P 500 ticker list
# ============================================================
def get_sp500_tickers() -> list:
    """Fetch current S&P 500 tickers from Wikipedia, with a fallback list."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Risk Radar Research"})
        resp = urllib.request.urlopen(req, timeout=20)
        html = resp.read().decode("utf-8")
        tickers = re.findall(r'<td[^>]*><a[^>]*>([A-Z]{1,5})</a>', html)
        if len(tickers) > 400:
            # Deduplicate while preserving order.
            seen, ordered = set(), []
            for t in tickers[:520]:
                if t not in seen:
                    seen.add(t)
                    ordered.append(t)
            log(f"Fetched {len(ordered)} S&P 500 tickers from Wikipedia.")
            return ordered
    except Exception as e:
        log(f"Wikipedia ticker fetch failed ({e}). Using fallback top 100 list.")

    return [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "BRK-B", "TSLA", "UNH", "LLY",
        "JPM", "XOM", "V", "JNJ", "PG", "MA", "AVGO", "HD", "COST", "MRK",
        "ABBV", "CVX", "CRM", "KO", "PEP", "WMT", "ADBE", "BAC", "CSCO", "MCD",
        "TMO", "NFLX", "ACN", "AMD", "LIN", "ABT", "ORCL", "DHR", "TXN", "PM",
        "WFC", "DIS", "CMCSA", "NEE", "VZ", "INTC", "RTX", "HON", "UNP", "SPGI",
        "COP", "QCOM", "BMY", "AMGN", "LOW", "ELV", "CAT", "BA", "INTU", "GE",
        "MS", "ISRG", "DE", "BLK", "AMAT", "GS", "PLD", "NOW", "SYK", "MDLZ",
        "GILD", "ADP", "ADI", "TJX", "BKNG", "MMC", "VRTX", "LRCX", "SCHW", "CI",
        "CB", "REGN", "ZTS", "MO", "TMUS", "SO", "CME", "ETN", "BDX", "PGR",
        "DUK", "SLB", "BSX", "PANW", "AON", "ICE", "EQIX", "SNPS", "CL", "TGT",
    ]


# ============================================================
# SEC lookups
# ============================================================
def build_cik_map(tickers: list) -> dict:
    """Fetch the ticker -> CIK map from SEC ONCE and filter to our tickers."""
    url = "https://www.sec.gov/files/company_tickers.json"
    data = json.loads(sec_get(url, SEC_HEADERS_WWW, timeout=30))
    wanted = {t.upper().replace("-", ".") for t in tickers}
    cik_map = {}
    for entry in data.values():
        t = entry.get("ticker", "").upper()
        if t in wanted:
            cik_map[t] = str(entry["cik_str"]).zfill(10)
    return cik_map


def recent_10k_filings(cik: str, keep: int) -> list:
    """Return the `keep` most recent 10-K filings for a CIK, newest first."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = json.loads(sec_get(url, SEC_HEADERS_DATA, timeout=30))
    company = data.get("name", "")
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accns = recent.get("accessionNumber", [])
    docs = recent.get("primaryDocument", [])

    filings = []
    for i, form in enumerate(forms):
        if form == "10-K":   # exclude 10-K/A amendments to avoid duplicate years
            date = dates[i] if i < len(dates) else ""
            filings.append({
                "form": form,
                "date": date,
                "year": int(date[:4]) if date else 0,
                "accession": accns[i].replace("-", "") if i < len(accns) else "",
                "accession_raw": accns[i] if i < len(accns) else "",
                "primary_doc": docs[i] if i < len(docs) else "",
                "cik": cik,
                "company_name": company,
            })

    filings.sort(key=lambda f: f["date"], reverse=True)
    return filings[:keep]


# ============================================================
# Item 1A extraction (HTML -> plain text first, then locate section)
# ============================================================
def extract_item_1a(html: str) -> str:
    """Robust Item 1A extractor that works on modern inline-XBRL 10-Ks."""
    text = html
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"<(script|style)[^>]*>.*?</\1>", " ", text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r"<(br|/p|/div|/tr|/td|/th|/li|/h[1-6])[^>]*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", "", text)
    text = unescape(text)
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)

    start_re = re.compile(r"Item\s*1A[\.\s\u2013\u2014\-:]*\s*Risk\s+Factors", re.IGNORECASE)
    starts = [m.start() for m in start_re.finditer(text)]
    if not starts:
        starts = [m.start() for m in re.finditer(r"Item\s*1A\b", text, re.IGNORECASE)]
    if not starts:
        return ""

    end_re = re.compile(r"Item\s*1B\b|Item\s*2[\.\s\u2013\u2014\-:]+\s*Propert", re.IGNORECASE)

    best_start, best_end, best_len = None, None, 0
    for s in starts:
        m = end_re.search(text, s + 50)
        end = m.start() if m else len(text)
        if end - s > best_len:
            best_len = end - s
            best_start, best_end = s, end
    if best_start is None:
        return ""

    inner = [m.start() for m in start_re.finditer(text, best_start + 20, best_end)]
    if inner:
        best_start = inner[-1]

    span = text[best_start:best_end].strip()
    span = re.sub(r"^Item\s*1A[\.\s\u2013\u2014\-:]*\s*Risk\s+Factors\s*", "", span, count=1, flags=re.IGNORECASE)
    span = span.strip()
    if len(span) > 200000:
        span = span[:200000]
    return span


def download_item_1a(filing: dict) -> tuple:
    """Download the filing and extract Item 1A. Returns (text, source_url)."""
    cik = filing["cik"].lstrip("0")
    accession = filing["accession"]
    primary_doc = filing["primary_doc"]

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
    html = sec_get(url, SEC_HEADERS_WWW, timeout=90)
    text = extract_item_1a(html)
    if text and len(text) >= MIN_VALID_CHARS:
        return text, url

    # Fallback: scan other .htm documents in the filing index.
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
    index_html = sec_get(index_url, SEC_HEADERS_WWW, timeout=60)
    htm = re.findall(r'href="([^"]+\.htm[l]?)"', index_html, re.IGNORECASE)
    htm = [f.split("/")[-1] for f in htm]
    htm = [f for f in htm if f != primary_doc and not f[:2].upper().startswith("R")]
    for alt in htm[:5]:
        alt_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{alt}"
        try:
            alt_html = sec_get(alt_url, SEC_HEADERS_WWW, timeout=60)
            alt_text = extract_item_1a(alt_html)
            if alt_text and len(alt_text) >= MIN_VALID_CHARS:
                return alt_text, alt_url
        except Exception:
            continue

    return (text or ""), url


def build_body(filing: dict, ticker: str, source_url: str, text: str) -> str:
    return (
        f"TICKER: {ticker}\n"
        f"COMPANY: {filing.get('company_name', '')}\n"
        f"FORM: {filing.get('form', '10-K')}\n"
        f"FILING_YEAR: {filing.get('year', '')}\n"
        f"FILING_DATE: {filing.get('date', '')}\n"
        f"ACCESSION: {filing.get('accession_raw', '')}\n"
        f"CIK: {filing.get('cik', '')}\n"
        f"SOURCE_URL: {source_url}\n"
        f"SECTION: Item 1A - Risk Factors\n"
        f"CHAR_COUNT: {len(text)}\n"
        f"{'-' * 60}\n\n"
    ) + text


# ============================================================
# Local progress (resume state)
# ============================================================
def load_progress() -> dict:
    if os.path.exists(PROGRESS_FILE):
        try:
            with open(PROGRESS_FILE) as f:
                data = json.load(f)
                data.setdefault("done", {})     # key "TICKER:YEAR" -> s3 key
                data.setdefault("failed", {})    # key "TICKER:YEAR" -> reason
                return data
        except Exception:
            pass
    return {"done": {}, "failed": {}}


def save_progress(progress: dict):
    tmp = PROGRESS_FILE + ".tmp"
    with open(tmp, "w") as f:
        json.dump(progress, f, indent=2)
    os.replace(tmp, PROGRESS_FILE)


# ============================================================
# S3
# ============================================================
def make_s3():
    try:
        import boto3
    except ImportError:
        log("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)
    return boto3.client("s3", region_name=REGION)


def upload(s3, key: str, body: str):
    s3.put_object(
        Bucket=BUCKET, Key=key,
        Body=body.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )


def write_manifest(s3, progress: dict):
    """Write a manifest.json of all successful filings to S3."""
    entries = [{"key": k, "s3_key": v} for k, v in sorted(progress["done"].items())]
    manifest = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "bucket": BUCKET,
        "count": len(entries),
        "filings": entries,
    }
    try:
        upload(s3, "manifest.json", json.dumps(manifest, indent=2))
    except Exception as e:
        log(f"    Could not write manifest.json: {e}")


# ============================================================
# Main loop
# ============================================================
def main():
    log("=" * 60)
    log("S&P 500 Risk Radar - bulk 10-K risk factor downloader")
    log(f"Bucket: {BUCKET}   Region: {REGION}   Keep: {RECENT_YEARS_TO_KEEP} most recent 10-Ks/company")
    log(f"Politeness: {SEC_SLEEP_SECONDS}s before each SEC request, +{BETWEEN_COMPANY_SLEEP}s between companies")
    log("=" * 60)

    s3 = make_s3()
    progress = load_progress()
    log(f"Resuming: {len(progress['done'])} filings already done, {len(progress['failed'])} previously failed.")

    tickers = get_sp500_tickers()
    log(f"Building CIK map for {len(tickers)} tickers (one SEC request)...")
    cik_map = build_cik_map(tickers)
    log(f"Matched CIKs for {len(cik_map)}/{len(tickers)} tickers.")

    total = len(cik_map)
    stored = skipped = failed = 0

    for idx, (ticker, cik) in enumerate(sorted(cik_map.items()), start=1):
        log(f"[{idx}/{total}] {ticker} (CIK {cik})")

        try:
            filings = recent_10k_filings(cik, RECENT_YEARS_TO_KEEP)
        except Exception as e:
            log(f"    Could not list filings: {e}")
            time.sleep(BETWEEN_COMPANY_SLEEP)
            continue

        if not filings:
            log("    No 10-K filings found.")
            time.sleep(BETWEEN_COMPANY_SLEEP)
            continue

        for filing in filings:
            year = filing["year"]
            pkey = f"{ticker}:{year}"

            if pkey in progress["done"]:
                skipped += 1
                log(f"    {year}: already done, skipping.")
                continue
            if pkey in progress["failed"]:
                skipped += 1
                log(f"    {year}: previously failed, skipping (delete from progress.json to retry).")
                continue

            s3_key = f"{PREFIX}/{year}/{ticker}.txt"
            try:
                text, source_url = download_item_1a(filing)
                if text and len(text) >= MIN_VALID_CHARS:
                    upload(s3, s3_key, build_body(filing, ticker, source_url, text))
                    progress["done"][pkey] = s3_key
                    stored += 1
                    log(f"    {year}: OK ({len(text):,} chars) -> s3://{BUCKET}/{s3_key}")
                else:
                    progress["failed"][pkey] = f"extraction too short ({len(text)} chars)"
                    failed += 1
                    log(f"    {year}: FAILED extraction ({len(text)} chars).")
            except Exception as e:
                progress["failed"][pkey] = str(e)
                failed += 1
                log(f"    {year}: ERROR {e}")

            # Save progress after EVERY filing so a crash loses nothing.
            save_progress(progress)

        time.sleep(BETWEEN_COMPANY_SLEEP)

    # Final manifest and summary.
    write_manifest(s3, progress)
    log("=" * 60)
    log(f"DONE. Stored this run: {stored}   Skipped: {skipped}   Failed this run: {failed}")
    log(f"Total in progress.json: {len(progress['done'])} done, {len(progress['failed'])} failed.")
    log(f"Manifest written to s3://{BUCKET}/manifest.json")
    log("If anything failed transiently, just run this script again to retry the rest.")
    log("=" * 60)


if __name__ == "__main__":
    main()
