#!/usr/bin/env python3
"""
S&P 500 Risk Radar - Single Filing Test Downloader (S3)
========================================================
A one-filing proof of concept. Downloads a company's 10-K from SEC EDGAR,
extracts Item 1A (Risk Factors) using the project's existing extraction
logic, and uploads the cleaned text to an S3 bucket.

This is hardcoded to Amazon (AMZN) so we can validate the full pipeline
(SEC fetch -> extract -> S3 upload) before building the bulk downloader.

Usage:
    # Real run (fetches from SEC, uploads to S3):
    python3 download_one_to_s3.py

    # Extract only, do NOT touch S3 (verify extraction first):
    python3 download_one_to_s3.py --dry-run

    # Override defaults if needed:
    python3 download_one_to_s3.py --year 2026 --bucket other-bucket --region us-east-1

Requirements:
    pip install boto3            (only needed for the real, non dry-run path)

AWS credentials:
    Uses standard boto3 credential resolution. Any of these work:
      - aws configure            (access key + secret in ~/.aws/credentials)
      - AWS_PROFILE=myprofile    (named profile or SSO session)
      - environment variables    (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    No keys are hardcoded in this script.
"""

import os
import sys
import json
import time
import argparse
import urllib.request
import urllib.error

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import re
from html import unescape


def extract_item_1a(html: str) -> str:
    """
    Robust Item 1A (Risk Factors) extractor for modern 10-K filings.

    Strategy: convert the HTML to clean plain text FIRST, then locate the
    section in the clean text. This is far more reliable for inline XBRL
    (iXBRL) filings like Amazon's, where the words "Item 1A" and "Risk
    Factors" are separated by many nested tags in the raw HTML.

    Steps:
      1. Strip scripts, styles, comments, and all tags to plain text.
      2. Find every "Item 1A ... Risk Factors" heading in the plain text.
      3. Find the matching end (Item 1B, or Item 2 Properties).
      4. Pick the longest span (the real section, not the table-of-contents
         entry, which is short).
    """
    # 1. HTML -> plain text
    text = html
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Turn block-level boundaries into spaces so words do not run together
    text = re.sub(r'<(br|/p|/div|/tr|/td|/th|/li|/h[1-6])[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)          # drop all remaining tags
    text = unescape(text)                         # decode &amp; &#160; etc.
    text = text.replace('\u00a0', ' ')            # non breaking space -> space
    text = re.sub(r'[ \t]+', ' ', text)           # collapse runs of spaces
    text = re.sub(r'\s{2,}', ' ', text)           # collapse remaining whitespace

    # 2. Start headings: "Item 1A" then (allowing punctuation/space) "Risk Factors"
    #    In clean text the two are close together even across original tags.
    start_re = re.compile(r'Item\s*1A[\.\s\u2013\u2014\-:]*\s*Risk\s+Factors', re.IGNORECASE)
    starts = [m.start() for m in start_re.finditer(text)]

    # Broader fallback: just "Item 1A" if the strict pattern found nothing
    if not starts:
        starts = [m.start() for m in re.finditer(r'Item\s*1A\b', text, re.IGNORECASE)]

    if not starts:
        return ""

    # 3. End headings: Item 1B (Unresolved Staff Comments) or Item 2 (Properties)
    end_re = re.compile(r'Item\s*1B\b|Item\s*2[\.\s\u2013\u2014\-:]+\s*Propert', re.IGNORECASE)

    # 4. For each start, measure the span to the next end; keep the longest.
    #    If a start is a table-of-contents entry, a LATER real "Item 1A Risk
    #    Factors" heading usually appears soon after inside the span, so we
    #    advance the start to the last such heading before the section end.
    best_start, best_end, best_len = None, None, 0
    for s in starts:
        m = end_re.search(text, s + 50)
        end = m.start() if m else len(text)
        span_len = end - s
        if span_len > best_len:
            best_len = span_len
            best_start, best_end = s, end

    if best_start is None:
        return ""

    # Within the chosen span, if another "Item 1A ... Risk Factors" heading
    # appears (the real one after a TOC entry), jump to the last occurrence.
    inner = [m.start() for m in start_re.finditer(text, best_start + 20, best_end)]
    if inner:
        best_start = inner[-1]

    span = text[best_start:best_end].strip()
    # Remove the leading "Item 1A. Risk Factors" heading itself
    span = re.sub(r'^Item\s*1A[\.\s\u2013\u2014\-:]*\s*Risk\s+Factors\s*',
                  '', span, count=1, flags=re.IGNORECASE)
    span = span.strip()

    # Cap enormous sections
    if len(span) > 200000:
        span = span[:200000]

    return span.strip()


# --- Defaults (locked in for this test) ---
DEFAULT_TICKER = "AMZN"          # Amazon
DEFAULT_YEAR = 2025              # filing year (Amazon files its FY2024 10-K in early 2025)
DEFAULT_BUCKET = "snp500-risk-radar-10k-data"
DEFAULT_REGION = "us-east-1"
DEFAULT_PREFIX = "risk-factors"
CONTACT_EMAIL = "velnraj@gmail.com"

# SEC asks for a descriptive User-Agent with real contact info.
SEC_HEADERS = {
    "User-Agent": f"S&P500 Risk Radar Research Tool {CONTACT_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}
SEC_DATA_HEADERS = {
    "User-Agent": f"S&P500 Risk Radar Research Tool {CONTACT_EMAIL}",
    "Accept-Encoding": "gzip, deflate",
    "Host": "data.sec.gov",
}

# SEC rate limit courtesy: max 10 req/sec. We stay well under.
RATE_LIMIT_DELAY = 0.2


def _get(url: str, host_headers: dict, timeout: int = 60) -> str:
    """Fetch a URL and return decoded text (handles gzip)."""
    import gzip
    req = urllib.request.Request(url, headers=host_headers)
    resp = urllib.request.urlopen(req, timeout=timeout)
    raw = resp.read()
    if resp.headers.get("Content-Encoding") == "gzip":
        raw = gzip.decompress(raw)
    return raw.decode("utf-8", errors="ignore")


def get_company_cik(ticker: str) -> str:
    """Look up the zero padded CIK for a ticker from SEC's mapping file."""
    url = "https://www.sec.gov/files/company_tickers.json"
    data = json.loads(_get(url, SEC_HEADERS, timeout=30))
    target = ticker.upper().replace("-", ".")
    for entry in data.values():
        if entry.get("ticker", "").upper() == target:
            return str(entry["cik_str"]).zfill(10)
    raise RuntimeError(f"Ticker {ticker} not found in SEC company list.")


def find_10k_for_year(cik: str, year: int) -> dict:
    """Find the 10-K filed in the given calendar year for this CIK."""
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = json.loads(_get(url, SEC_DATA_HEADERS, timeout=30))
    company_name = data.get("name", "")
    recent = data.get("filings", {}).get("recent", {})
    forms = recent.get("form", [])
    dates = recent.get("filingDate", [])
    accessions = recent.get("accessionNumber", [])
    primary_docs = recent.get("primaryDocument", [])

    for i, form in enumerate(forms):
        if form in ("10-K", "10-K/A"):
            filing_date = dates[i] if i < len(dates) else ""
            filing_year = int(filing_date[:4]) if filing_date else 0
            if filing_year == year:
                return {
                    "form": form,
                    "date": filing_date,
                    "year": filing_year,
                    "accession": accessions[i].replace("-", ""),
                    "accession_raw": accessions[i],
                    "primary_doc": primary_docs[i],
                    "cik": cik,
                    "company_name": company_name,
                }
    raise RuntimeError(f"No 10-K filed in {year} found for CIK {cik}.")


def download_item_1a(filing: dict) -> str:
    """Download the filing HTML and extract the Item 1A risk section."""
    cik = filing["cik"].lstrip("0")
    accession = filing["accession"]
    primary_doc = filing["primary_doc"]

    url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{primary_doc}"
    print(f"  [SEC] Fetching primary document: {url}")
    html = _get(url, SEC_HEADERS, timeout=90)

    item1a = extract_item_1a(html)
    if item1a and len(item1a) > 500:
        return item1a, url

    # Fallback: scan other .htm documents in the filing index.
    print("  [SEC] Primary doc extraction thin, trying other documents in the filing...")
    index_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/"
    import re
    index_html = _get(index_url, SEC_HEADERS, timeout=60)
    htm_files = re.findall(r'href="([^"]+\.htm[l]?)"', index_html, re.IGNORECASE)
    htm_files = [f.split("/")[-1] for f in htm_files]
    htm_files = [f for f in htm_files if f != primary_doc and not f[:2].upper().startswith("R")]

    for alt in htm_files[:6]:
        time.sleep(RATE_LIMIT_DELAY)
        alt_url = f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession}/{alt}"
        try:
            alt_html = _get(alt_url, SEC_HEADERS, timeout=60)
            alt_text = extract_item_1a(alt_html)
            if alt_text and len(alt_text) > 500:
                print(f"  [SEC] Extracted from alternate document: {alt}")
                return alt_text, alt_url
        except Exception:
            continue

    return item1a or "", url


def build_object_body(filing: dict, ticker: str, source_url: str, text: str) -> str:
    """Wrap the risk text with a small, self-documenting provenance header."""
    header = (
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
    )
    return header + text


def upload_to_s3(bucket: str, region: str, key: str, body: str):
    """Upload the text body to S3."""
    try:
        import boto3
    except ImportError:
        print("ERROR: boto3 not installed. Run: pip install boto3")
        sys.exit(1)

    s3 = boto3.client("s3", region_name=region)
    s3.put_object(
        Bucket=bucket,
        Key=key,
        Body=body.encode("utf-8"),
        ContentType="text/plain; charset=utf-8",
    )
    return f"s3://{bucket}/{key}"


def main():
    parser = argparse.ArgumentParser(description="Download one 10-K risk section and upload to S3 (test).")
    parser.add_argument("--ticker", default=DEFAULT_TICKER, help="Ticker (default: AMZN)")
    parser.add_argument("--year", type=int, default=DEFAULT_YEAR, help="Filing year (default: 2025)")
    parser.add_argument("--bucket", default=DEFAULT_BUCKET, help="S3 bucket name")
    parser.add_argument("--region", default=DEFAULT_REGION, help="AWS region")
    parser.add_argument("--prefix", default=DEFAULT_PREFIX, help="S3 key prefix")
    parser.add_argument("--dry-run", action="store_true", help="Extract only, do not upload to S3")
    args = parser.parse_args()

    print("=" * 64)
    print(f"  S&P 500 Risk Radar - single filing test")
    print(f"  Ticker: {args.ticker}   Filing year: {args.year}")
    print(f"  Bucket: {args.bucket}   Region: {args.region}")
    print(f"  Mode:   {'DRY RUN (no S3 upload)' if args.dry_run else 'REAL (will upload to S3)'}")
    print("=" * 64)

    print(f"\n  [1] Looking up CIK for {args.ticker}...")
    cik = get_company_cik(args.ticker)
    print(f"      CIK = {cik}")

    time.sleep(RATE_LIMIT_DELAY)
    print(f"  [2] Finding {args.year} 10-K...")
    filing = find_10k_for_year(cik, args.year)
    print(f"      {filing['company_name']}  |  {filing['form']}  |  filed {filing['date']}  |  accession {filing['accession_raw']}")

    time.sleep(RATE_LIMIT_DELAY)
    print(f"  [3] Downloading and extracting Item 1A (Risk Factors)...")
    text, source_url = download_item_1a(filing)

    if not text or len(text) < 500:
        print(f"\n  RESULT: extraction FAILED (got {len(text)} chars). The parser may not handle this filing format.")
        sys.exit(1)

    print(f"      Extracted {len(text):,} characters.")
    preview = text[:400].replace("\n", " ")
    print(f"\n  --- PREVIEW (first 400 chars) ---\n  {preview}...\n  ---------------------------------")

    key = f"{args.prefix}/{args.year}/{args.ticker.upper()}.txt"
    body = build_object_body(filing, args.ticker.upper(), source_url, text)

    if args.dry_run:
        print(f"\n  [4] DRY RUN: would upload to s3://{args.bucket}/{key}")
        print(f"      ({len(body):,} bytes including provenance header). Not uploaded.")
        print("\n  Dry run complete. Extraction works. Re-run without --dry-run to upload.")
        return

    print(f"\n  [4] Uploading to s3://{args.bucket}/{key} ...")
    uri = upload_to_s3(args.bucket, args.region, key, body)
    print(f"      Uploaded: {uri}")
    print("\n  SUCCESS. One filing is in S3. If this looks right, we build the bulk downloader next.")


if __name__ == "__main__":
    main()
