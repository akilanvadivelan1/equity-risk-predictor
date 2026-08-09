#!/usr/bin/env python3
"""
ERPSA — S&P 500 10-K Filing Downloader
========================================
Downloads and extracts Item 1A (Risk Factors) from all S&P 500 companies
for the specified years and stores them in a PostgreSQL database.

Usage:
    # Download all S&P 500 companies, 2025 and 2026 filings:
    python3 download_sp500.py

    # Download specific tickers:
    python3 download_sp500.py --tickers AAPL,MSFT,TGT

    # Download specific years:
    python3 download_sp500.py --years 2024,2025,2026

Environment Variables:
    DATABASE_URL  — PostgreSQL connection string (provided by Render)
                   Example: postgresql://user:pass@host:5432/dbname

Run on Render as a "Background Worker" or "Cron Job" (once daily or weekly).
"""

import os
import sys
import json
import time
import re
import urllib.request
import urllib.error
from datetime import datetime
from typing import List, Dict, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_cleaning import clean_text_preserve_structure

# ─── Configuration ───
SEC_HEADERS = {
    'User-Agent': 'ERPSA Research Tool admin@example.com',
    'Accept': 'application/json',
}

# Rate limit: SEC asks for max 10 requests/second
RATE_LIMIT_DELAY = 0.15  # 150ms between requests (~6 req/sec, conservative)

# Default years to download
DEFAULT_YEARS = [2025, 2026]


# ─── S&P 500 Ticker List ───
def get_sp500_tickers() -> List[str]:
    """
    Get current S&P 500 tickers from Wikipedia.
    Falls back to a hardcoded list if Wikipedia fetch fails.
    """
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    headers = {'User-Agent': 'ERPSA Research Tool'}

    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode('utf-8')

        # Extract tickers from the Wikipedia table
        # Pattern: ticker symbols in the first column of the table
        tickers = re.findall(r'<td[^>]*><a[^>]*>([A-Z]{1,5})</a>', html)

        if len(tickers) > 400:
            print(f"  [SP500] Fetched {len(tickers)} tickers from Wikipedia")
            return tickers[:505]  # S&P 500 sometimes has 503-505 due to dual-class shares
    except Exception as e:
        print(f"  [SP500] Wikipedia fetch failed: {e}")

    # Fallback: top 100 S&P 500 by market cap (as of 2025)
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


# ─── Database Functions ───
def get_db_connection():
    """Get PostgreSQL connection using DATABASE_URL."""
    try:
        import psycopg2
    except ImportError:
        print("ERROR: psycopg2 not installed. Run: pip install psycopg2-binary")
        sys.exit(1)

    db_url = os.environ.get('DATABASE_URL')
    if not db_url:
        print("ERROR: DATABASE_URL environment variable not set.")
        print("  On Render: add a PostgreSQL database and it's set automatically.")
        print("  Locally: export DATABASE_URL=postgresql://user:pass@localhost:5432/erpsa")
        sys.exit(1)

    return psycopg2.connect(db_url)


def init_database():
    """Create tables if they don't exist."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS filings (
            id SERIAL PRIMARY KEY,
            ticker VARCHAR(10) NOT NULL,
            year INTEGER NOT NULL,
            filing_date DATE,
            accession VARCHAR(30),
            item1a_text TEXT,
            item1a_length INTEGER DEFAULT 0,
            company_name VARCHAR(255),
            cik VARCHAR(12),
            downloaded_at TIMESTAMP DEFAULT NOW(),
            UNIQUE(ticker, year)
        )
    """)

    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_filings_ticker ON filings(ticker)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_filings_year ON filings(year)
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS idx_filings_ticker_year ON filings(ticker, year)
    """)

    conn.commit()
    cur.close()
    conn.close()
    print("  [DB] Database initialized.")


def save_filing(ticker: str, year: int, filing_date: str, accession: str,
                item1a_text: str, company_name: str, cik: str):
    """Save or update a filing in the database."""
    conn = get_db_connection()
    cur = conn.cursor()

    cur.execute("""
        INSERT INTO filings (ticker, year, filing_date, accession, item1a_text,
                            item1a_length, company_name, cik)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, year)
        DO UPDATE SET
            item1a_text = EXCLUDED.item1a_text,
            item1a_length = EXCLUDED.item1a_length,
            downloaded_at = NOW()
    """, (ticker, year, filing_date, accession, item1a_text,
          len(item1a_text) if item1a_text else 0, company_name, cik))

    conn.commit()
    cur.close()
    conn.close()


def get_existing_filings() -> set:
    """Get set of (ticker, year) pairs already downloaded."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute("SELECT ticker, year FROM filings WHERE item1a_length > 500")
    existing = {(row[0], row[1]) for row in cur.fetchall()}
    cur.close()
    conn.close()
    return existing


# ─── SEC EDGAR Functions ───
def get_company_cik(ticker: str) -> Optional[str]:
    """Look up company CIK from ticker."""
    url = 'https://www.sec.gov/files/company_tickers.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        ticker_upper = ticker.upper().replace('-', '.')
        for entry in data.values():
            if entry.get('ticker', '').upper() == ticker_upper:
                return str(entry['cik_str']).zfill(10)
    except Exception as e:
        print(f"    [EDGAR] CIK lookup failed for {ticker}: {e}")
    return None


def get_10k_filings_for_years(cik: str, years: List[int]) -> List[Dict]:
    """Get 10-K filings for specific years."""
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)

    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())

        company_name = data.get('name', '')
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])

        filings = []
        for i, form in enumerate(forms):
            if form in ('10-K', '10-K/A'):
                filing_date = dates[i] if i < len(dates) else ''
                year = int(filing_date[:4]) if filing_date else 0
                if year in years:
                    filings.append({
                        'form': form,
                        'date': filing_date,
                        'year': year,
                        'accession': accessions[i].replace('-', '') if i < len(accessions) else '',
                        'accession_raw': accessions[i] if i < len(accessions) else '',
                        'primary_doc': primary_docs[i] if i < len(primary_docs) else '',
                        'cik': cik,
                        'company_name': company_name,
                    })

        return filings
    except Exception as e:
        print(f"    [EDGAR] Filing list failed for CIK {cik}: {e}")
    return []


def extract_item_1a(html: str) -> str:
    """Extract Item 1A from filing HTML (same logic as app.py)."""
    # Import from app.py to reuse the same extraction logic
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('app', os.path.join(os.path.dirname(__file__), 'app.py'))
        app_mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(app_mod)
        return app_mod.extract_item_1a(html)
    except:
        # Fallback: simple extraction
        start_patterns = [
            re.compile(r'(?:Item|ITEM)\s*1A[\.\s\u2014\u2013\-:]*\s*(?:Risk\s*Factors|RISK\s*FACTORS)', re.IGNORECASE),
        ]
        end_patterns = [
            re.compile(r'(?:Item|ITEM)\s*1B', re.IGNORECASE),
            re.compile(r'(?:Item|ITEM)\s+2\.', re.IGNORECASE),
        ]

        start_pos = None
        for p in start_patterns:
            m = p.search(html)
            if m:
                start_pos = m.start()
                break
        if not start_pos:
            return ""

        end_pos = len(html)
        for p in end_patterns:
            m = p.search(html, start_pos + 200)
            if m:
                end_pos = min(end_pos, m.start())
                break

        section = html[start_pos:end_pos]
        cleaned = clean_text_preserve_structure(section)
        cleaned = re.sub(r'^\s*(?:Item|ITEM)\s*1A.*?(?:Risk\s*Factors|RISK\s*FACTORS)\s*', '', cleaned, count=1)
        return cleaned.strip()


def download_filing(filing: Dict) -> str:
    """Download and extract Item 1A from a single filing."""
    cik = filing['cik']
    accession = filing['accession']
    primary_doc = filing['primary_doc']

    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"

    try:
        req = urllib.request.Request(url, headers=SEC_HEADERS)
        resp = urllib.request.urlopen(req, timeout=60)
        html = resp.read().decode('utf-8', errors='ignore')

        item1a = extract_item_1a(html)
        if item1a and len(item1a) > 500:
            return item1a

        # Fallback: try other .htm files in the filing
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/"
        req2 = urllib.request.Request(index_url, headers=SEC_HEADERS)
        resp2 = urllib.request.urlopen(req2, timeout=30)
        index_html = resp2.read().decode('utf-8', errors='ignore')

        htm_files = re.findall(r'href="([^"]+\.htm[l]?)"', index_html, re.IGNORECASE)
        htm_files = [f for f in htm_files if f != primary_doc and not f.startswith('R')]

        for alt_doc in htm_files[:5]:
            alt_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{alt_doc}"
            try:
                req3 = urllib.request.Request(alt_url, headers=SEC_HEADERS)
                resp3 = urllib.request.urlopen(req3, timeout=30)
                alt_html = resp3.read().decode('utf-8', errors='ignore')
                alt_text = extract_item_1a(alt_html)
                if alt_text and len(alt_text) > 500:
                    return alt_text
            except:
                continue
            time.sleep(RATE_LIMIT_DELAY)

    except Exception as e:
        print(f"    [DOWNLOAD] Error: {e}")

    return ""


# ─── Main Download Loop ───
def download_all(tickers: List[str], years: List[int], skip_existing: bool = True):
    """Download all filings for given tickers and years."""
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  ERPSA — S&P 500 Filing Downloader                         ║
║  Tickers: {len(tickers):<5} | Years: {years}              ║
╚══════════════════════════════════════════════════════════════╝
""")

    # Initialize database
    init_database()

    # Check what's already downloaded
    existing = set()
    if skip_existing:
        existing = get_existing_filings()
        print(f"  [DB] Already have {len(existing)} filings in database. Skipping those.")

    # Get CIK mapping
    print(f"\n  [STEP 1] Looking up CIK numbers for {len(tickers)} tickers...")
    ticker_cik_url = 'https://www.sec.gov/files/company_tickers.json'
    req = urllib.request.Request(ticker_cik_url, headers=SEC_HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        all_companies = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  ERROR: Could not fetch ticker list: {e}")
        return

    cik_map = {}
    for entry in all_companies.values():
        t = entry.get('ticker', '').upper()
        if t in [tk.upper() for tk in tickers]:
            cik_map[t] = str(entry['cik_str']).zfill(10)

    print(f"  [STEP 1] Found CIKs for {len(cik_map)}/{len(tickers)} tickers")

    # Download filings
    print(f"\n  [STEP 2] Downloading filings...")
    total = len(cik_map) * len(years)
    downloaded = 0
    failed = 0
    skipped = 0

    for idx, (ticker, cik) in enumerate(cik_map.items()):
        print(f"\n  [{idx+1}/{len(cik_map)}] {ticker} (CIK: {cik})")

        # Check which years we still need
        years_needed = [y for y in years if (ticker, y) not in existing]
        if not years_needed:
            print(f"    Already have all years. Skipping.")
            skipped += len(years)
            continue

        # Get filing list
        time.sleep(RATE_LIMIT_DELAY)
        filings = get_10k_filings_for_years(cik, years_needed)

        if not filings:
            print(f"    No 10-K filings found for years {years_needed}")
            failed += len(years_needed)
            continue

        for filing in filings:
            year = filing['year']
            if (ticker, year) in existing:
                skipped += 1
                continue

            print(f"    Downloading {year} 10-K...", end=" ", flush=True)
            time.sleep(RATE_LIMIT_DELAY)

            item1a = download_filing(filing)

            if item1a and len(item1a) > 500:
                save_filing(
                    ticker=ticker,
                    year=year,
                    filing_date=filing['date'],
                    accession=filing['accession_raw'],
                    item1a_text=item1a,
                    company_name=filing.get('company_name', ''),
                    cik=cik,
                )
                downloaded += 1
                print(f"OK ({len(item1a):,} chars)")
            else:
                # Save empty record so we don't retry
                save_filing(
                    ticker=ticker,
                    year=year,
                    filing_date=filing.get('date', ''),
                    accession=filing.get('accession_raw', ''),
                    item1a_text='',
                    company_name=filing.get('company_name', ''),
                    cik=cik,
                )
                failed += 1
                print(f"FAILED (could not extract Item 1A)")

    # Summary
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║  DOWNLOAD COMPLETE                                         ║
╠══════════════════════════════════════════════════════════════╣
║  Downloaded: {downloaded:<6}                                    ║
║  Failed:     {failed:<6}                                    ║
║  Skipped:    {skipped:<6} (already in DB)                   ║
║  Total:      {downloaded + failed + skipped:<6}                                    ║
╚══════════════════════════════════════════════════════════════╝
""")


# ─── CLI Entry Point ───
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download S&P 500 10-K filings')
    parser.add_argument('--tickers', type=str, default=None,
                       help='Comma-separated list of tickers (default: S&P 500)')
    parser.add_argument('--years', type=str, default=None,
                       help='Comma-separated years (default: 2025,2026)')
    parser.add_argument('--no-skip', action='store_true',
                       help='Re-download even if already in database')
    args = parser.parse_args()

    # Parse tickers
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(',')]
    else:
        tickers = get_sp500_tickers()

    # Parse years
    if args.years:
        years = [int(y.strip()) for y in args.years.split(',')]
    else:
        years = DEFAULT_YEARS

    download_all(tickers, years, skip_existing=not args.no_skip)
