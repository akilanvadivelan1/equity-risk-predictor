#!/usr/bin/env python3
"""
ERPSA — Equity Risk Predictor & Sentiment Analyzer
Web Application v1.0

A complete web interface that:
1. Fetches real 10-K filings from SEC EDGAR (free, no API key)
2. Extracts Item 1A (Risk Factors) automatically
3. Compares risk language across years
4. Scores and explains each risk in plain English

Run: python3 app.py
Open: http://localhost:8888
"""

import sys
import os
import re
import json
import time
import urllib.request
import urllib.error
from html import unescape
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse, quote
from typing import Optional, Dict, List, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from text_cleaning import clean_text_preserve_structure
from risk_section_parser import parse_risk_sections
from risk_matcher import match_risk_categories, RiskChangeStatus
from risk_classifier import classify_risk_changes
from step_3_scoring import run_scoring
from sentiment_scorer import score_text_sentiment
from lm_dictionary import get_dictionary


# =========================================================
# SEC EDGAR Integration
# =========================================================

SEC_HEADERS = {
    'User-Agent': 'ERPSA Research Tool admin@example.com',
    'Accept': 'application/json',
}

# Cache to avoid re-fetching
_filing_cache: Dict[str, str] = {}


def get_company_cik(ticker: str) -> Optional[str]:
    """Look up company CIK number from ticker."""
    url = 'https://www.sec.gov/files/company_tickers.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        ticker_upper = ticker.upper()
        for entry in data.values():
            if entry.get('ticker', '').upper() == ticker_upper:
                return str(entry['cik_str']).zfill(10)
    except Exception as e:
        print(f"  [EDGAR] Error looking up ticker {ticker}: {e}")
    return None


def get_10k_filings(cik: str) -> List[Dict]:
    """Get list of 10-K filings for a company."""
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())

        filings = []
        recent = data.get('filings', {}).get('recent', {})
        forms = recent.get('form', [])
        dates = recent.get('filingDate', [])
        accessions = recent.get('accessionNumber', [])
        primary_docs = recent.get('primaryDocument', [])

        for i, form in enumerate(forms):
            if form in ('10-K', '10-K/A'):
                filing_date = dates[i] if i < len(dates) else ''
                year = int(filing_date[:4]) if filing_date else 0
                filings.append({
                    'form': form,
                    'date': filing_date,
                    'year': year,
                    'accession': accessions[i].replace('-', '') if i < len(accessions) else '',
                    'accession_raw': accessions[i] if i < len(accessions) else '',
                    'primary_doc': primary_docs[i] if i < len(primary_docs) else '',
                    'cik': cik,
                })

        return filings[:15]  # Last 15 filings
    except Exception as e:
        print(f"  [EDGAR] Error fetching filings for CIK {cik}: {e}")
    return []


# =========================================================
# S3 Storage (primary source for pre-extracted risk factors)
# =========================================================

RISK_S3_BUCKET = os.environ.get('RISK_S3_BUCKET', 'snp500-risk-radar-10k-data')
RISK_S3_PREFIX = os.environ.get('RISK_S3_PREFIX', 'risk-factors')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# In-memory caches so we hit S3 at most once per ticker/year.
_s3_manifest_cache: Optional[Dict] = None
_s3_year_cache: Dict[str, List[int]] = {}


def _get_s3_client():
    """Return a boto3 S3 client, or None if boto3/credentials are unavailable."""
    try:
        import boto3
        return boto3.client('s3', region_name=AWS_REGION)
    except Exception as e:
        print(f"  [S3] boto3 unavailable ({e}). S3 reads disabled.")
        return None


def _strip_provenance_header(body: str) -> str:
    """Remove the provenance header the downloader writes above the risk text."""
    marker = '-' * 60
    if marker in body:
        return body.split(marker, 1)[1].strip()
    return body.strip()


def list_s3_years_for_ticker(ticker: str) -> List[int]:
    """
    Return the filing years available in S3 for a ticker, newest first.

    Discovers years by listing objects under each risk-factors/<year>/ prefix
    and checking for <TICKER>.txt. Results are cached in memory.
    """
    ticker = ticker.upper()
    if ticker in _s3_year_cache:
        return _s3_year_cache[ticker]

    s3 = _get_s3_client()
    if s3 is None:
        return []

    years = []
    try:
        # List the year "folders" under the prefix, then check each for the ticker.
        resp = s3.list_objects_v2(
            Bucket=RISK_S3_BUCKET,
            Prefix=f"{RISK_S3_PREFIX}/",
            Delimiter='/',
        )
        year_prefixes = [p['Prefix'] for p in resp.get('CommonPrefixes', [])]
        for yp in year_prefixes:
            # yp looks like "risk-factors/2025/"
            m = re.search(r'/(\d{4})/$', yp)
            if not m:
                continue
            year = int(m.group(1))
            key = f"{RISK_S3_PREFIX}/{year}/{ticker}.txt"
            try:
                s3.head_object(Bucket=RISK_S3_BUCKET, Key=key)
                years.append(year)
            except Exception:
                continue
    except Exception as e:
        print(f"  [S3] Could not list years for {ticker}: {e}")

    years.sort(reverse=True)
    _s3_year_cache[ticker] = years
    return years


def read_s3_risk_text(ticker: str, year: int) -> Optional[str]:
    """Read the pre-extracted risk section for a ticker/year from S3."""
    if not ticker or not year:
        return None
    s3 = _get_s3_client()
    if s3 is None:
        return None
    key = f"{RISK_S3_PREFIX}/{int(year)}/{ticker.upper()}.txt"
    try:
        obj = s3.get_object(Bucket=RISK_S3_BUCKET, Key=key)
        body = obj['Body'].read().decode('utf-8', errors='ignore')
        text = _strip_provenance_header(body)
        print(f"  [S3] Loaded {ticker} {year} from s3://{RISK_S3_BUCKET}/{key} ({len(text):,} chars)")
        return text
    except Exception:
        return None   # not in S3, caller falls back to live SEC


def get_company_name_from_s3(ticker: str, year: int) -> str:
    """Read the COMPANY field from a stored file's provenance header, if present."""
    s3 = _get_s3_client()
    if s3 is None:
        return ticker.upper()
    key = f"{RISK_S3_PREFIX}/{int(year)}/{ticker.upper()}.txt"
    try:
        obj = s3.get_object(Bucket=RISK_S3_BUCKET, Key=key)
        head = obj['Body'].read(400).decode('utf-8', errors='ignore')
        m = re.search(r'COMPANY:\s*(.+)', head)
        if m:
            return m.group(1).strip()
    except Exception:
        pass
    return ticker.upper()


def _check_database(ticker: str, year: int) -> Optional[str]:
    """Check if a filing is already in the PostgreSQL database."""
    db_url = os.environ.get('DATABASE_URL')
    if not db_url or not ticker or not year:
        return None

    try:
        import psycopg2
        conn = psycopg2.connect(db_url)
        cur = conn.cursor()
        cur.execute(
            "SELECT item1a_text FROM filings WHERE ticker = %s AND year = %s AND item1a_length > 500",
            (ticker.upper(), year)
        )
        row = cur.fetchone()
        cur.close()
        conn.close()
        if row and row[0]:
            return row[0]
    except ImportError:
        # psycopg2 not installed — skip database check
        pass
    except Exception as e:
        print(f"  [DB] Database check failed (non-critical): {e}")
    return None


def fetch_filing_text(filing: Dict, ticker: str = '') -> str:
    """Download and extract Item 1A from a 10-K filing.
    
    First checks the PostgreSQL database for pre-downloaded filings.
    Falls back to live SEC EDGAR fetch if not in database.
    """
    cache_key = filing.get('accession', '')
    if cache_key in _filing_cache:
        return _filing_cache[cache_key]

    year = filing.get('year', 0)

    # ─── Check S3 first (primary source, pre-extracted risk factors) ───
    s3_text = read_s3_risk_text(ticker, year)
    if s3_text and len(s3_text) > 500:
        _filing_cache[cache_key] = s3_text
        return s3_text

    # ─── Then the PostgreSQL database, if configured ───
    db_text = _check_database(ticker, year)
    if db_text:
        _filing_cache[cache_key] = db_text
        print(f"  [DB] Found pre-downloaded filing for {ticker} {year} ({len(db_text):,} chars)")
        return db_text

    cik = filing['cik']
    accession = filing['accession']
    accession_raw = filing.get('accession_raw', '')
    primary_doc = filing['primary_doc']

    # Try to get the filing document
    url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{primary_doc}"
    print(f"  [EDGAR] Fetching: {url}")

    req = urllib.request.Request(url, headers=SEC_HEADERS)
    try:
        resp = urllib.request.urlopen(req, timeout=60)
        html = resp.read().decode('utf-8', errors='ignore')

        # Extract Item 1A section
        item1a_text = extract_item_1a(html)

        if item1a_text and len(item1a_text) > 500:
            _filing_cache[cache_key] = item1a_text
            print(f"  [EDGAR] Extracted {len(item1a_text)} chars of Item 1A text")
            return item1a_text

        # ─── Fallback 1: Try the filing index to find other documents ───
        print(f"  [EDGAR] Primary doc extraction failed, trying filing index...")
        index_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/"
        req2 = urllib.request.Request(index_url, headers=SEC_HEADERS)
        try:
            resp2 = urllib.request.urlopen(req2, timeout=30)
            index_html = resp2.read().decode('utf-8', errors='ignore')

            # Look for .htm files in the index (exclude the primary doc we already tried)
            htm_files = re.findall(r'href="([^"]+\.htm)"', index_html, re.IGNORECASE)
            htm_files = [f for f in htm_files if f != primary_doc and 'R' not in f[:2]]

            for alt_doc in htm_files[:5]:  # Try up to 5 alternative documents
                alt_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{alt_doc}"
                print(f"  [EDGAR] Trying alternative: {alt_doc}")
                try:
                    req3 = urllib.request.Request(alt_url, headers=SEC_HEADERS)
                    resp3 = urllib.request.urlopen(req3, timeout=30)
                    alt_html = resp3.read().decode('utf-8', errors='ignore')
                    alt_text = extract_item_1a(alt_html)
                    if alt_text and len(alt_text) > 500:
                        _filing_cache[cache_key] = alt_text
                        print(f"  [EDGAR] Success with {alt_doc}: {len(alt_text)} chars")
                        return alt_text
                except:
                    continue
                time.sleep(0.3)  # Rate limit courtesy
        except Exception as e2:
            print(f"  [EDGAR] Index fallback failed: {e2}")

        # ─── Fallback 2: Try the full submission text file ───
        print(f"  [EDGAR] Trying full submission text...")
        # The full text file uses the accession number formatted with dashes
        if accession_raw:
            txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{accession_raw}.txt"
            print(f"  [EDGAR] Full text URL: {txt_url}")
            try:
                req4 = urllib.request.Request(txt_url, headers=SEC_HEADERS)
                resp4 = urllib.request.urlopen(req4, timeout=60)
                full_text = resp4.read().decode('utf-8', errors='ignore')
                full_item1a = extract_item_1a(full_text)
                if full_item1a and len(full_item1a) > 500:
                    _filing_cache[cache_key] = full_item1a
                    print(f"  [EDGAR] Success with full text: {len(full_item1a)} chars")
                    return full_item1a
            except Exception as e3:
                print(f"  [EDGAR] Full text fallback failed: {e3}")

        # ─── Fallback 3: Try SEC EDGAR viewer (sections API) ───
        # This endpoint returns filing sections in a more parseable format
        print(f"  [EDGAR] Trying EDGAR section viewer...")
        try:
            # Try the filing with .htm extension variations
            base_name = primary_doc.rsplit('.', 1)[0] if '.' in primary_doc else primary_doc
            for ext in ['.htm', '.html', '-0001.htm', '_htm.xml']:
                try_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{base_name}{ext}"
                try:
                    req5 = urllib.request.Request(try_url, headers=SEC_HEADERS)
                    resp5 = urllib.request.urlopen(req5, timeout=30)
                    alt_html = resp5.read().decode('utf-8', errors='ignore')
                    alt_text = extract_item_1a(alt_html)
                    if alt_text and len(alt_text) > 500:
                        _filing_cache[cache_key] = alt_text
                        print(f"  [EDGAR] Success with {base_name}{ext}: {len(alt_text)} chars")
                        return alt_text
                except:
                    continue
        except:
            pass

        # ─── Fallback 4: Brute force — try ALL .htm files in the filing ───
        print(f"  [EDGAR] Brute force: trying all available documents...")
        try:
            # Re-fetch index if needed and try ALL .htm/.html files
            idx_url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={cik.lstrip('0')}&type=10-K&dateb=&owner=include&count=1&search_text=&accession={accession_raw}"
            # Actually just try common filing patterns
            common_patterns = [
                f"{ticker.lower()}-{filing.get('date','')[:4]}0928.htm",
                f"{ticker.lower()}-{filing.get('date','')[:4]}1231.htm",
                f"{ticker.lower()}-{filing.get('date','')[:4]}0630.htm",
                f"{ticker.lower()}-{filing.get('date','')[:4]}0331.htm",
                f"{ticker.lower()}20{filing.get('date','')[:4][-2:]}10k.htm",
                "0001.htm",
                "d10k.htm",
                "form10-k.htm",
                "form10k.htm",
            ]
            for pattern in common_patterns:
                try_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{pattern}"
                try:
                    req6 = urllib.request.Request(try_url, headers=SEC_HEADERS)
                    resp6 = urllib.request.urlopen(req6, timeout=20)
                    alt_html = resp6.read().decode('utf-8', errors='ignore')
                    alt_text = extract_item_1a(alt_html)
                    if alt_text and len(alt_text) > 500:
                        _filing_cache[cache_key] = alt_text
                        print(f"  [EDGAR] Success with pattern {pattern}: {len(alt_text)} chars")
                        return alt_text
                except:
                    continue
                time.sleep(0.2)
        except:
            pass

        return f"[Could not extract Item 1A from this filing. Tried primary document and alternatives. The filing may use a format our parser cannot yet handle. Filing URL: {url}]"
    except Exception as e:
        print(f"  [EDGAR] Error fetching document: {e}")
        return f"[Error fetching filing: {str(e)}]"


def extract_item_1a(html: str) -> str:
    """
    Extract Item 1A (Risk Factors) section from a 10-K HTML document.

    Handles multiple formats:
    - Standard HTML filings with clear Item headers
    - Inline XBRL (iXBRL) filings (modern format used by Apple, etc.)
    - Filings with table of contents (skips TOC entries)
    - Filings with various formatting styles (bold, caps, spans, divs)
    """
    # Strategy: convert the HTML to clean plain text FIRST, then locate the
    # section in the clean text. This is far more reliable for inline XBRL
    # (iXBRL) filings (Amazon, Apple, etc.), where "Item 1A" and "Risk Factors"
    # are separated by many nested tags in the raw HTML. Validated on Amazon's
    # 2025 10-K (was extracting 116 chars, now extracts the full section).
    text = html
    text = re.sub(r'<!--.*?-->', ' ', text, flags=re.DOTALL)
    text = re.sub(r'<(script|style)[^>]*>.*?</\1>', ' ', text, flags=re.DOTALL | re.IGNORECASE)
    # Turn block-level boundaries into spaces so words do not run together
    text = re.sub(r'<(br|/p|/div|/tr|/td|/th|/li|/h[1-6])[^>]*>', ' ', text, flags=re.IGNORECASE)
    text = re.sub(r'<[^>]+>', '', text)          # drop all remaining tags
    text = unescape(text)                         # decode &amp; &#160; etc.
    text = text.replace('\u00a0', ' ')            # non breaking space -> space
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\s{2,}', ' ', text)

    # Start headings: "Item 1A" then (allowing punctuation/space) "Risk Factors"
    start_re = re.compile(r'Item\s*1A[\.\s\u2013\u2014\-:]*\s*Risk\s+Factors', re.IGNORECASE)
    starts = [m.start() for m in start_re.finditer(text)]
    if not starts:
        starts = [m.start() for m in re.finditer(r'Item\s*1A\b', text, re.IGNORECASE)]
    if not starts:
        return ""

    # End headings: Item 1B (Unresolved Staff Comments) or Item 2 (Properties)
    end_re = re.compile(r'Item\s*1B\b|Item\s*2[\.\s\u2013\u2014\-:]+\s*Propert', re.IGNORECASE)

    # Pick the longest Item 1A -> end span (the real section, not a TOC entry)
    best_start, best_end, best_len = None, None, 0
    for s in starts:
        m = end_re.search(text, s + 50)
        end = m.start() if m else len(text)
        if end - s > best_len:
            best_len = end - s
            best_start, best_end = s, end
    if best_start is None:
        return ""

    # If the chosen span begins at a TOC entry, a later real heading usually
    # appears inside it. Jump to the last such heading before the end.
    inner = [m.start() for m in start_re.finditer(text, best_start + 20, best_end)]
    if inner:
        best_start = inner[-1]

    span = text[best_start:best_end].strip()
    # Remove the leading "Item 1A. Risk Factors" heading itself
    span = re.sub(r'^Item\s*1A[\.\s\u2013\u2014\-:]*\s*Risk\s+Factors\s*',
                  '', span, count=1, flags=re.IGNORECASE)
    span = span.strip()

    # Limit to reasonable size (some filings are enormous)
    if len(span) > 200000:
        span = span[:200000]

    return span.strip()


def get_available_years(ticker: str) -> Dict:
    """Get available 10-K filing years for a ticker."""
    cik = get_company_cik(ticker)
    if not cik:
        return {'error': f'Ticker "{ticker}" not found in SEC EDGAR.', 'years': [], 'company': ''}

    # Get company name
    url = f'https://data.sec.gov/submissions/CIK{cik}.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)
    company_name = ticker.upper()
    try:
        resp = urllib.request.urlopen(req, timeout=15)
        data = json.loads(resp.read().decode())
        company_name = data.get('name', ticker.upper())
    except:
        pass

    filings = get_10k_filings(cik)
    years = []
    for f in filings:
        if f['year'] > 0:
            years.append({
                'year': f['year'],
                'date': f['date'],
                'form': f['form'],
            })

    return {
        'ticker': ticker.upper(),
        'company': company_name,
        'cik': cik,
        'years': years,
        'filings': filings,
    }


# =========================================================
# Financial Data & Stock Analysis
# =========================================================

def fetch_company_financials(cik: str, ticker: str) -> Dict:
    """
    Fetch key financial data from SEC EDGAR's XBRL companyfacts API.
    Returns revenue, net income, total assets, liabilities, cash, and more.
    """
    url = f'https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json'
    req = urllib.request.Request(url, headers=SEC_HEADERS)

    try:
        resp = urllib.request.urlopen(req, timeout=20)
        data = json.loads(resp.read().decode())
    except Exception as e:
        print(f"  [FINANCIALS] Error fetching XBRL data: {e}")
        return {}

    facts = data.get('facts', {})
    us_gaap = facts.get('us-gaap', {})

    def get_annual_values(concept: str, max_entries: int = 5) -> List[Dict]:
        """Extract annual (10-K) values for a concept."""
        concept_data = us_gaap.get(concept, {})
        units = concept_data.get('units', {})
        # Try USD first, then shares
        entries = units.get('USD', units.get('shares', []))
        # Filter to annual (10-K) filings only
        annual = [e for e in entries if e.get('form') == '10-K']
        # Sort by end date descending
        annual.sort(key=lambda x: x.get('end', ''), reverse=True)
        # Deduplicate by fiscal year
        seen_years = set()
        unique = []
        for entry in annual:
            end_date = entry.get('end', '')
            year = end_date[:4] if end_date else ''
            if year and year not in seen_years:
                seen_years.add(year)
                unique.append({
                    'value': entry.get('val', 0),
                    'end': end_date,
                    'year': int(year) if year else 0,
                    'filed': entry.get('filed', ''),
                })
            if len(unique) >= max_entries:
                break
        return unique

    # Pull key financial metrics
    financials = {
        'revenue': get_annual_values('Revenues') or get_annual_values('RevenueFromContractWithCustomerExcludingAssessedTax') or get_annual_values('SalesRevenueNet'),
        'net_income': get_annual_values('NetIncomeLoss'),
        'total_assets': get_annual_values('Assets'),
        'total_liabilities': get_annual_values('Liabilities'),
        'stockholders_equity': get_annual_values('StockholdersEquity'),
        'cash': get_annual_values('CashAndCashEquivalentsAtCarryingValue'),
        'operating_income': get_annual_values('OperatingIncomeLoss'),
        'total_debt': get_annual_values('LongTermDebt') or get_annual_values('LongTermDebtNoncurrent'),
        'shares_outstanding': get_annual_values('CommonStockSharesOutstanding'),
        'eps': get_annual_values('EarningsPerShareDiluted'),
    }

    return financials


def fetch_zacks_rank(ticker: str) -> Dict:
    """
    Fetch the current Zacks Rank for a ticker from zacks.com.
    
    Zacks Rank is based on 4 factors:
    1. Agreement — % of analysts revising estimates in same direction
    2. Magnitude — size of recent consensus estimate changes
    3. Upside — difference between most accurate estimate and consensus
    4. Surprise — recent earnings surprise history
    
    Returns:
        Dict with rank (1-5), signal text, and metadata.
        Returns empty dict if fetch fails.
    """
    url = f'https://www.zacks.com/stock/quote/{ticker.upper()}'
    headers = {
        'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
        'Accept': 'text/html,application/xhtml+xml',
    }

    try:
        req = urllib.request.Request(url, headers=headers)
        resp = urllib.request.urlopen(req, timeout=15)
        html = resp.read().decode('utf-8', errors='ignore')

        # Extract Zacks Rank from the page
        # The rank is typically in a span with class "rank_chip" or in data attributes
        rank = None
        rank_text = None

        # Pattern 1: rank_chip class with rank number
        match = re.search(r'class="rank_chip[^"]*ranklist_(\d)"', html)
        if match:
            rank = int(match.group(1))

        # Pattern 2: zacks rank text
        if not rank:
            match = re.search(r'<p[^>]*class="rank_view"[^>]*>.*?(\d)\s*-\s*(Strong\s*Buy|Buy|Hold|Sell|Strong\s*Sell)', html, re.DOTALL | re.IGNORECASE)
            if match:
                rank = int(match.group(1))

        # Pattern 3: Look for rank in JSON data on page
        if not rank:
            match = re.search(r'"zacks_rank"\s*:\s*"?(\d)"?', html)
            if match:
                rank = int(match.group(1))

        # Pattern 4: Direct text match
        if not rank:
            match = re.search(r'Zacks\s+Rank\s*:?\s*#?(\d)\s*[-–]\s*(Strong\s*Buy|Buy|Hold|Sell|Strong\s*Sell)', html, re.IGNORECASE)
            if match:
                rank = int(match.group(1))

        if rank and 1 <= rank <= 5:
            rank_labels = {1: 'Strong Buy', 2: 'Buy', 3: 'Hold', 4: 'Sell', 5: 'Strong Sell'}
            rank_colors = {1: '#22c55e', 2: '#4ade80', 3: '#fbbf24', 4: '#fb923c', 5: '#ef4444'}
            return {
                'rank': rank,
                'signal': rank_labels.get(rank, 'Unknown'),
                'color': rank_colors.get(rank, '#9ca3af'),
                'source': 'Zacks Investment Research',
                'description': 'Based on earnings estimate revisions by Wall Street analysts',
                'available': True,
            }

        # If we got the page but couldn't parse the rank
        print(f"  [ZACKS] Could not parse rank from page for {ticker}")
        return {'available': False, 'reason': 'Could not parse Zacks Rank from page'}

    except urllib.error.HTTPError as e:
        print(f"  [ZACKS] HTTP {e.code} for {ticker}")
        return {'available': False, 'reason': f'HTTP {e.code}'}
    except Exception as e:
        print(f"  [ZACKS] Error fetching Zacks rank for {ticker}: {e}")
        return {'available': False, 'reason': str(e)}


def generate_stock_analysis(financials: Dict, ticker: str, company: str, current_year: int) -> Dict:
    """
    Generate a comprehensive stock analysis from financial data.
    Returns structured data for frontend rendering.
    """
    analysis = {
        'company': company,
        'ticker': ticker,
        'metrics': [],
        'health_score': 0,
        'health_label': '',
        'summary': '',
        'strengths': [],
        'concerns': [],
        'trend_analysis': [],
    }

    if not financials or not any(financials.values()):
        analysis['summary'] = 'Financial data not available from SEC EDGAR for this company.'
        return analysis

    def latest(key):
        vals = financials.get(key, [])
        return vals[0]['value'] if vals else None

    def prior(key):
        vals = financials.get(key, [])
        return vals[1]['value'] if len(vals) > 1 else None

    def fmt_billions(val):
        if val is None: return 'N/A'
        if abs(val) >= 1e9: return f"${val/1e9:.1f}B"
        if abs(val) >= 1e6: return f"${val/1e6:.0f}M"
        return f"${val:,.0f}"

    def fmt_pct(val):
        if val is None: return 'N/A'
        return f"{val*100:.1f}%"

    def yoy_change(key):
        curr = latest(key)
        prev = prior(key)
        if curr and prev and prev != 0:
            return (curr - prev) / abs(prev)
        return None

    # ─── Compute Key Metrics ───
    rev = latest('revenue')
    rev_prior = prior('revenue')
    ni = latest('net_income')
    ni_prior = prior('net_income')
    assets = latest('total_assets')
    liabilities = latest('total_liabilities')
    equity = latest('stockholders_equity')
    cash = latest('cash')
    debt = latest('total_debt')
    op_income = latest('operating_income')

    health_points = 0
    max_points = 0

    # Revenue
    if rev:
        rev_growth = yoy_change('revenue')
        analysis['metrics'].append({
            'name': 'Revenue',
            'value': fmt_billions(rev),
            'change': f"{rev_growth*100:+.1f}%" if rev_growth is not None else None,
            'trend': 'up' if rev_growth and rev_growth > 0 else 'down' if rev_growth and rev_growth < 0 else 'flat',
        })
        if rev_growth is not None:
            max_points += 20
            if rev_growth > 0.05: health_points += 20
            elif rev_growth > 0: health_points += 12
            elif rev_growth > -0.05: health_points += 6
            if rev_growth > 0.1:
                analysis['strengths'].append(f"Strong revenue growth of {rev_growth*100:.1f}% year-over-year")
            elif rev_growth < -0.05:
                analysis['concerns'].append(f"Revenue declined {rev_growth*100:.1f}% — a warning sign for future performance")

    # Net Income / Profitability
    if ni is not None and rev:
        profit_margin = ni / rev if rev != 0 else 0
        ni_growth = yoy_change('net_income')
        analysis['metrics'].append({
            'name': 'Net Income',
            'value': fmt_billions(ni),
            'change': f"{ni_growth*100:+.1f}%" if ni_growth is not None else None,
            'trend': 'up' if ni_growth and ni_growth > 0 else 'down' if ni_growth and ni_growth < 0 else 'flat',
        })
        analysis['metrics'].append({
            'name': 'Profit Margin',
            'value': fmt_pct(profit_margin),
            'change': None,
            'trend': 'up' if profit_margin > 0.15 else 'flat' if profit_margin > 0.05 else 'down',
        })
        max_points += 20
        if profit_margin > 0.15: health_points += 20
        elif profit_margin > 0.08: health_points += 14
        elif profit_margin > 0: health_points += 8
        if profit_margin > 0.2:
            analysis['strengths'].append(f"Excellent profit margin of {profit_margin*100:.1f}% — highly profitable business")
        elif ni < 0:
            analysis['concerns'].append(f"Company is unprofitable (net loss of {fmt_billions(ni)})")

    # Balance Sheet Health
    if assets and liabilities:
        debt_ratio = liabilities / assets
        analysis['metrics'].append({
            'name': 'Debt-to-Assets Ratio',
            'value': f"{debt_ratio:.2f}",
            'change': None,
            'trend': 'up' if debt_ratio > 0.7 else 'flat' if debt_ratio > 0.5 else 'down',
        })
        max_points += 20
        if debt_ratio < 0.5: health_points += 20
        elif debt_ratio < 0.65: health_points += 14
        elif debt_ratio < 0.8: health_points += 8
        if debt_ratio > 0.8:
            analysis['concerns'].append(f"High leverage — liabilities are {debt_ratio*100:.0f}% of total assets, indicating financial stress")
        elif debt_ratio < 0.4:
            analysis['strengths'].append(f"Conservative balance sheet with low leverage ({debt_ratio*100:.0f}% debt-to-assets)")

    # Cash Position
    if cash and assets:
        cash_ratio = cash / assets
        analysis['metrics'].append({
            'name': 'Cash & Equivalents',
            'value': fmt_billions(cash),
            'change': None,
            'trend': 'up' if cash_ratio > 0.15 else 'flat',
        })
        max_points += 20
        if cash_ratio > 0.2: health_points += 20
        elif cash_ratio > 0.1: health_points += 14
        elif cash_ratio > 0.05: health_points += 8
        if cash_ratio > 0.25:
            analysis['strengths'].append(f"Strong cash position ({cash_ratio*100:.0f}% of assets) — well-prepared for downturns")
        elif cash_ratio < 0.03:
            analysis['concerns'].append(f"Very low cash reserves ({cash_ratio*100:.1f}% of assets) — vulnerable to liquidity crunches")

    # Equity
    if equity:
        analysis['metrics'].append({
            'name': 'Stockholders Equity',
            'value': fmt_billions(equity),
            'change': None,
            'trend': 'up' if equity > 0 else 'down',
        })
        max_points += 20
        if equity > 0: health_points += 15
        if equity and assets and equity / assets > 0.4: health_points += 5
        if equity < 0:
            analysis['concerns'].append("Negative stockholders' equity — the company owes more than it owns (common for tech companies with heavy buybacks)")

    # Total Assets
    if assets:
        analysis['metrics'].append({
            'name': 'Total Assets',
            'value': fmt_billions(assets),
            'change': None,
            'trend': 'flat',
        })

    # ─── Trend Analysis (YoY comparisons) ───
    trends = []
    for key, label in [('revenue', 'Revenue'), ('net_income', 'Net Income'), ('total_assets', 'Total Assets')]:
        vals = financials.get(key, [])
        if len(vals) >= 3:
            recent_vals = [v['value'] for v in vals[:4]]
            years = [v['year'] for v in vals[:4]]
            # Check if trending up or down
            if all(recent_vals[i] >= recent_vals[i+1] for i in range(len(recent_vals)-1)):
                trends.append(f"{label} has grown consistently over the last {len(recent_vals)} years")
            elif all(recent_vals[i] <= recent_vals[i+1] for i in range(len(recent_vals)-1)):
                trends.append(f"{label} has declined consistently over the last {len(recent_vals)} years — concerning pattern")
    analysis['trend_analysis'] = trends

    # ─── Overall Health Score ───
    if max_points > 0:
        health_pct = (health_points / max_points) * 100
        analysis['health_score'] = round(health_pct)
        if health_pct >= 75: analysis['health_label'] = 'STRONG'
        elif health_pct >= 55: analysis['health_label'] = 'HEALTHY'
        elif health_pct >= 35: analysis['health_label'] = 'MODERATE'
        else: analysis['health_label'] = 'WEAK'
    else:
        analysis['health_score'] = 0
        analysis['health_label'] = 'UNKNOWN'

    # ─── Summary ───
    summary_parts = []
    summary_parts.append(f"{company} ({ticker})")
    if rev:
        summary_parts.append(f"generated {fmt_billions(rev)} in revenue")
    if ni is not None:
        if ni > 0:
            summary_parts.append(f"with {fmt_billions(ni)} in net income")
        else:
            summary_parts.append(f"but posted a net loss of {fmt_billions(abs(ni))}")

    rev_growth = yoy_change('revenue')
    if rev_growth is not None:
        if rev_growth > 0:
            summary_parts.append(f"(up {rev_growth*100:.1f}% year-over-year)")
        else:
            summary_parts.append(f"(down {abs(rev_growth)*100:.1f}% year-over-year)")

    analysis['summary'] = ' '.join(summary_parts) + '.'

    return analysis


def compute_recommendation(stock_analysis: Dict, risks: List[Dict]) -> Dict:
    """
    Compute a buy/sell recommendation by combining:
    - Financial health score (0-100)
    - Average risk probability from textual analysis
    - Number of high-priority risks
    - Revenue/earnings trends

    Returns a recommendation dict with signal, score, and explanation.
    
    Scale:
        STRONG BUY  — financials strong + risks low/stable
        BUY         — financials healthy + risks moderate
        HOLD        — mixed signals or insufficient data
        SELL        — financials weakening + risks elevated
        STRONG SELL — financials deteriorating + risks very high
    """
    health_score = stock_analysis.get('health_score', 50)
    health_label = stock_analysis.get('health_label', 'UNKNOWN')
    strengths = stock_analysis.get('strengths', [])
    concerns = stock_analysis.get('concerns', [])

    # Calculate risk metrics
    active_risks = [r for r in risks if r.get('status') not in ('UNCHANGED', None)]
    high_risks = [r for r in active_risks if r.get('probability', 0) >= 50]
    moderate_risks = [r for r in active_risks if 20 <= r.get('probability', 0) < 50]

    avg_risk = 0
    if active_risks:
        avg_risk = sum(r.get('probability', 0) for r in active_risks) / len(active_risks)

    max_risk = max((r.get('probability', 0) for r in risks), default=0)
    new_risks = [r for r in risks if r.get('status') == 'NEW']

    # ─── Scoring Matrix ───
    # Financial component (0-50 points, higher = better)
    financial_points = health_score / 2  # Maps 0-100 to 0-50

    # Risk component (0-50 points, higher = WORSE risk = lower recommendation)
    # Invert: low risk = high points for recommendation
    risk_penalty = 0
    risk_penalty += min(avg_risk * 0.4, 25)          # avg risk contributes up to 25
    risk_penalty += len(high_risks) * 5               # each high risk costs 5
    risk_penalty += len(new_risks) * 3                # new risks cost 3 each
    risk_penalty = min(risk_penalty, 50)

    risk_points = 50 - risk_penalty  # Invert: less risk = more points

    # Combined score (0-100, higher = more bullish)
    combined = financial_points + risk_points

    # Growth bonus/penalty
    metrics = stock_analysis.get('metrics', [])
    for m in metrics:
        if m.get('name') == 'Revenue' and m.get('change'):
            try:
                rev_change = float(m['change'].replace('%', '').replace('+', ''))
                if rev_change > 10: combined += 5
                elif rev_change > 5: combined += 3
                elif rev_change < -5: combined -= 5
                elif rev_change < -10: combined -= 8
            except:
                pass
        if m.get('name') == 'Net Income' and m.get('change'):
            try:
                ni_change = float(m['change'].replace('%', '').replace('+', ''))
                if ni_change < -15: combined -= 5
                elif ni_change > 15: combined += 3
            except:
                pass

    # Clamp
    combined = max(0, min(100, combined))

    # ─── Map to Recommendation ───
    if combined >= 75:
        signal = 'STRONG BUY'
        color = '#22c55e'
        emoji = '&#9650;&#9650;'
    elif combined >= 60:
        signal = 'BUY'
        color = '#4ade80'
        emoji = '&#9650;'
    elif combined >= 40:
        signal = 'HOLD'
        color = '#fbbf24'
        emoji = '&#9654;'
    elif combined >= 25:
        signal = 'SELL'
        color = '#fb923c'
        emoji = '&#9660;'
    else:
        signal = 'STRONG SELL'
        color = '#ef4444'
        emoji = '&#9660;&#9660;'

    # ─── Generate Explanation ───
    explanation_parts = []

    # Financial health component
    if health_score >= 70:
        explanation_parts.append(f"Financials are strong (health score: {health_score}/100).")
    elif health_score >= 50:
        explanation_parts.append(f"Financials are generally healthy (health score: {health_score}/100).")
    elif health_score >= 30:
        explanation_parts.append(f"Financial health is moderate with some concerns (score: {health_score}/100).")
    else:
        explanation_parts.append(f"Financial health is weak (score: {health_score}/100) — significant warning signs.")

    # Risk component
    if len(high_risks) == 0 and avg_risk < 15:
        explanation_parts.append("Risk analysis shows minimal changes in filing language — stable outlook.")
    elif len(high_risks) == 0 and avg_risk < 30:
        explanation_parts.append("Some risk language has shifted but no high-priority signals detected.")
    elif len(high_risks) <= 2:
        explanation_parts.append(f"{len(high_risks)} high-priority risk(s) detected in the filing — the company is warning about significant new threats.")
    else:
        explanation_parts.append(f"{len(high_risks)} high-priority risks detected — multiple areas of significant concern in the company's own disclosures.")

    if new_risks:
        explanation_parts.append(f"{len(new_risks)} brand-new risk disclosure(s) appeared that didn't exist last year.")

    # Combined verdict
    if signal == 'STRONG BUY':
        explanation_parts.append("The combination of strong financials and stable/low-risk language is the best possible signal. Historically, these companies outperform.")
    elif signal == 'BUY':
        explanation_parts.append("Overall positive picture. Some changes in risk language to monitor, but financials support continued strength.")
    elif signal == 'HOLD':
        explanation_parts.append("Mixed signals — either financials are softening, risk language is shifting, or both. Monitor closely over the next quarter.")
    elif signal == 'SELL':
        explanation_parts.append("Concerning combination: elevated risk language with weakening fundamentals. The 'Lazy Prices' research shows this pattern precedes negative events.")
    else:
        explanation_parts.append("Severe warning: major new risk disclosures combined with deteriorating financial metrics. This is the highest-danger combination identified by academic research.")

    return {
        'signal': signal,
        'score': round(combined),
        'color': color,
        'emoji': emoji,
        'explanation': ' '.join(explanation_parts),
        'components': {
            'financial_points': round(financial_points, 1),
            'risk_points': round(risk_points, 1),
            'high_risk_count': len(high_risks),
            'new_risk_count': len(new_risks),
            'avg_risk_probability': round(avg_risk, 1),
        }
    }



# =========================================================
# Risk Explanation Generator
# =========================================================

def generate_risk_explanation(risk_score, classification) -> str:
    """Generate a plain-English explanation of why a risk scored the way it did."""
    prob = risk_score.preliminary_probability
    status = risk_score.status
    title = risk_score.title

    explanation_parts = []

    # Status-based opening
    if status == RiskChangeStatus.NEW:
        explanation_parts.append(
            f"This is a <strong>brand new risk</strong> that did not exist in last year's filing. "
            f"When a company adds an entirely new risk disclosure, it means something has changed in their business "
            f"environment that their lawyers felt MUST be reported to investors. "
            f"According to the 'Lazy Prices' research, new disclosures are among the strongest warning signals."
        )
    elif status == RiskChangeStatus.MODIFIED:
        sim = risk_score.textual_detail.body_similarity if risk_score.textual_detail else 0
        if sim < 0.4:
            explanation_parts.append(
                f"This risk was <strong>dramatically rewritten</strong> from last year (only {sim*100:.0f}% similar to prior version). "
                f"A major rewrite means the company's exposure to this risk has fundamentally changed — they couldn't just copy-paste "
                f"last year's language because the situation is significantly different now."
            )
        else:
            explanation_parts.append(
                f"This risk was <strong>modified</strong> from last year ({sim*100:.0f}% similar to prior version). "
                f"The company kept the same general risk topic but updated the language — "
                f"often adding stronger warnings, new specifics, or escalating the severity of the threat."
            )

        # Sentence-level detail
        if risk_score.textual_detail:
            added = risk_score.textual_detail.added_sentence_count
            rewritten = risk_score.textual_detail.rewritten_sentence_count
            if added > 0 or rewritten > 0:
                parts = []
                if added > 0:
                    parts.append(f"{added} entirely new sentence{'s' if added > 1 else ''}")
                if rewritten > 0:
                    parts.append(f"{rewritten} rewritten sentence{'s' if rewritten > 1 else ''}")
                explanation_parts.append(
                    f"Specifically: {' and '.join(parts)} were detected in this section."
                )

    elif status == RiskChangeStatus.UNCHANGED:
        explanation_parts.append(
            f"This risk is <strong>unchanged boilerplate</strong> — the exact same language as last year. "
            f"No signal here. Companies copy-paste risks that haven't evolved, which actually means things are stable."
        )
        return ' '.join(explanation_parts)

    elif status == RiskChangeStatus.REMOVED:
        explanation_parts.append(
            f"This risk was <strong>removed</strong> — it existed last year but is gone now. "
            f"This could mean the risk resolved (good) or the company is trying to minimize attention to it (concerning)."
        )
        return ' '.join(explanation_parts)

    # Sentiment-based explanation
    if risk_score.sentiment_detail and risk_score.sentiment_detail.changed_sentiment:
        sent = risk_score.sentiment_detail.changed_sentiment
        if sent.is_heavily_negative:
            explanation_parts.append(
                f"The language used is <strong>heavily negative</strong> — words like 'adverse,' 'impair,' 'unable,' "
                f"'material loss' appear frequently ({sent.negative_density*100:.1f}% of words are negative). "
                f"This is significantly above normal corporate disclosure levels."
            )
        if sent.is_highly_uncertain:
            explanation_parts.append(
                f"There is <strong>high uncertainty language</strong> — words like 'may,' 'could,' 'uncertain,' "
                f"'unpredictable' ({sent.uncertainty_density*100:.1f}% density). "
                f"The company is hedging heavily about what might happen."
            )
        if sent.is_constrained:
            explanation_parts.append(
                f"The text contains <strong>constraining language</strong> — words indicating the company feels "
                f"limited, obligated, or restricted in how it can respond to this risk."
            )

    # Tone delta for MODIFIED
    if (status == RiskChangeStatus.MODIFIED and
            risk_score.sentiment_detail and
            risk_score.sentiment_detail.sentiment_delta):
        delta = risk_score.sentiment_detail.sentiment_delta
        if delta.tone_worsened:
            explanation_parts.append(
                f"<strong>The tone has worsened</strong> compared to last year — the language shifted from relatively "
                f"neutral to more alarming. This year-over-year darkening of tone is a key signal from the research."
            )

    # Probability-based conclusion
    if prob >= 70:
        explanation_parts.append(
            f"<strong>Bottom line:</strong> At {prob:.0f}%, this is a very high-priority signal. "
            f"Based on academic research, risks scoring this high have historically preceded "
            f"significant negative events (stock drops, earnings misses, or operational crises) within 12 months."
        )
    elif prob >= 50:
        explanation_parts.append(
            f"<strong>Bottom line:</strong> At {prob:.0f}%, this deserves close attention. "
            f"The combination of significant textual changes and negative language suggests "
            f"this risk is evolving in a concerning direction."
        )
    elif prob >= 25:
        explanation_parts.append(
            f"<strong>Bottom line:</strong> At {prob:.0f}%, this is a moderate signal worth monitoring. "
            f"Some changes detected but not yet at crisis-level language."
        )

    return ' '.join(explanation_parts)



# =========================================================
# Tone Scoring, Highlighting, and Serialization (Option A)
# =========================================================

import html as _html_escape
from lm_dictionary import NEGATIVE_WORDS, UNCERTAINTY_WORDS

# Tone labels, mild -> severe (never a positive scale; risk factors are about
# how bad things could get, so we lead with negative + uncertainty only).
TONE_LABELS = ["mild", "moderate", "serious", "severe"]


def compute_tone(sentiment_detail) -> Dict:
    """
    Turn Loughran-McDonald negative + uncertainty densities into a 0-100 tone
    score and a mild/moderate/serious/severe label with a plain-English caption.
    Uses only the changed/new wording (sentiment_detail.changed_sentiment).
    """
    default = {'score': 0, 'label': 'mild', 'caption': 'The wording is measured, in line with typical filing language.'}
    if not sentiment_detail or not getattr(sentiment_detail, 'changed_sentiment', None):
        return default

    s = sentiment_detail.changed_sentiment
    neg = getattr(s, 'negative_density', 0.0)
    unc = getattr(s, 'uncertainty_density', 0.0)

    # Map densities to 0-100. Negative weighs more than uncertainty.
    # ~0.15 negative density is very high in a 10-K, so normalize against that.
    neg_component = min(neg / 0.15, 1.0)
    unc_component = min(unc / 0.12, 1.0)
    raw = 0.65 * neg_component + 0.35 * unc_component
    score = int(round(min(max(raw, 0.0), 1.0) * 100))

    # Year-over-year worsening nudges the tone up a band.
    delta = getattr(sentiment_detail, 'sentiment_delta', None)
    worsened = bool(delta and getattr(delta, 'tone_worsened', False))
    if worsened:
        score = min(score + 10, 100)

    if score >= 75:
        label = 'severe'
    elif score >= 50:
        label = 'serious'
    elif score >= 25:
        label = 'moderate'
    else:
        label = 'mild'

    caption_map = {
        'severe': 'This section reads as strongly negative and uncertain, well above a typical filing. It comes across as a serious warning.',
        'serious': 'This section leans notably negative and uncertain, more cautious than typical filing language.',
        'moderate': 'This section carries some negative and hedging language, a step up from routine wording.',
        'mild': 'The wording is measured, close to routine filing language.',
    }
    caption = caption_map[label]
    if worsened:
        caption += ' The tone also darkened compared with last year.'

    return {'score': score, 'label': label, 'caption': caption}


_word_re = re.compile(r"[A-Za-z][A-Za-z'\-]*")


def highlight_wording(sentences: List[str], limit: int = 6) -> List[str]:
    """
    Return up to `limit` sentences as HTML with negative words wrapped in a
    red span and uncertainty words in an amber span. Input text is escaped
    first so filing content cannot inject markup.
    """
    out = []
    for sent in sentences[:limit]:
        def repl(m):
            w = m.group(0)
            lw = w.lower()
            if lw in NEGATIVE_WORDS:
                return f'<span class="w-neg">{w}</span>'
            if lw in UNCERTAINTY_WORDS:
                return f'<span class="w-unc">{w}</span>'
            return w
        escaped = _html_escape.escape(sent)
        # Re-run the matcher over the escaped text (word chars are unaffected by escaping).
        highlighted = _word_re.sub(repl, escaped)
        out.append(highlighted)
    return out


def serialize_risk(score_obj, classification) -> Dict:
    """
    Build the Option A per-risk dict for the frontend: title, status, score,
    tone (bar + caption), a plain-English note, and the highlighted changed
    wording for the expandable 'see the wording' detail.
    """
    status = score_obj.status.value
    tone = compute_tone(getattr(score_obj, 'sentiment_detail', None))

    # Collect the wording to highlight: changed sentences (MODIFIED) or key
    # sentences (NEW). These come from the classifier.
    sentences: List[str] = []
    if classification is not None:
        if getattr(classification, 'changed_sentences', None):
            sentences = [c.sentence for c in classification.changed_sentences]
        elif getattr(classification, 'key_sentences', None):
            sentences = list(classification.key_sentences)

    # Friendly status label for the chip.
    status_label = {
        'NEW': 'NEW', 'MODIFIED': 'REWRITTEN',
        'UNCHANGED': 'UNCHANGED', 'REMOVED': 'REMOVED',
    }.get(status, status)

    return {
        'title': score_obj.title[:140],
        'status': status,
        'status_label': status_label,
        'score': int(round(score_obj.preliminary_probability)),
        'level': score_obj.risk_level_label,
        'tone_score': tone['score'],
        'tone_label': tone['label'],
        'tone_caption': tone['caption'],
        'note': generate_risk_explanation(score_obj, classification) if classification else '',
        'wording': highlight_wording(sentences),
    }


def build_headline(scoring, current_year: int, prior_year: int) -> Dict:
    """
    Build the top risk-signal panel: an overall mild/moderate/serious/severe
    band and a one-line plain-English summary of what changed this year.
    """
    risks = scoring.risk_scores
    new_count = sum(1 for r in risks if r.status == RiskChangeStatus.NEW)
    modified_count = sum(1 for r in risks if r.status == RiskChangeStatus.MODIFIED)
    unchanged_count = sum(1 for r in risks if r.status == RiskChangeStatus.UNCHANGED)

    # Overall band from the average tone of changed risks, plus new-risk pressure.
    changed = [r for r in risks if r.status in (RiskChangeStatus.NEW, RiskChangeStatus.MODIFIED)]
    tones = [compute_tone(getattr(r, 'sentiment_detail', None))['score'] for r in changed]
    avg_tone = int(round(sum(tones) / len(tones))) if tones else 0
    overall = min(avg_tone + (8 if new_count >= 2 else 0), 100)

    if overall >= 75:
        band = 'severe'
    elif overall >= 50:
        band = 'serious'
    elif overall >= 25:
        band = 'moderate'
    else:
        band = 'mild'

    def plural(n, s):
        return f"{n} {s}" + ("" if n == 1 else "s")

    parts = []
    if new_count:
        parts.append(plural(new_count, "new risk") + " appeared")
    if modified_count:
        parts.append(plural(modified_count, "risk") + " rewritten with changed wording")
    change_phrase = ", and ".join(parts) if parts else "little changed in the risk language"

    band_phrase = {
        'severe': "grew much more cautious",
        'serious': "grew noticeably more cautious",
        'moderate': "shifted somewhat",
        'mild': "stayed largely steady",
    }[band]

    summary = (
        f"The risk language {band_phrase} this year. "
        f"{change_phrase[0].upper() + change_phrase[1:]}. "
        f"{unchanged_count} of the risk sections are unchanged from last year."
    )

    return {'band': band, 'score': overall, 'summary': summary,
            'new_count': new_count, 'modified_count': modified_count,
            'unchanged_count': unchanged_count}



# =========================================================
# HTML Templates
# =========================================================

HOME_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>S&amp;P 500 Risk Radar</title>
<style>
    :root {
        --bg: #ffffff;
        --bg-alt: #f7f9fc;
        --navy: #0b1b34;
        --ink: #0f172a;
        --slate: #64748b;
        --line: #e5e9f0;
        --accent: #2563eb;
        --accent-soft: #eff4ff;
        --red: #dc2626;
        --amber: #b45309;
        --shadow: 0 1px 3px rgba(15,23,42,0.06), 0 8px 24px rgba(15,23,42,0.05);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--bg); color: var(--ink); -webkit-font-smoothing: antialiased; line-height: 1.5; }
    a { color: inherit; }
    .wrap { max-width: 1120px; margin: 0 auto; padding: 0 32px; }

    /* ---------- Nav ---------- */
    .nav { position: sticky; top: 0; z-index: 50; background: rgba(255,255,255,0.85); backdrop-filter: saturate(180%) blur(12px); border-bottom: 1px solid var(--line); }
    .nav-inner { max-width: 1120px; margin: 0 auto; padding: 14px 32px; display: flex; align-items: center; gap: 44px; }
    .brand { display: flex; align-items: center; gap: 10px; text-decoration: none; }
    .brand .logo { width: 28px; height: 28px; }
    .brand h1 { font-size: 17px; color: var(--ink); font-weight: 700; letter-spacing: -0.2px; white-space: nowrap; }
    .brand h1 span { color: var(--accent); }
    .menu { display: flex; align-items: center; gap: 30px; }
    .menu a { color: var(--slate); text-decoration: none; font-size: 14.5px; font-weight: 500; transition: color 0.15s; }
    .menu a:hover { color: var(--ink); }
    .menu a .ext { font-size: 11px; opacity: 0.6; }
    .nav-actions { margin-left: auto; }
    .signin-btn { position: relative; padding: 8px 18px; border: 1px solid var(--line); border-radius: 8px; background: #f1f5f9; color: #94a3b8; font-size: 14px; font-weight: 600; cursor: not-allowed; font-family: inherit; }
    .signin-btn::after { content: "Coming soon"; position: absolute; top: 128%; right: 0; background: var(--navy); color: #e2e8f0; font-size: 11px; font-weight: 500; padding: 6px 10px; border-radius: 6px; white-space: nowrap; opacity: 0; pointer-events: none; transition: opacity 0.15s; }
    .signin-btn:hover::after { opacity: 1; }

    /* ---------- Hero ---------- */
    .hero { text-align: center; padding: 84px 32px 48px; max-width: 860px; margin: 0 auto; }
    .eyebrow { display: inline-block; font-size: 13px; font-weight: 600; letter-spacing: 0.4px; color: var(--accent); background: var(--accent-soft); padding: 6px 14px; border-radius: 999px; margin-bottom: 24px; }
    .hero h2 { font-size: 52px; font-weight: 800; line-height: 1.08; letter-spacing: -1.2px; color: var(--ink); margin-bottom: 22px; }
    .hero h2 span { color: var(--accent); }
    .hero p { font-size: 19px; color: var(--slate); line-height: 1.65; max-width: 680px; margin: 0 auto 32px; }
    .hero p b { color: var(--ink); font-weight: 600; }
    .cta { display: inline-block; padding: 14px 34px; background: var(--accent); color: #fff; border-radius: 10px; text-decoration: none; font-weight: 600; font-size: 16px; transition: all 0.15s; box-shadow: 0 6px 18px rgba(37,99,235,0.25); }
    .cta:hover { transform: translateY(-1px); background: #1d4fd7; }
    .cta.ghost { background: transparent; color: var(--accent); box-shadow: none; border: 1px solid var(--line); margin-left: 10px; }
    .cta.ghost:hover { background: var(--accent-soft); }

    /* ---------- Section scaffolding ---------- */
    section { padding: 72px 0; }
    .section-alt { background: var(--bg-alt); border-top: 1px solid var(--line); border-bottom: 1px solid var(--line); }
    .section-head { text-align: center; max-width: 640px; margin: 0 auto 48px; }
    .section-head .kicker { font-size: 13px; font-weight: 700; letter-spacing: 0.6px; text-transform: uppercase; color: var(--accent); margin-bottom: 12px; }
    .section-head h3 { font-size: 34px; font-weight: 800; letter-spacing: -0.6px; color: var(--ink); margin-bottom: 14px; }
    .section-head p { font-size: 17px; color: var(--slate); line-height: 1.6; }

    /* ---------- Carousel ---------- */
    .carousel { max-width: 1000px; margin: 0 auto; }
    .slide { display: none; grid-template-columns: 1fr 340px; gap: 0; background: var(--bg); border: 1px solid var(--line); border-radius: 16px; overflow: hidden; box-shadow: var(--shadow); }
    .slide.active { display: grid; }
    .slide-chart { padding: 26px 22px 18px; border-right: 1px solid var(--line); }
    .slide-chart .co { display: flex; align-items: baseline; gap: 10px; margin-bottom: 4px; }
    .slide-chart .co .tk { font-size: 18px; font-weight: 800; color: var(--ink); }
    .slide-chart .co .nm { font-size: 14px; color: var(--slate); }
    .slide-chart .period { font-size: 12.5px; color: var(--slate); margin-bottom: 12px; }
    .slide-chart svg { width: 100%; height: auto; display: block; }
    .slide-info { padding: 28px 26px; background: var(--bg); }
    .info-block { margin-bottom: 22px; }
    .info-block:last-child { margin-bottom: 0; }
    .info-block .lbl { font-size: 11.5px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 6px; }
    .info-block.risk .lbl { color: var(--amber); }
    .info-block.event .lbl { color: var(--accent); }
    .info-block.stock .lbl { color: var(--red); }
    .info-block p { font-size: 14px; color: #334155; line-height: 1.55; }
    .info-block p b { color: var(--ink); }

    .carousel-controls { display: flex; align-items: center; justify-content: center; gap: 20px; margin-top: 24px; }
    .dots { display: flex; gap: 10px; }
    .dot { width: 30px; height: 30px; border-radius: 50%; border: 1px solid var(--line); background: #fff; color: var(--slate); font-size: 13px; font-weight: 600; cursor: pointer; transition: all 0.15s; }
    .dot:hover { border-color: var(--accent); color: var(--accent); }
    .dot.active { background: var(--accent); border-color: var(--accent); color: #fff; }
    .arrow-btn { width: 38px; height: 38px; border-radius: 50%; border: 1px solid var(--line); background: #fff; color: var(--ink); font-size: 16px; cursor: pointer; transition: all 0.15s; }
    .arrow-btn:hover { border-color: var(--accent); color: var(--accent); }

    /* ---------- Two lenses ---------- */
    .lens-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 24px; max-width: 940px; margin: 0 auto; }
    .lens { background: var(--bg); border: 1px solid var(--line); border-radius: 14px; padding: 30px; box-shadow: var(--shadow); }
    .lens .tag { font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; margin-bottom: 10px; }
    .lens.numbers .tag { color: var(--amber); }
    .lens.words .tag { color: var(--accent); }
    .lens h4 { font-size: 20px; color: var(--ink); margin-bottom: 8px; }
    .lens .desc { color: var(--slate); font-size: 14.5px; line-height: 1.6; margin-bottom: 16px; }
    .lens ul { list-style: none; }
    .lens li { color: #334155; font-size: 14px; padding: 6px 0 6px 22px; position: relative; }
    .lens li::before { content: ""; position: absolute; left: 2px; top: 13px; width: 6px; height: 6px; border-radius: 50%; background: var(--accent); }

    /* ---------- How it works ---------- */
    .steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; max-width: 940px; margin: 0 auto; }
    .step { text-align: center; padding: 12px; }
    .step .num { width: 46px; height: 46px; background: var(--accent-soft); color: var(--accent); border-radius: 12px; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px; font-weight: 800; font-size: 18px; }
    .step h4 { color: var(--ink); margin-bottom: 8px; font-size: 17px; }
    .step p { color: var(--slate); font-size: 14.5px; line-height: 1.6; }

    /* ---------- Learning ---------- */
    .learn-grid { display: grid; grid-template-columns: repeat(3, 1fr); gap: 22px; max-width: 1000px; margin: 0 auto; }
    .learn-card { background: var(--bg); border: 1px solid var(--line); border-radius: 14px; overflow: hidden; box-shadow: var(--shadow); transition: transform 0.15s, box-shadow 0.15s; text-decoration: none; display: block; width: 100%; text-align: left; font-family: inherit; cursor: pointer; padding: 0; }
    .learn-card:hover { transform: translateY(-3px); box-shadow: 0 12px 30px rgba(15,23,42,0.1); }
    .learn-card .thumb { height: 8px; }
    .learn-card .thumb.t1 { background: linear-gradient(90deg, #f59e0b, #dc2626); }
    .learn-card .thumb.t2 { background: linear-gradient(90deg, #3b82f6, #dc2626); }
    .learn-card .thumb.t3 { background: linear-gradient(90deg, #6366f1, #dc2626); }
    .learn-card .body { padding: 22px; }
    .learn-card .meta { font-size: 12px; color: var(--slate); margin-bottom: 10px; }
    .learn-card h4 { font-size: 17px; color: var(--ink); line-height: 1.35; margin-bottom: 10px; }
    .learn-card p { font-size: 14px; color: var(--slate); line-height: 1.55; }
    .learn-more { text-align: center; margin-top: 36px; }
    .learn-more a { color: var(--accent); font-weight: 600; text-decoration: none; font-size: 15px; }
    .learn-more a:hover { text-decoration: underline; }

    /* ---------- About ---------- */
    .about { display: grid; grid-template-columns: 132px 1fr; gap: 32px; align-items: start; max-width: 820px; margin: 0 auto; }
    .about .avatar { width: 132px; height: 132px; border-radius: 20px; box-shadow: var(--shadow); }
    .about h3 { font-size: 28px; color: var(--ink); margin-bottom: 6px; letter-spacing: -0.4px; }
    .about .role { font-size: 14px; color: var(--accent); font-weight: 600; margin-bottom: 16px; }
    .about p { font-size: 15.5px; color: #334155; line-height: 1.7; margin-bottom: 14px; }
    .about .why { border-left: 3px solid var(--accent); padding-left: 16px; color: #1e293b; font-style: italic; }
    .about .sub-link { display: inline-block; margin-top: 8px; color: var(--accent); font-weight: 600; text-decoration: none; font-size: 15px; }
    .about .sub-link:hover { text-decoration: underline; }

    /* ---------- Research strip ---------- */
    .research { background: var(--navy); color: #fff; }
    .research .wrap { display: grid; grid-template-columns: 220px 1fr; gap: 40px; align-items: center; }
    .research .big { font-size: 60px; font-weight: 800; color: #60a5fa; letter-spacing: -2px; line-height: 1; }
    .research .big span { display: block; font-size: 14px; color: #94a3b8; font-weight: 500; margin-top: 8px; letter-spacing: 0; }
    .research h3 { font-size: 24px; margin-bottom: 12px; }
    .research p { color: #cbd5e1; font-size: 15px; line-height: 1.65; }

    /* ---------- Case study detail (inline expandable) ---------- */
    .case-detail { display: none; max-width: 820px; margin: 32px auto 0; background: var(--bg); border: 1px solid var(--line); border-radius: 16px; box-shadow: var(--shadow); overflow: hidden; }
    .case-detail.open { display: block; }
    .case-detail .cd-head { padding: 24px 28px; border-bottom: 1px solid var(--line); display: flex; align-items: center; justify-content: space-between; gap: 16px; }
    .case-detail .cd-head .cd-title { font-size: 20px; font-weight: 800; color: var(--ink); }
    .case-detail .cd-head .cd-tag { font-size: 12px; color: var(--slate); margin-top: 4px; }
    .case-detail .cd-close { border: 1px solid var(--line); background: #fff; color: var(--slate); border-radius: 8px; padding: 6px 12px; font-size: 13px; font-weight: 600; cursor: pointer; font-family: inherit; white-space: nowrap; }
    .case-detail .cd-close:hover { border-color: var(--accent); color: var(--accent); }
    .case-detail .cd-body { padding: 26px 28px; }
    .case-detail .cd-body h5 { font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: var(--accent); margin: 18px 0 6px; }
    .case-detail .cd-body h5:first-child { margin-top: 0; }
    .case-detail .cd-body p { font-size: 15px; color: #334155; line-height: 1.7; }
    .case-detail .cd-note { margin-top: 22px; padding: 12px 16px; background: var(--accent-soft); border-radius: 10px; font-size: 13px; color: #1e3a8a; }

    /* ---------- Academic research ---------- */
    .intro-note { max-width: 720px; margin: 0 auto 40px; text-align: center; color: #334155; font-size: 15.5px; line-height: 1.7; }
    .papers { max-width: 1000px; margin: 0 auto; display: grid; gap: 16px; }
    .paper { background: var(--bg); border: 1px solid var(--line); border-radius: 12px; box-shadow: var(--shadow); overflow: hidden; }
    .paper .p-head { padding: 20px 24px; cursor: pointer; display: flex; align-items: flex-start; gap: 16px; }
    .paper .p-head:hover { background: var(--bg-alt); }
    .paper .p-badge { flex-shrink: 0; width: 108px; text-align: center; font-size: 11px; font-weight: 700; letter-spacing: 0.2px; padding: 6px 8px; border-radius: 8px; margin-top: 1px; background: var(--accent-soft); color: var(--accent); line-height: 1.3; }
    .paper .p-main { flex: 1; }
    .paper .p-title { font-size: 16px; font-weight: 700; color: var(--ink); line-height: 1.35; }
    .paper .p-meta { font-size: 13px; color: var(--slate); margin-top: 4px; }
    .paper .p-toggle { flex-shrink: 0; color: var(--slate); font-size: 20px; line-height: 1; transition: transform 0.2s; margin-top: 2px; }
    .paper.open .p-toggle { transform: rotate(45deg); }
    .paper .p-detail { display: none; padding: 0 24px 22px 24px; }
    .paper.open .p-detail { display: block; }
    .paper .p-detail p { font-size: 14.5px; color: #334155; line-height: 1.65; }
    .paper .p-detail a { color: var(--accent); font-weight: 600; text-decoration: none; font-size: 14px; display: inline-block; margin-top: 10px; }
    .paper .p-detail a:hover { text-decoration: underline; }

    .footer { padding: 40px 0; text-align: center; color: var(--slate); font-size: 12.5px; line-height: 1.7; border-top: 1px solid var(--line); }

    @media (max-width: 860px) {
        .nav-inner { gap: 20px; }
        .menu { display: none; }
        .hero h2 { font-size: 34px; }
        .slide.active { grid-template-columns: 1fr; }
        .slide-chart { border-right: none; border-bottom: 1px solid var(--line); }
        .lens-grid, .steps, .learn-grid, .research .wrap, .about { grid-template-columns: 1fr; }
        .about .avatar { margin: 0 auto; }
    }
</style>
</head>
<body>

<!-- ============ NAV ============ -->
<nav class="nav">
    <div class="nav-inner">
        <a class="brand" href="#top">
            <svg class="logo" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="14" stroke="#2563eb" stroke-width="2" opacity="0.35"/>
                <circle cx="16" cy="16" r="8.5" stroke="#2563eb" stroke-width="2" opacity="0.6"/>
                <circle cx="16" cy="16" r="2.6" fill="#dc2626"/>
                <line x1="16" y1="16" x2="27" y2="6.5" stroke="#2563eb" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <h1>S&amp;P 500 <span>Risk Radar</span></h1>
        </a>
        <div class="menu">
            <a href="#top">Home</a>
            <a href="/analyze">Analyze</a>
            <a href="#learning">Learning</a>
            <a href="#research">Research</a>
            <a href="https://akilanvadivelan.substack.com" target="_blank" rel="noopener">Substack <span class="ext">&#8599;</span></a>
            <a href="#about">About</a>
        </div>
        <div class="nav-actions">
            <button class="signin-btn" disabled title="Coming soon">Sign In</button>
        </div>
    </div>
</nav>

<!-- ============ HERO ============ -->
<div id="top" class="hero">
    <span class="eyebrow">For everyday investors, powered by research</span>
    <h2>We read the warning signs that companies have<br><span>buried in their filings</span></h2>
    <p>
        Every S&amp;P 500 company reveals what is going wrong, in the <b>numbers behind its business</b>
        (profits, debt, inventory, cash flow) and in the <b>risk warnings inside its own reports</b>.
        S&amp;P 500 Risk Radar examines both closely, tracks what has changed since last year, and
        surfaces the signals before the stock price falls.
    </p>
    <a href="/analyze" class="cta">Start Analysis</a>
    <a href="#examples" class="cta ghost">See real examples</a>
</div>

<!-- ============ CAROUSEL ============ -->
<section id="examples" class="section-alt">
    <div class="section-head">
        <div class="kicker">Real Cases</div>
        <h3>The warning came first. Then the price.</h3>
        <p>Three real S&amp;P 500 companies where the risk was disclosed months before the stock reacted.</p>
    </div>

    <div class="carousel">
        <!-- ===== SLIDE 1: TARGET ===== -->
        <div class="slide active" data-slide="0">
            <div class="slide-chart">
                <div class="co"><span class="tk">TGT</span><span class="nm">Target Corporation</span></div>
                <div class="period">Monthly close, May 2021 to June 2022</div>
                <svg viewBox="0 0 860 360" role="img" aria-label="Target monthly stock price">
                    <g stroke="#e5e9f0" stroke-width="1">
                        <line x1="70" y1="300" x2="820" y2="300"/>
                        <line x1="70" y1="210" x2="820" y2="210"/>
                        <line x1="70" y1="120" x2="820" y2="120"/>
                        <line x1="70" y1="30" x2="820" y2="30"/>
                    </g>
                    <g fill="#94a3b8" font-size="11" text-anchor="end">
                        <text x="60" y="304">$150</text><text x="60" y="214">$190</text>
                        <text x="60" y="124">$230</text><text x="60" y="34">$270</text>
                    </g>
                    <!-- window between risk-disclosed and fall -->
                    <rect x="470" y="30" width="300" height="270" fill="#f59e0b" opacity="0.06"/>
                    <polyline fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"
                        points="70,168.4 120,126.1 170,89.7 220,50.1 270,83.2 320,123.6 370,57.2 420,96.6 470,116.8 520,144.2 570,143.8 620,163.8 670,123.0 720,153.1 770,273.9 820,285.3"/>
                    <!-- risk exposed arrow (~Jan 2022, x=470) -->
                    <g>
                        <line x1="470" y1="60" x2="470" y2="108" stroke="#b45309" stroke-width="1.5" marker-end="url(#amberArrow)"/>
                        <text x="470" y="52" fill="#b45309" font-size="11.5" text-anchor="middle" font-weight="600">Risk exposed</text>
                    </g>
                    <!-- stock fell arrow (May 18, x=770) -->
                    <g>
                        <line x1="770" y1="228" x2="770" y2="262" stroke="#dc2626" stroke-width="1.5" marker-end="url(#redArrow)"/>
                        <circle cx="770" cy="273.9" r="4.5" fill="#dc2626"/>
                        <text x="770" y="220" fill="#dc2626" font-size="11.5" text-anchor="middle" font-weight="700">Stock fell 24.9%</text>
                    </g>
                    <!-- gap label -->
                    <text x="620" y="322" fill="#b45309" font-size="11" text-anchor="middle">about 4 months between warning and fall</text>
                    <g fill="#94a3b8" font-size="10.5" text-anchor="middle">
                        <text x="70" y="318">May '21</text><text x="420" y="318">Dec '21</text><text x="820" y="318">Jun '22</text>
                    </g>
                    <defs>
                        <marker id="amberArrow" markerWidth="8" markerHeight="8" refX="4" refY="7" orient="auto"><path d="M0,0 L4,7 L8,0" fill="#b45309"/></marker>
                        <marker id="redArrow" markerWidth="8" markerHeight="8" refX="4" refY="7" orient="auto"><path d="M0,0 L4,7 L8,0" fill="#dc2626"/></marker>
                    </defs>
                </svg>
            </div>
            <div class="slide-info">
                <div class="info-block risk">
                    <div class="lbl">The risk in the report</div>
                    <p>New language about <b>excess inventory</b>, supply chain disruption, and rising costs appeared, replacing calmer boilerplate from the year before.</p>
                </div>
                <div class="info-block event">
                    <div class="lbl">What happened at earnings</div>
                    <p>On May 18, 2022, Target cut its operating margin guidance from over 8 percent to about 6 percent, citing the very cost and inventory pressures it had flagged.</p>
                </div>
                <div class="info-block stock">
                    <div class="lbl">What happened to the stock</div>
                    <p>Shares fell <b>24.9 percent in a single day</b> to $161.61, the worst session since 1987, erasing roughly $25 billion in value.</p>
                </div>
            </div>
        </div>

        <!-- ===== SLIDE 2: SOUTHWEST ===== -->
        <div class="slide" data-slide="1">
            <div class="slide-chart">
                <div class="co"><span class="tk">LUV</span><span class="nm">Southwest Airlines</span></div>
                <div class="period">Monthly close, January 2022 to January 2023</div>
                <svg viewBox="0 0 860 360" role="img" aria-label="Southwest monthly stock price">
                    <g stroke="#e5e9f0" stroke-width="1">
                        <line x1="70" y1="300" x2="820" y2="300"/>
                        <line x1="70" y1="210" x2="820" y2="210"/>
                        <line x1="70" y1="120" x2="820" y2="120"/>
                        <line x1="70" y1="30" x2="820" y2="30"/>
                    </g>
                    <g fill="#94a3b8" font-size="11" text-anchor="end">
                        <text x="60" y="304">$28</text><text x="60" y="214">$35</text>
                        <text x="60" y="124">$43</text><text x="60" y="34">$50</text>
                    </g>
                    <rect x="445" y="30" width="313" height="270" fill="#f59e0b" opacity="0.06"/>
                    <polyline fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"
                        points="70,103.6 132.5,89.2 195,128.8 257.5,80.2 320,72.6 382.5,101.8 445,192.9 507.5,173.5 570,194.1 632.5,260.5 695,190.3 757.5,151.7 820,243.5"/>
                    <!-- risk exposed (~Jul 2022, x=445) -->
                    <g>
                        <line x1="445" y1="150" x2="445" y2="184" stroke="#b45309" stroke-width="1.5" marker-end="url(#amberArrow2)"/>
                        <text x="445" y="142" fill="#b45309" font-size="11.5" text-anchor="middle" font-weight="600">Risk exposed</text>
                    </g>
                    <!-- stock fell (late Dec, x=757.5 -> Jan low 820) -->
                    <g>
                        <line x1="820" y1="200" x2="820" y2="232" stroke="#dc2626" stroke-width="1.5" marker-end="url(#redArrow2)"/>
                        <circle cx="820" cy="243.5" r="4.5" fill="#dc2626"/>
                        <text x="800" y="192" fill="#dc2626" font-size="11.5" text-anchor="end" font-weight="700">Meltdown, fell 15.6%</text>
                    </g>
                    <text x="632" y="322" fill="#b45309" font-size="11" text-anchor="middle">about 5 months between warning and fall</text>
                    <g fill="#94a3b8" font-size="10.5" text-anchor="middle">
                        <text x="70" y="318">Jan '22</text><text x="445" y="318">Jul '22</text><text x="820" y="318">Jan '23</text>
                    </g>
                    <defs>
                        <marker id="amberArrow2" markerWidth="8" markerHeight="8" refX="4" refY="7" orient="auto"><path d="M0,0 L4,7 L8,0" fill="#b45309"/></marker>
                        <marker id="redArrow2" markerWidth="8" markerHeight="8" refX="4" refY="7" orient="auto"><path d="M0,0 L4,7 L8,0" fill="#dc2626"/></marker>
                    </defs>
                </svg>
            </div>
            <div class="slide-info">
                <div class="info-block risk">
                    <div class="lbl">The risk in the report</div>
                    <p>Filings repeatedly flagged reliance on <b>aging technology and crew scheduling systems</b>, an operational risk that had been disclosed for years.</p>
                </div>
                <div class="info-block event">
                    <div class="lbl">What happened at the event</div>
                    <p>A December 2022 winter storm overwhelmed those systems, forcing thousands of cancellations and stranding travelers over the holidays.</p>
                </div>
                <div class="info-block stock">
                    <div class="lbl">What happened to the stock</div>
                    <p>Shares fell about <b>15.6 percent in December 2022</b> as the operational meltdown played out in public.</p>
                </div>
            </div>
        </div>

        <!-- ===== SLIDE 3: SVB ===== -->
        <div class="slide" data-slide="2">
            <div class="slide-chart">
                <div class="co"><span class="tk">SIVB</span><span class="nm">SVB Financial (Silicon Valley Bank)</span></div>
                <div class="period">Monthly close, March 2022 to March 2023</div>
                <svg viewBox="0 0 860 360" role="img" aria-label="SVB monthly stock price">
                    <g stroke="#e5e9f0" stroke-width="1">
                        <line x1="70" y1="300" x2="820" y2="300"/>
                        <line x1="70" y1="210" x2="820" y2="210"/>
                        <line x1="70" y1="120" x2="820" y2="120"/>
                        <line x1="70" y1="30" x2="820" y2="30"/>
                    </g>
                    <g fill="#94a3b8" font-size="11" text-anchor="end">
                        <text x="60" y="304">$80</text><text x="60" y="214">$253</text>
                        <text x="60" y="124">$427</text><text x="60" y="34">$600</text>
                    </g>
                    <rect x="70" y="30" width="750" height="270" fill="#f59e0b" opacity="0.05"/>
                    <polyline fill="none" stroke="#2563eb" stroke-width="2.5" stroke-linejoin="round" stroke-linecap="round"
                        points="70,51.1 123.6,88.3 177.1,87.9 230.7,136.4 284.3,132.0 337.9,130.5 391.4,167.2 445,221.6 498.6,222.6 552.1,222.0 605.7,224.6 659.3,191.9 712.9,194.6 766.4,202.5 820,286.5"/>
                    <!-- risk exposed early (interest rate risk building, x=70) -->
                    <g>
                        <line x1="120" y1="90" x2="120" y2="62" stroke="#b45309" stroke-width="1.5" marker-end="url(#amberArrow3)"/>
                        <text x="120" y="112" fill="#b45309" font-size="11.5" text-anchor="middle" font-weight="600">Risk building</text>
                    </g>
                    <!-- stock fell (Mar 9, x=820) -->
                    <g>
                        <line x1="820" y1="243" x2="820" y2="275" stroke="#dc2626" stroke-width="1.5" marker-end="url(#redArrow3)"/>
                        <circle cx="820" cy="286.5" r="4.5" fill="#dc2626"/>
                        <text x="800" y="235" fill="#dc2626" font-size="11.5" text-anchor="end" font-weight="700">Collapsed, halted Mar 10</text>
                    </g>
                    <text x="440" y="322" fill="#b45309" font-size="11" text-anchor="middle">about 12 months of quiet decline, then failure in days</text>
                    <g fill="#94a3b8" font-size="10.5" text-anchor="middle">
                        <text x="70" y="318">Mar '22</text><text x="445" y="318">Sep '22</text><text x="820" y="318">Mar '23</text>
                    </g>
                    <defs>
                        <marker id="amberArrow3" markerWidth="8" markerHeight="8" refX="4" refY="1" orient="auto"><path d="M0,8 L4,1 L8,8" fill="#b45309"/></marker>
                        <marker id="redArrow3" markerWidth="8" markerHeight="8" refX="4" refY="7" orient="auto"><path d="M0,0 L4,7 L8,0" fill="#dc2626"/></marker>
                    </defs>
                </svg>
            </div>
            <div class="slide-info">
                <div class="info-block risk">
                    <div class="lbl">The risk in the report</div>
                    <p>Filings disclosed a large bond portfolio exposed to <b>rising interest rates</b>, along with a deposit base concentrated in tech startups.</p>
                </div>
                <div class="info-block event">
                    <div class="lbl">What happened at the event</div>
                    <p>In March 2023 the bank sold bonds at a loss and tried to raise capital. Depositors rushed to withdraw, triggering a classic bank run.</p>
                </div>
                <div class="info-block stock">
                    <div class="lbl">What happened to the stock</div>
                    <p>Shares crashed from about <b>$268 to $106 on March 9</b>, then trading was halted on March 10 as regulators shut the bank down.</p>
                </div>
            </div>
        </div>
    </div>

    <div class="carousel-controls">
        <button class="arrow-btn" onclick="move(-1)" aria-label="Previous">&#8592;</button>
        <div class="dots">
            <button class="dot active" onclick="go(0)">1</button>
            <button class="dot" onclick="go(1)">2</button>
            <button class="dot" onclick="go(2)">3</button>
        </div>
        <button class="arrow-btn" onclick="move(1)" aria-label="Next">&#8594;</button>
    </div>
</section>

<!-- ============ TWO LENSES ============ -->
<section>
    <div class="section-head">
        <div class="kicker">The Method</div>
        <h3>Two ways to see risk</h3>
        <p>The numbers show where a company has been. The risk warnings hint at where it is going. Most investors watch only one. We read both.</p>
    </div>
    <div class="lens-grid">
        <div class="lens numbers">
            <div class="tag">The Numbers, looks at the past</div>
            <h4>Financial health</h4>
            <p class="desc">Signals hiding in the balance sheet and income statement, the story the reported figures tell.</p>
            <ul>
                <li>Revenue and margin trends, the profit squeeze</li>
                <li>Inventory piling up faster than sales</li>
                <li>Receivables and cash flow quality</li>
                <li>Debt, leverage, and interest coverage</li>
                <li>Guidance cuts against prior expectations</li>
            </ul>
        </div>
        <div class="lens words">
            <div class="tag">The Words, looks at the future</div>
            <h4>Risk factor language</h4>
            <p class="desc">What companies are legally required to disclose, and how their tone shifts when trouble is coming.</p>
            <ul>
                <li>Brand new risks that were not there last year</li>
                <li>Boilerplate turning specific and urgent</li>
                <li>Negative and uncertainty word density rising</li>
                <li>Risks quietly dropped or downplayed</li>
                <li>Shifts the market has not priced in yet</li>
            </ul>
        </div>
    </div>
</section>

<!-- ============ HOW IT WORKS ============ -->
<section class="section-alt">
    <div class="section-head">
        <div class="kicker">How It Works</div>
        <h3>Three simple steps</h3>
    </div>
    <div class="steps">
        <div class="step">
            <div class="num">1</div>
            <h4>Pick a company</h4>
            <p>Choose any company in the S&amp;P 500. We pull its official reports and yearly filings automatically.</p>
        </div>
        <div class="step">
            <div class="num">2</div>
            <h4>We read everything</h4>
            <p>We compare this year to last year, both the numbers and the risk warnings, and pinpoint what changed.</p>
        </div>
        <div class="step">
            <div class="num">3</div>
            <h4>See the warning signs</h4>
            <p>Get a plain English breakdown of each risk with a score from 0 to 100. Higher means a louder warning.</p>
        </div>
    </div>
</section>

<!-- ============ LEARNING ============ -->
<section id="learning">
    <div class="section-head">
        <div class="kicker">Learning</div>
        <h3>What the filings were telling us</h3>
        <p>Short, plain English case studies on the risks we find hiding inside company reports.</p>
    </div>
    <div class="learn-grid">
        <button class="learn-card" onclick="openCase('tgt')">
            <div class="thumb t1"></div>
            <div class="body">
                <div class="meta">Case study &middot; Retail</div>
                <h4>What Target's 2022 filing quietly said before the fall</h4>
                <p>How language about excess inventory foreshadowed the worst trading day in 35 years.</p>
            </div>
        </button>
        <button class="learn-card" onclick="openCase('luv')">
            <div class="thumb t2"></div>
            <div class="body">
                <div class="meta">Case study &middot; Airlines</div>
                <h4>Southwest warned about its own systems for years</h4>
                <p>An operational risk sat in plain sight until a winter storm turned it into a meltdown.</p>
            </div>
        </button>
        <button class="learn-card" onclick="openCase('sivb')">
            <div class="thumb t3"></div>
            <div class="body">
                <div class="meta">Case study &middot; Banking</div>
                <h4>The interest rate risk that ended Silicon Valley Bank</h4>
                <p>A bond portfolio and a concentrated deposit base, both public, both overlooked.</p>
            </div>
        </button>
    </div>

    <!-- Inline expandable case study details -->
    <div id="case-tgt" class="case-detail">
        <div class="cd-head">
            <div>
                <div class="cd-title">Target (TGT), 2022</div>
                <div class="cd-tag">Retail &middot; the profit squeeze that showed up in words first</div>
            </div>
            <button class="cd-close" onclick="closeCase()">Close</button>
        </div>
        <div class="cd-body">
            <h5>What changed in the filing</h5>
            <p>Compared with the prior year, Target's risk language grew more specific about excess inventory, supply chain disruption, and rising costs. The calm, repeated boilerplate of earlier filings gave way to sharper, more concrete warnings.</p>
            <h5>What happened next</h5>
            <p>On May 18, 2022, Target reported first quarter results and cut its operating margin outlook from over 8 percent to about 6 percent, pointing to the same inventory and cost pressures. The stock fell 24.9 percent in a single session to $161.61, its worst day since 1987, erasing roughly $25 billion in value.</p>
            <h5>The takeaway</h5>
            <p>The numbers still looked healthy on the surface. The words were already less confident. Reading both together gave a fuller picture than watching the price alone.</p>
            <div class="cd-note">An observation, not a prediction. We are exploring whether the effects described in the research appear in this company's story.</div>
        </div>
    </div>

    <div id="case-luv" class="case-detail">
        <div class="cd-head">
            <div>
                <div class="cd-title">Southwest Airlines (LUV), 2022</div>
                <div class="cd-tag">Airlines &middot; a known operational risk that finally broke</div>
            </div>
            <button class="cd-close" onclick="closeCase()">Close</button>
        </div>
        <div class="cd-body">
            <h5>What the filing said</h5>
            <p>For years, Southwest's filings flagged its reliance on aging technology and crew scheduling systems as an operational risk. The warning was consistent and public, but it read as routine and drew little attention.</p>
            <h5>What happened next</h5>
            <p>A severe winter storm in December 2022 overwhelmed those very systems, forcing thousands of cancellations and stranding travelers over the holidays. The stock fell about 15.6 percent that month as the meltdown played out.</p>
            <h5>The takeaway</h5>
            <p>A risk that a company repeats year after year is easy to tune out. Tracking which disclosed risks are most exposed can help decide which ones deserve a second look.</p>
            <div class="cd-note">An observation, not a prediction. We are exploring whether the effects described in the research appear in this company's story.</div>
        </div>
    </div>

    <div id="case-sivb" class="case-detail">
        <div class="cd-head">
            <div>
                <div class="cd-title">SVB Financial, Silicon Valley Bank (SIVB), 2022 to 2023</div>
                <div class="cd-tag">Banking &middot; interest rate risk that was disclosed all along</div>
            </div>
            <button class="cd-close" onclick="closeCase()">Close</button>
        </div>
        <div class="cd-body">
            <h5>What the filing said</h5>
            <p>SVB's filings described a large bond portfolio exposed to rising interest rates and a deposit base heavily concentrated in technology startups. Both of these were stated plainly in its reports.</p>
            <h5>What happened next</h5>
            <p>As rates rose through 2022, the stock drifted down from about $559 to the low $200s. In March 2023 the bank sold bonds at a loss and tried to raise capital. Depositors rushed to withdraw, the stock crashed from about $268 to $106 on March 9, and trading was halted on March 10 as regulators closed the bank.</p>
            <h5>The takeaway</h5>
            <p>The two ingredients of the failure were both in the filings well before the collapse. The story moved slowly for a year, then very fast in a matter of days.</p>
            <div class="cd-note">An observation, not a prediction. We are exploring whether the effects described in the research appear in this company's story.</div>
        </div>
    </div>
</section>

<!-- ============ RESEARCH STRIP ============ -->
<section class="research">
    <div class="wrap">
        <div class="big">22%<span>per year in abnormal returns (Lazy Prices, 2020)</span></div>
        <div>
            <h3>Grounded in decades of research</h3>
            <p>The finding that filing language predicts future problems is not new. In one landmark study, buying companies whose filings barely changed and selling those whose filings changed a lot earned roughly 22 percent per year. The reason it keeps working is that almost nobody reads these documents. See the papers below, including two from the University of Washington.</p>
        </div>
    </div>
</section>

<!-- ============ ACADEMIC RESEARCH ============ -->
<section id="research">
    <div class="section-head">
        <div class="kicker">Academic Research</div>
        <h3>Standing on the work of others</h3>
    </div>
    <p class="intro-note">
        This project takes its inspiration from finance and accounting professors who have studied, over many years,
        how the risk language in company filings can foreshadow what comes next. S&amp;P 500 Risk Radar does not claim
        new research. It is a student's attempt to explore, in real companies, whether the patterns these researchers
        described actually show up in the real world. The papers below are the foundation for that idea.
    </p>

    <div class="papers">
        <!-- Lazy Prices -->
        <div class="paper">
            <div class="p-head" onclick="togglePaper(this)">
                <span class="p-badge">Harvard &amp; DePaul</span>
                <div class="p-main">
                    <div class="p-title">Lazy Prices</div>
                    <div class="p-meta">Lauren Cohen, Christopher Malloy, and Quoc Nguyen (2020) &middot; The Journal of Finance</div>
                </div>
                <span class="p-toggle">+</span>
            </div>
            <div class="p-detail">
                <p>The anchor paper for this project. Studying two decades of company filings, the authors showed that changes in filing language, especially in the risk and management sections, predict weaker future returns. A strategy of buying companies whose filings barely changed and selling those whose filings changed a lot earned roughly 22 percent per year. The reason it works is investor inattention. Almost nobody reads these long documents closely.</p>
                <a href="https://onlinelibrary.wiley.com/doi/10.1111/jofi.12885" target="_blank" rel="noopener">View the paper &#8599;</a>
            </div>
        </div>

        <!-- Loughran McDonald -->
        <div class="paper">
            <div class="p-head" onclick="togglePaper(this)">
                <span class="p-badge">Notre Dame</span>
                <div class="p-main">
                    <div class="p-title">When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks</div>
                    <div class="p-meta">Tim Loughran and Bill McDonald (2011) &middot; The Journal of Finance</div>
                </div>
                <span class="p-toggle">+</span>
            </div>
            <div class="p-detail">
                <p>Built the finance specific word lists that this project relies on for reading tone. The authors showed that general purpose sentiment dictionaries misread financial writing, because words like liability or tax are neutral in a finance context. Their negative and uncertainty word lists became the standard tool for measuring tone in filings, and Risk Radar uses this approach for the words half of its analysis.</p>
                <a href="https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1540-6261.2010.01625.x" target="_blank" rel="noopener">View the paper &#8599;</a>
            </div>
        </div>

        <!-- Campbell et al -->
        <div class="paper">
            <div class="p-head" onclick="togglePaper(this)">
                <span class="p-badge">Georgia &amp; Arizona</span>
                <div class="p-main">
                    <div class="p-title">The Information Content of Mandatory Risk Factor Disclosures in Corporate Filings</div>
                    <div class="p-meta">John Campbell, Hsinchun Chen, Dan Dhaliwal, Hsin-min Lu, and Logan Steele (2014) &middot; Review of Accounting Studies</div>
                </div>
                <span class="p-toggle">+</span>
            </div>
            <div class="p-detail">
                <p>Examined whether the risk factor section is meaningful or just boilerplate. The authors found that firms facing greater risk disclose more risk factors, and that the type of risk a firm describes, whether financial, legal, or otherwise, lines up with the actual risk it faces. This supports the idea that the risk section carries real information about the company.</p>
                <a href="https://papers.ssrn.com/sol3/papers.cfm?abstract_id=1694279" target="_blank" rel="noopener">View the paper &#8599;</a>
            </div>
        </div>

        <!-- Kravet Muslu -->
        <div class="paper">
            <div class="p-head" onclick="togglePaper(this)">
                <span class="p-badge">UConn &amp; UT Dallas</span>
                <div class="p-main">
                    <div class="p-title">Textual Risk Disclosures and Investors' Risk Perceptions</div>
                    <div class="p-meta">Todd Kravet and Volkan Muslu (2013) &middot; Review of Accounting Studies</div>
                </div>
                <span class="p-toggle">+</span>
            </div>
            <div class="p-detail">
                <p>Studied what happens when a firm increases its risk language from one year to the next. The authors found that these annual increases are followed by higher stock return volatility and trading volume around and after the filing, along with more spread out analyst forecasts. In short, when companies say more about risk, the market treats them as riskier.</p>
                <a href="https://link.springer.com/article/10.1007/s11142-013-9228-9" target="_blank" rel="noopener">View the paper &#8599;</a>
            </div>
        </div>

        <!-- Hope Hu Lu -->
        <div class="paper">
            <div class="p-head" onclick="togglePaper(this)">
                <span class="p-badge">Toronto</span>
                <div class="p-main">
                    <div class="p-title">The Benefits of Specific Risk-Factor Disclosures</div>
                    <div class="p-meta">Ole-Kristian Hope, Danqi Hu, and Hai Lu (2016) &middot; Review of Accounting Studies</div>
                </div>
                <span class="p-toggle">+</span>
            </div>
            <div class="p-detail">
                <p>Asked whether it matters how specific a risk factor is, rather than just how many there are. The authors found that more specific risk factors, as opposed to vague boilerplate, are more useful to investors and analysts. This matters for Risk Radar, because a generic warning and a concrete, detailed one are not the same signal.</p>
                <a href="https://link.springer.com/article/10.1007/s11142-016-9371-1" target="_blank" rel="noopener">View the paper &#8599;</a>
            </div>
        </div>

        <!-- Gaulin -->
        <div class="paper">
            <div class="p-head" onclick="togglePaper(this)">
                <span class="p-badge">Rice</span>
                <div class="p-main">
                    <div class="p-title">The Information Content of Risk Factor Disclosures in Quarterly Reports</div>
                    <div class="p-meta">Maclean Gaulin (2015) &middot; Accounting Horizons</div>
                </div>
                <span class="p-toggle">+</span>
            </div>
            <div class="p-detail">
                <p>Looked at what happens when firms update their risk factors during the year. The study found that companies that add or change risk factors tend to have lower future unexpected earnings and are more likely to suffer sharp negative earnings surprises. In other words, updates to the risk section often arrive as an early warning of bad news.</p>
                <a href="https://publications.aaahq.org/accounting-horizons/article-abstract/29/4/887/2223/" target="_blank" rel="noopener">View the paper &#8599;</a>
            </div>
        </div>
    </div>

    <p class="intro-note" style="margin-top:36px; margin-bottom:0; font-size:13.5px; color:var(--slate);">
        A note on Washington state. The landmark Lazy Prices research was presented at the University of Washington
        and Washington State University while it was being developed, and the Foster School of Business at the
        University of Washington has active researchers in financial reporting and disclosure. This work has deep
        roots close to home.
    </p>
</section>

<!-- ============ ABOUT ============ -->
<section id="about" class="section-alt">
    <div class="section-head">
        <div class="kicker">About</div>
        <h3>Who built this</h3>
    </div>
    <div class="about">
        <svg class="avatar" viewBox="0 0 132 132">
            <rect width="132" height="132" rx="20" fill="#0b1b34"/>
            <polyline points="18,96 42,74 62,84 84,50 108,60" fill="none" stroke="#2563eb" stroke-width="3.5" stroke-linecap="round" stroke-linejoin="round" opacity="0.85"/>
            <circle cx="108" cy="60" r="4.5" fill="#dc2626"/>
            <text x="66" y="46" text-anchor="middle" fill="#ffffff" font-size="30" font-weight="800" font-family="sans-serif">AV</text>
        </svg>
        <div>
            <h3>Akilan Vadivelan</h3>
            <div class="role">North Creek High School, Class of 2027 &middot; future finance</div>
            <p>I am a high school senior planning a career in finance. I believe finance is the backbone of everyday life and of every organization, whether a young startup or a century old giant. I am drawn to the stories behind the numbers, especially how struggling companies use strategy and finance to reinvent themselves, which I explore in my Substack.</p>
            <p class="why">The idea is simple. The warning signs are already public, buried in filings that few people read. S&amp;P 500 Risk Radar is my attempt to surface them early, and to test a question that fascinates me. Can the words predict the fall before the numbers do?</p>
            <a class="sub-link" href="https://akilanvadivelan.substack.com" target="_blank" rel="noopener">Read my Substack &#8599;</a>
        </div>
    </div>
</section>

<!-- ============ FOOTER ============ -->
<div class="footer">
    <div class="wrap">
        S&amp;P 500 Risk Radar<br>
        Built on Cohen, Malloy and Nguyen "Lazy Prices" (2020) and Loughran and McDonald Financial Sentiment (2011)<br>
        Price data from company investor relations. Filings from SEC EDGAR. Not investment advice.
    </div>
</div>

<script>
    var slides = document.querySelectorAll('.slide');
    var dots = document.querySelectorAll('.dot');
    var cur = 0;
    function render() {
        slides.forEach(function(s, i){ s.classList.toggle('active', i === cur); });
        dots.forEach(function(d, i){ d.classList.toggle('active', i === cur); });
    }
    function go(i) { cur = (i + slides.length) % slides.length; render(); }
    function move(d) { go(cur + d); }

    // Inline case study open/close
    function openCase(id) {
        document.querySelectorAll('.case-detail').forEach(function(c){ c.classList.remove('open'); });
        var el = document.getElementById('case-' + id);
        if (el) {
            el.classList.add('open');
            el.scrollIntoView({ behavior: 'smooth', block: 'center' });
        }
    }
    function closeCase() {
        document.querySelectorAll('.case-detail').forEach(function(c){ c.classList.remove('open'); });
    }

    // Academic paper expand/collapse
    function togglePaper(head) {
        head.parentElement.classList.toggle('open');
    }
</script>
</body>
</html>
"""



ANALYZE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Analyze - S&amp;P 500 Risk Radar</title>
<style>
    :root {
        --bg: #ffffff; --bg-alt: #f7f9fc; --navy: #0b1b34; --ink: #0f172a;
        --slate: #64748b; --line: #e5e9f0; --accent: #2563eb; --accent-soft: #eff4ff;
        --red: #dc2626; --amber: #b45309;
        --shadow: 0 1px 3px rgba(15,23,42,0.06), 0 8px 24px rgba(15,23,42,0.05);
    }
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; background: var(--bg-alt); color: var(--ink); -webkit-font-smoothing: antialiased; line-height: 1.5; }
    a { color: inherit; }

    .nav { position: sticky; top: 0; z-index: 50; background: rgba(255,255,255,0.9); backdrop-filter: saturate(180%) blur(12px); border-bottom: 1px solid var(--line); }
    .nav-inner { max-width: 1000px; margin: 0 auto; padding: 14px 24px; display: flex; align-items: center; gap: 40px; }
    .brand { display: flex; align-items: center; gap: 10px; text-decoration: none; }
    .brand .logo { width: 26px; height: 26px; }
    .brand h1 { font-size: 16px; color: var(--ink); font-weight: 700; letter-spacing: -0.2px; white-space: nowrap; }
    .brand h1 span { color: var(--accent); }
    .menu { display: flex; align-items: center; gap: 26px; }
    .menu a { color: var(--slate); text-decoration: none; font-size: 14px; font-weight: 500; }
    .menu a:hover { color: var(--ink); }
    .nav-actions { margin-left: auto; }
    .signin-btn { padding: 8px 16px; border: 1px solid var(--line); border-radius: 8px; background: #f1f5f9; color: #94a3b8; font-size: 13px; font-weight: 600; cursor: not-allowed; font-family: inherit; }

    .container { max-width: 820px; margin: 0 auto; padding: 40px 24px 80px; }

    .search { background: var(--bg); border: 1px solid var(--line); border-radius: 16px; padding: 32px; box-shadow: var(--shadow); text-align: center; }
    .search h2 { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px; }
    .search p { color: var(--slate); font-size: 15px; margin-bottom: 24px; }
    .search-row { display: flex; gap: 10px; max-width: 440px; margin: 0 auto; }
    .search-row input { flex: 1; background: var(--bg-alt); border: 1px solid var(--line); border-radius: 10px; color: var(--ink); padding: 13px 16px; font-size: 16px; font-family: inherit; text-transform: uppercase; }
    .search-row input:focus { outline: none; border-color: var(--accent); background: #fff; }
    .search-row button { padding: 13px 26px; border: none; border-radius: 10px; background: var(--accent); color: #fff; font-size: 15px; font-weight: 600; cursor: pointer; font-family: inherit; }
    .search-row button:hover { background: #1d4fd7; }
    .search-row button:disabled { background: #cbd5e1; cursor: not-allowed; }
    .hint { margin-top: 14px; font-size: 13px; color: var(--slate); }

    .status { margin-top: 18px; font-size: 14px; min-height: 20px; }
    .status.error { color: var(--red); }
    .status.info { color: var(--slate); }

    .loading { display: none; text-align: center; padding: 40px; color: var(--slate); }
    .loading.show { display: block; }
    .spinner { width: 34px; height: 34px; border: 3px solid var(--line); border-top-color: var(--accent); border-radius: 50%; margin: 0 auto 14px; animation: spin 0.8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }

    .results { display: none; margin-top: 28px; }
    .results.show { display: block; }

    .result-head { margin-bottom: 20px; }
    .result-head .co { font-size: 24px; font-weight: 800; letter-spacing: -0.4px; }
    .result-head .co .tk { color: var(--accent); }
    .result-head .years { color: var(--slate); font-size: 14px; margin-top: 4px; }

    .signal { background: var(--bg); border: 1px solid var(--line); border-radius: 16px; padding: 26px; box-shadow: var(--shadow); margin-bottom: 22px; }
    .signal .lbl { font-size: 12px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: var(--slate); margin-bottom: 12px; }
    .signal .band { font-size: 30px; font-weight: 800; letter-spacing: -0.5px; text-transform: capitalize; margin-bottom: 14px; }
    .signal .summary { color: #334155; font-size: 15px; line-height: 1.65; }
    .meter { height: 10px; background: var(--bg-alt); border-radius: 6px; overflow: hidden; margin-bottom: 16px; border: 1px solid var(--line); }
    .meter .fill { height: 100%; border-radius: 6px; transition: width 0.7s ease; }

    .band-mild    { color: #15803d; } .fill-mild    { background: linear-gradient(90deg,#4ade80,#22c55e); }
    .band-moderate{ color: #b45309; } .fill-moderate{ background: linear-gradient(90deg,#fbbf24,#f59e0b); }
    .band-serious { color: #c2410c; } .fill-serious { background: linear-gradient(90deg,#fb923c,#ea580c); }
    .band-severe  { color: #dc2626; } .fill-severe  { background: linear-gradient(90deg,#f87171,#dc2626); }

    .section-label { font-size: 13px; font-weight: 700; letter-spacing: 0.5px; text-transform: uppercase; color: var(--slate); margin: 6px 0 14px; }

    .card { background: var(--bg); border: 1px solid var(--line); border-radius: 14px; padding: 22px 24px; box-shadow: var(--shadow); margin-bottom: 16px; }
    .card .top { display: flex; justify-content: space-between; align-items: flex-start; gap: 14px; margin-bottom: 12px; }
    .card .title { font-size: 16px; font-weight: 700; color: var(--ink); line-height: 1.35; }
    .card .score { flex-shrink: 0; text-align: right; }
    .card .score .n { font-size: 24px; font-weight: 800; line-height: 1; }
    .card .score .of { font-size: 11px; color: var(--slate); }
    .chip { display: inline-block; font-size: 10.5px; font-weight: 700; letter-spacing: 0.4px; padding: 4px 9px; border-radius: 999px; margin-bottom: 10px; }
    .chip.new { background: #fef2f2; color: #dc2626; }
    .chip.rewritten { background: #fff7ed; color: #c2410c; }
    .chip.removed { background: #f1f5f9; color: #64748b; }

    .tone { margin: 12px 0; }
    .tone .tone-top { display: flex; justify-content: space-between; font-size: 12px; color: var(--slate); margin-bottom: 5px; }
    .tone .tone-label { font-weight: 700; text-transform: capitalize; }
    .tone .caption { font-size: 13.5px; color: #334155; margin-top: 8px; line-height: 1.55; }

    .note { font-size: 13.5px; color: #475569; line-height: 1.6; margin-top: 12px; }
    .note strong { color: var(--ink); }

    .see-wording { margin-top: 14px; }
    .see-wording summary { cursor: pointer; color: var(--accent); font-size: 13.5px; font-weight: 600; list-style: none; }
    .see-wording summary::-webkit-details-marker { display: none; }
    .see-wording summary::before { content: "\\25B8 "; }
    .see-wording[open] summary::before { content: "\\25BE "; }
    .wording-box { margin-top: 12px; padding: 14px 16px; background: var(--bg-alt); border-radius: 10px; font-size: 14px; line-height: 1.8; color: #334155; }
    .wording-box p { margin-bottom: 8px; }
    .w-neg { background: #fee2e2; color: #b91c1c; padding: 0 3px; border-radius: 3px; font-weight: 600; }
    .w-unc { background: #fef3c7; color: #92400e; padding: 0 3px; border-radius: 3px; font-weight: 600; }
    .legend { margin-top: 10px; font-size: 12px; color: var(--slate); }
    .legend .w-neg, .legend .w-unc { font-weight: 600; }

    .tone-key { background: var(--accent-soft); border-radius: 12px; padding: 16px 18px; font-size: 13px; color: #1e3a5f; line-height: 1.6; margin-bottom: 22px; }
    .tone-key b { color: var(--navy); }

    .unchanged { margin-top: 18px; }
    .unchanged summary { cursor: pointer; color: var(--slate); font-size: 14px; font-weight: 600; }
    .unchanged ul { margin: 12px 0 0 4px; list-style: none; }
    .unchanged li { color: var(--slate); font-size: 13px; padding: 4px 0; }

    .foot-note { margin-top: 26px; padding: 14px 18px; background: var(--bg); border: 1px solid var(--line); border-radius: 12px; font-size: 12.5px; color: var(--slate); line-height: 1.6; text-align: center; }

    @media (max-width: 640px) { .menu { display: none; } .search-row { flex-direction: column; } }
</style>
</head>
<body>

<nav class="nav">
    <div class="nav-inner">
        <a class="brand" href="/">
            <svg class="logo" viewBox="0 0 32 32" fill="none">
                <circle cx="16" cy="16" r="14" stroke="#2563eb" stroke-width="2" opacity="0.35"/>
                <circle cx="16" cy="16" r="8.5" stroke="#2563eb" stroke-width="2" opacity="0.6"/>
                <circle cx="16" cy="16" r="2.6" fill="#dc2626"/>
                <line x1="16" y1="16" x2="27" y2="6.5" stroke="#2563eb" stroke-width="2" stroke-linecap="round"/>
            </svg>
            <h1>S&amp;P 500 <span>Risk Radar</span></h1>
        </a>
        <div class="menu">
            <a href="/">Home</a>
            <a href="/analyze">Analyze</a>
            <a href="/#learning">Learning</a>
            <a href="/#research">Research</a>
            <a href="/#about">About</a>
        </div>
        <div class="nav-actions">
            <button class="signin-btn" disabled title="Coming soon">Sign In</button>
        </div>
    </div>
</nav>

<div class="container">
    <div class="search">
        <h2>Analyze a company's risk</h2>
        <p>Enter an S&amp;P 500 ticker. We compare its two most recent annual filings.</p>
        <div class="search-row">
            <input type="text" id="ticker" placeholder="e.g. AMZN" autocomplete="off" spellcheck="false">
            <button id="go-btn" onclick="analyze()">Analyze</button>
        </div>
        <div class="hint">Covers S&amp;P 500 companies as of 30 August 2026.</div>
        <div class="status" id="status"></div>
    </div>

    <div class="loading" id="loading">
        <div class="spinner"></div>
        <div id="loading-text">Reading the filings and comparing the risk language...</div>
    </div>

    <div class="results" id="results"></div>
</div>

<script>
    var tickerEl = document.getElementById('ticker');
    var statusEl = document.getElementById('status');
    var loadingEl = document.getElementById('loading');
    var resultsEl = document.getElementById('results');
    var goBtn = document.getElementById('go-btn');

    tickerEl.addEventListener('keypress', function(e) { if (e.key === 'Enter') analyze(); });

    function setStatus(msg, cls) {
        statusEl.textContent = msg || '';
        statusEl.className = 'status ' + (cls || '');
    }

    async function analyze() {
        var ticker = (tickerEl.value || '').trim().toUpperCase();
        if (!ticker) { setStatus('Please enter a ticker.', 'error'); return; }

        setStatus('');
        resultsEl.classList.remove('show');
        resultsEl.innerHTML = '';
        document.getElementById('loading-text').textContent =
            'Reading ' + ticker + "'s two most recent filings and comparing the risk language. This can take up to half a minute.";
        loadingEl.classList.add('show');
        goBtn.disabled = true;

        try {
            var resp = await fetch('/api/analyze', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ ticker: ticker })
            });
            var data = await resp.json();
            loadingEl.classList.remove('show');
            goBtn.disabled = false;

            if (data.error) {
                setStatus(data.message || 'Something went wrong.', 'error');
                return;
            }
            render(data);
        } catch (err) {
            loadingEl.classList.remove('show');
            goBtn.disabled = false;
            setStatus('Could not reach the server. Please try again.', 'error');
        }
    }

    function esc(s) {
        var d = document.createElement('div');
        d.textContent = s == null ? '' : String(s);
        return d.innerHTML;
    }

    function render(data) {
        var band = data.headline.band;
        var html = '';

        html += '<div class="result-head">';
        html += '  <div class="co"><span class="tk">' + esc(data.ticker) + '</span> &mdash; ' + esc(data.company) + '</div>';
        html += '  <div class="years">Showing the two most recent filings we have: ' + esc(data.current_year) + ' and ' + esc(data.prior_year) + '.</div>';
        html += '</div>';

        // Signal panel
        html += '<div class="signal">';
        html += '  <div class="lbl">Risk signal</div>';
        html += '  <div class="band band-' + band + '">' + band + '</div>';
        html += '  <div class="meter"><div class="fill fill-' + band + '" style="width:' + data.headline.score + '%"></div></div>';
        html += '  <div class="summary">' + esc(data.headline.summary) + '</div>';
        html += '</div>';

        // Tone key
        html += '<div class="tone-key">';
        html += '  <b>How to read tone.</b> Negative words describe harm, decline, or failure (adverse, impair, loss). ';
        html += '  Uncertainty words are hedging language a company uses when it is unsure (may, could, uncertain). ';
        html += '  More of these, especially more than last year, means the tone is darkening. That is the warning the research points to.';
        html += '</div>';

        if (data.risks.length) {
            html += '<div class="section-label">What changed this year</div>';
            data.risks.forEach(function(r) { html += renderCard(r); });
        } else {
            html += '<div class="card"><div class="note">No changed risks were detected between these two years. Most of the filing is unchanged.</div></div>';
        }

        // Unchanged
        if (data.unchanged_titles && data.unchanged_titles.length) {
            html += '<details class="unchanged"><summary>' + data.unchanged_titles.length + ' risks unchanged from last year (no signal)</summary><ul>';
            data.unchanged_titles.forEach(function(t) { html += '<li>' + esc(t) + '</li>'; });
            html += '</ul></details>';
        }

        html += '<div class="foot-note">An observation, not a prediction. We explore whether the patterns academic research describes show up in real companies. This looks only at the risk wording. A company\\'s financial numbers are a separate lens we plan to add. Not investment advice.</div>';

        resultsEl.innerHTML = html;
        resultsEl.classList.add('show');
        // Scroll so the company header stays in view (offset for the sticky nav),
        // keeping the user in context rather than jumping to the risk panel.
        var top = resultsEl.getBoundingClientRect().top + window.pageYOffset - 72;
        window.scrollTo({ top: top, behavior: 'smooth' });
    }

    function renderCard(r) {
        var chipClass = r.status === 'NEW' ? 'new' : (r.status === 'MODIFIED' ? 'rewritten' : 'removed');
        var toneBand = r.tone_label;
        var h = '<div class="card">';
        h += '  <div class="top">';
        h += '    <div><span class="chip ' + chipClass + '">' + esc(r.status_label) + '</span><div class="title">' + esc(r.title) + '</div></div>';
        h += '    <div class="score"><div class="n band-' + toneBand + '">' + r.score + '</div><div class="of">out of 100</div></div>';
        h += '  </div>';

        // Tone bar
        h += '  <div class="tone">';
        h += '    <div class="tone-top"><span>Tone of the wording</span><span class="tone-label band-' + toneBand + '">' + toneBand + '</span></div>';
        h += '    <div class="meter"><div class="fill fill-' + toneBand + '" style="width:' + r.tone_score + '%"></div></div>';
        h += '    <div class="caption">' + esc(r.tone_caption) + '</div>';
        h += '  </div>';

        if (r.note) { h += '  <div class="note">' + r.note + '</div>'; }

        if (r.wording && r.wording.length) {
            h += '  <details class="see-wording"><summary>See the wording</summary>';
            h += '    <div class="wording-box">';
            r.wording.forEach(function(w) { h += '<p>' + w + '</p>'; });
            h += '      <div class="legend"><span class="w-neg">negative words</span> &nbsp; <span class="w-unc">uncertainty words</span></div>';
            h += '    </div>';
            h += '  </details>';
        }

        h += '</div>';
        return h;
    }
</script>
</body>
</html>"""



# =========================================================
# HTTP Handler
# =========================================================

class ERPSAHandler(BaseHTTPRequestHandler):
    """HTTP request handler for ERPSA web app."""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == '/' or path == '':
            self._serve_html(HOME_PAGE)
        elif path == '/analyze':
            self._serve_html(ANALYZE_PAGE)
        elif path == '/api/lookup':
            self._handle_lookup(parsed)
        else:
            self.send_response(404)
            self.end_headers()
            self.wfile.write(b'Not Found')

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path == '/api/analyze':
            self._handle_analyze()
        else:
            self.send_response(404)
            self.end_headers()

    def _serve_html(self, html):
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html.encode('utf-8'))

    def _serve_json(self, data):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(data).encode('utf-8'))

    def _handle_lookup(self, parsed):
        """
        Ticker-only lookup. Given just a ticker, discover the two most recent
        filing years available in S3 and return them for display. Falls back to
        live SEC EDGAR only if the ticker is not present in S3 at all.
        """
        params = parse_qs(parsed.query)
        ticker = params.get('ticker', [''])[0].strip().upper()
        if not ticker:
            self._serve_json({'error': 'No ticker provided'})
            return

        print(f"  [LOOKUP] Ticker: {ticker}")

        # Primary path: what do we have in S3?
        s3_years = list_s3_years_for_ticker(ticker)
        if s3_years:
            latest = s3_years[0]
            prior = s3_years[1] if len(s3_years) > 1 else None
            company = get_company_name_from_s3(ticker, latest)
            result = {
                'ticker': ticker,
                'company': company,
                'source': 's3',
                'available_years': s3_years,
                'latest_year': latest,
                'prior_year': prior,
                'can_compare': prior is not None,
            }
            if prior is None:
                result['message'] = (
                    f"We only have one year of filings for {ticker} so far "
                    f"({latest}). A year over year comparison needs two."
                )
            self._serve_json(result)
            return

        # Fallback: not in S3 yet, tell the user (avoid slow live path by default).
        self._serve_json({
            'ticker': ticker,
            'company': ticker,
            'source': 'none',
            'available_years': [],
            'latest_year': None,
            'prior_year': None,
            'can_compare': False,
            'message': f"We do not have filings for {ticker} yet.",
        })

    def _handle_analyze(self):
        """
        Ticker-only analysis. Reads the two most recent years for the ticker
        from S3, runs the risk pipeline live, and returns the Option A payload
        (headline band + per-risk cards with tone bars and highlighted wording).
        No Zacks, no buy/sell recommendation. Numbers lens comes later.
        """
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
            ticker = str(data.get('ticker', '')).strip().upper()
            if not ticker:
                self._serve_json({'error': 'No ticker provided.'})
                return

            print(f"  [ANALYZE] {ticker}: discovering years in S3...")
            years = list_s3_years_for_ticker(ticker)

            if not years:
                self._serve_json({
                    'error': 'not_in_list',
                    'message': (
                        f"We do not have {ticker} in our data. This tool covers "
                        f"S&P 500 companies as of 30 August 2026 only."
                    ),
                })
                return

            if len(years) < 2:
                self._serve_json({
                    'error': 'one_year_only',
                    'message': (
                        f"We only have one year of filings for {ticker} so far "
                        f"({years[0]}). A year over year comparison needs two."
                    ),
                })
                return

            current_year, prior_year = years[0], years[1]
            company = get_company_name_from_s3(ticker, current_year)
            print(f"  [ANALYZE] {ticker}: comparing {current_year} vs {prior_year}")

            current_text = read_s3_risk_text(ticker, current_year)
            prior_text = read_s3_risk_text(ticker, prior_year)
            if not current_text or not prior_text:
                self._serve_json({
                    'error': 'read_failed',
                    'message': f"We could not read the stored filings for {ticker}. Please try again.",
                })
                return

            # ─── Run the risk pipeline (live) ───
            clean_current = clean_text_preserve_structure(current_text)
            clean_prior = clean_text_preserve_structure(prior_text)
            sections_current = parse_risk_sections(clean_current)
            sections_prior = parse_risk_sections(clean_prior)
            matches = match_risk_categories(sections_current, sections_prior)
            change_report = classify_risk_changes(
                matches=matches, ticker=ticker,
                current_year=current_year, prior_year=prior_year,
                total_current=len(sections_current),
                total_prior=len(sections_prior),
            )
            scoring = run_scoring(change_report, verbose=False)

            # ─── Serialize into Option A shape ───
            changed, unchanged = [], []
            for i, r in enumerate(scoring.risk_scores):
                classification = change_report.classifications[i] if i < len(change_report.classifications) else None
                card = serialize_risk(r, classification)
                if r.status == RiskChangeStatus.UNCHANGED:
                    unchanged.append(card['title'])
                else:
                    changed.append(card)

            changed.sort(key=lambda c: c['score'], reverse=True)
            headline = build_headline(scoring, current_year, prior_year)

            result = {
                'ticker': ticker,
                'company': company,
                'current_year': current_year,
                'prior_year': prior_year,
                'headline': headline,
                'risks': changed,
                'unchanged_titles': unchanged,
            }
            print(f"  [ANALYZE] {ticker}: done, {len(changed)} changed risks, {len(unchanged)} unchanged.")
            self._serve_json(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._serve_json({'error': 'server_error', 'message': f'Analysis error: {str(e)}'})

    def log_message(self, format, *args):
        """Custom log format."""
        pass


# =========================================================
# Main
# =========================================================

def main():
    port = int(os.environ.get('PORT', 8888))
    server = HTTPServer(('0.0.0.0', port), ERPSAHandler)

    import webbrowser
    print(f"""
S&P 500 Risk Radar
Server running at: http://localhost:{port}

  1. Open the site and go to Analyze.
  2. Enter an S&P 500 ticker (for example AMZN).
  3. We read the two most recent filings from S3 and compare the risk language.

  Risk text is read from S3 (bucket: {RISK_S3_BUCKET}, region: {AWS_REGION}).
  Press Ctrl+C to stop.
""")
    # Only auto-open browser when running locally (not on cloud servers)
    if not os.environ.get('RENDER') and not os.environ.get('PORT'):
        webbrowser.open(f'http://localhost:{port}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
