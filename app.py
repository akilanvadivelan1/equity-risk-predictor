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


def fetch_filing_text(filing: Dict) -> str:
    """Download and extract Item 1A from a 10-K filing."""
    cache_key = filing.get('accession', '')
    if cache_key in _filing_cache:
        return _filing_cache[cache_key]

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

        # ─── Fallback 2: Try the full-text submission viewer ───
        print(f"  [EDGAR] Trying full submission text...")
        full_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/0000000000-00-000000-index.htm"
        # Actually, try the raw filing text
        txt_url = f"https://www.sec.gov/Archives/edgar/data/{cik.lstrip('0')}/{accession}/{accession_raw.replace('-','')}.txt" if accession_raw else None
        if txt_url:
            try:
                req4 = urllib.request.Request(txt_url, headers=SEC_HEADERS)
                resp4 = urllib.request.urlopen(req4, timeout=60)
                full_text = resp4.read().decode('utf-8', errors='ignore')
                full_item1a = extract_item_1a(full_text)
                if full_item1a and len(full_item1a) > 500:
                    _filing_cache[cache_key] = full_item1a
                    print(f"  [EDGAR] Success with full text: {len(full_item1a)} chars")
                    return full_item1a
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
    text = html

    # ─── Strategy 1: Find ALL occurrences of Item 1A, skip TOC entries ───
    # The trick: Table of Contents entries are SHORT (just a link/reference)
    # The ACTUAL section header is followed by substantial content

    # Broader set of start patterns to catch more formatting styles
    start_patterns = [
        # Standard patterns
        re.compile(r'(?:>|"|;|\n)\s*(?:Item|ITEM)\s*1A[\.\s\u2014\u2013\-:]*\s*(?:Risk\s*Factors|RISK\s*FACTORS)', re.IGNORECASE),
        re.compile(r'(?:Item|ITEM)\s+1A[\.\s\u2014\u2013\-:]+\s*Risk\s*Factor', re.IGNORECASE),
        re.compile(r'<b[^>]*>\s*Item\s*1A', re.IGNORECASE),
        re.compile(r'<span[^>]*>\s*Item\s*1A', re.IGNORECASE),
        re.compile(r'font-weight:\s*(?:bold|700)[^>]*>\s*Item\s*1A', re.IGNORECASE),
        # XBRL-tagged
        re.compile(r'<ix:[^>]*>\s*Item\s*1A', re.IGNORECASE),
        # Just "ITEM 1A" in caps (common in older filings)
        re.compile(r'ITEM\s+1A\.?\s+RISK\s+FACTORS', re.IGNORECASE),
    ]

    # End patterns (Item 1B or Item 2)
    end_patterns = [
        re.compile(r'(?:>|"|;|\n)\s*(?:Item|ITEM)\s*1B[\.\s\u2014\u2013\-:]*\s*(?:Unresolved|UNRESOLVED)', re.IGNORECASE),
        re.compile(r'(?:>|"|;|\n)\s*(?:Item|ITEM)\s+1B[\.\s\u2014\u2013\-:]', re.IGNORECASE),
        re.compile(r'(?:>|"|;|\n)\s*(?:Item|ITEM)\s+2[\.\s\u2014\u2013\-:]+\s*(?:Propert|PROPERT)', re.IGNORECASE),
        re.compile(r'(?:Item|ITEM)\s+2[\.\s\u2014\u2013\-:]+\s*Propert', re.IGNORECASE),
        re.compile(r'ITEM\s+1B\.?\s', re.IGNORECASE),
        re.compile(r'ITEM\s+2\.?\s+PROPERTIES', re.IGNORECASE),
    ]

    # Find ALL start matches
    all_starts = []
    for pattern in start_patterns:
        for match in pattern.finditer(text):
            all_starts.append(match.start())

    if not all_starts:
        # Last resort: very broad search
        broad = re.compile(r'Item\s*1A', re.IGNORECASE)
        for match in broad.finditer(text):
            all_starts.append(match.start())

    if not all_starts:
        return ""

    # Sort and deduplicate (keep unique positions that are >500 chars apart)
    all_starts.sort()
    unique_starts = [all_starts[0]]
    for s in all_starts[1:]:
        if s - unique_starts[-1] > 500:
            unique_starts.append(s)

    # Strategy: The REAL Item 1A content section is the one followed by the most text
    # before Item 1B/2. TOC entries are followed by very little before the next item.
    best_start = None
    best_length = 0

    for start_pos in unique_starts:
        # Find end after this start
        end_pos = len(text)
        for pattern in end_patterns:
            match = pattern.search(text, start_pos + 200)
            if match:
                end_pos = min(end_pos, match.start())
                break

        section_length = end_pos - start_pos

        # Skip if too short (likely a TOC entry or heading reference)
        if section_length < 2000:
            continue

        # The longest section is most likely the actual content
        if section_length > best_length:
            best_length = section_length
            best_start = start_pos

    if best_start is None:
        # Fallback: just use the last occurrence (usually the content, not TOC)
        best_start = unique_starts[-1]

    # Find end from best start
    end_pos = len(text)
    for pattern in end_patterns:
        match = pattern.search(text, best_start + 200)
        if match:
            candidate = match.start()
            if candidate < end_pos:
                end_pos = candidate

    # Extract the section
    section_html = text[best_start:end_pos]

    # Clean HTML
    cleaned = clean_text_preserve_structure(section_html)

    # Remove any CSS/style artifacts that leak through at the beginning
    cleaned = re.sub(r'^[^A-Za-z]*(?:font-[^"]*"?>?\s*)?', '', cleaned)

    # Remove the "Item 1A. Risk Factors" header itself (may appear at the start)
    cleaned = re.sub(r'^\s*(?:Item|ITEM)\s*1A[\.\s\u2014\u2013\-:]*\s*(?:Risk\s*Factors|RISK\s*FACTORS)?\s*',
                     '', cleaned, count=1)

    # Remove any remaining page numbers or form references at the top
    cleaned = re.sub(r'^\s*\d+\s*\n', '', cleaned)
    cleaned = re.sub(r'^\s*(?:Table of Contents|INDEX)\s*\n', '', cleaned, flags=re.IGNORECASE)

    # Limit to reasonable size (some filings are enormous)
    if len(cleaned) > 150000:
        cleaned = cleaned[:150000]

    return cleaned.strip()


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
# HTML Templates
# =========================================================

HOME_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERPSA - Equity Risk Predictor</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0e17; color: #e2e8f0; min-height: 100vh; }
        .nav { background: #111827; border-bottom: 1px solid #1f2937; padding: 16px 40px; display: flex; align-items: center; justify-content: space-between; }
        .nav h1 { font-size: 20px; color: #60a5fa; }
        .nav a { color: #9ca3af; text-decoration: none; margin-left: 24px; font-size: 14px; }
        .nav a:hover { color: #60a5fa; }
        .hero { text-align: center; padding: 80px 40px 60px; max-width: 900px; margin: 0 auto; }
        .hero h2 { font-size: 42px; font-weight: 700; margin-bottom: 20px; line-height: 1.2; }
        .hero h2 span { color: #60a5fa; }
        .hero p { font-size: 18px; color: #9ca3af; line-height: 1.7; margin-bottom: 30px; }
        .hero .cta { display: inline-block; padding: 14px 36px; background: linear-gradient(135deg, #3b82f6, #2563eb); color: white; border-radius: 8px; text-decoration: none; font-weight: 600; font-size: 16px; transition: all 0.2s; }
        .hero .cta:hover { transform: translateY(-2px); box-shadow: 0 8px 25px rgba(59,130,246,0.3); }
        .how-it-works { max-width: 1000px; margin: 0 auto; padding: 60px 40px; }
        .how-it-works h3 { text-align: center; font-size: 28px; margin-bottom: 40px; color: #f1f5f9; }
        .steps { display: grid; grid-template-columns: repeat(3, 1fr); gap: 24px; }
        .step { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 28px; text-align: center; }
        .step .num { width: 40px; height: 40px; background: #1e3a5f; color: #60a5fa; border-radius: 50%; display: flex; align-items: center; justify-content: center; margin: 0 auto 16px; font-weight: 700; }
        .step h4 { color: #f1f5f9; margin-bottom: 10px; font-size: 16px; }
        .step p { color: #9ca3af; font-size: 14px; line-height: 1.6; }
        .research { max-width: 800px; margin: 0 auto; padding: 40px; }
        .research .card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 28px; margin-bottom: 20px; }
        .research .card h4 { color: #60a5fa; margin-bottom: 8px; }
        .research .card p { color: #9ca3af; font-size: 14px; line-height: 1.6; }
        .research .card .stat { font-size: 32px; font-weight: 700; color: #f59e0b; }
        .footer { text-align: center; padding: 40px; color: #4b5563; font-size: 12px; border-top: 1px solid #1f2937; margin-top: 40px; }
        @media (max-width: 768px) { .steps { grid-template-columns: 1fr; } .hero h2 { font-size: 28px; } }
    </style>
</head>
<body>
    <div class="nav">
        <h1>ERPSA</h1>
        <div>
            <a href="/">Home</a>
            <a href="/analyze">Analyze</a>
        </div>
    </div>

    <div class="hero">
        <h2>Predict Corporate Risk<br><span>Before the Numbers Show It</span></h2>
        <p>
            ERPSA reads what companies are legally forced to tell you in their SEC filings,
            detects when their language shifts from routine to alarming, and scores the probability
            of bad things happening — months before Wall Street notices.
        </p>
        <a href="/analyze" class="cta">Start Analysis</a>
    </div>

    <div class="how-it-works">
        <h3>How It Works</h3>
        <div class="steps">
            <div class="step">
                <div class="num">1</div>
                <h4>Enter a Ticker</h4>
                <p>Type any publicly-traded company's stock ticker (like AAPL, TSLA, TGT). We pull their actual 10-K filings directly from the SEC.</p>
            </div>
            <div class="step">
                <div class="num">2</div>
                <h4>Pick Two Years</h4>
                <p>Choose which years to compare. The system extracts the "Risk Factors" section from each filing and compares them word-by-word.</p>
            </div>
            <div class="step">
                <div class="num">3</div>
                <h4>See the Signals</h4>
                <p>Get a scored breakdown of every risk: what changed, how severe the language is, and what it means in plain English. High scores = danger ahead.</p>
            </div>
        </div>
    </div>

    <div class="research">
        <div class="card">
            <div class="stat">22%/year</div>
            <h4>Academic Backing: "Lazy Prices" (Harvard, 2020)</h4>
            <p>A portfolio strategy that simply buys stocks of companies with unchanged filings and sells those with changed filings earned 22% per year in abnormal returns. The research proves that textual changes predict future problems — but almost nobody reads these documents.</p>
        </div>
        <div class="card">
            <h4>Why This Works</h4>
            <p>Companies are legally required to disclose risks. They KNOW about problems before the numbers show it. But they bury the warnings in 200-page documents using dense legal language. A computer that reads everything, every year, and measures what changed — has an enormous edge over humans who just look at stock prices.</p>
        </div>
    </div>

    <div class="footer">
        ERPSA v1.0 | Built on: Cohen et al. "Lazy Prices" (2020) + Loughran & McDonald Financial Sentiment (2011)<br>
        Data from SEC EDGAR (free, public) | Not investment advice
    </div>
</body>
</html>"""



ANALYZE_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>ERPSA - Analyze</title>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: #0a0e17; color: #e2e8f0; min-height: 100vh; }
        .nav { background: #111827; border-bottom: 1px solid #1f2937; padding: 16px 40px; display: flex; align-items: center; justify-content: space-between; }
        .nav h1 { font-size: 20px; color: #60a5fa; }
        .nav a { color: #9ca3af; text-decoration: none; margin-left: 24px; font-size: 14px; }
        .nav a:hover { color: #60a5fa; }
        .container { max-width: 1100px; margin: 0 auto; padding: 40px; }
        h2 { font-size: 28px; margin-bottom: 8px; }
        .subtitle { color: #9ca3af; margin-bottom: 30px; font-size: 15px; }

        .input-section { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 28px; margin-bottom: 24px; }
        .input-section h3 { color: #60a5fa; margin-bottom: 16px; font-size: 16px; }
        .form-row { display: flex; gap: 16px; align-items: end; flex-wrap: wrap; }
        .form-group { display: flex; flex-direction: column; }
        .form-group label { font-size: 12px; color: #9ca3af; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 6px; }
        input, select { background: #0a0e17; border: 1px solid #374151; border-radius: 8px; color: #e2e8f0; padding: 10px 14px; font-size: 14px; }
        input:focus, select:focus { outline: none; border-color: #60a5fa; }
        select { min-width: 160px; }
        .btn { padding: 10px 20px; border: none; border-radius: 8px; font-size: 14px; font-weight: 600; cursor: pointer; transition: all 0.2s; }
        .btn-blue { background: #3b82f6; color: white; }
        .btn-blue:hover { background: #2563eb; }
        .btn-blue:disabled { background: #374151; color: #6b7280; cursor: not-allowed; }
        .btn-green { background: #10b981; color: white; }
        .btn-green:hover { background: #059669; }
        .btn-gray { background: #374151; color: #9ca3af; }
        .btn-gray:hover { background: #4b5563; }

        .status { margin-top: 12px; font-size: 13px; color: #9ca3af; min-height: 20px; }
        .status.error { color: #f87171; }
        .status.success { color: #34d399; }

        .results { display: none; margin-top: 30px; }
        .results.show { display: block; }
        .results-header { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 24px; margin-bottom: 20px; }
        .results-header h3 { color: #60a5fa; margin-bottom: 12px; font-size: 18px; }
        .stats-row { display: flex; gap: 16px; flex-wrap: wrap; }
        .stat-box { background: #0a0e17; border-radius: 8px; padding: 14px 20px; min-width: 130px; }
        .stat-box .value { font-size: 22px; font-weight: 700; }
        .stat-box .label { font-size: 11px; color: #9ca3af; text-transform: uppercase; }

        .risk-card { background: #111827; border: 1px solid #1f2937; border-radius: 12px; padding: 24px; margin-bottom: 16px; border-left: 5px solid #374151; transition: all 0.3s; }
        .risk-card:hover { border-color: #60a5fa; }
        .risk-card.very-high { border-left-color: #ef4444; }
        .risk-card.high { border-left-color: #f97316; }
        .risk-card.medium-high { border-left-color: #eab308; }
        .risk-card.medium { border-left-color: #a3e635; }
        .risk-card.low { border-left-color: #22c55e; }
        .risk-card .top-row { display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 12px; }
        .risk-card .title { font-size: 16px; font-weight: 600; flex: 1; margin-right: 16px; }
        .risk-card .score { font-size: 32px; font-weight: 700; line-height: 1; }
        .risk-card .score.very-high { color: #fca5a5; }
        .risk-card .score.high { color: #fdba74; }
        .risk-card .score.medium-high { color: #fde047; }
        .risk-card .score.medium { color: #d9f99d; }
        .risk-card .score.low { color: #86efac; }

        .risk-card .badges { display: flex; gap: 8px; margin-bottom: 12px; flex-wrap: wrap; }
        .badge { display: inline-block; padding: 3px 10px; border-radius: 20px; font-size: 11px; font-weight: 600; text-transform: uppercase; }
        .badge.new { background: #7f1d1d; color: #fca5a5; }
        .badge.modified { background: #78350f; color: #fdba74; }
        .badge.unchanged { background: #14532d; color: #86efac; }
        .badge.removed { background: #1f2937; color: #9ca3af; }
        .badge.level { background: #1e3a5f; color: #93c5fd; }

        .risk-card .explanation { background: #0a0e17; border-radius: 8px; padding: 16px; margin-top: 12px; font-size: 14px; line-height: 1.7; color: #d1d5db; }
        .risk-card .explanation strong { color: #fbbf24; }

        .risk-card .signals { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-top: 12px; }
        .signal-item { background: #0a0e17; padding: 8px 12px; border-radius: 6px; font-size: 12px; }
        .signal-item .name { color: #9ca3af; }
        .signal-item .val { color: #e2e8f0; font-weight: 600; }

        .bar { height: 8px; background: #1f2937; border-radius: 4px; margin-top: 10px; overflow: hidden; }
        .bar-fill { height: 100%; border-radius: 4px; transition: width 1s ease; }
        .bar-fill.very-high { background: linear-gradient(90deg, #dc2626, #f87171); }
        .bar-fill.high { background: linear-gradient(90deg, #ea580c, #fb923c); }
        .bar-fill.medium-high { background: linear-gradient(90deg, #ca8a04, #facc15); }
        .bar-fill.medium { background: linear-gradient(90deg, #65a30d, #a3e635); }
        .bar-fill.low { background: linear-gradient(90deg, #16a34a, #4ade80); }

        .loading-overlay { display: none; position: fixed; top: 0; left: 0; right: 0; bottom: 0; background: rgba(10,14,23,0.85); z-index: 1000; align-items: center; justify-content: center; flex-direction: column; }
        .loading-overlay.show { display: flex; }
        .spinner { width: 48px; height: 48px; border: 4px solid #1f2937; border-top-color: #60a5fa; border-radius: 50%; animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .loading-text { margin-top: 16px; color: #9ca3af; font-size: 14px; }

        .footer { text-align: center; padding: 30px; color: #4b5563; font-size: 12px; margin-top: 40px; }
        @media (max-width: 768px) { .form-row { flex-direction: column; } .signals { grid-template-columns: 1fr; } }
    </style>
</head>
<body>
    <div class="nav">
        <h1>ERPSA</h1>
        <div>
            <a href="/">Home</a>
            <a href="/analyze">Analyze</a>
        </div>
    </div>

    <div class="container">
        <h2>Risk Factor Analysis</h2>
        <p class="subtitle">Enter a stock ticker and select years to compare their 10-K risk factor disclosures.</p>

        <div class="input-section">
            <h3>Step 1: Look Up Company</h3>
            <div class="form-row">
                <div class="form-group">
                    <label>Stock Ticker</label>
                    <input type="text" id="ticker" placeholder="e.g. TGT, AAPL, TSLA" style="width:160px;" value="">
                </div>
                <button class="btn btn-blue" onclick="lookupCompany()">Look Up</button>
            </div>
            <div class="status" id="lookup-status"></div>
        </div>

        <div class="input-section" id="year-section" style="display:none;">
            <h3>Step 2: Select Years to Compare</h3>
            <div id="company-info" style="margin-bottom:16px;color:#9ca3af;font-size:14px;"></div>
            <div class="form-row">
                <div class="form-group">
                    <label>Current Year (newer)</label>
                    <select id="year-current"></select>
                </div>
                <div class="form-group">
                    <label>Prior Year (older)</label>
                    <select id="year-prior"></select>
                </div>
                <button class="btn btn-green" onclick="runAnalysis()">Analyze Risks</button>
            </div>
            <div class="status" id="analysis-status"></div>
        </div>

        <div class="results" id="results"></div>
    </div>

    <div class="loading-overlay" id="loading">
        <div class="spinner"></div>
        <div class="loading-text" id="loading-text">Fetching filings from SEC EDGAR...</div>
    </div>

    <div class="footer">
        ERPSA v1.0 | Data: SEC EDGAR (free, public) | Not investment advice
    </div>

    <script>
        let companyData = null;

        async function lookupCompany() {
            const ticker = document.getElementById('ticker').value.trim().toUpperCase();
            if (!ticker) { setStatus('lookup-status', 'Please enter a ticker symbol.', 'error'); return; }

            setStatus('lookup-status', 'Looking up ' + ticker + ' on SEC EDGAR...', '');
            document.getElementById('year-section').style.display = 'none';
            document.getElementById('results').classList.remove('show');

            try {
                const resp = await fetch('/api/lookup?ticker=' + encodeURIComponent(ticker));
                const data = await resp.json();
                if (data.error) { setStatus('lookup-status', data.error, 'error'); return; }

                companyData = data;
                setStatus('lookup-status', 'Found: ' + data.company + ' (CIK: ' + data.cik + ')', 'success');

                // Populate year dropdowns
                const years = data.years;
                const selCurrent = document.getElementById('year-current');
                const selPrior = document.getElementById('year-prior');
                selCurrent.innerHTML = '';
                selPrior.innerHTML = '';

                years.forEach((y, i) => {
                    const opt1 = new Option(y.year + ' (' + y.date + ')', i);
                    const opt2 = new Option(y.year + ' (' + y.date + ')', i);
                    selCurrent.add(opt1);
                    selPrior.add(opt2);
                });

                // Default: current = first, prior = second
                if (years.length >= 2) {
                    selCurrent.selectedIndex = 0;
                    selPrior.selectedIndex = 1;
                }

                document.getElementById('company-info').innerHTML =
                    '<strong>' + data.company + '</strong> (' + data.ticker + ') — ' + years.length + ' annual filings available';
                document.getElementById('year-section').style.display = 'block';
            } catch (err) {
                setStatus('lookup-status', 'Network error: ' + err.message, 'error');
            }
        }

        async function runAnalysis() {
            if (!companyData) return;

            const currentIdx = parseInt(document.getElementById('year-current').value);
            const priorIdx = parseInt(document.getElementById('year-prior').value);

            if (currentIdx === priorIdx) {
                setStatus('analysis-status', 'Please select two different years.', 'error');
                return;
            }

            showLoading('Fetching 10-K filings from SEC EDGAR... (this may take 10-30 seconds)');

            try {
                const resp = await fetch('/api/analyze', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        ticker: companyData.ticker,
                        cik: companyData.cik,
                        current_filing: companyData.filings[currentIdx],
                        prior_filing: companyData.filings[priorIdx],
                    }),
                });
                const data = await resp.json();
                hideLoading();

                if (data.error) {
                    setStatus('analysis-status', data.error, 'error');
                    return;
                }

                setStatus('analysis-status', 'Analysis complete!', 'success');
                displayResults(data);
            } catch (err) {
                hideLoading();
                setStatus('analysis-status', 'Error: ' + err.message, 'error');
            }
        }

        function displayResults(data) {
            const container = document.getElementById('results');
            const risks = data.risks || [];

            const high = risks.filter(r => r.probability >= 50);
            const med = risks.filter(r => r.probability >= 20 && r.probability < 50);
            const low = risks.filter(r => r.probability < 20 && r.status !== 'UNCHANGED');
            const unchanged = risks.filter(r => r.status === 'UNCHANGED');

            let html = `
                <div class="results-header">
                    <h3>${data.ticker} — Risk Analysis (FY${data.current_year} vs FY${data.prior_year})</h3>
                    <div class="stats-row">
                        <div class="stat-box"><div class="value">${risks.length}</div><div class="label">Risks Found</div></div>
                        <div class="stat-box"><div class="value" style="color:#fca5a5">${high.length}</div><div class="label">High Priority</div></div>
                        <div class="stat-box"><div class="value" style="color:#fdba74">${med.length}</div><div class="label">Moderate</div></div>
                        <div class="stat-box"><div class="value" style="color:#86efac">${unchanged.length}</div><div class="label">Unchanged</div></div>
                    </div>
                </div>
            `;

            const allSorted = risks.filter(r => r.status !== 'UNCHANGED').sort((a,b) => b.probability - a.probability);
            allSorted.forEach(r => { html += renderRisk(r); });

            if (unchanged.length > 0) {
                html += '<div style="margin-top:20px;padding:16px;background:#111827;border-radius:12px;border:1px solid #1f2937;">';
                html += '<h4 style="color:#22c55e;margin-bottom:8px;">Unchanged Risks (Boilerplate — No Signal)</h4>';
                html += '<p style="color:#9ca3af;font-size:13px;margin-bottom:12px;">These risks use the exact same language as last year. No change = no danger signal.</p>';
                unchanged.forEach(r => {
                    html += '<div style="padding:6px 0;font-size:13px;color:#6b7280;">• ' + r.title + '</div>';
                });
                html += '</div>';
            }

            container.innerHTML = html;
            container.classList.add('show');
            container.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }

        function renderRisk(risk) {
            const level = risk.probability >= 70 ? 'very-high' :
                          risk.probability >= 50 ? 'high' :
                          risk.probability >= 35 ? 'medium-high' :
                          risk.probability >= 20 ? 'medium' : 'low';
            return `
                <div class="risk-card ${level}">
                    <div class="top-row">
                        <div class="title">${risk.title}</div>
                        <div class="score ${level}">${risk.probability}%</div>
                    </div>
                    <div class="badges">
                        <span class="badge ${risk.status.toLowerCase()}">${risk.status}</span>
                        <span class="badge level">${risk.level}</span>
                    </div>
                    <div class="bar"><div class="bar-fill ${level}" style="width:${risk.probability}%"></div></div>
                    <div class="explanation">${risk.explanation}</div>
                    <div class="signals">
                        <div class="signal-item"><span class="name">Signal 1 (Text Changed):</span> <span class="val">${(risk.textual_score * 100).toFixed(0)}%</span></div>
                        <div class="signal-item"><span class="name">Signal 2 (Negative Tone):</span> <span class="val">${(risk.sentiment_score * 100).toFixed(0)}%</span></div>
                    </div>
                </div>
            `;
        }

        function setStatus(id, msg, cls) {
            const el = document.getElementById(id);
            el.textContent = msg;
            el.className = 'status ' + (cls || '');
        }
        function showLoading(msg) { document.getElementById('loading-text').textContent = msg; document.getElementById('loading').classList.add('show'); }
        function hideLoading() { document.getElementById('loading').classList.remove('show'); }

        // Allow Enter key on ticker input
        document.addEventListener('DOMContentLoaded', () => {
            document.getElementById('ticker').addEventListener('keypress', (e) => {
                if (e.key === 'Enter') lookupCompany();
            });
        });
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
        """Handle ticker lookup — returns available years."""
        params = parse_qs(parsed.query)
        ticker = params.get('ticker', [''])[0].strip()
        if not ticker:
            self._serve_json({'error': 'No ticker provided'})
            return

        print(f"  [LOOKUP] Looking up ticker: {ticker}")
        result = get_available_years(ticker)
        self._serve_json(result)

    def _handle_analyze(self):
        """Handle full analysis request."""
        content_length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_length).decode('utf-8')

        try:
            data = json.loads(body)
            ticker = data.get('ticker', 'UNKNOWN')
            current_filing = data.get('current_filing', {})
            prior_filing = data.get('prior_filing', {})

            print(f"  [ANALYZE] Running analysis for {ticker}")
            print(f"    Current: {current_filing.get('date', '?')}")
            print(f"    Prior: {prior_filing.get('date', '?')}")

            # Fetch filings
            print(f"  [ANALYZE] Fetching current year filing...")
            current_text = fetch_filing_text(current_filing)
            time.sleep(0.5)  # SEC rate limiting courtesy

            print(f"  [ANALYZE] Fetching prior year filing...")
            prior_text = fetch_filing_text(prior_filing)

            if current_text.startswith('[') or prior_text.startswith('['):
                self._serve_json({
                    'error': f'Could not extract risk factors. '
                             f'Current: {"OK" if not current_text.startswith("[") else current_text[:100]}. '
                             f'Prior: {"OK" if not prior_text.startswith("[") else prior_text[:100]}.'
                })
                return

            # Run pipeline
            print(f"  [ANALYZE] Running scoring pipeline...")
            clean_current = clean_text_preserve_structure(current_text)
            clean_prior = clean_text_preserve_structure(prior_text)

            sections_current = parse_risk_sections(clean_current)
            sections_prior = parse_risk_sections(clean_prior)

            print(f"    Parsed: {len(sections_current)} current sections, {len(sections_prior)} prior sections")

            matches = match_risk_categories(sections_current, sections_prior)
            change_report = classify_risk_changes(
                matches=matches,
                ticker=ticker,
                current_year=current_filing.get('year', 0),
                prior_year=prior_filing.get('year', 0),
                total_current=len(sections_current),
                total_prior=len(sections_prior),
            )

            scoring = run_scoring(change_report, verbose=False)

            # Build response with explanations
            risks = []
            for i, r in enumerate(scoring.risk_scores):
                classification = change_report.classifications[i] if i < len(change_report.classifications) else None
                explanation = generate_risk_explanation(r, classification) if classification else ""

                risks.append({
                    'title': r.title[:120],
                    'status': r.status.value,
                    'probability': round(r.preliminary_probability, 1),
                    'level': r.risk_level_label,
                    'textual_score': round(r.textual_change_score, 3),
                    'sentiment_score': round(r.sentiment_score, 3),
                    'explanation': explanation,
                })

            risks.sort(key=lambda x: x['probability'], reverse=True)

            result = {
                'ticker': ticker,
                'current_year': current_filing.get('year', 0),
                'prior_year': prior_filing.get('year', 0),
                'risks': risks,
                'total_current': len(sections_current),
                'total_prior': len(sections_prior),
            }

            print(f"  [ANALYZE] Complete. {len(risks)} risks scored.")
            self._serve_json(result)

        except Exception as e:
            import traceback
            traceback.print_exc()
            self._serve_json({'error': f'Analysis error: {str(e)}'})

    def log_message(self, format, *args):
        """Custom log format."""
        pass


# =========================================================
# Main
# =========================================================

def main():
    port = 8888
    server = HTTPServer(('0.0.0.0', port), ERPSAHandler)

    import webbrowser
    print(f"""
╔══════════════════════════════════════════════════════════════════╗
║  ERPSA — Equity Risk Predictor & Sentiment Analyzer            ║
║  Version 1.0                                                   ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                ║
║  Server running at: http://localhost:{port}                      ║
║                                                                ║
║  How to use:                                                   ║
║    1. Enter a stock ticker (AAPL, TGT, TSLA, etc.)             ║
║    2. Click "Look Up" to fetch available filing years           ║
║    3. Select two years and click "Analyze Risks"               ║
║    4. Review scored risks with plain-English explanations       ║
║                                                                ║
║  Data source: SEC EDGAR (free, public, no API key needed)      ║
║  Press Ctrl+C to stop                                          ║
╚══════════════════════════════════════════════════════════════════╝
""")
    webbrowser.open(f'http://localhost:{port}')

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nServer stopped.")
        server.server_close()


if __name__ == "__main__":
    main()
