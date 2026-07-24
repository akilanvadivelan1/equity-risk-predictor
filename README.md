# ERPSA — Equity Risk Predictor & Sentiment Analyzer

An AI-driven text analysis application that ingests unstructured financial texts (SEC 10-K filings), filters out repetitive boilerplate, isolates corporate and systemic shifts in risk language, and assigns a **Sentiment-Weighted Probability Score (0-100%)** to risks materializing within the next 12 months.

## Problem Statement

Institutional investors spend thousands of hours manually reading corporate filings. Publicly traded companies disclose everything that might harm their business in **Item 1A: Risk Factors** of their annual 10-K filings. However, much of this text is legally obligated boilerplate repeated year after year. Human analysts suffer from "disclosure fatigue" and miss subtle changes in vocabulary that signal an impending crisis.

## How It Works

### Pipeline Architecture

```
SEC EDGAR API -> 10-K Filing -> Item 1A Extraction -> Text Cleaning
    -> Risk Section Parsing (by topic)
    -> Cross-Year Risk Matching (semantic alignment)
    -> Change Classification (UNCHANGED / MODIFIED / NEW / REMOVED)
    -> Sentence-Level Change Extraction
    -> Sentiment Scoring -> Risk Probability (0-100%)
```

### Key Modules

| Module | Purpose |
|--------|---------|
| `text_cleaning.py` | HTML stripping & text normalization |
| `risk_section_parser.py` | Parses Item 1A into titled risk sections by topic |
| `risk_matcher.py` | Matches risk categories across filing years |
| `risk_classifier.py` | Classifies changes & extracts sentence-level modifications |
| `step_2d_risk_analysis.py` | Unified pipeline entry point |

### Classification System

- UNCHANGED — Boilerplate language retained, no action needed
- MODIFIED — Same risk topic, but language has shifted (analyze further)
- NEW — Brand new risk disclosure (high priority signal)
- REMOVED — Risk dropped from prior year's filing

## Quick Start

```python
from step_2d_risk_analysis import run_risk_analysis

report = run_risk_analysis(
    raw_text_current=raw_text_current,
    raw_text_prior=raw_text_prior,
    ticker="TGT",
    current_year=2022,
    prior_year=2021
)

print(report.summary())
for risk in report.modified_risks:
    print(f"MODIFIED: {risk.title}: {risk.change_summary}")
for risk in report.new_risks:
    print(f"NEW: {risk.title}: {risk.change_summary}")
```

## Requirements

- Python 3.9+
- `edgartools` — SEC EDGAR API access
- No external NLP dependencies for core analysis (pure Python)

## Project Status

- [x] SEC EDGAR connection & ticker lookup (Step 2A)
- [x] Item 1A extraction & cleaning (Step 2B)
- [x] Year-over-year text comparison (Step 2C)
- [x] Topic-aware risk change analysis (Step 2D - Improved)
- [x] Sentence-level change extraction (Step 2E)
- [ ] Sentiment scoring engine
- [ ] Risk probability calculation (0-100%)
- [ ] Multi-year trend analysis
- [ ] Dashboard / visualization

## License

MIT
