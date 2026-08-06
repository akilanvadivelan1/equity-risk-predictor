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
| `step_2d_risk_analysis.py` | Step 2D pipeline entry point (change detection) |
| `lm_dictionary.py` | Loughran-McDonald financial sentiment dictionary (1,000+ words) |
| `sentiment_scorer.py` | Sentiment scoring engine (Signal 2: tone & severity) |
| `textual_change_scorer.py` | Textual change magnitude scorer (Signal 1) |
| `step_3_scoring.py` | Step 3 combined scoring pipeline (Signals 1+2 → probability) |
| `test_scoring_engine.py` | Verification tests for the scoring engine |

### Classification System

- UNCHANGED — Boilerplate language retained, no action needed
- MODIFIED — Same risk topic, but language has shifted (analyze further)
- NEW — Brand new risk disclosure (high priority signal)
- REMOVED — Risk dropped from prior year's filing

## Quick Start

```python
from step_3_scoring import run_full_analysis

# Full pipeline: text cleaning → parsing → matching →
# classification → textual scoring → sentiment scoring → probability
scoring = run_full_analysis(
    raw_text_current=raw_text_current,
    raw_text_prior=raw_text_prior,
    ticker="TGT",
    current_year=2022,
    prior_year=2021
)

# View high-priority risks
print(scoring.summary())
for risk in scoring.high_priority_risks:
    print(f"{risk.title}: {risk.preliminary_probability:.1f}% ({risk.risk_level_label})")
```

### Step-by-Step Usage

```python
# Step 2D: Detect and classify risk changes
from step_2d_risk_analysis import run_risk_analysis

change_report = run_risk_analysis(
    raw_text_current=raw_text_current,
    raw_text_prior=raw_text_prior,
    ticker="TGT",
    current_year=2022,
    prior_year=2021
)

# Step 3: Score risks for probability
from step_3_scoring import run_scoring

scoring = run_scoring(change_report)
print(scoring.summary())
```

### Standalone Sentiment Analysis

```python
from sentiment_scorer import score_text_sentiment

score = score_text_sentiment(
    "We may be unable to prevent significant data breaches that "
    "could adversely affect our operations."
)
print(f"Negative density: {score.negative_density:.3f}")
print(f"Uncertainty density: {score.uncertainty_density:.3f}")
print(f"Heavily negative? {score.is_heavily_negative}")
```

## Requirements

- Python 3.9+
- `edgartools` — SEC EDGAR API access
- No external NLP dependencies for core analysis (pure Python)

## Running Tests

```bash
python3 test_scoring_engine.py
```

Runs 4 test suites verifying dictionary classification, sentiment ordering, full pipeline scoring with realistic scenarios, and edge cases.

## Scoring Framework

The risk probability score combines multiple signals validated by academic research:

| Signal | Weight | Source | Status |
|--------|--------|--------|--------|
| **Signal 1:** Textual Change Magnitude | 0.30 | Cohen et al. "Lazy Prices" (2020) | ✅ Implemented |
| **Signal 2:** Sentiment Direction | 0.25 | Loughran & McDonald (2011) | ✅ Implemented |
| **Signal 3:** Financial Ratio Deterioration | 0.25 | Campbell, Hilscher & Szilagyi (2008) | ⬜ Planned |
| **Signal 4:** Text-Financial Interaction | 0.20 | Jiang et al. (2025) | ⬜ Planned |

**Current output (Signals 1+2):** Preliminary probability score (0-85%) indicating risk materialization likelihood based on textual analysis alone.

## Project Status

- [x] SEC EDGAR connection & ticker lookup (Step 2A)
- [x] Item 1A extraction & cleaning (Step 2B)
- [x] Year-over-year text comparison (Step 2C)
- [x] Topic-aware risk change analysis (Step 2D - Improved)
- [x] Sentence-level change extraction (Step 2E)
- [x] Loughran-McDonald sentiment dictionary (1,000+ finance-specific words)
- [x] Sentiment scoring engine (Signal 2)
- [x] Textual change magnitude scoring (Signal 1)
- [x] Combined probability scoring (Step 3 — Signals 1+2)
- [ ] Financial ratio integration (Signal 3 — Campbell-Hilscher model)
- [ ] Text-financial interaction model (Signal 4)
- [ ] Multi-year trend analysis
- [ ] Backtesting against historical distress events
- [ ] Dashboard / visualization

## License

MIT
