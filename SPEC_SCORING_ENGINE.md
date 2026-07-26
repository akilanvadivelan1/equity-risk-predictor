# ERPSA Spec: Sentiment-Weighted Probability Score (0-100%)

## Overview

The Sentiment-Weighted Probability Score is the final quantitative output of ERPSA. It assigns a mathematical probability (0-100%) that a disclosed risk will materialize within the next 12 months, based on a multi-signal framework validated by academic research.

## Scoring Framework

The score combines four distinct signal families:

### Signal 1: Textual Change Magnitude

**What it measures:** How much the risk language changed year-over-year.

**Source:** ERPSA Step 2D output (MODIFIED/NEW classifications, sentence-level changes)

**Academic validation:** Cohen, Malloy & Nguyen, "Lazy Prices" (Journal of Finance, 2020). Changes to 10-K language predict future earnings, profitability, and bankruptcies. A portfolio strategy based on textual changes earns 22%+ annualized abnormal returns.

**Inputs:**
- Risk section classification (UNCHANGED / MODIFIED / NEW / REMOVED)
- Body similarity score (0.0 - 1.0)
- Number of added sentences
- Number of rewritten sentences
- Percentage of section that is new content

**Scoring logic:**
- UNCHANGED sections: base score near 0 (no signal)
- MODIFIED sections: score proportional to (1 - body_similarity) and count of changed sentences
- NEW sections: elevated base score (entirely new risk disclosure is a strong signal)

---

### Signal 2: Sentiment Direction (Loughran-McDonald Dictionary)

**What it measures:** The tone and severity of the risk language, using a finance-specific sentiment lexicon.

**Academic validation:** Loughran & McDonald, "When Is a Liability Not a Liability?" (Journal of Finance, 2011). General-purpose sentiment dictionaries misclassify 75% of negative words in financial text. The Loughran-McDonald dictionary is the standard for financial NLP.

**Key word categories to use:**
- Negative words (e.g., "adversely," "unable," "impairment," "deterioration")
- Uncertainty words (e.g., "may," "could," "uncertain," "unpredictable")
- Constraining words (e.g., "obligated," "required," "restricted," "committed")
- Litigious words (e.g., "alleged," "lawsuit," "violation," "penalty")

**Additional validation:** Bodnaruk, Loughran & McDonald, "Using 10-K Text to Gauge Financial Constraints" (JFQA, 2015). The frequency of constraining words predicts liquidity events (dividend cuts, equity issuances) better than traditional financial constraint indexes.

**Scoring logic:**
- Compute sentiment scores for the changed/new sentences specifically (not the entire filing)
- Weight negative and uncertainty words more heavily in modified sections
- Compare sentiment density (negative words per sentence) vs prior year
- A shift from neutral/positive to negative language amplifies the score

---

### Signal 3: Financial Ratio Deterioration (Campbell-Hilscher-Szilagyi Model)

**What it measures:** Quantitative financial health indicators from the company's actual P&L, balance sheet, and market data.

**Academic validation:** Campbell, Hilscher & Szilagyi, "In Search of Distress Risk" (Journal of Finance, 2008). Their hazard model using accounting and market-based variables predicts corporate failure more accurately than Altman Z-Score or Ohlson O-Score.

**Key ratios to incorporate:**
- NIMTAAVG: Net Income / Market Total Assets (profitability trend)
- TLMTA: Total Liabilities / Market Total Assets (leverage)
- CASHMTA: Cash & Short-term Investments / Market Total Assets (liquidity)
- EXRETAVG: Excess return vs market (stock momentum)
- SIGMA: Equity volatility (market risk perception)
- RSIZE: Log of firm market cap / total market cap (relative size)
- MB: Market-to-book ratio (valuation stress)
- PRICE: Log of stock price, capped (penny stock indicator)

**Scoring logic:**
- Compute each ratio for the current period
- Compare vs historical averages and industry benchmarks
- Flag ratios that have deteriorated significantly year-over-year
- A deteriorating financial profile raises the base probability

---

### Signal 4: Text-Financial Interaction (Alignment Score)

**What it measures:** Whether the textual risk signals ALIGN with deteriorating financial metrics — the most dangerous combination.

**Academic validation:**
- "Uncovering Financial Distress with Textual Risk Disclosures" (Information Systems Frontiers, 2025): LLM-based textual predictors provide incremental information BEYOND conventional accounting ratios. Combined models significantly outperform either signal alone.
- Campbell et al. (2013): Increases in risk disclosures predict increased stock volatility.
- "Attentive Options Traders" (JFQA, 2025): Textual changes predict option-implied crash risk.

**Core principle:** Text changes LEAD, numbers CONFIRM.

A textual shift that aligns with deteriorating financials is much more dangerous than either signal alone. Conversely, new risk language with stable/improving financials may indicate proactive disclosure (lower actual risk).

**Interaction matrix:**

| Text Signal | Financials Improving | Financials Stable | Financials Deteriorating |
|---|---|---|---|
| UNCHANGED | Very Low (0-10%) | Very Low (0-10%) | Low-Medium (15-35%) |
| MODIFIED (mild) | Low (5-15%) | Low-Medium (15-30%) | Medium-High (40-65%) |
| MODIFIED (severe) | Medium (20-35%) | Medium-High (35-55%) | High (60-85%) |
| NEW risk | Medium (25-40%) | High (45-65%) | Very High (70-95%) |

**Scoring logic:**
- If text signals and financial signals agree (both negative): multiply/amplify the score
- If text signals are negative but financials are stable: moderate the score (may be proactive disclosure)
- If financials deteriorate but text is unchanged: moderate concern (company may be hiding risk)
- If both are positive/stable: score near zero

---

## Final Score Computation

```
P(risk_materialization) = w1 * TextualChangeMagnitude
                        + w2 * SentimentDirection
                        + w3 * FinancialRatioDeteriation
                        + w4 * InteractionAmplifier

Where:
  w1 = 0.30 (Textual change magnitude)
  w2 = 0.25 (Sentiment direction)
  w3 = 0.25 (Financial ratio deterioration)
  w4 = 0.20 (Interaction/alignment amplifier)

Final score clamped to [0, 100]
```

**Note:** Weights are initial estimates and should be calibrated against historical outcomes (actual distress events, stock crashes, earnings misses).

---

## Calibration & Validation

The scoring system should be validated against historical case studies:

| Company | Year | Expected Score | Actual Outcome |
|---------|------|----------------|----------------|
| Target ($TGT) | 2022 | 75-85% (Inventory) | Stock dropped 24.9% in single session |
| Southwest Airlines ($LUV) | 2022 | 60-75% (Operations) | Massive holiday meltdown, operational crisis |
| Silicon Valley Bank ($SIVB) | 2022 | 80-90% (Interest Rate) | Total bank failure March 2023 |
| Bed Bath & Beyond ($BBBY) | 2022 | 85-95% (Going Concern) | Bankruptcy filed Feb 2023 |

---

## Academic References

1. Cohen, L., Malloy, C., & Nguyen, Q. (2020). "Lazy Prices." Journal of Finance, 75(3), 1371-1415.
2. Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability?" Journal of Finance, 66(1), 35-65.
3. Campbell, J.Y., Hilscher, J., & Szilagyi, J. (2008). "In Search of Distress Risk." Journal of Finance, 63(6), 2899-2939.
4. Bodnaruk, A., Loughran, T., & McDonald, B. (2015). "Using 10-K Text to Gauge Financial Constraints." JFQA, 50(4), 623-646.
5. Campbell, J.L., et al. (2013). "Textual Risk Disclosures and Investors' Risk Perceptions." Review of Accounting Studies, 19, 396-455.
6. Jiang, L., et al. (2025). "Uncovering Financial Distress with Textual Risk Disclosures in Annual Reports: Insights from Large Language Models." Information Systems Frontiers.
7. Li, F. (2008). "Annual Report Readability, Current Earnings, and Earnings Persistence." Journal of Accounting and Economics, 45(2-3), 221-247.
8. "Attentive Options Traders: Textual Changes to 10-Ks and Option Volatility Smirk" (2025). JFQA.

---

## Implementation Priority

1. **Phase 1 (Current):** Textual Change Magnitude — DONE (Step 2D)
2. **Phase 2 (Next):** Sentiment Direction — Implement Loughran-McDonald scoring on extracted sentences
3. **Phase 3:** Financial Ratio Integration — Pull financial data via API, compute Campbell-Hilscher ratios
4. **Phase 4:** Interaction Model — Combine all signals with calibrated weights
5. **Phase 5:** Backtesting — Validate against historical distress events
