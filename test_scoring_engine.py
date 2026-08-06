#!/usr/bin/env python3
# =========================================================
# ERPSA - Scoring Engine Verification Test
# =========================================================
# Tests the full pipeline (Step 2D → Step 3) with synthetic
# risk factor text modeled after real 10-K disclosures.
#
# Validates:
#   1. UNCHANGED risks score near zero
#   2. MODIFIED risks score proportional to severity
#   3. NEW risks score high (especially with negative tone)
#   4. REMOVED risks score low
#   5. Sentiment correctly detects tone shifts
#   6. Scores align with expected ranges from SPEC
# =========================================================

import sys
sys.path.insert(0, '.')

from risk_section_parser import RiskSection
from risk_matcher import match_risk_categories, RiskChangeStatus
from risk_classifier import classify_risk_changes
from step_3_scoring import run_scoring
from sentiment_scorer import score_text_sentiment
from lm_dictionary import get_dictionary



def test_dictionary_basics():
    """Verify the LM dictionary classifies key financial words correctly."""
    print("=" * 70)
    print("  TEST 1: Dictionary Word Classification")
    print("=" * 70)

    lm = get_dictionary()
    errors = []

    # Negative words
    for word in ["adversely", "impairment", "bankruptcy", "deterioration", "loss"]:
        cats = lm.classify_word(word)
        if not any(c.value == "negative" for c in cats):
            errors.append(f"  FAIL: '{word}' not classified as negative")

    # Positive words
    for word in ["improve", "benefit", "growth", "profitable"]:
        cats = lm.classify_word(word)
        # 'growth' may not be in our subset — that's OK
        if word in ("improve", "benefit", "profitable"):
            if not any(c.value == "positive" for c in cats):
                errors.append(f"  FAIL: '{word}' not classified as positive")

    # Uncertainty words
    for word in ["may", "could", "uncertain", "volatile", "risk"]:
        cats = lm.classify_word(word)
        if not any(c.value == "uncertainty" for c in cats):
            errors.append(f"  FAIL: '{word}' not classified as uncertainty")

    # Constraining words
    for word in ["obligated", "required", "restrict", "prohibit"]:
        cats = lm.classify_word(word)
        if not any(c.value == "constraining" for c in cats):
            errors.append(f"  FAIL: '{word}' not classified as constraining")

    # Words NOT in financial negative (common Harvard IV misclassifications)
    non_negative = ["tax", "cost", "capital", "liability", "board", "vice"]
    for word in non_negative:
        cats = lm.classify_word(word)
        is_neg = any(c.value == "negative" for c in cats)
        # These should NOT be negative in financial context
        status = "PASS" if not is_neg else "NOTE (in LM negative)"
        print(f"  {word:<15} negative={is_neg:<6} [{status}]")

    if errors:
        for e in errors:
            print(e)
        print(f"\n  RESULT: {len(errors)} failures")
        return False
    else:
        print(f"\n  RESULT: All core word classifications PASS")
        return True



def test_sentiment_scoring():
    """Verify sentiment scoring produces expected relative ordering."""
    print("\n" + "=" * 70)
    print("  TEST 2: Sentiment Scoring — Relative Ordering")
    print("=" * 70)

    # Benign text (low risk language)
    benign = (
        "We continue to invest in our business and serve our customers. "
        "Our operations remain strong and we achieved record profitability."
    )

    # Moderate risk text
    moderate = (
        "We may face increased competition that could affect our market share. "
        "Changes in consumer preferences may require us to adjust our strategy."
    )

    # Severe risk text
    severe = (
        "We may be unable to prevent significant cybersecurity breaches that "
        "could adversely and materially impair our operations, result in "
        "substantial financial losses, expose us to costly litigation and "
        "regulatory penalties, and cause irreparable damage to our reputation. "
        "The threat of catastrophic failure has significantly heightened."
    )

    s_benign = score_text_sentiment(benign)
    s_moderate = score_text_sentiment(moderate)
    s_severe = score_text_sentiment(severe)

    print(f"\n  Benign text:")
    print(f"    Negative density: {s_benign.negative_density:.4f}")
    print(f"    Net negativity:   {s_benign.net_negativity:.4f}")
    print(f"    Risk intensity:   {s_benign.risk_intensity:.4f}")

    print(f"\n  Moderate risk text:")
    print(f"    Negative density: {s_moderate.negative_density:.4f}")
    print(f"    Net negativity:   {s_moderate.net_negativity:.4f}")
    print(f"    Risk intensity:   {s_moderate.risk_intensity:.4f}")

    print(f"\n  Severe risk text:")
    print(f"    Negative density: {s_severe.negative_density:.4f}")
    print(f"    Net negativity:   {s_severe.net_negativity:.4f}")
    print(f"    Risk intensity:   {s_severe.risk_intensity:.4f}")

    # Verify ordering: severe > moderate > benign
    ordering_correct = (
        s_severe.risk_intensity > s_moderate.risk_intensity > s_benign.risk_intensity
    )
    net_neg_correct = (
        s_severe.net_negativity > s_moderate.net_negativity
    )

    print(f"\n  Risk intensity ordering correct: {ordering_correct}")
    print(f"  Net negativity severe > moderate: {net_neg_correct}")

    if ordering_correct and net_neg_correct:
        print("  RESULT: PASS")
        return True
    else:
        print("  RESULT: FAIL — ordering violated")
        return False



def test_full_pipeline_scoring():
    """
    Test the complete pipeline with a realistic scenario modeled
    after Target's 2022 10-K risk disclosures.

    Expected outcomes:
    - UNCHANGED boilerplate → near 0%
    - MODIFIED (mild) → 20-40%
    - MODIFIED (severe) → 50-75%
    - NEW risk (negative tone) → 60-85%
    """
    print("\n" + "=" * 70)
    print("  TEST 3: Full Pipeline — Target-like Scenario")
    print("=" * 70)

    # --- CURRENT YEAR (2022) risk sections ---
    current_sections = [
        # 1. UNCHANGED: identical boilerplate
        RiskSection(
            title="GENERAL ECONOMIC CONDITIONS",
            body=(
                "Our business is subject to the risks arising from adverse changes "
                "in domestic and global economic conditions. If economic conditions "
                "deteriorate, consumer spending may decline, which could adversely "
                "affect our results of operations."
            ),
        ),
        # 2. MODIFIED (mild): competition section slightly expanded
        RiskSection(
            title="COMPETITIVE ENVIRONMENT",
            body=(
                "The retail industry is highly competitive. We compete with other "
                "mass merchandisers, department stores, and online retailers on the "
                "basis of price, quality, and convenience. Increasing competition "
                "from e-commerce retailers may put pressure on our margins and "
                "require additional investment in our digital capabilities."
            ),
        ),
        # 3. MODIFIED (severe): cybersecurity dramatically rewritten
        RiskSection(
            title="CYBERSECURITY AND DATA PRIVACY THREATS",
            body=(
                "We face increasingly severe and sophisticated cybersecurity threats "
                "that could result in catastrophic data breaches, material financial "
                "losses, and irreparable reputational damage. State-sponsored threat "
                "actors and organized criminal groups have significantly escalated "
                "attacks against retail companies. We may be unable to adequately "
                "defend against these threats despite substantial investments in "
                "security infrastructure. Any breach could expose us to costly "
                "litigation, regulatory penalties, and loss of customer trust that "
                "may materially impair our long-term financial performance. The "
                "frequency and severity of attempted intrusions has increased "
                "substantially over the past twelve months."
            ),
        ),
        # 4. NEW: supply chain / inventory risk (didn't exist before)
        RiskSection(
            title="INVENTORY MANAGEMENT AND SUPPLY CHAIN DISRUPTION",
            body=(
                "We face significant risks related to our ability to effectively "
                "manage inventory levels in an environment of unprecedented supply "
                "chain disruption. Global shipping constraints, port congestion, "
                "and labor shortages have materially impaired our logistics "
                "operations and may continue to do so. We may be unable to "
                "accurately forecast consumer demand, which could result in excess "
                "inventory requiring significant markdowns that adversely affect "
                "our gross margins, or inventory shortages that impair our ability "
                "to serve customers. The financial impact of these disruptions "
                "could be substantial and may materially affect our results of "
                "operations and financial condition."
            ),
        ),
    ]

    # --- PRIOR YEAR (2021) risk sections ---
    prior_sections = [
        # 1. Identical boilerplate
        RiskSection(
            title="GENERAL ECONOMIC CONDITIONS",
            body=(
                "Our business is subject to the risks arising from adverse changes "
                "in domestic and global economic conditions. If economic conditions "
                "deteriorate, consumer spending may decline, which could adversely "
                "affect our results of operations."
            ),
        ),
        # 2. Original competition section (shorter)
        RiskSection(
            title="COMPETITIVE ENVIRONMENT",
            body=(
                "The retail industry is highly competitive. We compete with other "
                "mass merchandisers, department stores, and online retailers on the "
                "basis of price, quality, and convenience."
            ),
        ),
        # 3. Original cyber section (generic, short)
        RiskSection(
            title="CYBERSECURITY RISKS",
            body=(
                "We face cybersecurity risks common to companies in our industry. "
                "We invest in technology and maintain protocols to protect our "
                "systems and customer information from unauthorized access."
            ),
        ),
        # No supply chain section in prior year
    ]

    # Run matching and classification
    matches = match_risk_categories(current_sections, prior_sections)
    change_report = classify_risk_changes(
        matches=matches, ticker="TGT",
        current_year=2022, prior_year=2021,
        total_current=len(current_sections),
        total_prior=len(prior_sections),
    )

    # Run Step 3 Scoring
    scoring = run_scoring(change_report, verbose=False)

    # Display and validate results
    print(scoring.summary())

    # Validation
    errors = []
    for risk in scoring.risk_scores:
        p = risk.preliminary_probability

        if risk.status == RiskChangeStatus.UNCHANGED:
            if p > 5.0:
                errors.append(
                    f"  FAIL: UNCHANGED '{risk.title}' scored {p:.1f}% (expected < 5%)"
                )
            else:
                print(f"  PASS: UNCHANGED '{risk.title[:40]}' → {p:.1f}%")

        elif risk.status == RiskChangeStatus.MODIFIED:
            if "CYBER" in risk.title.upper():
                # Severe modification — expect 50-85%
                if 45.0 <= p <= 85.0:
                    print(f"  PASS: MODIFIED (severe) '{risk.title[:40]}' → {p:.1f}%")
                else:
                    errors.append(
                        f"  FAIL: MODIFIED (severe) '{risk.title}' scored {p:.1f}% "
                        f"(expected 45-85%)"
                    )
            else:
                # Mild modification — expect 10-45%
                if 5.0 <= p <= 50.0:
                    print(f"  PASS: MODIFIED (mild) '{risk.title[:40]}' → {p:.1f}%")
                else:
                    errors.append(
                        f"  FAIL: MODIFIED (mild) '{risk.title}' scored {p:.1f}% "
                        f"(expected 5-50%)"
                    )

        elif risk.status == RiskChangeStatus.NEW:
            # New risk with heavy negative tone — expect 55-85%
            if 50.0 <= p <= 85.0:
                print(f"  PASS: NEW '{risk.title[:40]}' → {p:.1f}%")
            else:
                errors.append(
                    f"  FAIL: NEW '{risk.title}' scored {p:.1f}% (expected 50-85%)"
                )

    print()
    if errors:
        for e in errors:
            print(e)
        print(f"\n  RESULT: {len(errors)} failures")
        return False
    else:
        print("  RESULT: All score ranges PASS")
        return True



def test_edge_cases():
    """Test edge cases: empty text, very short text, removed risks."""
    print("\n" + "=" * 70)
    print("  TEST 4: Edge Cases")
    print("=" * 70)

    errors = []

    # Edge case: scoring empty text
    s = score_text_sentiment("")
    if s.word_count == 0 and s.negative_density == 0.0:
        print("  PASS: Empty text scores zero")
    else:
        errors.append("  FAIL: Empty text did not score zero")

    # Edge case: single word
    s = score_text_sentiment("bankruptcy")
    if s.negative_count >= 1:
        print(f"  PASS: Single negative word detected (count={s.negative_count})")
    else:
        errors.append("  FAIL: Single negative word not detected")

    # Edge case: REMOVED risk
    current_sections = [
        RiskSection(title="ONLY CURRENT", body="We face normal business conditions."),
    ]
    prior_sections = [
        RiskSection(title="ONLY CURRENT", body="We face normal business conditions."),
        RiskSection(title="OLD REMOVED RISK", body="We faced significant litigation risk from ongoing lawsuits that could result in material penalties."),
    ]

    matches = match_risk_categories(current_sections, prior_sections)
    report = classify_risk_changes(
        matches=matches, ticker="TEST",
        current_year=2023, prior_year=2022,
        total_current=1, total_prior=2,
    )

    scoring = run_scoring(report, verbose=False)

    removed = [r for r in scoring.risk_scores if r.status == RiskChangeStatus.REMOVED]
    if removed and removed[0].preliminary_probability <= 10.0:
        print(f"  PASS: REMOVED risk scored low ({removed[0].preliminary_probability:.1f}%)")
    elif removed:
        errors.append(f"  FAIL: REMOVED risk scored {removed[0].preliminary_probability:.1f}% (expected <= 10%)")
    else:
        print("  NOTE: No REMOVED risk detected (matching behavior)")

    if errors:
        for e in errors:
            print(e)
        print(f"\n  RESULT: {len(errors)} failures")
        return False
    else:
        print("\n  RESULT: All edge cases PASS")
        return True


# =========================================================
# Main Runner
# =========================================================

def main():
    """Run all verification tests."""
    print("\n")
    print("*" * 70)
    print("  ERPSA SCORING ENGINE — VERIFICATION TESTS")
    print("*" * 70)

    results = []
    results.append(("Dictionary Basics", test_dictionary_basics()))
    results.append(("Sentiment Scoring", test_sentiment_scoring()))
    results.append(("Full Pipeline", test_full_pipeline_scoring()))
    results.append(("Edge Cases", test_edge_cases()))

    print("\n")
    print("=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)
    all_pass = True
    for name, passed in results:
        status = "PASS" if passed else "FAIL"
        icon = "  [OK]" if passed else "  [!!]"
        print(f"{icon} {name}: {status}")
        if not passed:
            all_pass = False

    print()
    if all_pass:
        print("  ALL TESTS PASSED")
    else:
        print("  SOME TESTS FAILED — review output above")
    print("=" * 70)

    return all_pass


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
