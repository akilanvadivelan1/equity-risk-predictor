# =========================================================
# STEP 2D (IMPROVED): Topic-Aware Risk Change Analysis
# =========================================================
# Replaces the original Step 2D blind text-diff approach.
#
# NEW APPROACH:
#   1. Parse Item 1A into titled risk sections (by topic)
#   2. Match risk categories across years (by semantic similarity)
#   3. Classify each as UNCHANGED / MODIFIED / NEW / REMOVED
#   4. For MODIFIED risks: show exactly which sentences changed
#   5. For NEW risks: highlight key sentences
# =========================================================

import sys
sys.path.insert(0, '.')

from text_cleaning import clean_text, clean_text_preserve_structure
from risk_section_parser import parse_risk_sections, RiskSection
from risk_matcher import match_risk_categories, RiskChangeStatus
from risk_classifier import classify_risk_changes, RiskChangeReport


def run_risk_analysis(
    raw_text_current: str,
    raw_text_prior: str,
    ticker: str = "UNKNOWN",
    current_year: int = 0,
    prior_year: int = 0,
    verbose: bool = True
) -> RiskChangeReport:
    """
    Complete topic-aware risk change analysis pipeline.

    Args:
        raw_text_current: Raw Item 1A HTML/text from the current year's 10-K.
        raw_text_prior: Raw Item 1A HTML/text from the prior year's 10-K.
        ticker: Stock ticker symbol (for report labeling).
        current_year: Filing year for current 10-K.
        prior_year: Filing year for prior 10-K.
        verbose: If True, prints progress and results to console.

    Returns:
        RiskChangeReport with full classification details.
    """
    # PHASE 1: Clean and Parse
    if verbose:
        print("=" * 80)
        print(f"  STEP 2D: TOPIC-AWARE RISK CHANGE ANALYSIS ({ticker})")
        print(f"  Comparing FY{current_year} vs FY{prior_year}")
        print("=" * 80)
        print(f"\n[Phase 1] Cleaning text and parsing risk sections...")

    clean_current = clean_text_preserve_structure(raw_text_current)
    clean_prior = clean_text_preserve_structure(raw_text_prior)

    sections_current = parse_risk_sections(clean_current)
    sections_prior = parse_risk_sections(clean_prior)

    if verbose:
        print(f"  -> Current Year ({current_year}): {len(sections_current)} risk sections")
        print(f"  -> Prior Year   ({prior_year}): {len(sections_prior)} risk sections")
        print(f"\n  Current Year Risk Categories:")
        for i, s in enumerate(sections_current[:10], 1):
            print(f"    [{i:2d}] {s.title[:70]}{'...' if len(s.title) > 70 else ''}")
        if len(sections_current) > 10:
            print(f"    ... and {len(sections_current) - 10} more")

    # PHASE 2: Match Categories
    if verbose:
        print(f"\n[Phase 2] Matching risk categories across years...")

    matches = match_risk_categories(sections_current, sections_prior)

    if verbose:
        matched = sum(1 for m in matches if m.status in (RiskChangeStatus.UNCHANGED, RiskChangeStatus.MODIFIED))
        new = sum(1 for m in matches if m.status == RiskChangeStatus.NEW)
        removed = sum(1 for m in matches if m.status == RiskChangeStatus.REMOVED)
        print(f"  -> Matched: {matched} | New: {new} | Removed: {removed}")

    # PHASE 3: Classify
    if verbose:
        print(f"\n[Phase 3] Classifying risk changes...")

    report = classify_risk_changes(
        matches=matches, ticker=ticker,
        current_year=current_year, prior_year=prior_year,
        total_current=len(sections_current), total_prior=len(sections_prior)
    )

    # PHASE 4: Display
    if verbose:
        _print_report(report)

    return report



def _print_report(report: RiskChangeReport):
    """Print formatted report to console."""
    print("\n")
    print(report.summary())

    if report.modified_risks:
        print("\n" + "=" * 80)
        print(f"  MODIFIED RISKS — Language Shifted ({len(report.modified_risks)})")
        print("=" * 80)
        for idx, risk in enumerate(report.modified_risks, 1):
            print(f"\n  [{idx}] {risk.title}")
            print(f"      Body Similarity: {risk.body_similarity * 100:.1f}%")
            print(f"      {risk.change_summary}")
            if risk.changed_sentences:
                print(f"      Specific Changes:")
                for change in risk.changed_sentences[:5]:
                    icon = "[NEW]" if change.change_type == "added" else "[REWRITTEN]"
                    print(f"        {icon} (match: {change.similarity_to_prior * 100:.0f}%)")
                    print(f"          \"{change.sentence[:150]}{'...' if len(change.sentence) > 150 else ''}\"")
                    if change.prior_sentence and change.change_type == "rewritten":
                        print(f"          Was: \"{change.prior_sentence[:150]}{'...' if len(change.prior_sentence) > 150 else ''}\"")
                if len(risk.changed_sentences) > 5:
                    print(f"        ... and {len(risk.changed_sentences) - 5} more")
            print(f"      {'─' * 60}")

    if report.new_risks:
        print("\n" + "=" * 80)
        print(f"  NEW RISKS — Brand New Disclosures ({len(report.new_risks)})")
        print("=" * 80)
        for idx, risk in enumerate(report.new_risks, 1):
            print(f"\n  [{idx}] {risk.title}")
            print(f"      {risk.change_summary}")
            if risk.key_sentences:
                print(f"      Key Sentences:")
                for s_idx, sent in enumerate(risk.key_sentences[:3], 1):
                    print(f"        [{s_idx}] \"{sent[:200]}{'...' if len(sent) > 200 else ''}\"")
                if len(risk.key_sentences) > 3:
                    print(f"        ... and {len(risk.key_sentences) - 3} more")
            print(f"      {'─' * 60}")

    if report.removed_risks:
        print("\n" + "=" * 80)
        print(f"  REMOVED RISKS — Dropped From Filing ({len(report.removed_risks)})")
        print("=" * 80)
        for idx, risk in enumerate(report.removed_risks, 1):
            print(f"  [{idx}] {risk.title}")

    if report.unchanged_risks:
        print("\n" + "-" * 80)
        print(f"  UNCHANGED (boilerplate, filtered out): {len(report.unchanged_risks)}")
        print("-" * 80)
        for idx, risk in enumerate(report.unchanged_risks[:5], 1):
            print(f"  [{idx}] {risk.title} (similarity: {risk.body_similarity * 100:.0f}%)")
        if len(report.unchanged_risks) > 5:
            print(f"  ... and {len(report.unchanged_risks) - 5} more")

    print("\n" + "=" * 80)
    print("  ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    print("Step 2D Risk Analysis Module loaded successfully.")
    print("Use run_risk_analysis() with raw Item 1A text from two years.")
