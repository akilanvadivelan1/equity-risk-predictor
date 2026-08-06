# =========================================================
# STEP 3: Combined Risk Scoring Pipeline
# =========================================================
# Integrates Signal 1 (Textual Change Magnitude) and
# Signal 2 (Sentiment Direction) into a combined preliminary
# risk probability score.
#
# This module sits downstream of Step 2D and produces the
# first quantitative risk estimate for each identified risk.
#
# Future phases will add Signal 3 (Financial Ratios) and
# Signal 4 (Text-Financial Interaction) to produce the
# final 0-100% probability score.
# =========================================================

import sys
sys.path.insert(0, '.')

from dataclasses import dataclass, field
from typing import List, Optional

from risk_classifier import RiskChangeReport, RiskClassification
from risk_matcher import RiskChangeStatus
from textual_change_scorer import (
    score_textual_changes,
    TextualChangeReport,
    TextualChangeScore,
)
from sentiment_scorer import (
    score_risk_sentiment,
    SentimentReport,
    RiskSentimentResult,
)



# =========================================================
# Data Classes for Combined Score
# =========================================================

@dataclass
class RiskProbabilityScore:
    """Combined risk probability score for a single risk section."""
    title: str
    status: RiskChangeStatus

    # Signal 1: Textual Change Magnitude (0.0 - 1.0)
    textual_change_score: float = 0.0

    # Signal 2: Sentiment Direction (0.0 - 1.0)
    sentiment_score: float = 0.0

    # Signal 3: Financial Ratio Deterioration (placeholder, 0.0 - 1.0)
    financial_score: float = 0.0

    # Signal 4: Text-Financial Interaction (placeholder, 0.0 - 1.0)
    interaction_score: float = 0.0

    # Combined preliminary probability (0 - 100%)
    preliminary_probability: float = 0.0

    # Risk level classification
    risk_level: str = ""

    # Detail references
    textual_detail: Optional[TextualChangeScore] = None
    sentiment_detail: Optional[RiskSentimentResult] = None

    @property
    def risk_level_label(self) -> str:
        """Human-readable risk level based on probability."""
        p = self.preliminary_probability
        if p >= 70:
            return "VERY HIGH"
        elif p >= 50:
            return "HIGH"
        elif p >= 35:
            return "MEDIUM-HIGH"
        elif p >= 20:
            return "MEDIUM"
        elif p >= 10:
            return "LOW-MEDIUM"
        else:
            return "LOW"

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"  [{self.status.value}] {self.title}",
            f"      Preliminary Probability: {self.preliminary_probability:.1f}% ({self.risk_level_label})",
            f"      Signal 1 (Textual Change): {self.textual_change_score:.3f}",
            f"      Signal 2 (Sentiment):      {self.sentiment_score:.3f}",
        ]
        if self.financial_score > 0:
            lines.append(f"      Signal 3 (Financial):      {self.financial_score:.3f}")
        if self.interaction_score > 0:
            lines.append(f"      Signal 4 (Interaction):    {self.interaction_score:.3f}")
        return '\n'.join(lines)



@dataclass
class ScoringReport:
    """Complete scoring report combining all signals."""
    ticker: str
    current_year: int
    prior_year: int
    risk_scores: List[RiskProbabilityScore] = field(default_factory=list)

    # Sub-reports for drill-down
    textual_report: Optional[TextualChangeReport] = None
    sentiment_report: Optional[SentimentReport] = None

    @property
    def actionable_risks(self) -> List[RiskProbabilityScore]:
        """Risks with probability >= 20% (worth investigating)."""
        return [r for r in self.risk_scores if r.preliminary_probability >= 20.0]

    @property
    def high_priority_risks(self) -> List[RiskProbabilityScore]:
        """Risks with probability >= 50% (significant concern)."""
        return [r for r in self.risk_scores if r.preliminary_probability >= 50.0]

    @property
    def top_risk(self) -> Optional[RiskProbabilityScore]:
        """The highest-scored risk."""
        if not self.risk_scores:
            return None
        return max(self.risk_scores, key=lambda r: r.preliminary_probability)

    def summary(self) -> str:
        """Full formatted report."""
        lines = [
            "",
            "=" * 80,
            f"  STEP 3: RISK PROBABILITY SCORING — {self.ticker}",
            f"  Comparing: FY{self.current_year} vs FY{self.prior_year}",
            f"  Signals Active: 1 (Textual Change) + 2 (Sentiment Direction)",
            f"  Signals Pending: 3 (Financial Ratios) + 4 (Interaction)",
            "=" * 80,
            "",
            f"  Total Risks Scored: {len(self.risk_scores)}",
            f"  Actionable (>= 20%): {len(self.actionable_risks)}",
            f"  High Priority (>= 50%): {len(self.high_priority_risks)}",
            "",
        ]

        if self.high_priority_risks:
            lines.append("  " + "-" * 70)
            lines.append("  HIGH PRIORITY RISKS:")
            lines.append("  " + "-" * 70)
            for risk in sorted(self.high_priority_risks,
                             key=lambda r: r.preliminary_probability, reverse=True):
                lines.append(risk.summary())
                lines.append("")

        actionable_only = [r for r in self.actionable_risks
                          if r.preliminary_probability < 50.0]
        if actionable_only:
            lines.append("  " + "-" * 70)
            lines.append("  MODERATE RISKS (Worth Monitoring):")
            lines.append("  " + "-" * 70)
            for risk in sorted(actionable_only,
                             key=lambda r: r.preliminary_probability, reverse=True):
                lines.append(risk.summary())
                lines.append("")

        low_risks = [r for r in self.risk_scores
                    if r.preliminary_probability < 20.0
                    and r.status != RiskChangeStatus.UNCHANGED]
        if low_risks:
            lines.append("  " + "-" * 70)
            lines.append(f"  LOW RISK / REMOVED: {len(low_risks)}")
            lines.append("  " + "-" * 70)
            for risk in low_risks:
                lines.append(f"    [{risk.status.value}] {risk.title}: "
                           f"{risk.preliminary_probability:.1f}%")

        unchanged = [r for r in self.risk_scores
                    if r.status == RiskChangeStatus.UNCHANGED]
        if unchanged:
            lines.append("")
            lines.append(f"  UNCHANGED (boilerplate, filtered): {len(unchanged)}")

        lines.append("")
        lines.append("=" * 80)
        lines.append("  NOTE: Scores are PRELIMINARY (Signals 1+2 only).")
        lines.append("  Final scores will incorporate Financial Ratios (Signal 3)")
        lines.append("  and Text-Financial Interaction (Signal 4).")
        lines.append("=" * 80)

        return '\n'.join(lines)



# =========================================================
# Combined Scoring Engine
# =========================================================

class RiskScoringEngine:
    """
    Combines Signal 1 and Signal 2 into a preliminary risk probability.

    From SPEC_SCORING_ENGINE.md:
        P(risk) = w1 * TextualChangeMagnitude
                + w2 * SentimentDirection
                + w3 * FinancialRatioDeteriation   (future)
                + w4 * InteractionAmplifier         (future)

    Current weights (Signals 1+2 only, renormalized):
        w1 = 0.55 (Textual change — slightly higher without financials)
        w2 = 0.45 (Sentiment direction)

    When Signals 3+4 are added, these will shift to:
        w1 = 0.30, w2 = 0.25, w3 = 0.25, w4 = 0.20
    """

    # Weights for current phase (Signals 1+2 only)
    # Renormalized from spec: 0.30/(0.30+0.25) and 0.25/(0.30+0.25)
    WEIGHT_TEXTUAL = 0.55
    WEIGHT_SENTIMENT = 0.45

    # Scale factor: raw 0-1 scores mapped to 0-100 probability
    # Capped below 100 since we're missing Signals 3+4
    MAX_PRELIMINARY_PROBABILITY = 85.0  # cap without financial data

    def score(self, change_report: RiskChangeReport, verbose: bool = False) -> ScoringReport:
        """
        Run the complete scoring pipeline.

        Args:
            change_report: Output from Step 2D (risk_classifier).
            verbose: Print progress to console.

        Returns:
            ScoringReport with combined probability scores.
        """
        if verbose:
            print(f"\n[Step 3] Scoring risks for {change_report.ticker}...")
            print(f"  Running Signal 1 (Textual Change Magnitude)...")

        # Signal 1: Textual Change Magnitude
        textual_report = score_textual_changes(change_report)

        if verbose:
            print(f"  Running Signal 2 (Sentiment Direction)...")

        # Signal 2: Sentiment Direction
        sentiment_report = score_risk_sentiment(change_report)

        if verbose:
            print(f"  Combining signals into preliminary probability...")

        # Combine
        scoring_report = self._combine_signals(
            change_report, textual_report, sentiment_report
        )

        if verbose:
            print(f"  Done. {len(scoring_report.high_priority_risks)} high-priority risks identified.")

        return scoring_report


    def _combine_signals(
        self,
        change_report: RiskChangeReport,
        textual_report: TextualChangeReport,
        sentiment_report: SentimentReport,
    ) -> ScoringReport:
        """Combine Signal 1 and Signal 2 into probability scores."""
        report = ScoringReport(
            ticker=change_report.ticker,
            current_year=change_report.current_year,
            prior_year=change_report.prior_year,
            textual_report=textual_report,
            sentiment_report=sentiment_report,
        )

        # Align scores by index (they come from the same classifications list)
        for i, classification in enumerate(change_report.classifications):
            textual_score = textual_report.scores[i] if i < len(textual_report.scores) else None
            sentiment_result = sentiment_report.risk_results[i] if i < len(sentiment_report.risk_results) else None

            risk_prob = self._compute_probability(
                classification, textual_score, sentiment_result
            )
            report.risk_scores.append(risk_prob)

        return report

    def _compute_probability(
        self,
        classification: RiskClassification,
        textual: Optional[TextualChangeScore],
        sentiment: Optional[RiskSentimentResult],
    ) -> RiskProbabilityScore:
        """Compute combined probability for one risk."""
        result = RiskProbabilityScore(
            title=classification.title,
            status=classification.status,
            textual_detail=textual,
            sentiment_detail=sentiment,
        )

        # Get raw scores (default to 0 if not available)
        t_score = textual.magnitude_score if textual else 0.0
        s_score = sentiment.sentiment_score if sentiment else 0.0

        result.textual_change_score = t_score
        result.sentiment_score = s_score

        # UNCHANGED risks: minimal probability
        if classification.status == RiskChangeStatus.UNCHANGED:
            result.preliminary_probability = round(t_score * 5.0, 1)  # ~1%
            result.risk_level = result.risk_level_label
            return result

        # REMOVED risks: low fixed probability
        if classification.status == RiskChangeStatus.REMOVED:
            result.preliminary_probability = 5.0
            result.risk_level = result.risk_level_label
            return result

        # MODIFIED and NEW: weighted combination
        raw_combined = (
            self.WEIGHT_TEXTUAL * t_score +
            self.WEIGHT_SENTIMENT * s_score
        )

        # Amplification: both signals strong = compounding danger
        if t_score > 0.5 and s_score > 0.5:
            amplifier = 1.0 + 0.15 * min(t_score, s_score)
            raw_combined = min(raw_combined * amplifier, 1.0)

        # Scale to probability (0-100) with cap
        probability = raw_combined * self.MAX_PRELIMINARY_PROBABILITY

        # Floor for NEW risks (minimum 25% if any negative content)
        if classification.status == RiskChangeStatus.NEW and s_score > 0.2:
            probability = max(probability, 25.0)

        result.preliminary_probability = round(
            min(max(probability, 0.0), self.MAX_PRELIMINARY_PROBABILITY), 1
        )
        result.risk_level = result.risk_level_label
        return result



# =========================================================
# Public API — Main Entry Point
# =========================================================

def run_scoring(
    change_report: RiskChangeReport,
    verbose: bool = True
) -> ScoringReport:
    """
    Run the complete Step 3 scoring pipeline.

    Takes Step 2D output and produces preliminary risk probabilities
    by combining Signal 1 (Textual Change) and Signal 2 (Sentiment).

    Args:
        change_report: Output from Step 2D (classify_risk_changes).
        verbose: If True, prints progress and results to console.

    Returns:
        ScoringReport with per-risk probability scores.

    Usage:
        from step_2d_risk_analysis import run_risk_analysis
        from step_3_scoring import run_scoring

        # Step 2D
        report = run_risk_analysis(raw_current, raw_prior, ticker="TGT", ...)

        # Step 3
        scoring = run_scoring(report)
        print(scoring.summary())
    """
    engine = RiskScoringEngine()
    scoring_report = engine.score(change_report, verbose=verbose)

    if verbose:
        print(scoring_report.summary())

    return scoring_report


# =========================================================
# Full Pipeline: Step 2D + Step 3 in one call
# =========================================================

def run_full_analysis(
    raw_text_current: str,
    raw_text_prior: str,
    ticker: str = "UNKNOWN",
    current_year: int = 0,
    prior_year: int = 0,
    verbose: bool = True,
) -> ScoringReport:
    """
    Complete analysis pipeline: text cleaning → parsing → matching →
    classification → textual scoring → sentiment scoring → combined probability.

    This is the top-level entry point for the full ERPSA pipeline.

    Args:
        raw_text_current: Raw Item 1A HTML/text from current year's 10-K.
        raw_text_prior: Raw Item 1A HTML/text from prior year's 10-K.
        ticker: Stock ticker symbol.
        current_year: Filing year for current 10-K.
        prior_year: Filing year for prior 10-K.
        verbose: Print progress and results.

    Returns:
        ScoringReport with full analysis results.
    """
    from step_2d_risk_analysis import run_risk_analysis

    # Step 2D: Topic-aware risk change analysis
    change_report = run_risk_analysis(
        raw_text_current=raw_text_current,
        raw_text_prior=raw_text_prior,
        ticker=ticker,
        current_year=current_year,
        prior_year=prior_year,
        verbose=verbose,
    )

    # Step 3: Scoring
    scoring_report = run_scoring(change_report, verbose=verbose)

    return scoring_report


if __name__ == "__main__":
    print("Step 3 Scoring Pipeline loaded successfully.")
    print("Use run_scoring() with a RiskChangeReport from Step 2D.")
    print("Or use run_full_analysis() for the complete pipeline.")
