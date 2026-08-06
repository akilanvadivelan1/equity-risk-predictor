# =========================================================
# ERPSA - Phase 2: Sentiment Scoring Engine
# =========================================================
# Scores the tone and severity of changed/new risk text using
# the Loughran-McDonald financial sentiment dictionary.
#
# This module takes Step 2D output (RiskChangeReport) and
# produces per-risk sentiment scores that feed into the final
# probability calculation.
#
# Academic basis: Loughran & McDonald (2011), Bodnaruk et al. (2015)
# =========================================================

import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional

from lm_dictionary import (
    LoughranMcDonaldDictionary,
    SentimentCategory,
    get_dictionary,
)
from risk_classifier import (
    RiskClassification,
    RiskChangeReport,
    SentenceChange,
)
from risk_matcher import RiskChangeStatus



# =========================================================
# Data Classes for Sentiment Results
# =========================================================

@dataclass
class SentimentScore:
    """Sentiment analysis result for a single risk section."""
    # Category densities (proportion of words in each category)
    negative_density: float = 0.0
    positive_density: float = 0.0
    uncertainty_density: float = 0.0
    litigious_density: float = 0.0
    constraining_density: float = 0.0

    # Raw counts
    negative_count: int = 0
    positive_count: int = 0
    uncertainty_count: int = 0
    litigious_count: int = 0
    constraining_count: int = 0
    word_count: int = 0

    # Composite metrics
    net_negativity: float = 0.0       # (negative - positive) density
    risk_intensity: float = 0.0       # combined negative + uncertainty + constraining
    modal_weakness: float = 0.0       # weak modal density (hedging language)

    @property
    def is_heavily_negative(self) -> bool:
        """Whether the text is dominated by negative sentiment."""
        return self.negative_density > 0.10 and self.net_negativity > 0.08

    @property
    def is_highly_uncertain(self) -> bool:
        """Whether the text has high uncertainty language."""
        return self.uncertainty_density > 0.08

    @property
    def is_constrained(self) -> bool:
        """Whether the text indicates significant constraints."""
        return self.constraining_density > 0.05



@dataclass
class SentimentDelta:
    """Year-over-year change in sentiment for a risk section."""
    negative_delta: float = 0.0       # positive = MORE negative language this year
    positive_delta: float = 0.0       # negative = LESS positive language this year
    uncertainty_delta: float = 0.0
    constraining_delta: float = 0.0
    litigious_delta: float = 0.0
    net_negativity_delta: float = 0.0  # most important: overall tone shift

    @property
    def tone_worsened(self) -> bool:
        """Whether the tone shifted toward more negative/uncertain."""
        return self.net_negativity_delta > 0.02 or self.uncertainty_delta > 0.03

    @property
    def tone_improved(self) -> bool:
        """Whether the tone shifted toward more positive."""
        return self.net_negativity_delta < -0.02


@dataclass
class RiskSentimentResult:
    """Complete sentiment analysis for one risk section."""
    title: str
    status: RiskChangeStatus
    body_similarity: float

    # Sentiment of changed/new content only
    changed_sentiment: SentimentScore = field(default_factory=SentimentScore)

    # Sentiment of full current body
    current_sentiment: SentimentScore = field(default_factory=SentimentScore)

    # Sentiment of prior body (for MODIFIED risks)
    prior_sentiment: Optional[SentimentScore] = None

    # Year-over-year delta (for MODIFIED risks)
    sentiment_delta: Optional[SentimentDelta] = None

    # Final sentiment-based score (0.0 - 1.0)
    sentiment_score: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        parts = [f"[{self.status.value}] {self.title}"]
        parts.append(f"  Sentiment Score: {self.sentiment_score:.2f}")
        parts.append(f"  Negative density: {self.changed_sentiment.negative_density:.3f}")
        parts.append(f"  Uncertainty density: {self.changed_sentiment.uncertainty_density:.3f}")
        parts.append(f"  Net negativity: {self.changed_sentiment.net_negativity:.3f}")
        if self.sentiment_delta and self.status == RiskChangeStatus.MODIFIED:
            parts.append(f"  Tone delta: {self.sentiment_delta.net_negativity_delta:+.3f}")
            if self.sentiment_delta.tone_worsened:
                parts.append("  ** TONE WORSENED **")
        return '\n'.join(parts)



@dataclass
class SentimentReport:
    """Complete sentiment analysis report for all risks in a filing comparison."""
    ticker: str
    current_year: int
    prior_year: int
    risk_results: List[RiskSentimentResult] = field(default_factory=list)

    @property
    def modified_results(self) -> List[RiskSentimentResult]:
        return [r for r in self.risk_results if r.status == RiskChangeStatus.MODIFIED]

    @property
    def new_results(self) -> List[RiskSentimentResult]:
        return [r for r in self.risk_results if r.status == RiskChangeStatus.NEW]

    @property
    def high_risk_results(self) -> List[RiskSentimentResult]:
        """Results with sentiment score >= 0.5 (significant concern)."""
        return [r for r in self.risk_results if r.sentiment_score >= 0.5]

    @property
    def average_sentiment_score(self) -> float:
        """Average sentiment score across all non-UNCHANGED risks."""
        scored = [r for r in self.risk_results
                  if r.status != RiskChangeStatus.UNCHANGED]
        if not scored:
            return 0.0
        return sum(r.sentiment_score for r in scored) / len(scored)

    def summary(self) -> str:
        """Human-readable report summary."""
        lines = [
            "=" * 70,
            f"  SENTIMENT ANALYSIS REPORT: {self.ticker}",
            f"  Comparing: FY{self.current_year} vs FY{self.prior_year}",
            "=" * 70,
            "",
            f"  Risks Analyzed: {len(self.risk_results)}",
            f"  Modified (tone-scored): {len(self.modified_results)}",
            f"  New (tone-scored): {len(self.new_results)}",
            f"  High Risk (score >= 0.5): {len(self.high_risk_results)}",
            f"  Average Sentiment Score: {self.average_sentiment_score:.2f}",
            "",
        ]
        return '\n'.join(lines)



# =========================================================
# Sentiment Scorer Engine
# =========================================================

class SentimentScorer:
    """
    Scores risk sections based on Loughran-McDonald sentiment analysis.

    Scoring approach (from SPEC_SCORING_ENGINE.md Signal 2):
    - Scores changed/new sentences specifically (not entire filing)
    - Weights negative and uncertainty words more heavily
    - Compares sentiment density vs prior year
    - A shift from neutral to negative amplifies the score
    """

    # Weights for combining category signals into final score
    WEIGHT_NEGATIVE = 0.35
    WEIGHT_UNCERTAINTY = 0.25
    WEIGHT_CONSTRAINING = 0.20
    WEIGHT_LITIGIOUS = 0.10
    WEIGHT_TONE_DELTA = 0.10

    # Normalization constants (typical density ranges in 10-K text)
    # Based on empirical observation of risk factor sections
    NORM_NEGATIVE = 0.15      # density of 0.15 maps to score 1.0
    NORM_UNCERTAINTY = 0.12
    NORM_CONSTRAINING = 0.08
    NORM_LITIGIOUS = 0.06
    NORM_TONE_DELTA = 0.05    # delta of 0.05 maps to score 1.0

    def __init__(self, dictionary: Optional[LoughranMcDonaldDictionary] = None):
        """Initialize with a dictionary (uses default singleton if not provided)."""
        self.lm = dictionary or get_dictionary()

    def score_report(self, change_report: RiskChangeReport) -> SentimentReport:
        """
        Score all risk sections in a RiskChangeReport.

        Args:
            change_report: Output from Step 2D (risk_classifier.classify_risk_changes)

        Returns:
            SentimentReport with per-risk sentiment scores.
        """
        sentiment_report = SentimentReport(
            ticker=change_report.ticker,
            current_year=change_report.current_year,
            prior_year=change_report.prior_year,
        )

        for classification in change_report.classifications:
            result = self._score_classification(classification)
            sentiment_report.risk_results.append(result)

        return sentiment_report


    def _score_classification(self, classification: RiskClassification) -> RiskSentimentResult:
        """Score a single risk classification."""
        result = RiskSentimentResult(
            title=classification.title,
            status=classification.status,
            body_similarity=classification.body_similarity,
        )

        if classification.status == RiskChangeStatus.UNCHANGED:
            # No sentiment scoring needed — boilerplate
            result.sentiment_score = 0.0
            return result

        elif classification.status == RiskChangeStatus.MODIFIED:
            result = self._score_modified(classification, result)

        elif classification.status == RiskChangeStatus.NEW:
            result = self._score_new(classification, result)

        elif classification.status == RiskChangeStatus.REMOVED:
            # Removed risks get a low positive score (risk was dropped)
            result.sentiment_score = 0.05
            return result

        return result

    def _score_modified(
        self,
        classification: RiskClassification,
        result: RiskSentimentResult
    ) -> RiskSentimentResult:
        """Score a MODIFIED risk section with year-over-year comparison."""
        # Score the changed sentences specifically
        changed_text = self._extract_changed_text(classification)
        result.changed_sentiment = self._compute_sentiment(changed_text)

        # Score full current and prior bodies for delta
        result.current_sentiment = self._compute_sentiment(classification.current_body)
        if classification.prior_body:
            result.prior_sentiment = self._compute_sentiment(classification.prior_body)
            result.sentiment_delta = self._compute_delta(
                result.current_sentiment, result.prior_sentiment
            )

        # Compute final score
        result.sentiment_score = self._compute_modified_score(result)
        return result

    def _score_new(
        self,
        classification: RiskClassification,
        result: RiskSentimentResult
    ) -> RiskSentimentResult:
        """Score a NEW risk section (no prior year comparison available)."""
        # For new risks, score the full body and key sentences
        result.current_sentiment = self._compute_sentiment(classification.current_body)

        # Also score key sentences if available (they're the most important parts)
        if classification.key_sentences:
            key_text = ' '.join(classification.key_sentences)
            result.changed_sentiment = self._compute_sentiment(key_text)
        else:
            result.changed_sentiment = result.current_sentiment

        # Compute final score (new risks get a baseline bump)
        result.sentiment_score = self._compute_new_score(result)
        return result


    def _extract_changed_text(self, classification: RiskClassification) -> str:
        """Extract only the changed/new sentences from a MODIFIED risk."""
        if not classification.changed_sentences:
            # Fallback: use the full current body
            return classification.current_body

        parts = []
        for change in classification.changed_sentences:
            parts.append(change.sentence)
        return ' '.join(parts)

    def _compute_sentiment(self, text: str) -> SentimentScore:
        """Compute sentiment scores for a text block."""
        if not text or not text.strip():
            return SentimentScore()

        counts = self.lm.score_text(text)
        densities = self.lm.score_text_density(text)

        score = SentimentScore(
            negative_density=densities.get('negative', 0.0),
            positive_density=densities.get('positive', 0.0),
            uncertainty_density=densities.get('uncertainty', 0.0),
            litigious_density=densities.get('litigious', 0.0),
            constraining_density=densities.get('constraining', 0.0),
            negative_count=counts.get('negative', 0),
            positive_count=counts.get('positive', 0),
            uncertainty_count=counts.get('uncertainty', 0),
            litigious_count=counts.get('litigious', 0),
            constraining_count=counts.get('constraining', 0),
            word_count=counts.get('word_count', 0),
            net_negativity=densities.get('negative', 0.0) - densities.get('positive', 0.0),
            risk_intensity=(
                densities.get('negative', 0.0) +
                densities.get('uncertainty', 0.0) +
                densities.get('constraining', 0.0)
            ),
            modal_weakness=densities.get('modal_weak', 0.0),
        )
        return score

    def _compute_delta(
        self,
        current: SentimentScore,
        prior: SentimentScore
    ) -> SentimentDelta:
        """Compute the year-over-year change in sentiment."""
        return SentimentDelta(
            negative_delta=current.negative_density - prior.negative_density,
            positive_delta=current.positive_density - prior.positive_density,
            uncertainty_delta=current.uncertainty_density - prior.uncertainty_density,
            constraining_delta=current.constraining_density - prior.constraining_density,
            litigious_delta=current.litigious_density - prior.litigious_density,
            net_negativity_delta=current.net_negativity - prior.net_negativity,
        )


    def _compute_modified_score(self, result: RiskSentimentResult) -> float:
        """
        Compute final sentiment score for a MODIFIED risk.

        Combines:
        - Negativity of the changed sentences
        - Uncertainty in the changed sentences
        - Constraining language
        - Litigious language
        - Year-over-year tone shift (delta)

        Returns: Score between 0.0 and 1.0
        """
        sent = result.changed_sentiment

        # Normalize each signal to 0-1 range
        neg_signal = min(sent.negative_density / self.NORM_NEGATIVE, 1.0)
        unc_signal = min(sent.uncertainty_density / self.NORM_UNCERTAINTY, 1.0)
        con_signal = min(sent.constraining_density / self.NORM_CONSTRAINING, 1.0)
        lit_signal = min(sent.litigious_density / self.NORM_LITIGIOUS, 1.0)

        # Tone delta signal (only if we have prior comparison)
        delta_signal = 0.0
        if result.sentiment_delta:
            # Positive delta = tone worsened = higher score
            raw_delta = max(result.sentiment_delta.net_negativity_delta, 0.0)
            delta_signal = min(raw_delta / self.NORM_TONE_DELTA, 1.0)

        # Weighted combination
        score = (
            self.WEIGHT_NEGATIVE * neg_signal +
            self.WEIGHT_UNCERTAINTY * unc_signal +
            self.WEIGHT_CONSTRAINING * con_signal +
            self.WEIGHT_LITIGIOUS * lit_signal +
            self.WEIGHT_TONE_DELTA * delta_signal
        )

        # Amplification: if both negative AND uncertainty are high, boost
        if neg_signal > 0.5 and unc_signal > 0.5:
            score = min(score * 1.15, 1.0)

        # Amplification: if tone worsened significantly, boost
        if result.sentiment_delta and result.sentiment_delta.tone_worsened:
            score = min(score * 1.10, 1.0)

        return round(min(max(score, 0.0), 1.0), 4)

    def _compute_new_score(self, result: RiskSentimentResult) -> float:
        """
        Compute final sentiment score for a NEW risk.

        New risks get a baseline bump (0.15) because the very existence
        of a new risk disclosure is a signal, per "Lazy Prices" findings.
        The sentiment of the new text further modulates the score.

        Returns: Score between 0.0 and 1.0
        """
        NEW_RISK_BASELINE = 0.15

        sent = result.changed_sentiment

        # Normalize signals
        neg_signal = min(sent.negative_density / self.NORM_NEGATIVE, 1.0)
        unc_signal = min(sent.uncertainty_density / self.NORM_UNCERTAINTY, 1.0)
        con_signal = min(sent.constraining_density / self.NORM_CONSTRAINING, 1.0)
        lit_signal = min(sent.litigious_density / self.NORM_LITIGIOUS, 1.0)

        # Weighted combination (no delta for new risks — there's no prior)
        content_score = (
            0.40 * neg_signal +
            0.30 * unc_signal +
            0.20 * con_signal +
            0.10 * lit_signal
        )

        # Combine baseline + content
        score = NEW_RISK_BASELINE + (1.0 - NEW_RISK_BASELINE) * content_score

        # Amplification: heavily negative new risk
        if neg_signal > 0.6 and unc_signal > 0.4:
            score = min(score * 1.20, 1.0)

        return round(min(max(score, 0.0), 1.0), 4)



# =========================================================
# Public API Functions
# =========================================================

def score_risk_sentiment(change_report: RiskChangeReport) -> SentimentReport:
    """
    Score all risks in a change report for sentiment severity.

    This is the primary entry point for Phase 2 scoring.

    Args:
        change_report: Output from Step 2D (classify_risk_changes).

    Returns:
        SentimentReport with per-risk sentiment scores.

    Usage:
        from risk_classifier import classify_risk_changes
        from sentiment_scorer import score_risk_sentiment

        report = classify_risk_changes(matches, ticker="TGT", ...)
        sentiment = score_risk_sentiment(report)

        for risk in sentiment.high_risk_results:
            print(f"{risk.title}: {risk.sentiment_score:.2f}")
    """
    scorer = SentimentScorer()
    return scorer.score_report(change_report)


def score_text_sentiment(text: str) -> SentimentScore:
    """
    Standalone function to score any text block for sentiment.

    Useful for ad-hoc analysis outside the pipeline.

    Args:
        text: Any text to analyze.

    Returns:
        SentimentScore with all category densities and counts.
    """
    scorer = SentimentScorer()
    return scorer._compute_sentiment(text)


def compare_text_sentiment(current_text: str, prior_text: str) -> Dict[str, any]:
    """
    Compare sentiment between two text blocks.

    Args:
        current_text: Current year text.
        prior_text: Prior year text.

    Returns:
        Dictionary with 'current', 'prior', 'delta' SentimentScores.
    """
    scorer = SentimentScorer()
    current = scorer._compute_sentiment(current_text)
    prior = scorer._compute_sentiment(prior_text)
    delta = scorer._compute_delta(current, prior)
    return {
        'current': current,
        'prior': prior,
        'delta': delta,
    }
