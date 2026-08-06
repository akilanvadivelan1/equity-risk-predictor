# =========================================================
# ERPSA - Phase 2: Textual Change Magnitude Scorer (Signal 1)
# =========================================================
# Scores how much the risk language changed year-over-year.
#
# Academic basis: Cohen, Malloy & Nguyen, "Lazy Prices" (2020)
#   - Changes to 10-K language predict future earnings,
#     profitability, and bankruptcies.
#   - A portfolio based on textual changes earns 22%+ per year.
#
# Inputs (from Step 2D):
#   - Risk section classification (UNCHANGED/MODIFIED/NEW/REMOVED)
#   - Body similarity score (0.0 - 1.0)
#   - Number of added sentences
#   - Number of rewritten sentences
#   - Percentage of section that is new content
#
# Output:
#   - Textual change magnitude score (0.0 - 1.0) per risk section
# =========================================================

import re
from dataclasses import dataclass, field
from typing import List, Optional

from risk_classifier import RiskClassification, RiskChangeReport
from risk_matcher import RiskChangeStatus



# =========================================================
# Data Classes
# =========================================================

@dataclass
class TextualChangeScore:
    """Textual change magnitude score for a single risk section."""
    title: str
    status: RiskChangeStatus

    # Input metrics from Step 2D
    body_similarity: float = 0.0
    added_sentence_count: int = 0
    rewritten_sentence_count: int = 0
    total_changed_sentences: int = 0
    new_content_percentage: float = 0.0   # 0.0 - 1.0

    # Component scores (each 0.0 - 1.0)
    dissimilarity_score: float = 0.0      # 1 - body_similarity
    change_volume_score: float = 0.0      # based on sentence counts
    new_content_score: float = 0.0        # based on % new content

    # Final textual change magnitude (0.0 - 1.0)
    magnitude_score: float = 0.0

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"[{self.status.value}] {self.title}",
            f"  Magnitude Score: {self.magnitude_score:.3f}",
            f"  Body Similarity: {self.body_similarity:.3f}",
            f"  Dissimilarity: {self.dissimilarity_score:.3f}",
            f"  Change Volume: {self.change_volume_score:.3f} "
            f"(added={self.added_sentence_count}, rewritten={self.rewritten_sentence_count})",
            f"  New Content %: {self.new_content_percentage:.1%}",
        ]
        return '\n'.join(lines)


@dataclass
class TextualChangeReport:
    """Complete textual change magnitude report for all risks."""
    ticker: str
    current_year: int
    prior_year: int
    scores: List[TextualChangeScore] = field(default_factory=list)

    @property
    def modified_scores(self) -> List[TextualChangeScore]:
        return [s for s in self.scores if s.status == RiskChangeStatus.MODIFIED]

    @property
    def new_scores(self) -> List[TextualChangeScore]:
        return [s for s in self.scores if s.status == RiskChangeStatus.NEW]

    @property
    def high_change_scores(self) -> List[TextualChangeScore]:
        """Scores with magnitude >= 0.5 (significant textual change)."""
        return [s for s in self.scores if s.magnitude_score >= 0.5]

    @property
    def average_magnitude(self) -> float:
        """Average magnitude across non-UNCHANGED risks."""
        scored = [s for s in self.scores
                  if s.status != RiskChangeStatus.UNCHANGED]
        if not scored:
            return 0.0
        return sum(s.magnitude_score for s in scored) / len(scored)

    def summary(self) -> str:
        """Human-readable report summary."""
        lines = [
            "=" * 70,
            f"  TEXTUAL CHANGE MAGNITUDE REPORT: {self.ticker}",
            f"  Comparing: FY{self.current_year} vs FY{self.prior_year}",
            "=" * 70,
            "",
            f"  Risks Analyzed: {len(self.scores)}",
            f"  Modified: {len(self.modified_scores)}",
            f"  New: {len(self.new_scores)}",
            f"  High Change (magnitude >= 0.5): {len(self.high_change_scores)}",
            f"  Average Magnitude: {self.average_magnitude:.3f}",
            "",
        ]
        return '\n'.join(lines)



# =========================================================
# Textual Change Magnitude Scorer
# =========================================================

class TextualChangeMagnitudeScorer:
    """
    Scores the magnitude of textual changes in risk disclosures.

    Scoring logic (from SPEC_SCORING_ENGINE.md Signal 1):
    - UNCHANGED: score near 0 (no signal)
    - MODIFIED: proportional to (1 - body_similarity) and changed sentences
    - NEW: elevated base score (new disclosure = strong signal)
    - REMOVED: moderate score (risk dropped — could be good or hiding)
    """

    # Weights for MODIFIED risk score components
    WEIGHT_DISSIMILARITY = 0.45    # how different the body text is
    WEIGHT_CHANGE_VOLUME = 0.35    # how many sentences changed
    WEIGHT_NEW_CONTENT = 0.20      # what % of the section is new

    # Normalization: number of changed sentences that maps to score 1.0
    MAX_CHANGED_SENTENCES = 12

    # Base scores for status types
    UNCHANGED_BASE = 0.02          # negligible signal
    NEW_RISK_BASE = 0.65           # new disclosure is inherently a strong signal
    REMOVED_RISK_BASE = 0.25       # risk dropped — moderate concern

    def score_report(self, change_report: RiskChangeReport) -> TextualChangeReport:
        """
        Score all risk classifications for textual change magnitude.

        Args:
            change_report: Output from Step 2D (risk_classifier).

        Returns:
            TextualChangeReport with per-risk magnitude scores.
        """
        report = TextualChangeReport(
            ticker=change_report.ticker,
            current_year=change_report.current_year,
            prior_year=change_report.prior_year,
        )

        for classification in change_report.classifications:
            score = self._score_classification(classification)
            report.scores.append(score)

        return report


    def _score_classification(self, classification: RiskClassification) -> TextualChangeScore:
        """Score a single risk classification."""
        result = TextualChangeScore(
            title=classification.title,
            status=classification.status,
            body_similarity=classification.body_similarity,
        )

        if classification.status == RiskChangeStatus.UNCHANGED:
            result.magnitude_score = self.UNCHANGED_BASE
            return result

        elif classification.status == RiskChangeStatus.MODIFIED:
            return self._score_modified(classification, result)

        elif classification.status == RiskChangeStatus.NEW:
            return self._score_new(classification, result)

        elif classification.status == RiskChangeStatus.REMOVED:
            result.magnitude_score = self.REMOVED_RISK_BASE
            return result

        return result

    def _score_modified(
        self,
        classification: RiskClassification,
        result: TextualChangeScore
    ) -> TextualChangeScore:
        """Score a MODIFIED risk section."""
        # Count changed sentences
        added = sum(1 for s in classification.changed_sentences
                    if s.change_type == "added")
        rewritten = sum(1 for s in classification.changed_sentences
                        if s.change_type == "rewritten")
        total_changed = added + rewritten

        result.added_sentence_count = added
        result.rewritten_sentence_count = rewritten
        result.total_changed_sentences = total_changed

        # Compute new content percentage
        result.new_content_percentage = self._compute_new_content_pct(
            classification, added, rewritten
        )

        # Component 1: Dissimilarity (1 - body_similarity)
        # body_similarity of 0.9 means 10% different → low score
        # body_similarity of 0.3 means 70% different → high score
        result.dissimilarity_score = 1.0 - classification.body_similarity

        # Component 2: Change volume (normalized sentence count)
        # More changed sentences = stronger signal
        result.change_volume_score = min(
            total_changed / self.MAX_CHANGED_SENTENCES, 1.0
        )
        # Weight added sentences slightly more than rewritten
        if total_changed > 0:
            added_ratio = added / total_changed
            # Boost if most changes are additions (brand new language)
            volume_boost = 1.0 + (0.2 * added_ratio)
            result.change_volume_score = min(
                result.change_volume_score * volume_boost, 1.0
            )

        # Component 3: New content percentage
        result.new_content_score = min(result.new_content_percentage / 0.6, 1.0)

        # Weighted combination
        magnitude = (
            self.WEIGHT_DISSIMILARITY * result.dissimilarity_score +
            self.WEIGHT_CHANGE_VOLUME * result.change_volume_score +
            self.WEIGHT_NEW_CONTENT * result.new_content_score
        )

        # Amplification: very low similarity + many changes = severe rewrite
        if result.dissimilarity_score > 0.6 and result.change_volume_score > 0.5:
            magnitude = min(magnitude * 1.15, 1.0)

        result.magnitude_score = round(min(max(magnitude, 0.0), 1.0), 4)
        return result


    def _score_new(
        self,
        classification: RiskClassification,
        result: TextualChangeScore
    ) -> TextualChangeScore:
        """
        Score a NEW risk section.

        New risks always receive an elevated base score because the
        existence of a new disclosure is itself a strong signal
        (per "Lazy Prices" findings). The length/complexity of the
        new risk further modulates the score upward.
        """
        result.new_content_percentage = 1.0  # 100% new by definition
        result.dissimilarity_score = 1.0
        result.new_content_score = 1.0

        # Modulate based on the substance of the new disclosure
        body_length = len(classification.current_body)
        key_sentence_count = len(classification.key_sentences) if classification.key_sentences else 0

        # Longer, more detailed new disclosures are stronger signals
        # (brief boilerplate additions are weaker)
        length_factor = min(body_length / 1000.0, 1.0)  # 1000+ chars = full signal
        detail_factor = min(key_sentence_count / 5.0, 1.0)  # 5+ key sentences = full

        substance_score = 0.5 * length_factor + 0.5 * detail_factor
        result.change_volume_score = substance_score

        # Final: base + modulation
        magnitude = self.NEW_RISK_BASE + (1.0 - self.NEW_RISK_BASE) * substance_score

        result.magnitude_score = round(min(max(magnitude, 0.0), 1.0), 4)
        return result

    def _compute_new_content_pct(
        self,
        classification: RiskClassification,
        added: int,
        rewritten: int
    ) -> float:
        """
        Estimate what percentage of the current risk body is new content.

        Uses the ratio of changed sentence text to total body text.
        """
        if not classification.current_body:
            return 0.0

        total_body_len = len(classification.current_body)
        if total_body_len == 0:
            return 0.0

        # Sum the length of all changed sentences
        changed_text_len = sum(
            len(s.sentence) for s in classification.changed_sentences
        )

        # For rewritten sentences, count ~50% as "new" (the rest overlaps)
        rewritten_text_len = sum(
            len(s.sentence) for s in classification.changed_sentences
            if s.change_type == "rewritten"
        )
        effective_new_len = changed_text_len - (rewritten_text_len * 0.5)

        return min(max(effective_new_len / total_body_len, 0.0), 1.0)



# =========================================================
# Public API Functions
# =========================================================

def score_textual_changes(change_report: RiskChangeReport) -> TextualChangeReport:
    """
    Score all risks for textual change magnitude (Signal 1).

    This is the primary entry point for Signal 1 scoring.

    Args:
        change_report: Output from Step 2D (classify_risk_changes).

    Returns:
        TextualChangeReport with per-risk magnitude scores.

    Usage:
        from risk_classifier import classify_risk_changes
        from textual_change_scorer import score_textual_changes

        report = classify_risk_changes(matches, ticker="TGT", ...)
        textual = score_textual_changes(report)

        for score in textual.high_change_scores:
            print(f"{score.title}: magnitude={score.magnitude_score:.3f}")
    """
    scorer = TextualChangeMagnitudeScorer()
    return scorer.score_report(change_report)


def score_single_classification(classification: RiskClassification) -> TextualChangeScore:
    """
    Score a single risk classification for textual change magnitude.

    Useful for ad-hoc scoring outside the full pipeline.
    """
    scorer = TextualChangeMagnitudeScorer()
    return scorer._score_classification(classification)
