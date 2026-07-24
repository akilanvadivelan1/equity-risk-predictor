# =========================================================
# ERPSA - Step 2D (Improved): Cross-Year Risk Category Matcher
# =========================================================
# Matches risk sections across two filing years by topic using
# multi-signal scoring: title keywords, title text, body content.
# =========================================================

import difflib
import re
from dataclasses import dataclass
from enum import Enum
from typing import Optional, List

from risk_section_parser import RiskSection


class RiskChangeStatus(Enum):
    """Classification of how a risk section changed year-over-year."""
    UNCHANGED = "UNCHANGED"
    MODIFIED = "MODIFIED"
    NEW = "NEW"
    REMOVED = "REMOVED"


@dataclass
class RiskMatch:
    """Result of matching a current-year risk to a prior-year risk."""
    current_section: RiskSection
    prior_section: Optional[RiskSection]
    status: RiskChangeStatus
    title_similarity: float
    body_similarity: float
    keyword_overlap: float
    combined_score: float


class RiskCategoryMatcher:
    """Matches risk sections across two filing years by topic/category."""

    WEIGHT_TITLE_KEYWORDS = 0.35
    WEIGHT_TITLE_TEXT = 0.30
    WEIGHT_BODY_SIMILARITY = 0.35

    def __init__(self, match_threshold=0.25, unchanged_threshold=0.90, modified_threshold=0.40):
        self.match_threshold = match_threshold
        self.unchanged_threshold = unchanged_threshold
        self.modified_threshold = modified_threshold

    def match_sections(self, current_sections: List[RiskSection], prior_sections: List[RiskSection]) -> List[RiskMatch]:
        """Match current-year risk sections to prior-year sections by topic."""
        results = []

        # Compute similarity matrix
        similarity_matrix = []
        for i, curr in enumerate(current_sections):
            row = []
            for j, prior in enumerate(prior_sections):
                scores = self._compute_similarity(curr, prior)
                row.append((j, scores))
            similarity_matrix.append(row)

        # Greedy matching
        match_candidates = []
        for i, row in enumerate(similarity_matrix):
            best_j, best_scores = max(row, key=lambda x: x[1]['combined'])
            match_candidates.append((i, best_j, best_scores))

        match_candidates.sort(key=lambda x: x[2]['combined'], reverse=True)

        matched_current = set()
        matched_prior = set()

        for i, best_j, scores in match_candidates:
            if i in matched_current:
                continue
            if scores['combined'] >= self.match_threshold and best_j not in matched_prior:
                matched_current.add(i)
                matched_prior.add(best_j)
                status = self._classify_change(scores['body_similarity'])
                results.append(RiskMatch(
                    current_section=current_sections[i],
                    prior_section=prior_sections[best_j],
                    status=status,
                    title_similarity=scores['title_text'],
                    body_similarity=scores['body_similarity'],
                    keyword_overlap=scores['keyword_overlap'],
                    combined_score=scores['combined']
                ))

        # Unmatched current = NEW
        for i, curr in enumerate(current_sections):
            if i not in matched_current:
                results.append(RiskMatch(
                    current_section=curr, prior_section=None,
                    status=RiskChangeStatus.NEW,
                    title_similarity=0.0, body_similarity=0.0,
                    keyword_overlap=0.0, combined_score=0.0
                ))

        # Unmatched prior = REMOVED
        for j, prior in enumerate(prior_sections):
            if j not in matched_prior:
                results.append(RiskMatch(
                    current_section=prior, prior_section=prior,
                    status=RiskChangeStatus.REMOVED,
                    title_similarity=0.0, body_similarity=0.0,
                    keyword_overlap=0.0, combined_score=0.0
                ))

        return results

    def _compute_similarity(self, current: RiskSection, prior: RiskSection) -> dict:
        keyword_overlap = self._jaccard_similarity(current.title_keywords, prior.title_keywords)
        title_text_sim = difflib.SequenceMatcher(None, current.title.lower(), prior.title.lower()).ratio()
        body_sim = difflib.SequenceMatcher(None, current.body[:2000], prior.body[:2000]).ratio()
        combined = (
            self.WEIGHT_TITLE_KEYWORDS * keyword_overlap +
            self.WEIGHT_TITLE_TEXT * title_text_sim +
            self.WEIGHT_BODY_SIMILARITY * body_sim
        )
        return {'keyword_overlap': keyword_overlap, 'title_text': title_text_sim,
                'body_similarity': body_sim, 'combined': combined}

    def _classify_change(self, body_similarity: float) -> RiskChangeStatus:
        if body_similarity >= self.unchanged_threshold:
            return RiskChangeStatus.UNCHANGED
        else:
            return RiskChangeStatus.MODIFIED

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        if not set_a and not set_b:
            return 0.0
        intersection = set_a & set_b
        union = set_a | set_b
        return len(intersection) / len(union) if union else 0.0


def match_risk_categories(
    current_sections: List[RiskSection],
    prior_sections: List[RiskSection],
    match_threshold: float = 0.25,
    unchanged_threshold: float = 0.90,
    modified_threshold: float = 0.40
) -> List[RiskMatch]:
    """Match and classify risk sections across two filing years."""
    matcher = RiskCategoryMatcher(match_threshold, unchanged_threshold, modified_threshold)
    return matcher.match_sections(current_sections, prior_sections)
