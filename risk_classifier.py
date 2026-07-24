# =========================================================
# ERPSA - Step 2D (Improved): Risk Change Classifier & Reporter
# =========================================================
# Classifies matched risk sections and performs sentence-level
# diffing for MODIFIED risks to identify exactly what changed.
# =========================================================

import difflib
import re
from dataclasses import dataclass, field
from typing import List

from risk_section_parser import RiskSection
from risk_matcher import RiskMatch, RiskChangeStatus


@dataclass
class SentenceChange:
    """A single sentence that changed within a MODIFIED risk section."""
    sentence: str
    change_type: str           # "added" or "rewritten"
    similarity_to_prior: float
    prior_sentence: str = ""


@dataclass
class RiskClassification:
    """Detailed classification of a single risk section's change status."""
    title: str
    status: RiskChangeStatus
    body_similarity: float
    combined_match_score: float
    current_body: str
    prior_body: str = ""
    changed_sentences: List[SentenceChange] = field(default_factory=list)
    key_sentences: List[str] = field(default_factory=list)
    change_summary: str = ""


@dataclass
class RiskChangeReport:
    """Complete classification report for a year-over-year comparison."""
    ticker: str
    current_year: int
    prior_year: int
    total_current_risks: int
    total_prior_risks: int
    classifications: List[RiskClassification] = field(default_factory=list)

    @property
    def unchanged_risks(self) -> List[RiskClassification]:
        return [c for c in self.classifications if c.status == RiskChangeStatus.UNCHANGED]

    @property
    def modified_risks(self) -> List[RiskClassification]:
        return [c for c in self.classifications if c.status == RiskChangeStatus.MODIFIED]

    @property
    def new_risks(self) -> List[RiskClassification]:
        return [c for c in self.classifications if c.status == RiskChangeStatus.NEW]

    @property
    def removed_risks(self) -> List[RiskClassification]:
        return [c for c in self.classifications if c.status == RiskChangeStatus.REMOVED]

    def summary(self) -> str:
        lines = [
            "=" * 80,
            f"  RISK CHANGE CLASSIFICATION REPORT: {self.ticker}",
            f"  Comparing: FY{self.current_year} vs FY{self.prior_year}",
            "=" * 80,
            "",
            f"  Total Risk Sections (Current Year): {self.total_current_risks}",
            f"  Total Risk Sections (Prior Year)  : {self.total_prior_risks}",
            "",
            "  Classification Breakdown:",
            f"  {'─' * 50}",
            f"  UNCHANGED (boilerplate, no action needed) : {len(self.unchanged_risks)}",
            f"  MODIFIED  (language shifted, analyze)     : {len(self.modified_risks)}",
            f"  NEW       (brand new risk disclosed)      : {len(self.new_risks)}",
            f"  REMOVED   (risk dropped from prior year)  : {len(self.removed_risks)}",
            f"  {'─' * 50}",
            "",
        ]
        return '\n'.join(lines)



class RiskChangeClassifier:
    """Produces detailed classifications with sentence-level change analysis."""

    def __init__(self, min_sentence_length: int = 40):
        self.min_sentence_length = min_sentence_length

    def classify(self, matches: List[RiskMatch], ticker: str = "",
                 current_year: int = 0, prior_year: int = 0,
                 total_current: int = 0, total_prior: int = 0) -> RiskChangeReport:
        report = RiskChangeReport(
            ticker=ticker, current_year=current_year, prior_year=prior_year,
            total_current_risks=total_current, total_prior_risks=total_prior
        )
        for match in matches:
            report.classifications.append(self._classify_match(match))
        return report

    def _classify_match(self, match: RiskMatch) -> RiskClassification:
        if match.status == RiskChangeStatus.UNCHANGED:
            return RiskClassification(
                title=match.current_section.title,
                status=RiskChangeStatus.UNCHANGED,
                body_similarity=match.body_similarity,
                combined_match_score=match.combined_score,
                current_body=match.current_section.body,
                prior_body=match.prior_section.body if match.prior_section else "",
                change_summary="No significant changes. Boilerplate retained."
            )
        elif match.status == RiskChangeStatus.MODIFIED:
            changed = self._extract_sentence_changes(
                match.current_section.body,
                match.prior_section.body if match.prior_section else ""
            )
            added = [s for s in changed if s.change_type == "added"]
            rewritten = [s for s in changed if s.change_type == "rewritten"]
            parts = []
            if added:
                parts.append(f"{len(added)} new sentence(s) added")
            if rewritten:
                parts.append(f"{len(rewritten)} sentence(s) rewritten")
            summary = f"MODIFIED: {'; '.join(parts)}." if parts else "Minor modifications."
            return RiskClassification(
                title=match.current_section.title,
                status=RiskChangeStatus.MODIFIED,
                body_similarity=match.body_similarity,
                combined_match_score=match.combined_score,
                current_body=match.current_section.body,
                prior_body=match.prior_section.body if match.prior_section else "",
                changed_sentences=changed,
                change_summary=summary
            )
        elif match.status == RiskChangeStatus.NEW:
            key = self._extract_key_sentences(match.current_section.body)
            return RiskClassification(
                title=match.current_section.title,
                status=RiskChangeStatus.NEW,
                body_similarity=0.0, combined_match_score=0.0,
                current_body=match.current_section.body,
                key_sentences=key,
                change_summary=f"NEW RISK: Entirely new disclosure with {len(key)} key sentences."
            )
        else:  # REMOVED
            return RiskClassification(
                title=match.current_section.title,
                status=RiskChangeStatus.REMOVED,
                body_similarity=0.0, combined_match_score=0.0,
                current_body="",
                prior_body=match.prior_section.body if match.prior_section else "",
                change_summary="REMOVED: Risk dropped from current filing."
            )

    def _extract_sentence_changes(self, current_body: str, prior_body: str) -> List[SentenceChange]:
        curr_sentences = self._split_sentences(current_body)
        prior_sentences = self._split_sentences(prior_body)
        prior_set = set(prior_sentences)
        changes = []

        for sent in curr_sentences:
            if len(sent) < self.min_sentence_length:
                continue
            if sent in prior_set:
                continue
            best_ratio = 0.0
            best_prior = ""
            for p_sent in prior_sentences:
                ratio = difflib.SequenceMatcher(None, sent, p_sent).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_prior = p_sent
            if best_ratio < 0.30:
                changes.append(SentenceChange(sentence=sent, change_type="added",
                                              similarity_to_prior=best_ratio))
            elif best_ratio < 0.90:
                changes.append(SentenceChange(sentence=sent, change_type="rewritten",
                                              similarity_to_prior=best_ratio, prior_sentence=best_prior))
        return changes

    def _extract_key_sentences(self, body: str) -> List[str]:
        sentences = self._split_sentences(body)
        return [s for s in sentences if len(s) >= self.min_sentence_length][:10]

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        raw = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw if len(s.strip()) > 0]


def classify_risk_changes(matches: List[RiskMatch], ticker: str = "",
                          current_year: int = 0, prior_year: int = 0,
                          total_current: int = 0, total_prior: int = 0) -> RiskChangeReport:
    """Classify all matched risk sections and produce a detailed report."""
    classifier = RiskChangeClassifier()
    return classifier.classify(matches=matches, ticker=ticker,
                               current_year=current_year, prior_year=prior_year,
                               total_current=total_current, total_prior=total_prior)
