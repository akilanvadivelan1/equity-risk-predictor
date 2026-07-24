# =========================================================
# ERPSA - Step 2D (Improved): Risk Section Parser
# =========================================================
# Parses Item 1A (Risk Factors) text into individually titled
# risk sections. Detects title/header patterns and groups
# content under each risk topic.
# =========================================================

import re
from dataclasses import dataclass, field
from typing import List


@dataclass
class RiskSection:
    """Represents a single titled risk section from Item 1A."""
    title: str
    body: str
    title_keywords: set = field(default_factory=set)

    def __post_init__(self):
        if not self.title_keywords:
            self.title_keywords = self._extract_keywords(self.title)

    @staticmethod
    def _extract_keywords(text: str) -> set:
        """Extract meaningful keywords from a title, filtering stopwords."""
        stopwords = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'from', 'is', 'are', 'was', 'were',
            'be', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did',
            'will', 'would', 'could', 'should', 'may', 'might', 'shall',
            'can', 'our', 'we', 'us', 'its', 'their', 'that', 'this',
            'these', 'those', 'it', 'if', 'not', 'no', 'as', 'such',
            'related', 'risks', 'risk', 'certain', 'other', 'general',
            'factors', 'regarding', 'concerning', 'associated', 'about'
        }
        words = re.findall(r'[a-z]+', text.lower())
        return {w for w in words if w not in stopwords and len(w) > 2}


class RiskSectionParser:
    """
    Parses Item 1A text into structured risk sections by detecting
    title/header patterns used in SEC 10-K filings.
    """

    TITLE_PATTERNS = [
        re.compile(r'^[A-Z][A-Z\s,&\-\/\(\)\'\"]{15,150}$'),
        re.compile(r'^Risks?\s+(Related\s+to|Regarding|Concerning|Associated\s+with)\b', re.IGNORECASE),
        re.compile(r'^[A-Z][\w\s,&\-\/\(\)\'\"]{5,80}(Risk|Risks|Uncertainty|Uncertainties)\s*$', re.IGNORECASE),
    ]

    ANTI_TITLE_PATTERNS = [
        re.compile(r'^\d'),
        re.compile(r'^(Item|ITEM)\s+\d'),
        re.compile(r'^(See|Note|For\s+more|Refer\s+to)', re.IGNORECASE),
    ]

    def __init__(self, min_body_length: int = 100):
        self.min_body_length = min_body_length

    def parse(self, text: str) -> List[RiskSection]:
        """Parse Item 1A text into a list of RiskSection objects."""
        if '\n' in text:
            sections = self._parse_structured(text)
            if len(sections) >= 2:
                return sections

        sections = self._parse_flat_by_title_patterns(text)
        if len(sections) >= 2:
            return sections

        return self._parse_by_heuristic_breaks(text)

    def _parse_structured(self, text: str) -> List[RiskSection]:
        """Parse text with paragraph breaks."""
        paragraphs = [p.strip() for p in re.split(r'\n\n+', text) if p.strip()]
        sections = []
        current_title = None
        current_body_parts = []

        for para in paragraphs:
            if self._is_likely_title(para):
                if current_title is not None and current_body_parts:
                    body = ' '.join(current_body_parts)
                    if len(body) >= self.min_body_length:
                        sections.append(RiskSection(title=current_title, body=body))
                current_title = para.strip()
                current_body_parts = []
            else:
                if current_title is None:
                    current_title = self._generate_title_from_content(para)
                current_body_parts.append(para)

        if current_title is not None and current_body_parts:
            body = ' '.join(current_body_parts)
            if len(body) >= self.min_body_length:
                sections.append(RiskSection(title=current_title, body=body))

        return sections

    def _parse_flat_by_title_patterns(self, text: str) -> List[RiskSection]:
        """Parse flat text by detecting title-like phrases."""
        title_regex = re.compile(
            r'(Risks?\s+(?:Related\s+to|Regarding|Concerning|Associated\s+with)\s+[\w\s,&\-\/\(\)\'\"]+?)(?=\s+(?:We|Our|The|If|A\s|In\s|This))',
            re.IGNORECASE
        )
        title_matches = list(title_regex.finditer(text))

        if not title_matches:
            return self._parse_flat_by_sentence_headers(text)

        sections = []
        for i, match in enumerate(title_matches):
            title = match.group(1).strip()
            body_start = match.end()
            body_end = title_matches[i + 1].start() if i + 1 < len(title_matches) else len(text)
            body = text[body_start:body_end].strip()
            if len(body) >= self.min_body_length:
                sections.append(RiskSection(title=title, body=body))

        return sections

    def _parse_flat_by_sentence_headers(self, text: str) -> List[RiskSection]:
        """Parse flat text by detecting sentences that look like titles."""
        sentences = self._split_into_sentences(text)
        sections = []
        current_title = None
        current_body_parts = []

        for sent in sentences:
            if self._is_likely_title(sent):
                if current_title and current_body_parts:
                    body = ' '.join(current_body_parts)
                    if len(body) >= self.min_body_length:
                        sections.append(RiskSection(title=current_title, body=body))
                current_title = sent.strip()
                current_body_parts = []
            else:
                current_body_parts.append(sent)

        if current_title and current_body_parts:
            body = ' '.join(current_body_parts)
            if len(body) >= self.min_body_length:
                sections.append(RiskSection(title=current_title, body=body))

        return sections

    def _parse_by_heuristic_breaks(self, text: str) -> List[RiskSection]:
        """Fallback: split into fixed-size chunks."""
        sentences = self._split_into_sentences(text)
        sections = []
        chunk_size = 8

        for i in range(0, len(sentences), chunk_size):
            chunk = sentences[i:i + chunk_size]
            body = ' '.join(chunk)
            if len(body) < self.min_body_length:
                continue
            title = self._generate_title_from_content(chunk[0])
            sections.append(RiskSection(title=title, body=body))

        return sections

    def _is_likely_title(self, text: str) -> bool:
        """Determine if a text segment is likely a risk section title."""
        s = text.strip()
        if len(s) > 150 or len(s) < 15:
            return False

        for anti_pattern in self.ANTI_TITLE_PATTERNS:
            if anti_pattern.search(s):
                return False

        if s.endswith('.'):
            return False

        for pattern in self.TITLE_PATTERNS:
            if pattern.match(s):
                return True

        if len(s) < 100:
            words = s.split()
            if words:
                capitalized = sum(1 for w in words if w[0].isupper())
                if capitalized / len(words) > 0.6:
                    return True

        return False

    def _generate_title_from_content(self, first_sentence: str) -> str:
        title = first_sentence[:80].strip()
        if len(first_sentence) > 80:
            title = title.rsplit(' ', 1)[0] + "..."
        return title

    @staticmethod
    def _split_into_sentences(text: str) -> List[str]:
        raw_sentences = re.split(r'(?<=[.!?])\s+', text)
        return [s.strip() for s in raw_sentences if len(s.strip()) > 0]


def parse_risk_sections(cleaned_text: str, min_body_length: int = 100) -> List[RiskSection]:
    """
    Parse cleaned Item 1A text into structured risk sections.

    Args:
        cleaned_text: Output of clean_text() or clean_text_preserve_structure().
        min_body_length: Minimum chars for a section body to be valid.

    Returns:
        List of RiskSection objects with title, body, and keywords.
    """
    parser = RiskSectionParser(min_body_length=min_body_length)
    return parser.parse(cleaned_text)
