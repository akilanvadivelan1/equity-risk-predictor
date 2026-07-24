# =========================================================
# ERPSA - Shared Text Cleaning Utilities
# =========================================================

import re
from html import unescape


def clean_text(text: str) -> str:
    """Strip HTML tags and excess whitespace from raw SEC filing text.
    
    Uses pure Python (no bs4 dependency) for portability.
    """
    # Remove HTML comments
    cleaned = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove script and style blocks entirely
    cleaned = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with spaces
    cleaned = re.sub(r'<(br|p|div|li|tr|td|th|h[1-6])[^>]*/?\s*>', ' ', cleaned, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Decode HTML entities
    cleaned = unescape(cleaned)
    # Replace multiple spaces/newlines with a single space
    cleaned = re.sub(r'\s+', ' ', cleaned)
    # Remove navigation boilerplate
    cleaned = re.sub(r'Table of Contents', '', cleaned, flags=re.IGNORECASE)
    return cleaned.strip()


def clean_text_preserve_structure(text: str) -> str:
    """Strip HTML but preserve paragraph boundaries as double-newlines.
    
    This variant keeps structural breaks between sections, which is 
    critical for the risk section parser to detect title boundaries.
    """
    # Remove HTML comments
    cleaned = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
    # Remove script and style blocks
    cleaned = re.sub(r'<(script|style)[^>]*>.*?</\1>', '', cleaned, flags=re.DOTALL | re.IGNORECASE)
    # Replace block-level tags with double newlines (preserve structure)
    cleaned = re.sub(r'</?(p|div|li|tr|h[1-6]|blockquote|section|article)[^>]*>', '\n\n', cleaned, flags=re.IGNORECASE)
    # Replace <br> with single newline
    cleaned = re.sub(r'<br[^>]*/?\s*>', '\n', cleaned, flags=re.IGNORECASE)
    # Remove all remaining HTML tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Decode HTML entities
    cleaned = unescape(cleaned)
    # Collapse multiple newlines to exactly two
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    # Collapse multiple spaces on same line to one
    cleaned = re.sub(r'[^\S\n]+', ' ', cleaned)
    # Remove boilerplate
    cleaned = re.sub(r'Table of Contents', '', cleaned, flags=re.IGNORECASE)
    # Strip leading/trailing whitespace per line
    lines = [line.strip() for line in cleaned.split('\n')]
    cleaned = '\n'.join(lines)
    # Collapse multiple newlines again
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    return cleaned.strip()
