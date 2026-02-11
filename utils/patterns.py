"""Regex patterns for detecting sensitive data in text."""

import re

# Phone numbers (French and international)
PHONE_PATTERNS = [
    re.compile(r"\b(?:\+33|0033|0)\s*[1-9](?:[\s.\-]?\d{2}){4}\b"),  # French
    re.compile(r"\b\+?\d{1,3}[\s.\-]?\(?\d{1,4}\)?(?:[\s.\-]?\d{2,4}){2,4}\b"),  # International
]

# Email addresses
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b")

# French business identifiers
SIRET_PATTERN = re.compile(r"\b\d{3}[\s.]?\d{3}[\s.]?\d{3}[\s.]?\d{5}\b")  # 14 digits
SIREN_PATTERN = re.compile(r"\bSIREN[\s:]*\d{3}[\s.]?\d{3}[\s.]?\d{3}\b", re.IGNORECASE)
RCS_PATTERN = re.compile(r"\bRCS\s+[A-Z\s]+\d{3}[\s.]?\d{3}[\s.]?\d{3}\b", re.IGNORECASE)
TVA_PATTERN = re.compile(r"\bFR\s?\d{2}[\s.]?\d{3}[\s.]?\d{3}[\s.]?\d{3}\b")  # TVA intracommunautaire
APE_PATTERN = re.compile(r"\bAPE[\s:]*\d{4}[A-Z]\b", re.IGNORECASE)

# Postal addresses (French)
CEDEX_PATTERN = re.compile(r"\bCEDEX(?:\s+\d+)?\b", re.IGNORECASE)
POSTAL_CODE_PATTERN = re.compile(r"\b\d{5}\s+[A-Z][A-Za-zÀ-ÿ\s\-]+\b")
ADDRESS_PATTERNS = [
    re.compile(r"\b\d+[\s,]+(?:rue|avenue|boulevard|impasse|chemin|allée|place|route|voie|passage)\s+[A-Za-zÀ-ÿ\s\-]+", re.IGNORECASE),
    re.compile(r"\bBP\s*\d+\b", re.IGNORECASE),  # Boîte postale
    re.compile(r"\bZI\s+[A-Za-zÀ-ÿ\s\-]+", re.IGNORECASE),  # Zone industrielle
    re.compile(r"\bZA\s+[A-Za-zÀ-ÿ\s\-]+", re.IGNORECASE),  # Zone d'activité
]

# Website URLs
URL_PATTERN = re.compile(r"\b(?:https?://)?(?:www\.)?[a-zA-Z0-9\-]+\.[a-zA-Z]{2,}(?:/\S*)?\b")

# Fax numbers
FAX_PATTERN = re.compile(r"\b(?:fax|télécopie|t[eé]l[eé]copie)[\s:.]*(?:\+33|0)\s*[1-9](?:[\s.\-]?\d{2}){4}\b", re.IGNORECASE)


def find_sensitive_matches(text: str) -> list[dict]:
    """Scan text for all sensitive data patterns.

    Args:
        text: Text to scan.

    Returns:
        List of dicts with keys: type, match, start, end.
    """
    results = []

    for pattern in PHONE_PATTERNS:
        for m in pattern.finditer(text):
            results.append({"type": "phone", "match": m.group(), "start": m.start(), "end": m.end()})

    for m in EMAIL_PATTERN.finditer(text):
        results.append({"type": "email", "match": m.group(), "start": m.start(), "end": m.end()})

    for name, pattern in [
        ("siret", SIRET_PATTERN),
        ("siren", SIREN_PATTERN),
        ("rcs", RCS_PATTERN),
        ("tva", TVA_PATTERN),
        ("ape", APE_PATTERN),
        ("fax", FAX_PATTERN),
    ]:
        for m in pattern.finditer(text):
            results.append({"type": name, "match": m.group(), "start": m.start(), "end": m.end()})

    for m in CEDEX_PATTERN.finditer(text):
        results.append({"type": "address", "match": m.group(), "start": m.start(), "end": m.end()})

    for m in POSTAL_CODE_PATTERN.finditer(text):
        results.append({"type": "address", "match": m.group(), "start": m.start(), "end": m.end()})

    for pattern in ADDRESS_PATTERNS:
        for m in pattern.finditer(text):
            results.append({"type": "address", "match": m.group(), "start": m.start(), "end": m.end()})

    for m in URL_PATTERN.finditer(text):
        results.append({"type": "url", "match": m.group(), "start": m.start(), "end": m.end()})

    return results
