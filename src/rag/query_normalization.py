"""Small, auditable English wording normalization; no inferred facts or model call."""

import re

# Domain wording, not a learned paraphraser. Keep quoted identifiers untouched.
RULES = [
    (r"\bbackup copies\b", "snapshots"),
    (r"\bbackups\b", "snapshots"),
    (r"\bis responsible for\b", "owns"),
    (r"\bresponsible for\b", "owns"),
    (r"\bescalating\b", "escalation"),
    (r"\bfrequently\b", "often"),
    (r"\b(?:generated|created)\b", "taken"),
]


def normalize_query(text):
    pieces = re.split(r'(`[^`]*`|"[^"]*")', text)
    for i in range(0, len(pieces), 2):
        for pattern, replacement in RULES:
            pieces[i] = re.sub(pattern, replacement, pieces[i], flags=re.I)
    return " ".join("".join(pieces).split())
