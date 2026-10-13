"""Conservative lexical coverage and source provenance; not semantic entailment."""

import re
from src.rag.vectorstore import _tokens
from src.rag.query_normalization import normalize_query

STOP = set(
    "a an the is are was were be been being to of on in at for from with by and or as it its this that these those what which who where when why how do does did can could would should will may must many much often long minimum available used use using provide provides required deadline commitment target format mechanism time take takes after before only own s plan plans integration protocols support supported handles handle happens often quickly receiving should".split()
)
ALIASES = {
    "approves": "approval",
    "approve": "approval",
    "approved": "approval",
    "requests": "request",
    "revoked": "revoke",
    "revocation": "revoke",
    "exported": "export",
    "exports": "export",
    "retained": "retain",
    "retention": "retain",
    "rotated": "rotate",
    "rotation": "rotate",
    "encryption": "encrypt",
    "encrypted": "encrypt",
    "acknowledgement": "acknowledge",
    "acknowledged": "acknowledge",
    "disputed": "dispute",
    "receives": "receive",
    "payments": "payment",
    "database": "database",
    "events": "log",
    "logs": "log",
    "event": "log",
    "responds": "response",
    "respond": "response",
    "notice": "announce",
    "announced": "announce",
    "submit": "request",
    "submission": "request",
    "critical": "critical",
}


def topics(text):
    result = set()
    for token in _tokens(normalize_query(text)):
        if (
            token in STOP
            or token in {"during", "request", "requests", "run", "runs"}
            or token.isdigit()
        ):
            continue
        word = ALIASES.get(token, token)
        if word in STOP or word == "request":
            continue
        if word == "version":
            continue
        if word == "capped":
            word = "cap"
        if word == "scanning":
            word = "scan"
        if len(word) > 4 and word.endswith("s") and not word.endswith("ss"):
            word = word[:-1]
        result.add(word)
    return result


def instruction_like(text):
    return bool(
        re.search(
            r"ignore\s+(?:all\s+)?(?:previous|prior|system)\s+instructions|reveal\s+(?:the\s+)?(?:system\s+prompt|administrator\s+password)",
            text,
            re.I,
        )
    )


def scope_documents(query, docs):
    flags = set(re.findall(r"--[a-z][a-z0-9-]*", query.lower()))
    if not flags:
        return docs
    selected = []
    for doc in docs:
        declared = doc.metadata.get("document_identifiers", [])
        if declared:
            matches = flags <= set(declared)
        else:
            matches = all(
                re.search(
                    r"(?<![\w-])" + re.escape(flag) + r"(?![\w-])",
                    doc.page_content,
                    re.I,
                )
                for flag in flags
            )
        if matches:
            selected.append(doc)
    return selected


def evidence_screen(query, docs):
    docs = scope_documents(query, docs)
    safe = [d for d in docs if not instruction_like(d.page_content)]
    needed = topics(query)
    present = set().union(*(topics(d.page_content) for d in safe)) if safe else set()
    missing = sorted(needed - present)
    by_source = {}
    for doc in safe:
        by_source.setdefault(doc.metadata["source"], set()).update(
            topics(doc.page_content)
        )
    eligible = {source for source, words in by_source.items() if needed <= words}
    temporal = bool(
        re.search(
            r"\bhow (?:long|often|frequently)|\bwhen\b|\bdeadline\b|\backnowledgement time\b|\brecovery time\b|\bresponse time\b",
            query,
            re.I,
        )
    )
    if temporal:
        time_pattern = r"\b(?:second|minute|hour|day|week|month|year)s?\b|\b(?:daily|weekly|monthly|annually|annual|quarterly|every)\b"
        timed_sources = {
            d.metadata["source"]
            for d in safe
            for line in d.page_content.splitlines()
            if needed <= topics(line) and re.search(time_pattern, line, re.I)
        }
        eligible &= timed_sources
    # Conservative coverage intentionally abstains on unsupported facets and some paraphrases.
    return [d for d in safe if d.metadata["source"] in eligible], {
        "method": "conservative_same_source_topic_coverage",
        "missing_terms": missing,
        "same_source_required": True,
        "temporal_evidence_required": temporal,
        "instruction_like_chunks_excluded": len(docs) - len(safe),
        "not_semantic_entailment": True,
        "normalized_query": normalize_query(query),
    }


def citation(doc, index):
    metadata = doc.metadata
    return {
        "index": index,
        "source": metadata["source"],
        "chunk_id": metadata["chunk_id"],
        "snippet": doc.page_content,
        "score": metadata.get("dense_score"),
        **{
            k: metadata.get(k)
            for k in [
                "upstream_url",
                "upstream_revision",
                "source_sha256",
                "start_char",
                "end_char",
                "start_line",
                "end_line",
            ]
        },
    }
