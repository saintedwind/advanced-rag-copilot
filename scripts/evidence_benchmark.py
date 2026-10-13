"""Real-embedding development benchmark with positive and related-unanswerable cases."""

import hashlib
import json
import sys
from pathlib import Path
from qdrant_client import QdrantClient
from src.app.dependencies import AppSettings
from src.rag.ingestion import load_and_split
from src.rag.pipeline import RAGPipeline
from src.rag.vectorstore import HybridRetriever

ROOT = Path(__file__).resolve().parents[1]
CHALLENGE_CASES = [
    ("How often are dependency scans run?", "security.md", "every merge request"),
    ("How long are Standard audit events retained?", "privacy.md", "180 days"),
    ("Who approves refund requests?", "billing.md", "billing owner"),
    ("How often are access reviews run?", "security.md", "quarterly"),
    ("How frequently are backup copies generated?", "backup.md", "six hours"),
    (
        "Who is responsible for escalating an incident?",
        "incidents.md",
        "incident commander",
    ),
    ("How long does refund approval take?", None, None),
    ("How long is API token retention?", None, None),
]
CASES = [
    ("How often are database snapshots taken?", "backup.md", "six hours"),
    ("How long are snapshots retained?", "backup.md", "35 days"),
    ("Who approves restore requests?", "backup.md", "database operator"),
    ("What is the recovery time objective?", "backup.md", "four hours"),
    ("How often are restore drills run?", "backup.md", "quarterly"),
    ("What is the Standard API rate limit?", "api.md", "120 requests"),
    ("Which header should clients honor after HTTP 429?", "api.md", "Retry-After"),
    ("When do idempotency keys expire?", "api.md", "24 hours"),
    ("Who can revoke API tokens?", "api.md", "administrators"),
    ("Which mechanism is used for pagination?", "api.md", "opaque cursor"),
    ("How long does workspace deletion take?", "privacy.md", "30 days"),
    ("Which format is used for data exports?", "privacy.md", "CSV"),
    ("Where is personal data hosted?", "privacy.md", "EU region"),
    ("What happens to deletion during legal holds?", "privacy.md", "suspend deletion"),
    ("What is the severity-one acknowledgement time?", "incidents.md", "15 minutes"),
    ("How often are incident status updates posted?", "incidents.md", "30 minutes"),
    ("When are postmortems delivered?", "incidents.md", "five business days"),
    ("Who owns incident escalation?", "incidents.md", "incident commander"),
    ("What is the minimum TLS version?", "security.md", "1.2"),
    ("How many reviewers approve production access?", "security.md", "two reviewers"),
    ("When does an emergency access session expire?", "security.md", "60 minutes"),
    ("What is the invoice payment deadline?", "billing.md", "14 calendar days"),
    ("What is the invoice dispute deadline?", "billing.md", "seven calendar days"),
    ("What is the Business service credit cap?", "billing.md", "20 percent"),
    (
        "Which languages does the support desk handle?",
        "localization.md",
        "English and French",
    ),
    (
        "Which language is emergency phone support available in?",
        "localization.md",
        "English",
    ),
    ("What is the backup storage vendor?", None, None),
    ("What is the API GPU quota?", None, None),
    ("What is the workspace deletion fee?", None, None),
    ("What is the incident commander's salary?", None, None),
    ("What is the hardware security key vendor?", None, None),
    ("Which bank account receives invoice payments?", None, None),
    ("What is the administrator password?", None, None),
    ("Which languages are supported by the mobile application?", None, None),
    ("What is the capital of Argentina?", None, None),
    ("What is the price of Bitcoin today?", None, None),
]


def main(label="current"):
    root = ROOT / "data/evaluation_docs"
    settings = AppSettings(source_root=str(root))
    retriever = HybridRetriever(settings, client=QdrantClient(":memory:"))
    rows = []
    try:
        chunks, count = load_and_split(root, allowed_root=root)
        retriever.add_documents(chunks)
        pipeline = RAGPipeline(retriever, settings=settings)
        cases = CHALLENGE_CASES if label == "challenge" else CASES
        for question, source, phrase in cases:
            result = pipeline.answer(question, 3, False)
            for c in result["citations"]:
                raw = (root / c["source"]).read_bytes()
                text = raw.decode().replace("\r\n", "\n").replace("\r", "\n")
                assert hashlib.sha256(raw).hexdigest() == c["source_sha256"]
                assert text[c["start_char"] : c["end_char"]] == c["snippet"]
            passed = (
                result["status"] == "insufficient_evidence"
                if source is None
                else (
                    result["status"] == "evidence_found"
                    and source in [c["source"] for c in result["citations"]]
                    and phrase.lower() in result["answer"].lower()
                )
            )
            rows.append(
                {
                    "question": question,
                    "expected_source": source,
                    "expected_phrase": phrase,
                    "passed": passed,
                    "result": result,
                }
            )
        manifest = {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.glob("*.md"))
        }
        report = {
            "scope": "authored synthetic English development cases; not a blind holdout or customer accuracy estimate",
            "embedding": settings.embedding_model,
            "provider": settings.llm_provider,
            "documents": count,
            "source_manifest": manifest,
            "total": len(rows),
            "passed": sum(x["passed"] for x in rows),
            "supported_passed": sum(x["passed"] for x in rows if x["expected_source"]),
            "supported_total": sum(bool(x["expected_source"]) for x in rows),
            "abstention_passed": sum(
                x["passed"] for x in rows if not x["expected_source"]
            ),
            "abstention_total": sum(not x["expected_source"] for x in rows),
            "cases": rows,
        }
        (ROOT / f"reports/evidence-{label}.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    k: v
                    for k, v in report.items()
                    if k not in ["cases", "source_manifest"]
                },
                indent=2,
            )
        )
        print("Failed:", [r["question"] for r in rows if not r["passed"]])
    finally:
        retriever.close()


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "current")
