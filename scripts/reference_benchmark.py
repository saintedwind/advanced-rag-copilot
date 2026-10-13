"""Frozen real-document transfer evaluation. Questions are authored, not customer logs."""

import hashlib
import json
from pathlib import Path
from qdrant_client import QdrantClient
from src.app.dependencies import AppSettings
from src.rag.corpora import verify_curl
from src.rag.ingestion import load_and_split
from src.rag.vectorstore import HybridRetriever
from src.rag.pipeline import RAGPipeline

ROOT = Path(__file__).resolve().parents[1]
# Fixed before the first run on this corpus. Do not silently edit failed cases.
CASES = [
    ("What does --head fetch?", "head.md", "headers only"),
    ("Which HTTP command does --head use?", "head.md", "HEAD"),
    ("What does --head display for an FTP URL?", "head.md", "file size"),
    ("What does --fail return for HTTP errors?", "fail.md", "22"),
    ("Does --fail output the response body?", "fail.md", "no response body"),
    ("What HTTP response codes cause --fail errors?", "fail.md", "400 or greater"),
    (
        "Which option saves content along with the error code?",
        "fail.md",
        "--fail-with-body",
    ),
    ("What certificate format does --cacert require?", "cacert.md", "PEM"),
    (
        "Does --cacert support multiple CA certificates?",
        "cacert.md",
        "multiple CA certificates",
    ),
    (
        "Which environment variable does --cacert override?",
        "cacert.md",
        "CURL_CA_BUNDLE",
    ),
    ("What does --connect-timeout limit?", "connect-timeout.md", "connection phase"),
    (
        "Does --connect-timeout accept decimal values?",
        "connect-timeout.md",
        "decimal values",
    ),
    (
        "Which handshakes complete the connection phase?",
        "connect-timeout.md",
        "TCP, TLS or QUIC",
    ),
    (
        "Which decimal separator does --connect-timeout require?",
        "connect-timeout.md",
        "dot",
    ),
    ("What does --max-time limit?", "max-time.md", "each transfer"),
    ("Does --max-time accept decimal values?", "max-time.md", "decimal values"),
    (
        "When is the maximum time counter reset during retries?",
        "max-time.md",
        "reset each time",
    ),
    ("Which option follows redirects?", "location.md", "--location"),
    (
        "Which response codes change a POST to GET with --location?",
        "location.md",
        "301, 302, or 303",
    ),
    ("Which option limits the amount of redirects?", "location.md", "--max-redirs"),
    ("What is the default number of retries?", "retry.md", "0"),
    ("What is the initial wait before curl retries?", "retry.md", "one second"),
    (
        "How does --retry-delay change the backoff algorithm?",
        "retry-delay.md",
        "default backoff",
    ),
    ("Which header can override the retry delay?", "retry-delay.md", "Retry-After"),
    ("Which vendor sells the --cacert certificate file?", None, None),
    ("What is the monthly subscription price of --retry?", None, None),
    ("What is the --head API GPU quota?", None, None),
    ("Which bank account receives curl payments?", None, None),
    ("What is the --connect-timeout storage retention period?", None, None),
    ("What is the --location user password?", None, None),
    ("What is the capital of Argentina?", None, None),
    ("How long does --fail refund approval take?", None, None),
]


def main():
    directory, manifest = verify_curl()
    settings = AppSettings(source_root=str(directory))
    retriever = HybridRetriever(settings, client=QdrantClient(":memory:"))
    rows = []
    try:
        docs, count = load_and_split(directory, allowed_root=directory)
        retriever.add_documents(docs)
        pipeline = RAGPipeline(retriever, settings=settings)
        for question, source, phrase in CASES:
            result = pipeline.answer(question, 3, False)
            for c in result["citations"]:
                raw = (directory / c["source"]).read_bytes()
                text = raw.decode().replace("\r\n", "\n").replace("\r", "\n")
                assert hashlib.sha256(raw).hexdigest() == c["source_sha256"]
                assert text[c["start_char"] : c["end_char"]] == c["snippet"]
            passed = (
                (result["status"] == "insufficient_evidence")
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
        report = {
            "scope": "Real curl documentation, authored fixed English questions; results inspected during development, not a blind holdout or customer accuracy estimate",
            "revision": manifest["revision"],
            "source_manifest": manifest,
            "questions_sha256": hashlib.sha256(json.dumps(CASES).encode()).hexdigest(),
            "embedding": settings.embedding_model,
            "documents": count,
            "total": len(rows),
            "passed": sum(r["passed"] for r in rows),
            "supported_total": 24,
            "supported_passed": sum(r["passed"] for r in rows if r["expected_source"]),
            "abstention_total": 8,
            "abstention_passed": sum(
                r["passed"] for r in rows if not r["expected_source"]
            ),
            "cases": rows,
        }
        (ROOT / "reports/reference-curl.json").write_text(
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
    main()
