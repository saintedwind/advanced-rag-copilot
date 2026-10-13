"""Live loopback HTTP and real embedding check. No paid API calls."""

import json
import os
import socket
import subprocess
import sys
import time
import sys
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[1]
OPENER = build_opener(ProxyHandler({}))
CASES = [
    ("What uptime commitment does the Business plan provide?", "cloud_sla.md", "99.95"),
    ("How much notice is required for scheduled maintenance?", "cloud_sla.md", "72"),
    ("How long are audit logs retained on Business plans?", "security_guide.md", "365"),
    ("Which format is available for exported audit logs?", "security_guide.md", "JSON"),
    (
        "What protocols support enterprise identity integration?",
        "security_guide.md",
        "SAML",
    ),
    ("How often are enterprise encryption keys rotated?", "security_guide.md", "90"),
    ("What is the deadline to submit a service credit request?", "cloud_sla.md", "30"),
    (
        "What is the critical incident support response for the Business plan?",
        "cloud_sla.md",
        "one-hour",
    ),
    ("What is the capital of Argentina?", None, None),
    ("What is the CEO personal bank account number?", None, None),
    ("What is the price of a Bitcoin today?", None, None),
]


def request(path, body=None):
    req = Request(
        "http://127.0.0.1:8000" + path,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    with OPENER.open(req, timeout=90) as r:
        return json.load(r)


def main():
    mode = os.environ.get("LLM_PROVIDER", "extractive")
    evaluation = "--evaluation" in sys.argv
    cases = CASES
    if evaluation:
        from scripts.evidence_benchmark import CASES as evaluation_cases

        cases = evaluation_cases
        os.environ["SOURCE_ROOT"] = str(ROOT / "data/evaluation_docs")
        os.environ["QDRANT_PATH"] = str(ROOT / ".runtime/qdrant-evaluation")
    label = ("evaluation-" if evaluation else "") + mode
    sock = socket.socket()
    try:
        sock.bind(("127.0.0.1", 8000))
    finally:
        sock.close()
    log_path = ROOT / "reports" / f"live-{label}.log"
    rows = []
    with log_path.open("w", encoding="utf-8") as log:
        p = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "src.app.api:app",
                "--host",
                "127.0.0.1",
                "--port",
                "8000",
            ],
            cwd=ROOT,
            stdout=log,
            stderr=log,
        )
        try:
            deadline = time.monotonic() + 300
            while True:
                if p.poll() is not None:
                    raise RuntimeError(f"Service exited; inspect {log_path}")
                try:
                    health = request("/health")
                    break
                except OSError:
                    if time.monotonic() > deadline:
                        raise TimeoutError("Startup timed out")
                    time.sleep(1)
            ingestion = request("/ingest", {})
            for q, source, phrase in cases:
                r = request(
                    "/query",
                    {
                        "query": q,
                        "top_k": 3,
                        "enable_reranking": False,
                        "use_agentic_flow": True,
                    },
                )
                sources = [c["source"] for c in r["citations"]]
                for citation in r["citations"]:
                    stored = request("/evidence/" + citation["chunk_id"])
                    assert stored["text"] == citation["snippet"]
                    assert (
                        stored["metadata"]["source_sha256"] == citation["source_sha256"]
                    )
                passed = (
                    (r["status"] == "insufficient_evidence")
                    if source is None
                    else (
                        r["status"] == "evidence_found"
                        and source in sources
                        and phrase.lower() in r["answer"].lower()
                    )
                )
                rows.append(
                    {
                        "question": q,
                        "expected_source": source,
                        "expected_phrase": phrase,
                        "passed": passed,
                        **r,
                    }
                )
            cached = request(
                "/query",
                {
                    "query": cases[0][0],
                    "top_k": 3,
                    "enable_reranking": False,
                    "use_agentic_flow": True,
                },
            )
            result = {
                "date": time.strftime("%Y-%m-%d"),
                "mode": mode,
                "transport": "actual loopback HTTP / uvicorn",
                "embedding": "actual FastEmbed BAAI/bge-small-en-v1.5",
                "synthetic_corpus": True,
                "health": health,
                "ingestion": ingestion,
                "passed": sum(r["passed"] for r in rows),
                "total": len(rows),
                "cache_verified": cached["cache_hit"],
                "cases": rows,
            }
            (ROOT / "reports" / f"live-{label}.json").write_text(
                json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            print(
                json.dumps({k: v for k, v in result.items() if k != "cases"}, indent=2)
            )
        finally:
            p.terminate()
            try:
                p.wait(timeout=15)
            except subprocess.TimeoutExpired:
                p.kill()
                p.wait(timeout=5)


if __name__ == "__main__":
    main()
