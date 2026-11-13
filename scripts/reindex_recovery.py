"""Exercise interrupted re-indexing over loopback with real curl data and embeddings."""

import argparse
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
QUESTION = {"query": "What certificate format does --cacert require?", "top_k": 3}
FAULT = """
import os
import uvicorn
from src.app.api import create_app
from src.rag.pipeline import RAGPipeline

pipeline = RAGPipeline()
client = pipeline.retriever.client
write = client.upsert
def partial_write(collection_name, points, **options):
    point = next(p for p in points if p.payload['metadata']['source'] == 'location.md')
    write(collection_name, points=[point], **options)
    print('Injected failure after one persisted curl point', flush=True)
    raise OSError('Simulated interruption after Qdrant acknowledged one write')
client.upsert = partial_write
try:
    uvicorn.run(create_app(pipeline), host='127.0.0.1', port=int(os.environ['RECOVERY_PORT']))
finally:
    pipeline.retriever.close()
"""


@contextmanager
def service(project, env, log_path, *, interrupt=False):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
    command = (
        ["-c", FAULT]
        if interrupt
        else [
            "-m",
            "uvicorn",
            "src.app.api:app",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    )
    with (
        log_path.open("wb") as log,
        subprocess.Popen(
            [sys.executable, *command],
            cwd=project,
            env=dict(env, RECOVERY_PORT=str(port)),
            stdout=log,
            stderr=log,
        ) as process,
    ):
        try:
            with httpx.Client(
                base_url=f"http://127.0.0.1:{port}", timeout=90, trust_env=False
            ) as http:
                for _ in range(300):
                    if process.poll() is not None:
                        raise RuntimeError(f"Service exited. See {log_path}")
                    try:
                        http.get("/health", timeout=1).raise_for_status()
                    except httpx.TransportError:
                        time.sleep(0.2)
                    else:
                        break
                else:
                    raise RuntimeError(f"Service did not start. See {log_path}")
                yield http
        finally:
            process.terminate()
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=15)


def fingerprint(http):
    health = http.get("/health").raise_for_status().json()
    return {name: health[name] for name in ("version", "chunks", "corpus_sha256")}


def check_citation(http, project):
    response = http.post("/query", json=QUESTION).raise_for_status().json()
    assert response["status"] == "evidence_found" and "PEM" in response["answer"]
    citation = next(c for c in response["citations"] if c["source"] == "cacert.md")
    original = (project / "data/curl_docs/cacert.md").read_text(encoding="utf-8")
    assert (
        original[citation["start_char"] : citation["end_char"]] == citation["snippet"]
    )
    return citation


def run(project, output):
    env = dict(os.environ, QDRANT_URL="", LLM_PROVIDER="extractive")
    with tempfile.TemporaryDirectory(prefix="reindex-", dir=output.parent) as temporary:
        run_dir = Path(temporary)
        docs = run_dir / "docs"
        docs.mkdir()
        shutil.copyfile(project / "data/curl_docs/cacert.md", docs / "cacert.md")
        env.update(QDRANT_PATH=str(run_dir / "qdrant"), SOURCE_ROOT=str(docs))
        with service(project, env, output.with_suffix(".initial.log")) as http:
            http.post("/ingest", json={}).raise_for_status()
            initial = fingerprint(http)
            check_citation(http, project)
        fault_log = output.with_suffix(".interrupted.log")
        with service(project, env, fault_log, interrupt=True) as http:
            failed = http.post("/ingest", json={"collection": "curl"})
            assert failed.status_code == 503
            after_failure = fingerprint(http)
        assert "Injected failure after one persisted curl point" in fault_log.read_text(
            encoding="utf-8"
        )
        with service(project, env, output.with_suffix(".reopened.log")) as http:
            reopened = fingerprint(http)
            check_citation(http, project)
            retry = (
                http.post("/ingest", json={"collection": "curl"})
                .raise_for_status()
                .json()
            )
            assert retry["documents_ingested"] == 8
            assert retry["corpus_sha256"] != initial["corpus_sha256"]
            active = fingerprint(http)
            assert active["corpus_sha256"] == retry["corpus_sha256"]
            full_citation = check_citation(http, project)
            assert full_citation["upstream_revision"]
        with service(project, env, output.with_suffix(".final.log")) as http:
            assert fingerprint(http) == active
            assert (
                check_citation(http, project)["chunk_id"] == full_citation["chunk_id"]
            )
        return {
            "initial": initial,
            "interrupted_http_status": failed.status_code,
            "after_failure": after_failure,
            "after_restart": reopened,
            "after_successful_retry_and_restart": active,
            "old_index_preserved": initial == after_failure == reopened,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expect-legacy", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    result = run(args.project.resolve(), output)
    text = json.dumps(result, indent=2)
    output.write_text(text + "\n", encoding="utf-8")
    print(text)
    assert result["old_index_preserved"] == (not args.expect_legacy)


if __name__ == "__main__":
    main()
