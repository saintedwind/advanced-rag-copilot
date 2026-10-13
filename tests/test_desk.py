import json

import numpy as np
import pytest
from fastapi.testclient import TestClient
from langchain_core.documents import Document

from src.app.api import create_app
from src.app.dependencies import AppSettings
from src.rag.graph import build_rag_graph
from src.rag.ingestion import load_and_split
from src.rag.pipeline import RAGPipeline
from src.rag.vectorstore import HybridRetriever, _tokens


class Embeddings:
    """Deterministic test double, NOT a neural retrieval benchmark."""

    def encode(self, text):
        return np.array(
            [1.0, 0.0, 0.0]
            if "uptime" in text.lower()
            else [0.0, 1.0, 0.0]
            if "audit" in text.lower()
            else [0.0, 0.0, 1.0]
        )

    def embed(self, texts):
        return iter(self.encode(t) for t in texts)

    def query_embed(self, text):
        return iter([self.encode(text)])


@pytest.fixture
def setup(tmp_path):
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "sla.md").write_text(
        "Business uptime is 99.95 percent. Critical response is one hour.",
        encoding="utf-8",
    )
    (docs / "security.md").write_text(
        "Audit logs are retained for 365 days.", encoding="utf-8"
    )
    s = AppSettings(
        qa_reader_provider="none",
        source_root=str(docs),
        qdrant_path=str(tmp_path / "db"),
        evidence_threshold=0.8,
    )
    r = HybridRetriever(s, embedder=Embeddings())
    yield s, r, docs
    r.close()


def index(r, docs):
    chunks, _ = load_and_split(docs, allowed_root=docs)
    r.add_documents(chunks)
    return chunks


def test_unicode():
    assert _tokens("هوش مصنوعي و كتاب") == ["هوش", "مصنوعی", "و", "کتاب"]


def test_idempotent_ingestion(setup):
    _, r, docs = setup
    first = index(r, docs)
    index(r, docs)
    assert len(r._docs) == len(first) == 2
    assert {d.metadata["chunk_id"] for d in r._docs} == {
        d.metadata["chunk_id"] for d in first
    }


def test_removed_source_is_deleted(setup):
    _, r, docs = setup
    index(r, docs)
    (docs / "security.md").unlink()
    index(r, docs)
    assert len(r._docs) == 1
    assert r._docs[0].metadata["source"] == "sla.md"


def test_changed_source_replaces_chunk(setup):
    _, r, docs = setup
    first = index(r, docs)
    (docs / "sla.md").write_text("Business uptime is 99.99 percent.", encoding="utf-8")
    index(r, docs)
    assert len(r._docs) == 2
    old = next(d for d in first if d.metadata["source"] == "sla.md")
    assert old.metadata["chunk_id"] not in {d.metadata["chunk_id"] for d in r._docs}


def test_restart_rebuilds_lexical_index(setup):
    s, r, docs = setup
    index(r, docs)
    r.close()
    r2 = HybridRetriever(s, embedder=Embeddings())
    try:
        assert len(r2._docs) == 2 and r2._bm25 is not None
        assert r2.search("uptime", k=1)[0].metadata["source"] == "sla.md"
    finally:
        r2.close()


def test_evidence_gate_and_no_false_high_confidence(setup):
    s, r, docs = setup
    index(r, docs)
    p = RAGPipeline(r, settings=s)
    result = p.answer("uptime", 1, False)
    assert result["status"] == "evidence_found" and result["confidence"] == "unverified"
    assert result["citations"][0]["source"] == "sla.md"
    assert p.answer("banana", 1, False)["status"] == "insufficient_evidence"


def test_graph_preserves_request_and_documents():
    class P:
        def retrieve(self, q, k, rerank):
            assert k == 2 and rerank is False
            return [Document(page_content="selected")]

        def synthesize(self, q, docs):
            assert q == "original" and docs[0].page_content == "selected"
            return {"status": "evidence_found"}

    result = build_rag_graph(P()).invoke(
        {
            "question": "original",
            "original_question": "original",
            "top_k": 2,
            "rerank": False,
            "iterations": 0,
        }
    )
    assert result["result"]["status"] == "evidence_found"


def test_path_escape_and_overlap(setup, tmp_path):
    _, _, docs = setup
    with pytest.raises(ValueError):
        load_and_split(tmp_path, allowed_root=docs)
    with pytest.raises(ValueError):
        load_and_split(docs, 100, 100, allowed_root=docs)


def test_api_cache_invalidation_and_validation(setup):
    s, r, docs = setup
    with TestClient(create_app(RAGPipeline(r, settings=s), s)) as client:
        assert client.get("/health").json()["status"] == "empty"
        assert client.post("/ingest", json={}).status_code == 200
        payload = {
            "query": "uptime",
            "top_k": 1,
            "enable_reranking": False,
            "use_agentic_flow": False,
        }
        assert not client.post("/query", json=payload).json()["cache_hit"]
        assert client.post("/query", json=payload).json()["cache_hit"]
        client.post("/ingest", json={})
        assert not client.post("/query", json=payload).json()["cache_hit"]
        assert client.post("/query", json={"query": " "}).status_code == 422
        assert client.post("/ingest", json={"source_dir": "../"}).status_code == 400
        assert client.get("/").status_code == 200


@pytest.mark.parametrize(
    "choice",
    [
        {"finish_reason": "length", "message": {"content": "{}"}},
        {
            "finish_reason": "stop",
            "message": {"content": '{"steps":[99],"supported":true}'},
        },
        {
            "finish_reason": "stop",
            "message": {"content": '{"steps":[true],"supported":true}'},
        },
        {
            "finish_reason": "stop",
            "message": {"content": '{"steps":[1,1],"supported":true}'},
        },
    ],
)
def test_bad_model_output_is_explicit_failure(setup, monkeypatch, choice):
    s, r, docs = setup
    index(r, docs)
    s.llm_provider = "local"
    p = RAGPipeline(r, settings=s)

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def read(self, *args):
            return json.dumps({"choices": [choice]}).encode()

    class Opener:
        def open(self, *args, **kwargs):
            return Response()

    monkeypatch.setattr("src.rag.pipeline.build_opener", lambda *args: Opener())
    result = p.answer("uptime", 1, False)
    assert (
        result["status"] == "generation_unavailable" and result["confidence"] == "low"
    )


def test_no_context_never_calls_model(setup, monkeypatch):
    s, r, docs = setup
    index(r, docs)
    s.llm_provider = "local"
    p = RAGPipeline(r, settings=s)

    def forbidden(*args):
        raise AssertionError("Unexpected model call")

    monkeypatch.setattr("src.rag.pipeline.build_opener", forbidden)
    assert p.answer("banana", 1, False)["status"] == "insufficient_evidence"
