import hashlib
from pathlib import Path
import pytest
from langchain_core.documents import Document
from src.rag.query_normalization import normalize_query
from src.rag.evidence import scope_documents
from src.rag.ingestion import load_and_split
from src.rag.corpora import verify_curl
from src.rag.span_reader import SpanReader
from src.app.schemas import IngestionRequest


def test_normalization_preserves_quoted_identifiers_numbers_and_unknown_facets():
    result = normalize_query(
        'Who is responsible for escalating an incident at 14:30? "backup copies" `--max-time`'
    )
    assert "owns escalation" in result and "14:30" in result
    assert '"backup copies"' in result and "`--max-time`" in result
    assert "vendor" in normalize_query("Which vendor stores backup copies?")


def test_known_paraphrases_resolve_without_query_specific_answer_text():
    assert (
        normalize_query("How frequently are backup copies generated?")
        == "How often are snapshots taken?"
    )
    assert (
        normalize_query("Who is responsible for escalating an incident?")
        == "Who owns escalation an incident?"
    )


def test_named_option_cannot_use_a_neighbor_that_only_mentions_it():
    wrong = Document(
        page_content="See --connect-timeout. Decimal values allowed.",
        metadata={"document_identifiers": ["--max-time"]},
    )
    right = Document(
        page_content="Decimal values allowed.",
        metadata={"document_identifiers": ["--connect-timeout"]},
    )
    assert scope_documents(
        "Does --connect-timeout accept decimals?", [wrong, right]
    ) == [right]


def test_reference_offsets_still_point_into_unmodified_upstream_source():
    directory, manifest = verify_curl()
    chunks, _ = load_and_split(directory, allowed_root=directory)
    for chunk in chunks:
        raw = (directory / chunk.metadata["source"]).read_bytes()
        text = raw.decode().replace("\r\n", "\n").replace("\r", "\n")
        m = chunk.metadata
        assert text[m["start_char"] : m["end_char"]] == chunk.page_content
        assert m["source_sha256"] == hashlib.sha256(raw).hexdigest()
        assert "SPDX-License-Identifier" not in chunk.page_content
        assert m["document_identifiers"]


def test_missing_protected_facet_never_loads_reader(tmp_path, monkeypatch):
    reader = SpanReader(tmp_path)

    def forbidden():
        raise AssertionError("Unexpected model load")

    monkeypatch.setattr(reader, "load", forbidden)
    selected, status = reader.rescue(
        "Which vendor stores snapshots?", [], {"missing_terms": ["vendor"]}
    )
    assert not selected and status["status"] == "disabled_or_protected_facet"


def test_reader_failure_is_explicit_and_cannot_override_option_scope(
    tmp_path, monkeypatch
):
    reader = SpanReader(tmp_path)
    wrong = Document(
        page_content="--connect-timeout is mentioned.",
        metadata={"document_identifiers": ["--max-time"]},
    )
    selected, status = reader.rescue("What is --connect-timeout?", [wrong], {})
    assert not selected and status["status"] == "no_safe_candidates"
    good = Document(
        page_content="Certificate format is PEM.", metadata={"chunk_id": "one"}
    )
    monkeypatch.setattr(
        reader, "load", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    selected, status = reader.rescue("What certificate format?", [good], {})
    assert not selected and status["status"] == "reader_unavailable"


def test_named_collection_does_not_accept_an_arbitrary_path():
    with pytest.raises(ValueError):
        IngestionRequest(collection="curl", source_dir="../")


def test_reader_cannot_return_text_outside_its_source(tmp_path, monkeypatch):
    reader = SpanReader(tmp_path)
    monkeypatch.setattr(reader, "load", lambda: None)
    monkeypatch.setattr(
        reader,
        "score",
        lambda *args: {
            "margin": 20.0,
            "start_char": 0,
            "end_char": 3,
            "text": "invented",
        },
    )
    selected, status = reader.rescue(
        "What certificate format?",
        [Document(page_content="PEM certificate format", metadata={"chunk_id": "one"})],
        {},
    )
    assert not selected and status["status"] == "no_answer_span"


def test_transient_reader_failure_is_not_cached(tmp_path, monkeypatch):
    from fastapi.testclient import TestClient
    from tests.test_desk import Embeddings
    from src.app.dependencies import AppSettings
    from src.app.api import create_app
    from src.rag.pipeline import RAGPipeline
    from src.rag.vectorstore import HybridRetriever

    root = tmp_path / "docs"
    root.mkdir()
    (root / "sla.md").write_text("Business uptime is 99.95 percent.", encoding="utf-8")
    settings = AppSettings(source_root=str(root), qdrant_path=str(tmp_path / "db"))
    retriever = HybridRetriever(settings, embedder=Embeddings())
    pipeline = RAGPipeline(retriever, settings=settings)
    monkeypatch.setattr(
        pipeline.reader, "load", lambda: (_ for _ in ()).throw(RuntimeError("offline"))
    )
    try:
        with TestClient(create_app(pipeline, settings)) as client:
            client.post("/ingest", json={}).raise_for_status()
            for _ in range(2):
                result = client.post(
                    "/query",
                    json={"query": "uptime warranty", "use_agentic_flow": False},
                ).json()
                assert not result["cache_hit"]
                assert (
                    result["evidence_screen"]["reader"]["status"]
                    == "reader_unavailable"
                )
    finally:
        retriever.close()
