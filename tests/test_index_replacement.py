from contextlib import closing
from shutil import copyfile

from fastapi.testclient import TestClient
from qdrant_client import QdrantClient, models

from src.app.api import create_app
from src.app.dependencies import AppSettings
from src.rag.corpora import verify_curl
from src.rag.ingestion import load_and_split
from src.rag.pipeline import RAGPipeline
from src.rag.vectorstore import HybridRetriever
from tests.test_desk import Embeddings


def test_failed_reindex_keeps_the_active_corpus_after_restart(tmp_path, monkeypatch):
    upstream, _ = verify_curl()
    root = tmp_path / "docs"
    root.mkdir()
    copyfile(upstream / "cacert.md", root / "cacert.md")
    settings = AppSettings(
        source_root=str(root),
        qdrant_path=str(tmp_path / "qdrant"),
        qa_reader_provider="none",
    )
    retriever = HybridRetriever(settings, embedder=Embeddings())
    try:
        with TestClient(
            create_app(RAGPipeline(retriever, settings=settings), settings)
        ) as client:
            initial = client.post("/ingest", json={}).raise_for_status().json()
            existing = retriever._docs[0]
            receipt = client.get(f"/evidence/{existing.metadata['chunk_id']}").json()
            upsert = retriever.client.upsert

            def partial_write(collection_name, points, **options):
                point = next(
                    p
                    for p in points
                    if p.payload["metadata"]["source"] == "location.md"
                )
                upsert(collection_name, points=[point], **options)
                raise OSError("Injected interruption after one committed Qdrant point")

            with monkeypatch.context() as patch:
                patch.setattr(retriever.client, "upsert", partial_write)
                assert (
                    client.post("/ingest", json={"collection": "curl"}).status_code
                    == 503
                )
            assert (
                client.get("/health").json()["corpus_sha256"]
                == initial["corpus_sha256"]
            )
            assert (
                client.get(f"/evidence/{existing.metadata['chunk_id']}").json()
                == receipt
            )
    finally:
        retriever.close()

    reopened = HybridRetriever(settings, embedder=Embeddings())
    try:
        with TestClient(
            create_app(RAGPipeline(reopened, settings=settings), settings)
        ) as client:
            health = client.get("/health").json()
            assert health["corpus_sha256"] == initial["corpus_sha256"]
            assert health["chunks"] == initial["chunks_created"]
            assert (
                client.get(f"/evidence/{existing.metadata['chunk_id']}").json()
                == receipt
            )
            assert len(reopened.client.get_collections().collections) == 1
            replacement = (
                client.post("/ingest", json={"collection": "curl"})
                .raise_for_status()
                .json()
            )
            assert replacement["corpus_sha256"] != initial["corpus_sha256"]
            assert replacement["documents_ingested"] == 8
            assert len(reopened.client.get_collections().collections) == 1
    finally:
        reopened.close()

    final = HybridRetriever(settings, embedder=Embeddings())
    try:
        assert final.corpus_sha256 == replacement["corpus_sha256"]
        assert len(final._docs) == replacement["chunks_created"]
        assert {doc.metadata["source"] for doc in final._docs} == {
            p.name for p in upstream.glob("*.md")
        }
    finally:
        final.close()


def test_legacy_collection_loads_then_migrates_without_overwriting_it(tmp_path):
    upstream, _ = verify_curl()
    docs, _ = load_and_split(upstream, allowed_root=upstream)
    original = docs[0]
    settings = AppSettings(qdrant_path=str(tmp_path / "qdrant"))
    with closing(QdrantClient(path=settings.qdrant_path)) as client:
        client.create_collection(
            settings.qdrant_collection,
            vectors_config=models.VectorParams(size=3, distance=models.Distance.COSINE),
        )
        client.upsert(
            settings.qdrant_collection,
            [
                models.PointStruct(
                    id=1,
                    vector=Embeddings().encode(original.page_content).tolist(),
                    payload={
                        "text": original.page_content,
                        "metadata": original.metadata,
                        "embedding_model": settings.embedding_model,
                    },
                )
            ],
            wait=True,
        )
    migrated = HybridRetriever(settings, embedder=Embeddings())
    try:
        assert (
            migrated.search("certificate", k=1)[0].page_content == original.page_content
        )
        migrated.add_documents(docs)
        assert len(migrated._docs) == len(docs)
        assert migrated.client.count(settings.qdrant_collection, exact=True).count == 1
        fingerprint = migrated.corpus_sha256
    finally:
        migrated.close()
    reopened = HybridRetriever(settings, embedder=Embeddings())
    try:
        assert reopened.corpus_sha256 == fingerprint
        assert len(reopened._docs) == len(docs)
    finally:
        reopened.close()
