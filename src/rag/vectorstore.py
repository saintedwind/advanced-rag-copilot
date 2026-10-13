"""Persistent Qdrant vectors and rebuilt BM25 with weighted rank fusion."""

import re
import threading
import unicodedata
import uuid

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from src.app.dependencies import get_settings


def _tokens(text):
    text = (
        unicodedata.normalize("NFKC", text)
        .lower()
        .translate(str.maketrans({"\u064a": "\u06cc", "\u0643": "\u06a9"}))
    )
    return re.findall(r"[^\W_]+", text.replace("\u200c", " "), re.UNICODE)


class HybridRetriever:
    def __init__(self, settings=None, embedder=None, client=None):
        self.settings = settings or get_settings()
        self.lock = threading.RLock()
        self.client = client or (
            QdrantClient(
                url=self.settings.qdrant_url,
                api_key=self.settings.qdrant_api_key or None,
            )
            if self.settings.qdrant_url
            else QdrantClient(path=self.settings.qdrant_path)
        )
        if embedder is None:
            from fastembed import TextEmbedding

            embedder = TextEmbedding(
                model_name=self.settings.embedding_model,
                cache_dir=self.settings.embedding_cache,
                threads=2,
            )
        self.embedder = embedder
        self._docs = []
        self._bm25 = None
        self.generation = 0
        self.refresh()

    def refresh(self):
        self._docs = []
        if self.client.collection_exists(self.settings.qdrant_collection):
            offset = None
            while True:
                rows, offset = self.client.scroll(
                    self.settings.qdrant_collection,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                )
                if any(
                    r.payload.get("embedding_model") != self.settings.embedding_model
                    for r in rows
                ):
                    raise ValueError(
                        "Embedding model mismatch; select a new collection or rebuild the index"
                    )
                self._docs.extend(
                    Document(
                        page_content=r.payload["text"], metadata=r.payload["metadata"]
                    )
                    for r in rows
                )
                if offset is None:
                    break
        self._docs.sort(key=lambda d: d.metadata["chunk_id"])
        self._bm25 = (
            BM25Okapi([_tokens(d.page_content) for d in self._docs])
            if self._docs
            else None
        )
        self.generation += 1

    def add_documents(self, docs):
        if not docs:
            raise ValueError("No non-empty documents to ingest")
        with self.lock:
            vectors = list(self.embedder.embed([d.page_content for d in docs]))
            if len(vectors) != len(docs):
                raise ValueError("Embedding count does not match document count")
            if not self.client.collection_exists(self.settings.qdrant_collection):
                self.client.create_collection(
                    self.settings.qdrant_collection,
                    vectors_config=models.VectorParams(
                        size=len(vectors[0]), distance=models.Distance.COSINE
                    ),
                )
            # Replace the selected corpus, removing stale chunks from modified sources.
            points = [
                models.PointStruct(
                    id=str(uuid.uuid5(uuid.NAMESPACE_URL, d.metadata["chunk_id"])),
                    vector=v.tolist(),
                    payload={
                        "text": d.page_content,
                        "metadata": d.metadata,
                        "embedding_model": self.settings.embedding_model,
                    },
                )
                for d, v in zip(docs, vectors)
            ]
            self.client.upsert(
                self.settings.qdrant_collection, points=points, wait=True
            )
            keep = [p.id for p in points]
            self.client.delete(
                self.settings.qdrant_collection,
                points_selector=models.FilterSelector(
                    filter=models.Filter(must_not=[models.HasIdCondition(has_id=keep)])
                ),
                wait=True,
            )
            self.refresh()

    def search(self, query, k=10, alpha=None):
        with self.lock:
            if not self._docs:
                return []
            vector = next(iter(self.embedder.query_embed(query))).tolist()
            hits = self.client.query_points(
                self.settings.qdrant_collection,
                query=vector,
                limit=k,
                with_payload=True,
            ).points
            dense = [
                Document(
                    page_content=h.payload["text"],
                    metadata={**h.payload["metadata"], "dense_score": h.score},
                )
                for h in hits
            ]
            scores = self._bm25.get_scores(_tokens(query))
            lexical = [
                self._docs[i]
                for i in sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
                if scores[i] > 0
            ]
            weight = self.settings.hybrid_alpha if alpha is None else alpha
            if not 0 <= weight <= 1:
                raise ValueError("Alpha must be in [0,1]")
            fused = {}
            for ranking, factor in ((dense, weight), (lexical, 1 - weight)):
                if factor == 0:
                    continue
                for rank, doc in enumerate(ranking, 1):
                    key = doc.metadata["chunk_id"]
                    old = fused.get(key, (doc, 0))
                    fused[key] = (old[0], old[1] + factor / (60 + rank))
            return [
                Document(
                    page_content=d.page_content,
                    metadata={**d.metadata, "rrf_score": score},
                )
                for d, score in sorted(
                    fused.values(), key=lambda p: (-p[1], p[0].metadata["chunk_id"])
                )[:k]
            ]

    def close(self):
        self.client.close()
