"""Persistent Qdrant vectors and rebuilt BM25 with weighted rank fusion."""

import hashlib
import json
import logging
import re
import threading
import unicodedata
import uuid

from langchain_core.documents import Document
from qdrant_client import QdrantClient, models
from rank_bm25 import BM25Okapi

from src.app.dependencies import get_settings

logger = logging.getLogger(__name__)


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
        self.generation = 0
        self._prefix = f"{self.settings.qdrant_collection}__desk_"
        self._alias = self._prefix + "active"
        self._active = next(
            (
                alias.collection_name
                for alias in self.client.get_aliases().aliases
                if alias.alias_name == self._alias
            ),
            None,
        )
        if self._active is None and self.client.collection_exists(
            self.settings.qdrant_collection
        ):
            self._active = self.settings.qdrant_collection
        docs = self._read_collection(self._active) if self._active is not None else []
        self._docs, self._bm25, self.corpus_sha256 = self._prepare_snapshot(docs)

    def _read_collection(self, name):
        docs = []
        offset = None
        while True:
            rows, offset = self.client.scroll(
                name, limit=256, offset=offset, with_payload=True
            )
            for row in rows:
                if row.payload.get("embedding_model") != self.settings.embedding_model:
                    raise ValueError(
                        "Embedding model mismatch; select a new collection or rebuild the index"
                    )
                docs.append(
                    Document(
                        page_content=row.payload["text"],
                        metadata=row.payload["metadata"],
                    )
                )
            if offset is None:
                return docs

    @staticmethod
    def _prepare_snapshot(docs):
        docs = sorted(docs, key=lambda d: d.metadata["chunk_id"])
        fingerprint = hashlib.sha256(
            json.dumps(
                [
                    (d.metadata["chunk_id"], d.metadata.get("source_sha256"))
                    for d in docs
                ],
                separators=(",", ":"),
            ).encode()
        ).hexdigest()
        bm25 = BM25Okapi([_tokens(d.page_content) for d in docs]) if docs else None
        return docs, bm25, fingerprint

    def _discard_collection(self, name):
        try:
            if not self.client.delete_collection(name):
                logger.warning("Qdrant did not remove collection %s", name)
        except Exception:
            logger.warning(
                "Unable to remove inactive collection %s", name, exc_info=True
            )

    def add_documents(self, docs):
        if not docs:
            raise ValueError("No non-empty documents to ingest")
        with self.lock:
            vectors = list(self.embedder.embed([d.page_content for d in docs]))
            if len(vectors) != len(docs):
                raise ValueError("Embedding count does not match document count")
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
                for d, v in zip(docs, vectors, strict=True)
            ]
            if len({point.id for point in points}) != len(points):
                raise ValueError("Duplicate chunk identities")

            staged = self._prefix + uuid.uuid4().hex[:16]
            created = self.client.create_collection(
                staged,
                vectors_config=models.VectorParams(
                    size=len(vectors[0]), distance=models.Distance.COSINE
                ),
            )
            if not created:
                raise RuntimeError("Qdrant did not create the staging collection")
            try:
                for start in range(0, len(points), 256):
                    self.client.upsert(
                        staged, points=points[start : start + 256], wait=True
                    )
                stored = self._read_collection(staged)
                expected = {
                    d.metadata["chunk_id"]: (d.page_content, d.metadata) for d in docs
                }
                actual = {
                    d.metadata["chunk_id"]: (d.page_content, d.metadata) for d in stored
                }
                if len(stored) != len(docs) or actual != expected:
                    raise RuntimeError(
                        "Staged collection does not match the requested corpus"
                    )
                snapshot = self._prepare_snapshot(stored)
            except Exception:
                self._discard_collection(staged)
                raise

            # A timeout during activation is ambiguous. Leave both copies intact
            # so a restart can follow the stored alias without losing either index.
            previous = self._active
            activated = self.client.update_collection_aliases(
                [
                    models.DeleteAliasOperation(
                        delete_alias=models.DeleteAlias(alias_name=self._alias)
                    ),
                    models.CreateAliasOperation(
                        create_alias=models.CreateAlias(
                            collection_name=staged, alias_name=self._alias
                        )
                    ),
                ]
            )
            if not activated:
                raise RuntimeError("Qdrant did not confirm index activation")
            self._active = staged
            self._docs, self._bm25, self.corpus_sha256 = snapshot
            self.generation += 1
            if previous is not None and previous.startswith(self._prefix):
                self._discard_collection(previous)

    def search(self, query, k=10, alpha=None):
        with self.lock:
            if not self._docs:
                return []
            vector = next(iter(self.embedder.query_embed(query))).tolist()
            hits = self.client.query_points(
                self._active,
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
