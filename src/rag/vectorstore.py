"""Qdrant dense storage with BM25 + dense reciprocal-rank fusion."""
import re
from typing import Sequence
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Qdrant
from rank_bm25 import BM25Okapi
from qdrant_client import QdrantClient
from src.app.dependencies import AppSettings, get_settings


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z0-9_]+", text.lower())


class HybridRetriever:
    """Combines Qdrant similarity and corpus-local BM25 using reciprocal ranks."""
    def __init__(self, settings: AppSettings | None = None):
        self.settings = settings or get_settings()
        self.client = QdrantClient(url=self.settings.qdrant_url, api_key=self.settings.qdrant_api_key or None)
        self.embeddings: Embeddings = OpenAIEmbeddings(model=self.settings.embedding_model, api_key=self.settings.openai_api_key)
        self.store = Qdrant(client=self.client, collection_name=self.settings.qdrant_collection, embeddings=self.embeddings)
        self._bm25: BM25Okapi | None = None
        self._docs: list[Document] = []

    def add_documents(self, docs: Sequence[Document]) -> None:
        if not docs:
            return
        self.store.add_documents(list(docs))
        self._docs.extend(docs)
        self._bm25 = BM25Okapi([_tokens(d.page_content) for d in self._docs]) if self._docs else None

    def search(self, query: str, k: int = 10, alpha: float | None = None) -> list[Document]:
        weight = self.settings.hybrid_alpha if alpha is None else alpha
        dense = self.store.similarity_search(query, k=k)
        lexical: list[Document] = []
        if self._bm25 and self._docs:
            scores = self._bm25.get_scores(_tokens(query))
            lexical = [self._docs[i] for i in sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k] if scores[i] > 0]
        fused: dict[str, tuple[Document, float]] = {}
        for ranking, factor in ((dense, weight), (lexical, 1.0 - weight)):
            for rank, doc in enumerate(ranking, start=1):
                key = str(doc.metadata.get("chunk_id") or doc.page_content[:120])
                old = fused.get(key, (doc, 0.0))
                fused[key] = (doc, old[1] + factor / (60 + rank))
        return [doc for doc, _ in sorted(fused.values(), key=lambda item: item[1], reverse=True)[:k]]
