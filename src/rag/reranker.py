"""Optional FlashRank reranker with a safe retrieval-order fallback."""

import threading
from typing import Sequence

from langchain_core.documents import Document

from src.app.dependencies import get_settings


class DocumentReranker:
    def __init__(self, settings=None):
        self.settings = settings or get_settings()
        self.lock = threading.Lock()
        self._compressor = None
        if self.settings.reranker_provider.lower() == "flashrank":
            try:
                from flashrank import Ranker
                from langchain_community.document_compressors import FlashrankRerank

                self._compressor = FlashrankRerank(
                    client=Ranker(model_name=self.settings.reranker_model),
                    top_n=self.settings.top_n_rerank,
                )
            except (ImportError, RuntimeError, OSError):
                self._compressor = None

    @property
    def ready(self) -> bool:
        return self._compressor is not None

    def rerank(
        self, query: str, docs: Sequence[Document], top_n: int = 4
    ) -> list[Document]:
        if not docs:
            return []
        if self._compressor is None:
            return list(docs)[:top_n]
        with self.lock:
            self._compressor.top_n = top_n
            try:
                return list(self._compressor.compress_documents(list(docs), query))
            except Exception:
                return list(docs)[:top_n]
