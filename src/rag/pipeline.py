"""Grounded RAG answer synthesis, fallback behavior, and citations."""
from typing import Any
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from src.app.dependencies import get_settings
from src.rag.reranker import DocumentReranker
from src.rag.vectorstore import HybridRetriever

PROMPT = ChatPromptTemplate.from_messages([("system", "Answer only from supplied context. If evidence is insufficient, say so clearly. Cite claims with [n] markers matching the context numbering."), ("human", "Question: {question}\n\nContext:\n{context}")])


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever | None = None, reranker: DocumentReranker | None = None):
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or DocumentReranker()
        self.llm = ChatOpenAI(model=get_settings().openai_model, temperature=0, api_key=get_settings().openai_api_key)

    def retrieve(self, query: str, top_k: int = 5, rerank: bool = True) -> list[Document]:
        candidates = self.retriever.search(query, k=max(top_k, get_settings().top_k_retrieval))
        return self.reranker.rerank(query, candidates, top_n=top_k) if rerank else candidates[:top_k]

    def answer(self, query: str, top_k: int = 5, rerank: bool = True) -> dict[str, Any]:
        docs = self.retrieve(query, top_k, rerank)
        if not docs:
            return {"answer": "I could not find relevant evidence in the indexed knowledge base.", "citations": [], "confidence": "low", "route_taken": "no_context"}
        context = "\n\n".join(f"[{i}] {doc.page_content}" for i, doc in enumerate(docs, 1))
        try:
            response = (PROMPT | self.llm).invoke({"question": query, "context": context})
            text = str(response.content)
        except Exception:
            text = "I found potentially relevant material, but answer generation is temporarily unavailable. Please retry."
        citations = [{"source": str(d.metadata.get("source", "unknown")), "page": d.metadata.get("page"), "chunk_id": d.metadata.get("chunk_id"), "snippet": d.page_content[:240]} for d in docs]
        return {"answer": text, "citations": citations, "confidence": "medium" if docs else "low", "route_taken": "retrieval_synthesized"}
