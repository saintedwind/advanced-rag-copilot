"""Production FastAPI application exposing ingestion, health, and query endpoints."""
import time
from fastapi import FastAPI, HTTPException
from src.app.dependencies import get_settings
from src.app.schemas import HealthResponse, IngestionRequest, IngestionResponse, QueryRequest, QueryResponse
from src.rag.graph import build_rag_graph
from src.rag.ingestion import load_and_split
from src.rag.pipeline import RAGPipeline
from src.rag.vectorstore import HybridRetriever

app = FastAPI(title="Advanced RAG Copilot", version="1.0.0", description="Self-corrective enterprise hybrid RAG service")
retriever = HybridRetriever()
pipeline = RAGPipeline(retriever=retriever)
agent_graph = build_rag_graph(pipeline=pipeline)


@app.get("/health", response_model=HealthResponse, tags=["Observability"])
def health_check():
    connected = True
    try:
        retriever.client.get_collections()
    except Exception:
        connected = False
    return HealthResponse(status="healthy" if connected else "degraded", version="1.0.0", vector_db_connected=connected, reranker_ready=pipeline.reranker.ready)


@app.post("/ingest", response_model=IngestionResponse, tags=["RAG"])
def ingest_documents(payload: IngestionRequest):
    t0 = time.perf_counter()
    try:
        chunks, doc_count = load_and_split(payload.source_dir or "data/sample_docs", payload.chunk_size, payload.chunk_overlap)
        retriever.add_documents(chunks)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return IngestionResponse(status="success", documents_ingested=doc_count, chunks_created=len(chunks), collection_name=get_settings().qdrant_collection, elapsed_seconds=round(time.perf_counter() - t0, 3))


@app.post("/query", response_model=QueryResponse, tags=["RAG"])
def answer_query(payload: QueryRequest):
    t0 = time.perf_counter()
    if payload.use_agentic_flow:
        res = agent_graph.invoke({"question": payload.query, "documents": [], "answer": "", "citations": [], "needs_rewrite": False, "iterations": 0})
        ans, cits = res["answer"], res["citations"]
        route = "langgraph_agentic_workflow"
    else:
        res = pipeline.answer(payload.query, top_k=payload.top_k, rerank=payload.enable_reranking)
        ans, cits, route = res["answer"], res["citations"], res.get("route_taken", "direct_pipeline")
    return QueryResponse(query=payload.query, answer=ans, citations=cits, confidence="high" if cits else "low", route_taken=route, elapsed_seconds=round(time.perf_counter() - t0, 3))


@app.post("/chat", response_model=QueryResponse, tags=["RAG"])
def chat(payload: QueryRequest):
    return answer_query(payload)
