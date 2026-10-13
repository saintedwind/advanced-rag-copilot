"""Local evaluation API. No import-time model downloads or service connections."""

import copy
import logging
import threading
import time
from collections import OrderedDict
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse

from src.app.dependencies import get_settings
from src.app.schemas import IngestionRequest, QueryRequest, QueryResponse
from src.rag.graph import build_rag_graph
from src.rag.ingestion import load_and_split
from src.rag.pipeline import RAGPipeline


def create_app(pipeline=None, settings=None):
    settings = settings or get_settings()

    @asynccontextmanager
    async def lifespan(app):
        app.state.pipeline = pipeline or RAGPipeline(settings=settings)
        app.state.graph = build_rag_graph(app.state.pipeline)
        app.state.cache = OrderedDict()
        app.state.lock = threading.RLock()
        yield
        if pipeline is None:
            app.state.pipeline.retriever.close()

    app = FastAPI(title="Knowledge Evidence Desk", version="0.2.0", lifespan=lifespan)

    @app.get("/")
    def home():
        return FileResponse(Path(__file__).parent / "index.html")

    @app.get("/health")
    def health():
        p = app.state.pipeline
        try:
            p.retriever.client.get_collections()
            connected = True
        except Exception:
            connected = False
        return {
            "status": "ready"
            if connected and p.retriever._docs
            else "empty"
            if connected
            else "degraded",
            "version": "0.2.0",
            "vector_db_connected": connected,
            "chunks": len(p.retriever._docs),
            "embedding_model": settings.embedding_model,
            "generation_mode": settings.llm_provider,
            "reranker_ready": p.reranker.ready,
            "cache": "exact_query_bounded_128",
        }

    @app.post("/ingest")
    def ingest(payload: IngestionRequest):
        t = time.perf_counter()
        try:
            with app.state.lock:
                chunks, count = load_and_split(
                    payload.source_dir or settings.source_root,
                    payload.chunk_size,
                    payload.chunk_overlap,
                    allowed_root=settings.source_root,
                )
                if not chunks:
                    raise ValueError("No readable documents")
                app.state.pipeline.retriever.add_documents(chunks)
                app.state.cache.clear()
        except (ValueError, UnicodeError) as error:
            raise HTTPException(400, detail=str(error))
        except Exception:
            logging.exception("Indexing failed")
            raise HTTPException(
                503, detail="Indexing unavailable; inspect service diagnostics"
            )
        return {
            "status": "success",
            "documents_ingested": count,
            "chunks_created": len(chunks),
            "elapsed_seconds": round(time.perf_counter() - t, 3),
        }

    @app.post("/query", response_model=QueryResponse)
    def query(payload: QueryRequest):
        t = time.perf_counter()
        with app.state.lock:
            key = (
                app.state.pipeline.retriever.generation,
                payload.query,
                payload.top_k,
                payload.enable_reranking,
                payload.use_agentic_flow,
                settings.llm_provider,
            )
            if key in app.state.cache:
                result = copy.deepcopy(app.state.cache[key])
                result.update(
                    cache_hit=True, elapsed_seconds=round(time.perf_counter() - t, 3)
                )
                return result
            try:
                if payload.use_agentic_flow:
                    result = app.state.graph.invoke(
                        {
                            "question": payload.query,
                            "original_question": payload.query,
                            "documents": [],
                            "iterations": 0,
                            "top_k": payload.top_k,
                            "rerank": payload.enable_reranking,
                        }
                    )["result"]
                    result["route_taken"] = "langgraph/" + result["route_taken"]
                else:
                    result = app.state.pipeline.answer(
                        payload.query, payload.top_k, payload.enable_reranking
                    )
            except Exception:
                logging.exception("Query failed")
                raise HTTPException(503, detail="Query processing unavailable")
            result.update(
                query=payload.query, elapsed_seconds=round(time.perf_counter() - t, 3)
            )
            if result["status"] in {"evidence_found", "insufficient_evidence"}:
                app.state.cache[key] = copy.deepcopy(result)
                while len(app.state.cache) > 128:
                    app.state.cache.popitem(last=False)
            return result

    app.add_api_route("/chat", query, methods=["POST"], response_model=QueryResponse)
    return app


app = create_app()
