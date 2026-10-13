"""Self-corrective retrieval graph using LangGraph."""
from typing import TypedDict
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from src.rag.pipeline import RAGPipeline


class GraphState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    citations: list[dict]
    needs_rewrite: bool
    iterations: int


def build_rag_graph(pipeline: RAGPipeline | None = None):
    p = pipeline or RAGPipeline()

    def retrieve_node(state: GraphState) -> GraphState:
        docs = p.retrieve(state["question"], top_k=4, rerank=True)
        return {**state, "documents": docs, "iterations": state.get("iterations", 0) + 1}

    def grade_node(state: GraphState) -> GraphState:
        q_terms = set(state["question"].lower().split())
        relevant = [d for d in state["documents"] if any(t in d.page_content.lower() for t in q_terms if len(t) > 3)]
        docs = relevant or state["documents"]
        return {**state, "documents": docs, "needs_rewrite": len(docs) == 0 and state.get("iterations", 0) < 2}

    def rewrite_node(state: GraphState) -> GraphState:
        terms = [t for t in state["question"].replace("?", "").split() if len(t) > 2]
        return {**state, "question": " ".join(terms)}

    def generate_node(state: GraphState) -> GraphState:
        res = p.answer(state["question"], top_k=len(state["documents"]) or 4, rerank=False)
        return {**state, "answer": res["answer"], "citations": res["citations"]}

    def route_decision(state: GraphState) -> str:
        return "rewrite" if state.get("needs_rewrite") else "generate"

    workflow = StateGraph(GraphState)
    workflow.add_node("retrieve", retrieve_node)
    workflow.add_node("grade", grade_node)
    workflow.add_node("rewrite", rewrite_node)
    workflow.add_node("generate", generate_node)
    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "grade")
    workflow.add_conditional_edges("grade", route_decision, {"rewrite": "rewrite", "generate": "generate"})
    workflow.add_edge("rewrite", "retrieve")
    workflow.add_edge("generate", END)
    return workflow.compile()
