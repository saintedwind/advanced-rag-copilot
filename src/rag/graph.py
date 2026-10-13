"""Bounded retrieval graph; synthesis uses exactly the selected documents."""

from typing import TypedDict

from langgraph.graph import END, StateGraph


class GraphState(TypedDict, total=False):
    question: str
    original_question: str
    documents: list
    result: dict
    iterations: int
    top_k: int
    rerank: bool


def build_rag_graph(pipeline):
    def retrieve(state):
        return {
            "documents": pipeline.retrieve(
                state["question"], state["top_k"], state["rerank"]
            ),
            "iterations": state.get("iterations", 0) + 1,
        }

    def rewrite(state):
        return {"question": " ".join(state["question"].replace("?", " ").split())}

    def generate(state):
        return {
            "result": pipeline.synthesize(
                state["original_question"], state["documents"]
            )
        }

    graph = StateGraph(GraphState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("rewrite", rewrite)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_conditional_edges(
        "retrieve",
        lambda s: (
            "rewrite" if not s["documents"] and s["iterations"] < 2 else "generate"
        ),
        {"rewrite": "rewrite", "generate": "generate"},
    )
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    return graph.compile()
