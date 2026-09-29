"""LangGraph construction.

START -> retrieve -> check_retrieval --(no chunk above threshold)--> fallback -> END
                            |
                            v
                        generate --(not answerable / declined)--> fallback
                            |
                            v
                        validate --(unsupported claims)---------> fallback
                            |
                            v
                        finalize -> END
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from app.rag.graph.edges import (
    CHECK_RETRIEVAL,
    FALLBACK,
    FINALIZE,
    GENERATE,
    RETRIEVE,
    VALIDATE,
    after_check_retrieval,
    after_generate,
    after_validate,
)
from app.rag.graph.nodes import RAGNodes
from app.rag.graph.state import RAGState

RAGGraph = CompiledStateGraph[RAGState, None, RAGState, RAGState]


def build_rag_graph(nodes: RAGNodes) -> RAGGraph:
    graph = StateGraph(RAGState)
    graph.add_node(RETRIEVE, nodes.retrieve)
    graph.add_node(CHECK_RETRIEVAL, nodes.check_retrieval)
    graph.add_node(GENERATE, nodes.generate)
    graph.add_node(VALIDATE, nodes.validate)
    graph.add_node(FINALIZE, nodes.finalize)
    graph.add_node(FALLBACK, nodes.fallback)

    graph.add_edge(START, RETRIEVE)
    graph.add_edge(RETRIEVE, CHECK_RETRIEVAL)
    graph.add_conditional_edges(
        CHECK_RETRIEVAL, after_check_retrieval, {GENERATE: GENERATE, FALLBACK: FALLBACK}
    )
    graph.add_conditional_edges(GENERATE, after_generate, {VALIDATE: VALIDATE, FALLBACK: FALLBACK})
    graph.add_conditional_edges(VALIDATE, after_validate, {FINALIZE: FINALIZE, FALLBACK: FALLBACK})
    graph.add_edge(FINALIZE, END)
    graph.add_edge(FALLBACK, END)
    return graph.compile()
