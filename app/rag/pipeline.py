"""The RAG pipeline: LangGraph workflow behind the ``QuestionAnswerer`` interface."""

from __future__ import annotations

from typing import cast

from app.domain.interfaces import AnswerGenerator, GroundingValidator, Retriever
from app.domain.models import Answer, AnswerStatus, ConfidenceScore
from app.rag.graph.nodes import RAGNodes
from app.rag.graph.state import RAGState
from app.rag.graph.workflow import RAGGraph, build_rag_graph
from app.rag.grounding.confidence import ConfidencePolicy


class RAGPipeline:
    """Compiles the graph once; ``answer`` runs it for one question."""

    def __init__(
        self,
        retriever: Retriever,
        generator: AnswerGenerator,
        validator: GroundingValidator,
        policy: ConfidencePolicy,
    ) -> None:
        self._graph: RAGGraph = build_rag_graph(RAGNodes(retriever, generator, validator, policy))

    @property
    def graph(self) -> RAGGraph:
        return self._graph

    def answer(self, question: str, top_k: int) -> Answer:
        # invoke() is typed as returning a plain dict; the graph's output schema is RAGState.
        final = cast(RAGState, self._graph.invoke({"question": question, "top_k": top_k}))
        return to_answer(final)


def to_answer(state: RAGState) -> Answer:
    status = state["status"]
    draft = state.get("draft")
    verdict = state.get("verdict")
    answered = status is AnswerStatus.ANSWERED
    return Answer(
        question=state["question"],
        text=state["answer"],
        status=status,
        confidence=state.get("confidence") or ConfidenceScore.none(),
        retrieved=tuple(state.get("retrieved", [])),
        context_chunk_ids=frozenset(result.chunk.id for result in state.get("context", [])),
        cited_chunk_ids=draft.cited_chunk_ids if answered and draft else (),
        unsupported_claims=verdict.unsupported_claims if verdict else (),
    )
