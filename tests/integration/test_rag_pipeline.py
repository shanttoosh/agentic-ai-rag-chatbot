"""The full LangGraph workflow wired with fakes: every route through the graph."""

from __future__ import annotations

from app.core.constants import FALLBACK_ANSWER
from app.domain.models import AnswerStatus
from app.rag.generation.generator import GenerationOutput, GroundedAnswerGenerator
from app.rag.graph.edges import CHECK_RETRIEVAL, FALLBACK, FINALIZE, GENERATE, RETRIEVE, VALIDATE
from app.rag.grounding.confidence import ConfidencePolicy
from app.rag.grounding.validator import LLMGroundingValidator, ValidationOutput
from app.rag.pipeline import RAGPipeline
from tests.fakes import FakeLLM, FakeRetriever, make_result, refusal

RELEVANT = [
    make_result(0.58, "page18_chunk00", page=18),
    make_result(0.44, "page08_chunk00", page=8),
    make_result(0.21, "page60_chunk00", page=60),
]
IRRELEVANT = [
    make_result(0.12, "page05_chunk00", page=5),
    make_result(0.09, "page60_chunk00", page=60),
]


def _pipeline(
    results: list, generator_llm: FakeLLM, validator_llm: FakeLLM, policy: ConfidencePolicy
) -> RAGPipeline:
    return RAGPipeline(
        FakeRetriever(results),
        GroundedAnswerGenerator(generator_llm),
        LLMGroundingValidator(validator_llm),
        policy,
    )


def test_graph_has_the_expected_nodes(policy: ConfidencePolicy) -> None:
    graph = _pipeline(RELEVANT, FakeLLM(), FakeLLM(), policy).graph.get_graph()
    assert {RETRIEVE, CHECK_RETRIEVAL, GENERATE, VALIDATE, FINALIZE, FALLBACK} <= set(graph.nodes)


def test_answerable_question_returns_grounded_answer(policy: ConfidencePolicy) -> None:
    generator = FakeLLM(
        GenerationOutput(
            answerable=True,
            answer="Agentic AI refers to systems capable of autonomous decision-making (p. 18).",
            cited_excerpts=[1],
        )
    )
    validator = FakeLLM(ValidationOutput(unsupported_claims=[], supported=True))

    answer = _pipeline(RELEVANT, generator, validator, policy).answer(
        "What is Agentic AI?", top_k=5
    )

    assert answer.status is AnswerStatus.ANSWERED
    assert answer.grounded
    assert answer.text.startswith("Agentic AI refers to")
    assert answer.cited_pages == (18,)
    assert 0 < answer.confidence.value <= 1
    assert answer.confidence.relevant_chunks == 2
    # Only chunks above the threshold reach the LLM; all retrieved chunks are returned.
    assert answer.context_chunk_ids == {"page18_chunk00", "page08_chunk00"}
    assert len(answer.retrieved) == 3
    assert "page60_chunk00" not in generator.calls[0]["prompt"]


def test_low_relevance_query_falls_back_without_calling_the_llm(policy: ConfidencePolicy) -> None:
    generator, validator = FakeLLM(), FakeLLM()

    answer = _pipeline(IRRELEVANT, generator, validator, policy).answer(
        "Capital of France?", top_k=5
    )

    assert answer.text == FALLBACK_ANSWER
    assert answer.status is AnswerStatus.LOW_RETRIEVAL_SCORE
    assert not answer.grounded
    assert answer.confidence.value == 0.0
    assert answer.confidence.top_score == 0.12  # raw score still visible
    assert generator.calls == [] and validator.calls == []


def test_empty_index_falls_back(policy: ConfidencePolicy) -> None:
    answer = _pipeline([], FakeLLM(), FakeLLM(), policy).answer("anything", top_k=5)
    assert answer.status is AnswerStatus.LOW_RETRIEVAL_SCORE
    assert answer.retrieved == ()


def test_generator_finds_no_answer(policy: ConfidencePolicy) -> None:
    generator = FakeLLM(GenerationOutput(answerable=False, answer="", cited_excerpts=[]))
    validator = FakeLLM()

    answer = _pipeline(RELEVANT, generator, validator, policy).answer(
        "How do I fine-tune Llama?", top_k=5
    )

    assert answer.status is AnswerStatus.NOT_IN_CONTEXT
    assert answer.text == FALLBACK_ANSWER
    assert validator.calls == []


def test_unsupported_answer_is_withheld(policy: ConfidencePolicy) -> None:
    generator = FakeLLM(
        GenerationOutput(
            answerable=True, answer="Agentic AI was invented in 1956 (p. 18).", cited_excerpts=[1]
        )
    )
    validator = FakeLLM(ValidationOutput(unsupported_claims=["invented in 1956"], supported=False))

    answer = _pipeline(RELEVANT, generator, validator, policy).answer(
        "When was Agentic AI invented?", top_k=5
    )

    assert answer.status is AnswerStatus.FAILED_VALIDATION
    assert answer.text == FALLBACK_ANSWER
    assert "1956" not in answer.text
    assert answer.unsupported_claims == ("invented in 1956",)
    assert answer.confidence.value == 0.0
    assert answer.cited_chunk_ids == ()


def test_uncited_answer_fails_validation_without_judge(policy: ConfidencePolicy) -> None:
    generator = FakeLLM(GenerationOutput(answerable=True, answer="Some answer.", cited_excerpts=[]))
    validator = FakeLLM()

    answer = _pipeline(RELEVANT, generator, validator, policy).answer("q", top_k=5)

    assert answer.status is AnswerStatus.FAILED_VALIDATION
    assert validator.calls == []


def test_validator_refusal_withholds_the_answer(policy: ConfidencePolicy) -> None:
    generator = FakeLLM(
        GenerationOutput(answerable=True, answer="An answer (p. 18).", cited_excerpts=[1])
    )
    answer = _pipeline(RELEVANT, generator, FakeLLM(refusal()), policy).answer("q", top_k=5)
    assert answer.status is AnswerStatus.FAILED_VALIDATION
    assert answer.text == FALLBACK_ANSWER


def test_model_refusal_returns_fallback(policy: ConfidencePolicy) -> None:
    answer = _pipeline(RELEVANT, FakeLLM(refusal()), FakeLLM(), policy).answer("q", top_k=5)
    assert answer.status is AnswerStatus.DECLINED
    assert answer.text == FALLBACK_ANSWER


def test_top_k_is_passed_to_retriever(policy: ConfidencePolicy) -> None:
    retriever = FakeRetriever(IRRELEVANT)
    RAGPipeline(
        retriever, GroundedAnswerGenerator(FakeLLM()), LLMGroundingValidator(FakeLLM()), policy
    ).answer("q", top_k=3)
    assert retriever.calls == [("q", 3)]
