"""Composition root: builds services from settings and concrete infrastructure.

Each ``open_*`` factory is a context manager that closes its provider clients on exit.
"""

from __future__ import annotations

from collections.abc import Generator
from contextlib import ExitStack, contextmanager

import anthropic
import groq
from pinecone import Pinecone

from app.application.chat.service import ChatService
from app.application.ingestion.chunker import ParagraphChunker
from app.application.ingestion.service import IngestionService
from app.core.config import Settings
from app.core.constants import DOCUMENT_SOURCE, EBOOK_CHAPTERS
from app.domain.interfaces import StructuredLLM
from app.infrastructure.documents.pdf_loader import PyMuPDFLoader
from app.infrastructure.embeddings.pinecone_embedder import PineconeEmbedder
from app.infrastructure.llm.claude import ClaudeStructuredLLM
from app.infrastructure.llm.groq_llm import GroqStructuredLLM
from app.infrastructure.vector_store.pinecone_store import PineconeVectorStore
from app.rag.generation.generator import GroundedAnswerGenerator
from app.rag.grounding.confidence import ConfidencePolicy
from app.rag.grounding.validator import LLMGroundingValidator
from app.rag.pipeline import RAGPipeline
from app.rag.retrieval.retriever import VectorRetriever


def confidence_policy(settings: Settings) -> ConfidencePolicy:
    return ConfidencePolicy(
        threshold=settings.retrieval_threshold, ceiling=settings.confidence_score_ceiling
    )


def _pinecone(settings: Settings, stack: ExitStack) -> tuple[PineconeEmbedder, PineconeVectorStore]:
    client = Pinecone(api_key=settings.require_secret("pinecone"), source_tag="agentic_ai_rag")
    stack.callback(client.close)
    embedder = PineconeEmbedder(
        client,
        model=settings.embedding_model,
        dimension=settings.embedding_dimension,
        batch_size=settings.embedding_batch_size,
    )
    store = PineconeVectorStore(
        client,
        index_name=settings.pinecone_index_name,
        namespace=settings.pinecone_namespace,
        dimension=settings.embedding_dimension,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
    )
    stack.callback(store.close)
    return embedder, store


def _llms(settings: Settings, stack: ExitStack) -> tuple[StructuredLLM, StructuredLLM]:
    """(generator LLM, validator LLM) for the configured provider, sharing one client."""
    generation = (settings.effective_llm_model, settings.llm_effort)
    validation = (settings.effective_validator_model, settings.validator_effort)
    if settings.llm_provider == "groq":
        groq_client = groq.Groq(
            api_key=settings.require_secret("groq"), timeout=settings.llm_timeout_seconds
        )
        stack.callback(groq_client.close)
        return GroqStructuredLLM(groq_client, *generation), GroqStructuredLLM(
            groq_client, *validation
        )
    claude_client = anthropic.Anthropic(
        api_key=settings.require_secret("anthropic"), timeout=settings.llm_timeout_seconds
    )
    stack.callback(claude_client.close)
    return ClaudeStructuredLLM(claude_client, *generation), ClaudeStructuredLLM(
        claude_client, *validation
    )


def build_chunker(settings: Settings) -> ParagraphChunker:
    return ParagraphChunker(settings.chunk_size, settings.chunk_overlap, chapters=EBOOK_CHAPTERS)


@contextmanager
def open_chat_service(settings: Settings) -> Generator[ChatService, None, None]:
    with ExitStack() as stack:
        embedder, store = _pinecone(settings, stack)
        generation_llm, validation_llm = _llms(settings, stack)
        pipeline = RAGPipeline(
            VectorRetriever(embedder, store),
            GroundedAnswerGenerator(generation_llm),
            LLMGroundingValidator(validation_llm),
            confidence_policy(settings),
        )
        yield ChatService(pipeline, default_top_k=settings.top_k)


@contextmanager
def open_ingestion_service(settings: Settings) -> Generator[IngestionService, None, None]:
    with ExitStack() as stack:
        embedder, store = _pinecone(settings, stack)
        yield IngestionService(
            loader=PyMuPDFLoader(DOCUMENT_SOURCE),
            chunker=build_chunker(settings),
            embedder=embedder,
            store=store,
            batch_size=settings.embedding_batch_size,
        )


@contextmanager
def open_vector_store(settings: Settings) -> Generator[PineconeVectorStore, None, None]:
    with ExitStack() as stack:
        _, store = _pinecone(settings, stack)
        yield store


@contextmanager
def open_retriever(settings: Settings) -> Generator[VectorRetriever, None, None]:
    with ExitStack() as stack:
        embedder, store = _pinecone(settings, stack)
        yield VectorRetriever(embedder, store)
