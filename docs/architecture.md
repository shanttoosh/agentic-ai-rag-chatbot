# Architecture

## Layers and dependency direction

```
            ┌───────────────────────┐        ┌──────────────────────┐
            │ ui/  (Streamlit)      │ HTTP → │ app/api  (FastAPI)   │  routes, schemas, error mapping
            └───────────────────────┘        └──────────┬───────────┘
                                                        │ ChatService
                                             ┌──────────▼───────────┐
                                             │ app/application      │  use cases: chat, ingestion,
                                             │                      │  chunking (pure Python)
                                             └──────────┬───────────┘
                                                        │ QuestionAnswerer (Protocol)
                                             ┌──────────▼───────────┐
                                             │ app/rag              │  LangGraph workflow, retrieval,
                                             │                      │  generation, grounding, confidence
                                             └──────────┬───────────┘
                                                        │ Embedder / VectorStore / StructuredLLM (Protocols)
                                             ┌──────────▼───────────┐
                                             │ app/infrastructure   │  Pinecone, Groq/Claude, PyMuPDF adapters
                                             └──────────────────────┘

        app/domain   models (frozen dataclasses) + interfaces (Protocols); imports nothing above it
        app/core     settings, exceptions, logging, constants; imported by every layer
        app/bootstrap.py   composition root: the only module that knows every concrete class
```

Rules the code follows:

- **`domain` depends on nothing** except the standard library and Pydantic (for the
  structured-output type bound on `StructuredLLM`).
- **`application` depends only on `domain`.** `ChatService` receives a `QuestionAnswerer`;
  it has no idea LangGraph exists. `IngestionService` receives a loader, chunker,
  embedder and store.
- **`rag` depends on `domain` Protocols**, never on Pinecone or Anthropic. LangGraph is
  confined to `app/rag/graph/`.
- **`infrastructure` implements the Protocols.** Provider SDK errors are translated into
  the app's exception hierarchy at this boundary, so nothing above it imports `pinecone`,
  `groq` or `anthropic`.
- **`api` is thin.** A route validates input with Pydantic, calls one service method, and
  maps the domain `Answer` to a response schema.

Swapping a provider means writing one adapter and changing `bootstrap.py`. The LLM
already works this way: `GroqStructuredLLM` (the default) and `ClaudeStructuredLLM` both
implement `StructuredLLM`, and `LLM_PROVIDER` selects one. OpenAI embeddings would
likewise be one class with `embed_documents` / `embed_query`.

## Module map

| Path | Responsibility |
|---|---|
| `app/main.py` | FastAPI app factory + lifespan (builds services, closes clients on shutdown) |
| `app/bootstrap.py` | Wires concrete classes; `open_*` context managers own client lifetimes |
| `app/api/` | `POST /api/chat`, `GET /health`, dependency injection, exception → HTTP mapping |
| `app/schemas/` | Pydantic request/response models (the public API contract) |
| `app/core/` | `Settings` (pydantic-settings), exception hierarchy, JSON logging, constants |
| `app/domain/models/` | `Document`, `Page`, `DocumentChunk`, `RetrievalResult`, `DraftAnswer`, `GroundingVerdict`, `ConfidenceScore`, `Answer` |
| `app/domain/interfaces/` | `DocumentLoader`, `Chunker`, `Embedder`, `VectorStore`, `Retriever`, `StructuredLLM`, `AnswerGenerator`, `GroundingValidator`, `QuestionAnswerer` |
| `app/application/chat/` | `ChatService`: question normalisation, top-k policy, timing/logging |
| `app/application/ingestion/` | `ParagraphChunker` (structure-aware chunking), `IngestionService` (idempotent indexing) |
| `app/rag/graph/` | LangGraph state, nodes, routing functions, graph construction |
| `app/rag/retrieval/` | `VectorRetriever`, threshold filters |
| `app/rag/generation/` | Grounded answer generator, prompt formatting |
| `app/rag/grounding/` | LLM-judge validator, confidence heuristic |
| `app/prompts/` | System prompts as reviewable text files |
| `app/infrastructure/` | `PyMuPDFLoader`, `download_pdf`, `PineconeEmbedder`, `PineconeVectorStore`, `GroqStructuredLLM`, `ClaudeStructuredLLM` |
| `scripts/` | CLI entry points: download, create index, ingest, evaluate |
| `evaluation/` | Test-case dataset and pure metric functions |
| `ui/` | Streamlit client of the HTTP API |

## Request lifecycle (`POST /api/chat`)

1. FastAPI validates the body (`ChatRequest`: stripped, 1-2000 chars, `top_k` 1-20) and
   returns a 422 on failure.
2. `get_chat_service` reads the `ChatService` built at startup from `app.state`. If
   startup failed on missing configuration, it raises the stored `ConfigurationError`,
   which becomes a 503.
3. `ChatService.ask` normalises whitespace and calls `RAGPipeline.answer`.
4. The LangGraph graph runs `retrieve → check_retrieval → generate → validate → finalize`.
   Any node can route to `fallback` instead (see [rag-pipeline.md](rag-pipeline.md)).
5. The final state becomes a domain `Answer`, then `ChatResponse.from_answer` builds the
   JSON response.

The route is a plain `def`. The Pinecone, Groq and Anthropic SDKs are synchronous, so FastAPI
runs the handler in its thread pool and the event loop never blocks. The one piece of
shared mutable state, the lazily created Pinecone index handle, is guarded by a lock.

## Ingestion lifecycle (`python scripts/ingest.py`)

1. `download_pdf` downloads the eBook if it's missing. It writes to a `.part` file,
   checks the `%PDF-` magic bytes, then renames the file into place.
2. `PyMuPDFLoader` extracts text blocks page by page in reading order, removes footers,
   page numbers and glyph artifacts, and hashes the file bytes.
3. `ParagraphChunker` yields chunks with page, chapter and section metadata.
4. `IngestionService` computes a fingerprint from the PDF hash, the chunker config, the
   embedding model, and `INGESTION_VERSION`.
   - If the stored vectors already carry that fingerprint and the count matches,
     ingestion stops: no re-embedding.
   - Otherwise it embeds and upserts in batches, then deletes vectors whose IDs no longer
     exist. Queries keep working during a re-ingest, because the namespace is never
     emptied first.

## Error handling

| Exception (app/core/exceptions.py) | HTTP | When |
|---|---|---|
| `InvalidQuestionError` | 422 | Empty/oversized question reaching the service layer |
| `ConfigurationError` | 503 | Missing `PINECONE_API_KEY` / `GROQ_API_KEY` (or `ANTHROPIC_API_KEY`), invalid settings |
| `VectorStoreError` | 503 | Pinecone unreachable, index missing ("run ingest first") |
| `EmbeddingError` | 502 | Pinecone Inference failure or malformed vectors |
| `LLMError` | 502 | LLM unreachable, rate-limited, auth rejected, truncated or invalid output |
| `DocumentError` | 500 (CLI: exit 1) | PDF download/parse failure |
| anything else | 500 | Generic `"Internal server error"`; details only in logs |

Low relevance is **not an error**. A question with no chunk above the threshold returns
200 with the fallback answer and `"grounded": false`.

Messages are written to be safe for clients. Provider errors are chained (`raise ... from
exc`) so the logs keep the full cause, while the response never includes keys, hosts or
raw payloads.

## Logging

`app/core/logging.py` emits one JSON object per line (`LOG_FORMAT=json`) or readable
`key=value` text (`LOG_FORMAT=text`). Events are named (`retrieval.completed`,
`grounding.validated`, `chat.answered`, `llm.completed` with token usage, ...) and carry
their fields via `extra`, so each request can be traced end to end.
