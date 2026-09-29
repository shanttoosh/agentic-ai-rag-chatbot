# Agentic AI RAG Chatbot

A retrieval-augmented chatbot that answers questions **strictly from the
[Agentic AI eBook](https://konverge.ai/pdf/Ebook-Agentic-AI.pdf)** (Konverge AI, 60 pages).
If the answer isn't in the book, it says so:

> I couldn't find enough information about this in the provided Agentic AI eBook.

It's built with **LangGraph** (workflow), **Pinecone** (vector DB and hosted embeddings),
**Groq gpt-oss** (LLM), **FastAPI** (API) and **Streamlit** (demo UI).

## Overview

**Problem.** An LLM asked about "agentic AI" will happily answer from its training data.
The brief is the opposite: answer only from one document, and be honest when the document
doesn't cover something.

**Solution.** A LangGraph workflow that retrieves eBook chunks from Pinecone, drops weak
matches, generates a cited answer from the remaining chunks only, then has a second LLM
call **verify every claim** against those chunks. Anything unsupported is replaced by the
safe fallback. Every response includes the answer, the retrieved chunks with similarity
scores, a documented confidence heuristic, and a `grounded` flag.

## Architecture

```
User
 │  Streamlit UI (ui/)  ── or ──  curl / any HTTP client
 ▼
FastAPI  POST /api/chat          (app/api: validation, error mapping)
 ▼
ChatService                      (app/application: use case, no framework code)
 ▼
LangGraph RAG workflow           (app/rag/graph)
 ├─ retrieve ──────────► Pinecone: embed question (llama-text-embed-v2) → top-K chunks + scores
 ├─ check_retrieval ───► drop chunks below RETRIEVAL_THRESHOLD, compute confidence
 │        └─ none left ─────────────────────────────────────────────► fallback
 ├─ generate ──────────► LLM (gpt-oss-120b), strict JSON: {answerable, answer, cited_excerpts}
 │        └─ not answerable ────────────────────────────────────────► fallback
 ├─ validate ──────────► LLM judge (gpt-oss-20b), strict JSON: {unsupported_claims, supported}
 │        └─ any unsupported claim ─────────────────────────────────► fallback
 └─ finalize ──────────► validated answer
 ▼
{answer, grounded, confidence, retrieved_context[...]}
```

The code is layered: API → application → domain Protocols ← RAG / infrastructure
adapters. Business logic never imports FastAPI, Pinecone, Groq or LangGraph directly.
Details are in [docs/architecture.md](docs/architecture.md), the graph and prompts in
[docs/rag-pipeline.md](docs/rag-pipeline.md), and the reasoning behind each choice in
[docs/decisions.md](docs/decisions.md).

## Tech Stack

| Concern | Choice |
|---|---|
| Language | Python 3.11+ (type hints throughout, `mypy --strict` clean) |
| Workflow | LangGraph `StateGraph` with typed state and conditional edges |
| Vector DB | Pinecone serverless (cosine, 1024-d) |
| Embeddings | Pinecone Inference `llama-text-embed-v2` (asymmetric passage/query) |
| LLM | Groq `openai/gpt-oss-120b` (generation), `openai/gpt-oss-20b` (validation); strict JSON-schema outputs. Claude supported via `LLM_PROVIDER=anthropic` |
| PDF | PyMuPDF |
| API | FastAPI + Pydantic v2 |
| Config | pydantic-settings (`.env`) |
| UI | Streamlit (talks to the API over HTTP) |
| Quality | pytest (135 offline tests), Ruff (lint + format), mypy strict, GitHub Actions |

## Project Structure

```
app/
├── main.py                 FastAPI app factory + lifespan
├── bootstrap.py            composition root: the only place concrete providers are wired
├── api/                    routes (/health, /api/chat), DI, exception → HTTP mapping
├── schemas/                Pydantic request/response models
├── core/                   settings, exception hierarchy, JSON logging, constants
├── domain/
│   ├── models/             Document, DocumentChunk, RetrievalResult, Answer, ConfidenceScore …
│   └── interfaces/         Protocols: DocumentLoader, Chunker, Embedder, VectorStore,
│                           Retriever, StructuredLLM, AnswerGenerator, GroundingValidator
├── application/
│   ├── chat/service.py     ChatService
│   └── ingestion/          ParagraphChunker, IngestionService
├── rag/
│   ├── graph/              LangGraph state, nodes, edges, workflow
│   ├── retrieval/          VectorRetriever, threshold filters
│   ├── generation/         grounded generator, prompt formatting
│   ├── grounding/          LLM-judge validator, confidence heuristic
│   └── pipeline.py         RAGPipeline (implements QuestionAnswerer)
├── infrastructure/         PyMuPDF loader, downloader, Pinecone embedder/store, Groq + Claude LLMs
└── prompts/                system.txt, grounding.txt
scripts/                    download_document, create_index, ingest, evaluate
evaluation/                 rag_test_cases.json + metrics.py
ui/streamlit_app.py         demo UI
tests/                      unit/, integration/, api/, ui/  (fakes in tests/fakes.py)
docs/                       architecture, RAG pipeline, design decisions
```

## Setup

You need Python 3.11+, a **Groq** API key (free: <https://console.groq.com/keys>) and a
**Pinecone** API key (free Starter plan: <https://app.pinecone.io>).

```bash
git clone https://github.com/shanttoosh/agentic-ai-rag-chatbot.git
cd agentic-ai-rag-chatbot

python -m venv .venv
source .venv/bin/activate            # Windows PowerShell: .venv\Scripts\Activate.ps1
                                     # Windows cmd:        .venv\Scripts\activate.bat
pip install -r requirements.txt      # add dev tools with: pip install -r requirements-dev.txt

cp .env.example .env                 # Windows: copy .env.example .env
# then set GROQ_API_KEY and PINECONE_API_KEY in .env
```

`requirements.txt` ends with `-e .`, which installs the `app` package in editable mode so
the scripts can import it.

## Environment Variables

All settings live in `.env` (loaded by pydantic-settings). `.env` is gitignored, and
[.env.example](.env.example) documents every variable. The important ones:

| Variable | Default | Meaning |
|---|---|---|
| `GROQ_API_KEY` | (required) | LLM provider key |
| `PINECONE_API_KEY` | (required) | Vector DB + embeddings key |
| `PINECONE_INDEX_NAME` / `PINECONE_NAMESPACE` | `agentic-ai-ebook` | Where vectors live |
| `EMBEDDING_MODEL` / `EMBEDDING_DIMENSION` | `llama-text-embed-v2` / `1024` | Pinecone-hosted embedding model |
| `LLM_PROVIDER` | `groq` | `groq` or `anthropic` (then set `ANTHROPIC_API_KEY`) |
| `LLM_MODEL` / `VALIDATOR_MODEL` | `openai/gpt-oss-120b` / `openai/gpt-oss-20b` | Generation / validation models |
| `LLM_EFFORT` / `VALIDATOR_EFFORT` | `medium` / `low` | Reasoning effort |
| `TOP_K` | `5` | Chunks retrieved per question |
| `RETRIEVAL_THRESHOLD` | `0.33` | Minimum cosine similarity for a chunk to reach the LLM (calibrated, see below) |
| `CONFIDENCE_SCORE_CEILING` | `0.70` | Similarity treated as a "very strong" match |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `1000` / `150` | Characters |
| `LOG_FORMAT` | `json` | `json` or `text` |

Missing keys don't crash the server. `/health` still answers, and `/api/chat` returns
`503 {"error": "configuration_error", "message": "Missing required environment variable: …"}`.

## Ingestion

```bash
python scripts/ingest.py           # download → extract → clean → chunk → embed → upsert
python scripts/ingest.py --force   # re-embed even if nothing changed
```

It downloads the PDF to `data/raw/` if missing, writes the chunks to
`data/processed/chunks.jsonl` for inspection, creates the Pinecone index if needed, and
upserts the vectors.

Re-running is cheap. Each vector carries a fingerprint of (PDF bytes, chunking config,
embedding model). If the index already holds this exact ingestion, nothing is
re-embedded. Standalone steps are also available: `scripts/download_document.py` and
`scripts/create_index.py`.

Metadata stored with each vector:

```json
{"chunk_id": "page21_chunk00", "page_number": 21, "chunk_index": 0,
 "source": "Agentic AI eBook", "document_name": "Ebook-Agentic-AI.pdf",
 "chapter": "02 Anatomy of an Agentic AI System",
 "section": "2.3 Defining Characteristics of an Agent", "text": "…", "fingerprint": "…"}
```

## Run the API

```bash
uvicorn app.main:app --reload
```

Interactive docs are at <http://localhost:8000/docs>.

## Run the UI

```bash
streamlit run ui/streamlit_app.py      # http://localhost:8501; set API_URL if the API isn't on :8000
```

The UI shows:
- the answer, with its source pages;
- retrieval confidence, the grounded flag, the top similarity, and how many chunks were
  above the threshold;
- an expandable **Retrieved context** section listing every chunk Pinecone returned, with
  its page, score, and whether it was sent to the LLM and cited;
- a raw-JSON view of the API response.

Sample questions are one click away in the sidebar.

## Docker

```bash
docker compose up --build                              # API on :8000, UI on :8501
docker compose run --rm api python scripts/ingest.py   # one-off ingestion
```

## API Usage

```bash
curl http://localhost:8000/health
# {"status":"ok"}

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Agentic AI?"}'

curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the defining characteristics of an AI agent?", "top_k": 8}'
```

PowerShell:

```powershell
Invoke-RestMethod -Method Post -Uri http://localhost:8000/api/chat `
  -ContentType "application/json" -Body '{"question": "What is Agentic AI?"}'
```

| Status | When |
|---|---|
| 200 | Answer or safe fallback (see `grounded` / `status`) |
| 422 | Empty/whitespace question, question > 2000 chars, `top_k` outside 1-20 |
| 502 | Embedding or LLM provider failure (`embedding_error`, `llm_error`) |
| 503 | Missing configuration or Pinecone unavailable / index not ingested |

## API Response Example

This is a real response from the running API (`POST /api/chat` with
`{"question": "What is Agentic AI?"}`). It's trimmed to 2 of the 5 retrieved chunks, with
their text shortened.

```json
{
  "answer": "Agentic AI is an autonomous, proactive AI system that goes beyond reactive tools. It learns and adapts to new situations, makes independent decisions to achieve business goals, and acts as a super‑intelligent collaborator that can suggest improvements, draft proposals, track market trends, and even reach out to potential partners (p. 8). The book likens it to a coach, chef, coordinator, and project manager that continuously adapts strategies and drives progress with minimal supervision (p. 8). Unlike rule‑based automation, Agentic AI creates impact by anticipating needs, adjusting to disruptions, and aligning actions with real‑time objectives (p. 11).",
  "grounded": true,
  "status": "answered",
  "confidence": 0.794,
  "confidence_details": {
    "level": "high",
    "top_score": 0.6189,
    "mean_score": 0.5807,
    "relevant_chunks": 5,
    "retrieved_chunks": 5
  },
  "cited_pages": [
    8,
    9,
    11
  ],
  "retrieved_context": [
    {
      "chunk_id": "page08_chunk00",
      "page": 8,
      "section": null,
      "score": 0.6189,
      "above_threshold": true,
      "cited": true,
      "text": "Understanding the Shift from Reactive to Proactive Technology\nImagine Sarah, a busy entrepreneur juggling multiple projects. She's not just using a to …"
    },
    {
      "chunk_id": "page07_chunk00",
      "page": 7,
      "section": null,
      "score": 0.5867,
      "above_threshold": true,
      "cited": false,
      "text": "INTRODUCTION TO AGENTIC AI\nIn this section, we will define what Agentic AI is and, more importantly, what it’s not, as it’s often misunderstood. While …"
    }
  ]
}
```

For an off-topic question (*"What is the capital of France?"*), no chunk passes the
threshold (top score 0.020), so the LLM isn't called and the answer arrives in under a
second:

```json
{"answer": "I couldn't find enough information about this in the provided Agentic AI eBook.",
 "grounded": false, "status": "low_retrieval_score", "confidence": 0.0,
 "confidence_details": {"level": "none", "top_score": 0.0201, "mean_score": 0.0148,
                        "relevant_chunks": 0, "retrieved_chunks": 5},
 "cited_pages": [], "retrieved_context": ["…5 chunks, all below the threshold…"]}
```

`status` is one of:
- `answered`
- `low_retrieval_score`: no chunk passed the threshold, so the LLM was not called
- `not_in_context`: the generator found no answer in the chunks
- `failed_validation`: the validator found unsupported claims
- `declined`

Every status except `answered` returns the fallback text with `grounded: false` and
`confidence: 0`.

## Sample Questions

These are real outputs of the full pipeline: Pinecone retrieval → threshold → Groq
`gpt-oss-120b` answer → `gpt-oss-20b` validation. They were produced by
`python scripts/evaluate.py` and copied from
[evaluation/results/latest.md](evaluation/results/latest.md). The questions were chosen
by first confirming in the PDF text where each answer is.

### 1. What is Agentic AI?

> Agentic AI is an autonomous, goal‑driven form of artificial intelligence that goes beyond reactive tools. It learns and adapts to new situations, makes independent decisions, and proactively works toward business objectives (p. 9). Rather than merely processing inputs, it acts as a super‑intelligent collaborator—functioning as a coach, chef, coordinator, or project manager—suggesting improvements, drafting proposals, tracking trends, and even reaching out to partners without constant micromanagement (p. 8). It creates impact by anticipating needs, adjusting to disruptions, and aligning actions with real‑time goals (p. 11).

**Retrieved pages:** 8, 7, 9, 11, 53 · **cited pages:** 8, 9, 11 ·
**top similarity:** 0.619 · **confidence:** 0.79 (high) ·
**status:** `answered`

### 2. What are the defining characteristics of an AI agent?

> The eBook defines the following primary characteristics of an AI agent:
> - Autonomy: an autonomous, interactive, goal‑driven entity with its own state, behavior, decision‑making capabilities and the ability to self‑improve when performance falls short (p. 21).
> - Reactivity and Proactivity: can sense its environment and respond (reactive) and can take initiative toward its goals (proactive) (p. 21).
> - Beliefs, Desires, and Intentions (BDI): maintains beliefs about the world, desires as objectives, and intentions as plans of action (p. 21).
> - Social Ability & Communication: has a communication mechanism to interact with other agents or entities, enabling complex negotiation, coordination, and natural‑language exchange (p. 22).
> - Constitution: must follow regulations and policies, protect itself from compromise, and be trusted not to harm other agents (p. 22).
> - Memory: possesses long‑term memory of past interactions and short‑term memory of current context to support efficient task execution (p. 22).

**Retrieved pages:** 21, 22 · **cited pages:** 21, 22 ·
**top similarity:** 0.625 · **confidence:** 0.80 (high) ·
**status:** `answered`

### 3. How are AI agents different from LLMs?

> AI agents differ from LLMs in several key ways:
> - **Primary function**: Agents make decisions and take actions toward goals, showing autonomy and proactive behavior, while LLMs primarily understand and generate language (p. 10).
> - **Interactivity**: Agents operate continuously with minimal human input; LLMs respond reactively to prompts (p. 10).
> - **Decision‑making**: Agents have built‑in, context‑aware, goal‑driven decision‑making; LLMs lack inherent decision‑making capability (p. 10).
> - **Dependency**: Agents can function with little prompting and can integrate tools; LLMs need prompts to operate (p. 10).
> - **Autonomy & state**: Agents are autonomous, interactive, goal‑driven entities with their own state, behavior, and self‑improvement ability (p. 21). LLMs do not possess such autonomy (p. 9).

**Retrieved pages:** 10, 9, 21, 22 · **cited pages:** 9, 10, 21 ·
**top similarity:** 0.602 · **confidence:** 0.70 (medium) ·
**status:** `answered`

### 4. How does an agentic AI system decide which action to take?

> An agentic AI first perceives its environment through sensors and converts raw data into insights (p. 19). It then applies reasoning to detect patterns and draw conclusions (p. 19). Guided by its defined goals, the system plans a path to achieve those objectives (p. 19). The action selected is the one that best aligns with the goal, using internal models (model‑based agents) or direct goal prioritization (goal‑based agents) (p. 23). Memory and continuous learning from past interactions further refine the choice, enabling more effective and adaptive decision‑making (p. 20).

**Retrieved pages:** 20, 23, 11, 19, 17 · **cited pages:** 19, 20, 23 ·
**top similarity:** 0.564 · **confidence:** 0.70 (medium) ·
**status:** `answered`

### 5. What are the challenges of orchestrating multi-agent systems?

> The eBook lists several key challenges when orchestrating multi‑agent systems:
> - **Scalability** – Managing hundreds or thousands of agents requires robust infrastructure and algorithms that can dynamically add or remove agents without disrupting performance (p. 39).
> - **Reliability and fault tolerance** – The system must continue operating despite partial failures; a single agent’s failure must not jeopardize overall functionality (p. 39).
> - **Communication and coordination** – Ensuring seamless interaction among diverse agents is difficult due to differing functionalities, technologies, and standards; a uniform interaction protocol is often needed (p. 39).
> - **Interoperability issues** – Varying agent designs can cause compatibility problems that must be addressed (p. 39).
> - **Conflict management** – Autonomous agents may have overlapping goals or compete for resources, requiring automated mechanisms to resolve conflicts (p. 39).
> - **Agent selection** – When multiple suitable agents exist, the system must choose the most appropriate one based on context and efficiency (p. 39).
> - **Overall coordination failures** – Without effective orchestration, agents operate in silos, leading to high failure rates (70‑80%) and inefficiencies (p. 38).

**Retrieved pages:** 39, 37, 38, 41 · **cited pages:** 38, 39 ·
**top similarity:** 0.669 · **confidence:** 0.88 (high) ·
**status:** `answered`

### 6. How should an organization get started with implementing Agentic AI?

> To begin implementing Agentic AI, an organization should:
> - Set clear, business‑aligned goals and target high‑impact use cases such as predictive maintenance or personalization (p. 53).
> - Ensure data readiness and a robust, scalable infrastructure, investing in data quality and management practices (p. 53; p. 47).
> - Build cross‑functional teams that combine IT, operations, AI expertise, and domain knowledge (p. 53; p. 47).
> - Start with pilot projects or small‑scale experiments, using exhaustive validation and testing to uncover vulnerabilities (p. 46).
> - Track success with KPIs and continuously improve, establishing governance and performance metrics as the effort matures (p. 53; p. 52).

**Retrieved pages:** 53, 47, 46, 48, 52 · **cited pages:** 46, 47, 52, 53 ·
**top similarity:** 0.692 · **confidence:** 0.93 (high) ·
**status:** `answered`

### 7. What is the capital of France? *(out of scope)*

> I couldn't find enough information about this in the provided Agentic AI eBook.

**Retrieved pages:** 9, 19, 12 · **cited pages:** none ·
**top similarity:** 0.020 · **confidence:** 0.00 (none) ·
**status:** `low_retrieval_score`

### 8. How much does Konverge AI charge for its services? *(near-miss: the company is in the book, its pricing isn't)*

> I couldn't find enough information about this in the provided Agentic AI eBook.

**Retrieved pages:** 28, 2, 59, 4, 46 · **cited pages:** none ·
**top similarity:** 0.407 · **confidence:** 0.00 (none) ·
**status:** `not_in_context`

The out-of-scope question never reached the LLM (top score 0.020, below the 0.33
threshold). The near-miss did pass the threshold (0.407), because the chunks mention
Konverge AI, so the generator saw them. It found no pricing information and declined,
and the pipeline returned the fallback.

## RAG Pipeline

| Node | Role |
|---|---|
| `retrieve` | Embed the question (`input_type=query`), query Pinecone top-K with metadata and scores |
| `check_retrieval` | Keep chunks ≥ threshold; compute the confidence heuristic; if none remain, route to `fallback` without calling the LLM |
| `generate` | LLM answers from the relevant chunks only, as strict JSON `{answerable, answer, cited_excerpts}`; if not answerable, route to `fallback` |
| `validate` | Deterministic checks (non-empty, cites a provided excerpt), then an LLM judge returns `{unsupported_claims, supported}`; any unsupported claim routes to `fallback` |
| `finalize` / `fallback` | Validated answer, or the fixed fallback sentence with confidence 0 |

The routing functions are pure (`app/rag/graph/edges.py`), and nodes receive their
dependencies through the constructor (`RAGNodes`). The whole graph is exercised in
`tests/integration/test_rag_pipeline.py` with fake retriever and LLM objects.

## Chunking Strategy

- **Reading-order extraction.** PyMuPDF `get_text("blocks", sort=True)`, then cleanup:
  the running footer, page numbers, a bullet glyph that extracts as "բ", broken
  apostrophes, and "decision -making" style hyphen breaks.
- **One page per chunk.** Every chunk has exactly one page number, so page citations are
  precise.
- **Section-aware.** A numbered heading ("2.3 Defining Characteristics of an Agent")
  always starts a new chunk, so a chunk never straddles two sections.
- **1000 characters (about 200-250 tokens).** Pages hold 200-2,700 characters, so most
  become 1-3 chunks. At this size a chunk holds one idea (a bullet group, a table, a
  subsection): small enough to retrieve precisely, large enough to keep full sentences
  and their example.
- **150 characters overlap (about 15%).** Whole trailing sentences are carried into the
  next chunk when a section is split for size, so a sentence cut at a boundary survives.
  There's no overlap across headings (a new topic) or pages.
- **Contextual header.** The embedded text is `"<chapter> > <section>\n\n<chunk>"`, which
  helps short table fragments match topic-level questions. The stored and displayed text
  is the raw chunk.

The result is 106 chunks (median about 830 characters) from 60 pages.

## Retrieval Strategy

- **Top-K = 5** by cosine similarity (configurable per request, 1-20). That's enough to
  cover a section that spans 2-3 chunks without flooding the prompt.
- **Threshold = 0.33** (`RETRIEVAL_THRESHOLD`). Chunks below it are shown in the response
  but never sent to the LLM. If none pass, the pipeline returns the fallback without
  calling the LLM.
- **How 0.33 was chosen.** It was calibrated on the evaluation set with
  `llama-text-embed-v2`. The top similarity of each question fell into clearly separate
  bands:

  | Question type | Top-score range |
  |---|---|
  | Answerable (11) | **0.497 – 0.692** |
  | Clearly off-topic (France, FIFA, sourdough) | **0.020 – 0.165** |
  | Near-misses (LoRA, transformers) | 0.157 – 0.185 |
  | Near-miss (Konverge pricing) | 0.407: passes, then declined by the generator |

  0.33 is the midpoint of the gap (0.497 vs 0.165), so both sides have a 0.17 margin.
- **Why not 0.70?** It would have rejected *every* answerable question: the best match in
  the whole set scored 0.692.
- **Re-calibrate after changes.** If you change the embedding model or the chunking, run
  `scripts/evaluate.py`. It prints every score and the suggested threshold.

## Confidence Score

The confidence is a **retrieval-confidence heuristic** in [0, 1]. It is **not** a
calibrated probability that the answer is correct.

```
norm(s)    = clip((s − RETRIEVAL_THRESHOLD) / (CONFIDENCE_SCORE_CEILING − RETRIEVAL_THRESHOLD), 0, 1)
confidence = 0.5 · norm(top score)                        is there one strong match?
           + 0.3 · norm(mean of top-K scores)             relevant neighbourhood, or one lucky hit?
           + 0.2 · min(chunks above threshold / 3, 1)     enough supporting chunks?
level      : high ≥ 0.75, medium ≥ 0.50, low > 0, none = 0
```

`CONFIDENCE_SCORE_CEILING = 0.70` sits just above the strongest match seen in evaluation
(0.692), so a confidence of 1.0 is reserved for evidence stronger than anything
observed. With the first guess of 0.60, 7 of 11 answers showed 0.97-1.00, which looks
like over-claiming. At 0.70 the answered questions range from **0.36 to 0.93** and track
evidence strength:
- "How should an organization get started…" has five strongly matching chunks: **0.93,
  high**.
- "Why does an agent need long-term and short-term memory?" has one strong chunk and one
  weak one: **0.36, low**. It is still answered correctly, and the low number honestly
  signals thinner evidence.

Fallback answers report `confidence: 0`, since nothing is being asserted, while
`confidence_details` keeps the raw similarity numbers so you can see why. Each chunk's
raw cosine score is also in `retrieved_context[].score`.

## Hallucination Prevention

1. **Retrieval threshold.** Off-topic questions never reach the LLM.
2. **Grounded generation.** The prompt forbids outside knowledge, allows synthesis but
   not new conclusions, and requires `answerable=false` rather than a guess. Inline page
   references and excerpt citations are required, and citations to excerpts the model
   wasn't given are discarded.
3. **Validation node.** An answer without a valid citation fails immediately. Otherwise
   an LLM judge checks it claim by claim, and a single unsupported claim withholds the
   whole answer. Inconsistent or declined verdicts also fail safe.
4. **Safe fallback.** Every failure path returns the same fixed sentence with
   `grounded: false`.
5. **Also:** temperature 0, strict JSON-schema outputs (no free-text parsing), and
   HTML-escaped inputs so a question can't inject instructions by closing a tag.

## Evaluation

[evaluation/datasets/rag_test_cases.json](evaluation/datasets/rag_test_cases.json)
contains 17 cases:
- **11 answerable**, each with the page(s) that contain the answer, verified against the
  PDF text;
- **6 unanswerable**: 3 clearly off-topic (France, FIFA, sourdough) and 3 *near-misses*
  (LoRA fine-tuning, Konverge AI pricing, transformer attention). The near-misses sound
  related, may pass the retrieval threshold, and must be stopped by generation or
  validation.

```bash
python scripts/evaluate.py          # all cases
python scripts/evaluate.py --only a01 u05
```

It reports:
- answerable → grounded rate;
- unanswerable → fallback rate;
- expected-page hit rate (retrieval recall);
- mean confidence and mean latency;
- a suggested threshold.

Results are written to `evaluation/results/latest.{md,json}`.

**Full-pipeline results** (Pinecone + Groq, `RETRIEVAL_THRESHOLD=0.33`):

| Metric | Result |
|---|---|
| Answerable questions → grounded answer | **100%** (11/11) |
| Unanswerable questions → safe fallback | **100%** (6/6, including all 3 near-misses) |
| Expected page among retrieved chunks | **100%** (11/11) |
| Mean confidence of grounded answers | 0.72 |
| Mean latency per question | 5.7 s (under 1 s for fallbacks that skip the LLM) |

The per-question table is in [evaluation/results/latest.md](evaluation/results/latest.md).

With 17 questions this is a smoke test of the design, not a benchmark. A larger,
independently written question set is the obvious next step.

**LLM-layer check run during development.** Before Pinecone was connected, this ran the real Groq generator and validator over the real eBook chunks,
with a keyword-overlap stand-in for retrieval that forced *every* question, off-topic
ones included, through to the LLM. It tested layers 2-4 on their own:

- **Run 1** (`gpt-oss-120b` for both steps): 16 of 17 correct.
  - All 6 unanswerable questions, including the 3 near-misses, were declined.
  - In the one miss, the stand-in surfaced the wrong pages and the generator added a
    sentence that wasn't in them ("start with a pilot…"). The validator flagged exactly
    that sentence and the fallback was returned: a false "not found", never a
    fabrication.
- **Run 2** (tightened prompt, validator on `gpt-oss-20b`): 17 of 17 correct, at 2-13 s
  per question instead of 15-26 s.
- **Judge spot-check on `gpt-oss-20b`:**
  - flags the Run 1 invented sentence;
  - flags a planted false claim ("first defined by Alan Turing in 1950");
  - accepts a faithful paraphrase.


## Testing & Code Quality

```bash
pytest                                  # 135 tests, no network or API keys needed
RUN_LIVE_TESTS=1 pytest tests/integration/test_pinecone.py   # optional, against your index
ruff check . && ruff format --check .
mypy                                    # strict
```

What the tests cover:
- PDF extraction and cleaning (including a generated PDF and the real eBook);
- chunking, including sizes, overlap, sections and chapter resets;
- score normalisation and the confidence formula;
- retrieval filtering;
- the grounding pre-checks and the validator;
- prompt-injection escaping;
- idempotent ingestion and stale-vector cleanup;
- settings validation;
- both LLM adapters, with fake SDK clients;
- every route through the LangGraph workflow;
- API schemas, the 422/502/503/500 mappings and error redaction;
- the Streamlit UI, headless via `AppTest`.

CI (`.github/workflows/`) runs lint, format, mypy and pytest on every push.

## Limitations

- **Single document, single turn.** There is no conversation memory; each question is
  answered independently.
- **Per-page chunks.** A section that spans a page break becomes separate chunks. The
  shared section header in the embedded text mitigates this, but a sentence split across
  a page is split in the index too.
- **Images and diagrams aren't read.** Text inside images (the cover, some charts, the
  chapter title art) isn't extracted, so there's no OCR.
- **Tables are flattened** into row-ordered text blocks. The LLM handles this well, but
  cell alignment is lost.
- **Calibrated on a small set.** The threshold and confidence ceiling come from 17
  questions. They separate this set cleanly but should be re-checked on a larger one.
- **Page references come from the LLM.** Claims are validated against the excerpts, but
  the inline "(p. N)" is written by the model. Occasionally a supported claim cites a
  neighbouring excerpt's page: in sample 6, "pilot projects" is on p. 52, which was
  retrieved and cited, but the text says p. 46. `cited_pages` and `retrieved_context` are
  exact.
- **The confidence is a heuristic.** It measures retrieval strength, not answer
  correctness.
- **The LLM judge can err.** It is conservative by design, so a borderline paraphrase can
  be rejected (a false fallback) more often than a fabrication accepted.
- **Latency.** Two LLM calls per question; on Groq's free tier (8K tokens/minute per
  model) bursts of questions hit rate limits and slow down.

## Future Improvements

- **Hybrid search** (BM25 + dense, e.g. Pinecone sparse-dense) for exact terms like "BDI"
  or "MCP".
- **Reranking** of the top-20 with a cross-encoder (Pinecone `bge-reranker-v2-m3`) before
  the threshold.
- **Conversation memory** with a query-rewrite node for follow-ups ("tell me more about
  the second one").
- **Streaming responses** (SSE) so the answer appears while validation runs.
- **Citation extraction at sentence level**, highlighting the exact supporting span.
- **Observability**: LangSmith or OpenTelemetry traces per graph node, and token and cost
  metrics.
- **Automated RAG evaluation** in CI (faithfulness and answer relevance with RAGAS or an
  LLM-judge rubric), plus a larger question set.
- **Async provider clients** for higher concurrency.
- **Multi-document support**, with a namespace per document and metadata filters.

## License

Code: MIT (see [LICENSE](LICENSE)). The eBook is © Konverge AI. It is downloaded at
ingestion time and is not included in this repository.
