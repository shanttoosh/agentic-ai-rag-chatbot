# Design decisions

Each entry covers what was chosen, why, and what it costs.

### 1. LLM: Groq `openai/gpt-oss-120b` behind a `StructuredLLM` Protocol (Claude as alternative)

- **Why.** Groq is fast (about 1 s per call when not rate-limited), has a free tier, and
  supports **strict JSON-schema outputs** on gpt-oss. Strict mode means answers,
  citations and validator verdicts come back as validated Pydantic objects, never free
  text we'd have to parse.
- **Validator.** It defaults to `openai/gpt-oss-20b`. Judging "is this claim in the text?"
  is narrower than writing the answer, and on Groq's free tier each model has its own
  8K tokens/minute budget, so splitting generation and validation across models roughly
  doubles throughput.
- **Alternative.** `ClaudeStructuredLLM` implements the same Protocol. Setting
  `LLM_PROVIDER=anthropic` switches over with no other code changes, which also shows
  that the interfaces are real rather than decorative.
- **Cost.** Free-tier rate limits add latency under load (429 → SDK retry with backoff).
  A paid tier removes this.

### 2. Embeddings: Pinecone Inference `llama-text-embed-v2` (1024-d)

- **Why.** One provider and one API key for both vectors and search. The model is
  asymmetric (`input_type="passage"` for chunks, `"query"` for questions), which suits
  question → passage retrieval, and it's included in Pinecone's free tier.
- **Cost.** Similarity scores are model-specific, so the threshold must be calibrated for
  this model (see 6). Groq has no embeddings API, so it can't take this role.

### 3. Chunking: own structure-aware chunker, per page, 1000 chars / 150 overlap

- **Why.** The eBook is 60 designed pages, not continuous prose: headings, tables, bullet
  lists and callouts. The chunker:
  - keeps each chunk on one page, so every chunk has exactly one page number to cite;
  - starts a new chunk at every numbered section heading, so a chunk never mixes two
    sections;
  - packs whole sentences up to 1000 characters (about 200-250 tokens). Most pages
    become 1-3 chunks, and a chunk is roughly one idea: a bullet group, a table, a
    subsection;
  - carries 150 characters of overlap (whole trailing sentences) across size splits,
    but never across headings.
- **Metadata.** Chapter and section come from the table of contents and the numbered
  headings, and are prepended to the text that gets embedded. That helps short table
  chunks match topic-level questions.
- **Why not LangChain's splitter.** It's about 150 lines of pure, fully unit-tested
  Python with no framework dependency in core logic, and page-exact metadata is easier
  when we own the loop.
- **Cost.** A section that spans a page break becomes two chunks. The overlap doesn't
  bridge pages, but the shared section heading in the embedded text mitigates this.

### 4. PDF extraction: PyMuPDF `get_text("blocks", sort=True)`

The PDF's internal order follows how objects were drawn. For example, "1. Retail" came
*after* its bullets. Position-sorted blocks restore reading order and keep table cells
grouped by row.

Cleaning removes the running footer, bare page numbers, a bullet glyph that extracts as
the Armenian letter "բ", mis-encoded apostrophes, and "decision -making" hyphenation.
All of these were found by inspecting the extracted text, not guessed.

### 5. Idempotent ingestion via fingerprints; upsert-then-delete-stale

Each vector stores `fingerprint = sha256(pdf bytes | chunker config | embedding model |
version)`. If the last chunk carries the current fingerprint and the vector count
matches, re-running `ingest.py` does nothing, so no duplicate embeddings and no cost.

On change, new vectors are upserted first and vectors with obsolete IDs are deleted
afterwards. The namespace is never empty, so the API keeps answering during a re-ingest.
IDs are deterministic (`page12_chunk03`), so upserts overwrite rather than duplicate.

### 6. Retrieval threshold: 0.33, calibrated on the evaluation set

The brief suggested 0.70, but question-to-passage cosine similarity is much lower than
that with `llama-text-embed-v2`. In evaluation:

| Question type | Top-score range |
|---|---|
| Answerable (all 11) | 0.497 – 0.692 |
| Clearly off-topic | 0.020 – 0.165 |

0.70 would have rejected every answerable question. `scripts/evaluate.py` suggests the
midpoint of the gap, 0.331, so the threshold was set to **0.33**, which leaves a 0.17
margin on each side.

No answerable question lost a relevant chunk compared with the first default of 0.30.
Near-misses are not what the threshold is for: "Konverge AI pricing" scored 0.407 and
was correctly stopped by the generator.

The calibration uses only 17 questions, and it is model- and chunking-specific. Re-run
the evaluation after changing either.

### 7. Validation with an LLM judge, failing safe

A second call costs latency but is the only check that reads the answer against the
evidence. Cheap deterministic checks run first (no citation → fail without calling the
judge).

Anything ambiguous falls back: an inconsistent verdict, a refusal by the judge, a
truncated output. For a system whose one job is "don't make things up", a false "I
couldn't find it" is much cheaper than a confident fabrication.

In a live run, the judge caught a single invented sentence ("start with a pilot…") that
the generator added when retrieval surfaced the wrong pages. The smaller `gpt-oss-20b`
judge was then checked directly:
- it flagged that sentence;
- it flagged a planted false claim;
- it accepted a faithful paraphrase.

That is the evidence for defaulting the validator to 20b.

### 8. Confidence = retrieval heuristic; 0 for fallbacks

This follows the brief: a documented combination of top score, mean top-K score, and
the count of relevant chunks. It is not presented as a probability.

We deliberately don't fold the LLM's self-assessment into the number, so it stays
explainable. Fallbacks report 0 so a UI never shows "0.8 confidence" next to "I couldn't
find it".

The normalisation ceiling is **0.70**, just above the strongest match seen in evaluation
(0.692). With 0.60, 7 of 11 answers showed 0.97-1.00, which reads as over-claiming. At
0.70 answers range from 0.36 to 0.93 and follow evidence strength, so 1.0 is reserved
for matches stronger than anything observed.

### 9. Architecture: Protocols + composition root, no DI framework

- `domain/interfaces` defines Protocols (structural typing, no inheritance needed).
- `bootstrap.py` is the only module that imports concrete providers, and it exposes
  context managers so clients are always closed.
- FastAPI receives a fully built `ChatService` through `app.state` and a `Depends`
  function.
- No singletons and no module-level clients. Tests build the same services with fakes.

### 10. Sync endpoints

The provider SDKs are synchronous. A `def` route runs in FastAPI's thread pool, which is
correct and simple. `async def` with sync SDK calls inside would block the event loop.
Moving to async clients is a future step, not a requirement for this load.

### 11. Streamlit UI as a separate HTTP client

The UI imports nothing from `app/`. It calls `POST /api/chat` like any other client.
This keeps the API as the single backend (the brief's requirement), and the UI can show
exactly what the API returns: the answer, confidence, the grounded flag, and every
retrieved chunk with its score and whether it was cited.

## Deviations from the suggested project tree

| Suggested | Here | Reason |
|---|---|---|
| `api/v1/endpoints/*` | `api/routes/*`, mounted at `/api` | The brief requires `POST /api/chat`; versioning would be `/api/v1/chat`. Mounting under a new prefix is a one-line change. |
| `infrastructure/*/openai.py` | `groq_llm.py`, `claude.py`, `pinecone_embedder.py` | Providers chosen for this build (see 1-2). |
| `infrastructure/vector_store/pinecone.py` | `pinecone_store.py` | A module named `pinecone` inside the package invites confusion with the SDK. |
| `rag/retrieval/reranker.py` | omitted | No reranker is used. An empty placeholder file would be dead code. Listed as a future improvement. |
| `prompts/query_rewrite.txt` | omitted | The API is single-turn, so there's nothing to rewrite from. Would come with conversation memory. |
| `schemas/document.py` | `schemas/common.py` | The API exposes no document endpoints. Health and error schemas live here. |
| (none) | `app/bootstrap.py` | Composition root; keeps wiring out of `main.py` and the scripts. |
| (none) | `ui/streamlit_app.py` | Demo UI, requested separately. |
