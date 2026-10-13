# Knowledge Evidence Desk

[![Validation](https://github.com/saintedwind/advanced-rag-copilot/actions/workflows/validation.yml/badge.svg)](https://github.com/saintedwind/advanced-rag-copilot/actions/workflows/validation.yml)

A reproducible document-evidence portfolio by Peyman Gholamhosseini. Review synthetic operational policies or a pinned snapshot of real curl documentation, then inspect source passages, line ranges and provenance.

![Actual Chromium desktop capture](reports/desk-desktop.png)

[Mobile capture](reports/desk-mobile.png) | [Verified CI run](https://github.com/saintedwind/advanced-rag-copilot/actions/runs/38444286641)

## What runs

- FastAPI desk and API, Qdrant local persistence, FastEmbed dense retrieval and Unicode BM25, combined with weighted reciprocal rank fusion.
- Stable content-derived chunk IDs; re-ingestion replaces the selected corpus and removes stale chunks. BM25 is rebuilt from persisted payloads after restart.
- A cosine threshold gates retrieval. Its value is a heuristic, **not calibrated confidence**.
- A conservative same-source topic screen rejects missing question facets and recognized instruction-like source text. Time questions also require a relevant time expression. This is a lexical safeguard, not semantic entailment; unfamiliar paraphrases can be rejected.
- A bounded local SQuAD2 reader can rescue candidates rejected by the lexical screen. It selects only a source span, uses a fixed answer-versus-null margin of 3, respects named command-option scope and never generates new text. Its margin is not calibrated confidence. Missing protected facets and recognized instruction-like content are not rescued.
- Query normalization covers a small, documented set of wording variants and preserves quoted identifiers and numeric values. It is not a universal paraphraser.
- Citations include source-byte SHA-256, normalized-text character offsets and line ranges. `GET /evidence/{chunk_id}` returns the indexed passage; removed/replaced passages return 404. Every query includes a corpus fingerprint and the desk can export its evidence record.
- Default **extractive** mode returns source passages without a language-model call. Optional **local** mode asks a loopback llama.cpp model to select numbered source sentences; sentence embeddings rank the candidates, then the server renders those exact sentences and validates indices and completion status.
- LangGraph coordinates a bounded retrieval/normalization retry and synthesis. This is not autonomous reasoning or HyDE.
- Exact-query in-memory cache (128 entries), invalidated on ingestion. This is **not semantic caching**.
- Optional FlashRank; disabled by default. If unavailable, retrieval order is used.

## A two-minute review

1. Build the sample index and ask: **What is the Business plan critical response target?** Inspect the source's one-hour target and its distinction from resolution guarantees.
2. Ask: **What is the capital of Argentina?** The desk should abstain instead of treating an unrelated policy as an answer.
3. Repeat the supported question: inspect the exact-query cache hit. Re-index and verify that the cached result is invalidated.
4. Restart the service: the persisted vectors and rebuilt lexical index should still be available.

Version 0.4 checks: 35 behavioral/boundary tests; 11/11 original and 36/36 expanded extractive HTTP checks; the known paraphrase challenge improved from 6/8 to 8/8. Real-document evaluation is 30/32 (22/24 supported, 8/8 abstentions) with actual BGE/ONNX inference and loopback HTTP. Optional small-Qwen sentence selection on that real corpus is 26/32; full-passage extractive mode remains the default. These are evidence-presence checks on authored questions, not semantic completeness scores or a customer accuracy estimate. [Current validation and remaining failures](reports/reference-validation-v04.md).

![Actual real-document review](reports/desk-reference.png)

## Pipeline

```mermaid
flowchart LR
    D[Bounded UTF-8 documents] --> I[Stable chunks]
    I --> V[Persistent Qdrant + rebuilt BM25]
    Q[Question] --> V
    V --> R[Rank fusion + evidence gate]
    R --> E[Exact source passages]
    R --> L[Optional local sentence selection]
    E --> C[Inspectable citations]
    L --> C
```

## Run locally

Python 3.12 or newer. First startup downloads the embedding model; subsequent runs reuse `.runtime/models`.

```sh
python -m venv .venv
# Activate the environment for your shell.
python -m pip install -e ".[dev]"
uvicorn src.app.api:app --host 127.0.0.1 --port 8000
```

Open http://127.0.0.1:8000 and select **Build sample index**. API documentation: `/docs`.
Choose **Real curl documentation / pinned snapshot**, select **Build reference index**, and ask **What certificate format does --cacert require?** This demonstrates local reader rescue and upstream citations. Indexing replaces the active collection. API: `POST /ingest` with `{"collection":"curl"}`; arbitrary source paths cannot be combined with named collections.

```sh
curl -X POST http://127.0.0.1:8000/ingest -H "Content-Type: application/json" -d '{}'
curl -X POST http://127.0.0.1:8000/query -H "Content-Type: application/json" -d '{"query":"What is the Business plan critical response time?","top_k":3}'
python -m pytest -q
python -m scripts.evidence_benchmark current
python -m scripts.evidence_benchmark challenge
python -m scripts.live_check --evaluation
python -m scripts.reference_benchmark
python -m scripts.live_check --reference
```

Docker: `docker compose up --build` uses local Qdrant persistence and binds the host port to loopback. Allow time for the initial model download.

## Optional real local language model

Start an OpenAI-compatible llama.cpp server on **127.0.0.1:8081**, with a model supporting structured output. Set `LLM_PROVIDER=local` before starting the desk. No paid API or API key is used. Requests are limited to 96 output tokens and 3,000 context characters. Source selection can still be semantically wrong even when its citations are valid; review the original source.

## Boundaries and evidence

The three default documents and eight separate evaluation documents are **synthetic**, not customer policies. The evaluation set covers backups, API operations, privacy, incidents, billing, security, languages and untrusted archived content. The default embedding model is English-oriented; Persian token normalization is implemented but Persian retrieval quality is not claimed. This is a local single-process demonstration, not a hardened multi-tenant production service. No authentication, distributed cache or audited access control is included. Do not expose the service publicly or index confidential data without adding these controls.

The additional eight curl documents are actual upstream Markdown, pinned at commit `2b348649c68c1cd8cc1d055487f2fb2b59a353cb`. [Source manifest](data/curl_manifest.json) records exact byte digests; the [upstream license notice](data/curl-COPYING.txt) is retained. Indexing excludes curl build frontmatter but keeps line/character offsets into the unmodified file. Named `--options` are matched to declared document headings, so a neighboring option that merely mentions the name cannot substitute for its documentation. This intentionally limits cross-option synthesis. The `.gitattributes` rules preserve reference bytes across platforms. Refresh only explicitly with `python -m scripts.import_curl`.

The default reader is quantized tinyroberta-squad2 from deepset / ONNX Community, pinned and hash-verified; [model attribution and CC BY 4.0 notice](data/QA_MODEL_NOTICE.md). First needed use downloads approximately 82 MB of weights plus a 3.6 MB tokenizer into the ignored cache. CPU only, two inference threads, 384 input tokens and 48 answer tokens maximum; no generated tokens or paid API. Set `QA_READER_PROVIDER=none` for conservative-only mode. Reader failure falls back explicitly and is not cached as a permanent negative answer.

Ingestion is restricted to `SOURCE_ROOT`, UTF-8 Markdown/text, 200 files, 1 MB per file and 10 MB total. Ingestion replaces the corpus, rather than appending to it. A failed database write can leave a partial index; rebuild from the source directory. The exact-query cache is not persisted. Changing the embedding model requires a new collection/index.

See `reports/validation.md` for measured results and remaining limits. Unit tests use deterministic embeddings and must not be reported as neural retrieval evaluation. Real embedding and live HTTP checks are recorded separately.

## Design choices

The project prioritizes inspectable evidence over polished unsupported answers. Provider failures return `generation_unavailable` with low confidence; no result is labeled high confidence merely because retrieval found a passage. The UI inserts user/model/source content as text, never HTML.

File hashes identify bytes, not publisher authority or policy validity. Character offsets refer to UTF-8 text with CRLF/CR normalized to LF; they are not byte offsets. Indexed passages are retained snapshots and can differ from edited files until re-ingestion. Re-index older collections to populate provenance metadata. Same-source term coverage and time expressions cannot establish all relationships, resolve policy conflicts or ensure a complete answer. The instruction-like-text screen detects a narrow pattern family and is not a general prompt-injection defense.

MIT license. Embedding weights and optional runtime/model licenses remain their respective authors' licenses.
