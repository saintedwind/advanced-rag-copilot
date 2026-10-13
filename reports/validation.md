# Validation ledger

Review date: 2026-10-13. Python 3.12 / Windows CPU. This is a synthetic-document smoke test, not an independent benchmark or a customer deployment.

| Check | Observed result |
|---|---|
| Behavioral and boundary tests | 19 passed; deterministic embedding test double, actual local Qdrant |
| Live HTTP, real BGE embedding, extractive mode | 11/11 source/phrase or abstention checks passed |
| Live HTTP, real BGE embedding, Qwen3-0.6B Q8 local selection | 10/11 checks passed |
| Sample ingestion | 3 documents / 9 persisted chunks |
| Exact query cache | verified hit after a repeated request in both modes |
| Local successful model calls | median 2.077s end-to-end; 2034 recorded local tokens, no paid API |

Raw question-by-question results: [extractive](live-extractive.json), [local model](live-local.json). The initial critical-response check incorrectly expected “1 hour” rather than the source's “one-hour”; the matcher was corrected before re-running. The case set was also used during development, so these are development checks, not held-out accuracy estimates.

The first local-model run selected irrelevant opening sentences for scheduled-maintenance notice. Candidate sentences were subsequently ranked by embedding similarity and Markdown headings excluded. The final run still returned `generation_unavailable` for the audit-log export-format question. It exposes retrieved sources and low confidence instead of fabricating an answer. All three unrelated questions abstained without a model call. Default extractive mode remains the recommended presentation mode; the small local model is optional and experimental.

Model: official Qwen3-0.6B GGUF Q8_0, revision `23749fefcc72300e3a2ad315e1317431b06b590a`; SHA-256 `9465e63a22add5354d9bb4b99e90117043c7124007664907259bd16d043bb031`. Runtime: official llama.cpp b11476, context 4096, 4 CPU threads, one slot. Model processes were stopped after testing. Weights are not included in this repository.

Not yet verified here: production authentication, load/concurrency behavior, crash-atomic corpus replacement, remote Qdrant, optional FlashRank, RAGAS, Persian semantic quality, Docker runtime. Chromium acceptance is executed separately in CI; a workflow definition alone is not evidence of a pass.

## Changes from the initial upload

Created an executable test suite and README. Removed paid-provider defaults and unsupported HyDE/semantic-cache/production-grade claims. Added persistent BM25 rebuilding, stable chunk identities, stale-chunk deletion, request-option preservation through the graph, bounded ingestion, model-output validation, explicit failure/abstention states, exact cache invalidation and source inspection UI. Pinned the directly tested runtime packages.
