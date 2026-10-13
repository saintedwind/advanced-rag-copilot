# Evidence review release 0.3 — 2026-10-13

## Measured development results

| Check | Earlier version | Current version |
|---|---:|---:|
| Expanded suite, real BGE embeddings | 28/36 | 36/36 |
| Answerable questions in expanded suite | 26/26 | 26/26 |
| Correct abstentions in expanded suite | 2/10 | 10/10 |
| Further challenge suite | first screen: 3/8 | 6/8 |
| Original real HTTP extractive smoke suite | 11/11 | 11/11 |
| Expanded real HTTP extractive suite | not measured | 36/36 |
| Expanded local Qwen selector, real HTTP | not measured | 36/36 |
| Behavioral/boundary tests | 19 | 25 |

The suite contains eight authored synthetic English documents and 36 authored questions. Questions, normalization aliases and screens were inspected/refined during development. The challenge cases were also inspected and used for refinements: they are not an untouched holdout. Passing these checks is not a customer accuracy estimate or evidence of multilingual quality. Baseline, intermediate challenge and current raw outputs remain in evidence-baseline.json, evidence-challenge-initial.json, evidence-current.json and evidence-challenge.json. Every citation in the current offline benchmark is reconciled against the original file hash and exact normalized-text slice.

## Real execution

Loopback uvicorn, persistent Qdrant and actual cached BAAI/bge-small-en-v1.5 were used for both HTTP suites. The test reads each cited passage through the evidence endpoint and compares text and source hashes with the query result. Raw responses include corpus fingerprints and per-query screens. The model run uses cached Qwen3-0.6B Q8_0 through llama.cpp at 127.0.0.1:8081, with 96 output tokens maximum per call; it selects existing sentences and cannot invent new answer text. No paid endpoint or API key is required. Model output is still semantically unverified.

Desktop/mobile browser checks cover indexing, numeric policy evidence, expanded citations, downloading a complete JSON evidence record, unrelated and related-unanswerable questions, and horizontal overflow. Screenshots were captured with an isolated Windows Chrome test profile. They do not use the user's existing browser/profile.

## Known failures and limits

The current challenge falsely abstains on “How frequently are backup copies generated?” and “Who is responsible for escalating an incident?”. Conservative lexical coverage does not recognize those paraphrases. Main-suite success therefore must not be extrapolated to unrestricted questions. The temporal screen prevents one observed refund-duration false positive by requiring a relevant time expression; it is not a general relationship/entailment model.

Source-byte hashes and LF-normalized character/line offsets provide reproducible evidence, not source authority. Conflicting policy versions, cross-document synthesis, semantic completeness and arbitrary prompt injection are not solved. The recognized instruction-pattern screen can exclude legitimate discussions and can miss obfuscated attacks. Documents are synthetic and the optional local model is small; no production/customer or high-confidence claim is made. Default corpus replacement and the existing partial-write recovery limitation remain unchanged.
