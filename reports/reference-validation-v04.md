# Reference and paraphrase validation — 0.4 / 2026-10-13

## Measured outcomes

| Check | Result |
|---|---:|
| Behavioral/boundary tests | 35 passed |
| Original extractive HTTP smoke checks | 11/11 |
| Expanded synthetic evidence checks | 36/36 |
| Known paraphrase challenge | 8/8, previously 6/8 |
| Actual curl documentation, default full-passage mode | 30/32 |
| Supported reference questions | 22/24 |
| Unsupported reference questions, correct abstention | 8/8 |
| Optional small-Qwen sentence selection, real HTTP | 26/32 |

The first conservative transfer run was 25/32 (recorded in the development tool output). An intermediate reader run was 29/32; its raw result is retained in reference-curl-reader-initial.json. The final reference-curl.json and live-reference-extractive.json contain 30/32 and every citation. Queries were fixed before the first transfer run, but results were inspected and used to improve scope handling and frontmatter indexing. This is a development evaluation, not an untouched holdout. The pass criterion is expected source/phrase presence or abstention, not an assessment of every answer's semantic completeness.

## Real sources and models

Eight curl upstream documents were retrieved at immutable commit 2b348649c68c1cd8cc1d055487f2fb2b59a353cb. Original bytes, license notice, retrieval timestamp and digests are retained. No confidential customer documents or customer query logs are represented. Copyright/build frontmatter is excluded from indexed text; line/character offsets reconcile against the unmodified LF-normalized source. Named-option filtering prevents --connect-timeout being answered by a --max-time document that only references the former.

Actual BAAI/bge-small-en-v1.5 retrieval and quantized ONNX tinyroberta-squad2 run locally. The reader is pinned to revision 7c9f69b7e6228375169a4553bcfa6639152e3a69, with model/tokenizer byte checks. Fixed null-margin threshold: 3; no threshold tuning followed inspection of this corpus. At most three candidate chunks, 384 input tokens, 48-token spans, CPU-only and two inference threads. This extracts offsets into source text; it is not a generative or calibrated entailment model. Reader unavailable/protected-facet paths abstain or preserve the conservative result, and transient reader failure is not cached.

The optional Qwen3-0.6B sentence selector was tested through actual loopback HTTP: 26/32. Its four additional misses compared with full-passage mode are recorded, so the default remains extractive full passages. ONNX extraction and optional Qwen use no paid endpoint. The cached model/runtime processes were stopped after the check.

## UI and operational verification

Desktop/mobile checks cover sample indexing, source expansion, JSON download and abstention; they also switch to the named real corpus, verify a PEM answer through reader rescue and capture desk-reference.png. A transient browser export cancellation was corrected by attaching the download link and delaying blob-URL revocation. Tests use an isolated Windows Chrome profile, not the user's current browser.

Regression coverage includes quoted identifier/numeric preservation, option scope, original-source offsets after frontmatter removal, protected missing facets, reader failure, rejection of invented/out-of-bounds span text, named-collection/path exclusivity and transient-failure cache behavior. Model weights are excluded from Git; software, curl documents and model weights have separate notices.

## Remaining limitations

The real corpus still falsely abstains on “What HTTP response codes cause --fail errors?” and “Which decimal separator does --connect-timeout require?”. Those failures remain in the raw results. The 8/8 paraphrase challenge is a known regression suite, not general paraphrase quality. Margin 3 and the topic/facet guards are heuristics; no probability, complete entailment, general prompt-injection resistance, multilingual quality or production security claim is made. Conflicting policy versions, unrestricted cross-document reasoning, authentication and multi-tenant access control remain outside this local demo.

## Independent published verification

[GitHub Actions run](https://github.com/saintedwind/advanced-rag-copilot/actions/runs/38456113333) passed all stages in 1m 44s on Ubuntu 24.04 / Python 3.12: 35 tests, synthetic and real-document model checks, actual HTTP, Chromium acceptance and Docker ingestion / provenance. Tested code revision: `4d97210886906b4cd5cf68e141d059cb07490d9b`.

[Evidence bundle](https://github.com/saintedwind/advanced-rag-copilot/actions/runs/38456113333/artifacts/11841969688) includes raw results and v0.4 desktop/mobile/reference screenshots. Artifact ID: 11841969688; ZIP size: 531777 bytes; SHA-256: `2ee92703baa8abb3dd746950a3ae7e15393eaaebffd021e92ad9042eba3b9fd2`. These are hosted CI artifacts with a finite retention period; raw evaluation JSON is also committed in reports. Browser file upload failed during publication, so the new captures are provided in this verified bundle rather than represented by a missing repository image link.
