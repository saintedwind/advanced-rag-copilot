# Publication audit — 2026-11-01

## Scope

The code already matched GitHub. A previously local real-reference desktop capture is now included alongside the publication receipt.

## Validation

35 Python regression tests passed again on 2026-11-01. The first environment attempt failed locally; the recheck used the configured project dependencies and a workspace temporary directory. No test failure was hidden or reclassified as model evidence.

Published real curl retrieval evidence remains 30/32 extractive and 26/32 optional small-model selection. Local JSON differences were formatting only; canonical historical records are retained.

[Original evidence and limitations](reference-validation-v04.md)

## Limitations retained

Two known real-document false abstentions remain. Lexical guards and reader margins are heuristics, not semantic confidence. Screenshot is an earlier actual capture, not a fresh November experiment.

## Publication boundary

Source and selected attributed evidence only. No credentials, private documents, raw downloaded evaluation inputs, model weights, local databases, caches or temporary runtime state are published. Historical experiments are not renamed as fresh validation. No production/customer accuracy claim is added.

Pre-publication main revision: `a5b6c62d0195fb38b04bf6b8aaba1d9c959b30db`. GitHub Actions on the published source is the independent CI record; a historical green run is not substituted for a new code revision.
