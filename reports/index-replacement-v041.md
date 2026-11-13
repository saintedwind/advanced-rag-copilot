# Recovering from an interrupted index replacement

Version 0.4.0 upserts into the searchable collection and then removes old points. An interruption between those writes can leave a mixed corpus on disk even though the running process still reports the old fingerprint. Version 0.4.1 writes a separate collection, reads back every chunk and its metadata, builds BM25, and switches the active Qdrant alias only after validation. Successful activation invalidates the query cache. The first upgrade can read an existing collection without rebuilding it.

## Live verification

The comparison used two isolated Qdrant stores, real BGE-small-en-v1.5 embeddings, the local ONNX reader, and HTTP requests to Uvicorn on Windows. Each run first indexed the original `cacert.md` and verified that the returned PEM citation matched the source offsets. The replacement used all eight [pinned upstream curl documents](../data/curl_manifest.json). The test deliberately stopped the database write immediately after Qdrant persisted one `location.md` point. Both versions returned HTTP 503.

| Check | 0.4.0 | 0.4.1 |
|---|---|---|
| Active chunks before replacement | 3 | 3 |
| Active chunks after restart following the failed write | 4 | 3 |
| Corpus fingerprint after that restart | Changed | Preserved |
| Active chunks after a successful retry and restart | 18 | 18 |

All 37 automated tests passed, including legacy collection migration and interrupted replacement. The existing 32-question reference evaluation passed 30, matching the published 0.4.0 score. Two supported questions still received abstentions. The evaluation is scoped to the pinned curl corpus and the project's authored questions. [Recorded HTTP evaluation](live-reference-extractive.json).

## Operation and reproduction

The single API process serializes requests during ingestion. Plan for disk space for the active and staging collections and, on upgrade, any retained legacy collection. If Qdrant does not confirm alias activation, the service preserves both collections and continues using its previous in-memory snapshot. Restart to read the persisted alias. An abruptly interrupted or unconfirmed replacement can leave an inactive collection. Before any manual cleanup, stop the service and check the alias target. The failure test covers interruption after a committed point write in embedded Qdrant. It does not exercise disk corruption or several writers sharing a remote Qdrant namespace.

From the project directory with dependencies installed, run:

```sh
python -m pytest -q
python -m scripts.live_check --reference
python -m scripts.reindex_recovery --output .runtime/recovery-after.json
```

For the baseline comparison, pass `--project /path/to/0.4.0-checkout --expect-legacy` and a separate output path. Both runs use fresh databases, retain service logs alongside the JSON, and leave the application's existing Qdrant data untouched.
