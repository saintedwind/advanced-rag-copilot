"""Named public corpora with byte-verified upstream documentation provenance."""

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def verify_curl():
    manifest = json.loads(
        (ROOT / "data/curl_manifest.json").read_text(encoding="utf-8")
    )
    directory = ROOT / "data/curl_docs"
    if sorted(p.name for p in directory.glob("*.md")) != sorted(
        s["file"] for s in manifest["sources"]
    ):
        raise ValueError("Reference corpus inventory differs from source manifest")
    for source in manifest["sources"]:
        path = directory / source["file"]
        if (
            not path.resolve().is_relative_to(directory.resolve())
            or hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]
        ):
            raise ValueError("Reference document checksum mismatch")
    if (
        hashlib.sha256((ROOT / "data/curl-COPYING.txt").read_bytes()).hexdigest()
        != manifest["license_sha256"]
    ):
        raise ValueError("Reference license checksum mismatch")
    return directory, manifest
