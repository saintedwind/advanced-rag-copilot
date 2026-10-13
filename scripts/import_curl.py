"""Freeze exact upstream documentation bytes at a pinned commit, retaining its license."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[1]
REVISION = "2b348649c68c1cd8cc1d055487f2fb2b59a353cb"
FILES = [
    "retry",
    "retry-delay",
    "connect-timeout",
    "max-time",
    "fail",
    "location",
    "cacert",
    "head",
]


def main():
    directory = ROOT / "data/curl_docs"
    directory.mkdir(exist_ok=True)
    sources = []
    with httpx.Client(timeout=40, follow_redirects=True) as client:
        for name in FILES:
            url = f"https://raw.githubusercontent.com/curl/curl/{REVISION}/docs/cmdline-opts/{name}.md"
            response = client.get(url)
            response.raise_for_status()
            raw = response.content
            if not 100 <= len(raw) <= 100_000:
                raise ValueError("Unexpected upstream file size")
            raw.decode("utf-8")
            (directory / f"{name}.md").write_bytes(raw)
            sources.append(
                {
                    "file": f"{name}.md",
                    "url": url,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        license_url = f"https://raw.githubusercontent.com/curl/curl/{REVISION}/COPYING"
        license_response = client.get(license_url)
        license_response.raise_for_status()
        (ROOT / "data/curl-COPYING.txt").write_bytes(license_response.content)
    manifest = {
        "publisher": "curl project",
        "revision": REVISION,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
        "license": "curl license; original notice retained in curl-COPYING.txt",
        "license_url": license_url,
        "license_sha256": hashlib.sha256(license_response.content).hexdigest(),
        "sources": sources,
        "transformations": "none; exact UTF-8 Markdown source bytes",
        "scope": "Real open-source command documentation; not private customer documents or user query logs",
    }
    (ROOT / "data/curl_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Imported {len(sources)} attributed documents at {REVISION}")


if __name__ == "__main__":
    main()
