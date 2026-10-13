"""Bounded text ingestion with stable content-addressed identities."""

import hashlib
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split(source_dir, chunk_size=600, chunk_overlap=100, allowed_root=None):
    if not 100 <= chunk_size <= 4000 or not 0 <= chunk_overlap < chunk_size:
        raise ValueError("Overlap must be less than chunk size")
    root = Path(source_dir).resolve()
    if allowed_root and not root.is_relative_to(Path(allowed_root).resolve()):
        raise ValueError("Source directory is outside the configured document root")
    if not root.is_dir():
        raise ValueError("Document directory does not exist")
    files = sorted(
        p
        for p in root.rglob("*")
        if p.suffix.lower() in {".md", ".txt"} and p.is_file()
    )
    if len(files) > 200:
        raise ValueError("At most 200 source files are allowed")
    docs = []
    total = 0
    for path in files:
        if not path.resolve().is_relative_to(root):
            raise ValueError("Symlink leaves source directory")
        size = path.stat().st_size
        total += size
        if size > 1_000_000 or total > 10_000_000:
            raise ValueError("Document size budget exceeded")
        text = path.read_text(encoding="utf-8")
        if text.strip():
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.relative_to(root).as_posix(),
                        "file_name": path.name,
                        "file_type": path.suffix[1:],
                    },
                )
            )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True
    )
    chunks = splitter.split_documents(docs)
    for chunk in chunks:
        key = (
            chunk.metadata["source"]
            + "|"
            + str(chunk.metadata["start_index"])
            + "|"
            + chunk.page_content
        )
        chunk.metadata["chunk_id"] = hashlib.sha256(key.encode()).hexdigest()
    return chunks, len(docs)
