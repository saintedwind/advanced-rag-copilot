"""Document loading, metadata normalization, and recursive chunking."""
from pathlib import Path
from typing import Iterable
from langchain_core.documents import Document
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_split(source_dir: str, chunk_size: int = 600, chunk_overlap: int = 100) -> tuple[list[Document], int]:
    """Load UTF-8 text/Markdown files and return normalized chunks and source count."""
    root = Path(source_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Document directory not found: {root}")
    files = list(root.glob("**/*.md")) + list(root.glob("**/*.txt"))
    docs = []
    for path in sorted(set(files)):
        text = path.read_text(encoding="utf-8")
        if text.strip():
            docs.append(Document(page_content=text, metadata={"source": str(path), "file_name": path.name, "file_type": path.suffix.lstrip(".")}))
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap, add_start_index=True)
    chunks = splitter.split_documents(docs)
    for index, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = f"{Path(chunk.metadata['source']).stem}-{index}"
    return chunks, len(docs)
