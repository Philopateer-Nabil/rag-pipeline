"""Document ingestion supporting PDF, Markdown, and plain text files."""

from __future__ import annotations

from pathlib import Path

import structlog
from langchain_core.documents import Document

logger = structlog.get_logger()

_SUPPORTED_EXTENSIONS = {".pdf", ".md", ".txt"}


def _load_pdf(path: Path) -> list[Document]:
    """Load a PDF file using PyPDF loader."""
    from langchain_community.document_loaders import PyPDFLoader

    loader = PyPDFLoader(str(path))
    return loader.load()


def _load_markdown(path: Path) -> list[Document]:
    """Load a Markdown file as plain text with UTF-8 encoding."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


def _load_text(path: Path) -> list[Document]:
    """Load a plain text file."""
    from langchain_community.document_loaders import TextLoader

    loader = TextLoader(str(path), encoding="utf-8")
    return loader.load()


_LOADER_MAP = {
    ".pdf": _load_pdf,
    ".md": _load_markdown,
    ".txt": _load_text,
}


def load_document(path: str | Path) -> list[Document]:
    """Load a single document file.

    Args:
        path: Path to the document file.

    Returns:
        List of Document objects (a PDF may produce multiple pages).

    Raises:
        ValueError: If the file extension is not supported.
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    ext = path.suffix.lower()
    if ext not in _LOADER_MAP:
        raise ValueError(f"Unsupported file type: {ext}. Supported: {_SUPPORTED_EXTENSIONS}")

    logger.info("loading_document", path=str(path), type=ext)
    docs = _LOADER_MAP[ext](path)

    for doc in docs:
        doc.metadata["source"] = str(path)
        doc.metadata["file_type"] = ext

    logger.info("document_loaded", path=str(path), chunks=len(docs))
    return docs


def load_directory(directory: str | Path) -> list[Document]:
    """Load all supported documents from a directory (recursively).

    Args:
        directory: Path to the directory to scan.

    Returns:
        Combined list of Document objects from all supported files.

    Raises:
        FileNotFoundError: If the directory does not exist.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"Directory not found: {directory}")

    all_docs: list[Document] = []
    files = sorted(f for f in directory.rglob("*") if f.suffix.lower() in _SUPPORTED_EXTENSIONS)

    logger.info("scanning_directory", path=str(directory), files_found=len(files))

    for file_path in files:
        try:
            docs = load_document(file_path)
            all_docs.extend(docs)
        except Exception:
            logger.exception("document_load_failed", path=str(file_path))

    logger.info("directory_loaded", total_documents=len(all_docs))
    return all_docs
