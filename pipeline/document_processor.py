"""
Handles PDF loading, text extraction, and chunking for the RAG pipeline.
"""

import logging
from pathlib import Path
from typing import TypedDict

import fitz
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP


logger = logging.getLogger(__name__)


class PageRecord(TypedDict):
    content: str
    metadata: dict


class DocumentChunk(TypedDict):
    content: str
    metadata: dict


# reusing the same splitter instead of creating it on every call
_text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n\n", "\n", ". ", " "],
)


def load_pdf(pdf_path: Path) -> list[PageRecord]:
    """Extract text from each page of a PDF. Skips empty pages."""
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"{pdf_path.name!r} is not a PDF file")

    pages: list[PageRecord] = []

    with fitz.open(pdf_path) as doc:
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text").strip()
            if not text:
                continue

            pages.append({
                "content": text,
                "metadata": {
                    "source": pdf_path.name,
                    "page": page_num
                }
            })

    return pages


def split_documents(pages: list[PageRecord]) -> list[DocumentChunk]:
    """Split pages into smaller chunks for embedding."""
    chunks: list[DocumentChunk] = []

    for page in pages:
        split_texts = _text_splitter.split_text(page["content"])
        for chunk_id, text in enumerate(split_texts):
            chunks.append({
                "content": text,
                "metadata": {
                    **page["metadata"],
                    "chunk_id": chunk_id
                }
            })

    return chunks


def process_pdf_folder(pdf_folder: str | Path) -> list[DocumentChunk]:
    """
    Load and chunk all PDFs in a folder.
    Logs a warning and skips any file that fails to process.
    """
    pdf_folder = Path(pdf_folder)

    if not pdf_folder.is_dir():
        raise NotADirectoryError(f"Invalid directory: {pdf_folder}")

    pdf_files = sorted(pdf_folder.glob("*.pdf"))

    if not pdf_files:
        logger.warning("No PDF files found in '%s'.", pdf_folder)
        return []

    all_chunks: list[DocumentChunk] = []

    for pdf_file in pdf_files:
        try:
            pages = load_pdf(pdf_file)
            chunks = split_documents(pages)
            all_chunks.extend(chunks)
            logger.info("Processed '%s' — %d page(s), %d chunk(s)", pdf_file.name, len(pages), len(chunks))
        except Exception as e:
            logger.warning("Skipping '%s': %s", pdf_file.name, e)

    logger.info("Done. %d total chunk(s) ready for embedding.", len(all_chunks))
    return all_chunks