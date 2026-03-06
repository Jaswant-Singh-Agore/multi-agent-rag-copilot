"""
Manages the FAISS vector index — building, saving, loading, and retrieving.
"""

import logging
import pickle
from pathlib import Path

import faiss
import numpy as np

from config import FAISS_INDEX_PATH, TOP_K_RETRIEVAL


logger = logging.getLogger(__name__)

_INDEX_FILE = "index.bin"
_DOCS_FILE = "documents.pkl"


class FAISSStore:
    """Vector store backed by a FAISS flat inner-product index."""

    def __init__(self, embedder) -> None:
        self.embedder = embedder
        self.dimension: int = embedder.dimension
        self.index: faiss.IndexFlatIP | None = None
        self.documents: list[dict] = []

    def build_index(self, documents: list[dict]) -> None:
        """Embed documents and build a FAISS index from scratch."""
        if not documents:
            raise ValueError("Cannot build index from an empty document list.")

        logger.info("Building FAISS index for %d documents...", len(documents))

        texts = [doc["content"] for doc in documents]
        embeddings = self.embedder.embed_documents(texts).astype("float32")

        self.index = faiss.IndexFlatIP(self.dimension)
        self.index.add(embeddings)
        self.documents = documents

        logger.info("Index built — %d vectors stored.", self.index.ntotal)

    def save(self, index_dir: str | Path = FAISS_INDEX_PATH) -> None:
        """Persist the FAISS index and document store to disk."""
        if self.index is None:
            raise RuntimeError("No index to save. Call build_index() first.")

        index_dir = Path(index_dir)
        index_dir.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(index_dir / _INDEX_FILE))

        with open(index_dir / _DOCS_FILE, "wb") as f:
            pickle.dump(self.documents, f)

        logger.info("Index saved to '%s'.", index_dir)

    def load(self, index_dir: str | Path = FAISS_INDEX_PATH) -> bool:
        """
        Load a previously saved index from disk.
        Returns True on success, False if no saved index is found.
        """
        index_dir = Path(index_dir)
        index_path = index_dir / _INDEX_FILE
        docs_path = index_dir / _DOCS_FILE

        if not index_path.exists():
            logger.warning("No saved index found at '%s'.", index_dir)
            return False

        self.index = faiss.read_index(str(index_path))

        with open(docs_path, "rb") as f:
            self.documents = pickle.load(f)

        logger.info("Index loaded — %d vectors.", self.index.ntotal)
        return True

    def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL) -> list[dict]:
        """Return the top-k most relevant documents for a query."""
        if self.index is None:
            raise RuntimeError("Index not built. Call build_index() or load() first.")

        query_embedding = self.embedder.embed_query(query).astype("float32").reshape(1, -1)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "content": self.documents[idx]["content"],
                "metadata": self.documents[idx]["metadata"],
                "score": float(score)
            })

        return results