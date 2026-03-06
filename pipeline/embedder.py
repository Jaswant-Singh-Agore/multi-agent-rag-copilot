"""
Wraps SentenceTransformer to generate embeddings for documents and queries.
"""

import logging
from typing import Any

import numpy as np
from sentence_transformers import SentenceTransformer

from config import EMBEDDING_MODEL


logger = logging.getLogger(__name__)


class Embedder:
    """Handles embedding generation using a SentenceTransformer model."""

    def __init__(self, model_name: str = EMBEDDING_MODEL) -> None:
        logger.info("Loading embedding model: %s", model_name)
        self.model = SentenceTransformer(model_name)
        self.dimension: int = self.model.get_sentence_embedding_dimension()
        logger.info("Model ready — embedding dimension: %d", self.dimension)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        """Encode a list of document chunks into normalized embeddings."""
        if not texts:
            raise ValueError("Cannot embed an empty list of documents.")

        return self.model.encode(
            texts,
            batch_size=32,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    def embed_query(self, query: str) -> np.ndarray:
        """Encode a single query string into a normalized embedding."""
        if not query.strip():
            raise ValueError("Query string cannot be empty.")

        embedding = self.model.encode(
            [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return embedding[0]