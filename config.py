"""
Centralised configuration loaded from environment variables.
Fails fast if any required variable is missing.
"""

import os
from dotenv import load_dotenv

load_dotenv()


def _require(key: str) -> str:
    """Read a required env variable — raise early if it's missing."""
    val = os.getenv(key)
    if not val:
        raise EnvironmentError(f"Missing required environment variable: {key}")
    return val


# models
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"
LLM_MODEL = "meta-llama/Llama-3.1-8B-Instruct"
HF_API_TOKEN = _require("HF_TOKEN")

# vector stores
FAISS_INDEX_PATH = "data/faiss_index"
CHROMA_DB_PATH = "data/chroma_db"

# neo4j
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = _require("NEO4J_PASSWORD")

# rag parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64
TOP_K_RETRIEVAL = 5
CONFIDENCE_THRESH = 0.75