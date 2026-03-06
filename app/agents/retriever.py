"""
Dual retriever that combines FAISS and ChromaDB results for the RAG pipeline.
Exposes a LangGraph-compatible retriever_agent node.
"""

import logging

import chromadb

from config import CHROMA_DB_PATH, TOP_K_RETRIEVAL


logger = logging.getLogger(__name__)


class ChromaStore:
    """Thin wrapper around a ChromaDB persistent collection."""

    def __init__(self, embedder, path: str = CHROMA_DB_PATH) -> None:
        self.embedder = embedder
        self._client = chromadb.PersistentClient(path=path)
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self):
        return self._client.get_or_create_collection(
            name="knowledge_base",
            metadata={"hnsw:space": "cosine"},
        )

    def reset(self) -> None:
        """Drop and recreate the collection."""
        self._client.delete_collection("knowledge_base")
        self.collection = self._get_or_create_collection()
        logger.info("ChromaDB collection reset.")

    def add_documents(self, documents: list[dict]) -> None:
        """Add documents to the collection, skipping any already indexed."""
        if not documents:
            return

        existing_ids = set(self.collection.get()["ids"])
        new_docs = [
            (i, doc) for i, doc in enumerate(documents)
            if f"doc_{i}" not in existing_ids
        ]

        if not new_docs:
            logger.info("ChromaDB already up to date — nothing to add.")
            return

        indices, docs_to_add = zip(*new_docs)
        texts = [doc["content"] for doc in docs_to_add]
        metadatas = [doc["metadata"] for doc in docs_to_add]
        ids = [f"doc_{i}" for i in indices]
        embeddings = self.embedder.embed_documents(texts).tolist()

        self.collection.add(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("ChromaDB indexed %d new document(s).", len(docs_to_add))

    def search(
        self,
        query: str,
        top_k: int = TOP_K_RETRIEVAL,
        filter_source: str | None = None,
    ) -> list[dict]:
        """Query the collection and return results with normalised scores."""
        query_embedding = self.embedder.embed_query(query).tolist()
        where_filter = {"source": filter_source} if filter_source else None

        results = self.collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where_filter,
            include=["documents", "metadatas", "distances"],
        )

        chroma_results = []
        for doc, meta, dist in zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            chroma_results.append({
                "content": doc,
                "metadata": meta,
                "score": round(1 - dist, 4),  # cosine distance → similarity
                "retriever": "chromadb",
            })

        return chroma_results


def merge_results(
    faiss_results: list[dict],
    chroma_results: list[dict],
    top_k: int = TOP_K_RETRIEVAL,
) -> list[dict]:
    """
    Deduplicate and merge results from both retrievers.
    Uses the first 80 chars of content as a deduplication key.
    FAISS results are tagged and processed first.
    """
    seen: set[str] = set()
    merged: list[dict] = []

    for result in faiss_results:
        result["retriever"] = "faiss"
        key = result["content"][:80].strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(result)

    for result in chroma_results:
        key = result["content"][:80].strip().lower()
        if key not in seen:
            seen.add(key)
            merged.append(result)

    merged.sort(key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


def build_context_string(results: list[dict]) -> str:
    """Format retrieved chunks into a readable context block for the LLM."""
    if not results:
        return "No relevant context found."

    parts = []
    for i, result in enumerate(results, start=1):
        source = result["metadata"].get("source", "unknown")
        page = result["metadata"].get("page", "?")
        score = result["score"]
        parts.append(
            f"[Context {i}] Source: {source}, Page: {page}, Relevance: {score:.3f}\n"
            f"{result['content']}"
        )

    return "\n\n---\n\n".join(parts)


def make_retriever_agent(faiss_store, chroma_store: ChromaStore):
    """
    Factory that returns a LangGraph-compatible retriever node.
    Keeps the agent free of global state by closing over the stores.
    """
    def retriever_agent(state: dict) -> dict:
        query = state["query"]
        logger.info("Retriever: searching for '%s'", query[:60])

        faiss_results = faiss_store.retrieve(query, top_k=TOP_K_RETRIEVAL)
        logger.info("  FAISS → %d result(s)", len(faiss_results))

        chroma_results = chroma_store.search(query)
        logger.info("  ChromaDB → %d result(s)", len(chroma_results))

        merged = merge_results(faiss_results, chroma_results)
        logger.info("  Merged → %d unique chunk(s)", len(merged))

        return {
            **state,
            "faiss_results": faiss_results,
            "chroma_results": chroma_results,
            "merged_results": merged,
            "merged_context": build_context_string(merged),
            "retrieval_scores": [r["score"] for r in merged],
        }

    return retriever_agent