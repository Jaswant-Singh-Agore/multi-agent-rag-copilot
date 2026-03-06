"""
LangGraph node that generates grounded answers using an LLM and validates
them against the retrieved context before returning to the pipeline.
"""

import logging

from huggingface_hub import InferenceClient

from config import HF_API_TOKEN, LLM_MODEL, CONFIDENCE_THRESH


logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a precise knowledge base assistant.

Rules you MUST follow:
1. Answer ONLY using the provided Context and Graph Facts.
2. For every factual claim add a citation: [Source: filename, Page X]
3. If the answer is not in context say exactly:
   'I could not find this in the knowledge base.'
4. Never make up facts, numbers, or policies.
5. Be concise — 3 to 5 sentences maximum.
6. If Graph Facts are provided, use them to enrich your answer with relationship context."""

# words that add no signal to grounding checks
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "shall", "can", "need",
    "of", "in", "on", "at", "to", "for", "with", "by", "from",
    "and", "or", "but", "not", "this", "that", "it",
}

_NOT_FOUND_PHRASES = {
    "could not find",
    "not in the knowledge base",
    "no information",
    "i don't know",
    "not mentioned",
}


def _build_combined_context(merged_context: str, graph_context: str) -> str:
    """Combine document context and graph facts into a single context block."""
    parts = []
    if merged_context:
        parts.append(f"=== DOCUMENT CONTEXT ===\n{merged_context}")
    if graph_context:
        parts.append(f"=== KNOWLEDGE GRAPH FACTS ===\n{graph_context}")
    return "\n\n".join(parts) if parts else "No context available."


def _generate_answer(client: InferenceClient, query: str, context: str) -> str:
    """Call the LLM and return a grounded answer. Falls back gracefully on failure."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                f"Context:\n{context}"
                f"\n\nQuestion: {query}"
                f"\n\nAnswer (with citations):"
            ),
        },
    ]

    try:
        response = client.chat_completion(
            model=LLM_MODEL,
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        logger.warning("LLM call failed: %s — using fallback.", e)
        return _fallback_answer(query, context)


def _validate_grounding(answer: str, context: str) -> tuple[bool, str]:
    """
    Check whether the answer is grounded in the retrieved context.
    Uses word overlap ratio after removing stopwords.
    Returns (is_grounded, confidence_level).
    """
    answer_lower = answer.lower()

    if any(phrase in answer_lower for phrase in _NOT_FOUND_PHRASES):
        return False, "NOT_FOUND"

    answer_words = set(answer_lower.split()) - _STOPWORDS
    context_words = set(context.lower().split()) - _STOPWORDS

    if not answer_words:
        return False, "LOW"

    overlap_ratio = len(answer_words & context_words) / len(answer_words)

    if overlap_ratio >= CONFIDENCE_THRESH:
        return True, "HIGH"
    elif overlap_ratio >= 0.4:
        return True, "MEDIUM"
    else:
        return False, "LOW"


def _extract_sources(state: dict) -> list[dict]:
    """Deduplicate and collect source references from merged retrieval results."""
    sources = []
    seen: set[str] = set()

    for result in state.get("merged_results", []):
        source = result["metadata"].get("source", "")
        page = result["metadata"].get("page", "")
        key = f"{source}_p{page}"

        if key not in seen:
            seen.add(key)
            sources.append({
                "source": source,
                "page": page,
                "score": result.get("score", 0),
            })

    return sources


def _fallback_answer(query: str, context: str) -> str:
    """
    Simple keyword-based fallback when the LLM call fails.
    Tries to surface the most relevant context line for the query.
    """
    if not context or context == "No context available.":
        return "I could not find relevant information in the knowledge base for your query."

    query_words = query.lower().split()
    relevant_lines = [
        line for line in context.split("\n")
        if any(word in line.lower() for word in query_words) and len(line.strip()) > 20
    ]

    if relevant_lines:
        return f"Based on available context: {relevant_lines[0][:300]}"

    return "I could not find a specific answer to your query in the knowledge base."


def make_validator_agent(client: InferenceClient | None = None):
    """
    Factory that returns a LangGraph-compatible validator node.
    Accepts an optional InferenceClient for testing — creates one from config if not provided.
    """
    _client = client or InferenceClient(provider="novita", api_key=HF_API_TOKEN)

    def validator_agent(state: dict) -> dict:
        query = state["query"]
        logger.info("Validator Agent: generating answer for '%s'", query[:60])

        context = _build_combined_context(
            state.get("merged_context", ""),
            state.get("graph_context", ""),
        )

        answer = _generate_answer(_client, query, context)
        is_grounded, confidence = _validate_grounding(answer, context)
        sources = _extract_sources(state)

        logger.info("  Answer: %d char(s) | Confidence: %s | Grounded: %s",
                    len(answer), confidence, is_grounded)

        return {
            **state,
            "answer": answer,
            "confidence": confidence,
            "is_grounded": is_grounded,
            "sources": sources,
        }

    return validator_agent