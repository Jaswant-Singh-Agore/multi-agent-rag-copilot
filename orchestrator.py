"""
Builds and runs the LangGraph multi-agent pipeline.
Routing logic decides whether a query needs knowledge graph enrichment.
"""

import logging
from typing import TypedDict

from langgraph.graph import END, StateGraph

from app.agents.retriever import make_retriever_agent
from app.agents.graph_agent import make_graph_agent
from app.agents.validator import make_validator_agent
from pipeline.embedder import Embedder
from pipeline.faiss_store import FAISSStore
from pipeline.graph_builder import GraphBuilder
from app.agents.retriever import ChromaStore


logger = logging.getLogger(__name__)

# keywords that suggest the query needs relationship/graph context
_RELATIONAL_KEYWORDS = [
    "which", "who", "whose",
    "department", "team", "group",
    "reports to", "responsible",
    "related", "connected", "between",
    "relationship", "hierarchy",
    "belongs to", "part of",
]


class AgentState(TypedDict):
    query: str
    faiss_results: list
    chroma_results: list
    merged_results: list
    merged_context: str
    retrieval_scores: list
    graph_context: str
    graph_entities: list
    answer: str
    confidence: str
    is_grounded: bool
    sources: list


def _should_use_graph(state: AgentState) -> str:
    """Route to graph_agent for relational queries, validator_agent otherwise."""
    query = state["query"].lower()
    for keyword in _RELATIONAL_KEYWORDS:
        if keyword in query:
            logger.info("Routing to graph_agent (matched keyword: '%s').", keyword)
            return "graph_agent"
    logger.info("Routing to validator_agent (no relational keywords found).")
    return "validator_agent"


def build_agent_graph(retriever_agent, graph_agent, validator_agent):
    """
    Wire up the LangGraph pipeline.
    retriever always runs first, then conditionally routes to
    graph_agent or directly to validator_agent.
    """
    graph = StateGraph(AgentState)

    graph.add_node("retriever_agent", retriever_agent)
    graph.add_node("graph_agent", graph_agent)
    graph.add_node("validator_agent", validator_agent)

    graph.set_entry_point("retriever_agent")

    graph.add_conditional_edges(
        "retriever_agent",
        _should_use_graph,
        {
            "graph_agent": "graph_agent",
            "validator_agent": "validator_agent",
        },
    )

    graph.add_edge("graph_agent", "validator_agent")
    graph.add_edge("validator_agent", END)

    return graph.compile()


def run_pipeline(
    query: str,
    agent_graph,  # pass the compiled graph directly, not the agents
) -> dict:
    """
    Run the full multi-agent RAG pipeline for a given query.
    Returns the answer, confidence, grounding status, sources, and entities.
    """
    logger.info("Pipeline started: '%s'", query[:60])

    initial_state: AgentState = {
        "query": query,
        "faiss_results": [],
        "chroma_results": [],
        "merged_results": [],
        "merged_context": "",
        "retrieval_scores": [],
        "graph_context": "",
        "graph_entities": [],
        "answer": "",
        "confidence": "",
        "is_grounded": False,
        "sources": [],
    }

    final_state = agent_graph.invoke(initial_state)

    return {
        "answer": final_state["answer"],
        "confidence": final_state["confidence"],
        "is_grounded": final_state["is_grounded"],
        "sources": final_state["sources"],
        "graph_entities": final_state["graph_entities"],
        "retrieval_scores": final_state["retrieval_scores"],
    }