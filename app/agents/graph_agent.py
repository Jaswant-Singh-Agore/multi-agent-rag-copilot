"""
LangGraph node that enriches pipeline state with knowledge graph context.
"""

import logging

import spacy


logger = logging.getLogger(__name__)

# reuse the same model instance loaded in graph_builder
_nlp = spacy.load("en_core_web_sm")


def extract_entities(query: str) -> list[str]:
    """
    Extract named entities from a query string.
    Falls back to noun chunks if NER finds nothing.
    """
    doc = _nlp(query)

    entities = [ent.text for ent in doc.ents]
    if not entities:
        entities = [chunk.text for chunk in doc.noun_chunks][:3]

    return entities


def make_graph_agent(graph_builder):
    """
    Factory that returns a LangGraph-compatible graph agent node.
    Accepts a GraphBuilder instance so the node stays testable.
    """
    def graph_agent(state: dict) -> dict:
        query = state["query"]
        logger.info("Graph Agent: analyzing '%s'", query[:60])

        graph_context = graph_builder.get_graph_context(query)
        graph_entities = extract_entities(query)

        if graph_context:
            facts_count = graph_context.count("\n")
            logger.info("  Graph found %d relationship fact(s).", facts_count)
        else:
            logger.info("  Graph found no relevant relationships — falling back to FAISS context.")

        return {
            **state,
            "graph_context": graph_context,
            "graph_entities": graph_entities,
        }

    return graph_agent