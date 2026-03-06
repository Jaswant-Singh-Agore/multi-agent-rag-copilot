"""
Builds and queries a Neo4j knowledge graph from extracted document chunks.
Uses spaCy for entity and relation extraction.
"""

import logging
from typing import Any

import spacy
from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


logger = logging.getLogger(__name__)

# loaded once at module level — spaCy models are expensive to initialise
_nlp = spacy.load("en_core_web_sm")

_ENTITY_LABELS = {"PERSON", "ORG", "GPE", "PRODUCT", "EVENT", "LAW", "DATE", "MONEY", "PERCENT"}


class GraphBuilder:
    """Extracts entities/relations from text and stores them in Neo4j."""

    def __init__(self) -> None:
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        try:
            self.driver.verify_connectivity()
            logger.info("Connected to Neo4j at %s", NEO4J_URI)
        except ServiceUnavailable as e:
            raise ConnectionError(f"Could not connect to Neo4j: {e}") from e

    def close(self) -> None:
        """Close the Neo4j driver connection."""
        self.driver.close()

    # context manager support so callers can use `with GraphBuilder() as gb:`
    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

    def clear_graph(self) -> None:
        """Delete all nodes and relationships from the graph."""
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
        logger.info("Graph cleared.")

    def extract_entities(self, text: str) -> list[dict[str, str]]:
        """Return named entities filtered to the supported label set."""
        doc = _nlp(text)
        return [
            {"text": ent.text.strip(), "label": ent.label_}
            for ent in doc.ents
            if ent.label_ in _ENTITY_LABELS
        ]

    def extract_relations(self, text: str) -> list[dict[str, str]]:
        """
        Extract simple (subject, verb, object) triples using dependency parsing.
        Only considers direct objects, attributes, and prepositional objects.
        """
        doc = _nlp(text)
        relations = []

        for sent in doc.sents:
            for token in sent:
                if token.dep_ in ("nsubj", "nsubjpass") and token.head.pos_ == "VERB":
                    subject = token.text
                    relation = token.head.lemma_.upper()

                    for child in token.head.children:
                        if child.dep_ in ("dobj", "attr", "prep"):
                            relations.append({
                                "subject": subject,
                                "relation": relation,
                                "object": child.text
                            })

        return relations

    def build_graph(self, documents: list[dict]) -> None:
        """
        Build the knowledge graph from a list of document chunks.
        Clears any existing graph data before building.
        """
        if not documents:
            raise ValueError("No documents provided to build graph from.")

        self.clear_graph()

        with self.driver.session() as session:
            for doc in documents:
                text = doc["content"]
                source = doc["metadata"]["source"]
                page = doc["metadata"]["page"]

                for ent in self.extract_entities(text):
                    session.run(
                        """
                        MERGE (e:Entity {name: $name, label: $label})
                        ON CREATE SET e.source = $source, e.page = $page
                        ON MATCH SET e.source = $source
                        """,
                        name=ent["text"],
                        label=ent["label"],
                        source=source,
                        page=page,
                    )

                for rel in self.extract_relations(text):
                    session.run(
                        """
                        MERGE (a:Entity {name: $subject})
                        MERGE (b:Entity {name: $object})
                        MERGE (a)-[r:RELATION {type: $relation}]->(b)
                        ON CREATE SET r.source = $source, r.page = $page
                        """,
                        subject=rel["subject"],
                        object=rel["object"],
                        relation=rel["relation"],
                        source=source,
                        page=page,
                    )

        logger.info("Knowledge graph built from %d document(s).", len(documents))

    def query_graph(self, entity_name: str, depth: int = 2) -> list[dict[str, Any]]:
        """
        Find entities connected to *entity_name* up to *depth* hops away.
        Uses case-insensitive partial matching on the entity name.
        """
        # depth is an integer we control — safe to interpolate here
        query = f"""
            MATCH (a:Entity)
            WHERE toLower(a.name) CONTAINS toLower($name)
            MATCH (a)-[r*1..{depth}]-(b:Entity)
            RETURN a.name AS source,
                   b.name AS target,
                   b.label AS target_type,
                   b.source AS doc_source,
                   b.page AS page
            LIMIT 20
        """
        try:
            with self.driver.session() as session:
                result = session.run(query, name=entity_name)
                return [dict(r) for r in result]
        except Exception as e:
            logger.warning("Graph query failed for entity '%s': %s", entity_name, e)
            return []

    def get_graph_context(self, query: str) -> str:
        """
        Build a plain-text context block from graph facts relevant to the query.
        Falls back to noun chunks if no named entities are found.
        Returns an empty string if nothing relevant is in the graph.
        """
        doc = _nlp(query)

        entities = [ent.text for ent in doc.ents]
        if not entities:
            # noun chunks as a fallback when NER finds nothing
            entities = [chunk.text for chunk in doc.noun_chunks][:3]

        facts = []
        for entity in entities[:3]:
            for r in self.query_graph(entity):
                facts.append(
                    f"{r['source']} → {r['target']} "
                    f"[{r['target_type']}] "
                    f"(from {r['doc_source']}, page {r['page']})"
                )

        if not facts:
            return ""

        return "Knowledge Graph Facts:\n" + "\n".join(facts)