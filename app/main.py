"""
FastAPI application entry point for the Multi-Agent RAG Copilot.
Handles document ingestion, querying, and index management.
"""

import logging
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

from app.agents.retriever import make_retriever_agent
from app.agents.graph_agent import make_graph_agent
from app.agents.validator import make_validator_agent
from orchestrator import run_pipeline, build_agent_graph

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator

from config import FAISS_INDEX_PATH, CHROMA_DB_PATH
from pipeline.document_processor import process_pdf_folder
from pipeline.faiss_store import FAISSStore
from pipeline.embedder import Embedder
from pipeline.graph_builder import GraphBuilder
from app.agents.retriever import ChromaStore
from orchestrator import run_pipeline


logger = logging.getLogger(__name__)

embedder = Embedder()
faiss_store = FAISSStore(embedder)
chroma_store = ChromaStore(embedder)
graph_builder = GraphBuilder()

retriever_agent = make_retriever_agent(faiss_store, chroma_store)
graph_agent = make_graph_agent(graph_builder)
validator_agent = make_validator_agent()

agent_graph = build_agent_graph(retriever_agent, graph_agent, validator_agent)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting up — attempting to load existing FAISS index...")
    loaded = faiss_store.load()
    if loaded:
        logger.info("FAISS index loaded: %d vectors.", faiss_store.index.ntotal)
    else:
        logger.info("No FAISS index found — upload documents to get started.")
    yield
    logger.info("Shutting down — closing Neo4j connection.")
    graph_builder.close()


app = FastAPI(
    title="Multi-Agent RAG Copilot",
    description="RAG pipeline with LangGraph, FAISS, ChromaDB, and Neo4j.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Question cannot be empty.")
        return v.strip()


class QueryResponse(BaseModel):
    answer: str
    confidence: str
    is_grounded: bool
    sources: list
    graph_entities: list


@app.get("/")
async def root():
    return {
        "status": "running",
        "message": "Multi-Agent RAG Copilot API",
        "endpoints": {
            "upload": "POST /upload",
            "query": "POST /query",
            "health": "GET /health",
            "reset": "DELETE /reset",
            "docs": "GET /docs",
        },
    }


@app.get("/health")
async def health():
    faiss_ready = faiss_store.index is not None
    return {
        "status": "healthy",
        "faiss_ready": faiss_ready,
        "faiss_vectors": faiss_store.index.ntotal if faiss_ready else 0,
        "neo4j_ready": True,
    }


@app.post("/upload")
async def upload_documents(files: list[UploadFile] = File(...)):
    if not files:
        raise HTTPException(status_code=400, detail="No files provided.")

    non_pdfs = [f.filename for f in files if not f.filename.endswith(".pdf")]
    if non_pdfs:
        raise HTTPException(
            status_code=400,
            detail=f"Only PDF files are accepted. Rejected: {', '.join(non_pdfs)}",
        )

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)

        for file in files:
            dest = tmp_path / file.filename
            with open(dest, "wb") as f:
                shutil.copyfileobj(file.file, f)

        logger.info("Saved %d file(s) to temp dir. Processing...", len(files))
        documents = process_pdf_folder(tmp_path)

    if not documents:
        raise HTTPException(
            status_code=422,
            detail="Could not extract text from the uploaded PDFs.",
        )

    # index into all three stores
    chroma_store.reset()
    faiss_store.build_index(documents)
    faiss_store.save()
    chroma_store.add_documents(documents)
    graph_builder.build_graph(documents)

    logger.info("Indexing complete — %d chunk(s) ready.", len(documents))

    return {
        "status": "success",
        "files_uploaded": [f.filename for f in files],
        "total_chunks": len(documents),
        "message": "Documents indexed. Ready to query!",
    }


@app.post("/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    if faiss_store.index is None:
        raise HTTPException(
            status_code=503,
            detail="No documents indexed yet. Please upload PDFs first.",
        )

    try:
        result = run_pipeline(request.question, agent_graph)
        return QueryResponse(
            answer=result["answer"],
            confidence=result["confidence"],
            is_grounded=result["is_grounded"],
            sources=result["sources"],
            graph_entities=result["graph_entities"],
        )
    except Exception as e:
        logger.exception("Pipeline error for query '%s'.", request.question[:60])
        raise HTTPException(status_code=500, detail=f"Pipeline error: {e}")


@app.delete("/reset")
async def reset_knowledge_base():
    graph_builder.clear_graph()
    faiss_store.index = None
    faiss_store.documents = []

    # reset() releases ChromaDB file handles
    chroma_store.reset()

    faiss_path = Path(FAISS_INDEX_PATH)
    if faiss_path.exists():
        shutil.rmtree(faiss_path)
        logger.info("Removed '%s'.", faiss_path)

    return {"status": "reset", "message": "All indexes cleared."}

