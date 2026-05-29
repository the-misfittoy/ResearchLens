"""
ResearchLens — RAG Chain (Step 5)
==================================
The orchestrator that connects all components into a single pipeline.

This is the file you'd draw on a whiteboard in an interview:
  Question → Embed → Retrieve → Augment → Generate → Cited Answer

Interview concepts:
  - Pipeline architecture: Each step is modular and can be swapped independently.
  - Cross-paper synthesis: Our USP — the LLM can pull from multiple papers in one answer.
  - Source tracking: Every claim is traceable back to a specific paper and page.
"""

from .vector_store import search, get_stats, add_documents, clear_collection, get_collection
from .ingest import load_all_papers, load_single_pdf, extract_text_from_pdf, chunk_documents
from .llm import generate_answer, condense_query, rerank_chunks, route_query
from .config import TOP_K, PAPERS_DIR
from pathlib import Path


def ingest_papers(papers_dir: Path = None) -> dict:
    """
    Full ingestion pipeline: PDFs → chunks → embeddings → vector store.

    Args:
        papers_dir: Directory containing PDF files

    Returns:
        Dict with ingestion stats
    """
    if papers_dir is None:
        papers_dir = PAPERS_DIR

    # Step 1: Load and chunk all papers
    chunks = load_all_papers(papers_dir)

    if not chunks:
        return {"status": "error", "message": "No papers found to ingest"}

    # Step 2 & 3: Embed and store (handled inside add_documents)
    num_stored = add_documents(chunks)

    return {
        "status": "success",
        "chunks_stored": num_stored,
        "stats": get_stats(),
    }


def ingest_single_paper(pdf_path: Path) -> dict:
    """
    Ingest a single PDF file into the vector store.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        Dict with ingestion stats
    """
    chunks = load_single_pdf(pdf_path)

    if not chunks:
        return {"status": "error", "message": f"No text extracted from {pdf_path.name}"}

    num_stored = add_documents(chunks)

    return {
        "status": "success",
        "paper": pdf_path.name,
        "chunks_stored": num_stored,
    }


def query(question: str, chat_history: list = None, top_k: int = None, papers_filter: list = None) -> dict:
    """
    Full RAG pipeline: Question → Cited Answer with Conversational Memory and Filtering.

    The complete flow:
      0. Condense question using chat history if available (Conversational Memory)
      1. Embed the (possibly condensed) question
      2. Search vector store for top-K most relevant chunks (applying metadata filter if active)
      3. Build augmented prompt (question + context chunks)
      4. Send to Gemini LLM for answer generation (with call_with_retry protection)
      5. Return answer with source citations

    Args:
        question: The user's question about research papers
        chat_history: Streamlit session chat history for query condensation
        top_k: Number of chunks to retrieve (defaults to config)
        papers_filter: List of PDF filenames to restrict retrieval to

    Returns:
        Dict with keys:
          - answer: The generated response with citations
          - sources: List of unique source papers and pages used
          - chunks: The raw retrieved chunks (for transparency)
    """
    if top_k is None:
        top_k = TOP_K

    # Step 0: Conversational Memory (Query Condensation)
    search_query = question
    if chat_history and len(chat_history) > 0:
        print("[CHAT MEMORY] Condensing query using history...")
        search_query = condense_query(question, chat_history)
        print(f"   -> Standalone Search Query: '{search_query}'")

    # Step 0.5: Agentic Routing (Phase 4)
    stats = get_stats()
    route = route_query(search_query, stats["papers"])
    print(f"[AGENT ROUTER] Selected strategy: '{route['strategy']}' for papers: {route['target_papers']}")

    # Step 1 & 2: Retrieve relevant chunks
    if route["strategy"] == "summary" and route["target_papers"]:
        # Summarization Strategy: Retrieve page 1 (abstracts) for target papers
        print(f"[AGENT ROUTER] Fetching abstract (Page 1) chunks for: {route['target_papers']}")
        retrieved_chunks = []
        collection = get_collection()
        for paper in route["target_papers"]:
            try:
                # Query page 1 (abstract/intro) chunks for this paper
                abstract_res = collection.get(
                    where={"$and": [{"source": paper}, {"page": 1}]},
                    limit=5
                )
                if abstract_res and abstract_res["documents"]:
                    for doc, meta in zip(abstract_res["documents"], abstract_res["metadatas"]):
                        retrieved_chunks.append({"text": doc, "metadata": meta, "score": 1.0000})
            except Exception as e:
                print(f"[WARNING] Failed to fetch abstract chunks for {paper}: {e}")

        # If abstract chunks couldn't be loaded, fall back to standard search
        if not retrieved_chunks:
            retrieved_chunks = search(search_query, top_k=top_k * 2, papers_filter=route["target_papers"])
    else:
        # Standard Search Strategy: Retrieve double chunks for reranking
        retrieved_chunks = search(search_query, top_k=top_k * 2, papers_filter=papers_filter)

    # Phase 4: Semantic Reranking (LLM-based)
    if route["strategy"] == "search" and len(retrieved_chunks) > top_k:
        print(f"[RERANKER] Reranking {len(retrieved_chunks)} chunks down to {top_k} using Gemini...")
        retrieved_chunks = rerank_chunks(search_query, retrieved_chunks, top_k=top_k)
    else:
        # Ensure we only pass top_k to LLM if summarization or small list
        retrieved_chunks = retrieved_chunks[:top_k]

    if not retrieved_chunks:
        return {
            "answer": (
                "No relevant information found in the indexed papers. "
                "Please make sure papers have been ingested first or expand your filtering criteria."
            ),
            "sources": [],
            "chunks": [],
        }

    # Step 3 & 4: Generate answer with context
    # (build_prompt + LLM call happen inside generate_answer())
    answer = generate_answer(question, retrieved_chunks)

    # Step 5: Extract unique sources for citation summary
    sources = []
    seen = set()
    for chunk in retrieved_chunks:
        source_key = (chunk["metadata"]["source"], chunk["metadata"]["page"])
        if source_key not in seen:
            sources.append({
                "paper": chunk["metadata"]["source"].replace(".pdf", ""),
                "page": chunk["metadata"]["page"],
                "relevance": chunk["score"],
            })
            seen.add(source_key)

    return {
        "answer": answer,
        "sources": sources,
        "chunks": retrieved_chunks,
    }
