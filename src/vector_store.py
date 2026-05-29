"""
ResearchLens — Vector Store (Step 3)
=====================================
ChromaDB-backed vector storage for document chunks and similarity search.

Interview concepts:
  - Vector database: A database optimized for similarity search over high-dimensional vectors.
  - HNSW (Hierarchical Navigable Small World): The ANN algorithm ChromaDB uses internally.
    It builds a graph of vectors for fast approximate nearest-neighbor search.
  - Cosine similarity: Measures angle between vectors. 1 = identical, 0 = unrelated, -1 = opposite.
  - Persistent storage: ChromaDB saves to disk so you don't re-index every time.
"""

import chromadb
from .config import CHROMA_DIR, COLLECTION_NAME, RETRIEVAL_STRATEGY, SENTENCE_WINDOW_SIZE
from .embeddings import embed_texts, embed_query


def get_client() -> chromadb.PersistentClient:
    """Get a persistent ChromaDB client (stores data on disk)."""
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def get_collection(client: chromadb.PersistentClient = None):
    """
    Get or create the research papers collection.

    Uses cosine similarity as the distance metric — this matches
    how our embeddings are designed to be compared.
    """
    if client is None:
        client = get_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # Use cosine similarity
    )


def add_documents(chunks: list[dict]) -> int:
    """
    Add document chunks to the vector store.

    Pipeline:
      chunks → generate embeddings → store in ChromaDB with metadata

    Args:
        chunks: List of chunk dicts with 'text' and 'metadata' keys

    Returns:
        Number of chunks stored
    """
    collection = get_collection()

    texts = [c["text"] for c in chunks]
    metadatas = [c["metadata"] for c in chunks]

    # Create unique IDs for each chunk: "filename_page_suffix"
    # Suffix is sent_index for sentences and child_index for parent-child chunks
    ids = []
    for c in chunks:
        source = c["metadata"]["source"]
        page = c["metadata"]["page"]
        if c["metadata"].get("chunk_type") == "sentence":
            suffix = f"sent_{c['metadata']['sentence_index']}"
        else:
            suffix = f"child_{c['metadata'].get('chunk_index', '0')}"
        ids.append(f"{source}_{page}_{suffix}")

    # Generate embeddings using Gemini API
    print(f"[EMBED] Generating embeddings for {len(texts)} chunks...")
    embeddings = embed_texts(texts)

    # ChromaDB has a batch limit (~5000), add in batches to be safe
    batch_size = 500
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            embeddings=embeddings[i:end],
            metadatas=metadatas[i:end],
        )
        print(f"  [STORE] Stored batch {i // batch_size + 1} ({end}/{len(texts)} chunks)")

    print(f"[DONE] {len(texts)} chunks stored in vector database")
    return len(texts)


def search(
    query: str,
    top_k: int = 5,
    papers_filter: list[str] = None,
    strategy: str = None,
    window_size: int = None
) -> list[dict]:
    """
    Search for the most relevant chunks given a question using the active strategy.

    Args:
      query: The user's question
      top_k: Number of most-relevant chunks to return
      papers_filter: List of source PDF filenames to restrict the search to
      strategy: "parent_document" or "sentence_window" (defaults to config)
      window_size: Sentence window size for sentence_window strategy (defaults to config)

    Returns:
        List of dicts with keys: text, metadata, score
        Sorted by relevance (highest score first)
    """
    if strategy is None:
        strategy = RETRIEVAL_STRATEGY
    if window_size is None:
        window_size = SENTENCE_WINDOW_SIZE

    collection = get_collection()

    # Check if collection has any documents
    if collection.count() == 0:
        return []

    # Embed the query (using RETRIEVAL_QUERY task type)
    query_embedding = embed_query(query)

    # Determine chunk type filter
    chunk_type_filter = "child" if strategy == "parent_document" else "sentence"

    # Construct composite filter_dict using $and
    conditions = [{"chunk_type": chunk_type_filter}]
    if papers_filter:
        if len(papers_filter) == 1:
            conditions.append({"source": papers_filter[0]})
        else:
            conditions.append({"source": {"$in": papers_filter}})
            
    filter_dict = conditions[0] if len(conditions) == 1 else {"$and": conditions}

    # 1. Retrieve Dense (Vector) Results
    # Fetch top_k * 2 to give reranker more options to optimize
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k * 2, collection.count()),
        where=filter_dict,
        include=["documents", "metadatas", "distances"],
    )

    # 2. Retrieve Sparse (Keyword) Results
    # Extract keywords (lowercase words > 3 chars, removing common stopwords)
    stop_words = {"what", "when", "how", "why", "who", "with", "from", "their", "them", "then", "there", "they", "this", "that", "these", "those"}
    words = [w.strip("?,.!-()\"'").lower() for w in query.split()]
    keywords = [w for w in words if len(w) > 3 and w not in stop_words]

    sparse_results = []
    if keywords:
        for kw in keywords[:3]:  # Search up to top 3 keywords to prevent rate limits
            try:
                kw_res = collection.get(
                    where_document={"$contains": kw},
                    where=filter_dict,
                    limit=10,
                    include=["documents", "metadatas"]
                )
                if kw_res and kw_res["documents"]:
                    for doc, meta in zip(kw_res["documents"], kw_res["metadatas"]):
                        sparse_results.append((doc, meta))
            except Exception as e:
                print(f"[WARNING] Keyword search for '{kw}' failed: {e}")

    # 3. Reciprocal Rank Fusion (RRF) & Re-scoring
    rrf_scores = {}
    doc_lookup = {}

    def get_doc_id(meta: dict) -> str:
        source = meta["source"]
        page = meta["page"]
        if meta.get("chunk_type") == "sentence":
            suffix = f"sent_{meta['sentence_index']}"
        else:
            suffix = f"child_{meta.get('chunk_index', '0')}"
        return f"{source}_{page}_{suffix}"

    def reconstruct_context(doc: str, meta: dict) -> str:
        if strategy == "sentence_window":
            # Reconstruct sentence window using neighboring lookup
            source = meta["source"]
            page = meta["page"]
            target_idx = int(meta["sentence_index"])
            min_idx = max(0, target_idx - window_size)
            max_idx = target_idx + window_size
            
            try:
                window_res = collection.get(
                    where={
                        "$and": [
                            {"source": source},
                            {"page": page},
                            {"chunk_type": "sentence"},
                            {"sentence_index": {"$gte": min_idx}},
                            {"sentence_index": {"$lte": max_idx}}
                        ]
                    },
                    include=["documents", "metadatas"]
                )
                if window_res and window_res["documents"]:
                    # Sort sentences by their original sequence index
                    sorted_sents = sorted(
                        zip(window_res["documents"], window_res["metadatas"]),
                        key=lambda x: int(x[1]["sentence_index"])
                    )
                    return " ".join([s[0] for s in sorted_sents])
            except Exception as e:
                print(f"[WARNING] Sentence window reconstruction failed: {e}")
                return doc
        else:
            # Fall back to parent-chunk retrieval
            return meta.get("parent_text", doc)

    # Score Dense Results
    if dense_results and dense_results["documents"] and dense_results["documents"][0]:
        for rank, (doc, meta, dist) in enumerate(zip(dense_results["documents"][0], dense_results["metadatas"][0], dense_results["distances"][0])):
            doc_id = get_doc_id(meta)
            doc_lookup[doc_id] = {
                "text": reconstruct_context(doc, meta),
                "metadata": meta,
                "score": round(1 - dist, 4)
            }
            # RRF Formula: 1 / (60 + rank)
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))

    # Score Sparse Results
    seen_sparse_ids = set()
    for rank, (doc, meta) in enumerate(sparse_results):
        doc_id = get_doc_id(meta)
        if doc_id not in seen_sparse_ids:
            seen_sparse_ids.add(doc_id)
            if doc_id not in doc_lookup:
                # We give keyword-only matches a base semantic relevance score of 0.6
                doc_lookup[doc_id] = {
                    "text": reconstruct_context(doc, meta),
                    "metadata": meta,
                    "score": 0.6000
                }
            # Add sparse RRF weight
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + (1.0 / (60.0 + rank))

    # Sort all documents by their fused RRF score descending
    sorted_ids = sorted(rrf_scores.keys(), key=lambda k: rrf_scores[k], reverse=True)
    
    formatted = []
    for doc_id in sorted_ids:
        formatted.append(doc_lookup[doc_id])

    return formatted



def get_stats() -> dict:
    """
    Get statistics about the vector store.

    Returns:
        Dict with total_chunks, total_papers, and list of paper names
    """
    collection = get_collection()
    count = collection.count()

    if count > 0:
        # Fetch all metadata to get unique paper names
        all_data = collection.get(include=["metadatas"])
        sources = sorted(set(m["source"] for m in all_data["metadatas"]))
    else:
        sources = []

    return {
        "total_chunks": count,
        "total_papers": len(sources),
        "papers": sources,
    }


def clear_collection():
    """Delete all data and recreate an empty collection."""
    client = get_client()
    try:
        client.delete_collection(COLLECTION_NAME)
        print("[CLEAR] Collection cleared")
    except Exception:
        pass
    return get_collection(client)
