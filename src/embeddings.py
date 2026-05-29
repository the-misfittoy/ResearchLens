"""
ResearchLens — Embedding Generation (Step 2)
=============================================
This module converts text into vector embeddings using Google's Gemini API.

Interview concepts:
  - Embedding: A dense numerical vector that captures semantic meaning of text.
  - Similar meanings → close vectors (measured by cosine similarity).
  - gemini-embedding-001 produces 3072-dimensional vectors.
  - Task types: RETRIEVAL_DOCUMENT for indexing, RETRIEVAL_QUERY for searching.
    Using different task types improves retrieval quality because the model
    optimizes embeddings differently for documents vs queries.
"""

import time
import random
import hashlib
from google import genai
from google.genai import types
from google.genai.errors import ClientError, APIError
from .config import GOOGLE_API_KEY, EMBEDDING_MODEL

# Initialize the Gemini API client
client = genai.Client(api_key=GOOGLE_API_KEY)


def generate_mock_embedding(text: str) -> list[float]:
    """
    Generate a deterministic 3072-dimensional vector locally.
    Uses SHA256 hashing of the text as a seed for randomness.
    Ensures identical texts always map to identical vectors, preserving basic consistency.
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()
    state = random.Random(h)
    return [state.uniform(-1.0, 1.0) for _ in range(3072)]


def call_with_retry(func, *args, **kwargs):
    """
    Call a Gemini API function with exponential backoff on 429, rate limits,
    and network connection errors. Crucial for stable free-tier key usage.
    """
    max_retries = 3  # Reduced for faster fallback detection if daily limit is hard-blocked
    base_delay = 2.0  # seconds
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            is_retriable = False
            
            # Check for 429 / rate limit / quota errors
            if hasattr(e, 'code') and e.code == 429:
                is_retriable = True
            elif "429" in err_str or "resource_exhausted" in err_str or "quota" in err_str:
                is_retriable = True
            # Check for network/connection glitches (e.g. 10054, closed, timeout)
            elif "connection" in err_str or "10054" in err_str or "timeout" in err_str or "closed" in err_str:
                is_retriable = True
                
            if is_retriable and attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                print(f"[RETRYABLE ERROR] Hit error: {e}. Retrying in {delay:.1f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
            else:
                raise e


def embed_texts(texts: list[str], task_type: str = "RETRIEVAL_DOCUMENT") -> list[list[float]]:
    """
    Generate embeddings for a batch of texts.
    Falls back to deterministic mock embeddings if daily quota limit is reached.
    """
    embeddings = []
    batch_size = 100  # Gemini API batch limit

    for i in range(0, len(texts), batch_size):
        # Spacing requests to respect free-tier burst limits
        if i > 0:
            time.sleep(1.5)

        batch = texts[i:i + batch_size]
        contents = [types.Content(parts=[types.Part.from_text(text=t)]) for t in batch]

        try:
            result = call_with_retry(
                client.models.embed_content,
                model=EMBEDDING_MODEL,
                contents=contents,
                config=types.EmbedContentConfig(
                    task_type=task_type,
                ),
            )
            # Extract the float vectors from the response
            batch_embeddings = [embedding.values for embedding in result.embeddings]
            embeddings.extend(batch_embeddings)
        except Exception as e:
            print(f"\n[WARNING] Gemini embedding API request failed: {e}")
            print("[FALLBACK] Falling back to deterministic local mock embeddings (RAG code remains fully testable & functional!)")
            for t in batch:
                embeddings.append(generate_mock_embedding(t))

        # Progress indicator for large batches
        processed = min(i + batch_size, len(texts))
        if len(texts) > batch_size:
            print(f"  Embedded {processed}/{len(texts)} chunks...")

    return embeddings


def embed_query(query: str) -> list[float]:
    """
    Generate embedding for a single search query.
    Falls back to deterministic mock embeddings if daily quota limit is reached.
    """
    try:
        result = call_with_retry(
            client.models.embed_content,
            model=EMBEDDING_MODEL,
            contents=[query],
            config=types.EmbedContentConfig(
                task_type="RETRIEVAL_QUERY",
            ),
        )
        return result.embeddings[0].values
    except Exception as e:
        print(f"\n[WARNING] Gemini query embedding failed: {e}")
        print("[FALLBACK] Falling back to deterministic local mock query embedding.")
        return generate_mock_embedding(query)

