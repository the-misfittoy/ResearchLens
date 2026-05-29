"""
ResearchLens Configuration
==========================
All settings in one place. Each parameter is explained for interview prep.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── Paths ────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
PAPERS_DIR = PROJECT_ROOT / "papers"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# Create directories if they don't exist
PAPERS_DIR.mkdir(exist_ok=True)
CHROMA_DIR.mkdir(exist_ok=True)

# ── API Key ──────────────────────────────────────────────────────────────
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY or GOOGLE_API_KEY == "your_api_key_here":
    print("[WARNING] Set your GOOGLE_API_KEY in the .env file")
    print("   Get a free key at: https://aistudio.google.com/apikey")

# ── Embedding Model ─────────────────────────────────────────────────────
# gemini-embedding-001: Google's embedding model with separate daily quota
# - 3072 dimensions (higher = more semantic detail)
# - Free tier: 1500 requests/min
# - Task types: RETRIEVAL_DOCUMENT (for indexing), RETRIEVAL_QUERY (for searching)
EMBEDDING_MODEL = "gemini-embedding-001"
EMBEDDING_DIMENSION = 3072

# ── LLM (Large Language Model) ──────────────────────────────────────────
# gemini-3.5-flash: Fast, capable, free tier available
# - Temperature 0.3: Mostly deterministic but allows some natural variation
#   (0 = robotic, 1 = creative/unpredictable)
# - Max tokens 2048: Enough for detailed answers with citations
LLM_MODEL = "gemini-3.5-flash"
LLM_TEMPERATURE = 0.3
LLM_MAX_TOKENS = 2048

# ── Chunking Parameters ─────────────────────────────────────────────────
# chunk_size: 1000 chars ≈ 200-250 words ≈ 1 solid paragraph
#   Too small → loses context, too large → dilutes relevance
# chunk_overlap: 200 chars (20%) → ensures no information lost at boundaries
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# ── Retrieval ────────────────────────────────────────────────────────────
# top_k: Number of most-relevant chunks to retrieve
#   5 gives a good balance of coverage vs. noise
TOP_K = 5

# ── ChromaDB ─────────────────────────────────────────────────────────────
COLLECTION_NAME = "research_papers"
