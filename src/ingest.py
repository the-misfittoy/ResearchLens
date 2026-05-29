"""
ResearchLens — PDF Ingestion & Chunking (Step 1)
=================================================
This module handles:
  1. Reading PDF files and extracting text from each page
  2. Splitting text into overlapping chunks for vector storage

Interview concepts:
  - Why chunk? LLMs have limited context windows. We feed relevant chunks, not whole papers.
  - Why overlap? Prevents losing context at chunk boundaries.
  - RecursiveCharacterTextSplitter splits at natural boundaries (paragraphs → sentences → words).
"""

import fitz  # PyMuPDF — imported as 'fitz' for historical reasons
from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from .config import PAPERS_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def extract_text_from_pdf(pdf_path: Path) -> list[dict]:
    """
    Extract text from each page of a PDF file.

    Args:
        pdf_path: Path to the PDF file

    Returns:
        List of dicts with keys: text, source, page
    """
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc, start=1):
        text = page.get_text()
        # Skip empty pages (e.g., blank separators in papers)
        if text.strip():
            pages.append({
                "text": text.strip(),
                "source": pdf_path.name,
                "page": page_num,
            })

    doc.close()
    return pages


def chunk_documents(pages: list[dict]) -> list[dict]:
    """
    Split extracted pages into overlapping chunks with metadata.

    Uses RecursiveCharacterTextSplitter which tries to split at
    natural text boundaries in this priority order:
      1. Double newlines (paragraphs)
      2. Single newlines
      3. Periods followed by space (sentences)
      4. Spaces (words)
      5. Characters (last resort)

    Args:
        pages: List of page dicts from extract_text_from_pdf()

    Returns:
        List of chunk dicts with keys: text, metadata
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,  # Count characters (not tokens)
    )

    chunks = []
    for page in pages:
        splits = splitter.split_text(page["text"])
        for i, split_text in enumerate(splits):
            chunks.append({
                "text": split_text,
                "metadata": {
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": i,
                },
            })

    return chunks


def load_single_pdf(pdf_path: Path) -> list[dict]:
    """Load and chunk a single PDF file."""
    print(f"  [PDF] Processing: {pdf_path.name}")
    pages = extract_text_from_pdf(pdf_path)
    chunks = chunk_documents(pages)
    print(f"     -> {len(pages)} pages, {len(chunks)} chunks")
    return chunks


def load_all_papers(papers_dir: Path = None) -> list[dict]:
    """
    Load and chunk ALL PDFs from the papers directory.

    This is the main entry point for ingestion:
      papers/ folder → read PDFs → extract text → chunk → return

    Args:
        papers_dir: Directory containing PDF files (defaults to config)

    Returns:
        List of all chunks across all papers
    """
    if papers_dir is None:
        papers_dir = PAPERS_DIR

    all_chunks = []
    pdf_files = sorted(papers_dir.glob("*.pdf"))

    if not pdf_files:
        print(f"[WARNING] No PDF files found in {papers_dir}")
        print(f"   Add papers to the 'papers/' folder and try again.")
        return []

    print(f"[INFO] Found {len(pdf_files)} papers. Starting ingestion...\n")

    for pdf_path in pdf_files:
        try:
            chunks = load_single_pdf(pdf_path)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [ERROR] Error processing {pdf_path.name}: {e}")
            continue

    print(f"\n[DONE] Ingestion complete: {len(pdf_files)} papers -> {len(all_chunks)} chunks")
    return all_chunks
