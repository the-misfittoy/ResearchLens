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
import re
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


def split_into_sentences(text: str) -> list[str]:
    """
    Split page text into individual sentences.
    Robustly handles common academic abbreviations (e.g., et al., i.e., vs.)
    so sentences are not incorrectly fragmented.
    """
    # Replace multiple spaces and newlines with a single space
    clean_text = re.sub(r"\s+", " ", text).strip()
    
    # List of common academic and general abbreviations
    abbreviations = [
        "e.g.", "i.e.", "et al.", "al.", "vs.", "fig.", "eq.", "vol.", 
        "dept.", "univ.", "dr.", "mr.", "mrs.", "ms."
    ]
    
    # Temporarily hide dots inside abbreviations to prevent false sentence splits
    placeholder_text = clean_text
    for abbr in abbreviations:
        pattern = re.compile(re.escape(abbr), re.IGNORECASE)
        placeholder_abbr = abbr.replace(".", "<DOT>")
        placeholder_text = pattern.sub(placeholder_abbr, placeholder_text)
        
    # Split at period, question mark, or exclamation followed by a space and an uppercase character/number
    # (fixed-width look-behind is fully supported in Python's standard re module)
    sentence_end = r"(?<=\.|\?|!)\s+(?=[A-Z0-9])"
    sentences = re.split(sentence_end, placeholder_text)
    
    # Restore the dots in the final split sentences
    restored_sentences = []
    for s in sentences:
        s_clean = s.replace("<DOT>", ".").strip()
        if s_clean:
            restored_sentences.append(s_clean)
            
    return restored_sentences


def chunk_documents(pages: list[dict]) -> list[dict]:
    """
    Split extracted pages into:
      1. Parent-Child Chunks (for Parent-Document Retrieval)
      2. Individual Sentence Chunks (for Sentence Window Retrieval)
      
    Both chunk styles are written into ChromaDB with a 'chunk_type' key 
    in metadata, allowing search mode toggling at runtime.
    """
    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,          # Parent chunk size (~250 tokens)
        chunk_overlap=200,         # 20% overlap
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,           # Child chunk size (~75 tokens)
        chunk_overlap=50,          # 15-20% overlap
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for page in pages:
        # ─── Strategy 1: Parent-Child Chunking ───
        # 1. Split page into large Parent Chunks
        parent_splits = parent_splitter.split_text(page["text"])
        
        for parent_idx, parent_text in enumerate(parent_splits):
            # 2. Split each Parent Chunk into smaller Child Chunks
            child_splits = child_splitter.split_text(parent_text)
            
            for child_idx, child_text in enumerate(child_splits):
                chunks.append({
                    "text": child_text,  # We embed and search the small child text
                    "metadata": {
                        "chunk_type": "child",
                        "source": page["source"],
                        "page": page["page"],
                        "parent_text": parent_text,  # Store full parent paragraph in metadata
                        "chunk_index": f"{parent_idx}_{child_idx}",
                    },
                })

        # ─── Strategy 2: Sentence Window Chunking ───
        # Split page into clean individual sentences
        page_sentences = split_into_sentences(page["text"])
        
        for idx, sentence_text in enumerate(page_sentences):
            chunks.append({
                "text": sentence_text,  # We embed and search the single sentence
                "metadata": {
                    "chunk_type": "sentence",
                    "source": page["source"],
                    "page": page["page"],
                    "sentence_index": idx,  # Store sequence index for neighboring lookup
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
