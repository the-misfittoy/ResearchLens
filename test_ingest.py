import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.vector_store import clear_collection, get_stats
from src.rag_chain import ingest_single_paper

def main():
    print("=== Clearing Chroma DB collection ===")
    clear_collection()
    
    selected_papers = [
        "project description.pdf",
        "attention_is_all_you_need.pdf",
    ]
    
    print("\n=== Starting ingestion for selected papers ===")
    papers_dir = Path("papers")
    
    for filename in selected_papers:
        pdf_path = papers_dir / filename
        if not pdf_path.exists():
            print(f"[WARNING] File not found: {pdf_path}")
            continue
            
        print(f"\nIngesting {filename}...")
        try:
            result = ingest_single_paper(pdf_path)
            if result["status"] == "success":
                print(f"[SUCCESS] Ingested {filename}: {result['chunks_stored']} chunks stored!")
            else:
                print(f"[FAILED] Ingested {filename}: {result.get('message')}")
        except Exception as e:
            print(f"[ERROR] Error ingesting {filename}: {e}")
            
    print("\n=== Ingestion Stats ===")
    try:
        stats = get_stats()
        print(f"Total chunks: {stats['total_chunks']}")
        print(f"Total papers: {stats['total_papers']}")
        print(f"Papers: {', '.join(stats['papers'])}")
    except Exception as e:
        print(f"[ERROR] Could not print stats: {e}")

if __name__ == "__main__":
    main()
