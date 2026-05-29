"""
ResearchLens — Paper Download Script
======================================
Downloads landmark AI/ML research papers from ArXiv.
These are freely available academic papers.

Usage:
    python download_papers.py
"""

import sys
import io
import urllib.request
import os
from pathlib import Path

# Fix Windows console encoding for emoji/unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Project papers directory
PAPERS_DIR = Path(__file__).parent / "papers"
PAPERS_DIR.mkdir(exist_ok=True)

# ── Landmark AI/ML Papers ───────────────────────────────────────────────
# Each tuple: (filename, arxiv_pdf_url, description)
PAPERS = [
    (
        "attention_is_all_you_need.pdf",
        "https://arxiv.org/pdf/1706.03762",
        "Transformers — the foundation of modern AI (Vaswani et al., 2017)",
    ),
    (
        "bert.pdf",
        "https://arxiv.org/pdf/1810.04805",
        "BERT — Bidirectional pre-training for NLU (Devlin et al., 2018)",
    ),
    (
        "gpt2.pdf",
        "https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf",
        "GPT-2 — Language models as unsupervised multitask learners (Radford et al., 2019)",
    ),
    (
        "resnet.pdf",
        "https://arxiv.org/pdf/1512.03385",
        "ResNet — Deep residual learning (He et al., 2015)",
    ),
    (
        "word2vec.pdf",
        "https://arxiv.org/pdf/1301.3781",
        "Word2Vec — Efficient word representations (Mikolov et al., 2013)",
    ),
    (
        "rag_original.pdf",
        "https://arxiv.org/pdf/2005.11401",
        "RAG — Retrieval-Augmented Generation (Lewis et al., 2020)",
    ),
    (
        "lora.pdf",
        "https://arxiv.org/pdf/2106.09685",
        "LoRA — Low-Rank Adaptation for LLMs (Hu et al., 2021)",
    ),
    (
        "vision_transformer.pdf",
        "https://arxiv.org/pdf/2010.11929",
        "ViT — Vision Transformer (Dosovitskiy et al., 2020)",
    ),
    (
        "dropout.pdf",
        "https://arxiv.org/pdf/1207.0580",
        "Dropout — Regularization technique (Hinton et al., 2012)",
    ),
    (
        "adam_optimizer.pdf",
        "https://arxiv.org/pdf/1412.6980",
        "Adam — Adaptive learning rate optimizer (Kingma & Ba, 2014)",
    ),
]


def download_paper(filename: str, url: str, description: str) -> bool:
    """Download a single paper from the given URL."""
    filepath = PAPERS_DIR / filename

    if filepath.exists():
        print(f"  ✅ Already exists: {filename}")
        return True

    try:
        print(f"  ⬇️  Downloading: {description}")
        # Set a user-agent header so ArXiv doesn't block the request
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "ResearchLens/1.0 (Academic Project)"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read()

        filepath.write_bytes(data)
        size_mb = len(data) / (1024 * 1024)
        print(f"     ✅ Saved: {filename} ({size_mb:.1f} MB)")
        return True

    except Exception as e:
        print(f"     ❌ Failed: {filename} — {e}")
        return False


def main():
    """Download all research papers."""
    print("📚 ResearchLens — Paper Downloader")
    print(f"   Target directory: {PAPERS_DIR.resolve()}\n")

    success = 0
    failed = 0

    for filename, url, description in PAPERS:
        if download_paper(filename, url, description):
            success += 1
        else:
            failed += 1

    print(f"\n{'='*50}")
    print(f"✅ Downloaded: {success}/{len(PAPERS)}")
    if failed:
        print(f"❌ Failed: {failed} (you can manually download these)")
    print(f"\nPapers saved to: {PAPERS_DIR.resolve()}")

    # Calculate total size
    total_size = sum(
        f.stat().st_size for f in PAPERS_DIR.glob("*.pdf")
    )
    print(f"Total size: {total_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
