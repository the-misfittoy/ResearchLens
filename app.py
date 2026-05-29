"""
ResearchLens — Streamlit Frontend (Step 6)
===========================================
A premium-looking web UI for the RAG pipeline.

Usage:
    streamlit run app.py
"""

import streamlit as st
from pathlib import Path
import sys
import time

# Add project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

import fitz  # PyMuPDF for PDF page rendering
from src.rag_chain import query, ingest_papers, ingest_single_paper
from src.vector_store import get_stats, clear_collection
from src.config import PAPERS_DIR, TOP_K


# ── PDF Page Renderer (Enhancement: In-App PDF Page Viewer) ──────────────
def get_pdf_page_image(pdf_name: str, page_num: int) -> bytes:
    """Extract a PDF page as PNG bytes using PyMuPDF (fitz)."""
    pdf_path = PAPERS_DIR / pdf_name
    if not pdf_path.exists():
        return None
    try:
        doc = fitz.open(pdf_path)
        if 0 <= page_num - 1 < len(doc):
            page = doc.load_page(page_num - 1)
            pix = page.get_pixmap(dpi=150)
            return pix.tobytes("png")
    except Exception as e:
        print(f"[ERROR] Failed to render PDF page: {e}")
    return None


def select_page(pdf_name: str, page_num: int):
    st.session_state.selected_page = (pdf_name, page_num)


# ── Page Configuration ───────────────────────────────────────────────────
st.set_page_config(
    page_title="ResearchLens — AI Research Paper Q&A",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS for Premium Look ──────────────────────────────────────────
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .main-header h1 {
        color: #ffffff;
        font-size: 2.2rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .main-header p {
        color: rgba(255, 255, 255, 0.65);
        font-size: 1.05rem;
        margin: 0.4rem 0 0 0;
        font-weight: 300;
    }

    /* Source card styling */
    .source-card {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.08), rgba(139, 92, 246, 0.06));
        border: 1px solid rgba(99, 102, 241, 0.2);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin: 0.4rem 0;
        transition: all 0.2s ease;
    }
    .source-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-1px);
    }
    .source-card .paper-name {
        color: #818cf8;
        font-weight: 600;
        font-size: 0.9rem;
    }
    .source-card .page-info {
        color: rgba(255, 255, 255, 0.5);
        font-size: 0.8rem;
    }

    /* Stat cards in sidebar */
    .stat-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.1), rgba(59, 130, 246, 0.08));
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-radius: 12px;
        padding: 1rem;
        text-align: center;
        margin: 0.5rem 0;
    }
    .stat-card .stat-number {
        font-size: 2rem;
        font-weight: 700;
        color: #10b981;
        line-height: 1;
    }
    .stat-card .stat-label {
        color: rgba(255, 255, 255, 0.6);
        font-size: 0.8rem;
        margin-top: 0.3rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Chat message styling */
    .answer-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
        line-height: 1.7;
    }

    /* Chunk preview styling */
    .chunk-preview {
        background: rgba(0, 0, 0, 0.2);
        border-left: 3px solid #6366f1;
        padding: 0.8rem 1rem;
        margin: 0.5rem 0;
        border-radius: 0 8px 8px 0;
        font-size: 0.85rem;
        color: rgba(255, 255, 255, 0.7);
    }

    /* Button styling */
    .stButton > button {
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    /* Hide default Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Animated gradient border for active elements */
    @keyframes gradient-shift {
        0% { border-color: #6366f1; }
        50% { border-color: #8b5cf6; }
        100% { border-color: #6366f1; }
    }
</style>
""", unsafe_allow_html=True)


# ── Session State Initialization ─────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "ingested" not in st.session_state:
    st.session_state.ingested = False
if "selected_page" not in st.session_state:
    st.session_state.selected_page = None


# ── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔬 ResearchLens")
    st.markdown("---")

    # Collection Stats
    stats = get_stats()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_papers']}</div>
            <div class="stat-label">Papers</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="stat-card">
            <div class="stat-number">{stats['total_chunks']}</div>
            <div class="stat-label">Chunks</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Paper Management
    st.markdown("### 📄 Manage Papers")

    # Ingest all papers from folder
    if st.button("📥 Ingest Papers from Folder", use_container_width=True):
        with st.spinner("Ingesting papers..."):
            try:
                result = ingest_papers()
                if result["status"] == "success":
                    st.success(f"✅ Ingested {result['chunks_stored']} chunks!")
                    st.session_state.ingested = True
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                    st.error(
                        "🛑 **Gemini API Daily Quota Exhausted!** \n\n"
                        "You have reached Google's free-tier limit of **1,000 embedding requests per day** on your API key. \n\n"
                        "👉 **What you can do:**\n"
                        "1. Wait 24 hours for Google to reset your daily free quota.\n"
                        "2. Or, use a paid Gemini API key with higher limits.\n\n"
                        "*Note: You can still query any papers already indexed in the database (like Word2Vec)!*"
                    )
                else:
                    st.error(f"❌ Ingestion failed: {e}")

    # Upload individual PDF
    uploaded_file = st.file_uploader(
        "Upload a PDF",
        type=["pdf"],
        help="Upload a research paper to add to the knowledge base",
    )
    if uploaded_file is not None:
        # Save uploaded file
        save_path = PAPERS_DIR / uploaded_file.name
        save_path.write_bytes(uploaded_file.getvalue())

        with st.spinner(f"Ingesting {uploaded_file.name}..."):
            try:
                result = ingest_single_paper(save_path)
                if result["status"] == "success":
                    st.success(f"✅ Added {result['chunks_stored']} chunks from {uploaded_file.name}")
                    st.rerun()
                else:
                    st.error(f"❌ {result['message']}")
            except Exception as e:
                err_str = str(e).lower()
                if "quota" in err_str or "429" in err_str or "resource_exhausted" in err_str:
                    st.error(
                        "🛑 **Gemini API Daily Quota Exhausted!** \n\n"
                        "You have reached Google's free-tier limit of **1,000 embedding requests per day** on your API key. \n\n"
                        "👉 **What you can do:**\n"
                        "1. Wait 24 hours for Google to reset your daily free quota.\n"
                        "2. Or, use a paid Gemini API key with higher limits.\n\n"
                        "*Note: You can still query any papers already indexed in the database (like Word2Vec)!*"
                    )
                else:
                    st.error(f"❌ Ingestion failed: {e}")

    st.markdown("---")

    # Indexed Papers List
    if stats["papers"]:
        st.markdown("### 📚 Indexed Papers")
        for paper in stats["papers"]:
            st.markdown(f"- 📄 {paper.replace('.pdf', '')}")

    st.markdown("---")

    # Settings
    st.markdown("### ⚙️ Settings")

    # Metadata Filter (Phase 2)
    papers_filter = None
    if stats["papers"]:
        selected_papers = st.multiselect(
            "Filter by Papers",
            options=stats["papers"],
            default=stats["papers"],
            help="Select which papers to search inside"
        )
        papers_filter = selected_papers

    top_k = st.slider(
        "Chunks to retrieve (Top-K)",
        min_value=1,
        max_value=15,
        value=TOP_K,
        help="More chunks = broader context but more noise",
    )

    st.markdown("---")
    st.markdown("### 🔍 Retrieval Strategy")
    
    retrieval_mode = st.selectbox(
        "Retrieval Mode",
        options=["Auto-Detect (Recommended)", "Deep Context (Paragraphs)", "Focus Mode (Sentences)"],
        index=0,
        help=(
            "Auto-Detect: Automatically picks the best mode depending on your question.\n\n"
            "Deep Context (Paragraphs): Retrieves complete paragraphs around matching text. Ideal for understanding 'Why' or 'How' concepts.\n\n"
            "Focus Mode (Sentences): Retrieves exact matching sentences and their immediate surrounding lines. Ideal for specific facts, values, or metrics."
        )
    )
    
    if retrieval_mode == "Auto-Detect (Recommended)":
        strategy = "auto"
    elif retrieval_mode == "Deep Context (Paragraphs)":
        strategy = "parent_document"
    else:
        strategy = "sentence_window"
    
    window_size = 2
    if strategy == "sentence_window" or strategy == "auto":
        window_size = st.slider(
            "Context Window Size",
            min_value=1,
            max_value=5,
            value=2,
            help="Number of sentences to pull around the matching text for extra context."
        )

    st.markdown("---")

    # Clear database
    if st.button("🗑️ Clear Database", use_container_width=True):
        clear_collection()
        st.session_state.chat_history = []
        st.success("Database cleared!")
        st.rerun()

    # About
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: rgba(255,255,255,0.4); font-size: 0.75rem;">
            Built with RAG Pipeline<br>
            Gemini + ChromaDB + Streamlit
        </div>
        """,
        unsafe_allow_html=True,
    )


# ── Main Content ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🔬 ResearchLens</h1>
    <p>Ask questions across AI research papers — get cited, synthesized answers</p>
</div>
""", unsafe_allow_html=True)


# Check if papers are indexed
if stats["total_chunks"] == 0:
    st.info(
        "👋 **Welcome to ResearchLens!** \n\n"
        "To get started:\n"
        "1. Place your PDF papers in the `papers/` folder "
        "(or use `python download_papers.py` to get sample papers)\n"
        "2. Click **📥 Ingest Papers from Folder** in the sidebar\n"
        "3. Start asking questions!"
    )

# ── Split Screen Columns Layout (Enhancement: In-App PDF Page Viewer) ────
has_preview = "selected_page" in st.session_state and st.session_state.selected_page is not None

if has_preview:
    col_chat, col_preview = st.columns([1.1, 0.9])
else:
    col_chat = st.container()
    col_preview = None

with col_chat:
    # ── Chat Interface ───────────────────────────────────────────────────
    # Display chat history
    for entry in st.session_state.chat_history:
        # User message
        with st.chat_message("user"):
            st.markdown(entry["question"])

        # Assistant message
        with st.chat_message("assistant", avatar="🔬"):
            st.markdown(entry["answer"])

            # Source citations
            if entry["sources"]:
                with st.expander(f"📎 Sources ({len(entry['sources'])} references)", expanded=False):
                    for idx, src in enumerate(entry["sources"]):
                        st.markdown(
                            f"""<div class="source-card">
                                <span class="paper-name">📄 {src['paper']}</span>
                                <span class="page-info"> · Page {src['page']} · Relevance: {src['relevance']:.1%}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        # Add Page Preview Button
                        pdf_filename = f"{src['paper']}.pdf"
                        st.button(
                            f"🔍 View Page {src['page']}",
                            key=f"btn_hist_{src['paper']}_{src['page']}_{idx}",
                            on_click=select_page,
                            args=(pdf_filename, src["page"]),
                            use_container_width=True
                        )

            # Retrieved chunks (for transparency)
            if entry["chunks"]:
                with st.expander("🔍 Retrieved Chunks (Debug View)", expanded=False):
                    for i, chunk in enumerate(entry["chunks"], 1):
                        source = chunk["metadata"]["source"].replace(".pdf", "")
                        page = chunk["metadata"]["page"]
                        score = chunk.get("score", 0)
                        st.markdown(
                            f"""<div class="chunk-preview">
                                <strong>Chunk {i}</strong> · {source}, Page {page} · Score: {score:.4f}<br>
                                {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}
                            </div>""",
                            unsafe_allow_html=True,
                        )

    # ── Chat Input ───────────────────────────────────────────────────────
    user_question = st.chat_input(
        "Ask a question about your research papers...",
        disabled=(stats["total_chunks"] == 0),
    )

    if user_question:
        # Display user message immediately
        with st.chat_message("user"):
            st.markdown(user_question)

        # Generate response
        with st.chat_message("assistant", avatar="🔬"):
            with st.spinner("🔍 Searching papers and generating answer..."):
                start_time = time.time()
                result = query(
                    user_question,
                    chat_history=st.session_state.chat_history,
                    top_k=top_k,
                    papers_filter=papers_filter,
                    strategy=strategy,
                    window_size=window_size
                )
                elapsed = time.time() - start_time

            # Display answer
            st.markdown(result["answer"])
            st.caption(f"⏱️ Response generated in {elapsed:.1f}s")

            # Source citations
            if result["sources"]:
                with st.expander(f"📎 Sources ({len(result['sources'])} references)", expanded=True):
                    for idx, src in enumerate(result["sources"]):
                        st.markdown(
                            f"""<div class="source-card">
                                <span class="paper-name">📄 {src['paper']}</span>
                                <span class="page-info"> · Page {src['page']} · Relevance: {src['relevance']:.1%}</span>
                            </div>""",
                            unsafe_allow_html=True,
                        )
                        # Add Page Preview Button
                        pdf_filename = f"{src['paper']}.pdf"
                        st.button(
                            f"🔍 View Page {src['page']}",
                            key=f"btn_act_{src['paper']}_{src['page']}_{idx}",
                            on_click=select_page,
                            args=(pdf_filename, src["page"]),
                            use_container_width=True
                        )

            # Retrieved chunks
            if result["chunks"]:
                with st.expander("🔍 Retrieved Chunks (Debug View)", expanded=False):
                    for i, chunk in enumerate(result["chunks"], 1):
                        source = chunk["metadata"]["source"].replace(".pdf", "")
                        page = chunk["metadata"]["page"]
                        score = chunk.get("score", 0)
                        st.markdown(
                            f"""<div class="chunk-preview">
                                <strong>Chunk {i}</strong> · {source}, Page {page} · Score: {score:.4f}<br>
                                {chunk['text'][:300]}{'...' if len(chunk['text']) > 300 else ''}
                            </div>""",
                            unsafe_allow_html=True,
                        )

        # Save to chat history
        st.session_state.chat_history.append({
            "question": user_question,
            "answer": result["answer"],
            "sources": result["sources"],
            "chunks": result["chunks"],
        })
        st.rerun()

# ── Render PDF Page Previewer (Side-by-side) ─────────────────────────────
if has_preview and col_preview is not None:
    with col_preview:
        pdf_name, page_num = st.session_state.selected_page
        st.markdown(f"### 📄 Page Preview: {pdf_name.replace('.pdf', '')}")
        st.caption(f"Page {page_num} · Rendered locally from papers folder")
        
        # Add a Close button
        if st.button("❌ Close Preview", use_container_width=True):
            st.session_state.selected_page = None
            st.rerun()
            
        page_img = get_pdf_page_image(pdf_name, page_num)
        if page_img:
            st.image(page_img, use_container_width=True)
        else:
            st.error("Failed to load PDF page image.")
