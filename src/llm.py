"""
ResearchLens — LLM Integration (Step 4)
========================================
Google Gemini API integration for answer generation.

Interview concepts:
  - Prompt engineering: How you structure the prompt drastically affects output quality.
  - System prompt: Instructions that tell the LLM how to behave (persona, rules).
  - Temperature: Controls randomness. 0 = deterministic, 1 = creative. We use 0.3 for factual.
  - Context window: Max tokens the model can process. We keep it efficient with top-K retrieval.
  - Grounding: By providing retrieved chunks as context, we "ground" the LLM in real data,
    reducing hallucination. This is the core value proposition of RAG.
"""

from google import genai
from google.genai import types
from .config import GOOGLE_API_KEY, LLM_MODEL, LLM_TEMPERATURE, LLM_MAX_TOKENS
from .embeddings import call_with_retry

# Initialize the Gemini API client
client = genai.Client(api_key=GOOGLE_API_KEY)

# ── System Prompt ────────────────────────────────────────────────────────
# This is the "personality" and ruleset for ResearchLens.
# Notice: Rule #1 is critical — it prevents hallucination by forcing
# the model to only use provided context.

SYSTEM_PROMPT = """You are ResearchLens, an AI research assistant specializing in analyzing and synthesizing information from academic papers.

Your task is to answer questions using ONLY the provided research paper excerpts as context.

Rules:
1. Base your answer ONLY on the provided context. Do NOT use external knowledge.
2. Cite sources using [Paper Name, Page X] format for every claim.
3. If the context doesn't contain enough information to fully answer, say so honestly and answer what you can.
4. When information comes from multiple papers, SYNTHESIZE and COMPARE their perspectives — this is your key strength.
5. Be precise, clear, and academic in tone.
6. If papers present conflicting information, highlight the differences.
7. Structure longer answers with clear sections or bullet points for readability.
8. Always end with a brief summary if the answer is longer than 3 sentences."""


def build_prompt(question: str, context_chunks: list[dict]) -> str:
    """
    Build the augmented prompt by combining retrieved context with the question.

    This is the "Augmentation" step in RAG — we stuff relevant document
    chunks into the prompt so the LLM can reference them.

    Args:
        question: The user's question
        context_chunks: Retrieved chunks from the vector store

    Returns:
        The complete prompt string
    """
    # Format each chunk with its source information
    context_parts = []
    for i, chunk in enumerate(context_chunks, 1):
        source = chunk["metadata"]["source"].replace(".pdf", "")
        page = chunk["metadata"]["page"]
        score = chunk.get("score", 0)
        context_parts.append(
            f"--- Excerpt {i} [Source: {source}, Page {page}] "
            f"(Relevance: {score:.2f}) ---\n"
            f"{chunk['text']}\n"
        )

    context_str = "\n".join(context_parts)

    prompt = (
        f"Context from research papers:\n\n"
        f"{context_str}\n"
        f"---\n\n"
        f"Question: {question}\n\n"
        f"Provide a comprehensive answer based on the above context. "
        f"Cite sources for each claim using [Paper Name, Page X] format."
    )

    return prompt


def generate_answer(question: str, context_chunks: list[dict]) -> str:
    """
    Generate an answer using Gemini with retrieved context.

    This is the "Generation" step in RAG — the LLM produces an answer
    grounded in the provided document context.

    Args:
        question: The user's question
        context_chunks: Retrieved chunks with text, metadata, and scores

    Returns:
        The generated answer string
    """
    prompt = build_prompt(question, context_chunks)

    response = call_with_retry(
        client.models.generate_content,
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=LLM_TEMPERATURE,
            max_output_tokens=LLM_MAX_TOKENS,
        ),
    )

    return response.text


def condense_query(question: str, chat_history: list[dict]) -> str:
    """
    Condense a chat history and follow-up question into a standalone question.
    Crucial for Phase 1: Conversational Memory.
    """
    if not chat_history:
        return question

    history_parts = []
    # Only use the last 3 turns to prevent prompt bloat and keep context focused
    for turn in chat_history[-3:]:
        history_parts.append(f"User: {turn['question']}\nAssistant: {turn['answer']}")
    history_str = "\n\n".join(history_parts)

    condense_prompt = (
        f"Given the following conversation history and a follow-up question, "
        f"rewrite the follow-up question to be a standalone search query that can be "
        f"used to search a database of academic papers. The query must contain "
        f"all necessary background context from the history. "
        f"Do NOT answer the question, just rewrite it.\n\n"
        f"Conversation History:\n"
        f"{history_str}\n\n"
        f"Follow-up Question: {question}\n\n"
        f"Standalone Query:"
    )

    try:
        response = call_with_retry(
            client.models.generate_content,
            model=LLM_MODEL,
            contents=condense_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,  # Highly focused, deterministic rewriting
                max_output_tokens=256,
            ),
        )
        condensed = response.text.strip()
        # Clean up any formatting quotes the model might add
        if condensed.startswith('"') and condensed.endswith('"'):
            condensed = condensed[1:-1].strip()
        return condensed
    except Exception as e:
        print(f"[WARNING] Query condensation failed: {e}. Using original question.")
        return question


def rerank_chunks(query: str, chunks: list[dict], top_k: int = 5) -> list[dict]:
    """
    Rerank retrieved chunks using Gemini to select the absolute most relevant ones.
    Phase 4: Semantic Reranking (LLM-based).
    """
    if len(chunks) <= top_k:
        return chunks

    # Prepare chunks for LLM scoring
    chunks_str = []
    for idx, chunk in enumerate(chunks):
        chunks_str.append(f"[Excerpt {idx}] (Source: {chunk['metadata']['source']}, Page {chunk['metadata']['page']})\nText: {chunk['text'][:400]}...")
    chunks_input = "\n\n".join(chunks_str)

    rerank_prompt = (
        f"You are a search reranker. Given the following user question and a list of retrieved paper excerpts, "
        f"evaluate the relevance of each excerpt to the question. "
        f"Score each excerpt on a scale from 1 (unrelated) to 10 (perfect answer).\n\n"
        f"Question: {query}\n\n"
        f"Excerpts:\n"
        f"{chunks_input}\n\n"
        f"Response Format: Output a valid JSON list of objects containing 'index' and 'score', sorted by score descending. "
        f"Do NOT output anything else. Just the raw JSON block. Example:\n"
        f"[{{\"index\": 3, \"score\": 9.5}}, {{\"index\": 0, \"score\": 8.0}}]"
    )

    try:
        response = call_with_retry(
            client.models.generate_content,
            model=LLM_MODEL,
            contents=rerank_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=512,
                response_mime_type="application/json",
            ),
        )
        import json
        scores = json.loads(response.text.strip())
        
        # Sort chunks based on LLM scores
        scored_chunks = []
        for item in scores:
            idx = int(item["index"])
            if 0 <= idx < len(chunks):
                chunk = chunks[idx]
                # Update chunk relevance score with LLM reranker score normalized
                chunk["score"] = round(float(item["score"]) / 10.0, 4)
                scored_chunks.append(chunk)
                
        # Fill in any missing chunks in case the LLM missed them
        used_indices = {int(item["index"]) for item in scores if 0 <= int(item["index"]) < len(chunks)}
        for idx, chunk in enumerate(chunks):
            if idx not in used_indices:
                scored_chunks.append(chunk)
                
        return scored_chunks[:top_k]
    except Exception as e:
        print(f"[WARNING] LLM Reranking failed: {e}. Falling back to default ordering.")
        return chunks[:top_k]


def route_query(question: str, available_papers: list[str]) -> dict:
    """
    Agentic Query Router. Analyzes the question and routes it to the optimal strategy.
    Phase 4: Agentic Query Routing.
    """
    if not available_papers:
        return {"strategy": "search", "target_papers": []}

    papers_str = ", ".join(available_papers)
    router_prompt = (
        f"You are a query router for a research paper Q&A assistant. "
        f"Analyze the user question and classify it into one of two strategies:\n"
        f"1. 'summary': Use this if the user wants an overview, abstract, or summary of a specific paper "
        f"or wants to compare/contrast entire papers. Example: 'Summarize BERT' or 'Compare ResNet vs ViT'.\n"
        f"2. 'search': Use this for factual questions about specific details inside the papers. Example: 'What is the learning rate of Adam?' or 'How do skip connections work?'\n\n"
        f"Available papers in database: {papers_str}\n\n"
        f"User Question: {question}\n\n"
        f"Response Format: Output a JSON object containing 'strategy' ('summary' or 'search') and 'target_papers' (a list of matched filenames from the available papers list that the question is about). "
        f"Do NOT output anything else. Example:\n"
        f"{{\"strategy\": \"summary\", \"target_papers\": [\"bert.pdf\"]}}"
    )

    try:
        response = call_with_retry(
            client.models.generate_content,
            model=LLM_MODEL,
            contents=router_prompt,
            config=types.GenerateContentConfig(
                temperature=0.1,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
        import json
        return json.loads(response.text.strip())
    except Exception as e:
        print(f"[WARNING] Agentic Routing failed: {e}. Falling back to default search.")
        return {"strategy": "search", "target_papers": []}
