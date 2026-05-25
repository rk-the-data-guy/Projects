"""
rag_engine.py
Context assembly + Gemini 2.5 Flash answer generation with structured output.

Given a query and retrieved chunks (from retriever.py), this module:
  1. Assembles chunks into a numbered context block with citation markers
  2. Calls Gemini 2.5 Flash with a structured output schema (RAGAnswer)
  3. Returns the typed response including thin-context detection

Thin context is flagged when:
  - Fewer than 2 chunks are retrieved
  - The top chunk's relevance score is below 0.4
  - The LLM itself signals it cannot answer from the provided context
"""

import os
import logging
from typing import List, Dict, Any, Tuple
from openai import AsyncOpenAI
from dotenv import load_dotenv

from .models import RAGAnswer, Citation

load_dotenv()
logger = logging.getLogger(__name__)

API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not set")

_client = AsyncOpenAI(api_key=API_KEY)

SYSTEM_PROMPT = """\
You are a Marketing Insights Assistant for a brand strategy team.
Your job is to answer questions about campaigns, customer research, win/loss analysis, and marketing strategy using ONLY the provided context documents.

Rules:
1. Base your answer SOLELY on the provided context. Do not use outside knowledge.
2. Cite every factual claim using the citation number [N] that corresponds to the context document.
3. If the context does not contain enough information to fully answer the question, set thin_context to true and explain in thin_context_reason what additional documents or data would help.
4. Never invent statistics, campaign names, dates, or outcomes that are not in the context.
5. Set confidence between 0 and 100 based on how well the context supports your answer:
   - 80-100: Context directly and completely answers the question
   - 50-79: Context partially answers; some inference required
   - 20-49: Context is tangentially relevant; answer is speculative
   - 0-19: Context does not support the answer; thin_context should be true

Be direct and specific. Brand managers are preparing for planning meetings — they need actionable insights, not hedging.
"""

def _build_context_block(chunks: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """
    Build a numbered context block from retrieved chunks.
    Returns: (context_string, ordered_chunks_with_citation_numbers)
    """
    context_parts = []
    for i, chunk in enumerate(chunks, start=1):
        header = f"[{i}] {chunk['title']} ({chunk['doc_type'].replace('_', ' ').title()}, {chunk.get('date', 'n/d')})"
        context_parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(context_parts), chunks


def _detect_thin_context(chunks: List[Dict[str, Any]]) -> Tuple[bool, str]:
    """Pre-check if context is obviously thin before calling the LLM."""
    if not chunks:
        return True, "No relevant documents were found in the knowledge base for this query."
    if len(chunks) < 2:
        return True, "Only one document was retrieved; the answer may be incomplete. Additional campaign briefs, post-mortems, or research documents would improve coverage."
    top_score = chunks[0].get("_relevance_score", 1.0)
    if top_score < 0.35:
        return True, f"Retrieved documents have low relevance scores (top score: {top_score:.2f}). This topic may not be well-covered in the current knowledge base."
    return False, ""


async def generate_answer(
    query: str,
    chunks: List[Dict[str, Any]],
) -> Tuple[RAGAnswer, int, int]:
    """
    Generate a grounded answer from retrieved chunks.
    Returns: (RAGAnswer, prompt_tokens, completion_tokens)
    """
    pre_thin, pre_reason = _detect_thin_context(chunks)
    context_block, ordered_chunks = _build_context_block(chunks)

    user_message = f"""Context documents:
{context_block}

---

Question: {query}

Answer the question using only the context documents above. Include citation numbers [N] for every factual claim.
"""

    try:
        response = await _client.beta.chat.completions.parse(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ],
            response_format=RAGAnswer,
            temperature=0.1,
        )

        answer = response.choices[0].message.parsed
        if not answer:
            raise ValueError("Failed to parse OpenAI RAGAnswer response")

        # If pre-check flagged thin context but LLM didn't, trust the pre-check
        if pre_thin and not answer.thin_context:
            answer.thin_context = True
            answer.thin_context_reason = pre_reason

        # Enrich citations with relevance scores from retriever
        enriched_citations = []
        for citation in answer.citations:
            # Find the matching chunk to get its relevance score
            score = 0.7  # default
            for chunk in ordered_chunks:
                if chunk["doc_id"] == citation.doc_id:
                    score = chunk.get("_relevance_score", 0.7)
                    break
            enriched_citations.append(Citation(
                doc_id=citation.doc_id,
                title=citation.title,
                doc_type=citation.doc_type,
                snippet=citation.snippet,
                relevance_score=min(1.0, max(0.0, score)),
            ))
        answer.citations = enriched_citations

        prompt_tokens = response.usage.prompt_tokens if response.usage else 0
        completion_tokens = response.usage.completion_tokens if response.usage else 0

        return answer, prompt_tokens, completion_tokens

    except Exception as e:
        logger.error(f"RAG generation failed: {e}")
        raise
