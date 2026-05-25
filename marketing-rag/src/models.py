from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class Citation(BaseModel):
    """A single document citation from the retrieved context."""
    doc_id: str
    title: str
    doc_type: str
    snippet: str = Field(description="The relevant excerpt from the document")
    relevance_score: float = Field(ge=0.0, le=1.0)


class RAGAnswer(BaseModel):
    """Structured output from the RAG engine for a user question."""
    answer: str = Field(description="The grounded answer to the user question")
    citations: List[Citation] = Field(description="Source documents that support the answer")
    confidence: int = Field(ge=0, le=100, description="Confidence in the answer from 0 to 100")
    thin_context: bool = Field(description="True if retrieved context is insufficient to answer well")
    thin_context_reason: Optional[str] = Field(
        default=None,
        description="If thin_context is True, explain what additional information would help"
    )


class AskRequest(BaseModel):
    """API request body for the /api/ask endpoint."""
    query: str = Field(min_length=3, max_length=500)


class AskResponse(BaseModel):
    """API response body wrapping the RAGAnswer with latency metadata."""
    answer: RAGAnswer
    retrieval_ms: int
    generation_ms: int
    total_ms: int
    chunks_retrieved: int


# ── Eval schemas ───────────────────────────────────────────────────────────────

class EvalQAPair(BaseModel):
    """A single question-answer pair for evaluation."""
    id: str
    question: str
    ground_truth_doc_ids: List[str]
    reference_answer: str
    difficulty: Literal["easy", "medium", "hard"]
    category: str


class RetrievalEvalResult(BaseModel):
    """Evaluation result for a single retrieval query."""
    qa_id: str
    question: str
    ground_truth_doc_ids: List[str]
    retrieved_doc_ids: List[str]
    recall_at_1: float
    recall_at_3: float
    recall_at_5: float
    recall_at_10: float
    reciprocal_rank: float


class AnswerEvalResult(BaseModel):
    """LLM-as-judge evaluation result for a single answer."""
    qa_id: str
    question: str
    faithfulness_score: float = Field(ge=1.0, le=5.0)
    completeness_score: float = Field(ge=1.0, le=5.0)
    thin_context_correct: bool
    judge_reasoning: str
