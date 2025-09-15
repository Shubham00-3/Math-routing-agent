from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

class QuestionType(str, Enum):
    """Types of mathematical questions"""
    ALGEBRA = "algebra"
    CALCULUS = "calculus"
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    STATISTICS = "statistics"
    LINEAR_ALGEBRA = "linear_algebra"
    DISCRETE_MATH = "discrete_math"
    NUMBER_THEORY = "number_theory"

class SourceType(str, Enum):
    """Source of the solution"""
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"

class FeedbackType(str, Enum):
    """Types of feedback"""
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    DIFFICULTY = "difficulty"
    OVERALL = "overall"

# Request/Response Models
class MathQuestionRequest(BaseModel):
    """Math question request schema"""
    question: str = Field(..., min_length=5, max_length=1000)
    subject: Optional[QuestionType] = None
    difficulty_level: Optional[int] = Field(None, ge=1, le=10)
    context: Optional[str] = Field(None, max_length=500)
    
    @validator('question')
    def validate_question(cls, v):
        if not v.strip():
            raise ValueError('Question cannot be empty')
        return v.strip()

class Step(BaseModel):
    """Individual solution step"""
    step_number: int
    description: str
    formula: Optional[str] = None
    explanation: str
    visual_aid: Optional[str] = None  # LaTeX or image URL

class SolutionResponse(BaseModel):
    """Math solution response schema"""
    question: str
    solution_id: str
    steps: List[Step]
    final_answer: str
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    source: SourceType
    subject: QuestionType
    difficulty_level: int
    processing_time: float
    references: Optional[List[str]] = None
    created_at: datetime = Field(default_factory=datetime.utcnow)

class GuardrailsResult(BaseModel):
    """Guardrails validation result"""
    is_valid: bool
    confidence: float
    violations: List[str] = []
    category: str
    reasoning: str

class FeedbackRequest(BaseModel):
    """Human feedback request"""
    solution_id: str
    feedback_type: FeedbackType
    rating: int = Field(..., ge=1, le=5)
    comments: Optional[str] = Field(None, max_length=1000)
    improvement_suggestions: Optional[str] = Field(None, max_length=1000)
    user_id: Optional[str] = None

class FeedbackResponse(BaseModel):
    """Feedback processing response"""
    feedback_id: str
    processed: bool
    improvements_applied: List[str]
    next_suggestions: List[str]
    
# Knowledge Base Models
class KnowledgeEntry(BaseModel):
    """Knowledge base entry"""
    id: str
    question: str
    solution: SolutionResponse
    tags: List[str]
    difficulty: int
    subject: QuestionType
    embedding: List[float]
    usage_count: int = 0
    last_accessed: datetime = Field(default_factory=datetime.utcnow)

class SearchResult(BaseModel):
    """Web search result"""
    title: str
    url: str
    content: str
    relevance_score: float
    source: str
    extracted_at: datetime = Field(default_factory=datetime.utcnow)

# Error Models
class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: str
    code: int
    timestamp: datetime = Field(default_factory=datetime.utcnow)