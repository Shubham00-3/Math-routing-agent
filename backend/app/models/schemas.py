from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
from datetime import datetime
import uuid


class GuardrailsResult(BaseModel):
    """Result of guardrails validation"""
    is_valid: bool = Field(..., description="Whether the input/output passes validation")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Confidence score of the validation")
    violations: List[str] = Field(default=[], description="List of validation violations")
    category: str = Field(..., description="Category of the validation result")
    reasoning: str = Field(..., description="Explanation of the validation decision")


class QuestionType(str, Enum):
    """Mathematical subject categories"""
    ALGEBRA = "algebra"
    CALCULUS = "calculus"  
    GEOMETRY = "geometry"
    TRIGONOMETRY = "trigonometry"
    STATISTICS = "statistics"
    LINEAR_ALGEBRA = "linear_algebra"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    OTHER = "other"


class SourceType(str, Enum):
    """Source of solution information"""
    KNOWLEDGE_BASE = "knowledge_base"
    WEB_SEARCH = "web_search"
    HYBRID = "hybrid"
    STANDALONE = "standalone"


class FeedbackType(str, Enum):
    """Types of feedback categories"""
    ACCURACY = "accuracy"
    CLARITY = "clarity"
    COMPLETENESS = "completeness"
    HELPFULNESS = "helpfulness"
    CORRECTNESS = "correctness"


class Step(BaseModel):
    """Individual step in mathematical solution"""
    step_number: int = Field(..., description="Step number in sequence")
    description: str = Field(..., description="Clear description of the step")
    explanation: str = Field(default="", description="Detailed explanation of why this step is needed")
    formula: Optional[str] = Field(None, description="Mathematical formula used in this step")
    visual_aid: Optional[str] = Field(None, description="URL or description of visual aid")


class MathQuestionRequest(BaseModel):
    """Request for solving a mathematical problem"""
    question: str = Field(..., min_length=1, description="The mathematical question to solve")
    subject: Optional[QuestionType] = Field(None, description="Subject category of the question")
    difficulty_level: Optional[int] = Field(None, ge=1, le=10, description="Difficulty level 1-10")
    user_context: Optional[str] = Field(None, description="Additional context from user")
    preferred_methods: Optional[List[str]] = Field(default=[], description="Preferred solution methods")


class SolutionResponse(BaseModel):
    """Response containing mathematical solution"""
    question: str = Field(..., description="Original question")
    solution_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique solution ID")
    steps: List[Step] = Field(..., description="Step-by-step solution")
    final_answer: str = Field(..., description="Final answer to the question")
    confidence_score: float = Field(..., ge=0.0, le=1.0, description="Confidence in solution accuracy")
    source: SourceType = Field(..., description="Source of solution information")
    subject: QuestionType = Field(..., description="Detected subject category")
    difficulty_level: int = Field(..., ge=1, le=10, description="Estimated difficulty level")
    processing_time: float = Field(..., description="Time taken to generate solution in seconds")
    references: Optional[List[str]] = Field(default=[], description="References used")
    created_at: datetime = Field(default_factory=datetime.utcnow, description="Solution creation timestamp")


class FeedbackRequest(BaseModel):
    """Request for submitting feedback on a solution"""
    solution_id: str = Field(..., description="ID of solution being reviewed")
    user_id: Optional[str] = Field(None, description="ID of user providing feedback")
    feedback_type: FeedbackType = Field(..., description="Type of feedback being provided")
    rating: int = Field(..., ge=1, le=5, description="Rating from 1-5")
    correctness: bool = Field(..., description="Is the solution mathematically correct?")
    clarity: int = Field(..., ge=1, le=5, description="How clear is the explanation (1-5)?")
    helpfulness: int = Field(..., ge=1, le=5, description="How helpful was the solution (1-5)?")
    comments: Optional[str] = Field(None, description="Additional feedback comments")
    suggested_improvements: Optional[List[str]] = Field(default=[], description="Specific improvement suggestions")
    improvement_suggestions: Optional[str] = Field(None, description="User improvement suggestions")


class FeedbackResponse(BaseModel):
    """Response after processing feedback"""
    feedback_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique feedback ID")
    processed: bool = Field(..., description="Whether feedback was successfully processed")
    improvements_applied: List[str] = Field(default=[], description="Improvements applied based on feedback")
    next_suggestions: List[str] = Field(default=[], description="Suggestions for future improvements")


class SearchResult(BaseModel):
    """Web search result item"""
    title: str = Field(..., description="Title of the search result")
    url: str = Field(..., description="URL of the source")
    content: str = Field(..., description="Relevant content from the source")
    relevance_score: float = Field(..., ge=0.0, le=1.0, description="Relevance score")


class KnowledgeEntry(BaseModel):
    """Knowledge base entry"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Unique entry ID")
    question: str = Field(..., description="Mathematical question")
    solution: SolutionResponse = Field(..., description="Solution to the question")
    tags: List[str] = Field(default=[], description="Tags for categorization")
    difficulty: int = Field(..., ge=1, le=10, description="Difficulty level")
    subject: QuestionType = Field(..., description="Subject category")
    embedding: Optional[List[float]] = Field(None, description="Vector embedding")
    usage_count: int = Field(default=0, description="Number of times accessed")
    last_accessed: datetime = Field(default_factory=datetime.utcnow, description="Last access timestamp")