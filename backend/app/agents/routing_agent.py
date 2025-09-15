import logging
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum
import uuid
from datetime import datetime

from app.services.knowledge_base import KnowledgeBaseService
from app.services.web_search import MCPWebSearchService
from app.services.llm_service import MathLLMService
from app.models.schemas import (
    MathQuestionRequest, SolutionResponse, QuestionType, 
    SourceType, SearchResult, KnowledgeEntry, Step
)

logger = logging.getLogger(__name__)

class RoutingDecision(Enum):
    """Routing decision types"""
    KNOWLEDGE_BASE_ONLY = "knowledge_base_only"
    WEB_SEARCH_ONLY = "web_search_only"
    HYBRID = "hybrid"
    STANDALONE = "standalone"

class MathRoutingAgent:
    """Main routing agent for mathematical question processing"""
    
    def __init__(self):
        self.kb_service = KnowledgeBaseService()
        self.web_search_service = MCPWebSearchService()
        self.llm_service = MathLLMService()
        
        # Routing thresholds
        self.kb_similarity_threshold = 0.8
        self.web_search_threshold = 0.6
        self.hybrid_threshold = 0.7
    
    async def process_question(self, request: MathQuestionRequest) -> SolutionResponse:
        """Main entry point for processing mathematical questions"""
        try:
            logger.info(f"Processing question: {request.question[:100]}...")
            
            # Step 1: Make routing decision
            routing_decision, context_data = await self._make_routing_decision(request)
            logger.info(f"Routing decision: {routing_decision.value}")
            
            # Step 2: Generate solution based on routing decision
            solution = await self._generate_solution(request, routing_decision, context_data)
            
            # Step 3: Post-process solution
            final_solution = await self._post_process_solution(solution, context_data)
            
            logger.info(f"Solution generated with confidence: {final_solution.confidence_score}")
            return final_solution
            
        except Exception as e:
            logger.error(f"Error processing question: {str(e)}")
            return await self._create_fallback_solution(request, str(e))

    async def _make_routing_decision(
        self, 
        request: MathQuestionRequest
    ) -> Tuple[RoutingDecision, Dict[str, Any]]:
        """Determine the best routing strategy for the question"""
        context_data = {}
        
        # Search knowledge base
        kb_results = await self.kb_service.search_similar_questions(
            question=request.question,
            subject=request.subject,
            difficulty_range=(
                max(1, (request.difficulty_level or 5) - 2),
                min(10, (request.difficulty_level or 5) + 2)
            ) if request.difficulty_level else None,
            limit=3,
            score_threshold=0.5
        )
        
        context_data["knowledge_base_results"] = kb_results
        
        # Determine knowledge base confidence
        kb_confidence = 0.0
        if kb_results:
            # Take the highest similarity score
            kb_confidence = max(score for _, score in kb_results)
            context_data["knowledge_base"] = {
                "question": kb_results[0][0].question,
                "steps": [step.dict() for step in kb_results[0][0].solution.steps],
                "final_answer": kb_results[0][0].solution.final_answer,
                "confidence_score": kb_results[0][0].solution.confidence_score
            }
        
        # Make routing decision based on knowledge base confidence
        if kb_confidence >= self.kb_similarity_threshold:
            # High confidence in knowledge base - use it primarily
            return RoutingDecision.KNOWLEDGE_BASE_ONLY, context_data
        
        elif kb_confidence >= self.hybrid_threshold:
            # Medium confidence - supplement with web search
            web_results = await self._search_web(request)
            context_data["search_results"] = web_results
            return RoutingDecision.HYBRID, context_data
        
        else:
            # Low/No confidence in knowledge base - try web search
            web_results = await self._search_web(request)
            context_data["search_results"] = web_results
            
            if web_results and any(r.relevance_score >= self.web_search_threshold for r in web_results):
                return RoutingDecision.WEB_SEARCH_ONLY, context_data
            else:
                # No good sources - generate standalone solution
                return RoutingDecision.STANDALONE, context_data

    async def _search_web(self, request: MathQuestionRequest) -> List[SearchResult]:
        """Search web for mathematical content"""
        try:
            return await self.web_search_service.search_math_problems(
                query=request.question,
                subject=request.subject.value if request.subject else None,
                difficulty=self._map_difficulty_to_text(request.difficulty_level),
                max_results=5
            )
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []

    def _map_difficulty_to_text(self, difficulty_level: Optional[int]) -> Optional[str]:
        """Map numerical difficulty to text description"""
        if not difficulty_level:
            return None
        
        if difficulty_level <= 3:
            return "easy"
        elif difficulty_level <= 7:
            return "medium"
        else:
            return "hard"

    async def _generate_solution(
        self,
        request: MathQuestionRequest,
        routing_decision: RoutingDecision,
        context_data: Dict[str, Any]
    ) -> SolutionResponse:
        """Generate solution based on routing decision"""
        
        # Map routing decision to source type
        source_type_mapping = {
            RoutingDecision.KNOWLEDGE_BASE_ONLY: SourceType.KNOWLEDGE_BASE,
            RoutingDecision.WEB_SEARCH_ONLY: SourceType.WEB_SEARCH,
            RoutingDecision.HYBRID: SourceType.HYBRID,
            RoutingDecision.STANDALONE: SourceType.WEB_SEARCH  # Fallback
        }
        
        source_type = source_type_mapping[routing_decision]
        
        # Generate solution using LLM service
        solution = await self.llm_service.generate_solution(
            question=request.question,
            context_data=context_data,
            source_type=source_type,
            subject=request.subject
        )
        
        return solution

    async def _post_process_solution(
        self,
        solution: SolutionResponse,
        context_data: Dict[str, Any]
    ) -> SolutionResponse:
        """Post-process solution for quality and consistency"""
        try:
            # Add references based on source type
            if solution.source == SourceType.WEB_SEARCH or solution.source == SourceType.HYBRID:
                search_results = context_data.get("search_results", [])
                solution.references = [result.url for result in search_results[:3]]
            
            # Validate solution consistency
            if solution.source == SourceType.HYBRID:
                # Cross-validate with knowledge base if available
                kb_results = context_data.get("knowledge_base_results", [])
                if kb_results:
                    kb_answer = kb_results[0][0].solution.final_answer
                    if kb_answer.strip() != solution.final_answer.strip():
                        # Note discrepancy in confidence
                        solution.confidence_score *= 0.9
                        logger.warning("Discrepancy between knowledge base and generated solution")
            
            # Store successful solution in knowledge base for future use
            if solution.confidence_score >= 0.8:
                await self._store_solution_in_kb(solution)
            
            return solution
            
        except Exception as e:
            logger.error(f"Error in post-processing: {str(e)}")
            return solution

    async def _store_solution_in_kb(self, solution: SolutionResponse) -> None:
        """Store high-quality solution in knowledge base"""
        try:
            knowledge_entry = KnowledgeEntry(
                id=solution.solution_id,
                question=solution.question,
                solution=solution,
                tags=self._extract_tags_from_solution(solution),
                difficulty=solution.difficulty_level,
                subject=solution.subject,
                embedding=[],  # Will be generated during insertion
                usage_count=1,
                last_accessed=solution.created_at
            )
            
            await self.kb_service.add_knowledge_entry(knowledge_entry)
            logger.info(f"Stored solution in knowledge base: {solution.solution_id}")
            
        except Exception as e:
            logger.warning(f"Failed to store solution in knowledge base: {str(e)}")

    def _extract_tags_from_solution(self, solution: SolutionResponse) -> List[str]:
        """Extract relevant tags from solution"""
        tags = [solution.subject.value]
        
        # Add difficulty tag
        if solution.difficulty_level <= 3:
            tags.append("easy")
        elif solution.difficulty_level <= 7:
            tags.append("medium")
        else:
            tags.append("hard")
        
        # Extract mathematical concepts from steps
        concept_keywords = {
            "equation", "solve", "factor", "derivative", "integral", "matrix",
            "vector", "theorem", "proof", "formula", "calculate", "simplify"
        }
        
        solution_text = " ".join([step.description for step in solution.steps]).lower()
        for keyword in concept_keywords:
            if keyword in solution_text:
                tags.append(keyword)
        
        return list(set(tags))  # Remove duplicates

    async def _create_fallback_solution(
        self,
        request: MathQuestionRequest,
        error_msg: str
    ) -> SolutionResponse:
        """Create fallback solution when main processing fails"""
        return SolutionResponse(
            question=request.question,
            solution_id=str(uuid.uuid4()),
            steps=[
                Step(
                    step_number=1,
                    description="I apologize, but I encountered an error processing your question.",
                    explanation=f"Error details: {error_msg}",
                    formula=None,
                    visual_aid=None
                ),
                Step(
                    step_number=2,
                    description="Please try rephrasing your question or contact support if the issue persists.",
                    explanation="Sometimes breaking down complex questions into smaller parts can help.",
                    formula=None,
                    visual_aid=None
                )
            ],
            final_answer="Unable to generate solution due to system error.",
            confidence_score=0.0,
            source=SourceType.WEB_SEARCH,
            subject=request.subject or QuestionType.ALGEBRA,
            difficulty_level=request.difficulty_level or 5,
            processing_time=0.0,
            created_at=datetime.utcnow()
        )