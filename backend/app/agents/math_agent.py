# backend/app/agents/math_agent.py
import logging
import uuid
from typing import Dict, Any
from datetime import datetime

from app.agents.routing_agent import MathRoutingAgent
from app.agents.guardrails_agent import GuardrailsAgent
from app.agents.feedback_agent import FeedbackAgent
from app.models.schemas import (
    MathQuestionRequest, SolutionResponse, 
    FeedbackRequest, FeedbackResponse
)

logger = logging.getLogger(__name__)

class MathAgent:
    """Main orchestrating agent for mathematical problem solving"""
    
    def __init__(self):
        """Initialize all sub-agents"""
        self.routing_agent = MathRoutingAgent()
        self.guardrails_agent = GuardrailsAgent()
        self.feedback_agent = FeedbackAgent()
        logger.info("Math Agent initialized successfully")
    
    async def solve_math_problem(self, request: MathQuestionRequest) -> SolutionResponse:
        """
        Main entry point for solving mathematical problems
        
        Flow:
        1. Input Guardrails Validation
        2. Route to appropriate solution method
        3. Generate solution
        4. Output Guardrails Validation
        5. Return validated solution
        """
        try:
            logger.info(f"Solving math problem: {request.question[:50]}...")
            
            # Step 1: Input Guardrails
            input_validation = await self.guardrails_agent.validate_input(request.question)
            
            if not input_validation.is_valid:
                logger.warning(f"Input validation failed: {input_validation.violations}")
                return self._create_validation_error_response(
                    request, input_validation.reasoning
                )
            
            logger.info("✅ Input validation passed")
            
            # Step 2: Route and Generate Solution
            solution = await self.routing_agent.process_question(request)
            
            # Step 3: Output Guardrails
            solution_text = self._solution_to_text(solution)
            output_validation = await self.guardrails_agent.validate_output(
                request.question, solution_text
            )
            
            if not output_validation.is_valid:
                logger.warning(f"Output validation failed: {output_validation.violations}")
                # Lower confidence but don't reject completely
                solution.confidence_score *= 0.7
            
            logger.info(f"✅ Solution generated successfully: {solution.solution_id}")
            return solution
            
        except Exception as e:
            logger.error(f"Error in math problem solving: {str(e)}")
            return self._create_error_response(request, str(e))
    
    async def process_feedback(self, feedback_request: FeedbackRequest) -> FeedbackResponse:
        """Process human feedback for continuous improvement"""
        try:
            logger.info(f"Processing feedback for solution: {feedback_request.solution_id}")
            return await self.feedback_agent.process_feedback(feedback_request)
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            raise
    
    async def get_analytics(self, days: int = 7) -> Dict[str, Any]:
        """Get system analytics"""
        try:
            return await self.feedback_agent.get_feedback_analytics(time_period=days)
        except Exception as e:
            logger.error(f"Error getting analytics: {str(e)}")
            return {"error": str(e)}
    
    def _solution_to_text(self, solution: SolutionResponse) -> str:
        """Convert solution to text for validation"""
        text_parts = [f"Question: {solution.question}"]
        
        for step in solution.steps:
            text_parts.append(f"Step {step.step_number}: {step.description}")
            if step.explanation:
                text_parts.append(f"Explanation: {step.explanation}")
        
        text_parts.append(f"Final Answer: {solution.final_answer}")
        return "\n".join(text_parts)
    
    def _create_validation_error_response(
        self, 
        request: MathQuestionRequest, 
        reason: str
    ) -> SolutionResponse:
        """Create response for validation errors"""
        from app.models.schemas import Step, SourceType, QuestionType
        
        return SolutionResponse(
            question=request.question,
            solution_id=str(uuid.uuid4()),
            steps=[
                Step(
                    step_number=1,
                    description="Input validation failed",
                    explanation=f"Your question didn't pass our educational content filters: {reason}",
                    formula=None,
                    visual_aid=None
                ),
                Step(
                    step_number=2,
                    description="Please rephrase your question",
                    explanation="Make sure your question is mathematical and educational in nature.",
                    formula=None,
                    visual_aid=None
                )
            ],
            final_answer="Unable to process question due to validation issues.",
            confidence_score=0.0,
            source=SourceType.WEB_SEARCH,
            subject=request.subject or QuestionType.ALGEBRA,
            difficulty_level=request.difficulty_level or 5,
            processing_time=0.1,
            created_at=datetime.utcnow()
        )
    
    def _create_error_response(
        self, 
        request: MathQuestionRequest, 
        error_msg: str
    ) -> SolutionResponse:
        """Create response for system errors"""
        from app.models.schemas import Step, SourceType, QuestionType
        
        return SolutionResponse(
            question=request.question,
            solution_id=str(uuid.uuid4()),
            steps=[
                Step(
                    step_number=1,
                    description="System error occurred",
                    explanation=f"We encountered an error: {error_msg}",
                    formula=None,
                    visual_aid=None
                )
            ],
            final_answer="Unable to generate solution due to system error.",
            confidence_score=0.0,
            source=SourceType.WEB_SEARCH,
            subject=request.subject or QuestionType.ALGEBRA,
            difficulty_level=request.difficulty_level or 5,
            processing_time=0.1,
            created_at=datetime.utcnow()
        )