import logging
from typing import List, Dict, Any, Optional
import asyncio
from datetime import datetime, timedelta

from app.services.feedback_service import FeedbackAggregator
from app.services.dspy_optimizer import DSPyFeedbackOptimizer
from app.models.schemas import (
    FeedbackRequest, FeedbackResponse, SolutionResponse,
    FeedbackType, QuestionType
)
from app.models.database import get_db

logger = logging.getLogger(__name__)

class FeedbackAgent:
    """Main agent for handling human feedback and continuous learning"""
    
    def __init__(self):
        self.feedback_aggregator = FeedbackAggregator()
        self.dspy_optimizer = DSPyFeedbackOptimizer()
        self.optimization_schedule = timedelta(hours=24)  # Optimize daily
        self.last_optimization = None
        
    async def process_feedback(
        self, 
        feedback_request: FeedbackRequest
    ) -> FeedbackResponse:
        """Process individual feedback and trigger improvements"""
        try:
            logger.info(f"Processing feedback for solution {feedback_request.solution_id}")
            
            # Get database session
            db = next(get_db())
            
            # Process feedback through aggregator
            response = await self.feedback_aggregator.collect_feedback(
                feedback_request, db
            )
            
            # Check if it's time for optimization
            if await self._should_trigger_optimization():
                asyncio.create_task(self._trigger_optimization())
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return FeedbackResponse(
                feedback_id="error",
                processed=False,
                improvements_applied=[],
                next_suggestions=["Error processing feedback"]
            )

    async def get_feedback_analytics(
        self, 
        solution_id: Optional[str] = None,
        time_period: Optional[int] = 7  # days
    ) -> Dict[str, Any]:
        """Get feedback analytics and insights"""
        try:
            db = next(get_db())
            
            if solution_id:
                # Analytics for specific solution
                analysis = await self.feedback_aggregator._analyze_solution_feedback(
                    solution_id, db
                )
            else:
                # Overall analytics
                analysis = await self._get_overall_analytics(time_period, db)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error getting feedback analytics: {str(e)}")
            return {"error": str(e)}

    async def optimize_model_with_feedback(
        self, 
        force_optimization: bool = False
    ) -> Dict[str, Any]:
        """Trigger DSPy optimization using collected feedback"""
        try:
            if not force_optimization and not await self._should_trigger_optimization():
                return {
                    "status": "skipped",
                    "reason": "Not enough time since last optimization"
                }
            
            # Collect recent feedback data
            feedback_data = await self._collect_optimization_data()
            
            if not feedback_data:
                return {
                    "status": "skipped",
                    "reason": "No feedback data available"
                }
            
            # Run DSPy optimization
            optimization_result = await self.dspy_optimizer.optimize_with_feedback(
                feedback_data
            )
            
            # Update last optimization time
            self.last_optimization = datetime.utcnow()
            
            logger.info(f"Model optimization completed: {optimization_result}")
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in model optimization: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def generate_improved_solution(
        self,
        question: str,
        context: Dict[str, Any],
        feedback_history: Optional[List[FeedbackRequest]] = None
    ) -> SolutionResponse:
        """Generate solution using feedback-optimized model"""
        try:
            # Format feedback history
            feedback_context = ""
            if feedback_history:
                feedback_context = self._format_feedback_context(feedback_history)
            
            # Format context
            context_str = self._format_context_for_dspy(context)
            
            # Generate solution using optimized DSPy model
            solution_data = await self.dspy_optimizer.generate_optimized_solution(
                question=question,
                context=context_str,
                feedback_history=feedback_context,
                subject=context.get("subject", "general")
            )
            
            if "error" in solution_data:
                raise Exception(solution_data["error"])
            
            # Convert to SolutionResponse
            solution = self._convert_to_solution_response(
                question, solution_data, context
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Error generating improved solution: {str(e)}")
            # Fallback to regular generation
            raise

    async def _should_trigger_optimization(self) -> bool:
        """Determine if optimization should be triggered"""
        if not self.last_optimization:
            return True
        
        time_since_last = datetime.utcnow() - self.last_optimization
        return time_since_last >= self.optimization_schedule

    async def _trigger_optimization(self) -> None:
        """Trigger asynchronous optimization"""
        try:
            await self.optimize_model_with_feedback()
        except Exception as e:
            logger.error(f"Error in background optimization: {str(e)}")

    async def _collect_optimization_data(self) -> List[Dict[str, Any]]:
        """Collect feedback data for optimization"""
        try:
            db = next(get_db())
            
            # Get feedback from last week
            cutoff_time = datetime.utcnow() - timedelta(days=7)
            
            # This would be a real database query in production
            # For now, simulate some feedback data
            feedback_data = [
                {
                    "solution": {
                        "question": "Solve x^2 + 5x + 6 = 0",
                        "steps": [
                            {"step_number": 1, "description": "Factor the quadratic"},
                            {"step_number": 2, "description": "Set each factor to zero"}
                        ],
                        "final_answer": "x = -2 or x = -3",
                        "confidence_score": 0.9,
                        "subject": "algebra"
                    },
                    "feedback": {
                        "rating": 4,
                        "feedback_type": "clarity",
                        "comments": "Good solution but could explain factoring better",
                        "improvement_suggestions": "Add more detail in factoring step"
                    },
                    "context": {
                        "knowledge_base": {
                            "question": "Similar quadratic problem",
                            "final_answer": "x = -2 or x = -3"
                        }
                    }
                }
            ]
            
            return feedback_data
            
        except Exception as e:
            logger.error(f"Error collecting optimization data: {str(e)}")
            return []

    async def _get_overall_analytics(
        self, 
        time_period: int, 
        db
    ) -> Dict[str, Any]:
        """Get overall feedback analytics"""
        # This would involve database queries in production
        return {
            "period_days": time_period,
            "total_feedback": 150,
            "average_rating": 3.8,
            "improvement_areas": ["clarity", "step_explanations"],
            "top_subjects": ["algebra", "calculus", "geometry"],
            "optimization_history": self.dspy_optimizer.optimization_history
        }

    def _format_feedback_context(
        self, 
        feedback_history: List[FeedbackRequest]
    ) -> str:
        """Format feedback history for DSPy context"""
        if not feedback_history:
            return ""
        
        context_parts = []
        for feedback in feedback_history[-3:]:  # Last 3 feedback items
            context_parts.append(
                f"Previous feedback: {feedback.feedback_type.value} "
                f"rated {feedback.rating}/5. Comments: {feedback.comments or 'None'}"
            )
        
        return "\n".join(context_parts)

    def _format_context_for_dspy(self, context: Dict[str, Any]) -> str:
        """Format context data for DSPy processing"""
        context_parts = []
        
        if "knowledge_base" in context:
            kb_data = context["knowledge_base"]
            context_parts.append(f"Knowledge Base Solution: {kb_data}")
        
        if "search_results" in context:
            search_results = context["search_results"][:2]
            for result in search_results:
                context_parts.append(f"Web Source: {result.get('title', '')}")
        
        return "\n".join(context_parts)

    def _convert_to_solution_response(
        self, 
        question: str, 
        solution_data: Dict[str, Any], 
        context: Dict[str, Any]
    ) -> SolutionResponse:
        """Convert DSPy solution to SolutionResponse object"""
        from app.models.schemas import Step, SourceType
        import uuid
        
        steps = []
        for step_data in solution_data.get("steps", []):
            step = Step(
                step_number=step_data.get("step_number", 1),
                description=step_data.get("description", ""),
                explanation=step_data.get("explanation", ""),
                formula=step_data.get("formula"),
                visual_aid=step_data.get("visual_aid")
            )
            steps.append(step)
        
        # Determine subject
        subject_str = context.get("subject", "algebra")
        try:
            subject = QuestionType(subject_str)
        except ValueError:
            subject = QuestionType.ALGEBRA
        
        return SolutionResponse(
            question=question,
            solution_id=str(uuid.uuid4()),
            steps=steps,
            final_answer=solution_data.get("final_answer", ""),
            confidence_score=solution_data.get("confidence_score", 0.7),
            source=SourceType.HYBRID,  # Feedback-optimized is hybrid
            subject=subject,
            difficulty_level=5,  # Default
            processing_time=0.0,
            references=[],
            created_at=datetime.utcnow()
        )