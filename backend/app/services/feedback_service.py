import logging
import json
import uuid
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import and_, func, desc
import pandas as pd
import numpy as np
from collections import defaultdict

from app.models.schemas import (
    FeedbackRequest, FeedbackResponse, FeedbackType, 
    SolutionResponse, Step, QuestionType
)
from app.models.database import get_db
from app.core.config import settings

logger = logging.getLogger(__name__)

class FeedbackAggregator:
    """Aggregates and analyzes human feedback for solution improvement"""
    
    def __init__(self):
        self.feedback_window_hours = settings.FEEDBACK_AGGREGATION_WINDOW
        self.min_feedback_count = settings.MIN_FEEDBACK_COUNT
        
    async def collect_feedback(
        self, 
        feedback_request: FeedbackRequest,
        db: Session
    ) -> FeedbackResponse:
        """Collect and process human feedback"""
        try:
            # Store feedback in database
            feedback_id = await self._store_feedback(feedback_request, db)
            
            # Analyze aggregated feedback for this solution
            improvements = await self._analyze_solution_feedback(
                feedback_request.solution_id, db
            )
            
            # Generate improvement suggestions
            suggestions = await self._generate_improvement_suggestions(
                feedback_request, improvements
            )
            
            # Check if solution needs updating based on feedback
            should_update = await self._should_update_solution(
                feedback_request.solution_id, db
            )
            
            improvements_applied = []
            if should_update:
                improvements_applied = await self._apply_feedback_improvements(
                    feedback_request.solution_id, improvements, db
                )
            
            return FeedbackResponse(
                feedback_id=feedback_id,
                processed=True,
                improvements_applied=improvements_applied,
                next_suggestions=suggestions
            )
            
        except Exception as e:
            logger.error(f"Error processing feedback: {str(e)}")
            return FeedbackResponse(
                feedback_id="error",
                processed=False,
                improvements_applied=[],
                next_suggestions=[]
            )

    async def _store_feedback(
        self, 
        feedback_request: FeedbackRequest, 
        db: Session
    ) -> str:
        """Store feedback in database"""
        feedback_id = str(uuid.uuid4())
        
        # This would typically use SQLAlchemy models
        # For now, we'll simulate database storage
        feedback_data = {
            "id": feedback_id,
            "solution_id": feedback_request.solution_id,
            "feedback_type": feedback_request.feedback_type.value,
            "rating": feedback_request.rating,
            "comments": feedback_request.comments,
            "improvement_suggestions": feedback_request.improvement_suggestions,
            "user_id": feedback_request.user_id,
            "created_at": datetime.utcnow().isoformat()
        }
        
        # Store in database (implementation would go here)
        logger.info(f"Stored feedback {feedback_id} for solution {feedback_request.solution_id}")
        
        return feedback_id

    async def _analyze_solution_feedback(
        self, 
        solution_id: str, 
        db: Session
    ) -> Dict[str, Any]:
        """Analyze all feedback for a specific solution"""
        try:
            # Get all feedback for this solution from the last window period
            cutoff_time = datetime.utcnow() - timedelta(hours=self.feedback_window_hours)
            
            # This would be a database query in real implementation
            # For now, we'll simulate feedback data
            feedback_data = await self._get_solution_feedback(solution_id, cutoff_time, db)
            
            if len(feedback_data) < self.min_feedback_count:
                return {"insufficient_data": True}
            
            # Analyze feedback patterns
            analysis = {
                "total_feedback": len(feedback_data),
                "average_ratings": {},
                "common_issues": [],
                "improvement_areas": [],
                "sentiment_analysis": {},
                "feedback_trends": {}
            }
            
            # Calculate average ratings by feedback type
            for feedback_type in FeedbackType:
                type_feedback = [f for f in feedback_data if f["feedback_type"] == feedback_type.value]
                if type_feedback:
                    avg_rating = sum(f["rating"] for f in type_feedback) / len(type_feedback)
                    analysis["average_ratings"][feedback_type.value] = avg_rating
            
            # Extract common issues from comments
            all_comments = [f["comments"] for f in feedback_data if f["comments"]]
            analysis["common_issues"] = await self._extract_common_issues(all_comments)
            
            # Identify improvement areas
            low_rated_types = [
                ftype for ftype, rating in analysis["average_ratings"].items() 
                if rating < 3.0
            ]
            analysis["improvement_areas"] = low_rated_types
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing feedback for solution {solution_id}: {str(e)}")
            return {"error": str(e)}

    async def _get_solution_feedback(
        self, 
        solution_id: str, 
        cutoff_time: datetime, 
        db: Session
    ) -> List[Dict]:
        """Get feedback data for solution (simulated for now)"""
        # In real implementation, this would be a database query
        # Simulating some feedback data
        sample_feedback = [
            {
                "feedback_type": "accuracy",
                "rating": 4,
                "comments": "The solution is correct but could be clearer",
                "created_at": datetime.utcnow()
            },
            {
                "feedback_type": "clarity",
                "rating": 2,
                "comments": "Steps are confusing, need better explanations",
                "created_at": datetime.utcnow()
            },
            {
                "feedback_type": "completeness",
                "rating": 5,
                "comments": "Very thorough solution",
                "created_at": datetime.utcnow()
            }
        ]
        return sample_feedback

    async def _extract_common_issues(self, comments: List[str]) -> List[str]:
        """Extract common issues from feedback comments using NLP"""
        if not comments:
            return []
        
        # Simple keyword-based analysis (in production, use more sophisticated NLP)
        issue_keywords = {
            "clarity": ["unclear", "confusing", "hard to understand", "explain better"],
            "steps": ["missing step", "skip", "jump", "incomplete"],
            "accuracy": ["wrong", "incorrect", "error", "mistake"],
            "explanation": ["why", "how", "explain", "reasoning"],
            "formatting": ["format", "layout", "presentation", "organize"]
        }
        
        common_issues = []
        for issue_type, keywords in issue_keywords.items():
            count = sum(1 for comment in comments for keyword in keywords if keyword in comment.lower())
            if count >= 2:  # Threshold for considering it a common issue
                common_issues.append(f"{issue_type}: mentioned {count} times")
        
        return common_issues

    async def _generate_improvement_suggestions(
        self, 
        feedback_request: FeedbackRequest, 
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate actionable improvement suggestions"""
        suggestions = []
        
        if feedback_request.rating <= 2:
            suggestions.append("Consider regenerating this solution with enhanced explanations")
        
        if "clarity" in analysis.get("improvement_areas", []):
            suggestions.append("Add more detailed step-by-step explanations")
            suggestions.append("Include visual aids or diagrams where appropriate")
        
        if "accuracy" in analysis.get("improvement_areas", []):
            suggestions.append("Double-check mathematical calculations")
            suggestions.append("Verify solution against multiple sources")
        
        if feedback_request.improvement_suggestions:
            suggestions.append(f"User suggestion: {feedback_request.improvement_suggestions}")
        
        return suggestions

    async def _should_update_solution(self, solution_id: str, db: Session) -> bool:
        """Determine if solution should be updated based on feedback"""
        analysis = await self._analyze_solution_feedback(solution_id, db)
        
        if analysis.get("insufficient_data"):
            return False
        
        # Update if average rating is below threshold
        overall_rating = np.mean(list(analysis.get("average_ratings", {}).values()))
        return overall_rating < 3.0

    async def _apply_feedback_improvements(
        self, 
        solution_id: str, 
        improvements: Dict[str, Any], 
        db: Session
    ) -> List[str]:
        """Apply improvements to solution based on feedback"""
        applied_improvements = []
        
        # This would trigger solution regeneration with feedback context
        # For now, we'll log the improvements that would be applied
        if "clarity" in improvements.get("improvement_areas", []):
            applied_improvements.append("Enhanced step explanations")
        
        if "accuracy" in improvements.get("improvement_areas", []):
            applied_improvements.append("Verified mathematical accuracy")
        
        logger.info(f"Applied improvements to solution {solution_id}: {applied_improvements}")
        return applied_improvements