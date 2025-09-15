from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks
from typing import List, Optional
import logging
from datetime import datetime

from app.models.schemas import (
    FeedbackRequest, FeedbackResponse, FeedbackType,
    SolutionResponse
)
from app.agents.feedback_agent import FeedbackAgent
from app.core.security import get_current_user
from app.main import manager

logger = logging.getLogger(__name__)
router = APIRouter()

feedback_agent = FeedbackAgent()

@router.post("/feedback/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback_request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Submit feedback for a mathematical solution.
    
    - **solution_id**: ID of the solution to provide feedback for
    - **feedback_type**: Type of feedback (accuracy, clarity, etc.)
    - **rating**: Rating from 1-5
    - **comments**: Optional detailed comments
    - **improvement_suggestions**: Optional suggestions for improvement
    """
    try:
        # Add user context to feedback
        feedback_request.user_id = current_user.get('user_id')
        
        logger.info(f"Feedback submission for solution {feedback_request.solution_id}")
        
        # Process feedback
        response = await feedback_agent.process_feedback(feedback_request)
        
        # Send real-time update to user
        if current_user.get('user_id'):
            background_tasks.add_task(
                send_feedback_update,
                current_user['user_id'],
                response
            )
        
        # Trigger optimization if needed
        background_tasks.add_task(
            check_optimization_trigger,
            feedback_request.solution_id
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing feedback: {str(e)}"
        )

@router.get("/feedback/solution/{solution_id}")
async def get_solution_feedback(
    solution_id: str,
    current_user: dict = Depends(get_current_user)
):
    """Get all feedback for a specific solution"""
    try:
        # Get feedback analytics for the solution
        analytics = await feedback_agent.get_feedback_analytics(
            solution_id=solution_id
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error retrieving feedback for solution {solution_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving feedback"
        )

@router.get("/feedback/analytics")
async def get_feedback_analytics(
    days: Optional[int] = 7,
    current_user: dict = Depends(get_current_user)
):
    """Get overall feedback analytics"""
    try:
        analytics = await feedback_agent.get_feedback_analytics(
            time_period=days
        )
        
        return analytics
        
    except Exception as e:
        logger.error(f"Error retrieving feedback analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving analytics"
        )

@router.post("/feedback/optimize")
async def trigger_model_optimization(
    force: bool = False,
    current_user: dict = Depends(get_current_user)
):
    """Manually trigger model optimization using feedback"""
    try:
        # Check if user has admin permissions (in production)
        # For now, allow any authenticated user
        
        result = await feedback_agent.optimize_model_with_feedback(
            force_optimization=force
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error triggering optimization: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error triggering optimization"
        )

async def send_feedback_update(user_id: str, feedback_response: FeedbackResponse):
    """Send feedback processing update via WebSocket"""
    try:
        await manager.send_to_user(
            {
                "type": "feedback_processed",
                "feedback_id": feedback_response.feedback_id,
                "improvements_applied": feedback_response.improvements_applied,
                "suggestions": feedback_response.next_suggestions
            },
            user_id
        )
    except Exception as e:
        logger.error(f"Error sending feedback update: {str(e)}")

async def check_optimization_trigger(solution_id: str):
    """Check if optimization should be triggered"""
    try:
        # This would check feedback patterns and trigger optimization
        # Implementation depends on your optimization strategy
        pass
    except Exception as e:
        logger.error(f"Error checking optimization trigger: {str(e)}")

# backend/app/api/routes/analytics.py
from fastapi import APIRouter, HTTPException, Depends
from typing import Optional, List, Dict, Any
import logging
from datetime import datetime, timedelta

from app.core.security import get_current_user
from app.services.knowledge_base import KnowledgeBaseService
from app.agents.feedback_agent import FeedbackAgent

logger = logging.getLogger(__name__)
router = APIRouter()

kb_service = KnowledgeBaseService()
feedback_agent = FeedbackAgent()

@router.get("/analytics/dashboard")
async def get_dashboard_analytics(
    days: Optional[int] = 7,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive dashboard analytics"""
    try:
        # Knowledge base statistics
        kb_stats = await kb_service.get_collection_stats()
        
        # Feedback analytics
        feedback_analytics = await feedback_agent.get_feedback_analytics(
            time_period=days
        )
        
        # System performance metrics (would be from monitoring system)
        performance_metrics = {
            "avg_response_time": 2.3,
            "success_rate": 0.95,
            "total_queries": 1250,
            "uptime": 0.999
        }
        
        # Subject distribution
        subject_analytics = await get_subject_analytics(days)
        
        # Recent activity
        recent_activity = await get_recent_activity(limit=10)
        
        return {
            "knowledge_base": kb_stats,
            "feedback": feedback_analytics,
            "performance": performance_metrics,
            "subjects": subject_analytics,
            "recent_activity": recent_activity,
            "period_days": days,
            "generated_at": datetime.utcnow().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error generating dashboard analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating analytics"
        )

@router.get("/analytics/performance")
async def get_performance_metrics(
    days: Optional[int] = 7,
    current_user: dict = Depends(get_current_user)
):
    """Get detailed performance metrics"""
    try:
        # This would integrate with monitoring systems like Prometheus
        # For now, return simulated data
        
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=days)
        
        metrics = {
            "time_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat(),
                "days": days
            },
            "response_times": {
                "p50": 1.2,
                "p95": 3.5,
                "p99": 8.1,
                "avg": 2.3
            },
            "success_rates": {
                "overall": 0.95,
                "by_source": {
                    "knowledge_base": 0.98,
                    "web_search": 0.92,
                    "hybrid": 0.96,
                    "standalone": 0.89
                }
            },
            "error_rates": {
                "total_errors": 63,
                "error_rate": 0.05,
                "by_type": {
                    "timeout": 25,
                    "validation": 18,
                    "llm_error": 12,
                    "system_error": 8
                }
            },
            "resource_usage": {
                "cpu_avg": 45.2,
                "memory_avg": 68.7,
                "requests_per_minute": 8.3
            }
        }
        
        return metrics
        
    except Exception as e:
        logger.error(f"Error retrieving performance metrics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving performance metrics"
        )

@router.get("/analytics/subjects")
async def get_subject_analytics(
    days: Optional[int] = 7,
    current_user: dict = Depends(get_current_user)
):
    """Get analytics by mathematical subject"""
    try:
        # This would query the database for subject distribution
        # For now, return simulated data
        
        subject_data = {
            "algebra": {
                "count": 345,
                "avg_rating": 4.2,
                "avg_confidence": 0.87,
                "common_topics": ["quadratic equations", "linear systems", "polynomials"]
            },
            "calculus": {
                "count": 198,
                "avg_rating": 3.8,
                "avg_confidence": 0.79,
                "common_topics": ["derivatives", "integrals", "limits"]
            },
            "geometry": {
                "count": 156,
                "avg_rating": 4.1,
                "avg_confidence": 0.85,
                "common_topics": ["area calculation", "trigonometry", "proofs"]
            },
            "statistics": {
                "count": 87,
                "avg_rating": 3.9,
                "avg_confidence": 0.82,
                "common_topics": ["probability", "distributions", "hypothesis testing"]
            }
        }
        
        return subject_data
        
    except Exception as e:
        logger.error(f"Error retrieving subject analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving subject analytics"
        )

@router.get("/analytics/trends")
async def get_trend_analytics(
    days: Optional[int] = 30,
    current_user: dict = Depends(get_current_user)
):
    """Get trend analytics over time"""
    try:
        # Generate trend data (would be from time-series database)
        trends = {
            "daily_queries": generate_daily_trend(days),
            "avg_ratings": generate_rating_trend(days),
            "response_times": generate_response_time_trend(days),
            "subject_popularity": generate_subject_trend(days),
            "feedback_volume": generate_feedback_trend(days)
        }
        
        return trends
        
    except Exception as e:
        logger.error(f"Error retrieving trend analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error retrieving trend analytics"
        )

async def get_recent_activity(limit: int = 10) -> List[Dict[str, Any]]:
    """Get recent system activity"""
    # This would query recent activities from database
    # For now, return simulated data
    
    activities = [
        {
            "type": "solution_generated",
            "timestamp": datetime.utcnow().isoformat(),
            "subject": "algebra",
            "confidence": 0.92,
            "source": "knowledge_base"
        },
        {
            "type": "feedback_received", 
            "timestamp": (datetime.utcnow() - timedelta(minutes=5)).isoformat(),
            "rating": 5,
            "feedback_type": "clarity"
        },
        {
            "type": "optimization_triggered",
            "timestamp": (datetime.utcnow() - timedelta(hours=2)).isoformat(),
            "improvement": 0.15
        }
    ]
    
    return activities[:limit]

def generate_daily_trend(days: int) -> List[Dict[str, Any]]:
    """Generate daily query trend data"""
    import random
    
    trend_data = []
    base_date = datetime.utcnow() - timedelta(days=days)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        trend_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "queries": random.randint(50, 150),
            "unique_users": random.randint(20, 80)
        })
    
    return trend_data

def generate_rating_trend(days: int) -> List[Dict[str, Any]]:
    """Generate rating trend data"""
    import random
    
    trend_data = []
    base_date = datetime.utcnow() - timedelta(days=days)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        trend_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "avg_rating": round(random.uniform(3.5, 4.5), 2),
            "rating_count": random.randint(30, 100)
        })
    
    return trend_data

def generate_response_time_trend(days: int) -> List[Dict[str, Any]]:
    """Generate response time trend data"""
    import random
    
    trend_data = []
    base_date = datetime.utcnow() - timedelta(days=days)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        trend_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "avg_response_time": round(random.uniform(1.5, 3.5), 2),
            "p95_response_time": round(random.uniform(4.0, 8.0), 2)
        })
    
    return trend_data

def generate_subject_trend(days: int) -> List[Dict[str, Any]]:
    """Generate subject popularity trend"""
    subjects = ["algebra", "calculus", "geometry", "statistics"]
    trend_data = []
    base_date = datetime.utcnow() - timedelta(days=days)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        subject_counts = {subject: random.randint(10, 50) for subject in subjects}
        trend_data.append({
            "date": date.strftime("%Y-%m-%d"),
            **subject_counts
        })
    
    return trend_data

def generate_feedback_trend(days: int) -> List[Dict[str, Any]]:
    """Generate feedback volume trend"""
    import random
    
    trend_data = []
    base_date = datetime.utcnow() - timedelta(days=days)
    
    for i in range(days):
        date = base_date + timedelta(days=i)
        trend_data.append({
            "date": date.strftime("%Y-%m-%d"),
            "feedback_count": random.randint(15, 45),
            "avg_rating": round(random.uniform(3.5, 4.5), 2)
        })
    
    return trend_data
