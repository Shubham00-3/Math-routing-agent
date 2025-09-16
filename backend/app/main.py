# backend/app/main.py
from fastapi import FastAPI, HTTPException, Depends, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import List, Dict

# Import your agents and services
from app.agents.math_agent import MathAgent
from app.models.schemas import (
    MathQuestionRequest, SolutionResponse,
    FeedbackRequest, FeedbackResponse
)
from app.services.knowledge_base import KnowledgeBaseService
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize global agents
math_agent = None
kb_service = None

# WebSocket Connection Manager
class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}

    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket

    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]

    async def send_to_client(self, message: dict, client_id: str):
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

manager = ConnectionManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    global math_agent, kb_service
    
    # Startup
    logger.info("🚀 Starting Math Routing Agent API")
    
    try:
        # Initialize services
        logger.info("Initializing Knowledge Base Service...")
        kb_service = KnowledgeBaseService()
        await kb_service.initialize_collection()
        
        logger.info("Initializing Math Agent...")
        math_agent = MathAgent()
        
        logger.info("✅ All services initialized successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize services: {str(e)}")
        # For now, continue anyway to allow basic testing
        logger.warning("⚠️ Running in degraded mode")
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Math Routing Agent API")

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered mathematical tutoring system with human feedback learning",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Middleware setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.websocket("/socket.io/")
async def websocket_endpoint(websocket: WebSocket, client_id: str = "default_client"):
    await manager.connect(websocket, client_id)
    try:
        while True:
            data = await websocket.receive_text()
            logger.info(f"Received WebSocket message from {client_id}: {data}")
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        logger.info(f"Client {client_id} disconnected")


# Simple auth dependency (for now just return anonymous user)
async def get_current_user():
    return {"user_id": "anonymous", "username": "anonymous"}

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "🚀 Welcome to Math Routing Agent API!",
        "version": settings.APP_VERSION,
        "docs": "/docs",
        "health": "/health",
        "status": "running",
        "features": [
            "✅ AI-powered math problem solving",
            "✅ Knowledge base integration", 
            "✅ Web search capabilities",
            "✅ Human feedback learning",
            "✅ Input/Output guardrails"
        ]
    }
    
# History Endpoint
@app.get("/api/v1/math/history", response_model=List[SolutionResponse])
async def get_solution_history(limit: int = 10, offset: int = 0):
    """
    Get the history of previously solved math problems.
    """
    # This is a placeholder. In a real application, you would fetch this
    # from your database.
    history = await kb_service.get_recent_entries(limit=limit, offset=offset)
    return [entry.solution for entry in history]


# Health check endpoints
@app.get("/health")
async def health_check():
    """Basic health check endpoint"""
    global math_agent, kb_service
    
    services_status = {
        "math_agent": "initialized" if math_agent else "not_initialized",
        "knowledge_base": "initialized" if kb_service else "not_initialized"
    }
    
    return {
        "status": "healthy",
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "services": services_status,
        "message": "Math Routing Agent is running! 🎉"
    }

@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with system information"""
    global math_agent, kb_service
    
    # Check service health
    services = {}
    
    if kb_service:
        try:
            stats = await kb_service.get_collection_stats()
            services["knowledge_base"] = {
                "status": "healthy",
                "entries": stats.get("total_entries", 0)
            }
        except Exception as e:
            services["knowledge_base"] = {
                "status": "unhealthy",
                "error": str(e)
            }
    else:
        services["knowledge_base"] = {"status": "not_initialized"}
    
    services["math_agent"] = {
        "status": "healthy" if math_agent else "not_initialized"
    }
    
    overall_status = "healthy" if all(
        s.get("status") == "healthy" for s in services.values()
    ) else "degraded"
    
    return {
        "status": overall_status,
        "timestamp": time.time(),
        "version": settings.APP_VERSION,
        "services": services,
        "environment": "development" if settings.DEBUG else "production"
    }

# Main math solving endpoint
@app.post("/api/v1/math/solve", response_model=SolutionResponse)
async def solve_math_problem(
    request: MathQuestionRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """
    Solve a mathematical problem using the AI routing agent system.
    
    Features:
    - Input guardrails validation
    - Knowledge base search
    - Web search when needed
    - LLM-powered solution generation
    - Output validation
    """
    global math_agent
    
    try:
        start_time = time.time()
        
        logger.info(f"📝 Math solve request from {current_user.get('username')}: {request.question[:100]}...")
        
        # Check if math agent is available
        if not math_agent:
            logger.warning("Math agent not initialized, using fallback")
            return create_fallback_response(request)
        
        # Process the mathematical question using the integrated agent
        solution = await math_agent.solve_math_problem(request)
        
        # Add processing metadata
        solution.processing_time = time.time() - start_time
        
        logger.info(f"✅ Solution generated: {solution.solution_id} (confidence: {solution.confidence_score:.2f})")
        
        return solution
        
    except Exception as e:
        logger.error(f"❌ Error solving math problem: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating solution: {str(e)}"
        )

# Feedback endpoint
@app.post("/api/v1/feedback/submit", response_model=FeedbackResponse)
async def submit_feedback(
    feedback_request: FeedbackRequest,
    background_tasks: BackgroundTasks,
    current_user: dict = Depends(get_current_user)
):
    """Submit feedback for a mathematical solution"""
    global math_agent
    
    try:
        # Add user context to feedback
        feedback_request.user_id = current_user.get('user_id')
        
        logger.info(f"📝 Feedback submission for solution {feedback_request.solution_id}")
        
        if not math_agent:
            # Basic feedback response without processing
            return FeedbackResponse(
                feedback_id=str(uuid.uuid4()),
                processed=False,
                improvements_applied=[],
                next_suggestions=["Math agent not initialized - feedback stored but not processed"]
            )
        
        # Process feedback using the math agent
        response = await math_agent.process_feedback(feedback_request)
        
        logger.info(f"✅ Feedback processed: {response.feedback_id}")
        return response
        
    except Exception as e:
        logger.error(f"❌ Error processing feedback: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error processing feedback: {str(e)}"
        )

# Analytics endpoint
@app.get("/api/v1/analytics/dashboard")
async def get_dashboard_analytics(
    days: int = 7,
    current_user: dict = Depends(get_current_user)
):
    """Get comprehensive dashboard analytics"""
    global math_agent, kb_service
    
    try:
        analytics = {
            "period_days": days,
            "generated_at": time.time()
        }
        
        # Knowledge base stats
        if kb_service:
            try:
                kb_stats = await kb_service.get_collection_stats()
                analytics["knowledge_base"] = kb_stats
            except Exception as e:
                analytics["knowledge_base"] = {"error": str(e)}
        else:
            analytics["knowledge_base"] = {"status": "not_initialized"}
        
        # Feedback analytics
        if math_agent:
            try:
                feedback_analytics = await math_agent.get_analytics(days)
                analytics["feedback"] = feedback_analytics
            except Exception as e:
                analytics["feedback"] = {"error": str(e)}
        else:
            analytics["feedback"] = {"status": "not_initialized"}
        
        # Basic performance metrics
        analytics["performance"] = {
            "avg_response_time": 2.1,
            "success_rate": 0.95,
            "total_queries": 42,  # This would come from database
            "uptime": 0.99
        }
        
        return analytics
        
    except Exception as e:
        logger.error(f"❌ Error generating analytics: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Error generating analytics"
        )

# Test endpoint for development
@app.get("/api/v1/test")
async def test_connection():
    """Test endpoint to verify system status"""
    global math_agent, kb_service
    
    status = {
        "message": "🎯 Backend Connection Test Successful!",
        "timestamp": time.time(),
        "components": {
            "fastapi": "✅ Running",
            "math_agent": "✅ Ready" if math_agent else "❌ Not initialized",
            "knowledge_base": "✅ Ready" if kb_service else "❌ Not initialized"
        },
        "ready_for": []
    }
    
    if math_agent:
        status["ready_for"].extend([
            "✅ Math problem solving",
            "✅ Feedback processing"
        ])
    
    if kb_service:
        status["ready_for"].append("✅ Knowledge base queries")
    
    if not math_agent or not kb_service:
        status["ready_for"].append("⚠️ Running in degraded mode")
    
    return status

def create_fallback_response(request: MathQuestionRequest) -> SolutionResponse:
    """Create a fallback response when services are not available"""
    from app.models.schemas import Step, SourceType, QuestionType
    from datetime import datetime
    
    return SolutionResponse(
        question=request.question,
        solution_id=f"fallback_{uuid.uuid4().hex[:8]}",
        steps=[
            Step(
                step_number=1,
                description="🔄 System Initializing",
                explanation="The AI agents are still starting up. This is a basic response.",
                formula=None,
                visual_aid=None
            ),
            Step(
                step_number=2,
                description="📚 Setup Required",
                explanation="Complete the setup steps to enable full AI-powered solutions.",
                formula=None,
                visual_aid=None
            )
        ],
        final_answer="⚠️ Please complete setup for full AI capabilities",
        confidence_score=0.5,
        source=SourceType.WEB_SEARCH,
        subject=request.subject or QuestionType.ALGEBRA,
        difficulty_level=request.difficulty_level or 5,
        processing_time=0.1,
        created_at=datetime.utcnow()
    )

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred",
            "timestamp": time.time(),
            "request_id": str(uuid.uuid4())
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)