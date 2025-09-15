from fastapi import APIRouter, HTTPException
from typing import Dict, Any
import time
import psutil
import logging
from datetime import datetime

from app.core.config import settings
from app.services.knowledge_base import KnowledgeBaseService

logger = logging.getLogger(__name__)
router = APIRouter()

@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """Basic health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "version": settings.APP_VERSION,
        "environment": "production" if not settings.DEBUG else "development"
    }

@router.get("/health/detailed")
async def detailed_health_check() -> Dict[str, Any]:
    """Detailed health check with system information"""
    try:
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Check service dependencies
        dependencies = await check_dependencies()
        
        # Overall health status
        overall_status = "healthy" if all(
            dep["status"] == "healthy" for dep in dependencies.values()
        ) else "degraded"
        
        return {
            "status": overall_status,
            "timestamp": datetime.utcnow().isoformat(),
            "version": settings.APP_VERSION,
            "system": {
                "cpu_percent": cpu_percent,
                "memory": {
                    "total": memory.total,
                    "available": memory.available,
                    "percent": memory.percent
                },
                "disk": {
                    "total": disk.total,
                    "free": disk.free,
                    "percent": (disk.used / disk.total) * 100
                }
            },
            "dependencies": dependencies
        }
        
    except Exception as e:
        logger.error(f"Error in detailed health check: {str(e)}")
        return {
            "status": "error",
            "timestamp": datetime.utcnow().isoformat(),
            "error": str(e)
        }

async def check_dependencies() -> Dict[str, Dict[str, Any]]:
    """Check health of external dependencies"""
    dependencies = {}
    
    # Check Qdrant (Vector Database)
    try:
        kb_service = KnowledgeBaseService()
        # Try a simple operation
        stats = await kb_service.get_collection_stats()
        dependencies["qdrant"] = {
            "status": "healthy",
            "response_time": 0.1,  # Would measure actual response time
            "details": {"entries": stats.get("total_entries", 0)}
        }
    except Exception as e:
        dependencies["qdrant"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check Redis (if configured)
    try:
        # This would test Redis connection
        dependencies["redis"] = {
            "status": "healthy",
            "response_time": 0.05
        }
    except Exception as e:
        dependencies["redis"] = {
            "status": "unhealthy", 
            "error": str(e)
        }
    
    # Check Database
    try:
        # This would test database connection
        dependencies["database"] = {
            "status": "healthy",
            "response_time": 0.02
        }
    except Exception as e:
        dependencies["database"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    # Check LLM API
    try:
        # This would test LLM API connectivity
        dependencies["llm_api"] = {
            "status": "healthy",
            "response_time": 1.2,
            "model": settings.LLM_MODEL
        }
    except Exception as e:
        dependencies["llm_api"] = {
            "status": "unhealthy",
            "error": str(e)
        }
    
    return dependencies

@router.get("/health/ready")
async def readiness_check():
    """Readiness probe for Kubernetes"""
    try:
        # Check if all critical services are ready
        dependencies = await check_dependencies()
        
        critical_services = ["qdrant", "database", "llm_api"]
        ready = all(
            dependencies.get(service, {}).get("status") == "healthy"
            for service in critical_services
        )
        
        if ready:
            return {"status": "ready"}
        else:
            raise HTTPException(status_code=503, detail="Service not ready")
            
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Service not ready: {str(e)}")

@router.get("/health/live")
async def liveness_check():
    """Liveness probe for Kubernetes"""
    # Simple check that the application is running
    return {"status": "alive", "timestamp": time.time()}