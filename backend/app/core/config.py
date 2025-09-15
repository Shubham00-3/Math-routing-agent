from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings and environment variables"""
    
    # Application
    APP_NAME: str = "Math Routing Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 * 24 * 8  # 8 days
    
    # Database
    POSTGRES_USER: str = "mathagent"
    POSTGRES_PASSWORD: str = "password"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: str = "5432"
    POSTGRES_DB: str = "mathagent_db"
    
    # Add DATABASE_URL as a direct field (for .env compatibility)
    DATABASE_URL: Optional[str] = None
    
    def get_database_url(self) -> str:
        """Get database URL, preferring DATABASE_URL from env if set"""
        if self.DATABASE_URL:
            return self.DATABASE_URL
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_SERVER}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
    
    # Vector Database (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "math_knowledge_base"
    
    # Redis
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # LLM Configuration
    OPENAI_API_KEY: Optional[str] = None
    ANTHROPIC_API_KEY: Optional[str] = None
    LLM_MODEL: str = "gpt-4-turbo-preview"
    LLM_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2000
    
    # Search Configuration
    TAVILY_API_KEY: Optional[str] = None
    SEARCH_MAX_RESULTS: int = 5
    SEARCH_TIMEOUT: int = 10
    
    # Guardrails
    CONTENT_FILTER_THRESHOLD: float = 0.8
    MATH_RELEVANCE_THRESHOLD: float = 0.7
    
    # Feedback & Learning
    FEEDBACK_AGGREGATION_WINDOW: int = 24  # hours
    MIN_FEEDBACK_COUNT: int = 3
    DSPY_OPTIMIZATION_ENABLED: bool = True
    
    # MCP Configuration
    MCP_SERVER_HOST: str = "localhost"
    MCP_SERVER_PORT: int = 8080
    MCP_SEARCH_SERVER_PATH: str = "./mcp/servers/search_server.py"
    
    # Add this to handle any extra fields that might be in .env
    RUN_COMMAND: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        # Allow extra fields to prevent validation errors
        extra = "allow"

# Initialize settings
settings = Settings()