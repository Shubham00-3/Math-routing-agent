from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """Application settings from environment variables"""
    
    # Application Settings
    APP_NAME: str = "Math Routing Agent"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    SECRET_KEY: str = "math-agent-secret-key-change-in-production"
    
    # Database Configuration
    POSTGRES_USER: str = "mathagent"
    POSTGRES_PASSWORD: str = "password123"
    POSTGRES_SERVER: str = "localhost"
    POSTGRES_PORT: int = 5432
    POSTGRES_DB: str = "mathagent_db"
    DATABASE_URL: str = "postgresql://mathagent:password123@localhost:5432/mathagent_db"
    
    # Vector Database (Qdrant)
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_COLLECTION_NAME: str = "math_knowledge_base"
    
    # Redis Configuration
    REDIS_HOST: str = "localhost"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    # LLM Configuration - Your Groq/Gemini Setup
    GROQ_API_KEY: str = ""
    GOOGLE_API_KEY: str = ""
    OPENAI_API_KEY: str = ""  # Backup if needed
    LLM_MODEL: str = "groq/llama3-8b-8192"
    LLM_PROVIDER: str = "groq"
    LLM_TEMPERATURE: float = 0.1
    MAX_TOKENS: int = 2000
    
    # Search Configuration
    TAVILY_API_KEY: str = ""
    SEARCH_MAX_RESULTS: int = 5
    SEARCH_TIMEOUT: int = 10
    
    # Guardrails Settings
    CONTENT_FILTER_THRESHOLD: float = 0.8
    MATH_RELEVANCE_THRESHOLD: float = 0.7
    
    # Feedback & Learning
    FEEDBACK_AGGREGATION_WINDOW: int = 24
    MIN_FEEDBACK_COUNT: int = 3
    DSPY_OPTIMIZATION_ENABLED: bool = True
    
    # MCP Configuration
    MCP_SERVER_HOST: str = "localhost"
    MCP_SERVER_PORT: int = 8080
    MCP_SEARCH_SERVER_PATH: str = "../mcp/servers/search_server.py"
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Global settings instance
settings = Settings()