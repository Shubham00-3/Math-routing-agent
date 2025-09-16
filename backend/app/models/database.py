from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from typing import Generator
import logging

from app.core.config import settings

logger = logging.getLogger(__name__)

# Create SQLAlchemy engine
try:
    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,  # Log SQL queries in debug mode
        pool_pre_ping=True,   # Verify connections before use
        pool_recycle=300      # Recycle connections every 5 minutes
    )
    logger.info("Database engine created successfully")
except Exception as e:
    logger.warning(f"Failed to create database engine: {e}")
    logger.info("Running without database connection")
    engine = None

# Create SessionLocal class
if engine:
    SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
else:
    SessionLocal = None

# Create Base class for SQLAlchemy models
Base = declarative_base()


def get_db() -> Generator[Session, None, None]:
    """
    Database dependency function for FastAPI
    Yields a database session and ensures it's closed after use
    """
    if not SessionLocal:
        logger.warning("Database not available - returning None")
        yield None
        return
        
    db = SessionLocal()
    try:
        yield db
    except Exception as e:
        logger.error(f"Database session error: {e}")
        db.rollback()
        raise
    finally:
        db.close()


def init_db():
    """Initialize database tables"""
    if not engine:
        logger.warning("No database engine available - skipping table creation")
        return
        
    try:
        # Import all models here to ensure they're registered with Base
        # from app.models.feedback import Feedback  # Example
        # from app.models.solution import Solution  # Example
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("Database tables created successfully")
    except Exception as e:
        logger.error(f"Failed to create database tables: {e}")


def check_db_connection() -> bool:
    """Check if database connection is working"""
    if not engine:
        return False
        
    try:
        with engine.connect() as connection:
            connection.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False