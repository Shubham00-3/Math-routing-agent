#!/usr/bin/env python3
"""
Knowledge Base Population Script
Populates the Qdrant knowledge base with sample mathematical problems
"""

import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

backend_path = str(Path(__file__).parent.parent / "backend")
sys.path.append(backend_path)

from app.services.knowledge_base import KnowledgeBaseService
from app.models.schemas import (
    KnowledgeEntry, QuestionType, SolutionResponse, Step, SourceType
)
from app.core.config import settings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class KnowledgeBasePopulator:
    """Populate the knowledge base with sample mathematical problems"""
    
    def __init__(self):
        self.kb_service = KnowledgeBaseService()
        
    async def initialize(self):
        """Initialize the knowledge base service"""
        try:
            await self.kb_service.initialize_collection()
            logger.info("Knowledge base service initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge base service: {e}")
            raise

    async def load_sample_dataset(self, dataset_path: str) -> int:
        """Load sample dataset from JSON file"""
        try:
            # Load the dataset
            dataset_path = Path(dataset_path)
            if not dataset_path.exists():
                logger.error(f"Dataset file not found: {dataset_path}")
                return 0
                
            with open(dataset_path, 'r', encoding='utf-8') as f:
                sample_data = json.load(f)
            
            logger.info(f"Loaded {len(sample_data)} problems from {dataset_path}")
            
            # Process each problem
            loaded_count = 0
            for i, problem_data in enumerate(sample_data, 1):
                try:
                    entry = self._create_knowledge_entry(problem_data)
                    await self.kb_service.add_knowledge_entry(entry)
                    loaded_count += 1
                    logger.info(f"Loaded problem {i}/{len(sample_data)}: {problem_data['question'][:60]}...")
                except Exception as e:
                    logger.warning(f"Failed to load problem {i}: {str(e)}")
                    continue
            
            logger.info(f"Successfully loaded {loaded_count} problems into knowledge base")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    def _create_knowledge_entry(self, problem_data: Dict[str, Any]) -> KnowledgeEntry:
        """Create a KnowledgeEntry from problem data"""
        try:
            # Parse steps
            steps = []
            for step_data in problem_data.get('steps', []):
                step = Step(
                    step_number=step_data.get('step_number', 1),
                    description=step_data.get('description', ''),
                    explanation=step_data.get('explanation', ''),
                    formula=step_data.get('formula'),
                    visual_aid=step_data.get('visual_aid')
                )
                steps.append(step)
            
            # Create solution response
            solution = SolutionResponse(
                question=problem_data['question'],
                solution_id=str(uuid.uuid4()),
                steps=steps,
                final_answer=problem_data.get('final_answer', ''),
                confidence_score=problem_data.get('confidence_score', 0.9),
                source=SourceType.KNOWLEDGE_BASE,
                subject=QuestionType(problem_data.get('subject', 'algebra')),
                difficulty_level=problem_data.get('difficulty', 5),
                processing_time=0.0,
                references=problem_data.get('references', []),
                created_at=datetime.utcnow()
            )
            
            # Create knowledge entry
            entry = KnowledgeEntry(
                id=str(uuid.uuid4()),
                question=problem_data['question'],
                solution=solution,
                tags=problem_data.get('tags', []),
                difficulty=problem_data.get('difficulty', 5),
                subject=QuestionType(problem_data.get('subject', 'algebra')),
                embedding=[],  # Will be generated during insertion
                usage_count=0,
                last_accessed=datetime.utcnow()
            )
            
            return entry
            
        except Exception as e:
            logger.error(f"Error creating knowledge entry: {str(e)}")
            raise

    async def verify_knowledge_base(self) -> Dict[str, Any]:
        """Verify the knowledge base contents"""
        try:
            stats = await self.kb_service.get_collection_stats()
            
            # Test a sample search
            test_results = await self.kb_service.search_similar_questions(
                "quadratic equation", limit=3
            )
            
            verification = {
                "total_entries": stats.get("total_entries", 0),
                "collection_status": "healthy" if stats.get("total_entries", 0) > 0 else "empty",
                "sample_search_results": len(test_results),
                "subjects_available": list(set([
                    entry.get("subject", "unknown") for entry in test_results
                ]))
            }
            
            logger.info(f"Knowledge base verification: {verification}")
            return verification
            
        except Exception as e:
            logger.error(f"Error verifying knowledge base: {str(e)}")
            return {"error": str(e)}

    async def clear_knowledge_base(self):
        """Clear the knowledge base (for testing)"""
        try:
            # Delete the collection and recreate it
            self.kb_service.client.delete_collection(self.kb_service.collection_name)
            await self.kb_service.initialize_collection()
            logger.info("Knowledge base cleared successfully")
        except Exception as e:
            logger.error(f"Error clearing knowledge base: {str(e)}")
            raise

async def main():
    """Main function to populate the knowledge base"""
    populator = KnowledgeBasePopulator()
    
    try:
        # Initialize the knowledge base
        logger.info("��� Starting knowledge base population...")
        await populator.initialize()
        
        # Check if we should clear existing data
        clear_existing = input("Clear existing knowledge base? (y/N): ").lower().strip()
        if clear_existing == 'y':
            logger.info("Clearing existing knowledge base...")
            await populator.clear_knowledge_base()
        
        # Create sample dataset file if it doesn't exist
        dataset_path = Path("sample_math_dataset.json")
        if not dataset_path.exists():
            logger.info("Creating sample dataset file...")
            # You'll need to create this file with the JSON data from the previous artifact
            sample_data = []  # Add your sample data here
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(sample_data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created sample dataset: {dataset_path}")
        
        # Load the dataset
        loaded_count = await populator.load_sample_dataset(str(dataset_path))
        
        # Verify the results
        verification = await populator.verify_knowledge_base()
        
        # Print summary
        print("\n" + "="*50)
        print("��� KNOWLEDGE BASE POPULATION SUMMARY")
        print("="*50)
        print(f"✅ Problems loaded: {loaded_count}")
        print(f"��� Total entries: {verification.get('total_entries', 0)}")
        print(f"��� Search test results: {verification.get('sample_search_results', 0)}")
        print(f"��� Subjects available: {', '.join(verification.get('subjects_available', []))}")
        print(f"��� Status: {verification.get('collection_status', 'unknown')}")
        print("="*50)
        
        if loaded_count > 0:
            print("✅ Knowledge base populated successfully!")
            print("\nYou can now test these questions in your demo:")
            print("1. 'Solve the quadratic equation x² + 5x + 6 = 0'")
            print("2. 'Find the derivative of f(x) = 3x³ + 2x² - x + 5'")
            print("3. 'Calculate the area of a triangle with sides 3, 4, and 5'")
        else:
            print("❌ No problems were loaded. Please check the logs for errors.")
            
    except Exception as e:
        logger.error(f"Fatal error during population: {str(e)}")
        print(f"❌ Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
