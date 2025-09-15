import asyncio
import json
import csv
import logging
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

from app.services.knowledge_base import KnowledgeBaseService
from app.models.schemas import KnowledgeEntry, QuestionType, SolutionResponse, Step

logger = logging.getLogger(__name__)

class DatasetLoader:
    """Load mathematical datasets into the knowledge base"""
    
    def __init__(self, kb_service: KnowledgeBaseService):
        self.kb_service = kb_service
        
    async def load_math_dataset(self, dataset_path: str, dataset_type: str = "json") -> int:
        """Load mathematical dataset from file"""
        try:
            data_path = Path(dataset_path)
            if not data_path.exists():
                raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
            
            if dataset_type == "json":
                entries = await self._load_json_dataset(data_path)
            elif dataset_type == "csv":
                entries = await self._load_csv_dataset(data_path)
            else:
                raise ValueError(f"Unsupported dataset type: {dataset_type}")
            
            # Load entries into knowledge base
            loaded_count = 0
            for entry in entries:
                try:
                    await self.kb_service.add_knowledge_entry(entry)
                    loaded_count += 1
                except Exception as e:
                    logger.warning(f"Failed to load entry: {str(e)}")
                    continue
            
            logger.info(f"Successfully loaded {loaded_count} entries from {dataset_path}")
            return loaded_count
            
        except Exception as e:
            logger.error(f"Error loading dataset: {str(e)}")
            raise

    async def _load_json_dataset(self, file_path: Path) -> List[KnowledgeEntry]:
        """Load dataset from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        entries = []
        for item in data:
            entry = self._create_knowledge_entry_from_dict(item)
            if entry:
                entries.append(entry)
        
        return entries

    async def _load_csv_dataset(self, file_path: Path) -> List[KnowledgeEntry]:
        """Load dataset from CSV file"""
        entries = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                entry = self._create_knowledge_entry_from_dict(row)
                if entry:
                    entries.append(entry)
        
        return entries

    def _create_knowledge_entry_from_dict(self, data: Dict[str, Any]) -> Optional[KnowledgeEntry]:
        """Create KnowledgeEntry from dictionary data"""
        try:
            # Parse solution steps
            steps = []
            if isinstance(data.get('steps'), list):
                for i, step_data in enumerate(data['steps']):
                    if isinstance(step_data, dict):
                        step = Step(
                            step_number=step_data.get('step_number', i + 1),
                            description=step_data.get('description', ''),
                            formula=step_data.get('formula'),
                            explanation=step_data.get('explanation', ''),
                            visual_aid=step_data.get('visual_aid')
                        )
                        steps.append(step)
            elif isinstance(data.get('solution'), str):
                # Parse solution text into steps
                solution_text = data['solution']
                step_parts = solution_text.split('\n')
                for i, part in enumerate(step_parts):
                    if part.strip():
                        step = Step(
                            step_number=i + 1,
                            description=part.strip(),
                            explanation="",
                            formula=None,
                            visual_aid=None
                        )
                        steps.append(step)

            # Create solution response
            solution = SolutionResponse(
                question=data['question'],
                solution_id=str(uuid.uuid4()),
                steps=steps,
                final_answer=data.get('answer', data.get('final_answer', '')),
                confidence_score=float(data.get('confidence_score', 0.9)),
                source="knowledge_base",
                subject=QuestionType(data.get('subject', 'algebra')),
                difficulty_level=int(data.get('difficulty', data.get('difficulty_level', 5))),
                processing_time=0.0,
                references=data.get('references', []),
                created_at=datetime.utcnow()
            )

            # Create knowledge entry
            entry = KnowledgeEntry(
                id=str(uuid.uuid4()),
                question=data['question'],
                solution=solution,
                tags=data.get('tags', []),
                difficulty=int(data.get('difficulty', 5)),
                subject=QuestionType(data.get('subject', 'algebra')),
                embedding=[],  # Will be generated during insertion
                usage_count=0,
                last_accessed=datetime.utcnow()
            )

            return entry

        except Exception as e:
            logger.warning(f"Error creating knowledge entry from data: {str(e)}")
            return None

# Sample dataset creation for testing
async def create_sample_dataset():
    """Create a sample math dataset for testing"""
    sample_data = [
        {
            "question": "Solve the quadratic equation x² + 5x + 6 = 0",
            "subject": "algebra",
            "difficulty": 3,
            "steps": [
                {
                    "step_number": 1,
                    "description": "Identify coefficients a=1, b=5, c=6",
                    "explanation": "In the standard form ax² + bx + c = 0"
                },
                {
                    "step_number": 2,
                    "description": "Factor the quadratic expression",
                    "explanation": "Look for two numbers that multiply to 6 and add to 5"
                },
                {
                    "step_number": 3,
                    "description": "(x + 2)(x + 3) = 0",
                    "explanation": "2 × 3 = 6 and 2 + 3 = 5"
                },
                {
                    "step_number": 4,
                    "description": "Solve each factor: x = -2 or x = -3",
                    "explanation": "Set each factor equal to zero"
                }
            ],
            "final_answer": "x = -2 or x = -3",
            "tags": ["quadratic", "factoring", "algebra", "polynomials"],
            "confidence_score": 0.95
        },
        {
            "question": "Find the derivative of f(x) = 3x² + 2x - 1",
            "subject": "calculus",
            "difficulty": 4,
            "steps": [
                {
                    "step_number": 1,
                    "description": "Apply the power rule to each term",
                    "formula": "d/dx(x^n) = nx^(n-1)"
                },
                {
                    "step_number": 2,
                    "description": "For 3x²: d/dx(3x²) = 3 × 2x¹ = 6x",
                    "explanation": "Multiply coefficient by exponent, reduce exponent by 1"
                },
                {
                    "step_number": 3,
                    "description": "For 2x: d/dx(2x) = 2 × 1x⁰ = 2",
                    "explanation": "Derivative of x is 1"
                },
                {
                    "step_number": 4,
                    "description": "For constant -1: d/dx(-1) = 0",
                    "explanation": "Derivative of constant is 0"
                }
            ],
            "final_answer": "f'(x) = 6x + 2",
            "tags": ["derivative", "power rule", "calculus", "differentiation"],
            "confidence_score": 0.98
        }
    ]
    
    return sample_data