# evaluation/jee_bench/benchmark.py - Proper JEE Benchmark Implementation

import json
import logging
import re
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

# Add backend to path
sys.path.append(str(Path(__file__).parent.parent.parent / "backend"))

from app.agents.math_agent import MathAgent
from app.models.schemas import MathQuestionRequest, QuestionType

logger = logging.getLogger(__name__)

class JEEBenchmarkRunner:
    """JEE Benchmark evaluation system with proper answer extraction"""
    
    def __init__(self, questions_file: Optional[str] = None):
        self.math_agent = MathAgent()
        self.questions_file = questions_file or self._find_questions_file()
        self.results = []
        
        # Answer extraction patterns for multiple choice
        self.answer_patterns = [
            r'(?:FINAL\s+)?ANSWER\s*:?\s*([ABCD])\b',
            r'(?:THE\s+)?(?:CORRECT\s+)?ANSWER\s+IS\s*:?\s*([ABCD])\b',
            r'(?:OPTION\s+)?([ABCD])\s+IS\s+(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|SOLUTION)',
            r'\b([ABCD])\s*\)\s*IS\s+(?:THE\s+)?(?:CORRECT|RIGHT)',
            r'(?:CHOOSE|SELECT)\s+(?:OPTION\s+)?([ABCD])',
            r'(?:THEREFORE|HENCE),?\s*(?:THE\s+)?(?:ANSWER\s+IS\s+)?([ABCD])\b',
            # Last resort: find isolated A, B, C, or D near end of text
            r'(?:^|\s)([ABCD])(?:\s*$|\s*\.|\s+is\s+correct)',
        ]
    
    def _find_questions_file(self) -> str:
        """Find JEE questions data file"""
        possible_paths = [
            Path(__file__).parent / "data" / "jee_questions.json",
            Path(__file__).parent / "data" / "questions.json", 
            Path(__file__).parent / "jee_data.json"
        ]
        
        for path in possible_paths:
            if path.exists():
                return str(path)
                
        # Create sample questions if none found
        logger.warning("No JEE questions file found, creating sample questions")
        return self._create_sample_questions()
    
    def _create_sample_questions(self) -> str:
        """Create sample JEE questions for testing"""
        sample_questions = [
            {
                "id": 1,
                "question": "If x² - 5x + 6 = 0, then the values of x are:",
                "options": {
                    "A": "2, 3",
                    "B": "1, 6", 
                    "C": "-2, -3",
                    "D": "2, -3"
                },
                "correct_answer": "A",
                "subject": "algebra",
                "difficulty": "medium"
            },
            {
                "id": 2,
                "question": "What is the derivative of x³ + 2x²?",
                "options": {
                    "A": "3x² + 4x",
                    "B": "3x² + 2x",
                    "C": "x² + 4x", 
                    "D": "3x + 4"
                },
                "correct_answer": "A", 
                "subject": "calculus",
                "difficulty": "medium"
            }
        ]
        
        sample_file = Path(__file__).parent / "sample_questions.json"
        with open(sample_file, 'w') as f:
            json.dump(sample_questions, f, indent=2)
            
        return str(sample_file)
    
    async def run_benchmark(self, limit: Optional[int] = None) -> Dict[str, Any]:
        """Run the JEE benchmark evaluation"""
        try:
            logger.info("Starting JEE Bench benchmark...")
            
            # Load questions
            questions = self._load_questions()
            if limit:
                questions = questions[:limit]
                
            logger.info(f"Found {len(questions)} math questions to process.")
            
            total_questions = len(questions)
            correct_answers = 0
            
            # Process each question
            for i, question_data in enumerate(questions, 1):
                logger.info(f"--- Processing question {i}/{total_questions} ---")
                
                try:
                    result = await self._process_single_question(question_data, i)
                    self.results.append(result)
                    
                    if result["is_correct"]:
                        correct_answers += 1
                        logger.info(f"Result: ✅ CORRECT")
                    else:
                        logger.info(f"Result: ❌ INCORRECT")
                        
                    # Print debug info
                    self._print_debug_info(result)
                    logger.info("-" * 60)
                    
                except Exception as e:
                    logger.error(f"Error processing question {i}: {str(e)}")
                    error_result = {
                        "question_id": question_data.get("id", i),
                        "question": question_data.get("question", "")[:100] + "...",
                        "expected_answer": question_data.get("correct_answer", ""),
                        "generated_answer": "ERROR", 
                        "extracted_answer": "ERROR",
                        "is_correct": False,
                        "error": str(e)
                    }
                    self.results.append(error_result)
            
            # Calculate final metrics
            accuracy = (correct_answers / total_questions) * 100 if total_questions > 0 else 0
            
            summary = {
                "total_questions": total_questions,
                "correct_answers": correct_answers,
                "accuracy": accuracy,
                "results": self.results,
                "timestamp": datetime.utcnow().isoformat()
            }
            
            # Save results
            self._save_results(summary)
            
            logger.info("Benchmark finished!")
            logger.info(f"--- Benchmark Summary ---")
            logger.info(f"Total Questions Processed: {total_questions}")
            logger.info(f"Correct Answers: {correct_answers}")
            logger.info(f"Accuracy: {accuracy:.2f}%")
            logger.info("-" * 25)
            
            return summary
            
        except Exception as e:
            logger.error(f"Benchmark failed: {str(e)}")
            raise
    
    async def _process_single_question(
        self, 
        question_data: Dict[str, Any], 
        question_num: int
    ) -> Dict[str, Any]:
        """Process a single JEE question"""
        
        question_text = question_data.get("question", "")
        correct_answer = question_data.get("correct_answer", "")
        options = question_data.get("options", {})
        
        logger.info(f"Question: {question_text[:100]}...")
        logger.info(f"Expected Answer: {correct_answer}")
        
        # Create request
        request = MathQuestionRequest(
            question=question_text,
            subject=self._map_subject(question_data.get("subject", "")),
            difficulty_level=self._map_difficulty(question_data.get("difficulty", "medium"))
        )
        
        # Get solution from math agent
        solution = await self.math_agent.solve_math_problem(request)
        
        # Extract answer from solution
        extracted_answer = self._extract_answer_from_solution(solution)
        
        # Check correctness
        is_correct = extracted_answer.upper() == correct_answer.upper()
        
        return {
            "question_id": question_data.get("id", question_num),
            "question": question_text,
            "options": options,
            "expected_answer": correct_answer,
            "generated_answer": solution.final_answer,
            "extracted_answer": extracted_answer,
            "is_correct": is_correct,
            "confidence_score": solution.confidence_score,
            "source": solution.source.value,
            "steps_count": len(solution.steps),
            "steps": [step.description for step in solution.steps]
        }
    
    def _extract_answer_from_solution(self, solution) -> str:
        """Extract single letter answer from solution using multiple strategies"""
        
        # Combine all solution text
        full_text = f"{solution.final_answer} {' '.join([step.description for step in solution.steps])}"
        full_text = full_text.upper()
        
        logger.info(f"Extracting answer from: {full_text[:200]}...")
        
        # Strategy 1: Pattern matching
        for pattern in self.answer_patterns:
            match = re.search(pattern, full_text, re.IGNORECASE | re.MULTILINE)
            if match:
                answer = match.group(1).upper()
                if answer in ['A', 'B', 'C', 'D']:
                    logger.info(f"Pattern match found: {answer}")
                    return answer
        
        # Strategy 2: Look for isolated letters at end
        # Find all A, B, C, D occurrences
        letter_matches = re.findall(r'\b([ABCD])\b', full_text)
        if letter_matches:
            # Take the last occurrence (most likely to be final answer)
            last_letter = letter_matches[-1]
            logger.info(f"Last letter found: {last_letter}")
            return last_letter
        
        # Strategy 3: Look in final_answer specifically
        final_answer_clean = re.sub(r'[^ABCD]', '', solution.final_answer.upper())
        if len(final_answer_clean) == 1 and final_answer_clean in 'ABCD':
            logger.info(f"Single letter in final answer: {final_answer_clean}")
            return final_answer_clean
        
        # Strategy 4: Look for common answer phrases
        answer_phrases = {
            'OPTION A': 'A', 'CHOICE A': 'A', 'A)': 'A', '(A)': 'A',
            'OPTION B': 'B', 'CHOICE B': 'B', 'B)': 'B', '(B)': 'B', 
            'OPTION C': 'C', 'CHOICE C': 'C', 'C)': 'C', '(C)': 'C',
            'OPTION D': 'D', 'CHOICE D': 'D', 'D)': 'D', '(D)': 'D'
        }
        
        for phrase, letter in answer_phrases.items():
            if phrase in full_text:
                logger.info(f"Found answer phrase '{phrase}' -> {letter}")
                return letter
        
        # Strategy 5: If multiple letters found, use heuristics
        if letter_matches and len(letter_matches) > 1:
            # Prefer letters that appear near "answer", "correct", "final"
            answer_context = r'(?:ANSWER|CORRECT|FINAL|THEREFORE|HENCE).{0,20}([ABCD])'
            context_match = re.search(answer_context, full_text)
            if context_match:
                logger.info(f"Context-based answer: {context_match.group(1)}")
                return context_match.group(1)
        
        logger.warning("No clear answer found, returning UNKNOWN")
        return "UNKNOWN"
    
    def _load_questions(self) -> List[Dict[str, Any]]:
        """Load JEE questions from file"""
        try:
            with open(self.questions_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            # Handle different JSON structures
            if isinstance(data, list):
                return data
            elif isinstance(data, dict) and "questions" in data:
                return data["questions"]
            else:
                logger.error(f"Unexpected JSON structure in {self.questions_file}")
                return []
                
        except Exception as e:
            logger.error(f"Error loading questions from {self.questions_file}: {e}")
            return []
    
    def _map_subject(self, subject: str) -> Optional[QuestionType]:
        """Map subject string to QuestionType enum"""
        subject_mapping = {
            "algebra": QuestionType.ALGEBRA,
            "calculus": QuestionType.CALCULUS,
            "geometry": QuestionType.GEOMETRY,
            "trigonometry": QuestionType.TRIGONOMETRY,
            "statistics": QuestionType.STATISTICS,
            "probability": QuestionType.STATISTICS,
            "linear_algebra": QuestionType.LINEAR_ALGEBRA,
            "number_theory": QuestionType.NUMBER_THEORY,
            "combinatorics": QuestionType.COMBINATORICS
        }
        
        return subject_mapping.get(subject.lower(), QuestionType.ALGEBRA)
    
    def _map_difficulty(self, difficulty: str) -> int:
        """Map difficulty string to integer"""
        difficulty_mapping = {
            "easy": 3,
            "medium": 5, 
            "hard": 8,
            "very_hard": 10
        }
        
        return difficulty_mapping.get(difficulty.lower(), 5)
    
    def _print_debug_info(self, result: Dict[str, Any]):
        """Print debug information for a result"""
        logger.info("=== SOLUTION DEBUG ===")
        logger.info(f"Final Answer: '{result['generated_answer']}'")
        logger.info(f"Confidence: {result['confidence_score']}")
        logger.info(f"Source: {result['source']}")
        logger.info(f"Steps Count: {result['steps_count']}")
        logger.info("Steps:")
        for i, step in enumerate(result['steps'][:3], 1):  # Show first 3 steps
            logger.info(f"  {i}. {step}")
        logger.info("=" * 22)
        logger.info(f"    Generated: '{result['generated_answer']}'")
        logger.info(f"    Expected: '{result['expected_answer']}'") 
        logger.info(f"    Extracted: '{result['extracted_answer']}'")
    
    def _save_results(self, summary: Dict[str, Any]):
        """Save benchmark results to file"""
        try:
            results_dir = Path(__file__).parent / "results"
            results_dir.mkdir(exist_ok=True)
            
            results_file = results_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
            logger.info(f"Results saved to {results_file}")
            
            # Also save debug results
            debug_file = results_dir / "debug_benchmark_results.json"
            with open(debug_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
                
        except Exception as e:
            logger.error(f"Error saving results: {e}")

def run():
    """Main entry point for JEE benchmark"""
    import argparse
    
    parser = argparse.ArgumentParser(description="JEE Benchmark Runner")
    parser.add_argument("--limit", type=int, help="Limit number of questions to process")
    parser.add_argument("--questions-file", type=str, help="Path to questions JSON file")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('benchmark.log')
        ]
    )
    
    # Run benchmark
    try:
        runner = JEEBenchmarkRunner(questions_file=args.questions_file)
        results = asyncio.run(runner.run_benchmark(limit=args.limit))
        
        print(f"\n🎯 BENCHMARK COMPLETE")
        print(f"Accuracy: {results['accuracy']:.2f}%")
        print(f"Questions: {results['total_questions']}")
        print(f"Correct: {results['correct_answers']}")
        
        return results
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")
        raise

if __name__ == "__main__":
    run()