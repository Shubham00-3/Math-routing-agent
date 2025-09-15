import dspy
import logging
from typing import List, Dict, Any, Optional, Tuple
import json
import asyncio
from datetime import datetime
import numpy as np

from app.models.schemas import SolutionResponse, Step, FeedbackRequest, QuestionType
from app.core.config import settings

logger = logging.getLogger(__name__)

class MathSolutionSignature(dspy.Signature):
    """Signature for mathematical solution generation with feedback optimization"""
    
    question = dspy.InputField(desc="Mathematical question to solve")
    context = dspy.InputField(desc="Additional context from knowledge base or web search")
    feedback_history = dspy.InputField(desc="Previous feedback and improvements", default="")
    subject = dspy.InputField(desc="Mathematical subject area", default="general")
    
    solution_steps = dspy.OutputField(desc="Step-by-step solution as JSON array")
    final_answer = dspy.OutputField(desc="Final answer to the mathematical problem")
    confidence = dspy.OutputField(desc="Confidence score between 0.0 and 1.0")
    educational_notes = dspy.OutputField(desc="Educational tips and insights")

class FeedbackOptimizedMathSolver(dspy.Module):
    """DSPy module for mathematical problem solving optimized with human feedback"""
    
    def __init__(self):
        super().__init__()
        self.generate_solution = dspy.ChainOfThought(MathSolutionSignature)
        self.feedback_weight = 0.3  # Weight for feedback in optimization
        
    def forward(self, question, context="", feedback_history="", subject="general"):
        """Generate solution using DSPy with feedback context"""
        
        # Enhanced context with feedback integration
        enhanced_context = self._integrate_feedback_context(context, feedback_history)
        
        # Generate solution
        result = self.generate_solution(
            question=question,
            context=enhanced_context,
            feedback_history=feedback_history,
            subject=subject
        )
        
        return dspy.Prediction(
            solution_steps=result.solution_steps,
            final_answer=result.final_answer,
            confidence=result.confidence,
            educational_notes=result.educational_notes
        )
    
    def _integrate_feedback_context(self, context: str, feedback_history: str) -> str:
        """Integrate feedback history into solution context"""
        if not feedback_history:
            return context
        
        feedback_prompt = f"""
Previous Feedback and Improvements:
{feedback_history}

Please incorporate these learnings into your solution approach.
Focus on areas that received negative feedback in the past.

Original Context:
{context}
"""
        return feedback_prompt

class DSPyFeedbackOptimizer:
    """DSPy-based optimizer for improving solutions based on human feedback"""
    
    def __init__(self):
        # Initialize DSPy with your LLM
        if settings.LLM_MODEL.startswith("gpt"):
            dspy.settings.configure(
                lm=dspy.OpenAI(
                    model=settings.LLM_MODEL,
                    api_key=settings.OPENAI_API_KEY,
                    max_tokens=settings.MAX_TOKENS
                )
            )
        elif settings.LLM_MODEL.startswith("claude"):
            # Note: DSPy might not have direct Claude support, you may need to create a custom LM
            dspy.settings.configure(
                lm=self._create_claude_lm()
            )
        
        self.solver = FeedbackOptimizedMathSolver()
        self.training_data = []
        self.optimization_history = []
        
    def _create_claude_lm(self):
        """Create custom Claude language model for DSPy"""
        # This would need to be implemented as a custom DSPy LM
        # For now, fallback to OpenAI
        return dspy.OpenAI(
            model="gpt-4",
            api_key=settings.OPENAI_API_KEY,
            max_tokens=settings.MAX_TOKENS
        )

    async def optimize_with_feedback(
        self,
        feedback_data: List[Dict[str, Any]],
        optimization_metric: str = "overall_rating"
    ) -> Dict[str, Any]:
        """Optimize the math solver using collected feedback"""
        try:
            logger.info(f"Starting DSPy optimization with {len(feedback_data)} feedback samples")
            
            # Prepare training examples from feedback
            training_examples = await self._prepare_training_examples(feedback_data)
            
            if len(training_examples) < 5:
                logger.warning("Insufficient training data for optimization")
                return {"status": "insufficient_data", "examples_count": len(training_examples)}
            
            # Create evaluation metric based on feedback
            evaluation_metric = self._create_feedback_metric(optimization_metric)
            
            # Optimize using DSPy
            optimizer = dspy.BootstrapFewShot(
                metric=evaluation_metric,
                max_bootstrapped_demos=min(10, len(training_examples)),
                max_labeled_demos=min(5, len(training_examples) // 2)
            )
            
            # Compile optimized module
            optimized_solver = optimizer.compile(
                self.solver,
                trainset=training_examples[:20],  # Limit training set size
                valset=training_examples[20:25] if len(training_examples) > 20 else training_examples[:5]
            )
            
            # Update the solver with optimized version
            self.solver = optimized_solver
            
            # Evaluate improvement
            improvement_metrics = await self._evaluate_optimization(
                training_examples, evaluation_metric
            )
            
            optimization_result = {
                "status": "success",
                "examples_used": len(training_examples),
                "improvement_metrics": improvement_metrics,
                "optimization_timestamp": datetime.utcnow().isoformat()
            }
            
            self.optimization_history.append(optimization_result)
            logger.info("DSPy optimization completed successfully")
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in DSPy optimization: {str(e)}")
            return {"status": "error", "error": str(e)}

    async def _prepare_training_examples(
        self, 
        feedback_data: List[Dict[str, Any]]
    ) -> List[dspy.Example]:
        """Convert feedback data into DSPy training examples"""
        examples = []
        
        for feedback_item in feedback_data:
            try:
                # Extract solution and feedback information
                solution = feedback_item.get("solution", {})
                feedback = feedback_item.get("feedback", {})
                
                # Create example with feedback context
                example = dspy.Example(
                    question=solution.get("question", ""),
                    context=self._format_context(feedback_item.get("context", {})),
                    feedback_history=self._format_feedback_history(feedback),
                    subject=solution.get("subject", "general"),
                    solution_steps=json.dumps(solution.get("steps", [])),
                    final_answer=solution.get("final_answer", ""),
                    confidence=str(solution.get("confidence_score", 0.5)),
                    educational_notes=solution.get("educational_notes", ""),
                    feedback_rating=feedback.get("rating", 3)
                ).with_inputs("question", "context", "feedback_history", "subject")
                
                examples.append(example)
                
            except Exception as e:
                logger.warning(f"Error processing feedback item: {str(e)}")
                continue
        
        return examples

    def _format_context(self, context_data: Dict[str, Any]) -> str:
        """Format context data for DSPy training"""
        if not context_data:
            return ""
        
        formatted_parts = []
        
        if "knowledge_base" in context_data:
            kb_data = context_data["knowledge_base"]
            formatted_parts.append(f"Knowledge Base: {kb_data.get('question', '')} -> {kb_data.get('final_answer', '')}")
        
        if "search_results" in context_data:
            search_results = context_data["search_results"][:2]  # Limit to first 2 results
            for i, result in enumerate(search_results, 1):
                formatted_parts.append(f"Search Result {i}: {result.get('title', '')} - {result.get('content', '')[:200]}")
        
        return "\n".join(formatted_parts)

    def _format_feedback_history(self, feedback: Dict[str, Any]) -> str:
        """Format feedback history for training context"""
        if not feedback:
            return ""
        
        feedback_text = f"Previous Feedback - Rating: {feedback.get('rating', 'N/A')}/5\n"
        
        if feedback.get("comments"):
            feedback_text += f"Comments: {feedback.get('comments')}\n"
        
        if feedback.get("improvement_suggestions"):
            feedback_text += f"Suggestions: {feedback.get('improvement_suggestions')}\n"
        
        feedback_text += f"Feedback Type: {feedback.get('feedback_type', 'general')}"
        
        return feedback_text

    def _create_feedback_metric(self, optimization_metric: str):
        """Create evaluation metric based on feedback"""
        
        def feedback_metric(example, pred, trace=None):
            """Evaluate prediction quality based on feedback patterns"""
            try:
                # Base score from feedback rating
                feedback_rating = getattr(example, 'feedback_rating', 3)
                base_score = feedback_rating / 5.0
                
                # Bonus for generating steps (completeness)
                try:
                    steps = json.loads(pred.solution_steps)
                    step_bonus = min(len(steps) * 0.1, 0.3)
                except:
                    step_bonus = 0
                
                # Confidence calibration
                try:
                    confidence = float(pred.confidence)
                    confidence_bonus = 0.1 if 0.7 <= confidence <= 0.9 else 0
                except:
                    confidence_bonus = 0
                
                # Educational value (presence of educational notes)
                edu_bonus = 0.1 if pred.educational_notes and len(pred.educational_notes) > 50 else 0
                
                final_score = base_score + step_bonus + confidence_bonus + edu_bonus
                return min(final_score, 1.0)
                
            except Exception as e:
                logger.warning(f"Error in feedback metric calculation: {str(e)}")
                return 0.5
        
        return feedback_metric

    async def _evaluate_optimization(
        self, 
        examples: List[dspy.Example], 
        metric
    ) -> Dict[str, float]:
        """Evaluate optimization improvement"""
        if not examples:
            return {"improvement": 0.0}
        
        # Sample evaluation on a subset
        eval_examples = examples[:5]
        scores = []
        
        for example in eval_examples:
            try:
                pred = self.solver(
                    question=example.question,
                    context=example.context,
                    feedback_history=example.feedback_history,
                    subject=example.subject
                )
                score = metric(example, pred)
                scores.append(score)
            except Exception as e:
                logger.warning(f"Error evaluating example: {str(e)}")
                scores.append(0.5)
        
        return {
            "average_score": np.mean(scores),
            "score_std": np.std(scores),
            "samples_evaluated": len(scores)
        }

    async def generate_optimized_solution(
        self,
        question: str,
        context: str = "",
        feedback_history: str = "",
        subject: str = "general"
    ) -> Dict[str, Any]:
        """Generate solution using the optimized DSPy model"""
        try:
            # Use the optimized solver
            result = self.solver(
                question=question,
                context=context,
                feedback_history=feedback_history,
                subject=subject
            )
            
            # Parse and format the result
            solution_data = {
                "steps": self._parse_solution_steps(result.solution_steps),
                "final_answer": result.final_answer,
                "confidence_score": float(result.confidence) if result.confidence else 0.7,
                "educational_notes": result.educational_notes,
                "optimization_used": True,
                "generation_timestamp": datetime.utcnow().isoformat()
            }
            
            return solution_data
            
        except Exception as e:
            logger.error(f"Error generating optimized solution: {str(e)}")
            return {
                "error": str(e),
                "optimization_used": False
            }

    def _parse_solution_steps(self, steps_json: str) -> List[Dict[str, Any]]:
        """Parse solution steps from JSON string"""
        try:
            steps_data = json.loads(steps_json)
            
            # Ensure proper step format
            formatted_steps = []
            for i, step in enumerate(steps_data, 1):
                if isinstance(step, dict):
                    formatted_steps.append({
                        "step_number": step.get("step_number", i),
                        "description": step.get("description", ""),
                        "explanation": step.get("explanation", ""),
                        "formula": step.get("formula", None),
                        "visual_aid": step.get("visual_aid", None)
                    })
                elif isinstance(step, str):
                    formatted_steps.append({
                        "step_number": i,
                        "description": step,
                        "explanation": "",
                        "formula": None,
                        "visual_aid": None
                    })
            
            return formatted_steps
            
        except json.JSONDecodeError:
            logger.warning("Failed to parse solution steps JSON")
            return [{
                "step_number": 1,
                "description": steps_json,
                "explanation": "",
                "formula": None,
                "visual_aid": None
            }]
