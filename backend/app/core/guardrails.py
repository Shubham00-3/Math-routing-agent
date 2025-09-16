import re
import logging
from typing import List, Dict, Tuple, Optional
from langchain_community.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from app.models.schemas import GuardrailsResult, QuestionType
from app.core.config import settings
import httpx
import asyncio
from langchain_huggingface import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

class ContentValidator:
    """Content validation for educational math content"""
    
    def __init__(self):
        self.math_keywords = {
            "algebra": ["equation", "variable", "solve", "x", "y", "polynomial", "factor", "quadratic"],
            "calculus": ["derivative", "integral", "limit", "function", "differentiate", "integrate", "dx", "dy"],
            "geometry": ["triangle", "circle", "area", "perimeter", "angle", "vertex", "polygon", "theorem"],
            "trigonometry": ["sin", "cos", "tan", "sine", "cosine", "tangent", "radian", "degree"],
            "statistics": ["mean", "median", "mode", "standard deviation", "probability", "distribution"],
            "linear_algebra": ["matrix", "vector", "determinant", "eigenvalue", "dot product"],
            "number_theory": ["prime", "divisible", "gcd", "lcm", "modular", "congruent"]
        }
        
        self.prohibited_content = [
            "inappropriate", "violent", "harmful", "illegal", "adult", "nsfw",
            "politics", "religion", "discrimination", "hate", "offensive"
        ]
        
        self.educational_indicators = [
            "explain", "solve", "calculate", "find", "determine", "prove", "show",
            "step by step", "how to", "what is", "why", "derive", "simplify"
        ]

class InputGuardrails:
    """Input validation and filtering"""
    
    def __init__(self):
        self.validator = ContentValidator()
        self.llm = OpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.1,
            model_name="gpt-3.5-turbo"
        )
        self.classification_prompt = PromptTemplate(
            input_variables=["question"],
            template="""
            Analyze the following question and determine:
            1. Is this a mathematical/educational question? (Yes/No)
            2. What mathematical subject does it belong to? (algebra, calculus, geometry, etc.)
            3. Is the content appropriate for educational purposes? (Yes/No)
            4. Confidence score (0.0 to 1.0)
            5. Any content concerns or violations

            Question: {question}

            Respond in this format:
            Mathematical: [Yes/No]
            Subject: [subject]
            Appropriate: [Yes/No]
            Confidence: [0.0-1.0]
            Concerns: [list any concerns]
            Reasoning: [brief explanation]
            """
        )
        self.classification_chain = LLMChain(
            llm=self.llm,
            prompt=self.classification_prompt
        )

    async def validate_input(self, question: str) -> GuardrailsResult:
        """Validate input question against guardrails"""
        try:
            # Basic content filtering
            if await self._contains_prohibited_content(question):
                return GuardrailsResult(
                    is_valid=False,
                    confidence=0.9,
                    violations=["Prohibited content detected"],
                    category="content_violation",
                    reasoning="Question contains inappropriate content"
                )

            # # Math relevance check
            # math_relevance = await self._check_math_relevance(question)
            # if math_relevance < settings.MATH_RELEVANCE_THRESHOLD:
            #     return GuardrailsResult(
            #         is_valid=False,
            #         confidence=math_relevance,
            #         violations=["Not a mathematical question"],
            #         category="relevance_violation",
            #         reasoning="Question is not related to mathematics"
            #     )

            # LLM-based classification
            llm_result = await self._llm_classification(question)
            
            return GuardrailsResult(
                is_valid=llm_result["is_valid"],
                confidence=llm_result["confidence"],
                violations=llm_result["violations"],
                category=llm_result["category"],
                reasoning=llm_result["reasoning"]
            )

        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            return GuardrailsResult(
                is_valid=False,
                confidence=0.5,
                violations=["Validation error"],
                category="system_error",
                reasoning=f"System error during validation: {str(e)}"
            )

    async def _contains_prohibited_content(self, text: str) -> bool:
        """Check for prohibited content"""
        text_lower = text.lower()
        return any(prohibited in text_lower for prohibited in self.validator.prohibited_content)

    async def _check_math_relevance(self, question: str) -> float:
        """Calculate mathematical relevance score"""
        question_lower = question.lower()
        
        # Count math keywords
        math_keyword_count = 0
        total_keywords = 0
        
        for subject, keywords in self.validator.math_keywords.items():
            for keyword in keywords:
                total_keywords += 1
                if keyword in question_lower:
                    math_keyword_count += 1

        # Count educational indicators
        educational_count = sum(1 for indicator in self.validator.educational_indicators 
                                if indicator in question_lower)

        # Mathematical symbols and patterns
        math_symbols = len(re.findall(r'[+\-*/=<>∫∑∏√∞πθλΔ∂]', question))
        number_patterns = len(re.findall(r'\b\d+(?:\.\d+)?\b', question))
        
        # Boost score for basic arithmetic patterns
        if re.search(r'\d+\s*[+\-*/]\s*\d+', question):
            return 1.0  # Perfect score for basic arithmetic

        # Calculate relevance score
        keyword_score = math_keyword_count / max(total_keywords, 1) * 0.4
        educational_score = min(educational_count / 3, 1.0) * 0.3
        symbol_score = min(math_symbols / 5, 1.0) * 0.2
        number_score = min(number_patterns / 3, 1.0) * 0.1
        
        return keyword_score + educational_score + symbol_score + number_score

    async def _llm_classification(self, question: str) -> Dict:
        """Enhanced math detection without LLM"""
        question_lower = question.lower()
        
        # Math keywords and patterns
        math_keywords = [
            # Basic operations
            'solve', 'calculate', 'find', 'compute', 'determine',
            'what is', 'how much', 'how many', 'equal', 'equals',
            
            # Mathematical concepts
            'area', 'perimeter', 'volume', 'radius', 'diameter', 'circumference',
            'derivative', 'integral', 'limit', 'function', 'equation', 'formula',
            'triangle', 'circle', 'square', 'rectangle', 'polygon',
            'probability', 'statistics', 'mean', 'median', 'mode',
            'matrix', 'vector', 'algebra', 'geometry', 'calculus',
            
            # Mathematical symbols/expressions
            'x', 'y', 'z', 'sin', 'cos', 'tan', 'log', 'ln', 'sqrt', 'pi'
        ]
        
        # Check for math keywords
        has_math_keywords = any(keyword in question_lower for keyword in math_keywords)
        
        # Check for numbers
        has_numbers = bool(re.search(r'\d', question))
        
        # Check for math operators  
        has_operators = bool(re.search(r'[+\-*/=<>^]', question))
        
        # Check for variables (like "2x", "x^2", etc)
        has_variables = bool(re.search(r'[a-z]\s*[+\-*/=^]|[+\-*/=^]\s*[a-z]|\d+[a-z]|[a-z]\d+', question_lower))
        
        # Mathematical question patterns
        math_patterns = [
            r'solve\s+for\s+[a-z]',
            r'find\s+the\s+(area|volume|perimeter)',
            r'what\s+is\s+\d+.*[+\-*/].*\d+',
            r'calculate.*\d+',
            r'derivative\s+of',
            r'integral\s+of'
        ]
        
        has_math_patterns = any(re.search(pattern, question_lower) for pattern in math_patterns)
        
        # Score the question
        score = 0
        if has_math_keywords: score += 0.4
        if has_numbers: score += 0.2  
        if has_operators: score += 0.3
        if has_variables: score += 0.3
        if has_math_patterns: score += 0.5
        
        is_mathematical = score >= 0.3  # Lower threshold
        
        return {
            "is_valid": is_mathematical,
            "confidence": min(score, 1.0),
            "violations": [] if is_mathematical else ["Not detected as mathematical"],
            "category": "algebra" if is_mathematical else "unknown",
            "reasoning": f"Math detection score: {score:.2f}"
        }

class OutputGuardrails:
    """Output validation and filtering"""
    
    def __init__(self):
        self.validator = ContentValidator()
        self.llm = OpenAI(
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.1
        )
        self.quality_prompt = PromptTemplate(
            input_variables=["question", "solution"],
            template="""
            Evaluate the quality of this mathematical solution:
            
            Question: {question}
            Solution: {solution}
            
            Rate the solution on:
            1. Mathematical accuracy (0-1)
            2. Step-by-step clarity (0-1)
            3. Educational value (0-1)
            4. Completeness (0-1)
            5. Appropriateness for students (0-1)
            
            Also identify any issues:
            - Mathematical errors
            - Missing steps
            - Unclear explanations
            - Inappropriate content
            
            Format:
            Accuracy: [0.0-1.0]
            Clarity: [0.0-1.0]
            Educational: [0.0-1.0]
            Completeness: [0.0-1.0]
            Appropriate: [0.0-1.0]
            Overall: [0.0-1.0]
            Issues: [list any issues]
            Reasoning: [explanation]
            """
        )
        self.quality_chain = LLMChain(llm=self.llm, prompt=self.quality_prompt)

    async def validate_output(self, question: str, solution: str) -> GuardrailsResult:
        """Validate solution output"""
        try:
            # Basic content check
            if await self._contains_inappropriate_content(solution):
                return GuardrailsResult(
                    is_valid=False,
                    confidence=0.9,
                    violations=["Inappropriate content in solution"],
                    category="content_violation",
                    reasoning="Solution contains inappropriate content"
                )

            # Quality assessment
            quality_result = await self._assess_solution_quality(question, solution)
            
            is_valid = quality_result["overall_score"] >= settings.CONTENT_FILTER_THRESHOLD
            
            return GuardrailsResult(
                is_valid=is_valid,
                confidence=quality_result["overall_score"],
                violations=quality_result["issues"],
                category="quality_assessment",
                reasoning=quality_result["reasoning"]
            )

        except Exception as e:
            logger.error(f"Output validation error: {str(e)}")
            return GuardrailsResult(
                is_valid=False,
                confidence=0.5,
                violations=["Validation error"],
                category="system_error",
                reasoning=f"Error during output validation: {str(e)}"
            )

    async def _contains_inappropriate_content(self, text: str) -> bool:
        """Check solution for inappropriate content"""
        text_lower = text.lower()
        
        # Check for prohibited content
        if any(prohibited in text_lower for prohibited in self.validator.prohibited_content):
            return True
            
        # Check for non-educational language patterns
        non_educational_patterns = [
            r'\b(buy|sell|purchase|advertisement|ad)\b',
            r'\b(click here|visit|website|url)\b',
            r'\b(personal|private|contact|email|phone)\b'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in non_educational_patterns)

    async def _assess_solution_quality(self, question: str, solution: str) -> Dict:
        """Simple quality assessment without OpenAI"""
        # Basic quality checks
        has_steps = "step" in solution.lower()
        has_answer = len(solution) > 100
        has_explanation = any(word in solution.lower() for word in ['because', 'explanation', 'formula', 'calculate'])

        score = 0.7  # Base score
        if has_steps: score += 0.1
        if has_answer: score += 0.1  
        if has_explanation: score += 0.1

        return {
            "overall_score": min(score, 1.0),
            "detailed_scores": {"completeness": score},
            "issues": [],
            "reasoning": "Basic content analysis - no API costs"
        }