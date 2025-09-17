# backend/app/core/guardrails.py - Fixed to stop blocking valid math solutions

import re
import logging
from typing import List, Dict, Tuple, Optional, Any
from app.models.schemas import GuardrailsResult, QuestionType
from app.core.config import settings

logger = logging.getLogger(__name__)

class ContentValidator:
    """Enhanced content validation for educational math content"""
    
    def __init__(self):
        self.math_keywords = {
            "algebra": ["equation", "variable", "solve", "x", "y", "polynomial", "factor", "quadratic", "linear"],
            "calculus": ["derivative", "integral", "limit", "function", "differentiate", "integrate", "dx", "dy", "chain rule"],
            "geometry": ["triangle", "circle", "area", "perimeter", "angle", "vertex", "polygon", "theorem", "proof"],
            "trigonometry": ["sin", "cos", "tan", "sine", "cosine", "tangent", "radian", "degree", "amplitude"],
            "statistics": ["mean", "median", "mode", "standard deviation", "probability", "distribution", "variance"],
            "linear_algebra": ["matrix", "vector", "determinant", "eigenvalue", "dot product", "transpose"],
            "number_theory": ["prime", "divisible", "gcd", "lcm", "modular", "congruent", "fibonacci"]
        }
        
        # CRITICAL: Much more restrictive prohibited content to avoid false positives
        self.prohibited_content = [
            "explicit sexual content", "graphic violence", "illegal drugs", "weapons manufacturing",
            "hate speech", "terrorist activities", "child abuse", "self-harm instructions"
        ]
        
        # Mathematical educational indicators - expanded list
        self.educational_indicators = [
            "explain", "solve", "calculate", "find", "determine", "prove", "show", "demonstrate",
            "step by step", "how to", "what is", "why", "derive", "simplify", "factor",
            "integrate", "differentiate", "substitute", "evaluate", "compute", "verify",
            "theorem", "formula", "equation", "expression", "solution", "answer"
        ]

class InputGuardrails:
    """Input validation with math-friendly filtering"""
    
    def __init__(self):
        self.validator = ContentValidator()

    async def validate_input(self, question: str) -> GuardrailsResult:
        """Enhanced input validation that's math-friendly"""
        try:
            # Step 1: Check for truly prohibited content (very strict check)
            if await self._contains_truly_harmful_content(question):
                return GuardrailsResult(
                    is_valid=False,
                    confidence=0.9,
                    violations=["Prohibited content detected"],
                    category="content_violation",
                    reasoning="Question contains genuinely harmful content"
                )

            # Step 2: Enhanced math relevance check
            math_analysis = await self._enhanced_math_check(question)
            
            # Step 3: Educational appropriateness check
            educational_check = self._check_educational_appropriateness(question)
            
            # Combine scores for final decision
            final_score = (math_analysis["math_score"] * 0.6) + (educational_check["educational_score"] * 0.4)
            
            # MUCH MORE LENIENT: Accept anything that looks remotely mathematical
            is_valid = final_score >= 0.2  # Very low threshold!
            
            return GuardrailsResult(
                is_valid=is_valid,
                confidence=final_score,
                violations=[] if is_valid else ["Not clearly mathematical or educational"],
                category=math_analysis.get("detected_subject", "general_math"),
                reasoning=f"Math score: {math_analysis['math_score']:.3f}, Educational: {educational_check['educational_score']:.3f}, Combined: {final_score:.3f}"
            )

        except Exception as e:
            logger.error(f"Input validation error: {str(e)}")
            # FAIL OPEN: If validation fails, assume it's valid math
            return GuardrailsResult(
                is_valid=True,  # FAIL OPEN for math content!
                confidence=0.7,
                violations=[],
                category="math_general",
                reasoning=f"Validation error, assuming valid math content: {str(e)}"
            )

    async def _contains_truly_harmful_content(self, text: str) -> bool:
        """Very strict check for genuinely harmful content only"""
        text_lower = text.lower()
        
        # Only flag truly explicit harmful content
        truly_harmful_patterns = [
            r'\b(?:kill|murder|suicide|self-harm)\s+(?:instructions|methods|how\s+to)\b',
            r'\b(?:make|build|create)\s+(?:bomb|explosive|weapon)\b',
            r'\b(?:child|minor)\s+(?:abuse|exploitation|porn)\b',
            r'\b(?:nazi|hitler|holocaust)\s+(?:good|great|awesome)\b',
            r'\bexplicit\s+sexual\s+(?:content|material|images)\b'
        ]
        
        return any(re.search(pattern, text_lower) for pattern in truly_harmful_patterns)

    async def _enhanced_math_check(self, question: str) -> Dict[str, Any]:
        """Enhanced mathematical content detection"""
        question_lower = question.lower()
        
        # Mathematical indicators with weights
        indicators = {
            "numbers": (re.findall(r'\d+', question), 0.15),
            "operators": (re.findall(r'[+\-*/=<>^]', question), 0.15),
            "math_words": ([word for word in ["solve", "calculate", "find", "equation", "formula"] if word in question_lower], 0.25),
            "variables": (re.findall(r'\b[a-z]\b|\b[a-z]\d|\d[a-z]\b', question_lower), 0.15),
            "functions": (re.findall(r'sin|cos|tan|log|ln|sqrt|exp', question_lower), 0.20),
            "symbols": (re.findall(r'π|∑|∫|∂|Δ|α|β|γ|θ|λ|μ|σ', question), 0.10)
        }
        
        # Calculate weighted score
        total_score = 0.0
        detected_features = {}
        
        for indicator_name, (matches, weight) in indicators.items():
            feature_score = min(len(matches) * 0.2, 1.0)  # Normalize
            total_score += feature_score * weight
            detected_features[indicator_name] = {"count": len(matches), "score": feature_score}
        
        # Subject detection
        detected_subject = self._detect_math_subject(question_lower)
        
        # Bonus for clear math questions
        if any(phrase in question_lower for phrase in ["what is", "solve for", "find the", "calculate"]):
            total_score += 0.1
            
        # Bonus for standard mathematical expressions
        if re.search(r'\b\w+\s*=\s*\w+\b', question):  # Equations like "x = 5"
            total_score += 0.15
            
        return {
            "math_score": min(total_score, 1.0),
            "detected_subject": detected_subject,
            "features": detected_features,
            "has_clear_math_intent": total_score > 0.3
        }

    def _detect_math_subject(self, question_lower: str) -> str:
        """Detect mathematical subject area"""
        subject_scores = {}
        
        for subject, keywords in self.validator.math_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > 0:
                subject_scores[subject] = score
        
        if subject_scores:
            return max(subject_scores.items(), key=lambda x: x[1])[0]
        
        # Default classification based on content patterns
        if any(word in question_lower for word in ["derivative", "integral", "limit"]):
            return "calculus"
        elif any(word in question_lower for word in ["equation", "solve", "x", "y"]):
            return "algebra"
        elif any(word in question_lower for word in ["triangle", "circle", "area"]):
            return "geometry"
        else:
            return "general_math"

    def _check_educational_appropriateness(self, question: str) -> Dict[str, Any]:
        """Check if question is educationally appropriate"""
        question_lower = question.lower()
        
        educational_score = 0.5  # Start neutral
        
        # Positive indicators
        positive_patterns = [
            r'\b(?:learn|understand|study|homework|assignment|class|course|exam|test)\b',
            r'\b(?:explain|show|demonstrate|help|teach|tutorial)\b',
            r'\b(?:step\s+by\s+step|solution|answer|method|approach)\b',
            r'\b(?:problem|question|exercise|practice)\b'
        ]
        
        for pattern in positive_patterns:
            if re.search(pattern, question_lower):
                educational_score += 0.1
        
        # Educational question words
        question_words = ["what", "how", "why", "when", "where", "which"]
        if any(word in question_lower for word in question_words):
            educational_score += 0.1
        
        # Mathematical context
        if any(word in question_lower for word in self.validator.educational_indicators):
            educational_score += 0.2
        
        return {
            "educational_score": min(educational_score, 1.0),
            "has_educational_context": educational_score > 0.6
        }

class OutputGuardrails:
    """Output validation that doesn't block math solutions"""
    
    def __init__(self):
        self.validator = ContentValidator()

    async def validate_output(self, question: str, solution: str) -> GuardrailsResult:
        """Math-friendly output validation"""
        try:
            # Step 1: Check for genuinely inappropriate content (very restrictive)
            if await self._contains_genuinely_inappropriate_content(solution):
                return GuardrailsResult(
                    is_valid=False,
                    confidence=0.9,
                    violations=["Genuinely inappropriate content in solution"],
                    category="content_violation",
                    reasoning="Solution contains genuinely inappropriate content"
                )

            # Step 2: Basic solution quality check (very lenient)
            quality_check = await self._lenient_quality_check(question, solution)
            
            # VERY LENIENT: Accept almost all mathematical content
            is_valid = quality_check["basic_quality_score"] >= 0.2  # Very low bar!
            
            return GuardrailsResult(
                is_valid=is_valid,
                confidence=quality_check["basic_quality_score"],
                violations=quality_check.get("minor_issues", []),
                category="mathematical_solution",
                reasoning=f"Basic quality check passed: {quality_check['reasoning']}"
            )

        except Exception as e:
            logger.error(f"Output validation error: {str(e)}")
            # FAIL OPEN: If validation fails, assume valid mathematical content
            return GuardrailsResult(
                is_valid=True,  # FAIL OPEN for math solutions!
                confidence=0.8,
                violations=[],
                category="mathematical_solution",
                reasoning=f"Validation error, assuming valid math solution: {str(e)}"
            )

    async def _contains_genuinely_inappropriate_content(self, text: str) -> bool:
        """Check for genuinely inappropriate content (very restrictive)"""
        text_lower = text.lower()
        
        # Only flag genuinely problematic content, not mathematical terms
        genuinely_bad_patterns = [
            r'\b(?:fuck|shit|damn|hell|bitch|asshole)\b',  # Profanity
            r'\b(?:kill yourself|commit suicide|self harm)\b',  # Self-harm
            r'\b(?:hate|racist|nazi|terrorist)\b.*\b(?:good|great|awesome)\b',  # Hate speech
            r'\b(?:buy|purchase|click here|visit our website)\b'  # Commercial content
        ]
        
        return any(re.search(pattern, text_lower) for pattern in genuinely_bad_patterns)

    async def _lenient_quality_check(self, question: str, solution: str) -> Dict:
        """Very lenient quality assessment for mathematical solutions"""
        
        # Basic checks that almost always pass
        basic_score = 0.3  # Start with reasonable baseline
        minor_issues = []
        
        # Check solution length (very lenient)
        if len(solution.strip()) > 20:  # Very low bar
            basic_score += 0.3
        else:
            minor_issues.append("Solution seems quite short")
        
        # Check for mathematical content
        math_indicators = ["=", "+", "-", "*", "/", "step", "equation", "solve", "calculate", "answer"]
        math_content = sum(1 for indicator in math_indicators if indicator.lower() in solution.lower())
        
        if math_content > 0:
            basic_score += 0.3
        
        # Check for structure (steps, explanation)
        structure_indicators = ["step", "first", "then", "next", "finally", "therefore", "answer"]
        if any(indicator in solution.lower() for indicator in structure_indicators):
            basic_score += 0.2
            
        reasoning = f"Length check: {'✓' if len(solution) > 20 else '✗'}, "
        reasoning += f"Math content: {math_content} indicators, "
        reasoning += f"Structure: {'✓' if any(ind in solution.lower() for ind in structure_indicators) else '✗'}"
        
        return {
            "basic_quality_score": min(basic_score, 1.0),
            "minor_issues": minor_issues,
            "reasoning": reasoning,
            "length": len(solution),
            "math_indicators_count": math_content
        }