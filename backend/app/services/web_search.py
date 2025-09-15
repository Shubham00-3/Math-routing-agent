
# import logging
# from typing import List, Dict, Any, Optional, Union
# import asyncio
# import time
# import json
# import re
# import uuid
# from datetime import datetime
# import httpx

# from app.models.schemas import SolutionResponse, Step, QuestionType, SourceType
# from app.core.config import settings

# logger = logging.getLogger(__name__)

# class MathLLMService:
#     """LLM service supporting Groq, Gemini, and OpenAI"""
    
#     def __init__(self):
#         self.provider = getattr(settings, 'LLM_PROVIDER', 'groq').lower()
#         self.model = settings.LLM_MODEL
        
#         # Initialize based on provider
#         if self.provider == 'groq':
#             self.api_key = settings.GROQ_API_KEY
#             self.base_url = "https://api.groq.com/openai/v1"
#         elif self.provider == 'gemini':
#             self.api_key = settings.GOOGLE_API_KEY
#             self.base_url = "https://generativelanguage.googleapis.com/v1beta"
#         elif self.provider == 'openai':
#             self.api_key = settings.OPENAI_API_KEY
#             self.base_url = "https://api.openai.com/v1"
#         else:
#             raise ValueError(f"Unsupported LLM provider: {self.provider}")
        
#         logger.info(f"Initialized LLM service with provider: {self.provider}")
        
#         # System prompts for different scenarios
#         self.system_prompts = self._create_system_prompts()
    
#     def _create_system_prompts(self) -> Dict[str, str]:
#         """Create system prompts for different scenarios"""
#         return {
#             "knowledge_base": """You are a mathematical professor helping students learn mathematics. 
# You have been provided with a relevant solution from the knowledge base. 
# Your task is to:
# 1. Verify the solution is correct and complete
# 2. Enhance the explanation to make it clearer for students
# 3. Add educational insights and tips
# 4. Format the solution in clear, numbered steps
# 5. Ensure the final answer is clearly stated

# Focus on pedagogy - make complex concepts accessible while maintaining mathematical rigor.
# Always explain the reasoning behind each step.""",

#             "web_search": """You are a mathematical professor creating step-by-step solutions.
# You have been provided with web search results about a mathematical problem.
# Your task is to:
# 1. Synthesize information from the search results
# 2. Create a clear, educational step-by-step solution
# 3. Verify mathematical accuracy
# 4. Add explanations for each step
# 5. Include relevant formulas and concepts
# 6. Provide the final answer clearly

# If the search results are inconsistent or unclear, note this and provide the most likely correct solution with appropriate caveats.
# Always prioritize educational value and clarity.""",

#             "standalone": """You are a mathematical professor solving a problem without external resources.
# Your task is to:
# 1. Analyze the mathematical problem carefully
# 2. Create a step-by-step solution using your mathematical knowledge
# 3. Explain each step clearly for student understanding
# 4. Include relevant mathematical concepts and formulas
# 5. Provide the final answer
# 6. Note if you're uncertain about any aspect

# Be honest about limitations and suggest verification methods if needed."""
#         }

#     async def generate_solution(
#         self,
#         question: str,
#         context_data: Dict[str, Any],
#         source_type: SourceType,
#         subject: Optional[QuestionType] = None
#     ) -> SolutionResponse:
#         """Generate mathematical solution based on context and source type"""
#         try:
#             start_time = time.time()
            
#             # Select appropriate system prompt
#             system_prompt = self.system_prompts.get(source_type.value, self.system_prompts["standalone"])
            
#             # Prepare context
#             context_str = self._format_context(context_data, source_type)
            
#             # Create the full prompt
#             full_prompt = self._create_full_prompt(question, context_str, subject, system_prompt)
            
#             # Generate response based on provider
#             if self.provider == 'groq':
#                 response_text = await self._call_groq(full_prompt)
#             elif self.provider == 'gemini':
#                 response_text = await self._call_gemini(full_prompt)
#             elif self.provider == 'openai':
#                 response_text = await self._call_openai(full_prompt)
#             else:
#                 raise ValueError(f"Unsupported provider: {self.provider}")
            
#             # Parse response into structured solution
#             parsed_solution = await self._parse_solution_response(
#                 response_text, question, source_type, subject
#             )
            
#             # Calculate processing time
#             processing_time = time.time() - start_time
#             parsed_solution.processing_time = processing_time
            
#             logger.info(f"Generated solution using {self.provider} in {processing_time:.2f}s")
#             return parsed_solution
            
#         except Exception as e:
#             logger.error(f"Error generating solution: {str(e)}")
#             return self._create_error_solution(question, str(e), source_type, subject)

#     async def _call_groq(self, prompt: str) -> str:
#         """Call Groq API"""
#         try:
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(
#                     f"{self.base_url}/chat/completions",
#                     headers={
#                         "Authorization": f"Bearer {self.api_key}",
#                         "Content-Type": "application/json"
#                     },
#                     json={
#                         "model": self.model.replace("groq/", ""),  # Remove prefix
#                         "messages": [
#                             {"role": "user", "content": prompt}
#                         ],
#                         "temperature": settings.LLM_TEMPERATURE,
#                         "max_tokens": settings.MAX_TOKENS,
#                         "top_p": 1,
#                         "stream": False
#                     },
#                     timeout=30.0
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     return result["choices"][0]["message"]["content"]
#                 else:
#                     raise Exception(f"Groq API error: {response.status_code} - {response.text}")
                    
#         except Exception as e:
#             logger.error(f"Groq API call failed: {str(e)}")
#             raise

#     async def _call_gemini(self, prompt: str) -> str:
#         """Call Gemini API"""
#         try:
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(
#                     f"{self.base_url}/models/gemini-pro:generateContent",
#                     params={"key": self.api_key},
#                     headers={"Content-Type": "application/json"},
#                     json={
#                         "contents": [{
#                             "parts": [{"text": prompt}]
#                         }],
#                         "generationConfig": {
#                             "temperature": settings.LLM_TEMPERATURE,
#                             "maxOutputTokens": settings.MAX_TOKENS,
#                             "topP": 1,
#                             "topK": 1
#                         }
#                     },
#                     timeout=30.0
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     return result["candidates"][0]["content"]["parts"][0]["text"]
#                 else:
#                     raise Exception(f"Gemini API error: {response.status_code} - {response.text}")
                    
#         except Exception as e:
#             logger.error(f"Gemini API call failed: {str(e)}")
#             raise

#     async def _call_openai(self, prompt: str) -> str:
#         """Call OpenAI API (backup option)"""
#         try:
#             async with httpx.AsyncClient() as client:
#                 response = await client.post(
#                     f"{self.base_url}/chat/completions",
#                     headers={
#                         "Authorization": f"Bearer {self.api_key}",
#                         "Content-Type": "application/json"
#                     },
#                     json={
#                         "model": self.model,
#                         "messages": [
#                             {"role": "user", "content": prompt}
#                         ],
#                         "temperature": settings.LLM_TEMPERATURE,
#                         "max_tokens": settings.MAX_TOKENS
#                     },
#                     timeout=30.0
#                 )
                
#                 if response.status_code == 200:
#                     result = response.json()
#                     return result["choices"][0]["message"]["content"]
#                 else:
#                     raise Exception(f"OpenAI API error: {response.status_code} - {response.text}")
                    
#         except Exception as e:
#             logger.error(f"OpenAI API call failed: {str(e)}")
#             raise

#     def _create_full_prompt(self, question: str, context: str, subject: Optional[QuestionType], system_prompt: str) -> str:
#         """Create the complete prompt for the LLM"""
#         return f"""
# {system_prompt}

# Question: {question}
# Subject: {subject.value if subject else "general mathematics"}

# {context}

# Please provide a comprehensive step-by-step solution following this format:

# SOLUTION:
# Step 1: [Clear description]
# Explanation: [Why this step is necessary and how it works]
# Formula (if applicable): [Mathematical formula used]

# Step 2: [Clear description]
# Explanation: [Why this step is necessary and how it works]
# Formula (if applicable): [Mathematical formula used]

# [Continue for all steps...]

# FINAL ANSWER: [Clear, concise final answer]

# CONFIDENCE: [Your confidence level from 0.0 to 1.0]

# EDUCATIONAL NOTES: [Additional tips, common mistakes to avoid, or related concepts]
# """

#     def _format_context(self, context_data: Dict[str, Any], source_type: SourceType) -> str:
#         """Format context data for LLM prompt"""
#         if source_type == SourceType.KNOWLEDGE_BASE:
#             kb_data = context_data.get("knowledge_base", {})
#             if kb_data:
#                 return f"""
# KNOWLEDGE BASE SOLUTION:
# Question: {kb_data.get('question', '')}
# Answer: {kb_data.get('final_answer', '')}
# Confidence: {kb_data.get('confidence_score', 0.0)}
# """
#         elif source_type == SourceType.WEB_SEARCH:
#             search_results = context_data.get("search_results", [])
#             if search_results:
#                 formatted_results = []
#                 for i, result in enumerate(search_results[:3], 1):
#                     formatted_results.append(f"""
# SEARCH RESULT {i}:
# Title: {result.get('title', '')}
# Content: {result.get('content', '')[:300]}...
# """)
#                 return "\n".join(formatted_results)
        
#         return "No additional context provided."

#     async def _parse_solution_response(
#         self,
#         solution_text: str,
#         question: str,
#         source_type: SourceType,
#         subject: Optional[QuestionType]
#     ) -> SolutionResponse:
#         """Parse LLM response into structured SolutionResponse"""
#         try:
#             steps = []
#             final_answer = ""
#             confidence_score = 0.8
            
#             # Split response into sections
#             sections = solution_text.split('\n\n')
            
#             for section in sections:
#                 section = section.strip()
                
#                 if section.upper().startswith('STEP'):
#                     # Parse step
#                     lines = section.split('\n')
#                     step_line = lines[0]
                    
#                     # Extract step number and description
#                     step_match = re.search(r'Step\s+(\d+):\s*(.+)', step_line, re.IGNORECASE)
#                     if step_match:
#                         step_num = int(step_match.group(1))
#                         description = step_match.group(2)
                        
#                         explanation = ""
#                         formula = None
                        
#                         # Look for explanation and formula in following lines
#                         for line in lines[1:]:
#                             if line.lower().startswith('explanation:'):
#                                 explanation = line[12:].strip()
#                             elif line.lower().startswith('formula:'):
#                                 formula = line[8:].strip()
                        
#                         step = Step(
#                             step_number=step_num,
#                             description=description,
#                             explanation=explanation,
#                             formula=formula,
#                             visual_aid=None
#                         )
#                         steps.append(step)
                
#                 elif section.upper().startswith('FINAL ANSWER'):
#                     final_answer = section[12:].strip().rstrip('.')
                
#                 elif section.upper().startswith('CONFIDENCE'):
#                     try:
#                         conf_text = section[10:].strip()
#                         confidence_match = re.search(r'(\d+\.?\d*)', conf_text)
#                         if confidence_match:
#                             confidence_score = float(confidence_match.group(1))
#                             if confidence_score > 1.0:
#                                 confidence_score = confidence_score / 100.0
#                     except:
#                         confidence_score = 0.8
            
#             # Determine subject if not provided
#             if not subject:
#                 subject = self._determine_subject(question)
            
#             # If no steps were parsed, create a basic step
#             if not steps:
#                 steps = [
#                     Step(
#                         step_number=1,
#                         description="Solution provided",
#                         explanation=solution_text[:200] + "..." if len(solution_text) > 200 else solution_text,
#                         formula=None,
#                         visual_aid=None
#                     )
#                 ]
            
#             # If no final answer was found, try to extract from last step or response
#             if not final_answer:
#                 final_answer = "Please refer to the solution steps above."
            
#             # Create solution response
#             solution = SolutionResponse(
#                 question=question,
#                 solution_id=str(uuid.uuid4()),
#                 steps=steps,
#                 final_answer=final_answer,
#                 confidence_score=min(max(confidence_score, 0.0), 1.0),
#                 source=source_type,
#                 subject=subject or QuestionType.ALGEBRA,
#                 difficulty_level=self._estimate_difficulty(question, steps),
#                 processing_time=0.0,  # Will be set by caller
#                 references=[],
#                 created_at=datetime.utcnow()
#             )
            
#             return solution
            
#         except Exception as e:
#             logger.error(f"Error parsing solution response: {str(e)}")
#             return self._create_error_solution(question, str(e), source_type, subject)

#     def _determine_subject(self, question: str) -> QuestionType:
#         """Determine mathematical subject from question"""
#         question_lower = question.lower()
        
#         subject_keywords = {
#             QuestionType.ALGEBRA: ["equation", "variable", "solve", "polynomial", "factor", "quadratic", "linear", "x", "y"],
#             QuestionType.CALCULUS: ["derivative", "integral", "limit", "differentiate", "integrate", "dx", "dy"],
#             QuestionType.GEOMETRY: ["triangle", "circle", "area", "perimeter", "angle", "polygon", "theorem"],
#             QuestionType.TRIGONOMETRY: ["sin", "cos", "tan", "sine", "cosine", "tangent", "radian"],
#             QuestionType.STATISTICS: ["mean", "median", "mode", "probability", "distribution", "variance"],
#             QuestionType.LINEAR_ALGEBRA: ["matrix", "vector", "determinant", "eigenvalue", "dot product"],
#             QuestionType.NUMBER_THEORY: ["prime", "divisible", "gcd", "lcm", "modular", "congruent"]
#         }
        
#         max_score = 0
#         detected_subject = QuestionType.ALGEBRA
        
#         for subject, keywords in subject_keywords.items():
#             score = sum(1 for keyword in keywords if keyword in question_lower)
#             if score > max_score:
#                 max_score = score
#                 detected_subject = subject
        
#         return detected_subject

#     def _estimate_difficulty(self, question: str, steps: List[Step]) -> int:
#         """Estimate difficulty level based on question and solution complexity"""
#         difficulty = 5  # Default medium difficulty
        
#         # Adjust based on number of steps
#         step_count = len(steps)
#         if step_count <= 2:
#             difficulty -= 1
#         elif step_count >= 6:
#             difficulty += 1
        
#         # Adjust based on mathematical complexity indicators
#         question_lower = question.lower()
#         complex_terms = [
#             "derivative", "integral", "matrix", "theorem", "proof", "differential",
#             "eigenvalue", "polynomial", "trigonometric", "logarithmic"
#         ]
        
#         complexity_score = sum(1 for term in complex_terms if term in question_lower)
#         difficulty += min(complexity_score, 3)
        
#         return min(max(difficulty, 1), 10)

#     def _create_error_solution(
#         self, 
#         question: str, 
#         error_msg: str, 
#         source_type: SourceType,
#         subject: Optional[QuestionType]
#     ) -> SolutionResponse:
#         """Create error solution response"""
#         return SolutionResponse(
#             question=question,
#             solution_id=str(uuid.uuid4()),
#             steps=[
#                 Step(
#                     step_number=1,
#                     description=f"Error occurred during solution generation",
#                     explanation=f"Technical details: {error_msg}",
#                     formula=None,
#                     visual_aid=None
#                 )
#             ],
#             final_answer="Solution could not be generated due to an error.",
#             confidence_score=0.0,
#             source=source_type,
#             subject=subject or QuestionType.ALGEBRA,
#             difficulty_level=5,
#             processing_time=0.0,
#             created_at=datetime.utcnow()
#         )

import logging
from typing import List, Dict, Any, Optional
from app.models.schemas import SearchResult
from app.core.config import settings
import httpx

logger = logging.getLogger(__name__)

class MCPWebSearchService:
    """
    A service for performing web searches for mathematical problems using Tavily.
    """

    def __init__(self):
        self.tavily_api_key = settings.TAVILY_API_KEY
        self.client = httpx.AsyncClient(timeout=30.0)
        logger.info("MCPWebSearchService initialized.")

    async def search_math_problems(
        self,
        query: str,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_results: int = 5
    ) -> List[SearchResult]:

        logger.info(f"Performing web search for: {query}")
        # This is where you would integrate your Tavily search logic.
        # For now, it returns an empty list to allow the server to run.
        return []