import logging
import re
from typing import List, Dict, Any, Optional, Union
import pandas as pd
import asyncio
import time
import json
import uuid
from datetime import datetime

# Updated imports for Groq/Gemini support
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.callbacks.streaming_aiter import AsyncIteratorCallbackHandler

from app.models.schemas import SolutionResponse, Step, QuestionType, SourceType
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMCallbackHandler(AsyncIteratorCallbackHandler):
    """Async callback handler for LLM operations"""
    
    def __init__(self):
        self.start_time = None
        self.tokens_used = 0
    
    async def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs):
        self.start_time = time.time()
        logger.info("LLM generation started")
    
    async def on_llm_end(self, response, **kwargs):
        if self.start_time:
            duration = time.time() - self.start_time
            logger.info(f"LLM generation completed in {duration:.2f}s")

class MathLLMService:
    """LLM service specialized for mathematical problem solving"""
    
    def __init__(self):
        self.callback_handler = LLMCallbackHandler()
        
        # Initialize LLM based on your configuration
        if settings.LLM_PROVIDER == "groq":
            self.llm = ChatGroq(
                model=settings.LLM_MODEL.replace("groq/", ""),  # Remove groq/ prefix
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                groq_api_key=settings.GROQ_API_KEY,
                callbacks=[self.callback_handler]
            )
            logger.info(f"Initialized Groq LLM: {settings.LLM_MODEL}")
            
        elif settings.LLM_PROVIDER == "gemini":
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-pro",
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                google_api_key=settings.GOOGLE_API_KEY,
                callbacks=[self.callback_handler]
            )
            logger.info("Initialized Gemini LLM")
            
        else:
            logger.warning(f"Unsupported LLM provider: {settings.LLM_PROVIDER}")
            self.llm = None
            
        # System prompts for different scenarios
        self.system_prompts = self._create_system_prompts()

    def _create_system_prompts(self) -> Dict[str, str]:
        """Create system prompts for different scenarios"""
        return {
            "knowledge_base": """You are a mathematical professor helping students learn mathematics. 
You have been provided with a relevant solution from the knowledge base. 
Your task is to:
1. Verify the solution is correct and complete
2. Enhance the explanation to make it clearer for students
3. Add educational insights and tips
4. Format the solution in clear, numbered steps
5. Ensure the final answer is clearly stated

CRITICAL: For multiple choice questions:
- If ONE option is correct, provide only that letter (A, B, C, or D)
- If MULTIPLE options are correct, provide ALL correct letters together (like BCD or ACD)
- Do NOT explain WHY in the final answer - just give the letter(s)

Focus on pedagogy while maintaining mathematical rigor.""",

            "web_search": """You are a mathematical professor creating step-by-step solutions.
You have been provided with web search results about a mathematical problem.
Your task is to:
1. Synthesize information from the search results
2. Create a clear, educational step-by-step solution
3. Verify mathematical accuracy
4. Add explanations for each step
5. Include relevant formulas and concepts
6. Provide the final answer clearly

CRITICAL: For multiple choice questions:
- If ONE option is correct, provide only that letter (A, B, C, or D)  
- If MULTIPLE options are correct, provide ALL correct letters together (like BCD or ACD)
- Do NOT explain WHY in the final answer - just give the letter(s)

Always prioritize educational value and clarity.""",

            "hybrid": """You are a mathematical professor combining knowledge base information with web research.
You have both knowledge base content and web search results.
Your task is to:
1. Compare and validate information from both sources
2. Create the most accurate and educational solution
3. Note any discrepancies between sources
4. Provide a comprehensive step-by-step solution
5. Include multiple solution methods if available
6. Give the final answer with confidence level

CRITICAL: For multiple choice questions:
- If ONE option is correct, provide only that letter (A, B, C, or D)
- If MULTIPLE options are correct, provide ALL correct letters together (like BCD or ACD)  
- Do NOT explain WHY in the final answer - just give the letter(s)

Synthesize the best of both sources while maintaining educational focus.""",

            "standalone": """You are a mathematical professor solving a problem without external resources.
Your task is to:
1. Analyze the mathematical problem carefully
2. Create a step-by-step solution using your mathematical knowledge
3. Explain each step clearly for student understanding
4. Include relevant mathematical concepts and formulas
5. Provide the final answer
6. Note if you're uncertain about any aspect

CRITICAL: For multiple choice questions:
- If ONE option is correct, provide only that letter (A, B, C, or D)
- If MULTIPLE options are correct, provide ALL correct letters together (like BCD or ACD)
- Do NOT explain WHY in the final answer - just give the letter(s)

Be honest about limitations and suggest verification methods if needed."""
        }

    async def generate_solution(
        self,
        question: str,
        context_data: Dict[str, Any],
        source_type: SourceType,
        subject: Optional[QuestionType] = None
    ) -> SolutionResponse:
        """Generate mathematical solution based on context and source type"""
        try:
            start_time = time.time()
            
            # Select appropriate system prompt
            system_prompt = self.system_prompts.get(source_type.value, self.system_prompts["standalone"])
            
            # Create chat prompt
            prompt_template = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_prompt),
                HumanMessagePromptTemplate.from_template(self._create_human_prompt(source_type))
            ])
            
            # Prepare context
            context_str = self._format_context(context_data, source_type)
            
            # Format prompt inputs
            prompt_inputs = {
                "question": question,
                "context": context_str,
                "subject": subject.value if subject else "general mathematics"
            }
            
            # Generate response using LLM
            if self.llm is None:
                return self._create_error_solution(question, "LLM not initialized", source_type, subject)
            
            messages = prompt_template.format_prompt(**prompt_inputs).to_messages()
            
            # Use invoke instead of agenerate for newer langchain versions
            try:
                response = await self.llm.ainvoke(messages)
                solution_text = response.content
            except AttributeError:
                # Fallback for older versions
                response = self.llm.invoke(messages)
                solution_text = response.content
            
            # Parse response into structured solution
            parsed_solution = await self._parse_solution_response(
                solution_text, question, source_type, subject
            )
            
            # Calculate processing time
            processing_time = time.time() - start_time
            parsed_solution.processing_time = processing_time
            
            logger.info(f"Generated solution for question: {question[:50]}... in {processing_time:.2f}s")
            return parsed_solution
            
        except Exception as e:
            logger.error(f"Error generating solution: {str(e)}")
            return self._create_error_solution(question, str(e), source_type, subject)

    def _create_human_prompt(self, source_type: SourceType) -> str:
        """Create human prompt based on source type"""
        base_prompt = """
Question: {question}
Subject: {subject}

{context}

Please provide a comprehensive step-by-step solution following this format:

SOLUTION:
Step 1: [Clear description]
Explanation: [Why this step is necessary and how it works]
Formula (if applicable): [Mathematical formula used]

Step 2: [Clear description]
Explanation: [Why this step is necessary and how it works]
Formula (if applicable): [Mathematical formula used]

[Continue for all steps...]

FINAL ANSWER: For multiple choice questions with options (A), (B), (C), (D), your FINAL ANSWER must be ONLY ONE of the following: A, B, C, or D. If multiple options are correct, provide them as a single string (e.g., BCD, AD). For all other questions, provide the complete answer. Do not include any other text or explanation in the FINAL ANSWER section.

CONFIDENCE: [Your confidence level from 0.0 to 1.0]

EDUCATIONAL NOTES: [Additional tips, common mistakes to avoid, or related concepts]
"""
        return base_prompt

    def _format_context(self, context_data: Dict[str, Any], source_type: SourceType) -> str:
        """Format context data for LLM prompt"""
        if source_type == SourceType.KNOWLEDGE_BASE:
            kb_data = context_data.get("knowledge_base", {})
            if kb_data:
                return f"""
KNOWLEDGE BASE SOLUTION:
Question: {kb_data.get('question', '')}
Answer: {kb_data.get('final_answer', '')}
Confidence: {kb_data.get('confidence_score', 0.0)}
"""
            
        elif source_type == SourceType.WEB_SEARCH:
            search_results = context_data.get("search_results", [])
            if search_results:
                formatted_results = []
                for i, result in enumerate(search_results[:3], 1):
                    formatted_results.append(f"""
SEARCH RESULT {i}:
Title: {result.title}
Content: {result.content[:500]}...
Relevance: {result.relevance_score}
""")
                return "\n".join(formatted_results)
            
        elif source_type == SourceType.HYBRID:
            kb_context = self._format_context(context_data, SourceType.KNOWLEDGE_BASE)
            search_context = self._format_context(context_data, SourceType.WEB_SEARCH)
            return f"{kb_context}\n\n{search_context}"
            
        return "No additional context provided."

    async def _parse_solution_response(
        self,
        solution_text: str,
        question: str,
        source_type: SourceType,
        subject: Optional[QuestionType]
    ) -> SolutionResponse:
        """Parse LLM response into structured SolutionResponse with improved answer extraction"""
        try:
            steps = []
            final_answer = ""
            confidence_score = 0.8
            educational_notes = ""
            
            # Split response into sections
            sections = solution_text.split('\n\n')
            
            # First pass - extract structured content
            for section in sections:
                section = section.strip()
                
                if section.upper().startswith('STEP'):
                    # Parse step
                    lines = section.split('\n')
                    step_line = lines[0]
                    
                    # Extract step number and description
                    import re
                    step_match = re.search(r'Step\s+(\d+):\s*(.+)', step_line, re.IGNORECASE)
                    if step_match:
                        step_num = int(step_match.group(1))
                        description = step_match.group(2)
                        
                        current_step = Step(
                            step_number=step_num,
                            description=description,
                            explanation="",
                            formula=None,
                            visual_aid=None
                        )
                        
                        # Look for explanation and formula in following lines
                        for line in lines[1:]:
                            if line.lower().startswith('explanation:'):
                                current_step.explanation = line[12:].strip()
                            elif line.lower().startswith('formula:'):
                                current_step.formula = line[8:].strip()
                        
                        steps.append(current_step)
                
                elif section.upper().startswith('FINAL ANSWER'):
                    # FIXED: Better final answer extraction
                    final_answer = section[12:].strip()
                    # Remove leading colon and spaces
                    if final_answer.startswith(':'):
                        final_answer = final_answer[1:].strip()
                
                elif section.upper().startswith('CONFIDENCE'):
                    try:
                        import re
                        conf_text = section[10:].strip()
                        conf_match = re.search(r'(\d+\.?\d*)', conf_text)
                        if conf_match:
                            confidence_score = float(conf_match.group(1))
                            if confidence_score > 1.0:
                                confidence_score = confidence_score / 100.0
                    except:
                        confidence_score = 0.8
                
                elif section.upper().startswith('EDUCATIONAL NOTES'):
                    educational_notes = section[17:].strip()
            
            # IMPROVED: Second pass - try to extract answer if not found in structured format
            if not final_answer or final_answer in ["", "Please see the solution steps above for the complete answer."]:
                final_answer = self._extract_answer_from_text(solution_text, question)
            
            # Create fallback steps if none found
            if not steps:
                steps = [
                    Step(
                        step_number=1,
                        description="Solution analysis",
                        explanation=solution_text[:300] + "..." if len(solution_text) > 300 else solution_text,
                        formula=None,
                        visual_aid=None
                    )
                ]
            
            # Create solution response
            solution = SolutionResponse(
                question=question,
                solution_id=str(uuid.uuid4()),
                steps=steps,
                final_answer=final_answer,
                confidence_score=min(max(confidence_score, 0.0), 1.0),
                source=source_type,
                subject=subject or QuestionType.ALGEBRA,
                difficulty_level=5,  # Default difficulty
                processing_time=0.0,  # Will be set by caller
                references=[],
                created_at=datetime.utcnow()
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Error parsing solution response: {str(e)}")
            return self._create_error_solution(question, str(e), source_type, subject)

    def _extract_answer_from_text(self, solution_text: str, question: str) -> str:
        """Try to extract the final answer using multiple strategies - IMPROVED FOR MULTI-LETTER ANSWERS"""
        import re

        # Strategy 1: Look for explicit "FINAL ANSWER:"
        final_answer_match = re.search(r'FINAL ANSWER:\s*(.*)', solution_text, re.IGNORECASE | re.DOTALL)
        if final_answer_match:
            answer = final_answer_match.group(1).strip()
            # Further clean up, remove confidence etc.
            answer = answer.split('\n')[0].strip()
            if answer:
                return answer


        # Strategy 2: Look for multiple choice patterns if the first strategy fails
        if self._is_multiple_choice_question(question):
            # This pattern is more robust
            mc_patterns = [
                r'(?:answer|option|choice|therefore|result|conclusion)\s*:?\s*\(?([A-D]{1,4})\)?',
            ]
            text_lower = solution_text.lower()
            for pattern in mc_patterns:
                matches = re.findall(pattern, text_lower)
                if matches:
                    return matches[-1].upper()

        return "Unable to determine final answer"

    def _is_multiple_choice_question(self, question: str) -> bool:
        """Check if question is multiple choice"""
        return bool(re.search(r'\([ABCD]\)', question))

    def _create_error_solution(
        self, 
        question: str, 
        error_msg: str, 
        source_type: SourceType,
        subject: Optional[QuestionType]
    ) -> SolutionResponse:
        """Create error solution response"""
        return SolutionResponse(
            question=question,
            solution_id=str(uuid.uuid4()),
            steps=[
                Step(
                    step_number=1,
                    description=f"Error occurred during solution generation: {error_msg}",
                    explanation="Please try again or rephrase your question.",
                    formula=None,
                    visual_aid=None
                )
            ],
            final_answer="Solution could not be generated due to an error.",
            confidence_score=0.0,
            source=source_type,
            subject=subject or QuestionType.ALGEBRA,
            difficulty_level=5,
            processing_time=0.0,
            created_at=datetime.utcnow()
        )