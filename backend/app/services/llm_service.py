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
        context: str = "",
        source_type: SourceType = SourceType.STANDALONE,
        subject: Optional[QuestionType] = None
    ) -> SolutionResponse:
        """Generate mathematical solution with context support"""
        try:
            start_time = datetime.utcnow()
            
            # Select appropriate system prompt based on source type
            system_prompt = self.system_prompts.get(source_type.value, self.system_prompts["standalone"])
            
            # Build the prompt with context if provided
            if context and context.strip():
                user_prompt = f"""
                Context Information:
                {context}
                
                Question: {question}
                
                Please provide a step-by-step solution using the context information provided above.
                """
            else:
                user_prompt = f"""
                Question: {question}
                
                Please provide a step-by-step mathematical solution.
                """
            
            # Generate solution using LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ]
            
            logger.info(f"🤖 Generating solution using {source_type.value} approach")
            
            if self.llm is None:
                return self._create_error_solution(question, "LLM not initialized", source_type, subject)

            response = await self.llm.ainvoke(messages)
            solution_text = response.content
            logger.info(f"🐛 RAW LLM RESPONSE: {solution_text}")
            # Parse the response into structured format
            solution = await self._parse_solution_response(
                solution_text, question, source_type, subject
            )
            
            # Calculate processing time
            processing_time = (datetime.utcnow() - start_time).total_seconds()
            solution.processing_time = processing_time
            
            logger.info(f"✅ Solution generated in {processing_time:.2f}s")
            return solution
            
        except Exception as e:
            logger.error(f"❌ Error generating solution: {str(e)}")
            return self._create_error_solution(question, str(e), source_type, subject)

    async def _parse_solution_response(
        self,
        solution_text: str,
        question: str,
        source_type: SourceType,
        subject: Optional[QuestionType]
    ) -> SolutionResponse:
        """Parse LLM response into a structured SolutionResponse with explanations."""
        try:
            steps = []
            final_answer = ""
            
            # Clean the text by removing markdown bolding
            solution_text = solution_text.replace('**', '').strip()
    
            # Isolate the final answer first to prevent it from being parsed as a step
            final_answer_keywords = ["Final Answer:", "FINAL ANSWER:", "Conclusion:"]
            text_for_steps = solution_text
            
            for keyword in final_answer_keywords:
                if keyword in solution_text:
                    parts = solution_text.split(keyword, 1)
                    text_for_steps = parts[0]
                    final_answer = parts[1].strip()
                    break
    
            # Use regex to split the text into steps. This pattern looks for "Step X:"
            # and captures everything until the next "Step X:" or the end of the string.
            step_chunks = re.split(r'(?=Step\s+\d+:)', text_for_steps, flags=re.IGNORECASE)
            
            step_num_counter = 1
            for chunk in step_chunks:
                chunk = chunk.strip()
                if not chunk:
                    continue
    
                # Extract the step description and explanation from the chunk
                match = re.match(r'Step\s+\d+:\s*([^\n]+)\n?([\s\S]*)', chunk, re.IGNORECASE)
                if match:
                    description = match.group(1).strip()
                    explanation = match.group(2).strip()
                    
                    # If there's no real explanation, use the description as the main content
                    if not explanation:
                        explanation = description
                        description = f"Step {step_num_counter}" # Generic title
    
                    steps.append(Step(
                        step_number=step_num_counter,
                        description=description,
                        explanation=explanation
                    ))
                    step_num_counter += 1
    
            # If after all that, the final answer is still empty, create a sensible default
            if not final_answer:
                final_answer = "The solution is detailed in the steps above."
    
            # If no steps were parsed at all, use the whole text as the first step's explanation
            if not steps and solution_text:
                steps.append(Step(
                    step_number=1,
                    description="Solution Analysis",
                    explanation=solution_text
                ))
    
            return SolutionResponse(
                question=question,
                solution_id=str(uuid.uuid4()),
                steps=steps,
                final_answer=final_answer,
                confidence_score=0.85,  # Adjusted default confidence
                source=source_type,
                subject=subject or QuestionType.ALGEBRA,
                difficulty_level=5,
                processing_time=0.0,
                created_at=datetime.utcnow()
            )
    
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

