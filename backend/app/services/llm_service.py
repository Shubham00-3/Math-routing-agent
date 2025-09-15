import logging
from typing import List, Dict, Any, Optional, Union
from langchain.llms import OpenAI
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import BaseMessage, SystemMessage, HumanMessage, AIMessage
from langchain.callbacks import AsyncCallbackHandler
import asyncio
import time
import json

from app.models.schemas import SolutionResponse, Step, QuestionType, SourceType
from app.core.config import settings

logger = logging.getLogger(__name__)

class LLMCallbackHandler(AsyncCallbackHandler):
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
        
        # Initialize LLM based on configuration
        if settings.LLM_MODEL.startswith("gpt"):
            self.llm = ChatOpenAI(
                model_name=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                openai_api_key=settings.OPENAI_API_KEY,
                callbacks=[self.callback_handler]
            )
        elif settings.LLM_MODEL.startswith("claude"):
            self.llm = ChatAnthropic(
                model=settings.LLM_MODEL,
                temperature=settings.LLM_TEMPERATURE,
                max_tokens=settings.MAX_TOKENS,
                anthropic_api_key=settings.ANTHROPIC_API_KEY,
                callbacks=[self.callback_handler]
            )
        else:
            raise ValueError(f"Unsupported LLM model: {settings.LLM_MODEL}")
        
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

Focus on pedagogy - make complex concepts accessible while maintaining mathematical rigor.
Always explain the reasoning behind each step.""",

            "web_search": """You are a mathematical professor creating step-by-step solutions.
You have been provided with web search results about a mathematical problem.
Your task is to:
1. Synthesize information from the search results
2. Create a clear, educational step-by-step solution
3. Verify mathematical accuracy
4. Add explanations for each step
5. Include relevant formulas and concepts
6. Provide the final answer clearly

If the search results are inconsistent or unclear, note this and provide the most likely correct solution with appropriate caveats.
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

Synthesize the best of both sources while maintaining educational focus.""",

            "standalone": """You are a mathematical professor solving a problem without external resources.
Your task is to:
1. Analyze the mathematical problem carefully
2. Create a step-by-step solution using your mathematical knowledge
3. Explain each step clearly for student understanding
4. Include relevant mathematical concepts and formulas
5. Provide the final answer
6. Note if you're uncertain about any aspect

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
            
            # Generate response
            messages = prompt_template.format_prompt(**prompt_inputs).to_messages()
            response = await self.llm.agenerate([messages])
            
            # Parse response into structured solution
            solution_text = response.generations[0][0].text
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

FINAL ANSWER: [Clear, concise final answer]

CONFIDENCE: [Your confidence level from 0.0 to 1.0]

EDUCATIONAL NOTES: [Additional tips, common mistakes to avoid, or related concepts]
"""
        
        if source_type == SourceType.KNOWLEDGE_BASE:
            return base_prompt + "\nNote: Use the provided knowledge base solution as reference but enhance it for better student understanding."
        elif source_type == SourceType.WEB_SEARCH:
            return base_prompt + "\nNote: Synthesize the web search results to create the most accurate solution."
        elif source_type == SourceType.HYBRID:
            return base_prompt + "\nNote: Compare knowledge base and web search information to provide the best solution."
        else:
            return base_prompt + "\nNote: Solve using your mathematical knowledge, being clear about any assumptions."

    def _format_context(self, context_data: Dict[str, Any], source_type: SourceType) -> str:
        """Format context data for LLM prompt"""
        if source_type == SourceType.KNOWLEDGE_BASE:
            kb_data = context_data.get("knowledge_base", {})
            if kb_data:
                return f"""
KNOWLEDGE BASE SOLUTION:
Question: {kb_data.get('question', '')}
Steps: {self._format_steps(kb_data.get('steps', []))}
Answer: {kb_data.get('final_answer', '')}
Confidence: {kb_data.get('confidence_score', 0.0)}
"""
            
        elif source_type == SourceType.WEB_SEARCH:
            search_results = context_data.get("search_results", [])
            formatted_results = []
            for i, result in enumerate(search_results[:3], 1):
                formatted_results.append(f"""
SEARCH RESULT {i}:
Title: {result.get('title', '')}
Source: {result.get('url', '')}
Content: {result.get('content', '')[:500]}...
Relevance: {result.get('relevance_score', 0.0)}
""")
            return "\n".join(formatted_results)
            
        elif source_type == SourceType.HYBRID:
            kb_context = self._format_context(context_data, SourceType.KNOWLEDGE_BASE)
            search_context = self._format_context(context_data, SourceType.WEB_SEARCH)
            return f"{kb_context}\n\n{search_context}"
            
        return "No additional context provided."

    def _format_steps(self, steps: List[Dict]) -> str:
        """Format solution steps for context"""
        if not steps:
            return "No steps provided"
        
        formatted_steps = []
        for step in steps:
            step_text = f"Step {step.get('step_number', '')}: {step.get('description', '')}"
            if step.get('explanation'):
                step_text += f"\nExplanation: {step.get('explanation')}"
            if step.get('formula'):
                step_text += f"\nFormula: {step.get('formula')}"
            formatted_steps.append(step_text)
        
        return "\n\n".join(formatted_steps)

    async def _parse_solution_response(
        self,
        solution_text: str,
        question: str,
        source_type: SourceType,
        subject: Optional[QuestionType]
    ) -> SolutionResponse:
        """Parse LLM response into structured SolutionResponse"""
        try:
            steps = []
            final_answer = ""
            confidence_score = 0.8
            educational_notes = ""
            
            # Split response into sections
            sections = solution_text.split('\n\n')
            current_step = None
            
            for section in sections:
                section = section.strip()
                
                if section.upper().startswith('STEP'):
                    # Parse step
                    lines = section.split('\n')
                    step_line = lines[0]
                    
                    # Extract step number and description
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
                    final_answer = section[12:].strip()
                
                elif section.upper().startswith('CONFIDENCE'):
                    try:
                        conf_text = section[10:].strip()
                        confidence_score = float(re.search(r'(\d+\.?\d*)', conf_text).group(1))
                        if confidence_score > 1.0:
                            confidence_score = confidence_score / 100.0
                    except:
                        confidence_score = 0.8
                
                elif section.upper().startswith('EDUCATIONAL NOTES'):
                    educational_notes = section[17:].strip()
            
            # Determine subject if not provided
            if not subject:
                subject = await self._determine_subject(question)
            
            # Create solution response
            solution = SolutionResponse(
                question=question,
                solution_id=str(uuid.uuid4()),
                steps=steps,
                final_answer=final_answer,
                confidence_score=min(max(confidence_score, 0.0), 1.0),
                source=source_type,
                subject=subject or QuestionType.ALGEBRA,
                difficulty_level=await self._estimate_difficulty(question, steps),
                processing_time=0.0,  # Will be set by caller
                references=[],
                created_at=datetime.utcnow()
            )
            
            return solution
            
        except Exception as e:
            logger.error(f"Error parsing solution response: {str(e)}")
            return self._create_error_solution(question, str(e), source_type, subject)

    async def _determine_subject(self, question: str) -> QuestionType:
        """Determine mathematical subject from question"""
        question_lower = question.lower()
        
        subject_keywords = {
            QuestionType.ALGEBRA: ["equation", "variable", "solve", "polynomial", "factor", "quadratic", "linear"],
            QuestionType.CALCULUS: ["derivative", "integral", "limit", "differentiate", "integrate", "dx", "dy"],
            QuestionType.GEOMETRY: ["triangle", "circle", "area", "perimeter", "angle", "polygon", "theorem"],
            QuestionType.TRIGONOMETRY: ["sin", "cos", "tan", "sine", "cosine", "tangent", "radian"],
            QuestionType.STATISTICS: ["mean", "median", "mode", "probability", "distribution", "variance"],
            QuestionType.LINEAR_ALGEBRA: ["matrix", "vector", "determinant", "eigenvalue", "dot product"],
            QuestionType.NUMBER_THEORY: ["prime", "divisible", "gcd", "lcm", "modular", "congruent"]
        }
        
        max_score = 0
        detected_subject = QuestionType.ALGEBRA
        
        for subject, keywords in subject_keywords.items():
            score = sum(1 for keyword in keywords if keyword in question_lower)
            if score > max_score:
                max_score = score
                detected_subject = subject
        
        return detected_subject

    async def _estimate_difficulty(self, question: str, steps: List[Step]) -> int:
        """Estimate difficulty level based on question and solution complexity"""
        difficulty = 5  # Default medium difficulty
        
        # Adjust based on number of steps
        step_count = len(steps)
        if step_count <= 2:
            difficulty -= 1
        elif step_count >= 6:
            difficulty += 1
        
        # Adjust based on mathematical complexity indicators
        question_lower = question.lower()
        complex_terms = [
            "derivative", "integral", "matrix", "theorem", "proof", "differential",
            "eigenvalue", "polynomial", "trigonometric", "logarithmic"
        ]
        
        complexity_score = sum(1 for term in complex_terms if term in question_lower)
        difficulty += min(complexity_score, 3)
        
        return min(max(difficulty, 1), 10)

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