# # backend/app/agents/routing_agent.py - Fixed Routing Logic

# import logging
# from typing import Tuple, Dict, Any, Optional, List
# from enum import Enum
# import uuid
# from datetime import datetime

# from app.services.knowledge_base import KnowledgeBaseService
# from app.services.web_search import DirectWebSearchService as MCPWebSearchService
# from app.services.llm_service import MathLLMService
# from app.models.schemas import (
#     MathQuestionRequest, SolutionResponse, QuestionType, 
#     SourceType, SearchResult, KnowledgeEntry, Step
# )

# logger = logging.getLogger(__name__)

# class RoutingDecision(Enum):
#     """Routing decision types"""
#     KNOWLEDGE_BASE_ONLY = "knowledge_base_only"
#     WEB_SEARCH_ONLY = "web_search_only"
#     HYBRID = "hybrid"
#     STANDALONE = "standalone"

# class MathRoutingAgent:
#     """Enhanced routing agent with proper decision making"""
    
#     def __init__(self):
#         self.kb_service = KnowledgeBaseService()
#         self.web_search_service = MCPWebSearchService()
#         self.llm_service = MathLLMService()
        
#         # Updated routing thresholds for better performance
#         self.kb_similarity_threshold = 0.5  # Lower threshold to trigger KB more often
#         self.web_search_threshold = 0.5    # Lower threshold to trigger web search
#         self.hybrid_threshold = 0.6        # Threshold for hybrid approach
        
#         # MCP availability flag
#         self._mcp_available = None
    
#     async def process_question(self, request: MathQuestionRequest) -> SolutionResponse:
#         """Enhanced question processing with improved routing logic"""
#         try:
#             logger.info(f"🔄 Processing question: {request.question[:100]}...")
            
#             # Step 1: Check MCP availability
#             await self._check_mcp_availability()
            
#             # Step 2: Make routing decision with detailed logging
#             routing_decision, context_data = await self._make_routing_decision(request)
#             logger.info(f"📍 Routing decision: {routing_decision.value}")
#             logger.info(f"📊 Context scores: {context_data.get('scores', {})}")
            
#             # Step 3: Generate solution based on routing decision
#             solution = await self._generate_solution(request, routing_decision, context_data)
            
#             # Step 4: Post-process and validate solution
#             final_solution = await self._post_process_solution(solution, context_data)
            
#             logger.info(f"✅ Solution generated - Confidence: {final_solution.confidence_score:.2f}, Source: {final_solution.source.value}")
#             return final_solution
            
#         except Exception as e:
#             logger.error(f"❌ Error in routing agent: {str(e)}")
#             return await self._create_error_solution(request, str(e))

#     async def _check_mcp_availability(self):
#         """Check if MCP web search is available"""
#         if self._mcp_available is None:
#             try:
#                 self._mcp_available = await self.web_search_service.is_available()
#                 logger.info(f"🌐 MCP Web Search Available: {self._mcp_available}")
#             except Exception as e:
#                 logger.warning(f"⚠️ MCP availability check failed: {e}")
#                 self._mcp_available = False

#     async def _make_routing_decision(
#         self, 
#         request: MathQuestionRequest
#     ) -> Tuple[RoutingDecision, Dict[str, Any]]:
#         """Enhanced routing decision with detailed scoring"""
        
#         context_data = {
#             "scores": {},
#             "kb_results": [],
#             "reasoning": "",
#             "fallback_used": False
#         }
        
#         try:
#             # Step 1: Check Knowledge Base
#             logger.info("🔍 Searching knowledge base...")
#             kb_results = await self._search_knowledge_base(request.question)
#             context_data["kb_results"] = kb_results
            
#             kb_score = 0.0
#             if kb_results:
#                 # Use the best similarity score
#                 kb_score = max(result.get("similarity_score", 0.0) for result in kb_results)
#                 logger.info(f"📚 KB best match score: {kb_score:.3f}")
#             else:
#                 logger.info("📚 No KB results found")
            
#             context_data["scores"]["knowledge_base"] = kb_score
            
#             # Step 2: Evaluate question complexity and type
#             question_analysis = self._analyze_question_complexity(request.question)
#             context_data["scores"]["complexity"] = question_analysis["complexity_score"]
#             context_data["question_analysis"] = question_analysis
            
#             # Step 3: Make routing decision based on scores and availability
#             if kb_score >= self.kb_similarity_threshold:
#                 if self._mcp_available and kb_score < 0.9:  # High but not perfect match
#                     decision = RoutingDecision.HYBRID
#                     context_data["reasoning"] = f"Good KB match ({kb_score:.3f}) + web verification available"
#                 else:
#                     decision = RoutingDecision.KNOWLEDGE_BASE_ONLY
#                     context_data["reasoning"] = f"Strong KB match ({kb_score:.3f})"
            
#             elif self._mcp_available:
#                 # If web search is available, use it when KB score is low, regardless of question type
#                 if kb_score > self.hybrid_threshold:  # 0.6
#                     decision = RoutingDecision.HYBRID
#                     context_data["reasoning"] = f"Moderate KB match ({kb_score:.3f}) + web search available for verification"
#                 elif kb_score > self.web_search_threshold:  # 0.5
#                     # For scores between 0.5-0.6, use hybrid approach
#                     decision = RoutingDecision.HYBRID  
#                     context_data["reasoning"] = f"Moderate KB match ({kb_score:.3f}), hybrid approach with web verification"
#                 else:
#                     # For scores below 0.5, use web search primarily
#                     decision = RoutingDecision.WEB_SEARCH_ONLY
#                     context_data["reasoning"] = f"Low KB match ({kb_score:.3f}), web search for better results"
            
#             else:
#                 # Fallback to standalone when web search isn't available
#                 decision = RoutingDecision.STANDALONE
#                 context_data["reasoning"] = "No strong KB match, web search unavailable - using standalone LLM"
#                 context_data["fallback_used"] = True
                
#                 # Log this prominently since it indicates a limitation
#                 logger.warning(f"⚠️ Using STANDALONE mode - KB score: {kb_score:.3f}, MCP available: {self._mcp_available}")
            
#             logger.info(f"🎯 Decision reasoning: {context_data['reasoning']}")
#             return decision, context_data
            
#         except Exception as e:
#             logger.error(f"❌ Error in routing decision: {str(e)}")
#             # Emergency fallback
#             return RoutingDecision.STANDALONE, {
#                 "scores": {"error": True},
#                 "reasoning": f"Error in routing: {str(e)}",
#                 "fallback_used": True
#             }

#     def _analyze_question_complexity(self, question: str) -> Dict[str, Any]:
#         """Analyze question to determine complexity and search needs"""
#         question_lower = question.lower()
        
#         # Indicators that suggest external search might be needed
#         external_indicators = [
#             "current", "latest", "recent", "new", "today", "2024", "2025",
#             "state of the art", "modern", "contemporary", "updated",
#             "real world", "industry", "practical application"
#         ]
        
#         # Complex math topics that might benefit from multiple sources
#         complex_topics = [
#             "differential equations", "topology", "abstract algebra",
#             "real analysis", "complex analysis", "number theory",
#             "mathematical physics", "advanced", "graduate level"
#         ]
        
#         # Basic topics likely in knowledge base
#         basic_topics = [
#             "basic", "simple", "elementary", "fundamental",
#             "addition", "subtraction", "multiplication", "division",
#             "linear equation", "quadratic", "basic calculus"
#         ]
        
#         needs_external = any(indicator in question_lower for indicator in external_indicators)
#         is_complex = any(topic in question_lower for topic in complex_topics)
#         is_basic = any(topic in question_lower for topic in basic_topics)
        
#         # Calculate complexity score
#         complexity_score = 0.5  # baseline
#         if is_complex:
#             complexity_score += 0.3
#         if needs_external:
#             complexity_score += 0.2
#         if is_basic:
#             complexity_score -= 0.2
            
#         complexity_score = max(0.1, min(1.0, complexity_score))
        
#         return {
#             "complexity_score": complexity_score,
#             "needs_external_search": needs_external,
#             "is_complex_topic": is_complex,
#             "is_basic_topic": is_basic,
#             "indicators": {
#                 "external": [ind for ind in external_indicators if ind in question_lower],
#                 "complex": [topic for topic in complex_topics if topic in question_lower],
#                 "basic": [topic for topic in basic_topics if topic in question_lower]
#             }
#         }

#     async def _search_knowledge_base(self, question: str) -> List[Dict[str, Any]]:
#         """Search knowledge base with error handling"""
#         try:
#             results = await self.kb_service.search_similar_problems(question, top_k=3)
#             logger.info(f"📚 KB search returned {len(results)} results")
#             return results
#         except Exception as e:
#             logger.error(f"❌ KB search failed: {str(e)}")
#             return []

#     async def _generate_solution(
#         self,
#         request: MathQuestionRequest,
#         decision: RoutingDecision,
#         context_data: Dict[str, Any]
#     ) -> SolutionResponse:
#         """Generate solution based on routing decision"""
        
#         logger.info(f"🔧 Generating solution using: {decision.value}")
        
#         try:
#             if decision == RoutingDecision.KNOWLEDGE_BASE_ONLY:
#                 return await self._generate_kb_solution(request, context_data)
                
#             elif decision == RoutingDecision.WEB_SEARCH_ONLY:
#                 return await self._generate_web_solution(request, context_data)
                
#             elif decision == RoutingDecision.HYBRID:
#                 return await self._generate_hybrid_solution(request, context_data)
                
#             else:  # STANDALONE
#                 return await self._generate_standalone_solution(request, context_data)
                
#         except Exception as e:
#             logger.error(f"❌ Error generating solution: {str(e)}")
#             # Fallback to standalone
#             return await self._generate_standalone_solution(request, context_data)

#     async def _generate_web_solution(
#         self,
#         request: MathQuestionRequest, 
#         context_data: Dict[str, Any]
#     ) -> SolutionResponse:
#         """Generate solution using web search"""
#         logger.info("🌐 Starting web search solution generation...")
        
#         try:
#             # Determine search parameters
#             subject = request.subject.value if request.subject else None
#             difficulty = self._map_difficulty_level(request.difficulty_level)
            
#             # Perform web search
#             search_results = await self.web_search_service.search_math_problems(
#                 query=request.question,
#                 subject=subject,
#                 difficulty=difficulty,
#                 max_results=5
#             )
            
#             logger.info(f"🔍 Web search returned {len(search_results)} results")
            
#             if not search_results:
#                 logger.warning("⚠️ No web search results, falling back to standalone")
#                 return await self._generate_standalone_solution(request, context_data)
            
#             # Format search results for LLM
#             search_context = self._format_search_results(search_results)
            
#             # Generate solution using LLM with web context
#             solution = await self.llm_service.generate_solution(
#                 question=request.question,
#                 context=search_context,
#                 source_type=SourceType.WEB_SEARCH,
#                 subject=request.subject
#             )
            
#             logger.info("✅ Web search solution generated")
#             return solution
            
#         except Exception as e:
#             logger.error(f"❌ Web solution generation failed: {str(e)}")
#             # Fallback to standalone
#             return await self._generate_standalone_solution(request, context_data)

#     async def _generate_kb_solution(
#         self,
#         request: MathQuestionRequest,
#         context_data: Dict[str, Any]
#     ) -> SolutionResponse:
#         """Generate solution using knowledge base"""
#         logger.info("📚 Generating KB solution...")
        
#         try:
#             kb_results = context_data.get("kb_results", [])
            
#             if not kb_results:
#                 logger.warning("⚠️ No KB results available, using standalone")
#                 return await self._generate_standalone_solution(request, context_data)
            
#             # Format KB results for LLM
#             kb_context = self._format_kb_results(kb_results)
            
#             # Generate solution
#             solution = await self.llm_service.generate_solution(
#                 question=request.question,
#                 context=kb_context,
#                 source_type=SourceType.KNOWLEDGE_BASE,
#                 subject=request.subject
#             )
            
#             logger.info("✅ KB solution generated")
#             return solution
            
#         except Exception as e:
#             logger.error(f"❌ KB solution generation failed: {str(e)}")
#             return await self._generate_standalone_solution(request, context_data)

#     async def _generate_hybrid_solution(
#         self,
#         request: MathQuestionRequest,
#         context_data: Dict[str, Any]
#     ) -> SolutionResponse:
#         """Generate solution using both KB and web search"""
#         logger.info("🔀 Generating hybrid solution...")
        
#         try:
#             # Get KB context
#             kb_results = context_data.get("kb_results", [])
#             kb_context = self._format_kb_results(kb_results) if kb_results else ""
            
#             # Get web search results
#             subject = request.subject.value if request.subject else None
#             difficulty = self._map_difficulty_level(request.difficulty_level)
            
#             search_results = await self.web_search_service.search_math_problems(
#                 query=request.question,
#                 subject=subject,
#                 difficulty=difficulty,
#                 max_results=3  # Fewer results for hybrid
#             )
            
#             web_context = self._format_search_results(search_results) if search_results else ""
            
#             # Combine contexts
#             combined_context = f"""
#             KNOWLEDGE BASE INFORMATION:
#             {kb_context}
            
#             WEB SEARCH INFORMATION:
#             {web_context}
#             """
            
#             # Generate solution
#             solution = await self.llm_service.generate_solution(
#                 question=request.question,
#                 context=combined_context,
#                 source_type=SourceType.HYBRID,
#                 subject=request.subject
#             )
            
#             logger.info("✅ Hybrid solution generated")
#             return solution
            
#         except Exception as e:
#             logger.error(f"❌ Hybrid solution generation failed: {str(e)}")
#             return await self._generate_standalone_solution(request, context_data)

#     async def _generate_standalone_solution(
#         self,
#         request: MathQuestionRequest,
#         context_data: Dict[str, Any]
#     ) -> SolutionResponse:
#         """Generate solution using LLM only"""
#         logger.info("🤖 Generating standalone LLM solution...")
        
#         try:
#             solution = await self.llm_service.generate_solution(
#                 question=request.question,
#                 context="",
#                 source_type=SourceType.STANDALONE,
#                 subject=request.subject
#             )
            
#             logger.info("✅ Standalone solution generated")
#             return solution
            
#         except Exception as e:
#             logger.error(f"❌ Standalone solution failed: {str(e)}")
#             return await self._create_error_solution(request, str(e))

#     def _format_search_results(self, results: List[SearchResult]) -> str:
#         """Format search results for LLM context"""
#         if not results:
#             return ""
            
#         formatted_parts = []
#         for i, result in enumerate(results, 1):
#             formatted_parts.append(f"""
#             SOURCE {i}:
#             Title: {result.title}
#             Content: {result.content}
#             Relevance: {result.relevance_score:.2f}
#             URL: {result.url}
#             """)
        
#         return "\n".join(formatted_parts)

#     def _format_kb_results(self, results: List[Dict[str, Any]]) -> str:
#         """Format knowledge base results for LLM context"""
#         if not results:
#             return ""
            
#         formatted_parts = []
#         for i, result in enumerate(results, 1):
#             similarity = result.get("similarity_score", 0.0)
#             content = result.get("content", "")
#             solution = result.get("solution", "")
            
#             formatted_parts.append(f"""
#             KB ENTRY {i} (Similarity: {similarity:.2f}):
#             Problem: {content}
#             Solution: {solution}
#             """)
        
#         return "\n".join(formatted_parts)

#     def _map_difficulty_level(self, level: Optional[int]) -> Optional[str]:
#         """Map integer difficulty to string"""
#         if level is None:
#             return None
            
#         if level <= 3:
#             return "easy"
#         elif level <= 7:
#             return "medium"
#         else:
#             return "hard"

#     async def _post_process_solution(
#         self,
#         solution: SolutionResponse,
#         context_data: Dict[str, Any]
#     ) -> SolutionResponse:
#         """Post-process and enhance solution"""
#         try:
#             # Adjust confidence based on routing decision
#             if context_data.get("fallback_used", False):
#                 solution.confidence_score *= 0.8  # Lower confidence for fallback
                
#             # Add processing metadata
#             solution.processing_time = 0.0  # This would be calculated by caller
            
#             return solution
            
#         except Exception as e:
#             logger.error(f"❌ Error in post-processing: {str(e)}")
#             return solution

#     async def _create_error_solution(
#         self,
#         request: MathQuestionRequest,
#         error_msg: str
#     ) -> SolutionResponse:
#         """Create error solution response"""
#         return SolutionResponse(
#             question=request.question,
#             solution_id=str(uuid.uuid4()),
#             steps=[
#                 Step(
#                     step_number=1,
#                     description=f"Error occurred during processing: {error_msg}",
#                     explanation="Please try again or rephrase your question.",
#                     formula=None,
#                     visual_aid=None
#                 )
#             ],
#             final_answer="Unable to generate solution due to system error.",
#             confidence_score=0.0,
#             source=SourceType.STANDALONE,
#             subject=request.subject or QuestionType.ALGEBRA,
#             difficulty_level=request.difficulty_level or 5,
#             processing_time=0.0,
#             created_at=datetime.utcnow()
#         )
# backend/app/agents/routing_agent.py - Updated to use MCPWebSearchService

import logging
from typing import Tuple, Dict, Any, Optional, List
from enum import Enum
import uuid
from datetime import datetime

from app.services.knowledge_base import KnowledgeBaseService
# IMPORTANT: This import has been changed to use the new service
from app.services.web_search import MCPWebSearchService
from app.services.llm_service import MathLLMService
from app.models.schemas import (
    MathQuestionRequest, SolutionResponse, QuestionType, 
    SourceType, SearchResult, KnowledgeEntry, Step
)

logger = logging.getLogger(__name__)

class RoutingDecision(Enum):
    """Routing decision types"""
    KNOWLEDGE_BASE_ONLY = "knowledge_base_only"
    WEB_SEARCH_ONLY = "web_search_only"
    HYBRID = "hybrid"
    STANDALONE = "standalone"

class MathRoutingAgent:
    """Enhanced routing agent with proper decision making"""
    
    def __init__(self):
        self.kb_service = KnowledgeBaseService()
        # IMPORTANT: Now using the new MCP service
        self.web_search_service = MCPWebSearchService()
        self.llm_service = MathLLMService()
        
        # Updated routing thresholds for better performance
        self.kb_similarity_threshold = 0.5  # Lower threshold to trigger KB more often
        self.web_search_threshold = 0.5    # Lower threshold to trigger web search
        self.hybrid_threshold = 0.6        # Threshold for hybrid approach
        
        # MCP availability flag
        self._mcp_available = None
    
    async def process_question(self, request: MathQuestionRequest) -> SolutionResponse:
        """Enhanced question processing with improved routing logic"""
        try:
            logger.info(f"🔄 Processing question: {request.question[:100]}...")
            
            # Step 1: Check MCP availability
            await self._check_mcp_availability()
            
            # Step 2: Make routing decision with detailed logging
            routing_decision, context_data = await self._make_routing_decision(request)
            logger.info(f"📍 Routing decision: {routing_decision.value}")
            logger.info(f"📊 Context scores: {context_data.get('scores', {})}")
            
            # Step 3: Generate solution based on routing decision
            solution = await self._generate_solution(request, routing_decision, context_data)
            
            # Step 4: Post-process and validate solution
            final_solution = await self._post_process_solution(solution, context_data)
            
            logger.info(f"✅ Solution generated - Confidence: {final_solution.confidence_score:.2f}, Source: {final_solution.source.value}")
            return final_solution
            
        except Exception as e:
            logger.error(f"❌ Error in routing agent: {str(e)}")
            return await self._create_error_solution(request, str(e))

    async def _check_mcp_availability(self):
        """Check if MCP web search is available"""
        if self._mcp_available is None:
            try:
                self._mcp_available = await self.web_search_service.is_available()
                logger.info(f"🌐 MCP Web Search Available: {self._mcp_available}")
            except Exception as e:
                logger.warning(f"⚠️ MCP availability check failed: {e}")
                self._mcp_available = False

    async def _make_routing_decision(
        self, 
        request: MathQuestionRequest
    ) -> Tuple[RoutingDecision, Dict[str, Any]]:
        """Enhanced routing decision with detailed scoring"""
        
        context_data = {
            "scores": {},
            "kb_results": [],
            "reasoning": "",
            "fallback_used": False
        }
        
        try:
            # Step 1: Check Knowledge Base
            logger.info("🔍 Searching knowledge base...")
            kb_results = await self._search_knowledge_base(request.question)
            context_data["kb_results"] = kb_results
            
            kb_score = 0.0
            if kb_results:
                # Use the best similarity score
                kb_score = max(result.get("similarity_score", 0.0) for result in kb_results)
                logger.info(f"📚 KB best match score: {kb_score:.3f}")
            else:
                logger.info("📚 No KB results found")
            
            context_data["scores"]["knowledge_base"] = kb_score
            
            # Step 2: Evaluate question complexity and type
            question_analysis = self._analyze_question_complexity(request.question)
            context_data["scores"]["complexity"] = question_analysis["complexity_score"]
            context_data["question_analysis"] = question_analysis
            
            # Step 3: Make routing decision based on scores and availability
            if kb_score >= self.kb_similarity_threshold:
                if self._mcp_available and kb_score < 0.9:  # High but not perfect match
                    decision = RoutingDecision.HYBRID
                    context_data["reasoning"] = f"Good KB match ({kb_score:.3f}) + web verification available"
                else:
                    decision = RoutingDecision.KNOWLEDGE_BASE_ONLY
                    context_data["reasoning"] = f"Strong KB match ({kb_score:.3f})"
            
            elif self._mcp_available:
                # If web search is available, use it when KB score is low, regardless of question type
                if kb_score > self.hybrid_threshold:  # 0.6
                    decision = RoutingDecision.HYBRID
                    context_data["reasoning"] = f"Moderate KB match ({kb_score:.3f}) + web search available for verification"
                elif kb_score > self.web_search_threshold:  # 0.5
                    # For scores between 0.5-0.6, use hybrid approach
                    decision = RoutingDecision.HYBRID  
                    context_data["reasoning"] = f"Moderate KB match ({kb_score:.3f}), hybrid approach with web verification"
                else:
                    # For scores below 0.5, use web search primarily
                    decision = RoutingDecision.WEB_SEARCH_ONLY
                    context_data["reasoning"] = f"Low KB match ({kb_score:.3f}), web search for better results"
            
            else:
                # Fallback to standalone when web search isn't available
                decision = RoutingDecision.STANDALONE
                context_data["reasoning"] = "No strong KB match, web search unavailable - using standalone LLM"
                context_data["fallback_used"] = True
                
                # Log this prominently since it indicates a limitation
                logger.warning(f"⚠️ Using STANDALONE mode - KB score: {kb_score:.3f}, MCP available: {self._mcp_available}")
            
            logger.info(f"🎯 Decision reasoning: {context_data['reasoning']}")
            return decision, context_data
            
        except Exception as e:
            logger.error(f"❌ Error in routing decision: {str(e)}")
            # Emergency fallback
            return RoutingDecision.STANDALONE, {
                "scores": {"error": True},
                "reasoning": f"Error in routing: {str(e)}",
                "fallback_used": True
            }

    def _analyze_question_complexity(self, question: str) -> Dict[str, Any]:
        """Analyze question to determine complexity and search needs"""
        question_lower = question.lower()
        
        # Indicators that suggest external search might be needed
        external_indicators = [
            "current", "latest", "recent", "new", "today", "2024", "2025",
            "state of the art", "modern", "contemporary", "updated",
            "real world", "industry", "practical application"
        ]
        
        # Complex math topics that might benefit from multiple sources
        complex_topics = [
            "differential equations", "topology", "abstract algebra",
            "real analysis", "complex analysis", "number theory",
            "mathematical physics", "advanced", "graduate level"
        ]
        
        # Basic topics likely in knowledge base
        basic_topics = [
            "basic", "simple", "elementary", "fundamental",
            "addition", "subtraction", "multiplication", "division",
            "linear equation", "quadratic", "basic calculus"
        ]
        
        needs_external = any(indicator in question_lower for indicator in external_indicators)
        is_complex = any(topic in question_lower for topic in complex_topics)
        is_basic = any(topic in question_lower for topic in basic_topics)
        
        # Calculate complexity score
        complexity_score = 0.5  # baseline
        if is_complex:
            complexity_score += 0.3
        if needs_external:
            complexity_score += 0.2
        if is_basic:
            complexity_score -= 0.2
            
        complexity_score = max(0.1, min(1.0, complexity_score))
        
        return {
            "complexity_score": complexity_score,
            "needs_external_search": needs_external,
            "is_complex_topic": is_complex,
            "is_basic_topic": is_basic,
            "indicators": {
                "external": [ind for ind in external_indicators if ind in question_lower],
                "complex": [topic for topic in complex_topics if topic in question_lower],
                "basic": [topic for topic in basic_topics if topic in question_lower]
            }
        }

    async def _search_knowledge_base(self, question: str) -> List[Dict[str, Any]]:
        """Search knowledge base with error handling"""
        try:
            results = await self.kb_service.search_similar_problems(question, top_k=3)
            logger.info(f"📚 KB search returned {len(results)} results")
            return results
        except Exception as e:
            logger.error(f"❌ KB search failed: {str(e)}")
            return []

    async def _generate_solution(
        self,
        request: MathQuestionRequest,
        decision: RoutingDecision,
        context_data: Dict[str, Any]
    ) -> SolutionResponse:
        """Generate solution based on routing decision"""
        
        logger.info(f"🔧 Generating solution using: {decision.value}")
        
        try:
            if decision == RoutingDecision.KNOWLEDGE_BASE_ONLY:
                return await self._generate_kb_solution(request, context_data)
                
            elif decision == RoutingDecision.WEB_SEARCH_ONLY:
                return await self._generate_web_solution(request, context_data)
                
            elif decision == RoutingDecision.HYBRID:
                return await self._generate_hybrid_solution(request, context_data)
                
            else:  # STANDALONE
                return await self._generate_standalone_solution(request, context_data)
                
        except Exception as e:
            logger.error(f"❌ Error generating solution: {str(e)}")
            # Fallback to standalone
            return await self._generate_standalone_solution(request, context_data)

    async def _generate_web_solution(
        self,
        request: MathQuestionRequest, 
        context_data: Dict[str, Any]
    ) -> SolutionResponse:
        """Generate solution using web search"""
        logger.info("🌐 Starting web search solution generation...")
        
        try:
            # Determine search parameters
            subject = request.subject.value if request.subject else None
            difficulty = self._map_difficulty_level(request.difficulty_level)
            
            # Perform web search
            search_results = await self.web_search_service.search_math_problems(
                query=request.question,
                subject=subject,
                difficulty=difficulty,
                max_results=5
            )
            
            logger.info(f"🔍 Web search returned {len(search_results)} results")
            
            if not search_results:
                logger.warning("⚠️ No web search results, falling back to standalone")
                return await self._generate_standalone_solution(request, context_data)
            
            # Format search results for LLM
            search_context = self._format_search_results(search_results)
            
            # Generate solution using LLM with web context
            solution = await self.llm_service.generate_solution(
                question=request.question,
                context=search_context,
                source_type=SourceType.WEB_SEARCH,
                subject=request.subject
            )
            
            logger.info("✅ Web search solution generated")
            return solution
            
        except Exception as e:
            logger.error(f"❌ Web solution generation failed: {str(e)}")
            # Fallback to standalone
            return await self._generate_standalone_solution(request, context_data)

    async def _generate_kb_solution(
        self,
        request: MathQuestionRequest,
        context_data: Dict[str, Any]
    ) -> SolutionResponse:
        """Generate solution using knowledge base"""
        logger.info("📚 Generating KB solution...")
        
        try:
            kb_results = context_data.get("kb_results", [])
            
            if not kb_results:
                logger.warning("⚠️ No KB results available, using standalone")
                return await self._generate_standalone_solution(request, context_data)
            
            # Format KB results for LLM
            kb_context = self._format_kb_results(kb_results)
            
            # Generate solution
            solution = await self.llm_service.generate_solution(
                question=request.question,
                context=kb_context,
                source_type=SourceType.KNOWLEDGE_BASE,
                subject=request.subject
            )
            
            logger.info("✅ KB solution generated")
            return solution
            
        except Exception as e:
            logger.error(f"❌ KB solution generation failed: {str(e)}")
            return await self._generate_standalone_solution(request, context_data)

    async def _generate_hybrid_solution(
        self,
        request: MathQuestionRequest,
        context_data: Dict[str, Any]
    ) -> SolutionResponse:
        """Generate solution using both KB and web search"""
        logger.info("🔀 Generating hybrid solution...")
        
        try:
            # Get KB context
            kb_results = context_data.get("kb_results", [])
            kb_context = self._format_kb_results(kb_results) if kb_results else ""
            
            # Get web search results
            subject = request.subject.value if request.subject else None
            difficulty = self._map_difficulty_level(request.difficulty_level)
            
            search_results = await self.web_search_service.search_math_problems(
                query=request.question,
                subject=subject,
                difficulty=difficulty,
                max_results=3  # Fewer results for hybrid
            )
            
            web_context = self._format_search_results(search_results) if search_results else ""
            
            # Combine contexts
            combined_context = f"""
            KNOWLEDGE BASE INFORMATION:
            {kb_context}
            
            WEB SEARCH INFORMATION:
            {web_context}
            """
            
            # Generate solution
            solution = await self.llm_service.generate_solution(
                question=request.question,
                context=combined_context,
                source_type=SourceType.HYBRID,
                subject=request.subject
            )
            
            logger.info("✅ Hybrid solution generated")
            return solution
            
        except Exception as e:
            logger.error(f"❌ Hybrid solution generation failed: {str(e)}")
            return await self._generate_standalone_solution(request, context_data)

    async def _generate_standalone_solution(
        self,
        request: MathQuestionRequest,
        context_data: Dict[str, Any]
    ) -> SolutionResponse:
        """Generate solution using LLM only"""
        logger.info("🤖 Generating standalone LLM solution...")
        
        try:
            solution = await self.llm_service.generate_solution(
                question=request.question,
                context="",
                source_type=SourceType.STANDALONE,
                subject=request.subject
            )
            
            logger.info("✅ Standalone solution generated")
            return solution
            
        except Exception as e:
            logger.error(f"❌ Standalone solution failed: {str(e)}")
            return await self._create_error_solution(request, str(e))

    def _format_search_results(self, results: List[SearchResult]) -> str:
        """Format search results for LLM context"""
        if not results:
            return ""
            
        formatted_parts = []
        for i, result in enumerate(results, 1):
            formatted_parts.append(f"""
            SOURCE {i}:
            Title: {result.title}
            Content: {result.content}
            Relevance: {result.relevance_score:.2f}
            URL: {result.url}
            """)
        
        return "\n".join(formatted_parts)

    def _format_kb_results(self, results: List[Dict[str, Any]]) -> str:
        """Format knowledge base results for LLM context"""
        if not results:
            return ""
            
        formatted_parts = []
        for i, result in enumerate(results, 1):
            similarity = result.get("similarity_score", 0.0)
            content = result.get("content", "")
            solution = result.get("solution", "")
            
            formatted_parts.append(f"""
            KB ENTRY {i} (Similarity: {similarity:.2f}):
            Problem: {content}
            Solution: {solution}
            """)
        
        return "\n".join(formatted_parts)

    def _map_difficulty_level(self, level: Optional[int]) -> Optional[str]:
        """Map integer difficulty to string"""
        if level is None:
            return None
            
        if level <= 3:
            return "easy"
        elif level <= 7:
            return "medium"
        else:
            return "hard"

    async def _post_process_solution(
        self,
        solution: SolutionResponse,
        context_data: Dict[str, Any]
    ) -> SolutionResponse:
        """Post-process and enhance solution"""
        try:
            # Adjust confidence based on routing decision
            if context_data.get("fallback_used", False):
                solution.confidence_score *= 0.8  # Lower confidence for fallback
                
            # Add processing metadata
            solution.processing_time = 0.0  # This would be calculated by caller
            
            return solution
            
        except Exception as e:
            logger.error(f"❌ Error in post-processing: {str(e)}")
            return solution

    async def _create_error_solution(
        self,
        request: MathQuestionRequest,
        error_msg: str
    ) -> SolutionResponse:
        """Create error solution response"""
        return SolutionResponse(
            question=request.question,
            solution_id=str(uuid.uuid4()),
            steps=[
                Step(
                    step_number=1,
                    description=f"Error occurred during processing: {error_msg}",
                    explanation="Please try again or rephrase your question.",
                    formula=None,
                    visual_aid=None
                )
            ],
            final_answer="Unable to generate solution due to system error.",
            confidence_score=0.0,
            source=SourceType.STANDALONE,
            subject=request.subject or QuestionType.ALGEBRA,
            difficulty_level=request.difficulty_level or 5,
            processing_time=0.0,
            created_at=datetime.utcnow()
        )