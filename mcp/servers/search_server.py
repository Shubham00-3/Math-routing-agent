#search_server.py
import asyncio
import json
import logging
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime
import re

# MCP Server Implementation
from mcp.server import Server 
import mcp.server.stdio as stdio_server
from mcp.types import Tool, TextContent, ImageContent
import mcp.server.stdio

logger = logging.getLogger(__name__)

class MathSearchServer:
    """MCP Server for mathematical web search"""
    
    def __init__(self, tavily_api_key: str):
        self.tavily_api_key = tavily_api_key
        self.server = Server("math-search-server")
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Register tools
        self._register_tools()
    
    def _register_tools(self):
        """Register MCP tools"""
        
        @self.server.tool("search_math_problems")
        async def search_math_problems(
            query: str,
            difficulty: Optional[str] = None,
            subject: Optional[str] = None,
            max_results: int = 5
        ) -> List[TextContent]:
            """
            Search for mathematical problems and solutions on the web.
            
            Args:
                query: The mathematical question or topic to search for
                difficulty: Optional difficulty level (easy, medium, hard)
                subject: Optional math subject (algebra, calculus, geometry, etc.)
                max_results: Maximum number of results to return (default: 5)
            
            Returns:
                List of search results with mathematical content
            """
            try:
                results = await self._search_tavily(
                    query=query,
                    difficulty=difficulty,
                    subject=subject,
                    max_results=max_results
                )
                
                formatted_results = []
                for result in results:
                    content = self._format_search_result(result)
                    formatted_results.append(TextContent(type="text", text=content))
                
                return formatted_results
                
            except Exception as e:
                logger.error(f"Error in search_math_problems: {str(e)}")
                return [TextContent(
                    type="text", 
                    text=f"Search error: {str(e)}"
                )]

        @self.server.tool("extract_math_solution")
        async def extract_math_solution(
            url: str,
            question: str
        ) -> List[TextContent]:
            """
            Extract mathematical solution from a specific URL.
            
            Args:
                url: URL of the webpage containing the solution
                question: The original question to help with extraction
                
            Returns:
                Extracted solution with step-by-step explanation
            """
            try:
                content = await self._extract_page_content(url)
                solution = await self._parse_math_solution(content, question)
                
                return [TextContent(
                    type="text",
                    text=json.dumps(solution, indent=2)
                )]
                
            except Exception as e:
                logger.error(f"Error in extract_math_solution: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"Extraction error: {str(e)}"
                )]

        @self.server.tool("verify_math_answer")
        async def verify_math_answer(
            question: str,
            proposed_answer: str,
            solution_steps: str
        ) -> List[TextContent]:
            """
            Verify a mathematical answer by searching for similar problems.
            
            Args:
                question: The original mathematical question
                proposed_answer: The proposed answer to verify
                solution_steps: The solution steps to check
                
            Returns:
                Verification results with confidence score
            """
            try:
                verification = await self._verify_answer(
                    question, proposed_answer, solution_steps
                )
                
                return [TextContent(
                    type="text",
                    text=json.dumps(verification, indent=2)
                )]
                
            except Exception as e:
                logger.error(f"Error in verify_math_answer: {str(e)}")
                return [TextContent(
                    type="text",
                    text=f"Verification error: {str(e)}"
                )]

    async def _search_tavily(
        self, 
        query: str, 
        difficulty: Optional[str] = None,
        subject: Optional[str] = None,
        max_results: int = 5
    ) -> List[Dict[str, Any]]:
        """Search using Tavily API"""
        try:
            # Enhance query with math-specific terms
            enhanced_query = self._enhance_math_query(query, difficulty, subject)
            
            payload = {
                "api_key": self.tavily_api_key,
                "query": enhanced_query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": True,
                "max_results": max_results,
                "include_domains": [
                    "khanacademy.org",
                    "mathway.com", 
                    "wolfram.com",
                    "math.stackexchange.com",
                    "brilliant.org",
                    "chegg.com",
                    "symbolab.com",
                    "mathpapa.com"
                ]
            }
            
            response = await self.client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._filter_math_results(data.get("results", []))
            else:
                logger.error(f"Tavily API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error in Tavily search: {str(e)}")
            return []

    def _enhance_math_query(
        self, 
        query: str, 
        difficulty: Optional[str] = None,
        subject: Optional[str] = None
    ) -> str:
        """Enhance search query with mathematical context"""
        enhanced_parts = [query]
        
        # Add mathematical context
        if "solve" not in query.lower():
            enhanced_parts.append("solve")
        if "step by step" not in query.lower():
            enhanced_parts.append("step by step solution")
        
        # Add subject context
        if subject:
            enhanced_parts.append(f"{subject} mathematics")
        
        # Add difficulty context
        if difficulty:
            enhanced_parts.append(f"{difficulty} level")
        
        # Add general math terms
        enhanced_parts.extend([
            "mathematics", "explanation", "tutorial"
        ])
        
        return " ".join(enhanced_parts)

    def _filter_math_results(self, results: List[Dict]) -> List[Dict]:
        """Filter and score results for mathematical relevance"""
        filtered_results = []
        
        math_indicators = [
            "equation", "formula", "solve", "calculate", "mathematics",
            "algebra", "calculus", "geometry", "trigonometry", "solution",
            "step", "answer", "problem", "theorem", "proof"
        ]
        
        for result in results:
            content = (result.get("content", "") + " " + 
                      result.get("title", "")).lower()
            
            # Calculate math relevance score
            math_score = sum(1 for indicator in math_indicators 
                           if indicator in content) / len(math_indicators)
            
            if math_score > 0.1:  # Minimum threshold
                result["math_relevance_score"] = math_score
                filtered_results.append(result)
        
        # Sort by relevance score
        filtered_results.sort(
            key=lambda x: x.get("math_relevance_score", 0), 
            reverse=True
        )
        
        return filtered_results

    async def _extract_page_content(self, url: str) -> str:
        """Extract content from a specific webpage"""
        try:
            response = await self.client.get(url)
            if response.status_code == 200:
                # Basic HTML content extraction
                content = response.text
                
                # Remove HTML tags (basic cleaning)
                content = re.sub(r'<[^>]+>', '', content)
                content = re.sub(r'\s+', ' ', content).strip()
                
                return content
            else:
                return f"Failed to fetch content from {url}"
                
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return f"Error extracting content: {str(e)}"

    async def _parse_math_solution(self, content: str, question: str) -> Dict:
        """Parse mathematical solution from webpage content"""
        try:
            # Extract potential solution steps
            steps = []
            
            # Look for numbered steps
            step_patterns = [
                r'Step\s+(\d+)[:\.]?\s*([^\.]+\.)',
                r'(\d+)\.\s*([^\.]+\.)',
                r'Solution:\s*([^\.]+\.)'
            ]
            
            for pattern in step_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE)
                for match in matches:
                    if len(match) == 2:
                        step_num, description = match
                        steps.append({
                            "step_number": int(step_num) if step_num.isdigit() else len(steps) + 1,
                            "description": description.strip()
                        })
            
            # Extract final answer
            answer_patterns = [
                r'Answer[:\s]*([^\.]+\.)',
                r'Final Answer[:\s]*([^\.]+\.)',
                r'Therefore[:\s]*([^\.]+\.)',
                r'Result[:\s]*([^\.]+\.)'
            ]
            
            final_answer = ""
            for pattern in answer_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    final_answer = match.group(1).strip()
                    break
            
            return {
                "question": question,
                "steps": steps,
                "final_answer": final_answer,
                "source": "web_extraction",
                "extraction_confidence": min(len(steps) * 0.2, 1.0)
            }
            
        except Exception as e:
            logger.error(f"Error parsing math solution: {str(e)}")
            return {
                "question": question,
                "steps": [],
                "final_answer": "",
                "source": "web_extraction",
                "extraction_confidence": 0.0,
                "error": str(e)
            }

    async def _verify_answer(
        self, 
        question: str, 
        proposed_answer: str, 
        solution_steps: str
    ) -> Dict:
        """Verify answer by cross-referencing with web sources"""
        try:
            # Search for similar problems
            verification_query = f"verify {question} answer {proposed_answer}"
            results = await self._search_tavily(verification_query, max_results=3)
            
            # Analyze verification results
            confidence_score = 0.0
            supporting_sources = []
            contradicting_sources = []
            
            for result in results:
                content = result.get("content", "").lower()
                title = result.get("title", "").lower()
                
                # Check if the proposed answer appears in the content
                if proposed_answer.lower() in content or proposed_answer.lower() in title:
                    confidence_score += 0.3
                    supporting_sources.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "relevance": result.get("math_relevance_score", 0.0)
                    })
                else:
                    # Look for different answers that might contradict
                    contradicting_sources.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "content_snippet": content[:200]
                    })
            
            confidence_score = min(confidence_score, 1.0)
            
            return {
                "question": question,
                "proposed_answer": proposed_answer,
                "confidence_score": confidence_score,
                "verification_status": "verified" if confidence_score > 0.6 else "uncertain",
                "supporting_sources": supporting_sources,
                "contradicting_sources": contradicting_sources,
                "verification_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error verifying answer: {str(e)}")
            return {
                "question": question,
                "proposed_answer": proposed_answer,
                "confidence_score": 0.0,
                "verification_status": "error",
                "error": str(e)
            }

    def _format_search_result(self, result: Dict) -> str:
        """Format search result for display"""
        title = result.get("title", "")
        url = result.get("url", "")
        content = result.get("content", "")
        score = result.get("math_relevance_score", 0.0)
        
        # Truncate content if too long
        if len(content) > 500:
            content = content[:500] + "..."
        
        formatted = f"""
Title: {title}
URL: {url}
Relevance Score: {score:.2f}
Content: {content}
---
"""
        return formatted.strip()