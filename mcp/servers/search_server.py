# mcp/servers/search_server.py - Corrected MCP Implementation

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
import httpx
from datetime import datetime
import re
import sys

# Add the project root directory to the Python path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from dotenv import load_dotenv

# Correct MCP imports - using the standard MCP pattern
import mcp.server.stdio
from mcp import types
from mcp.server import Server

logger = logging.getLogger(__name__)

class MathSearchMCPServer:
    """Corrected MCP Server for mathematical web search"""

    def __init__(self, tavily_api_key: str):
        self.tavily_api_key = tavily_api_key
        self.client = httpx.AsyncClient(timeout=30.0)
        
        # Create the server
        self.app = Server("math-search-server")
        
        # Register handlers
        self.app.list_tools = self.list_tools
        self.app.call_tool = self.call_tool

    async def list_tools(self) -> List[types.Tool]:
        """List available tools"""
        return [
            types.Tool(
                name="search_math_problems",
                description="Search for mathematical problems and solutions on the web",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Math problem or question to search for"
                        },
                        "difficulty": {
                            "type": "string", 
                            "description": "Difficulty level (easy, medium, hard)",
                            "enum": ["easy", "medium", "hard"]
                        },
                        "subject": {
                            "type": "string",
                            "description": "Math subject area",
                            "enum": ["algebra", "calculus", "geometry", "trigonometry", "statistics"]
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return",
                            "minimum": 1,
                            "maximum": 10,
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            ),
            types.Tool(
                name="extract_math_solution",
                description="Extract mathematical solution from a specific URL",
                inputSchema={
                    "type": "object", 
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to extract solution from"
                        },
                        "question": {
                            "type": "string", 
                            "description": "Original math question"
                        }
                    },
                    "required": ["url", "question"]
                }
            ),
            types.Tool(
                name="verify_math_answer",
                description="Verify a mathematical answer by cross-referencing sources",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                            "description": "Original math question"  
                        },
                        "proposed_answer": {
                            "type": "string",
                            "description": "Proposed answer to verify"
                        },
                        "solution_steps": {
                            "type": "string",
                            "description": "Solution steps taken"
                        }
                    },
                    "required": ["question", "proposed_answer", "solution_steps"]
                }
            )
        ]

    async def call_tool(self, name: str, arguments: Dict[str, Any]) -> List[types.TextContent]:
        """Handle tool calls"""
        try:
            logger.info(f"Calling tool: {name} with args: {arguments}")
            
            if name == "search_math_problems":
                return await self.search_math_problems(**arguments)
            elif name == "extract_math_solution": 
                return await self.extract_math_solution(**arguments)
            elif name == "verify_math_answer":
                return await self.verify_math_answer(**arguments)
            else:
                return [types.TextContent(
                    type="text", 
                    text=f"Unknown tool: {name}"
                )]
                
        except Exception as e:
            logger.error(f"Error calling tool {name}: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"Tool error: {str(e)}"
            )]

    async def search_math_problems(
        self,
        query: str,
        difficulty: Optional[str] = None,
        subject: Optional[str] = None,
        max_results: int = 5
    ) -> List[types.TextContent]:
        """Search for mathematical problems and solutions"""
        try:
            logger.info(f"Searching for math problems: query='{query}', subject='{subject}', difficulty='{difficulty}'")
            
            results = await self._search_tavily(
                query=query,
                difficulty=difficulty, 
                subject=subject,
                max_results=max_results
            )

            if not results:
                return [types.TextContent(
                    type="text",
                    text="No relevant mathematical content found for your query."
                )]

            # Format results for better consumption
            formatted_content = self._format_multiple_results(results[:max_results])
            
            return [types.TextContent(
                type="text", 
                text=formatted_content
            )]

        except Exception as e:
            logger.error(f"Error in search_math_problems: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"Search error: {str(e)}"
            )]

    async def extract_math_solution(
        self,
        url: str,
        question: str
    ) -> List[types.TextContent]:
        """Extract mathematical solution from specific URL"""
        try:
            logger.info(f"Extracting solution from URL: {url}")
            
            content = await self._extract_page_content(url)
            solution = await self._parse_math_solution(content, question)

            return [types.TextContent(
                type="text",
                text=json.dumps(solution, indent=2)
            )]

        except Exception as e:
            logger.error(f"Error in extract_math_solution: {str(e)}")
            return [types.TextContent(
                type="text", 
                text=f"Extraction error: {str(e)}"
            )]

    async def verify_math_answer(
        self,
        question: str,
        proposed_answer: str,
        solution_steps: str
    ) -> List[types.TextContent]:
        """Verify mathematical answer against web sources"""
        try:
            logger.info(f"Verifying answer: {proposed_answer}")
            
            verification = await self._verify_answer(
                question, proposed_answer, solution_steps
            )

            return [types.TextContent(
                type="text",
                text=json.dumps(verification, indent=2)
            )]

        except Exception as e:
            logger.error(f"Error in verify_math_answer: {str(e)}")
            return [types.TextContent(
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
        """Search using Tavily API with enhanced math focus"""
        try:
            # Enhanced query building for better math results
            enhanced_query = self._build_enhanced_math_query(query, difficulty, subject)
            logger.info(f"Enhanced query: {enhanced_query}")

            payload = {
                "api_key": self.tavily_api_key,
                "query": enhanced_query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": True,
                "max_results": max_results * 2,  # Get more to filter better
                "include_domains": [
                    "khanacademy.org",
                    "mathway.com", 
                    "wolfram.com",
                    "math.stackexchange.com",
                    "brilliant.org",
                    "symbolab.com",
                    "mathpapa.com",
                    "coursehero.com",
                    "slader.com",
                    "chegg.com"
                ]
            }

            response = await self.client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                results = data.get("results", [])
                logger.info(f"Tavily returned {len(results)} results")
                # Filter and rank by mathematical relevance
                filtered_results = self._filter_and_rank_math_results(results)
                logger.info(f"Filtered to {len(filtered_results)} relevant results")
                return filtered_results
            else:
                logger.error(f"Tavily API error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"Error in Tavily search: {str(e)}")
            return []

    def _build_enhanced_math_query(
        self, 
        query: str, 
        difficulty: Optional[str] = None, 
        subject: Optional[str] = None
    ) -> str:
        """Build enhanced search query for mathematical content"""
        enhanced_parts = [query]

        # Add context for better math results
        if "solve" not in query.lower():
            enhanced_parts.append("solve")
        if "step" not in query.lower():
            enhanced_parts.append("step by step solution")

        if subject:
            enhanced_parts.append(f"{subject} mathematics")

        if difficulty:
            enhanced_parts.append(f"{difficulty} level")

        # Add mathematical context
        enhanced_parts.extend([
            "mathematics", "explanation", "tutorial", "formula"
        ])

        return " ".join(enhanced_parts)

    def _filter_and_rank_math_results(self, results: List[Dict]) -> List[Dict]:
        """Filter and rank results by mathematical relevance"""
        filtered_results = []

        # High-value math keywords
        math_indicators = [
            "equation", "formula", "solve", "calculate", "mathematics",
            "algebra", "calculus", "geometry", "trigonometry", "solution",
            "step", "answer", "problem", "theorem", "proof", "derivative",
            "integral", "function", "variable", "coefficient"
        ]

        for result in results:
            title = result.get("title", "").lower()
            content = result.get("content", "").lower()
            url = result.get("url", "").lower()
            
            # Combine all text for scoring
            all_text = f"{title} {content} {url}"

            # Calculate math relevance score
            math_score = sum(1 for indicator in math_indicators if indicator in all_text)
            math_score = math_score / len(math_indicators)  # Normalize

            # Bonus for trusted math domains
            trusted_domains = [
                "khanacademy", "wolfram", "mathway", "brilliant", 
                "symbolab", "mathpapa", "stackexchange"
            ]
            domain_bonus = 0.2 if any(domain in url for domain in trusted_domains) else 0
            
            final_score = math_score + domain_bonus

            if final_score > 0.1:  # Minimum threshold
                result["math_relevance_score"] = final_score
                filtered_results.append(result)

        # Sort by relevance score
        filtered_results.sort(
            key=lambda x: x.get("math_relevance_score", 0),
            reverse=True
        )

        return filtered_results

    def _format_multiple_results(self, results: List[Dict]) -> str:
        """Format multiple search results into readable text"""
        if not results:
            return "No search results found."
            
        formatted_parts = ["=== MATH SEARCH RESULTS ===\n"]
        
        for i, result in enumerate(results, 1):
            title = result.get("title", "No Title")
            url = result.get("url", "")
            content = result.get("content", "")
            score = result.get("math_relevance_score", 0.0)

            # Truncate long content
            if len(content) > 400:
                content = content[:400] + "..."

            formatted_parts.append(f"""
RESULT {i}:
Title: {title}
URL: {url}
Relevance: {score:.2f}
Content: {content}
{'='*50}
""")

        return "\n".join(formatted_parts)

    async def _extract_page_content(self, url: str) -> str:
        """Extract clean text content from webpage"""
        try:
            response = await self.client.get(url)
            if response.status_code == 200:
                # Basic HTML cleaning
                content = response.text
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL)
                content = re.sub(r'<[^>]+>', ' ', content)
                content = re.sub(r'\s+', ' ', content).strip()
                return content
            else:
                return f"Failed to fetch content from {url} (Status: {response.status_code})"

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {str(e)}")
            return f"Error extracting content: {str(e)}"

    async def _parse_math_solution(self, content: str, question: str) -> Dict:
        """Parse mathematical solution from webpage content"""
        try:
            steps = []
            
            # Multiple step detection patterns
            step_patterns = [
                r'Step\s+(\d+)[:\.]?\s*([^\n]+)',
                r'(\d+)\.\s*([^\n]+)',
                r'Solution[:\s]*Step\s+(\d+)[:\.]?\s*([^\n]+)'
            ]

            for pattern in step_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if len(match) == 2 and match[1].strip():
                        step_num = match[0] if match[0].isdigit() else str(len(steps) + 1)
                        description = match[1].strip()
                        if len(description) > 10:  # Filter out very short matches
                            steps.append({
                                "step_number": int(step_num),
                                "description": description
                            })

            # Look for final answer
            answer_patterns = [
                r'(?:Final\s+)?Answer[:\s]*([^\n.]+)',
                r'Therefore[:\s]*([^\n.]+)',
                r'Result[:\s]*([^\n.]+)',
                r'Solution[:\s]*([^\n.]+)'
            ]

            final_answer = ""
            for pattern in answer_patterns:
                match = re.search(pattern, content, re.IGNORECASE)
                if match and match.group(1).strip():
                    final_answer = match.group(1).strip()
                    break

            return {
                "question": question,
                "steps": steps,
                "final_answer": final_answer,
                "source": "web_extraction",
                "extraction_confidence": min(len(steps) * 0.15, 1.0),
                "steps_found": len(steps)
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
            # Create verification query
            verification_query = f"{question} solution {proposed_answer}"
            results = await self._search_tavily(verification_query, max_results=3)

            confidence_score = 0.0
            supporting_sources = []
            contradicting_sources = []

            for result in results:
                content = result.get("content", "").lower()
                title = result.get("title", "").lower()
                combined_text = f"{title} {content}"

                # Check if proposed answer appears in the content
                if proposed_answer.lower() in combined_text:
                    confidence_score += 0.3
                    supporting_sources.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "relevance": result.get("math_relevance_score", 0.0)
                    })
                else:
                    # Look for contradictory information
                    contradicting_sources.append({
                        "url": result.get("url", ""),
                        "title": result.get("title", ""),
                        "snippet": content[:200]
                    })

            confidence_score = min(confidence_score, 1.0)

            return {
                "question": question,
                "proposed_answer": proposed_answer,
                "confidence_score": confidence_score,
                "verification_status": "verified" if confidence_score > 0.5 else "uncertain",
                "supporting_sources": supporting_sources,
                "contradicting_sources": contradicting_sources,
                "verification_timestamp": datetime.utcnow().isoformat(),
                "sources_checked": len(results)
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

# Main execution
def main():
    """Main function to run the MCP server"""
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Load environment variables
    project_root = Path(__file__).resolve().parent.parent.parent
    dotenv_path = project_root / "backend" / ".env"

    if dotenv_path.exists():
        logger.info(f"Loading environment variables from {dotenv_path}")
        load_dotenv(dotenv_path=dotenv_path)
    else:
        logger.warning(f".env file not found at {dotenv_path}")

    # Get Tavily API key
    tavily_api_key = os.getenv("TAVILY_API_KEY")
    if not tavily_api_key:
        raise ValueError("TAVILY_API_KEY not found in environment variables")

    logger.info("TAVILY_API_KEY loaded successfully")

    # Create and run server
    try:
        search_server = MathSearchMCPServer(tavily_api_key=tavily_api_key)
        logger.info("Starting MCP Math Search Server...")
        
        # Run the server using stdio
        asyncio.run(mcp.server.stdio.stdio_server(search_server.app))
        
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        raise

if __name__ == "__main__":
    main()