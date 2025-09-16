# backend/app/services/web_search.py

import os
import logging
import asyncio
import subprocess
import json
from typing import List, Optional, Dict, Any
from pathlib import Path
import sys

from app.models.schemas import SearchResult
from app.core.config import settings

logger = logging.getLogger(__name__)

class MCPWebSearchService:
    """Web search service using MCP (Model Context Protocol) server"""

    def __init__(self):
        # Build an absolute path from the project root
        project_root = Path(__file__).parent.parent.parent.parent
        self.mcp_server_path = project_root / "mcp" / "servers" / "search_server.py"
        self.tavily_api_key = settings.TAVILY_API_KEY

        if not self.mcp_server_path.exists():
            logger.warning(f"MCP server script not found at the constructed path: {self.mcp_server_path}")
        else:
            logger.info(f"MCP server script found at: {self.mcp_server_path}")

        logger.info("MCPWebSearchService initialized.")
    
    async def search_math_problems(
        self,
        query: str,
        subject: Optional[str] = None,
        difficulty: Optional[str] = None,
        max_results: int = 5
    ) -> List[SearchResult]:
        """Search for mathematical problems using MCP server"""
        try:
            logger.info(f"Performing MCP web search for: {query}")
            
            # Call MCP server
            mcp_results = await self._call_mcp_server(
                tool_name="search_math_problems",
                arguments={
                    "query": query,
                    "subject": subject,
                    "difficulty": difficulty,
                    "max_results": max_results
                }
            )
            
            # Convert MCP results to SearchResult objects
            search_results = []
            if mcp_results and isinstance(mcp_results, list):
                for result_text in mcp_results:
                    if hasattr(result_text, 'text'):
                        parsed_result = self._parse_mcp_search_result(result_text.text)
                        if parsed_result:
                            search_results.append(parsed_result)
            
            logger.info(f"MCP search returned {len(search_results)} results")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in MCP web search: {str(e)}")
            # Fallback to direct Tavily API call if MCP fails
            return await self._fallback_tavily_search(query, subject, difficulty, max_results)
    
    async def extract_solution_from_url(
        self,
        url: str,
        question: str
    ) -> Dict[str, Any]:
        """Extract solution from specific URL using MCP server"""
        try:
            logger.info(f"Extracting solution from URL: {url}")
            
            mcp_result = await self._call_mcp_server(
                tool_name="extract_math_solution",
                arguments={
                    "url": url,
                    "question": question
                }
            )
            
            if mcp_result and len(mcp_result) > 0:
                result_text = mcp_result[0].text if hasattr(mcp_result[0], 'text') else str(mcp_result[0])
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"error": "Failed to parse MCP response", "raw_response": result_text}
            
            return {"error": "No response from MCP server"}
            
        except Exception as e:
            logger.error(f"Error extracting solution via MCP: {str(e)}")
            return {"error": str(e)}
    
    async def verify_math_answer(
        self,
        question: str,
        proposed_answer: str,
        solution_steps: str
    ) -> Dict[str, Any]:
        """Verify mathematical answer using MCP server"""
        try:
            logger.info(f"Verifying answer via MCP: {proposed_answer}")
            
            mcp_result = await self._call_mcp_server(
                tool_name="verify_math_answer",
                arguments={
                    "question": question,
                    "proposed_answer": proposed_answer,
                    "solution_steps": solution_steps
                }
            )
            
            if mcp_result and len(mcp_result) > 0:
                result_text = mcp_result[0].text if hasattr(mcp_result[0], 'text') else str(mcp_result[0])
                try:
                    return json.loads(result_text)
                except json.JSONDecodeError:
                    return {"error": "Failed to parse verification response", "raw_response": result_text}
            
            return {"error": "No verification response from MCP server"}
            
        except Exception as e:
            logger.error(f"Error verifying answer via MCP: {str(e)}")
            return {"error": str(e)}
    
    async def _call_mcp_server(self, tool_name: str, arguments: Dict[str, Any]) -> List[Any]:
        """Call MCP server with specified tool and arguments"""
        try:
            # Prepare MCP server command
            cmd = [
                sys.executable,
                str(self.mcp_server_path),
            ]
            
            # Prepare environment variables
            env = {
                "TAVILY_API_KEY": self.tavily_api_key,
                "PYTHONPATH": str(Path(__file__).parent.parent.parent),
                **dict(os.environ)  # Include existing environment
            }
            
            # Create MCP request
            mcp_request = {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": tool_name,
                    "arguments": arguments
                }
            }
            
            # Execute MCP server call
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env
            )
            
            # Send request and get response
            stdout, stderr = await process.communicate(
                input=json.dumps(mcp_request).encode('utf-8')
            )
            
            if process.returncode != 0:
                logger.error(f"MCP server error: {stderr.decode()}")
                return []
            
            # Parse MCP response
            response_text = stdout.decode()
            if response_text:
                response = json.loads(response_text)
                if "result" in response and "content" in response["result"]:
                    return response["result"]["content"]
            
            return []
            
        except Exception as e:
            logger.error(f"Error calling MCP server: {str(e)}")
            return []
    
    def _parse_mcp_search_result(self, result_text: str) -> Optional[SearchResult]:
        """Parse MCP search result text into SearchResult object"""
        try:
            # Parse the formatted search result
            lines = result_text.strip().split('\n')
            
            title = ""
            url = ""
            content = ""
            relevance_score = 0.0
            
            for line in lines:
                line = line.strip()
                if line.startswith("Title:"):
                    title = line[6:].strip()
                elif line.startswith("URL:"):
                    url = line[4:].strip()
                elif line.startswith("Content:"):
                    content = line[8:].strip()
                elif line.startswith("Relevance Score:"):
                    try:
                        relevance_score = float(line[16:].strip())
                    except ValueError:
                        relevance_score = 0.5
            
            if title and url:
                return SearchResult(
                    title=title,
                    url=url,
                    content=content,
                    relevance_score=relevance_score
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Error parsing MCP search result: {str(e)}")
            return None
    
    async def _fallback_tavily_search(
        self,
        query: str,
        subject: Optional[str],
        difficulty: Optional[str], 
        max_results: int
    ) -> List[SearchResult]:
        """Fallback direct Tavily search if MCP fails"""
        try:
            import httpx
            
            # Enhance query for mathematical content
            enhanced_query = f"{query} mathematics step by step solution"
            if subject:
                enhanced_query += f" {subject}"
            if difficulty:
                enhanced_query += f" {difficulty} level"
            
            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    "https://api.tavily.com/search",
                    json={
                        "api_key": self.tavily_api_key,
                        "query": enhanced_query,
                        "search_depth": "advanced",
                        "include_answer": True,
                        "max_results": max_results,
                        "include_domains": [
                            "khanacademy.org",
                            "mathway.com",
                            "wolfram.com",
                            "math.stackexchange.com",
                            "brilliant.org"
                        ]
                    }
                )
                
                if response.status_code == 200:
                    data = response.json()
                    results = []
                    
                    for item in data.get("results", []):
                        result = SearchResult(
                            title=item.get("title", ""),
                            url=item.get("url", ""),
                            content=item.get("content", "")[:500],
                            relevance_score=item.get("score", 0.5)
                        )
                        results.append(result)
                    
                    return results
            
            return []
            
        except Exception as e:
            logger.error(f"Fallback Tavily search failed: {str(e)}")
            return []