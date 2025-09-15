import json
import asyncio
import logging
from typing import List, Dict, Any, Optional
import subprocess
import tempfile
import os

from app.models.schemas import SearchResult
from app.core.config import settings

logger = logging.getLogger(__name__)

class MCPWebSearchService:
    """Web search service using MCP protocol"""
    
    def __init__(self):
        self.mcp_server_path = settings.MCP_SEARCH_SERVER_PATH
        self.tavily_api_key = settings.TAVILY_API_KEY
        
    async def search_math_problems(
        self,
        query: str,
        difficulty: Optional[str] = None,
        subject: Optional[str] = None,
        max_results: int = 5
    ) -> List[SearchResult]:
        """Search for mathematical problems using MCP server"""
        try:
            # Prepare MCP request
            mcp_request = {
                "method": "tools/call",
                "params": {
                    "name": "search_math_problems",
                    "arguments": {
                        "query": query,
                        "difficulty": difficulty,
                        "subject": subject,
                        "max_results": max_results
                    }
                }
            }
            
            # Call MCP server
            results = await self._call_mcp_server(mcp_request)
            
            # Convert to SearchResult objects
            search_results = []
            for result_data in results:
                if result_data.get("type") == "text":
                    # Parse the text content as search results
                    parsed_results = self._parse_search_results(result_data.get("text", ""))
                    search_results.extend(parsed_results)
            
            logger.info(f"Found {len(search_results)} search results for query: {query}")
            return search_results
            
        except Exception as e:
            logger.error(f"Error in MCP web search: {str(e)}")
            return []

    async def extract_solution_from_url(
        self,
        url: str,
        question: str
    ) -> Optional[Dict[str, Any]]:
        """Extract mathematical solution from URL using MCP"""
        try:
            mcp_request = {
                "method": "tools/call",
                "params": {
                    "name": "extract_math_solution",
                    "arguments": {
                        "url": url,
                        "question": question
                    }
                }
            }
            
            results = await self._call_mcp_server(mcp_request)
            
            if results and results[0].get("type") == "text":
                solution_data = json.loads(results[0].get("text", "{}"))
                return solution_data
            
            return None
            
        except Exception as e:
            logger.error(f"Error extracting solution from {url}: {str(e)}")
            return None

    async def verify_answer(
        self,
        question: str,
        proposed_answer: str,
        solution_steps: str
    ) -> Dict[str, Any]:
        """Verify mathematical answer using MCP"""
        try:
            mcp_request = {
                "method": "tools/call",
                "params": {
                    "name": "verify_math_answer",
                    "arguments": {
                        "question": question,
                        "proposed_answer": proposed_answer,
                        "solution_steps": solution_steps
                    }
                }
            }
            
            results = await self._call_mcp_server(mcp_request)
            
            if results and results[0].get("type") == "text":
                verification_data = json.loads(results[0].get("text", "{}"))
                return verification_data
            
            return {"verification_status": "error", "confidence_score": 0.0}
            
        except Exception as e:
            logger.error(f"Error verifying answer: {str(e)}")
            return {"verification_status": "error", "confidence_score": 0.0}

    async def _call_mcp_server(self, request: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call MCP server with request"""
        try:
            # Create temporary file for communication
            with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
                json.dump(request, f)
                request_file = f.name
            
            try:
                # Set environment variables
                env = os.environ.copy()
                env['TAVILY_API_KEY'] = self.tavily_api_key
                
                # Run MCP server
                process = await asyncio.create_subprocess_exec(
                    'python', self.mcp_server_path,
                    stdin=asyncio.subprocess.PIPE,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    env=env
                )
                
                # Send request
                request_json = json.dumps(request)
                stdout, stderr = await process.communicate(request_json.encode())
                
                if process.returncode == 0:
                    response = json.loads(stdout.decode())
                    return response.get("result", [])
                else:
                    logger.error(f"MCP server error: {stderr.decode()}")
                    return []
                    
            finally:
                # Clean up temporary file
                os.unlink(request_file)
                
        except Exception as e:
            logger.error(f"Error calling MCP server: {str(e)}")
            return []

    def _parse_search_results(self, text_content: str) -> List[SearchResult]:
        """Parse search results from MCP server response"""
        results = []
        
        try:
            # Split by separator
            result_blocks = text_content.split("---")
            
            for block in result_blocks:
                if not block.strip():
                    continue
                
                lines = block.strip().split('\n')
                result_data = {}
                
                for line in lines:
                    if ':' in line:
                        key, value = line.split(':', 1)
                        key = key.strip().lower().replace(' ', '_')
                        value = value.strip()
                        result_data[key] = value
                
                if 'title' in result_data and 'url' in result_data:
                    search_result = SearchResult(
                        title=result_data.get('title', ''),
                        url=result_data.get('url', ''),
                        content=result_data.get('content', ''),
                        relevance_score=float(result_data.get('relevance_score', 0.0)),
                        source="mcp_web_search"
                    )
                    results.append(search_result)
                    
        except Exception as e:
            logger.error(f"Error parsing search results: {str(e)}")
        
        return results
