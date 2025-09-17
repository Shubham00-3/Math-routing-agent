# backend/app/services/web_search.py - Direct Tavily Integration (Replace your current file)

import asyncio
import json
import logging
import re
from typing import Dict, List, Any, Optional
from datetime import datetime
import httpx

from app.models.schemas import SearchResult, SourceType
from app.core.config import settings

logger = logging.getLogger(__name__)

class DirectWebSearchService:
    """Direct Tavily web search - no MCP server or subprocess needed"""

    def __init__(self):
        self.tavily_api_key = getattr(settings, 'TAVILY_API_KEY', None)
        self.client = httpx.AsyncClient(timeout=30.0)
        
        if not self.tavily_api_key:
            logger.warning("⚠️ TAVILY_API_KEY not found - web search will be unavailable")
            self.is_available = False
        else:
            self.is_available = True
            logger.info("✅ Tavily web search initialized successfully")

    async def search_math_problems(
        self,
        query: str,
        difficulty: Optional[str] = None,
        subject: Optional[str] = None,
        max_results: int = 5
    ) -> List[SearchResult]:
        """Direct search using Tavily API - no subprocess needed"""
        
        if not self.is_available:
            logger.warning("🚫 Web search unavailable - missing API key")
            return []

        try:
            logger.info(f"🔍 Searching Tavily for: '{query}'")
            
            # Build enhanced query for better math results
            enhanced_query = self._build_enhanced_query(query, difficulty, subject)
            logger.info(f"📝 Enhanced query: '{enhanced_query}'")
            
            # Direct Tavily API call
            payload = {
                "api_key": self.tavily_api_key,
                "query": enhanced_query,
                "search_depth": "advanced",
                "include_answer": True,
                "include_raw_content": True,
                "max_results": max_results * 2,  # Get more to filter better
                "include_domains": [
                    "khanacademy.org", "mathway.com", "wolfram.com",
                    "math.stackexchange.com", "brilliant.org", "symbolab.com",
                    "mathpapa.com", "coursehero.com", "chegg.com"
                ],
                "exclude_domains": [
                    "pinterest.com", "instagram.com", "facebook.com", 
                    "twitter.com", "tiktok.com"  # Exclude non-educational domains
                ]
            }

            # Make the API request
            response = await self.client.post(
                "https://api.tavily.com/search",
                json=payload,
                headers={"Content-Type": "application/json"}
            )

            if response.status_code == 200:
                data = response.json()
                raw_results = data.get("results", [])
                logger.info(f"📊 Tavily returned {len(raw_results)} raw results")
                
                # Filter and convert to SearchResult objects
                filtered_results = self._filter_and_convert_results(raw_results, query, max_results)
                logger.info(f"✨ Filtered to {len(filtered_results)} relevant math results")
                
                return filtered_results
                
            else:
                logger.error(f"❌ Tavily API error: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            logger.error(f"💥 Error in web search: {str(e)}")
            return []

    def _build_enhanced_query(
        self, 
        query: str, 
        difficulty: Optional[str] = None, 
        subject: Optional[str] = None
    ) -> str:
        """Build enhanced search query for better math results"""
        
        enhanced_parts = [query]

        # Add context for better math results
        if "solve" not in query.lower():
            enhanced_parts.append("solve")
        if "step" not in query.lower():
            enhanced_parts.append("step by step solution")
        
        # Add subject context
        if subject:
            enhanced_parts.append(f"{subject} mathematics")
        
        # Add difficulty context  
        if difficulty:
            enhanced_parts.append(f"{difficulty} level")
        
        # Add educational keywords for better filtering
        enhanced_parts.extend([
            "mathematics", "explanation", "tutorial", "formula", "equation"
        ])

        return " ".join(enhanced_parts)

    def _filter_and_convert_results(
        self, 
        raw_results: List[Dict[str, Any]], 
        query: str,
        max_results: int
    ) -> List[SearchResult]:
        """Filter raw Tavily results and convert to SearchResult objects"""
        
        filtered_results = []
        
        # Mathematical relevance keywords
        math_keywords = [
            "equation", "formula", "solve", "calculate", "mathematics", "math",
            "algebra", "calculus", "geometry", "trigonometry", "solution",
            "step", "answer", "problem", "theorem", "proof", "derivative",
            "integral", "function", "variable", "expression"
        ]
        
        for result in raw_results:
            try:
                title = result.get("title", "").lower()
                content = result.get("content", "").lower()
                url = result.get("url", "").lower()
                
                # Calculate math relevance score
                all_text = f"{title} {content} {url}"
                math_score = sum(1 for keyword in math_keywords if keyword in all_text)
                math_score = math_score / len(math_keywords)  # Normalize to 0-1
                
                # Bonus for trusted educational domains
                trusted_domains = [
                    "khanacademy", "wolfram", "mathway", "brilliant", 
                    "symbolab", "mathpapa", "stackexchange", "coursehero"
                ]
                domain_bonus = 0.3 if any(domain in url for domain in trusted_domains) else 0
                
                final_relevance = math_score + domain_bonus
                
                # Only include results with reasonable math relevance
                if final_relevance > 0.1:  # Minimum threshold
                    search_result = SearchResult(
                        title=result.get("title", ""),
                        url=result.get("url", ""),
                        content=result.get("content", "")[:1000],  # Limit content length
                        source="tavily_direct",
                        relevance_score=min(final_relevance, 1.0),  # Cap at 1.0
                        timestamp=datetime.utcnow()
                    )
                    filtered_results.append(search_result)
                    
            except Exception as e:
                logger.warning(f"⚠️ Error processing search result: {e}")
                continue
        
        # Sort by relevance and return top results
        filtered_results.sort(key=lambda x: x.relevance_score, reverse=True)
        return filtered_results[:max_results]

    async def extract_math_solution(
        self,
        url: str,
        question: str
    ) -> Optional[Dict[str, Any]]:
        """Extract mathematical solution from a specific URL"""
        try:
            logger.info(f"🔗 Extracting solution from: {url}")
            
            # Fetch the webpage content
            response = await self.client.get(url)
            if response.status_code != 200:
                logger.error(f"❌ Failed to fetch {url}: status {response.status_code}")
                return None
            
            # Clean and parse the content
            content = self._clean_html_content(response.text)
            solution_data = self._parse_math_solution(content, question)
            
            logger.info(f"📄 Extracted {len(solution_data.get('steps', []))} steps")
            return solution_data
            
        except Exception as e:
            logger.error(f"💥 Error extracting from {url}: {str(e)}")
            return None

    def _clean_html_content(self, html_content: str) -> str:
        """Clean HTML content to extract text"""
        try:
            # Remove script and style tags
            content = re.sub(r'<script[^>]*>.*?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
            content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
            
            # Remove HTML tags
            content = re.sub(r'<[^>]+>', ' ', content)
            
            # Clean up whitespace
            content = re.sub(r'\s+', ' ', content).strip()
            
            return content
        except Exception as e:
            logger.warning(f"⚠️ Error cleaning HTML: {e}")
            return html_content

    def _parse_math_solution(self, content: str, question: str) -> Dict[str, Any]:
        """Parse mathematical solution from clean text content"""
        try:
            steps = []
            
            # Look for step patterns
            step_patterns = [
                r'Step\s+(\d+)[:\.]?\s*([^\n]+)',
                r'(\d+)\.\s*([^\n]+)',
                r'Solution[:\s]*Step\s+(\d+)[:\.]?\s*([^\n]+)'
            ]

            for pattern in step_patterns:
                matches = re.findall(pattern, content, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    if len(match) == 2 and len(match[1].strip()) > 15:  # Filter short matches
                        step_num = match[0] if match[0].isdigit() else str(len(steps) + 1)
                        description = match[1].strip()
                        
                        steps.append({
                            "step_number": int(step_num),
                            "description": description
                        })

            # Look for final answer
            answer_patterns = [
                r'(?:Final\s+)?Answer[:\s]*([^\n.]+)',
                r'Therefore[:\s]*([^\n.]+)',
                r'Result[:\s]*([^\n.]+)',
                r'The answer is[:\s]*([^\n.]+)'
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
                "extraction_confidence": min(len(steps) * 0.2, 1.0),
                "steps_found": len(steps)
            }

        except Exception as e:
            logger.error(f"💥 Error parsing solution: {str(e)}")
            return {
                "question": question,
                "steps": [],
                "final_answer": "",
                "source": "web_extraction", 
                "extraction_confidence": 0.0,
                "error": str(e)
            }

    async def verify_math_answer(
        self,
        question: str,
        proposed_answer: str,
        solution_steps: str
    ) -> Dict[str, Any]:
        """Verify answer by searching for similar problems"""
        try:
            logger.info(f"🔍 Verifying answer: '{proposed_answer}'")
            
            # Search for verification
            verification_query = f"{question} solution answer"
            search_results = await self.search_math_problems(
                query=verification_query,
                max_results=3
            )
            
            confidence_score = 0.0
            supporting_sources = []
            
            # Check if our answer appears in search results
            for result in search_results:
                content_lower = result.content.lower()
                title_lower = result.title.lower()
                
                if proposed_answer.lower() in content_lower or proposed_answer.lower() in title_lower:
                    confidence_score += 0.3
                    supporting_sources.append({
                        "url": result.url,
                        "title": result.title,
                        "relevance": result.relevance_score
                    })

            confidence_score = min(confidence_score, 1.0)

            return {
                "question": question,
                "proposed_answer": proposed_answer,
                "confidence_score": confidence_score,
                "verification_status": "verified" if confidence_score > 0.5 else "uncertain",
                "supporting_sources": supporting_sources,
                "sources_checked": len(search_results),
                "verification_timestamp": datetime.utcnow().isoformat()
            }

        except Exception as e:
            logger.error(f"💥 Error in verification: {str(e)}")
            return {
                "question": question,
                "proposed_answer": proposed_answer,
                "confidence_score": 0.0,
                "verification_status": "error",
                "error": str(e)
            }

    async def is_available(self) -> bool:
        """Check if web search is available"""
        return self.is_available and bool(self.tavily_api_key)

# For backward compatibility, create an alias
MCPWebSearchService = DirectWebSearchService