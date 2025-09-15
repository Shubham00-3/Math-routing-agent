async def test_mcp_search():
    """Test MCP web search functionality"""
    search_service = MCPWebSearchService()
    
    # Test search
    results = await search_service.search_math_problems(
        query="solve quadratic equation x^2 + 5x + 6 = 0",
        subject="algebra",
        difficulty="medium",
        max_results=3
    )
    
    print(f"Found {len(results)} results:")
    for result in results:
        print(f"Title: {result.title}")
        print(f"URL: {result.url}")
        print(f"Relevance: {result.relevance_score}")
        print("---")
    
    # Test extraction
    if results:
        solution = await search_service.extract_solution_from_url(
            results[0].url,
            "solve quadratic equation x^2 + 5x + 6 = 0"
        )
        print("Extracted solution:", solution)

if __name__ == "__main__":
    asyncio.run(test_mcp_search())