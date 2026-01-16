"""Tools for agents to use."""

from langchain_core.tools import tool
from typing import List


@tool
def retrieval_tool(query: str) -> str:
    """
    Search the vector database for relevant information.
    
    Args:
        query: The search query
        
    Returns:
        Relevant context from the database
    """
    # TODO: This will connect to Pinecone in the next phase
    # For now, return a placeholder
    return f"[Placeholder] Results for query: {query}"