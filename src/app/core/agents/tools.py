"""Tools for agents to use."""

from langchain_core.tools import tool
from ..retrieval.vector_store import vector_store_manager
from ..retrieval.serialization import serialize_chunks_with_ids


@tool
def retrieval_tool(query: str) -> str:
    """
    Search the vector database for relevant information.
    
    Args:
        query: The search query
        
    Returns:
        Relevant context from the database
    """
    # Search Pinecone
    results = vector_store_manager.search(query, k=5)
    
    if not results:
        return "No relevant information found in the database."
    
    # Serialize results
    formatted_context, citation_map = serialize_chunks_with_ids(results)
    
    return formatted_context