"""Tools for agents - Pinecone with Gemini embeddings."""

from langchain_core.tools import tool
from ..retrieval.vector_store import vector_store_manager


@tool
def retrieval_tool(query: str) -> str:
    """
    Search Pinecone vector database for relevant information.
    Uses FREE Gemini embeddings for semantic search.
    
    Args:
        query: The search query
        
    Returns:
        Relevant context from Pinecone
    """
    print(f"🔍 Searching Pinecone for: {query[:60]}...")
    
    # Semantic search in Pinecone
    results = vector_store_manager.search(query, k=4)
    
    if not results:
        return "No relevant information found in the database."
    
    # Format results with IDs
    context_parts = []
    for i, doc in enumerate(results, 1):
        chunk_id = f"C{i}"
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        
        context_parts.append(
            f"[{chunk_id}] (Page {page}, {source})\n{doc.page_content}"
        )
    
    formatted_context = "\n\n" + ("=" * 60 + "\n\n").join(context_parts)
    
    return formatted_context