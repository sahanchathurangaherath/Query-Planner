"""Document serialization utilities."""

from langchain_core.documents import Document
from typing import List, Dict, Tuple


def serialize_chunks_with_ids(docs: List[Document]) -> Tuple[str, Dict[str, dict]]:
    """
    Convert documents to formatted context string with stable IDs.
    
    Args:
        docs: List of LangChain Documents
        
    Returns:
        Tuple of (formatted_context, citation_map)
    """
    context_parts = []
    citation_map = {}
    
    for i, doc in enumerate(docs):
        chunk_id = f"C{i+1}"
        page = doc.metadata.get("page", "unknown")
        source = doc.metadata.get("source", "unknown")
        
        # Format chunk with ID
        chunk_text = f"[{chunk_id}] (Page {page})\n{doc.page_content}\n"
        context_parts.append(chunk_text)
        
        # Store citation info
        citation_map[chunk_id] = {
            "page": page,
            "source": source,
            "snippet": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content
        }
    
    formatted_context = "\n" + "="*60 + "\n".join(context_parts)
    
    return formatted_context, citation_map


def serialize_chunks_simple(docs: List[Document]) -> str:
    """
    Simple serialization without IDs (for backward compatibility).
    
    Args:
        docs: List of LangChain Documents
        
    Returns:
        Formatted context string
    """
    context_parts = []
    
    for i, doc in enumerate(docs):
        page = doc.metadata.get("page", "unknown")
        chunk_text = f"Chunk {i+1} (Page {page}):\n{doc.page_content}\n"
        context_parts.append(chunk_text)
    
    return "\n" + "="*60 + "\n".join(context_parts)