

from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from typing import List
import os


class PineconeVectorStoreManager:
    """Manages Pinecone vector store with Gemini embeddings."""
    
    def __init__(self):
        """Initialize Pinecone connection with Gemini embeddings."""
        print("\n Initializing Pinecone vector store...")
        
        # Initialize Pinecone
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")

        # Use  Gemini embeddings
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Connect to Pinecone index
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )
        
        # Get index stats
        index = self.pc.Index(self.index_name)
        stats = index.describe_index_stats()
        total_vectors = stats.get("total_vector_count", 0)
        
        print(f" Connected to Pinecone index: {self.index_name}")
        print(f" Total vectors in index: {total_vectors}")
        
        if total_vectors == 0:
            print(" Index is empty. Run setup script to add documents.")
        

    def search(self, query: str, k: int = 4) -> List[Document]:
        """
        Semantic search in Pinecone using Gemini embeddings.
        
        Args:
            query: Search query
            k: Number of results
            
        Returns:
            List of relevant documents
        """
        print(f" Searching Pinecone for: {query[:60]}...")
        
        results = self.vector_store.similarity_search(query, k=k)
        
        print(f" Found {len(results)} relevant documents from Pinecone")
        
        return results
    
    def add_documents(self, documents: List[Document]):
        """
        Add documents to Pinecone 
        
        Args:
            documents: List of documents to index
        """
        print(f" Adding {len(documents)} documents to Pinecone...")
        print("Generating embeddings with Gemini (FREE)...")
        
        self.vector_store.add_documents(documents)
        
        print(f"Successfully indexed {len(documents)} documents!")
       
    
    def get_retriever(self, k: int = 4):
        """Get LangChain retriever interface."""
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# Initialize global instance
print("\n" + "="*60)
print("Starting IKMS Query Planner ..")
print("="*60)

vector_store_manager = PineconeVectorStoreManager()

print("="*60)
print(" System ready for queries!")

print("="*60 + "\n")
