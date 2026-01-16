"""Vector store management with Pinecone."""

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from typing import List
from langchain_core.documents import Document


class VectorStoreManager:
    """Manages Pinecone vector store operations."""
    
    def __init__(self):
        """Initialize Pinecone and embeddings."""
        self.pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        self.index_name = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Initialize vector store
        self.vector_store = PineconeVectorStore(
            index_name=self.index_name,
            embedding=self.embeddings
        )

        print(f" Connected to Pinecone index: {self.index_name}")

    def index_pdf(self, pdf_path: str) -> dict:
        """
        Load a PDF, split into chunks, and index in Pinecone.
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            dict with indexing stats
        """
        print(f"\n Loading PDF: {pdf_path}")
        
        # Load PDF
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        
        print(f" Loaded {len(documents)} pages")
        
        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks")
        
        # Add to vector store
        print(f" Indexing chunks in Pinecone...")
        self.vector_store.add_documents(chunks)
        
        print(f"Indexed {len(chunks)} chunks successfully!")
        
        return {
            "pages": len(documents),
            "chunks": len(chunks),
            "status": "success"
        }
    
    def search(self, query: str, k: int = 5) -> List[Document]:
        """
        Search for relevant chunks.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of relevant documents
        """
        print(f" Searching for: {query[:50]}...")
        
        results = self.vector_store.similarity_search(query, k=k)
        
        print(f" Found {len(results)} results")
        
        return results
    
    def get_retriever(self, k: int = 5):
        """
        Get a LangChain retriever.
        
        Args:
            k: Number of results to return
            
        Returns:
            LangChain retriever
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})


# Global instance
vector_store_manager = VectorStoreManager()