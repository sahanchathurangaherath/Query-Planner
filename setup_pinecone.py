"""
ONE-TIME SETUP: Index a PDF into Pinecone with FREE Gemini embeddings.
Run this ONCE to add your documents to the vector database.

"""

from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import os
import sys

load_dotenv()
print(" INDEXING ......")
    

def index_pdf(pdf_path: str):
    """Index a PDF file into Pinecone using FREE Gemini embeddings."""
    
    print("\n" + "="*70)
    print("📄 PINECONE PDF INDEXING")
    print("="*70)
    print(f"PDF: {pdf_path}")
    print(f"💡 Using FREE Gemini embeddings - No cost!\n")
    
    # Check if file exists
    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        return False
    
    # Check if it's a PDF
    if not pdf_path.lower().endswith('.pdf'):
        print(f"❌ Error: File must be a PDF")
        return False
    
    try:
        # Initialize embeddings
        print("🔧 Initializing Gemini embeddings...")
        embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        
        # Load PDF
        print(f"📖 Loading PDF: {pdf_path}")
        loader = PyPDFLoader(pdf_path)
        documents = loader.load()
        print(f"✅ Loaded {len(documents)} pages")
        
        # Split into chunks
        print(f"✂️  Splitting into chunks...")
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = splitter.split_documents(documents)
        print(f"✅ Created {len(chunks)} chunks")
        
        # Show sample
        print(f"\n📝 Sample chunk:")
        print(f"   Page: {chunks[0].metadata.get('page', 'unknown')}")
        print(f"   Text: {chunks[0].page_content[:100]}...")
        
        # Get index name
        index_name = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")
        
        # Index in Pinecone
        print(f"\n🔄 Indexing into Pinecone (index: {index_name})...")
        print(f"⏳ Generating embeddings with Gemini (FREE)...")
        
        PineconeVectorStore.from_documents(
            documents=chunks,
            embedding=embeddings,
            index_name=index_name
        )
        
        print(f"\n✅ SUCCESS!")
        print(f"   Indexed: {len(chunks)} chunks from {len(documents)} pages")
        print(f"   Cost: $0.00 (Gemini embeddings are free!)")
        print(f"\n🚀 You can now query your PDF:")
        print(f"   uvicorn src.app.api:app --reload")
        print("\n" + "="*70 + "\n")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during indexing: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def check_index_stats():
    """Check current Pinecone index statistics."""
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index_name = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")
        index = pc.Index(index_name)
        stats = index.describe_index_stats()
        
        print("\n" + "="*70)
        print("📊 CURRENT INDEX STATUS")
        print("="*70)
        print(f"Index: {index_name}")
        print(f"Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"Dimension: {stats.get('dimension', 'unknown')}")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"⚠️  Could not check index stats: {e}\n")


if __name__ == "__main__":
    # Check environment
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in .env file")
        sys.exit(1)
    
    if not os.getenv("PINECONE_API_KEY"):
        print("❌ Error: PINECONE_API_KEY not found in .env file")
        sys.exit(1)
    
    # Show current index stats
    check_index_stats()
    
    # Get PDF path from command line
    if len(sys.argv) < 2:
        print("Usage: python scripts/setup_pinecone.py <path_to_pdf>")
        print("\nExample:")
        print("  python scripts/setup_pinecone.py documents/research_paper.pdf")
        print("  python scripts/setup_pinecone.py C:\\Users\\Sahan\\Documents\\my_document.pdf")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    
    # Index the PDF
    success = index_pdf(pdf_path)
    
    if success:
        # Show updated stats
        check_index_stats()
    else:
        sys.exit(1)