"""Clear all data from Pinecone index."""

from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

def clear_index():
    """Delete all vectors from Pinecone index."""
    
    index_name = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")
    
    print(f"\n⚠️  WARNING: This will delete ALL vectors from '{index_name}'")
    confirm = input("Type 'yes' to confirm: ")
    
    if confirm.lower() != 'yes':
        print("Cancelled.")
        return
    
    try:
        pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
        index = pc.Index(index_name)
        
        # Get current stats
        stats = index.describe_index_stats()
        total = stats.get('total_vector_count', 0)
        
        print(f"\n🗑️  Deleting {total} vectors...")
        
        # Delete all vectors
        index.delete(delete_all=True)
        
        print(f"✅ Index cleared!")
        print(f"\nRun setup_pinecone.py to add new documents.\n")
        
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    clear_index()