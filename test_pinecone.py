"""Test Pinecone connection."""

from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# List indexes
indexes = pc.list_indexes()
print("✅ Connected to Pinecone!")
print(f"Available indexes: {[idx.name for idx in indexes]}")

# Connect to your index
index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
stats = index.describe_index_stats()
print(f"\nIndex stats: {stats}")