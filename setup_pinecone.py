"""
INDEXING SCRIPT: Using official google-genai SDK
Target: Gemini Free Tier (100 RPM limit) with Auto-Retries
"""

import os
import sys
import time
from dotenv import load_dotenv
from pinecone import Pinecone
from google import genai
from google.genai import types
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# Initialize Clients
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")
index = pc.Index(index_name)

def index_pdf_with_genai_sdk(pdf_path: str):
    print(f"\n🚀 Starting Indexing: {pdf_path}")
    
    # 1. Load and Split
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(documents)
    print(f"✅ Split PDF into {len(chunks)} chunks.")

    # 2. Adjusted Batch Processing for 100 RPM limit
    batch_size = 20 # Lowered to 20 chunks per batch
    
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [chunk.page_content for chunk in batch]
        batch_num = (i // batch_size) + 1
        total_batches = (len(chunks) // batch_size) + 1
        
        print(f"📤 Processing batch {batch_num} of {total_batches}...")

        # 3. Automatic Retry Loop
        success = False
        retries = 0
        
        while not success and retries < 5:
            try:
                # Generate Embeddings
                result = client.models.embed_content(
                    model="gemini-embedding-001",
                    contents=texts,
                    config=types.EmbedContentConfig(output_dimensionality=768)
                )
                
                # Extract embeddings from result
                embeddings_list = [e.values for e in result.embeddings]
                
                # Prepare and Upsert to Pinecone
                vectors_to_upsert = []
                for j, emb in enumerate(embeddings_list):
                    chunk_id = f"id_{i + j}"
                    metadata = batch[j].metadata
                    metadata["text"] = texts[j] 
                    
                    vectors_to_upsert.append({
                        "id": chunk_id,
                        "values": emb,
                        "metadata": metadata
                    })
                
                index.upsert(vectors=vectors_to_upsert)
                success = True # Break out of the retry loop
                
                # Normal sleep to maintain ~80 requests per minute
                if i + batch_size < len(chunks):
                    print("💤 Normal wait: Sleeping 15s...")
                    time.sleep(15)

            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg:
                    retries += 1
                    wait_time = 20 * retries # Progressively wait longer (20s, 40s, 60s)
                    print(f"⚠️ Rate limit hit. Waiting {wait_time}s before retrying batch {batch_num}...")
                    time.sleep(wait_time)
                else:
                    print(f"❌ Fatal error in batch {batch_num}: {e}")
                    break # Break on non-rate-limit errors

    print("\n✅ Indexing Complete!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python script_name.py 'path/to/file.pdf'")
    else:
        index_pdf_with_genai_sdk(sys.argv[1])


        