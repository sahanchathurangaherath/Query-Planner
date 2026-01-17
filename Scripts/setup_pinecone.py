"""
ONE-TIME SETUP: Index documents into Pinecone with Gemini embeddings.
Run this ONCE before using the system.

Cost: $0.00 (Gemini embeddings are FREE!)
"""

from pinecone import Pinecone
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document
from dotenv import load_dotenv
import os

load_dotenv()

# Sample documents about vector databases
DOCUMENTS = [
    {
        "content": """Vector databases are specialized database systems designed to store and query 
        high-dimensional vectors efficiently. They are commonly used in machine learning applications, 
        particularly for similarity search and recommendation systems. Unlike traditional databases that 
        store structured data in rows and columns, vector databases store embedding vectors that represent 
        semantic meaning.""",
        "metadata": {"page": 1, "source": "vector_db_guide.pdf", "topic": "introduction"}
    },
    {
        "content": """The main advantages of vector databases include: Fast similarity search using 
        approximate nearest neighbor (ANN) algorithms, efficient handling of high-dimensional data, 
        semantic search capabilities that understand meaning beyond exact keyword matching, scalability 
        for handling billions of vectors, and integration with modern AI/ML pipelines.""",
        "metadata": {"page": 2, "source": "vector_db_guide.pdf", "topic": "advantages"}
    },
    {
        "content": """Common indexing strategies in vector databases include HNSW (Hierarchical Navigable 
        Small World), which creates a multi-layer graph structure for efficient search. LSH (Locality-Sensitive 
        Hashing) uses hash functions to map similar vectors to the same buckets. IVF (Inverted File Index) 
        partitions the vector space into clusters for faster search.""",
        "metadata": {"page": 3, "source": "vector_db_guide.pdf", "topic": "indexing"}
    },
    {
        "content": """Vector databases handle scalability through several mechanisms: horizontal sharding 
        to distribute data across multiple nodes, quantization to reduce memory footprint, and optimized 
        indexing structures that maintain performance as data grows. Many vector databases support distributed 
        architectures that can scale to billions of vectors.""",
        "metadata": {"page": 4, "source": "vector_db_guide.pdf", "topic": "scalability"}
    },
    {
        "content": """Compared to traditional relational databases, vector databases excel at finding similar 
        items based on semantic meaning rather than exact matches. Traditional databases use SQL queries and 
        B-tree indexes, while vector databases use distance metrics like cosine similarity or Euclidean distance. 
        This makes vector databases ideal for AI applications like RAG, recommendation engines, and semantic search.""",
        "metadata": {"page": 5, "source": "vector_db_guide.pdf", "topic": "comparison"}
    },
    {
        "content": """HNSW indexing provides logarithmic search complexity and excellent recall rates. It builds 
        a hierarchical graph where each layer has progressively fewer nodes. During search, the algorithm starts 
        at the top layer and navigates down, finding approximate nearest neighbors efficiently. HNSW is particularly 
        good for high-dimensional data and offers a good balance between speed and accuracy.""",
        "metadata": {"page": 6, "source": "vector_db_guide.pdf", "topic": "hnsw"}
    },
    {
        "content": """Popular vector database solutions include Pinecone (managed cloud service), Weaviate 
        (open-source with GraphQL API), Milvus (highly scalable open-source), Qdrant (Rust-based with good performance), 
        and Chroma (lightweight and developer-friendly). Each has different trade-offs in terms of performance, 
        features, and ease of use.""",
        "metadata": {"page": 7, "source": "vector_db_guide.pdf", "topic": "solutions"}
    },
    {
        "content": """Vector databases are essential for RAG (Retrieval-Augmented Generation) systems. In RAG, 
        documents are converted to embeddings and stored in a vector database. When a user asks a question, 
        the question is embedded and similar document chunks are retrieved. These chunks provide context to 
        the language model, enabling it to generate accurate, grounded answers.""",
        "metadata": {"page": 8, "source": "vector_db_guide.pdf", "topic": "rag"}
    },
    {
        "content": """Embedding models convert text into dense vector representations. Common models include 
        OpenAI's text-embedding-ada-002, sentence-transformers from HuggingFace, and Cohere embeddings. The 
        dimensionality typically ranges from 384 to 1536 dimensions. Higher dimensions can capture more nuance 
        but require more storage and computation.""",
        "metadata": {"page": 9, "source": "vector_db_guide.pdf", "topic": "embeddings"}
    },
    {
        "content": """Performance optimization in vector databases involves techniques like product quantization 
        (PQ) to compress vectors, filtering to reduce search space, and caching for frequently accessed vectors. 
        Batch operations and asynchronous indexing can improve throughput. Monitoring query latency and recall 
        metrics helps tune the system for optimal performance.""",
        "metadata": {"page": 10, "source": "vector_db_guide.pdf", "topic": "performance"}
    },
    {
        "content": """Machine learning applications using vector databases include recommendation systems that 
        suggest similar products, anomaly detection that finds outliers in high-dimensional data, image similarity 
        search for finding visually similar images, and question-answering systems that retrieve relevant context.""",
        "metadata": {"page": 11, "source": "ml_applications.pdf", "topic": "applications"}
    },
    {
        "content": """Concurrent write handling in vector databases is challenging due to index updates. Some 
        systems use write-ahead logs, others implement optimistic locking or MVCC (Multi-Version Concurrency Control). 
        Real-time systems may sacrifice some consistency for availability, while batch-oriented systems can rebuild 
        indexes periodically.""",
        "metadata": {"page": 12, "source": "ml_applications.pdf", "topic": "concurrency"}
    },
    {
        "content": """LSH (Locality-Sensitive Hashing) works by using hash functions that map similar vectors 
        to the same hash buckets with high probability. This allows approximate nearest neighbor search in 
        sublinear time. While LSH is fast, it may have lower recall compared to graph-based methods like HNSW.""",
        "metadata": {"page": 13, "source": "vector_db_guide.pdf", "topic": "lsh"}
    },
    {
        "content": """IVF (Inverted File Index) partitions the vector space using clustering algorithms like k-means. 
        During search, only the nearest clusters are examined, reducing the search space. IVF combined with 
        product quantization (IVFPQ) is widely used in production systems for billion-scale vector search.""",
        "metadata": {"page": 14, "source": "vector_db_guide.pdf", "topic": "ivf"}
    },
    {
        "content": """Distance metrics in vector databases include cosine similarity (measures angle between vectors), 
        Euclidean distance (measures straight-line distance), and dot product (measures alignment). The choice of 
        metric depends on whether vectors are normalized and the specific use case requirements.""",
        "metadata": {"page": 15, "source": "vector_db_guide.pdf", "topic": "metrics"}
    },
]


def setup_pinecone():
    """Index sample documents into Pinecone using FREE Gemini embeddings."""
    
    print("\n" + "="*70)
    print("📄 ONE-TIME PINECONE SETUP")
    print("="*70)
    print("\n💡 Using FREE Gemini embeddings - No cost!\n")
    
    # Initialize
    index_name = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")
    
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004",
        google_api_key=os.getenv("GOOGLE_API_KEY")
    )
    
    # Convert to documents
    docs = [
        Document(
            page_content=doc["content"],
            metadata=doc["metadata"]
        )
        for doc in DOCUMENTS
    ]
    
    print(f"📊 Preparing to index {len(docs)} documents...")
    print(f"🔄 Generating embeddings with Gemini (FREE)...")
    
    # Index in Pinecone
    PineconeVectorStore.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=index_name
    )
    
    print(f"\n✅ SUCCESS! Indexed {len(docs)} documents in Pinecone")
    print(f"💰 Cost: $0.00 (Gemini embeddings are free!)")
    print(f"\n🚀 You can now run your application:")
    print(f"   uvicorn src.app.api:app --reload")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    setup_pinecone()