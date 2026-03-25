"""
INDEXING SCRIPT: Local embeddings on laptop GPU (RTX 4060 8GB) + Pinecone
Recommended model: BAAI/bge-base-en-v1.5
- Uses CUDA if available
- Falls back to CPU if needed
- Keeps Pinecone dimension compatible (768)
"""

import os
import sys
import time
import math
import hashlib
from typing import List

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()

# =========================
# CONFIG
# =========================
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "ikms-rag")
PINECONE_NAMESPACE = os.getenv("PINECONE_NAMESPACE", "")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

LOCAL_EMBED_MODEL = os.getenv("LOCAL_EMBED_MODEL", "BAAI/bge-base-en-v1.5")
EMBED_DIM = int(os.getenv("EMBED_DIM", "768"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

EMBED_BATCH_SIZE = int(os.getenv("EMBED_BATCH_SIZE", "32"))
UPSERT_BATCH_SIZE = int(os.getenv("UPSERT_BATCH_SIZE", "100"))

# =========================
# CHECK ENV
# =========================
if not PINECONE_API_KEY:
    print("❌ Error: PINECONE_API_KEY not found in .env")
    sys.exit(1)

# =========================
# PINECONE
# =========================
pc = Pinecone(api_key=PINECONE_API_KEY)


def print_header(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def get_file_hash(file_path: str) -> str:
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def get_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def load_embedding_model():
    device = get_device()
    print(f"🔧 Loading embedding model: {LOCAL_EMBED_MODEL}")
    print(f"🖥️ Device: {device}")

    model = SentenceTransformer(LOCAL_EMBED_MODEL, device=device)

    # quick dimension check
    test_vec = model.encode(["test"], normalize_embeddings=True)
    actual_dim = len(test_vec[0])

    print(f"📏 Embedding dimension from model: {actual_dim}")

    if actual_dim != EMBED_DIM:
        raise ValueError(
            f"Embedding dimension mismatch. Model outputs {actual_dim}, "
            f"but EMBED_DIM is set to {EMBED_DIM}."
        )

    return model, device


def ensure_index_exists(index_name: str, dimension: int):
    existing_indexes = [idx["name"] for idx in pc.list_indexes()]
    if index_name in existing_indexes:
        print(f"✅ Pinecone index already exists: {index_name}")
        return

    print(f"🛠️ Creating Pinecone index: {index_name}")
    print(f"   Dimension: {dimension}")
    print(f"   Metric: cosine")
    print(f"   Cloud: {PINECONE_CLOUD}")
    print(f"   Region: {PINECONE_REGION}")

    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=ServerlessSpec(
            cloud=PINECONE_CLOUD,
            region=PINECONE_REGION,
        ),
    )

    print("⏳ Waiting for index to be ready...")
    while True:
        info = pc.describe_index(index_name)
        ready = info.status.get("ready", False)
        if ready:
            break
        time.sleep(2)

    print(f"✅ Index ready: {index_name}")


def get_index_dimension(index_name: str):
    try:
        info = pc.describe_index(index_name)
        return getattr(info, "dimension", None)
    except Exception as e:
        print(f"⚠️ Could not get index dimension: {e}")
        return None


def check_index_stats():
    try:
        index = pc.Index(PINECONE_INDEX_NAME)
        stats = index.describe_index_stats()

        print_header("📊 CURRENT INDEX STATUS")
        print(f"Index: {PINECONE_INDEX_NAME}")
        print(f"Namespace: {PINECONE_NAMESPACE or '(default)'}")
        print(f"Total vectors: {stats.get('total_vector_count', 0)}")
        print(f"Dimension: {stats.get('dimension', 'unknown')}")
        print("=" * 70 + "\n")
    except Exception as e:
        print(f"⚠️ Could not check index stats: {e}")


def load_and_split_pdf(pdf_path: str):
    print(f"📖 Loading PDF: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages")

    print("✂️ Splitting into chunks...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    print(f"✅ Created {len(chunks)} chunks")

    return documents, chunks


def enrich_chunk_metadata(chunks, pdf_path: str):
    file_name = os.path.basename(pdf_path)
    file_hash = get_file_hash(pdf_path)

    for i, chunk in enumerate(chunks):
        chunk.metadata["source_file"] = file_name
        chunk.metadata["source_path"] = pdf_path
        chunk.metadata["file_hash"] = file_hash
        chunk.metadata["chunk_index"] = i

    return chunks, file_hash


def make_vector_id(file_hash: str, chunk_index: int) -> str:
    return f"{file_hash}_{chunk_index}"


def embed_texts(model: SentenceTransformer, texts: List[str], batch_size: int):
    # BGE models work well with normalized embeddings for cosine similarity
    return model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )


def index_pdf_local(pdf_path: str):
    print_header("🚀 PINECONE INDEXING WITH LOCAL GPU EMBEDDINGS")

    if not os.path.exists(pdf_path):
        print(f"❌ Error: File not found: {pdf_path}")
        return False

    if not pdf_path.lower().endswith(".pdf"):
        print("❌ Error: File must be a PDF")
        return False

    try:
        print(f"PDF: {pdf_path}")
        print(f"Index: {PINECONE_INDEX_NAME}")
        print(f"Namespace: {PINECONE_NAMESPACE or '(default)'}")
        print(f"Embedding model: {LOCAL_EMBED_MODEL}")
        print(f"Expected dimension: {EMBED_DIM}")
        print(f"Embed batch size: {EMBED_BATCH_SIZE}")
        print(f"Upsert batch size: {UPSERT_BATCH_SIZE}\n")

        # 1. Load local model
        model, device = load_embedding_model()

        # 2. Ensure Pinecone index exists
        ensure_index_exists(PINECONE_INDEX_NAME, EMBED_DIM)

        # 3. Confirm index dimension
        actual_dim = get_index_dimension(PINECONE_INDEX_NAME)
        print(f"📏 Pinecone index dimension: {actual_dim}")

        if actual_dim is not None and actual_dim != EMBED_DIM:
            print(
                f"❌ Dimension mismatch: Pinecone index is {actual_dim}, "
                f"but local embedding model is {EMBED_DIM}"
            )
            print("   Delete/recreate the index or use a model with matching dimension.")
            return False

        # 4. Load and split PDF
        documents, chunks = load_and_split_pdf(pdf_path)
        if not chunks:
            print("❌ No chunks created from PDF.")
            return False

        chunks, file_hash = enrich_chunk_metadata(chunks, pdf_path)

        print("\n📝 Sample chunk:")
        print(f"   Page: {chunks[0].metadata.get('page', 'unknown')}")
        print(f"   Chunk index: {chunks[0].metadata.get('chunk_index')}")
        print(f"   Text: {chunks[0].page_content[:150]}...")

        # 5. Embed all chunks in local batches
        print("\n🧠 Generating embeddings locally...")
        all_vectors = []

        total_embed_batches = math.ceil(len(chunks) / EMBED_BATCH_SIZE)
        for i in range(0, len(chunks), EMBED_BATCH_SIZE):
            batch = chunks[i:i + EMBED_BATCH_SIZE]
            batch_num = (i // EMBED_BATCH_SIZE) + 1

            texts = [doc.page_content for doc in batch]

            print(f"⚙️ Embedding batch {batch_num}/{total_embed_batches} ({len(batch)} chunks)")
            embeddings = embed_texts(model, texts, EMBED_BATCH_SIZE)

            for j, emb in enumerate(embeddings):
                chunk = batch[j]
                vector_id = make_vector_id(file_hash, chunk.metadata["chunk_index"])

                metadata = dict(chunk.metadata)
                metadata["text"] = chunk.page_content

                all_vectors.append({
                    "id": vector_id,
                    "values": emb.tolist(),
                    "metadata": metadata
                })

        print(f"✅ Generated {len(all_vectors)} vectors locally on {device}")

        # 6. Upsert to Pinecone
        print("\n📤 Uploading vectors to Pinecone...")
        index = pc.Index(PINECONE_INDEX_NAME)

        total_upsert_batches = math.ceil(len(all_vectors) / UPSERT_BATCH_SIZE)
        for i in range(0, len(all_vectors), UPSERT_BATCH_SIZE):
            batch = all_vectors[i:i + UPSERT_BATCH_SIZE]
            batch_num = (i // UPSERT_BATCH_SIZE) + 1

            print(f"📦 Upserting batch {batch_num}/{total_upsert_batches} ({len(batch)} vectors)")
            index.upsert(
                vectors=batch,
                namespace=PINECONE_NAMESPACE if PINECONE_NAMESPACE else None,
            )

        print("\n✅ Indexing complete!")
        print(f"   Pages: {len(documents)}")
        print(f"   Chunks: {len(chunks)}")
        print(f"   Index: {PINECONE_INDEX_NAME}")
        print(f"   Namespace: {PINECONE_NAMESPACE or '(default)'}")
        print(f"   Model: {LOCAL_EMBED_MODEL}")
        print(f"   Device used: {device}")
        print("=" * 70 + "\n")

        return True

    except Exception as e:
        print(f"\n❌ Error during indexing: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("INDEXING ......")

    check_index_stats()

    if len(sys.argv) < 2:
        print("Usage: python scripts/setup_pinecone_local.py <path_to_pdf>")
        print("\nExample:")
        print('  python scripts/setup_pinecone_local.py "documents/AI.pdf"')
        sys.exit(1)

    pdf_path = sys.argv[1]

    success = index_pdf_local(pdf_path)

    if success:
        check_index_stats()
    else:
        sys.exit(1)