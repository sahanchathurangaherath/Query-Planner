"""FastAPI application."""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
print("Environment variables loaded.")

# Verify API keys
if not os.getenv("GOOGLE_API_KEY"):
    raise ValueError("GOOGLE_API_KEY not found in environment variables")

from fastapi import FastAPI, HTTPException,UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from .models import QARequest, QAResponse
from .core.agents.graph import qa_graph
import shutil
from pathlib import Path
from .core.retrieval.vector_store import vector_store_manager


app = FastAPI(title="IKMS Query Planner", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def root():
    """Health check endpoint."""
    return {
        "status": "running",
        "message": "IKMS Query Planner API",
        "version": "1.0.0"
    }

@app.post("/index-pdf")
async def index_pdf(file: UploadFile = File(...)):
    """
    Upload and index a PDF file.
    
    The PDF will be:
    1. Split into chunks
    2. Embedded using OpenAI
    3. Stored in Pinecone
    """
    try:
        # Create uploads directory if it doesn't exist
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = upload_dir / file.filename
        
        with file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        print(f"\n📤 Uploaded file: {file.filename}")
        
        # Index the PDF
        result = vector_store_manager.index_pdf(str(file_path))
        
        return {
            "status": "success",
            "filename": file.filename,
            "pages": result["pages"],
            "chunks": result["chunks"],
            "message": f"Successfully indexed {result['chunks']} chunks from {result['pages']} pages"
        }
        
    except Exception as e:
        print(f"ERROR indexing PDF: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))






@app.post("/qa", response_model=QAResponse)
def question_answer(request: QARequest):
    """
    Main QA endpoint with query planning.
    
    Flow:
    1. Planning Agent creates search plan
    2. Retrieval Agent searches with enhanced queries
    3. Summarization Agent generates answer
    4. Verification Agent checks quality
    """
    try:
        print(f"\n{'='*60}")
        print(f"NEW QUESTION: {request.question}")
        print(f"{'='*60}")
        
        # Run the graph
        initial_state = {
            "question": request.question,
            "plan": None,
            "sub_questions": None,
            "context": None,
            "answer": None
        }
        
        final_state = qa_graph.invoke(initial_state)
        
        print(f"\n{'='*60}")
        print(f"FINAL ANSWER: {final_state['answer'][:100]}...")
        print(f"{'='*60}\n")
        
        return QAResponse(
            question=request.question,
            plan=final_state.get("plan"),
            sub_questions=final_state.get("sub_questions"),
            answer=final_state["answer"],
            context=final_state.get("context")
        )
    
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
    

    
@app.get("/index-stats")
def get_index_stats():
    """Get statistics about the Pinecone index."""
    try:
        index = vector_store_manager.pc.Index(vector_store_manager.index_name)
        stats = index.describe_index_stats()
        
        return {
            "index_name": vector_store_manager.index_name,
            "dimension": stats.get("dimension"),
            "total_vectors": stats.get("total_vector_count", 0),
            "namespaces": stats.get("namespaces", {})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/health")
def health_check():
    """Detailed health check."""
    # from .core.retrieval.vector_store import vector_store_manager
    # from pinecone import Pinecone
    # import os
    
    # # Get Pinecone stats
    # pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    # index = pc.Index(os.getenv("PINECONE_INDEX_NAME"))
    # stats = index.describe_index_stats()
    
    return {
        "status": "healthy",
        "feature": "Query Planning Agent",
        "llm_provider": "Google Gemini ",
        "embedding_provider": "Google Gemini text-embedding-004 ",
        "vector_database": "Pinecone (Cloud)",
        "components": {
            "planning_agent": True,
            "retrieval_agent": True,
            "summarization_agent": True,
            "verification_agent": True,
            "pinecone_connected": True,
            "semantic_search": True
        }
    }
