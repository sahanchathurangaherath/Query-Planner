"""FastAPI application."""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .models import QARequest, QAResponse
from .core.agents.graph import qa_graph
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Verify API keys
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables")

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