from typing import TypedDict


class QAState(TypedDict):
    """State for the QA pipeline."""
    
    # Input
    question: str
    
    # Planning (NEW for Feature 1)
    plan: str | None
    sub_questions: list[str] | None
    
    # Retrieval
    context: str | None
    
    # Answer Generation
    answer: str | None