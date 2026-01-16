"""Pydantic models for API requests and responses."""

from pydantic import BaseModel
from typing import List, Optional


class QARequest(BaseModel):
    """Request model for QA endpoint."""
    question: str


class QAResponse(BaseModel):
    """Response model for QA endpoint."""
    question: str
    plan: Optional[str] = None
    sub_questions: Optional[List[str]] = None
    answer: str
    context: Optional[str] = None