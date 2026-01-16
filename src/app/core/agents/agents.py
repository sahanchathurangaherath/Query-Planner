"""Agent node functions for the QA pipeline."""

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from .state import QAState
from .prompts import (
    QUERY_PLANNER_PROMPT,
    RETRIEVAL_PROMPT,
    SUMMARIZATION_PROMPT,
    VERIFICATION_PROMPT
)
from .tools import retrieval_tool


# Initialize LLMs
planner_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
retrieval_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
summarization_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
verification_llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)


def planning_node(state: QAState) -> dict:
    """
    Planning Agent: Analyzes the question and creates a search plan.
    """
    question = state["question"]
    
    print(f"\n🧠 PLANNING AGENT: Analyzing question...")
    
    response = planner_llm.invoke([
        SystemMessage(content=QUERY_PLANNER_PROMPT),
        HumanMessage(content=f"User Question: {question}")
    ])
    
    text = response.content
    print(f"Planning output:\n{text}\n")
    
    # Parse the response
    plan = ""
    sub_questions = []
    
    if "PLAN:" in text:
        parts = text.split("PLAN:")[1].split("SUB_QUESTIONS:")
        plan = parts[0].strip()
    
    if "SUB_QUESTIONS:" in text:
        lines = text.split("SUB_QUESTIONS:")[1].strip().split("\n")
        sub_questions = [
            line.replace("-", "").replace("•", "").strip() 
            for line in lines 
            if line.strip() and not line.strip().startswith("Example")
        ]
    
    return {
        "plan": plan,
        "sub_questions": sub_questions
    }


def retrieval_node(state: QAState) -> dict:
    """
    Retrieval Agent: Searches for relevant information using the plan.
    """
    question = state["question"]
    plan = state.get("plan", "")
    sub_questions = state.get("sub_questions", [])
    
    print(f"\n🔍 RETRIEVAL AGENT: Searching for information...")
    
    # Strategy 1: Search with original question
    print(f"   → Searching with original question...")
    context_parts = [retrieval_tool.invoke({"query": question})]
    
    # Strategy 2: Search with each sub-question (if planning generated them)
    if sub_questions and len(sub_questions) > 0:
        print(f"   → Searching with {len(sub_questions)} sub-questions...")
        for i, sub_q in enumerate(sub_questions[:3]):  # Limit to 3 for efficiency
            print(f"      • Sub-question {i+1}: {sub_q[:50]}...")
            result = retrieval_tool.invoke({"query": sub_q})
            context_parts.append(result)
    
    # Combine all context
    combined_context = "\n\n" + "="*60 + "\n\n".join(context_parts)
    
    print(f"✅ Retrieved context from {len(context_parts)} search(es)")
    
    return {"context": combined_context}


def summarization_node(state: QAState) -> dict:
    """
    Summarization Agent: Creates an answer from the context.
    """
    question = state["question"]
    context = state.get("context", "")
    
    print(f"\n✍️ SUMMARIZATION AGENT: Generating answer...")
    
    response = summarization_llm.invoke([
        SystemMessage(content=SUMMARIZATION_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nContext:\n{context}")
    ])
    
    answer = response.content
    print(f"Generated answer: {answer[:100]}...\n")
    
    return {"answer": answer}


def verification_node(state: QAState) -> dict:
    """
    Verification Agent: Verifies and potentially corrects the answer.
    """
    question = state["question"]
    answer = state.get("answer", "")
    context = state.get("context", "")
    
    print(f"\n✅ VERIFICATION AGENT: Checking answer quality...")
    
    response = verification_llm.invoke([
        SystemMessage(content=VERIFICATION_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nAnswer: {answer}\n\nContext: {context}")
    ])
    
    verified_answer = response.content
    print(f"Verification complete.\n")
    
    return {"answer": verified_answer}