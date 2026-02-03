

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from .state import QAState
from .prompts import (
    QUERY_PLANNER_PROMPT,
    SUMMARIZATION_PROMPT,
    VERIFICATION_PROMPT
)
from .tools import retrieval_tool


# Initialize Gemini LLMs 
planner_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    convert_system_message_to_human=True
)

summarization_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    convert_system_message_to_human=True
)

verification_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
    convert_system_message_to_human=True
)


def planning_node(state: QAState) -> dict:
    """Planning Agent: Analyzes question and creates search plan."""
    question = state["question"]
    
    print(f"\n PLANNING AGENT: Analyzing question...")
    
    prompt = f"{QUERY_PLANNER_PROMPT}\n\nUser Question: {question}"
    response = planner_llm.invoke([HumanMessage(content=prompt)])
    
    text = response.content
    print(f"Planning output:\n{text}\n")
    
    # Parse response
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
    """Retrieval Agent: Multi-query search in Pinecone."""
    question = state["question"]
    plan = state.get("plan", "")
    sub_questions = state.get("sub_questions", [])
    
    print(f"\n RETRIEVAL AGENT: Searching Pinecone...")
    
    # Strategy 1: Search with original question
    print(f"   → Search 1: Original question")
    context_parts = [retrieval_tool.invoke({"query": question})]
    
    # Strategy 2: Search with sub-questions
    if sub_questions:
        print(f"   → Searches 2-{len(sub_questions)+1}: Sub-questions")
        for i, sub_q in enumerate(sub_questions[:3], 2):
            print(f"      • Search {i}: {sub_q[:50]}...")
            result = retrieval_tool.invoke({"query": sub_q})
            context_parts.append(result)
    
    # Combine all context
    combined_context = "\n\n" + ("="*60 + "\n\n").join(context_parts)
    
    print(f" Completed {len(context_parts)} Pinecone searches")
    
    return {"context": combined_context}


def summarization_node(state: QAState) -> dict:
    """Summarization Agent: Generate answer."""
    question = state["question"]
    context = state.get("context", "")
    
    print(f"\n SUMMARIZATION AGENT: Generating answer...")
    
    prompt = f"{SUMMARIZATION_PROMPT}\n\nQuestion: {question}\n\nContext:\n{context}"
    response = summarization_llm.invoke([HumanMessage(content=prompt)])
    
    answer = response.content
    print(f"Generated answer: {answer[:100]}...\n")
    
    return {"answer": answer}


}

def verification_node(state: QAState) -> dict:
    """
    Verification Agent: Validates and refines the answer.
    Returns the final polished answer.
    """
    question = state["question"]
    answer = state.get("answer", "")
    context = state.get("context", "")
    
    print(f"\n VERIFICATION AGENT: Reviewing answer quality...")
    
    # Improved prompt that focuses on output quality
    verification_prompt = f"""You are a Quality Verification Agent. Review the answer below and return the FINAL ANSWER.

Question:
{question}

Current Answer:
{answer}

Available Context:
{context[:500]}...

Your task:
1. If the answer is accurate and complete → Return it EXACTLY as written
2. If the answer needs improvement → Return an IMPROVED version
3. Remove any meta-commentary like "based on the context"
4. Ensure the answer directly addresses the question

IMPORTANT: Return ONLY the final answer text. Do NOT include:
- Your analysis or reasoning
- Phrases like "The answer is accurate" or "Return as-is"
- Meta-commentary about the answer quality

Final Answer:"""
    
    response = verification_llm.invoke([HumanMessage(content=verification_prompt)])
    verified_answer = response.content.strip()
    
    # Safety check: detect if LLM returned analysis instead of answer
    meta_phrases = [
        "the answer is",
        "return the answer",
        "as-is",
        "accurate and complete",
        "well-structured",
        "directly supported"
    ]
    
    # If response contains meta-commentary, use original answer
    if any(phrase in verified_answer.lower() for phrase in meta_phrases):
        print(" Verification returned analysis - using original answer")
        verified_answer = answer
    else:
        print(f" Verification complete - {len(verified_answer)} characters")
    
    print()
    
    return {"answer": verified_answer}



