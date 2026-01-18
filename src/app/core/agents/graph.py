

from langgraph.graph import StateGraph, START, END
from .state import QAState
from .agents import (
    planning_node,
    retrieval_node,
    summarization_node,
    verification_node
)


def create_qa_graph():
    """Create the QA workflow graph."""
    
    # Initialize graph
    graph = StateGraph(QAState)
    
    # Add nodes
    graph.add_node("planning", planning_node)
    graph.add_node("retrieval", retrieval_node)
    graph.add_node("summarization", summarization_node)
    graph.add_node("verification", verification_node)
    
    # Define flow: START → planning → retrieval → summarization → verification → END
    graph.add_edge(START, "planning")
    graph.add_edge("planning", "retrieval")
    graph.add_edge("retrieval", "summarization")
    graph.add_edge("summarization", "verification")
    graph.add_edge("verification", END)
    
    return graph.compile()


# Create the compiled graph
qa_graph = create_qa_graph()