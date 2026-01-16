"""
Streamlit Frontend for IKMS Query Planner
Feature 1: Query Planning & Decomposition Agent
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Configuration
API_URL = "http://localhost:8000"

# Page config
st.set_page_config(
    page_title="IKMS Query Planner",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .plan-box {
        background-color: #f0f8ff;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .sub-question {
        background-color: #fff4e6;
        padding: 0.8rem;
        border-radius: 5px;
        margin: 0.5rem 0;
        border-left: 3px solid #ff9800;
    }
    .answer-box {
        background-color: #f0fff4;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #4caf50;
        margin: 1rem 0;
    }
    .context-box {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 5px;
        border: 1px solid #ddd;
        max-height: 400px;
        overflow-y: auto;
    }
    .metric-card {
        background-color: #ffffff;
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🧠 IKMS Query Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Multi-Agent RAG with Intelligent Query Decomposition</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    **Feature 1: Query Planning Agent**
    
    This system uses a multi-agent approach:
    
    1. 🧠 **Planning Agent** - Analyzes your question and creates a search strategy
    2. 🔍 **Retrieval Agent** - Performs multiple targeted searches
    3. ✍️ **Summarization Agent** - Generates comprehensive answer
    4. ✅ **Verification Agent** - Ensures quality and accuracy
    """)
    
    st.divider()
    
    # API Health Check
    st.subheader("🔧 System Status")
    try:
        response = requests.get(f"{API_URL}/health", timeout=2)
        if response.status_code == 200:
            health = response.json()
            st.success("✅ API Connected")
            st.metric("Documents", health.get("documents_loaded", "N/A"))
        else:
            st.error("❌ API Error")
    except:
        st.error("❌ API Offline")
        st.info("Make sure backend is running:\n```\nuvicorn src.app.api:app --reload\n```")
    
    st.divider()
    
    # Example Questions
    st.subheader("💡 Example Questions")
    example_questions = [
        "What are vector databases?",
        "How do vector databases compare to traditional databases?",
        "What are the advantages of vector databases and how do they handle scalability?",
        "Explain HNSW indexing in vector databases",
        "What are the main applications of vector databases in machine learning?"
    ]
    
    for i, eq in enumerate(example_questions):
        if st.button(f"📝 {eq[:40]}...", key=f"example_{i}", use_container_width=True):
            st.session_state.question = eq

# Main content
st.divider()

# Question input
question = st.text_area(
    "❓ Ask a Question",
    value=st.session_state.get("question", ""),
    height=100,
    placeholder="Enter your question here... (e.g., What are the advantages of vector databases compared to traditional databases?)",
    key="question_input"
)

# Update session state
if question:
    st.session_state.question = question

# Submit button
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    submit_button = st.button("🚀 Generate Answer", type="primary", use_container_width=True)

# Process question
if submit_button and question:
    with st.spinner("🔄 Processing your question through the multi-agent pipeline..."):
        try:
            # Call API
            response = requests.post(
                f"{API_URL}/qa",
                json={"question": question},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                
                # Display results in tabs
                tab1, tab2, tab3, tab4 = st.tabs(["📊 Overview", "🧠 Planning", "🔍 Context", "📄 Full Response"])
                
                with tab1:
                    st.subheader("📊 Query Processing Overview")
                    
                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Sub-Questions Generated", len(result.get("sub_questions", [])))
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col2:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        st.metric("Retrieval Calls", len(result.get("sub_questions", [])) + 1)
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    with col3:
                        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                        answer_length = len(result.get("answer", "").split())
                        st.metric("Answer Length", f"{answer_length} words")
                        st.markdown('</div>', unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Final Answer
                    st.markdown("### ✅ Final Answer")
                    st.markdown(f'<div class="answer-box">{result.get("answer", "No answer generated")}</div>', unsafe_allow_html=True)
                
                with tab2:
                    st.subheader("🧠 Query Planning & Decomposition")
                    
                    # Search Plan
                    if result.get("plan"):
                        st.markdown("#### 📋 Search Strategy")
                        st.markdown(f'<div class="plan-box">{result["plan"]}</div>', unsafe_allow_html=True)
                    
                    # Sub-questions
                    if result.get("sub_questions"):
                        st.markdown("#### 🔍 Decomposed Sub-Questions")
                        st.markdown("The planning agent broke your complex question into these focused searches:")
                        
                        for i, sq in enumerate(result["sub_questions"], 1):
                            st.markdown(f'<div class="sub-question"><strong>Sub-Question {i}:</strong> {sq}</div>', unsafe_allow_html=True)
                    
                    # Pipeline visualization
                    st.markdown("#### 🔄 Processing Pipeline")
                    st.markdown("""
```
                    📥 User Question
                        ↓
                    🧠 Planning Agent (creates search plan)
                        ↓
                    🔍 Retrieval Agent (multiple targeted searches)
                        ↓
                    ✍️ Summarization Agent (generates answer)
                        ↓
                    ✅ Verification Agent (ensures quality)
                        ↓
                    📤 Final Answer
```
                    """)
                
                with tab3:
                    st.subheader("🔍 Retrieved Context")
                    
                    if result.get("context"):
                        st.markdown("#### 📚 Documents Retrieved")
                        st.markdown("These are the document chunks retrieved to answer your question:")
                        st.markdown(f'<div class="context-box">{result["context"]}</div>', unsafe_allow_html=True)
                        
                        # Context stats
                        context_length = len(result["context"].split())
                        num_chunks = result["context"].count("[C")
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Total Context Words", context_length)
                        with col2:
                            st.metric("Document Chunks", num_chunks)
                    else:
                        st.info("No context available")
                
                with tab4:
                    st.subheader("📄 Complete API Response")
                    st.json(result)
                
                # Success message
                st.success("✅ Question processed successfully!")
                
            else:
                st.error(f"❌ API Error: {response.status_code}")
                st.code(response.text)
                
        except requests.exceptions.Timeout:
            st.error("⏱️ Request timeout. The query might be too complex or the API is slow.")
        except requests.exceptions.ConnectionError:
            st.error("🔌 Cannot connect to API. Make sure the backend is running on http://localhost:8000")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><strong>IKMS Query Planner</strong> - Feature 1: Query Planning & Decomposition Agent</p>
    <p>Multi-Agent RAG System with Intelligent Search Strategy</p>
</div>
""", unsafe_allow_html=True)