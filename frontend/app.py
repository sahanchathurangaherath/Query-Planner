"""
IKMS Query Planner UI
Feature 1: Query Planning & Decomposition Agent
"""

import streamlit as st
import requests
import time

API_URL = "http://localhost:8000"

#  Page Config 
st.set_page_config(
    page_title="IKMS Query Planner",
    page_icon="🧠",
    layout="wide"
)

# Styles 
st.markdown("""
<style>
.header {
    font-size: 2.4rem;
    font-weight: 700;
    text-align: center;
}

.sub {
    text-align: center;
    color: #9ca3af;   /* better for dark mode */
    margin-bottom: 1.5rem;
}

.box {
    padding: 1rem;
    border-radius: 8px;
    margin-bottom: 1rem;
}


.plan {
    background: #102a43;
    color: #e3f2fd;          
    border-left: 4px solid #1f77b4;
}


.answer {
    background: #f0fff4;
    color: #1b1b1b;          
    border-left: 4px solid #2e7d32;
}

/* 🔍 Sub-Questions */
.subq {
    background: #fff4e6;
    color: #3e2723;          /* ✅ ADD THIS */
    border-left: 3px solid #ff9800;
}
</style>

""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown('<div class="header">🧠 IKMS Query Planner</div>', unsafe_allow_html=True)
st.markdown('<div class="sub">Multi-Agent RAG with Query Planning</div>', unsafe_allow_html=True)

# ---------------- Sidebar ----------------
with st.sidebar:
    st.subheader("📘 Feature Overview")
    st.markdown("""
** Query Planning Agent**

Pipeline:
1. 🧠 Plan the query  
2. 🔍 Retrieve per sub-question  
3. ✍️ Summarize  
4. ✅ Verify  
""")

    st.divider()

    st.subheader("🔧 Backend Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=5)
        
        if r.status_code == 200:
            health = r.json()
            st.success("API Connected")
        else:
            st.warning("API reachable but unhealthy")
    except:
        st.error("API Offline")

    st.divider()

    
# Input 
question = st.text_area(
    "❓ Ask a Question",
    value=st.session_state.get("question", ""),
    height=90
)

if st.button("🚀 Run Query", type="primary"):
    if not question.strip():
        st.warning("Please enter a question.")
        st.stop()

    with st.spinner("Running multi-agent pipeline..."):
        start = time.time()
        res = requests.post(
            f"{API_URL}/qa",
            json={"question": question},
            timeout=60
        )
        elapsed = time.time() - start

    if res.status_code != 200:
        st.error("API Error")
        st.code(res.text)
        st.stop()

    data = res.json()
    st.success(f"Completed in {elapsed:.1f}s")

    #  Answer
    st.markdown("### ✅ Final Answer")
    st.markdown(
        f'<div class="box answer">{data.get("answer","No answer")}</div>',
        unsafe_allow_html=True
    )

    # Planning 
    st.markdown("### 🧠 Query Planning")

    if data.get("plan"):
        st.markdown(
            f'<div class="box plan">{data["plan"]}</div>',
            unsafe_allow_html=True
        )
    else:
        st.info("Planner not triggered")

    if data.get("sub_questions"):
        st.markdown("#### 🔍 Decomposed Sub-Questions")
        for i, sq in enumerate(data["sub_questions"], 1):
            st.markdown(
                f'<div class="box subq"><b>{i}.</b> {sq}</div>',
                unsafe_allow_html=True
            )

    # Context 
    if data.get("context"):
        with st.expander("📚 Retrieved Context"):
            st.text(data["context"])


# Footer 
st.markdown(
    "<center style='color:#888'>IKMS • Query Planning & Decomposition</center>",
    unsafe_allow_html=True
)
