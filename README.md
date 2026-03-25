# IKMS Query Planner -  Query Planning & Decomposition Agent

An intelligent Multi-Agent RAG system that uses AI-powered query planning to break down complex questions into focused sub-queries for comprehensive answers. Built with Google Gemini (free tier) and Pinecone vector database.

---

## 🎯  Query Planning & Decomposition

### Problem
Traditional RAG systems perform a single retrieval call with the entire user question, often missing important aspects of complex, multi-part queries.

### Solution
An intelligent **Query Planning Agent** that:
1. ✅ Analyzes question structure and complexity
2. ✅ Creates a structured search strategy
3. ✅ Decomposes complex questions into focused sub-questions
4. ✅ Performs multiple targeted retrieval calls
5. ✅ Aggregates results for comprehensive answers

### Example

**Question:**
> "How do vector databases compare to traditional databases and how do they handle scalability?"

**Without Feature 1:**
- 1 generic search with entire question
- May miss specific details about advantages OR scalability
- Incomplete answer

**With Feature 1:**
- Creates search plan
- Generates 4 focused sub-questions:
  - "vector database advantages"
  - "vector database vs traditional database"
  - "vector database scalability"
  - "traditional database comparison"
- 5 total searches (1 original + 4 sub-questions)
- ✅ Comprehensive, detailed answer

---

## 🏗️ Architecture

### Multi-Agent Pipeline
```
User Question
     ↓
🧠 Planning Agent (Gemini 1.5 Flash)
     ├─ Analyzes question complexity
     ├─ Creates search strategy
     └─ Generates sub-questions
     ↓
🔍 Retrieval Agent (Multiple Searches)
     ├─ Query 1: Original question → Pinecone
     ├─ Query 2: Sub-question 1 → Pinecone
     ├─ Query 3: Sub-question 2 → Pinecone
     └─ Query 4: Sub-question 3 → Pinecone
     ↓
✍️ Summarization Agent 
     └─ Generates comprehensive answer
     ↓
✅ Verification Agent 
     └─ Validates quality
     ↓
Final Answer
```

### Technology Stack

- **LLM**: Google Gemini 1.5 Flash (FREE tier)
- **Embeddings**: Google Gemini embedding-001 (FREE)
- **Vector Database**: Pinecone (Cloud)
- **Framework**: LangChain v1.0 + LangGraph
- **Backend**: FastAPI
- **Frontend**: Streamlit
- **PDF Processing**: PyMuPDF

---



## 📋 Prerequisites

- Python 3.10+
- Google Gemini API key 
- Pinecone API key 

---

## 🚀 Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/Thilanjith/Query-Planning-Decomposition-Agent.git
cd ikms-query-planner
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate

# Mac/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Get API Keys

#### Google Gemini API Key (FREE):
1. Go to: https://aistudio.google.com/app/apikey
2. Click "Create API Key"
3. Copy your key (starts with `AIza...`)

#### Pinecone API Key (FREE):
1. Go to: https://app.pinecone.io/
2. Sign up for free account
3. Create a new project
4. Copy your API key from "API Keys" section

#### Create Pinecone Index:
1. In Pinecone console, click "Create Index"
2. Settings:
   - **Name**: `ikms-rag`
   - **Dimensions**: `768` (for Gemini embeddings)
   - **Metric**: `cosine`
   - **Cloud**: Choose free tier
3. Click "Create Index"

### 5. Configure Environment

Create `.env` file in project root:
```bash
touch .env
```

Add your API keys:
```env
# Google Gemini API Key 
GOOGLE_API_KEY=your-gemini-api-key-here

# Pinecone Configuration 
PINECONE_API_KEY=your-pinecone-api-key-here
PINECONE_INDEX_NAME=ikms-rag
```

**⚠️ Important:** Never commit `.env` to Git (already in `.gitignore`)

### 6. Index Your Documents
```bash
# Create scripts directory if not exists
mkdir -p scripts

# Index a PDF (one-time setup)
python scripts/setup_pinecone.py path/to/your/document.pdf
```

**Example:**
```bash
python scripts/setup_pinecone.py documents/research_paper.pdf
```

**Output:**
```
📄 PINECONE PDF INDEXING
PDF: documents/research_paper.pdf
💡 Using FREE Gemini embeddings - No cost!

✅ Loaded 25 pages
✅ Created 87 chunks
🔄 Indexing into Pinecone...
✅ SUCCESS! Indexed 87 chunks

```

### 7. Start Backend
```bash
uvicorn src.app.api:app --reload
```

Backend runs at: http://localhost:8000

**API Docs**: http://localhost:8000/docs

### 8. Start Frontend

In a new terminal:
```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Start Streamlit
streamlit run frontend/app.py
```

Frontend opens at: http://localhost:8501

---

## 📡 API Endpoints

### Health Check
```bash
GET http://localhost:8000/health
```

**Response:**
```json
{
  "status": "healthy",
  "feature": "Query Planning Agent",
  "llm_provider": "Google Gemini 1.5 Flash (FREE)",
  "embedding_provider": "Google Gemini embedding-001 (FREE)",
  "vector_database": "Pinecone (Cloud)",
  "total_vectors": 87
}
```

### Question Answering
```bash
POST http://localhost:8000/qa
Content-Type: application/json

{
  "question": "What are vector databases?"
}
```

**Response:**
```json
{
  "question": "What are vector databases?",
  "plan": "The question asks for a definition...",
  "sub_questions": [
    "vector database definition",
    "vector database purpose",
    "vector database features"
  ],
  "answer": "Vector databases are specialized...",
  "context": "[C1] (Page 1, source.pdf)\n..."
}
```

### Example with Python
```python
import requests

response = requests.post(
    "http://localhost:8000/qa",
    json={"question": "How do vector databases work?"}
)

result = response.json()
print(f"Plan: {result['plan']}")
print(f"Sub-questions: {result['sub_questions']}")
print(f"Answer: {result['answer']}")
```

---

## 📁 Project Structure
```
ikms-query-planner/
├── src/app/
│   ├── api.py                      # FastAPI application
│   ├── models/
│   │   └── __init__.py            # Pydantic models
│   └── core/
│       ├── agents/
│       │   ├── state.py           # QAState with plan & sub_questions
│       │   ├── prompts.py         # Agent system prompts
│       │   ├── agents.py          # Agent node functions
│       │   ├── graph.py           # LangGraph workflow
│       │   └── tools.py           # Retrieval tool
│       └── retrieval/
│           └── vector_store.py    # Pinecone vector store manager
│
├── frontend/
│   └── app.py                     # Streamlit UI
│
├── scripts/
│   ├── setup_pinecone.py          # Index PDFs into Pinecone
│   └── clear_pinecone.py          # Clear Pinecone index
│
├── documents/                      # Place your PDFs here
├── .env                           # API keys (not in repo)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🎨 Using the Frontend

### 1. Ask Questions



- Type your question in the text area
- Or click an example question from the sidebar
- Click "🚀 Generate Answer"

### 2. View Results in Tabs

**📊 Overview Tab:**
- Metrics: Sub-questions generated, retrieval calls, answer length
- Final answer display

**🧠 Planning Tab:**
- Search strategy/plan
- List of generated sub-questions
- Processing pipeline visualization
- Shows Feature 1's multi-query benefit

**🔍 Context Tab:**
- Retrieved document chunks
- Source information
- Context statistics

**📄 Raw Data Tab:**
- Complete JSON API response

---

## 🔧 Advanced Configuration

### Custom PDF Processing

**Index Multiple PDFs:**
```bash
# Put all PDFs in documents/ folder
# Then index them all
for file in documents/*.pdf; do
    python scripts/setup_pinecone.py "$file"
done
```

**Clear Pinecone Index:**
```bash
# Remove all vectors
python scripts/clear_pinecone.py
```

### Change Models

Edit `src/app/core/agents/agents.py`:
```python
# Use Gemini Pro instead of Flash (higher quality, same free tier)
planner_llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-pro",  
    temperature=0,
    convert_system_message_to_human=True
)
```

### Adjust Chunk Settings

Edit `scripts/setup_pinecone.py`:
```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,      # Increase from 1000
    chunk_overlap=300,    # Increase from 200
    separators=["\n\n", "\n", ". ", " ", ""]
)
```

### Change Ports
```bash
# Backend on different port
uvicorn src.app.api:app --port 8001 --reload

# Frontend on different port
streamlit run frontend/app.py --server.port 8502
```

---

## 🧪 Testing

### Test Backend
```bash
python test_backend.py
```

### Test with curl
```bash
# Health check
curl http://localhost:8000/health

# Ask a question (Windows PowerShell)
Invoke-RestMethod -Uri "http://localhost:8000/qa" -Method POST -ContentType "application/json" -Body '{"question":"What are vector databases?"}'
```

### Test PDF Processing
```bash
python test_pdf.py documents/your_document.pdf
```

---

## 🐛 Troubleshooting

### "Gemini API quota exceeded"
- Check your usage at: https://aistudio.google.com/
- Free tier: 1,500 requests/day
- Wait 24 hours for reset, or upgrade to paid tier

### "Pinecone index not found"
```bash
# Check your index name in .env matches Pinecone console
# Default: PINECONE_INDEX_NAME=ikms-rag
```

### "No vectors in index"
```bash
# Index a PDF first
python scripts/setup_pinecone.py documents/sample.pdf
```

### "PDF parsing errors"
```bash
# Install PyMuPDF for better PDF handling
pip install pymupdf
```

### "Module not found"
```bash
# Ensure virtual environment is activated
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# Reinstall dependencies
pip install -r requirements.txt
```



---



## 🎓 Learning Outcomes

This project demonstrates:

1. **Multi-Agent Coordination**
   - Specialized agents with distinct roles
   - Sequential pipeline with state propagation

2. **Query Planning & Decomposition**
   - Analyzing question complexity
   - Breaking down into sub-queries
   - Strategic retrieval planning

3. **LangGraph State Management**
   - Enhanced state schema
   - State flow through nodes

4. **Vector Database Integration**
   - Semantic search with Pinecone
   - Embedding generation with Gemini

5. **Prompt Engineering**
   - System prompts for specialized agents
   - Output format control

6. **API Design**
   - RESTful endpoints with FastAPI
   - Request/response models

---

## 🚀 Future Enhancements

Potential improvements beyond Feature 1:

- **Feature 2**: Multi-call retrieval with message organization
- **Feature 3**: Context critic & reranker agent
- **Feature 4**: Evidence-aware answers with chunk citations
- **Feature 5**: Conversational multi-turn QA with memory
- Query history tracking
- User feedback collection
- A/B testing different planning strategies

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open Pull Request

---

## 📝 License

MIT License - see LICENSE file for details.

---



## 🙏 Acknowledgments

- [Google Gemini](https://ai.google.dev/) for  LLM and embeddings
- [Pinecone](https://www.pinecone.io/) for vector database
- [LangChain](https://github.com/langchain-ai/langchain) for agent framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [FastAPI](https://fastapi.tiangolo.com/) for API framework
- [Streamlit](https://streamlit.io/) for frontend

---

## 📞 Support

Having issues?

1. Check [Troubleshooting](#-troubleshooting) section
2. Open new issue with:
   - Error message
   - Steps to reproduce
   - Environment details (OS, Python version)

---



