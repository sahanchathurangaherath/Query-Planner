# IKMS Query Planner

An intelligent Knowledge Management System (IKMS) that uses AI agents to plan, retrieve, and answer questions from your document corpus using advanced query planning and vector search.

## 🌟 Features

- **Multi-Agent Architecture**: Coordinated agents for planning, retrieval, summarization, and verification
- **Query Planning**: Automatically breaks down complex questions into sub-queries
- **Vector Search**: Efficient semantic search using Pinecone vector database
- **PDF Document Processing**: Upload and index PDF documents for Q&A
- **RESTful API**: FastAPI backend for easy integration
- **Interactive Frontend**: Streamlit-based user interface

## 🏗️ Architecture

The system consists of four specialized agents:

1. **Planning Agent**: Analyzes questions and creates search strategies
2. **Retrieval Agent**: Searches documents using enhanced queries
3. **Summarization Agent**: Generates comprehensive answers from retrieved context
4. **Verification Agent**: Ensures answer quality and relevance

## 📋 Prerequisites

- Python 3.8+
- OpenAI API key
- Pinecone API key (for vector database)
- Git

## 🚀 Installation

### 1. Clone the Repository

```bash
git clone https://github.com/sahanchathurangaherath/ikms-query-planner.git
cd ikms-query-planner
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac/WSL:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Environment Variables

Create a `.env` file in the project root:

```bash
touch .env
```

Add your API keys to `.env`:

```env
OPENAI_API_KEY=sk-your-openai-api-key-here
PINECONE_API_KEY=your-pinecone-api-key-here
```

**Important**: Never commit your `.env` file to Git. It's already included in `.gitignore`.

## 🎯 Usage

### Running the Backend API

```bash
# Make sure virtual environment is activated
source venv/bin/activate

# Run FastAPI server
uvicorn src.app.api:app --reload
```

The API will be available at `http://localhost:8000`

**API Documentation**: Visit `http://localhost:8000/docs` for interactive API documentation (Swagger UI)

### Running the Frontend

In a separate terminal:

```bash
# Activate virtual environment
source venv/bin/activate

# Run Streamlit app
streamlit run frontend/app.py
```

The frontend will open automatically in your browser at `http://localhost:8501`

## 📡 API Endpoints

### Health Check
```bash
GET /
```

### Question Answering
```bash
POST /qa
Content-Type: application/json

{
  "question": "What are the main advantages of vector databases?"
}
```

### Index PDF Document
```bash
POST /index-pdf
Content-Type: multipart/form-data

file: <your-pdf-file>
```

### Example with curl

```bash
# Ask a question
curl -X POST "http://localhost:8000/qa" \
  -H "Content-Type: application/json" \
  -d '{"question": "What is machine learning?"}'

# Upload a PDF
curl -X POST "http://localhost:8000/index-pdf" \
  -F "file=@your-document.pdf"
```

## 📁 Project Structure

```
ikms-query-planner/
├── src/
│   └── app/
│       ├── api.py                 # FastAPI application
│       ├── models.py              # Pydantic models
│       └── core/
│           └── agents/
│               ├── agents.py      # Agent definitions
│               ├── graph.py       # LangGraph workflow
│               └── state.py       # State management
├── frontend/
│   └── app.py                     # Streamlit UI
├── test_docs/                     # Sample documents
├── .env                           # Environment variables (not in repo)
├── .gitignore                     # Git ignore rules
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## 🔧 Configuration

### Model Selection

By default, the system uses `gpt-4o-mini`. To change the model, edit `src/app/core/agents/agents.py`:

```python
planner_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)  # Use cheaper model
```

### Port Configuration

To run on a different port:

```bash
# Backend
uvicorn src.app.api:app --port 8001 --reload

# Frontend
streamlit run frontend/app.py --server.port 8502
```

## 🛠️ Development

### Adding New Dependencies

```bash
# Install new package
pip install package-name

# Update requirements
pip freeze > requirements.txt
```

### Running Tests

```bash
pytest tests/
```

## 📊 Cost Considerations

- **OpenAI API**: Costs vary by model and usage
  - `gpt-4o-mini`: ~$0.15 per 1M tokens
  - `gpt-3.5-turbo`: ~$0.50 per 1M tokens
- **Pinecone**: Free tier available for development
- Set usage limits in your OpenAI dashboard to control costs

## 🐛 Troubleshooting

### "OpenAI API quota exceeded"
- Add credits to your OpenAI account at https://platform.openai.com/account/billing
- Set spending limits to control costs

### "streamlit: command not found"
```bash
pip install streamlit
```

### "Module not found" errors
```bash
# Ensure virtual environment is activated
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Port already in use
```bash
# Find and kill process using port 8000
lsof -ti:8000 | xargs kill -9

# Or use a different port
uvicorn src.app.api:app --port 8001 --reload
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👤 Author

**Sahan Chathuranga Herath**
- GitHub: [@sahanchathurangaherath](https://github.com/sahanchathurangaherath)

## 🙏 Acknowledgments

- [LangChain](https://github.com/langchain-ai/langchain) for the agent framework
- [LangGraph](https://github.com/langchain-ai/langgraph) for workflow orchestration
- [Pinecone](https://www.pinecone.io/) for vector database
- [FastAPI](https://fastapi.tiangolo.com/) for the API framework
- [Streamlit](https://streamlit.io/) for the frontend

## 📞 Support

If you have any questions or issues, please:
1. Check the [Issues](https://github.com/sahanchathurangaherath/ikms-query-planner/issues) page
2. Open a new issue if your problem isn't already listed
3. Provide detailed information about the error and your environment

---

**Note**: This project is under active development. Features and documentation may change.


