# GenAI Blog Assistant

The GenAI Blog Assistant is a FastAPI-based application designed to help analyze and refine draft blog posts.  

## Features
- Blog analysis: Accepts an array of blog texts and returns sentiment distributions, detected topics, and top keywords.  
- Keyword recommendation: Accepts a draft and an optional user profile and produces ranked phrase suggestions, readability and relevance scores, estimated tokens, and a combined quality score.  
- Agent demo: Demonstrates iterative drafting and keyword updates using LangGraph.  

## Architecture
- **API layer**: FastAPI routes for health, analysis, recommendations, and agent demo.  
- **Utility layer**: Tokenization, sentiment scoring, keyword extraction, syllable estimation, readability (Flesch), cosine similarity, and scoring functions.  
- **LLM layer**: Optional OpenAI integration via `OPENAI_API_KEY`. Falls back to heuristics if not provided.  
- **Security layer**: API key validation for access control.  
- **Workflow layer**: Agent loop with LangGraph simulating iterative drafting.  

## Key Benefits
- Runs fully without external models.  
- When provided with an OpenAI API key, leverages GPT-4o-mini for better keyword generation.  
- Combines deterministic algorithms with LLM outputs for traceable results.  

## Running Locally
1. Create and activate a virtual environment.  
2. Install dependencies with `pip install -r requirements.txt`.  
3. Set the `API_KEY` environment variable (for example: `export API_KEY=devkey`).  
4. Optionally set `OPENAI_API_KEY` if LLM suggestions are desired.  
5. Start with `uvicorn main:app --reload`.  
6. Open Swagger UI at http://127.0.0.1:8000/docs.  
