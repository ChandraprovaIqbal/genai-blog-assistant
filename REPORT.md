# REPORT

## Architecture
The system is a FastAPI web service with clean modular layers:  
- API layer: exposes `/api/analyze-blogs`, `/api/recommend-keywords`, and `/api/agent-demo`.  
- Utility layer: handles text tokenization, sentiment detection, keyword extraction, readability, and cosine similarity.  
- Model layer: optional OpenAI GPT-4o-mini integration via Python client.  
- Workflow layer: LangGraph agent loop to re-run recommendations with exponential backoff.  

## Model and Prompt Rationale
- Heuristics are used when no LLM is available, ensuring the system always works.  
- With OpenAI, GPT-4o-mini is prompted to return exactly five short keyword phrases in JSON array form.  
- Phrases are constrained to 2–4 words, keeping suggestions concise and parseable.  
- The temperature is set to 0.7 for variation while remaining relevant.  
- Prompts include context such as “legal/AI” to stay aligned with compliance themes.  

## Scoring Formula
- **Readability**: Flesch Reading Ease = 206.835 − 1.015 × (words per sentence) − 84.6 × (syllables per word).  
- **Readability match**: Difference from target levels (70 beginner, 60 intermediate, 50 advanced) subtracted from 100.  
- **Relevance**: Cosine similarity between draft keywords and extracted seed keywords.  
- **Combined score**: 0.45 × relevance + 0.35 × readability match + 0.20 × raw readability.  

## Token Efficiency
- Tokens are estimated as one per four characters with a minimum of 50 tokens.  
- Prompts explicitly request JSON arrays to reduce wasted completion tokens.  
- When no OpenAI key is present, the system falls back to heuristics, consuming zero tokens.  
- Responses include prompt, completion, and total tokens for transparency.  

## Conclusion
This design balances cost control and interpretability while meeting assignment deliverables. It demonstrates a hybrid workflow combining deterministic analysis with LLM-based keyword generation, while ensuring efficiency and reproducibility.
