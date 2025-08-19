I built the system using FastAPI. It gives me a simple way to create APIs. There are three main endpoints:

/api/analyze-blogs: takes blog text and gives back topics, keywords, and sentiment.

/api/recommend-keywords: takes a draft and suggests phrases, readability, relevance, token count, and a score.

/api/agent-demo: runs a small loop using LangGraph to show how an agent can update results as text grows.

The code uses small helper functions for text tasks. These include splitting words and sentences, checking reading ease, finding topics and keywords, and simple sentiment checks.
I added an API key check for security. The system can also call OpenAI if a key is set. If not, it still works with my own rule-based functions.

Model and Prompt Rationale

The main part does not need an LLM. I wrote simple rules for keywords, sentiment, and relevance. But I also let it call an OpenAI model if available.
When the LLM is used, I give it a short prompt: “Suggest 5 short phrases…” This is enough to get clean, list-style answers. If the model fails or gives extra text, I fall back on to my own phrase list.

I kept the LLM optional so the app can run anywhere, even with no API key. This makes testing very easy and shows that I understand both rule-based and model-based methods.

Scoring Formula

I used 3 main checks for the score:

Relevance score: cosine similarity between draft words and seed keywords.

Readability score: Flesch Reading Ease formula.

Target reading level match: beginner (70), intermediate (60), or advanced (50).

I combine them as:

score = 0.45 * relevance + 0.35 * match_to_level + 0.20 * readability

This mix keeps the balance. Relevance is the most important, then the reading level and  then raw readability. The score always stays between 0 and 100.

Token Efficiency

I have used a simple rule to estimate tokens: about 1 token per 4 characters plus 20. This helps me know cost when using OpenAI.
I also keep the LLM calls short. The prompt is one block of text, and the model only needs to return a JSON array. This saves tokens and keeps answers fast.