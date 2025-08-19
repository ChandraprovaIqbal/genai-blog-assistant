# main.py
from fastapi import FastAPI, Header, HTTPException, status, Depends
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, TypedDict
from collections import Counter
from langgraph.graph import StateGraph, END
import os, re, math, json, time

app = FastAPI(title="GenAI Blog Assistant", version="1.0")

# -------------------------------------------------------------------
# Security
# -------------------------------------------------------------------
from fastapi.security import APIKeyHeader
from fastapi import Depends

API_KEY = os.getenv("API_KEY", "devkey")

# This defines an API Key *scheme* named "X-API-Key" for Swagger UI
api_key_scheme = APIKeyHeader(name="X-API-Key", auto_error=False)

def require_api_key(x_api_key: Optional[str] = Depends(api_key_scheme)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        )



""" API_KEY = os.getenv("API_KEY", "devkey")

def require_api_key(x_api_key: Optional[str] = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key",
        ) """

# -------------------------------------------------------------------
# Optional LLM (OpenAI). The app runs without it.
# -------------------------------------------------------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
openai_client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        openai_client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        openai_client = None  # falls back to heuristics

# -------------------------------------------------------------------
# Lightweight text utilities used by both endpoints
# -------------------------------------------------------------------
STOP = set(("a an the and or but of to in on with for by from this that is are was were be been being as at "
            "it its into than then so such not no nor over under again further out off only own same too very "
            "can will just should now").split())

TOPIC_LEXICON = [
    "law","legal","contract","privacy","compliance","startup","ai","governance",
    "policy","gdpr","licensing","ip","regulation","risk","intellectual property"
]

POS = {"good","great","excellent","love","helpful","clear","positive","benefit","success"}
NEG = {"bad","poor","confusing","hate","terrible","negative","risk","issue","failure"}

def words(text: str) -> List[str]:
    return re.findall(r"[A-Za-z][A-Za-z\-']+", text.lower())

def sentences(text: str) -> List[str]:
    return [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]

def estimate_syllables(word: str) -> int:
    w = re.sub(r"[^a-z]", "", word.lower())
    if not w: return 1
    groups = re.findall(r"[aeiouy]+", w)
    count = len(groups)
    if w.endswith("e") and count > 1:
        count -= 1
    return max(1, count)

def flesch_reading_ease(text: str) -> float:
    W = words(text)
    S = max(1, len(sentences(text)))
    if not W: return 100.0
    syl = sum(estimate_syllables(w) for w in W)
    score = 206.835 - 1.015*(len(W)/S) - 84.6*(syl/len(W))
    return max(0.0, min(100.0, score))

def extract_topics(text: str) -> List[str]:
    low = text.lower()
    t = [kw for kw in TOPIC_LEXICON if kw in low]
    return t[:5] if t else ["general"]

def extract_keywords(text: str, k: int = 8) -> List[str]:
    toks = [t for t in words(text) if t not in STOP]
    common = Counter(toks).most_common(20)
    return [w for w,_ in common][:k]

def sentiment_scores(text: str) -> Dict[str,float]:
    w = set(words(text))
    pos = len(w & POS); neg = len(w & NEG)
    total = pos + neg
    if total == 0:
        return {"pos":0.3, "neg":0.2, "neu":0.5}
    pos_p = pos/total; neg_p = neg/total
    neu_p = max(0.0, 1.0 - (pos_p + neg_p))
    s = pos_p + neg_p + neu_p
    return {"pos":pos_p/s, "neg":neg_p/s, "neu":neu_p/s}

def cosine_sim(a: Counter, b: Counter) -> float:
    if not a or not b: return 0.0
    inter = set(a) & set(b)
    num = sum(a[x]*b[x] for x in inter)
    den = math.sqrt(sum(v*v for v in a.values())) * math.sqrt(sum(v*v for v in b.values()))
    return (num/den) if den else 0.0

def relevance_score(draft: str, seed_keywords: List[str]) -> float:
    da = Counter([t for t in words(draft) if t not in STOP])
    kb = Counter([t for t in seed_keywords if t not in STOP])
    return max(0.0, min(100.0, cosine_sim(da,kb)*100.0))

def token_estimate(text: str) -> int:
    # coarse estimate similar to OpenAI tokenization (~4 chars per token)
    return max(50, len(text)//4 + 20)

def combined_score(draft: str, seed_keywords: List[str], reading_level: str = "intermediate") -> float:
    read = flesch_reading_ease(draft)
    rel = relevance_score(draft, seed_keywords)
    target = {"beginner":70, "intermediate":60, "advanced":50}.get(reading_level.lower(), 60)
    read_match = max(0.0, 100.0 - abs(read - target))
    # weights chosen to balance content fit and readability
    score = 0.45*rel + 0.35*read_match + 0.20*read
    return max(0.0, min(100.0, score))

# -------------------------------------------------------------------
# Request / response models
# -------------------------------------------------------------------
class BlogInput(BaseModel):
    text: str

class AnalyzeBlogsRequest(BaseModel):
    blogs: List[BlogInput]

class BlogAnalysis(BaseModel):
    sentiment: Dict[str, float]   # {"pos":..,"neg":..,"neu":..}
    topics: List[str]
    keywords: List[str]

class AnalyzeBlogsResponse(BaseModel):
    results: List[BlogAnalysis]

class RecommendRequest(BaseModel):
    draft: str
    cursor: Optional[str] = None
    user_profile: Optional[Dict[str, Any]] = None

class Suggestion(BaseModel):
    phrase: str
    rank: float

class RecommendResponse(BaseModel):
    suggestions: List[Suggestion]
    readability_score: float
    relevance_score: float
    token_estimate: int
    combined_score: float
    usage: Optional[Dict[str, int]] = None  # present when LLM is used

# -------------------------------------------------------------------
# Endpoints
# -------------------------------------------------------------------
@app.get("/health")
def health():
    """Basic readiness probe and LLM availability flag."""
    return {"status": "ok", "llm_enabled": bool(openai_client)}

@app.post("/api/analyze-blogs", response_model=AnalyzeBlogsResponse, dependencies=[Depends(require_api_key)])
def analyze_blogs(payload: AnalyzeBlogsRequest):
    """
    Accepts an array of blog texts and returns, for each one:
    - sentiment distribution, detected topics, and top keywords.
    """
    results: List[BlogAnalysis] = []
    for b in payload.blogs:
        s = sentiment_scores(b.text)
        t = extract_topics(b.text)
        k = extract_keywords(b.text, 8)
        results.append(BlogAnalysis(sentiment=s, topics=t, keywords=k))
    return AnalyzeBlogsResponse(results=results)

@app.post("/api/recommend-keywords", response_model=RecommendResponse, dependencies=[Depends(require_api_key)])
def recommend_keywords(payload: RecommendRequest):
    """
    Accepts the current draft and (optional) user profile, and returns:
    - ranked phrase suggestions,
    - readability score, relevance score,
    - estimated tokens and a combined 0–100 score.
    Uses an LLM when configured; otherwise falls back to heuristics.
    """
    draft = payload.draft
    up = payload.user_profile or {}
    level = (up.get("reading_level") or "intermediate").lower()

    # default suggestions; switched if LLM refines them
    suggestions = ["definitions", "case study", "checklist", "best practices", "common pitfalls"]
    if "ai" in draft.lower():
        suggestions = ["model governance", "bias mitigation", "data privacy", "risk register", "explainability"]
    usage = None

    if openai_client:
        prompt = (
            "Suggest 5 short phrases (2–4 words) that would be natural next insertions "
            "for the following blog draft. Focus on legal/AI if relevant. "
            "Return ONLY a JSON array of strings.\n\nDRAFT:\n" + draft
        )
        try:
            resp = openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
            )
            raw = (resp.choices[0].message.content or "").strip()
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list) and parsed:
                    suggestions = [str(x) for x in parsed[:5]]
            except Exception:
                # tolerate non-JSON outputs
                lines = [ln.strip("-* \n") for ln in raw.splitlines() if ln.strip()]
                if lines:
                    suggestions = lines[:5]
            if resp.usage:
                usage = {
                    "prompt_tokens": resp.usage.prompt_tokens or 0,
                    "completion_tokens": resp.usage.completion_tokens or 0,
                    "total_tokens": resp.usage.total_tokens or 0,
                }
        except Exception:
            # network/model errors fall back to heuristics
            usage = None

    sugg_objs = [Suggestion(phrase=p, rank=1.0 - i*0.12) for i, p in enumerate(suggestions)]
    read = flesch_reading_ease(draft)
    seed = extract_keywords(draft, 10)
    rel = relevance_score(draft, seed)
    tokens = token_estimate(draft)
    combo = combined_score(draft, seed, level)

    return RecommendResponse(
        suggestions=sugg_objs,
        readability_score=read,
        relevance_score=rel,
        token_estimate=tokens,
        combined_score=combo,
        usage=usage
    )

# -------------------------------------------------------------------
# Agentic workflow demo (LangGraph + retries/backoff)
# -------------------------------------------------------------------
class AgentDemoRequest(BaseModel):
    draft: str = Field(..., description="Starting draft text")
    seconds: int = Field(6, description="Total runtime of the loop")
    interval: float = Field(2.0, description="Seconds between iterations")
    mutate: bool = Field(True, description="Append words each tick to simulate typing")

class AgentDemoResponse(BaseModel):
    ticks: int
    last_output: RecommendResponse

class AgentState(TypedDict, total=False):
    draft: str
    output: Dict[str, Any]

def _recommend_node(state: AgentState) -> AgentState:
    """Single LangGraph node that runs the recommend routine on current draft."""
    req = RecommendRequest(draft=state["draft"], user_profile={"reading_level": "intermediate"})
    out = recommend_keywords(req)  # reuses app logic; no HTTP round-trip
    return {"output": out.model_dump()}

# build a minimal graph once; we invoke it in the loop below
_graph = StateGraph(AgentState)
_graph.add_node("recommend", _recommend_node)
_graph.set_entry_point("recommend")
_graph.set_finish_point("recommend")
COMPILED_GRAPH = _graph.compile()

@app.post("/api/agent-demo", response_model=AgentDemoResponse, dependencies=[Depends(require_api_key)])
def agent_demo(payload: AgentDemoRequest):
    """
    Demonstrates an agent that periodically re-runs recommendations while the draft evolves.
    Includes retries with exponential backoff. Intended for assignment validation.
    """
    draft = payload.draft
    deadline = time.time() + max(1, payload.seconds)
    ticks = 0
    last: Optional[RecommendResponse] = None

    while time.time() < deadline:
        # retry w/ exponential backoff
        backoff = 0.3
        for _ in range(3):
            try:
                state_in: AgentState = {"draft": draft}
                result = COMPILED_GRAPH.invoke(state_in)
                # convert dict back to pydantic model
                last = RecommendResponse(**result["output"])
                break
            except Exception:
                time.sleep(backoff)
                backoff *= 2
        ticks += 1

        if payload.mutate:
            draft += " example"  # simulate user typing

        time.sleep(max(0.1, payload.interval))

    assert last is not None
    return AgentDemoResponse(ticks=ticks, last_output=last)
