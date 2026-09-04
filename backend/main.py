from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import knowledge_base
from gonka_client import ask_gonka, verify_claim

from web_scraper import is_url, fetch_web_content

app = FastAPI(title="MUBA Hacks - Gov Services Assistant")

# Allow the frontend (served from a different port/origin during dev) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class AskRequest(BaseModel):
    question: str = Field(min_length=2, max_length=2000)
    language: str = Field(default="en", pattern="^(en|ms|zh)$")


class VerifyRequest(BaseModel):
    # A claim to fact-check: either raw text (e.g. a tweet's wording pasted
    # in) or a URL (e.g. a tweet/article link) to scrape and check.
    input: str = Field(min_length=2, max_length=4000)
    language: str = Field(default="en", pattern="^(en|ms|zh)$")


class ChatRequest(BaseModel):
    # One box for everything — the chat itself decides whether this is a
    # question to answer or a claim to fact-check. See classify_message().
    message: str = Field(min_length=2, max_length=4000)
    language: str = Field(default="en", pattern="^(en|ms|zh)$")


# --- Router: decide Q&A vs claim-verification WITHOUT calling an LLM -------
# This is deliberately a plain, deterministic, inspectable function (not
# another Gonka call) — so the routing decision itself stays transparent
# and auditable, and so classifying a message doesn't cost an extra
# inference call before the "real" one even runs.

_QUESTION_STARTERS = (
    # English
    "what", "whats", "what's", "how", "when", "where", "why", "who", "which",
    "can", "could", "is", "are", "do", "does", "did", "will", "should",
    # Bahasa Malaysia
    "apa", "apakah", "bagaimana", "bagaimanakah", "bila", "bilakah",
    "di mana", "dimana", "kenapa", "mengapa", "siapa", "siapakah",
    "berapa", "berapakah", "adakah", "bolehkah",
    # Chinese
    "什么", "怎么", "怎样", "如何", "几时", "什么时候", "哪里", "为什么", "谁", "多少", "可以吗", "是不是",
)


def classify_message(text: str) -> str:
    """Returns "qa" or "claim".

    - A URL is always treated as a claim to verify (someone pasting a link
      wants it checked, not answered).
    - Anything ending in a question mark, or opening with a
      question-word in English / Bahasa Malaysia / Chinese, is routed to
      Q&A.
    - Everything else (a declarative statement) is treated as a claim to
      fact-check — that's the natural shape of "X costs RM50", "renewal
      takes 2 weeks", etc.
    """
    stripped = text.strip()
    if is_url(stripped):
        return "claim"
    if stripped.endswith("?") or stripped.endswith("？"):
        return "qa"

    lowered = stripped.lower()
    first_word = lowered.split()[0] if lowered.split() else ""
    for starter in _QUESTION_STARTERS:
        if lowered.startswith(starter) or first_word == starter:
            return "qa"

    return "claim"


# --- Shared pipelines -------------------------------------------------------

def _run_qa(question: str, language: str) -> dict:
    """Government-services Q&A pipeline: local KB retrieval, falling back
    to live web search only if local context looks irrelevant, then a
    multi-model consensus call via ask_gonka()."""
    web_context = ""
    web_used = False

    if is_url(question):
        print(f"🔗 [qa] URL detected, scraping: {question}")
        web_context = fetch_web_content(question, is_search=False)
        if "Error" not in web_context and "not configured" not in web_context:
            web_used = True

    context_entries = knowledge_base.retrieve(question, top_k=3)

    max_similarity = 0.0
    for entry in context_entries:
        if entry.get("similarity", 0) > max_similarity:
            max_similarity = entry.get("similarity", 0)
    print(f"📊 [qa] Highest local similarity: {max_similarity}")

    if max_similarity < 0.5 and not is_url(question):
        print("🌐 [qa] Local context insufficient. Falling back to Web Search...")
        web_context = fetch_web_content(question, is_search=True)
        if "Error" not in web_context and "not configured" not in web_context:
            web_used = True
            context_entries = [{
                "source": "Web Search (Firecrawl)",
                "source_url": "",
                "text_ms": web_context,
                "similarity": 0.9,
            }]
        else:
            print("⚠️ [qa] Web search failed, keeping local context.")

    if web_context and "Error" not in web_context and "not configured" not in web_context and not context_entries:
        context_entries.append({
            "source": "Web Scrape (Firecrawl)",
            "source_url": question if is_url(question) else "",
            "text_ms": web_context,
            "similarity": 0.9,
        })

    if not context_entries:
        return {
            "answer": "I could not find any relevant information, either locally or on the web. Please try rephrasing your question.",
            "confidence": 0,
            "truth_score": 0,
            "reasoning_trace": [],
            "source": None,
            "models_agree": None,
            "agreement_score": None,
            "model_votes": [],
            "gonka_request_id": None,
            "model": None,
            "retrieved": [],
            "web_used": web_used,
        }

    result = ask_gonka(question, context_entries, target_lang=language)

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "truth_score": result["truth_score"],
        "reasoning_trace": result["reasoning_trace"],
        "source": result["source"],
        "models_agree": result["models_agree"],
        "agreement_score": result["agreement_score"],
        "model_votes": result["model_votes"],
        "gonka_request_id": result["gonka_request_id"],
        "model": result["model"],
        "retrieved": context_entries,
        "web_used": web_used,
    }


def _run_claim_verification(raw_input: str, language: str) -> dict:
    """Claim/fact-check pipeline: extract the claim (scrape if it's a URL),
    gather evidence from BOTH the local KB and live web search, then a
    multi-model consensus truth score via verify_claim()."""
    raw_input = raw_input.strip()
    input_is_url = is_url(raw_input)
    claim_text = raw_input

    if input_is_url:
        print(f"🔗 [verify] URL detected, scraping claim source: {raw_input}")
        scraped = fetch_web_content(raw_input, is_search=False)
        if "Error" in scraped or "not configured" in scraped:
            return {
                "answer": f"Could not fetch content from that URL: {scraped}",
                "truth_score": 0,
                "reasoning_trace": [],
                "source": None,
                "models_agree": None,
                "agreement_score": None,
                "model_votes": [],
                "gonka_request_id": None,
                "model": None,
                "retrieved": [],
                "web_used": False,
                "input_type": "url",
            }
        claim_text = scraped

    context_entries = knowledge_base.retrieve(claim_text, top_k=3)
    max_local_similarity = max(
        (e.get("similarity", 0) for e in context_entries), default=0.0
    )

    web_used = False
    web_context = fetch_web_content(claim_text[:200], is_search=True)
    if (
        "Error" not in web_context
        and "not configured" not in web_context
        and "No search results" not in web_context
    ):
        web_used = True
        context_entries.append({
            "source": "Web Search (Firecrawl)",
            "source_url": "",
            "text_ms": web_context,
            "similarity": 0.9,
        })

    if not context_entries:
        return {
            "answer": "No supporting evidence found locally or on the web to verify this claim.",
            "truth_score": 0,
            "reasoning_trace": [],
            "source": None,
            "models_agree": None,
            "agreement_score": None,
            "model_votes": [],
            "gonka_request_id": None,
            "model": None,
            "retrieved": [],
            "web_used": web_used,
            "input_type": "url" if input_is_url else "text",
        }

    result = verify_claim(claim_text[:3000], context_entries, target_lang=language)

    return {
        "answer": result["answer"],
        "truth_score": result["truth_score"],
        "reasoning_trace": result["reasoning_trace"],
        "source": result["source"],
        "models_agree": result["models_agree"],
        "agreement_score": result["agreement_score"],
        "model_votes": result["model_votes"],
        "gonka_request_id": result["gonka_request_id"],
        "model": result["model"],
        "retrieved": context_entries,
        "web_used": web_used,
        "input_type": "url" if input_is_url else "text",
        "local_similarity": round(max_local_similarity, 3),
    }


# --- Endpoints ---------------------------------------------------------------

@app.post("/chat")
def chat(req: ChatRequest):
    """Single conversational entry point. Classifies the message, runs the
    matching pipeline, and returns one unified response shape so the
    frontend can render every reply the same way regardless of mode."""
    message = req.message.strip()
    mode = classify_message(message)

    if mode == "qa":
        result = _run_qa(message, req.language)
    else:
        result = _run_claim_verification(message, req.language)

    result["mode"] = mode
    return result


@app.post("/ask")
def ask(req: AskRequest):
    """Kept for backward compatibility / direct testing — same pipeline
    /chat uses when it classifies a message as Q&A."""
    return _run_qa(req.question, req.language)


@app.post("/verify")
def verify(req: VerifyRequest):
    """Kept for backward compatibility / direct testing — same pipeline
    /chat uses when it classifies a message as a claim to fact-check."""
    return _run_claim_verification(req.input, req.language)


@app.get("/health")
def health():
    return {"status": "ok", "kb_entries": len(knowledge_base.entries)}