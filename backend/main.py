from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from concurrent.futures import ThreadPoolExecutor
import re
import sys
from pathlib import Path

# When run locally via `cd backend && uvicorn main:app`, Python puts this
# file's own directory on sys.path automatically, so the bare `from rag
# import ...` style imports below just work. But Vercel imports this
# module as `backend.main:app` from the REPO ROOT (see pyproject.toml's
# [tool.vercel] entrypoint) — in that case `backend/` is NOT on sys.path,
# so the same imports would fail with ModuleNotFoundError. Adding it
# explicitly here makes both environments work without touching the
# import style used everywhere else in this file.
sys.path.insert(0, str(Path(__file__).resolve().parent))

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

# Local KB retrieval and live web search are independent lookups, so we run
# them concurrently instead of one-after-another wherever both are needed —
# same latency idea as the two Gonka model calls in gonka_client.py.
_io_executor = ThreadPoolExecutor(max_workers=4)


# --- Language auto-detection ------------------------------------------------
# Lightweight, dependency-free detection so replies match whatever language
# the person actually typed in, instead of relying on a manually-picked
# dropdown value that's easy to forget to change.

_MS_HINT_WORDS = {
    "yang", "dan", "saya", "boleh", "bolehkah", "apa", "apakah", "berapa",
    "berapakah", "bagaimana", "adakah", "tidak", "ini", "itu", "untuk",
    "dengan", "kalau", "nak", "mahu", "kena", "perlu", "macam", "mana",
    "kenapa", "mengapa", "bila", "bilakah", "lesen", "memandu", "kad",
    "pengenalan", "pasport", "warganegara", "kerajaan",
}


def detect_language(text: str) -> str:
    """Returns "en", "ms", or "zh". CJK characters are an unambiguous
    signal for Chinese. Otherwise, count how many whole words match a
    small Bahasa Malaysia function-word list vs. treat as English by
    default — good enough for short chat messages without needing an
    extra ML dependency or an extra Gonka call just to detect language."""
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"

    words = re.findall(r"[a-zA-Z']+", text.lower())
    if not words:
        return "en"
    ms_hits = sum(1 for w in words if w in _MS_HINT_WORDS)
    if ms_hits >= 1 and ms_hits / len(words) >= 0.15:
        return "ms"
    return "en"


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
    # "auto" (default) detects the message's language and replies in kind.
    # Pass "en"/"ms"/"zh" explicitly to force a reply language regardless
    # of what the message is written in.
    language: str = Field(default="auto", pattern="^(auto|en|ms|zh)$")


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

# Small talk / chit-chat that isn't a question OR a factual claim — sending
# these through the full RAG + web-search + two-model consensus pipeline
# wastes calls and produces a confusing "UNVERIFIABLE" verdict for what's
# really just a greeting. Short-circuit these before classify_message()
# even runs, with zero Gonka calls.
_SMALLTALK = {
    "hi", "hello", "hey", "hai", "helo", "yo", "sup",
    "assalamualaikum", "salam", "apa khabar", "khabar",
    "你好", "您好", "嗨", "哈喽",
    "thanks", "thank you", "thankyou", "terima kasih", "tq",
    "谢谢", "感谢",
    "ok", "okay", "test", "testing", "bye", "goodbye", "see you",
}


def is_smalltalk(text: str) -> bool:
    stripped = text.strip().lower().rstrip("!.?？！ ")
    return stripped in _SMALLTALK


_SMALLTALK_REPLY = {
    "en": "Hi! I'm MyAssist 👋 — ask me a question about MyKad, passport, or driving licence services, or paste a claim/URL and I'll fact-check it for you.",
    "ms": "Hai! Saya MyAssist 👋 — tanya saya soalan tentang perkhidmatan MyKad, pasport, atau lesen memandu, atau tampal sebarang dakwaan/URL untuk saya sahkan.",
    "zh": "你好！我是 MyAssist 👋 — 你可以问我关于 MyKad、护照或驾驶执照服务的问题，或者贴上一个说法或链接让我帮你核实。",
}


# --- Clarification: catch bare topic words before they hit either pipeline --
# A message like "identity card" / "kad pengenalan" / "身份证" is neither a
# question nor a checkable claim — it's the user naming a topic and
# expecting us to guess the rest. Instead of silently guessing (and, worse,
# running it through the CLAIM pipeline where it comes back a confusing
# "UNVERIFIABLE"), ask one quick follow-up — same as a human counter clerk
# would say "sure — what about it?" This is a deterministic check, zero
# extra Gonka calls, same reasoning as classify_message() below.

_CLARIFY_TOPIC_KEYWORDS = {
    "mykad": ["identity card", "ic", "mykad", "my kad", "kad pengenalan", "身份证"],
    "passport": ["passport", "pasport", "护照"],
    "license": [
        "driving license", "driving licence", "driver's license",
        "driver's licence", "lesen memandu", "lesen", "驾驶执照", "驾照",
    ],
}

_CLARIFY_QUESTIONS = {
    "mykad": {
        "en": "Sure — what would you like to know about MyKad? For example: how to apply, renewal steps/fee, or replacing a lost card.",
        "ms": "Baik — apa yang anda ingin tahu tentang MyKad? Contohnya: cara memohon, langkah/kos pembaharuan, atau menggantikan kad yang hilang.",
        "zh": "好的 — 您想了解身份证的哪方面？例如：如何申请、续期步骤/费用，或补办遗失的证件。",
    },
    "passport": {
        "en": "Sure — what would you like to know about the passport? For example: how to apply, renewal steps/fee, or processing time.",
        "ms": "Baik — apa yang anda ingin tahu tentang pasport? Contohnya: cara memohon, langkah/kos pembaharuan, atau tempoh pemprosesan.",
        "zh": "好的 — 您想了解护照的哪方面？例如：如何申请、续期步骤/费用，或办理所需时间。",
    },
    "license": {
        "en": "Sure — what would you like to know about the driving licence? For example: how to renew, the fee, or what to do if it's lost.",
        "ms": "Baik — apa yang anda ingin tahu tentang lesen memandu? Contohnya: cara memperbaharui, kosnya, atau apa perlu buat jika hilang.",
        "zh": "好的 — 您想了解驾驶执照的哪方面？例如：如何续期、费用，或遗失后该怎么办。",
    },
    "general": {
        "en": "Could you tell me a bit more about what you'd like to know? For example, is this about MyKad, a passport, or a driving licence — and are you asking about applying, renewing, fees, or something else?",
        "ms": "Boleh beritahu saya sedikit lagi tentang apa yang anda ingin tahu? Contohnya, adakah ini tentang MyKad, pasport, atau lesen memandu — dan adakah ia tentang permohonan, pembaharuan, kos, atau lain-lain?",
        "zh": "可以再告诉我多一点您想了解的内容吗？例如，您问的是身份证、护照，还是驾驶执照 — 是关于申请、续期、费用，还是其他方面？",
    },
}


def detect_topic(text: str) -> str:
    lowered = text.strip().lower()
    for topic, keywords in _CLARIFY_TOPIC_KEYWORDS.items():
        if any(kw in lowered for kw in keywords):
            return topic
    return "general"


def needs_clarification(text: str) -> bool:
    """True for a short, bare phrase that isn't a question and isn't a
    URL — e.g. "identity card", "mykad", "护照". Real questions have a
    "?" or a question-word (caught first); real claims to fact-check are
    almost always full sentences, so a handful of words (or, for
    space-less scripts like Chinese, a handful of characters) is a
    reasonable "too vague, just ask" signal for a hackathon MVP."""
    stripped = text.strip()
    if is_url(stripped) or stripped.endswith(("?", "？")):
        return False
    lowered = stripped.lower()
    first_word = lowered.split()[0] if lowered.split() else ""
    for starter in _QUESTION_STARTERS:
        if lowered.startswith(starter) or first_word == starter:
            return False
    if re.search(r"[\u4e00-\u9fff]", stripped):
        return len(stripped) <= 6
    return len(stripped.split()) <= 3


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

    # Local KB retrieval and live web search don't depend on each other —
    # run them at the same time instead of one after another.
    kb_future = _io_executor.submit(knowledge_base.retrieve, claim_text, top_k=3)
    web_future = _io_executor.submit(fetch_web_content, claim_text[:200], is_search=True)

    context_entries = kb_future.result()
    max_local_similarity = max(
        (e.get("similarity", 0) for e in context_entries), default=0.0
    )

    web_used = False
    web_context = web_future.result()
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
    language = detect_language(message) if req.language == "auto" else req.language

    if is_smalltalk(message):
        return {
            "answer": _SMALLTALK_REPLY.get(language, _SMALLTALK_REPLY["en"]),
            "mode": "chitchat",
            "truth_score": None,
            "confidence": None,
            "reasoning_trace": [],
            "source": None,
            "models_agree": None,
            "agreement_score": None,
            "model_votes": [],
            "gonka_request_id": None,
            "model": None,
            "retrieved": [],
            "web_used": False,
            "detected_language": language,
        }

    if needs_clarification(message):
        topic = detect_topic(message)
        return {
            "answer": _CLARIFY_QUESTIONS[topic].get(language, _CLARIFY_QUESTIONS[topic]["en"]),
            "mode": "clarify",
            "truth_score": None,
            "confidence": None,
            "reasoning_trace": [],
            "source": None,
            "models_agree": None,
            "agreement_score": None,
            "model_votes": [],
            "gonka_request_id": None,
            "model": None,
            "retrieved": [],
            "web_used": False,
            "detected_language": language,
        }

    mode = classify_message(message)

    if mode == "qa":
        result = _run_qa(message, language)
    else:
        result = _run_claim_verification(message, language)

    result["mode"] = mode
    result["detected_language"] = language
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