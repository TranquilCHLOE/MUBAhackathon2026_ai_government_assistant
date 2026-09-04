"""
Wrapper around the Gonka Router API.

Gonka Router is OpenAI-compatible, so we reuse the official `openai` SDK
and just point it at Gonka's base_url. This is the ONLY place in the
codebase that should call an LLM for reasoning/verification, per the
hackathon's mandatory rule.

Implements Multi-Model Consensus: every question/claim is sent to TWO
different Gonka-hosted models (GONKA_MODEL and GONKA_MODEL_2). Their
individual truth_score + reasoning_trace are reconciled in Python into a
single consensus result, and both raw votes are returned in `model_votes`
for transparency (show this in your demo to prove real cross-verification,
not a single model's opinion dressed up as two).

Double-check the base_url, available model names and auth header against
the current Gonka Router docs / workshop slides before the deadline —
routers occasionally rename models or move endpoints. GONKA_MODEL_2 below
is a PLACEHOLDER guess — confirm a real second model name from the Gonka
Router catalog and set GONKA_MODEL_2 in .env before you demo, or both
"votes" will silently be the same underlying model.
"""

import json
import os
from difflib import SequenceMatcher
from pathlib import Path
import re

from dotenv import load_dotenv
# pip install openai
from openai import OpenAI

# .env now lives at the project root, one level above backend/, regardless
# of what directory you run uvicorn from — so point at it explicitly
# instead of relying on load_dotenv()'s cwd-based default.
load_dotenv(dotenv_path=Path(__file__).resolve().parent.parent / ".env")

GONKA_API_KEY = os.environ["GONKA_API_KEY"]
GONKA_BASE_URL = os.environ.get("GONKA_BASE_URL", "https://api.gonkarouter.io/v1")
GONKA_MODEL = os.environ.get("GONKA_MODEL", "MiniMaxAI/MiniMax-M2.7")
# PLACEHOLDER — verify this against Gonka Router's model list and override
# via .env. It just needs to be a *different* model from GONKA_MODEL so the
# consensus step is a genuine cross-check.
GONKA_MODEL_2 = os.environ.get("GONKA_MODEL_2", "Qwen/Qwen2.5-72B-Instruct")

_client = OpenAI(
    api_key=GONKA_API_KEY,
    base_url=GONKA_BASE_URL,
)

_LANGUAGE_NAMES = {
    "ms": "Bahasa Malaysia",
    "en": "English",
    "zh": "Chinese",
}

_QA_SYSTEM_PROMPT = """You are a Malaysian government services assistant.
Answer ONLY using the context provided below. Do not invent facts, fees,
or procedures that are not in the context.

Stay strictly neutral and objective — report what the context says without
editorializing, and do not favor any political party, official, or agency.
Every step in "reasoning_trace" must cite the specific context entry (by
its bracketed source tag, e.g. "[Muba.pdf]") that supports it.

Respond with a single JSON object, no other text, in this exact shape:
{
  "answer": "<answer written entirely in the requested language>",
  "truth_score": <integer 0-100, how well the context supports this answer>,
  "reasoning_trace": ["<short reasoning step 1, citing a source tag>", "<short reasoning step 2, citing a source tag>", "..."],
  "source": "<the source field of the context entry you relied on most>"
}

If the context does not contain enough information to answer, set
"truth_score" to 0 and say so honestly in "answer".
"""

_VERIFY_SYSTEM_PROMPT = """You are a fact-checking assistant for claims
related to Malaysian government services. You will be given a CLAIM and
CONTEXT gathered from a local government-document knowledge base and/or
live web search.

Judge whether the CLAIM is TRUE, FALSE, or UNVERIFIABLE using ONLY the
CONTEXT given below. Do not rely on outside knowledge that isn't backed by
the CONTEXT — if the CONTEXT doesn't address the claim, say so.

Stay strictly neutral and objective. Do not soften or exaggerate the
verdict to be agreeable. Every step in "reasoning_trace" must cite the
specific context entry (by its bracketed source tag, e.g. "[Muba.pdf]" or
"[Web Search (Firecrawl)]") that the step's claim about the evidence rests
on — a verdict is only as strong as the evidence backing each step.

Respond with a single JSON object, no other text, in this exact shape:
{
  "answer": "<one to two sentence verdict, written entirely in the requested language, starting with TRUE, FALSE, or UNVERIFIABLE>",
  "truth_score": <integer 0-100, 0 = definitely false, 100 = definitely true, 50 = unverifiable / no evidence either way>,
  "reasoning_trace": ["<step 1: what the claim asserts>", "<step 2: what the context says, citing its source tag>", "<step 3: how they compare>"],
  "source": "<the source field of the context entry you relied on most>"
}
"""


def _extract_json(text: str) -> dict:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
        text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if match:
            try:
                return json.loads(match.group(0))
            except json.JSONDecodeError:
                pass
    return {"answer": text, "truth_score": 0, "reasoning_trace": [], "source": "unknown"}


def _call_model(model: str, system_prompt: str, user_prompt: str) -> dict:
    """One raw call to one Gonka-hosted model. Returns the parsed JSON plus
    the model name and Gonka's *real* request id for on-chain-proof
    transparency in the UI.

    Two things matter here for the receipt cross-check to actually work:
    - We use `.with_raw_response` to get at the HTTP response headers.
      The request id that GonkaRouter's `/v1/receipts/{id}` endpoint
      expects is the `X-Request-Id` RESPONSE HEADER — NOT the `id` field
      inside the JSON body (that body id is an internal devshard
      inference id, a different value, and won't resolve at /receipts).
    - We send `X-Gonka-No-Fallback: true` so a saturated upstream can't
      silently substitute a different model for the one we asked for —
      without this, our "two different models" consensus check could end
      up quietly comparing one model against itself.
    """
    raw_response = _client.chat.completions.with_raw_response.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        extra_headers={"X-Gonka-No-Fallback": "true"},
    )
    response = raw_response.parse()

    parsed = _extract_json(response.choices[0].message.content)
    parsed.setdefault("truth_score", 0)
    parsed.setdefault("reasoning_trace", [])
    parsed.setdefault("source", "unknown")
    parsed["model"] = model
    # The header lookup is case-insensitive (httpx.Headers), but Gonka
    # sends it as `X-Request-Id` — this is the id to show in the UI and
    # the id to hit GET /v1/receipts/{id} with.
    parsed["gonka_request_id"] = raw_response.headers.get("x-request-id")
    parsed["gonka_devshard_id"] = raw_response.headers.get("x-devshard-id")
    return parsed


def _reconcile(result_a: dict, result_b: dict) -> dict:
    """Multi-Model Consensus: combine two independent model votes into one
    result. Models are compared (not just averaged blindly) — if they
    disagree, that disagreement is surfaced and the truth score is pulled
    down to reflect the uncertainty, instead of being silently hidden."""
    answer_a = result_a.get("answer", "") or ""
    answer_b = result_b.get("answer", "") or ""
    score_a = result_a.get("truth_score", 0) or 0
    score_b = result_b.get("truth_score", 0) or 0

    agreement_score = SequenceMatcher(None, answer_a.lower(), answer_b.lower()).ratio()
    models_agree = agreement_score >= 0.6 and abs(score_a - score_b) <= 25

    # Prefer the more confident model's phrasing as the headline answer.
    final_answer = answer_a if score_a >= score_b else answer_b
    final_source = result_a.get("source") if score_a >= score_b else result_b.get("source")

    if models_agree:
        final_truth_score = round((score_a + score_b) / 2)
    else:
        # Disagreement penalty: trust the lower, more cautious score, and
        # discount it further since the models couldn't corroborate it.
        final_truth_score = max(0, round(min(score_a, score_b) * 0.7))

    reasoning_trace = list(result_a.get("reasoning_trace", [])) + list(result_b.get("reasoning_trace", []))
    if models_agree:
        reasoning_trace.append(
            f"Consensus: {result_a['model']} and {result_b['model']} agreed "
            f"(answer similarity {agreement_score:.2f}); scores averaged to {final_truth_score}."
        )
    else:
        reasoning_trace.append(
            f"Disagreement: {result_a['model']} and {result_b['model']} gave "
            f"differing answers (similarity {agreement_score:.2f}); truth score "
            f"discounted to {final_truth_score} to reflect the uncertainty."
        )

    return {
        "answer": final_answer,
        "truth_score": final_truth_score,
        "confidence": final_truth_score,  # kept for backward-compat with older frontend/main.py code
        "reasoning_trace": reasoning_trace,
        "source": final_source,
        "models_agree": models_agree,
        "agreement_score": round(agreement_score, 2),
        "model_votes": [
            {
                "model": result_a["model"],
                "answer": answer_a,
                "truth_score": score_a,
                "reasoning_trace": result_a.get("reasoning_trace", []),
                "gonka_request_id": result_a.get("gonka_request_id"),
                "gonka_devshard_id": result_a.get("gonka_devshard_id"),
            },
            {
                "model": result_b["model"],
                "answer": answer_b,
                "truth_score": score_b,
                "reasoning_trace": result_b.get("reasoning_trace", []),
                "gonka_request_id": result_b.get("gonka_request_id"),
                "gonka_devshard_id": result_b.get("gonka_devshard_id"),
            },
        ],
        # Kept as top-level fields too, for any code that still expects a
        # single request id / model name (points at the first model's call).
        "gonka_request_id": result_a.get("gonka_request_id"),
        "model": f"{result_a['model']} + {result_b['model']}",
    }


def ask_gonka(question: str, context_entries: list, target_lang: str = "en") -> dict:
    """
    Government-services Q&A. Sends the retrieved context + user question to
    TWO Gonka-hosted models and reconciles them into one consensus answer
    (see `_reconcile`). Returned dict includes `model_votes` with both raw
    per-model answers/scores/request ids for demo transparency.
    """
    context_block = "\n\n".join(
        f"[{e['source']}] {e['text_ms']}" for e in context_entries
    )
    language_name = _LANGUAGE_NAMES.get(target_lang, target_lang)

    user_prompt = (
        f"Context:\n{context_block}\n\n"
        f"Question ({language_name}): {question}\n\n"
        f"Answer in {language_name}."
    )

    result_a = _call_model(GONKA_MODEL, _QA_SYSTEM_PROMPT, user_prompt)
    result_b = _call_model(GONKA_MODEL_2, _QA_SYSTEM_PROMPT, user_prompt)
    return _reconcile(result_a, result_b)


def verify_claim(claim: str, context_entries: list, target_lang: str = "en") -> dict:
    """
    Fact-checking / claim verification. Sends the CLAIM plus supporting
    context (local KB and/or real-time web search results, gathered by
    main.py's /verify endpoint) to TWO Gonka-hosted models and reconciles
    them into a single Truth Score + reasoning trace.
    """
    context_block = "\n\n".join(
        f"[{e['source']}] {e['text_ms']}" for e in context_entries
    ) or "(no supporting context found)"
    language_name = _LANGUAGE_NAMES.get(target_lang, target_lang)

    user_prompt = (
        f"CLAIM: {claim}\n\n"
        f"CONTEXT:\n{context_block}\n\n"
        f"Respond in {language_name}."
    )

    result_a = _call_model(GONKA_MODEL, _VERIFY_SYSTEM_PROMPT, user_prompt)
    result_b = _call_model(GONKA_MODEL_2, _VERIFY_SYSTEM_PROMPT, user_prompt)
    return _reconcile(result_a, result_b)