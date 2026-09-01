"""
Wrapper around the Gonka Router API.

Gonka Router is OpenAI-compatible, so we reuse the official `openai` SDK
and just point it at Gonka's base_url. This is the ONLY place in the
codebase that should call an LLM for reasoning/verification, per the
hackathon's mandatory rule.

Double-check the base_url, available model names and auth header against
the current Gonka Router docs / workshop slides before the deadline —
routers occasionally rename models or move endpoints.
"""

import json
import os

from dotenv import load_dotenv
# pip install openai
from openai import OpenAI

load_dotenv()

GONKA_API_KEY = os.environ["GONKA_API_KEY"]
GONKA_BASE_URL = os.environ.get("GONKA_BASE_URL", "https://api.gonkarouter.io/v1")
GONKA_MODEL = os.environ.get("GONKA_MODEL", "MiniMaxAI/MiniMax-M2.7")

_client = OpenAI(
    api_key=GONKA_API_KEY,
    base_url=GONKA_BASE_URL,
)

_LANGUAGE_NAMES = {
    "ms": "Bahasa Malaysia",
    "en": "English",
    "zh": "Chinese",
}

_SYSTEM_PROMPT = """You are a Malaysian government services assistant.
Answer ONLY using the context provided below. Do not invent facts, fees,
or procedures that are not in the context.

Respond with a single JSON object, no other text, in this exact shape:
{
  "answer": "<answer written entirely in the requested language>",
  "confidence": <integer 0-100, how well the context supports this answer>,
  "source": "<the source field of the context entry you relied on most>"
}

If the context does not contain enough information to answer, set
"confidence" to 0 and say so honestly in "answer".
"""


def ask_gonka(question: str, context_entries: list, target_lang: str = "en") -> dict:
    """
    Sends the retrieved context + user question to Gonka Router and
    returns a dict with answer, confidence, source and the raw request id
    (for demo/judging transparency — show this id in your pitch video).
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

    response = _client.chat.completions.create(
        model=GONKA_MODEL,
        messages=[
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
    )

    raw_text = response.choices[0].message.content

    try:
        parsed = json.loads(raw_text)
    except json.JSONDecodeError:
        # Model didn't return clean JSON — fall back gracefully instead of crashing
        parsed = {"answer": raw_text, "confidence": 50, "source": "unclear"}

    parsed["gonka_request_id"] = getattr(response, "id", None)
    parsed["model"] = GONKA_MODEL
    return parsed
