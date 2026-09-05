# MyAssist — Government Services Assistant

**MUBA Hacks 2026 — Gonka Track**

## Links

- **Live Deployment:** _TODO: paste your Vercel deployment URL here_
- **Pitch Video (YouTube):https://youtu.be/nT3HOSg3qHE

## Project Description

MyAssist is a trilingual (Malay / English / Chinese) chat assistant for
Malaysian government services — MyKad (identity card), passport, and
driving licence. Users type a question, a claim, or even just a topic
in any of the three languages, and get back either:

- a **grounded answer** (Q&A), retrieved from official JPN/JPJ reference
  material and reasoned over by two independent AI models, or
- a **Truth Score verdict** (claim verification) when the input is a
  factual claim or a pasted URL/tweet about a government service, or
- a **clarifying follow-up question** when the input is too vague to
  answer safely (e.g. just "identity card") — instead of guessing.

Every answer is backed by a **multi-model consensus**: the same
question/claim is sent to two different models on the Gonka Router
concurrently, and their answers are reconciled in code — if they agree,
scores are averaged; if they disagree, the score is discounted and the
disagreement is shown, not hidden. Every response carries the real
Gonka Router request ID for each model call, so the reasoning is
independently verifiable and not just a single black-box opinion.

## Problem Statement

Malaysians looking up government service info (MyKad renewal, passport
fees, driving licence procedures) run into two problems:

1. **Language barriers** — official info is often only in Bahasa
   Malaysia, and no single centralized help channel answers fluently in
   Malay, English, and Chinese.
2. **Misinformation and stale info** — fees, procedures, and rules
   circulate on social media and WhatsApp forwards, often outdated or
   simply wrong, with no easy way for an ordinary citizen to check them
   against an authoritative source and get a transparent, non-partisan
   verdict rather than a single unaccountable answer.

MyAssist addresses both: it answers in whichever of the 3 languages the
user typed in, grounds every answer in retrieved official reference
text, and fact-checks claims with a transparent, multi-model,
disagreement-aware verdict instead of one model's unchecked opinion.

## Blockchain / Decentralized Technology Used

- **Gonka Network** (via `api.gonkarouter.io`, OpenAI-compatible
  gateway) — ALL reasoning/verification calls in this project are
  routed through Gonka Router, per the track's mandatory rule. See
  `backend/gonka_client.py`.
- **Multi-model consensus** — every question/claim is sent to two
  independently-selected Gonka-hosted models concurrently; their
  answers are reconciled into one Truth Score, so no single centralized
  model's opinion is presented as ground truth.
- **On-chain-style proof** — each model call returns Gonka's real
  `X-Request-Id` response header (not an internal body ID), which is
  surfaced in the UI/API response (`gonka_request_id` / `model_votes[].gonka_request_id`)
  so a judge can independently confirm the inference was actually run
  on the Gonka Network, not faked.

**Smart Contract Addresses (Testnet):** N/A — this project does not
deploy or call a custom smart contract. Verifiability comes from Gonka
Router's request-ID / receipt system described above, not an on-chain
contract we wrote. _(If your team did deploy a contract for this
submission, add its testnet address(es) and network here.)_

## Tech Stack

| Layer                    | Tech                                                                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------- |
| Frontend                 | Plain HTML/JS chat UI (`frontend/index.html`)                                                                                         |
| Backend                  | FastAPI (`backend/main.py`) — single `/chat` endpoint, plus `/ask` and `/verify` for direct testing                                   |
| Retrieval                | LangChain + Chroma vector store, multilingual sentence embeddings (`backend/rag.py`, `backend/ingest.py`) — purely local, no LLM call |
| Reasoning / verification | Gonka Router, two-model consensus (`backend/gonka_client.py`)                                                                         |
| Live web evidence        | Firecrawl (`backend/web_scraper.py`) — used as a fallback when local knowledge base similarity is low, or to scrape a pasted URL      |

## Setup and Installation

**Requirements:** Python 3.10+, a Gonka Router API key, a Firecrawl API key (optional but recommended).

```bash
# 1. Clone and enter the project
git clone <this-repo-url>
cd gov-services-assistant

# 2. Install backend dependencies
cd backend
pip install -r ../requirements.txt

# 3. Configure environment variables
# Create a .env file in the PROJECT ROOT (one level above backend/) with:
#   GONKA_API_KEY=your_key_here
#   GONKA_BASE_URL=https://api.gonkarouter.io/v1
#   GONKA_MODEL=<model 1 from Gonka's catalog>
#   GONKA_MODEL_2=<a different model 2, for real cross-verification>
#   FIRECRAWL_API_KEY=your_key_here   (optional — enables live web fallback)

# 4. Build the local knowledge base (run once, and again whenever files
#    under backend/data/ change)
python ingest.py

# 5. Start the backend
python -m uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` directly in a browser (or serve it with
any static file server). Confirm the `API_URL` constant in its
`<script>` tag matches where the backend is running
(`http://127.0.0.1:8000` by default).

Health check: `GET http://127.0.0.1:8000/health` should return
`{"status": "ok", "kb_entries": <n>}`.

## AI Tools Used During Development

In addition to Gonka Router (the in-product, mandatory inference
layer described above), the following AI coding assistants were used
during development of this project:

- **Claude** (Anthropic) — code generation, debugging, and refactoring assistance.
- **ChatGPT** (OpenAI) — code generation and debugging assistance.
- **DeepSeek** — code generation and debugging assistance.

## Team Members

_(fill in your team's names and roles below)_

| Name           | Role                |
| -------------- | ------------------- |
| _CHAN HUI ERN_ | _PROJECT DEVELOPER_ |
| _LIM JIAN JUN_ | _PROJECT DEVELOPER_ |

## Notes on Scope

This is an MVP built within a hackathon timeframe — it deliberately
covers only MyKad, passport, and driving licence to keep the scope
realistic, rather than half-covering every government service.
Fee/procedure information may change; the UI should always point users
back to the official JPN/JPJ portal for the current, authoritative
figures.
