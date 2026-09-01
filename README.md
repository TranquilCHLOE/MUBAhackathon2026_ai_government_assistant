# Government Services Assistant — MUBA Hacks 2026 (Gonka track)

A 3-language (Malay / English / Chinese) assistant that answers questions
about Malaysian government services (MyKad, passport, driving license)
using retrieval over official JPN/JPJ information, with all reasoning
and verification routed through **Gonka Router** as required by the
track rules.

## How it works

1. **Frontend** (`frontend/index.html`) — plain HTML/JS chat with a language dropdown.
2. **Backend** (`backend/main.py`) — FastAPI orchestrator, single `/ask` endpoint.
3. **Retrieval** (`backend/rag.py`) — multilingual sentence embeddings match
   the user's question (in any of the 3 languages) against the knowledge
   base, purely local, no LLM call.
4. **Reasoning** (`backend/gonka_client.py`) — the retrieved context and
   question are sent to **Gonka Router** (OpenAI-compatible API), which
   returns the answer, a confidence score, and the source it relied on.

Every response includes the Gonka Router request id — keep this visible
in your demo video as proof the mandatory integration is real and live.

## Setup

```bash
cd backend
pip install -r requirements.txt
cp .env.example .env   # then fill in your real GONKA_API_KEY (never commit .env)
uvicorn main:app --reload --port 8000
```

Open `frontend/index.html` directly in a browser (or serve it with any
static file server). Make sure `API_URL` in the `<script>` matches where
your backend is running.

## Before you submit — checklist

- [ ] Confirm the model name and base_url in `.env` still match the
      current Gonka Router docs (routers occasionally add/rename models).
- [ ] Expand `knowledge_base.json` with more entries from your compiled
      JPN/JPJ notes — 3 sample entries are included as a starting point.
- [ ] `git init` this repo no earlier than 26 August 2026 — organizers
      may inspect commit timestamps (Section 2 of the rules).
- [ ] Add a short note in your README/pitch declaring every AI tool used
      during development (e.g. Claude, Copilot), separate from the
      in-product Gonka Router usage.
- [ ] Record a demo video showing a real question being answered in each
      of the 3 languages, with the Gonka request id visible.
- [ ] Add a disclaimer in the UI that fee/procedure info may change —
      always verify with the official JPN/JPJ portal.
- [ ] Make sure `.env` (with your real key) is in `.gitignore` and was
      never committed — `.env.example` should only ever contain a
      placeholder, not a working key.

## Notes on scope

This is a minimal MVP skeleton, not a finished product — it deliberately
covers only MyKad, passport and driving license to keep the hacking
period realistic. Better to fully polish 2-3 service categories with a
working live demo than to half-cover everything.

## usually how they do this kind of project
Step 1: Gather Your Government DataCollect all official handbooks, FAQ spreadsheets, laws, or application guidelines into a folder. This is your system’s source of truth so the AI never guesses.

Step 2: Set Up a RAG Framework (e.g., LangChain)You will use a coding framework like LangChain or LlamaIndex to read those files. LangChain will chop the documents into small pieces and store them in a local Vector Database (like ChromaDB or FAISS).

Step 3: Use GonkaRouter to Answer the QuestionWhen a user asks a question, your code will look up the relevant laws in your database, pass them to GonkaRouter, and let the model generate the answer.Here is exactly how the backend Python code looks using GonkaRouter's AI endpoint:pythonimport openai

# 1. Initialize GonkaRouter
client = openai.OpenAI(
    base_url="https://api.gonkarouter.io/v1",  # Gonka's Gateway URL
    api_key="your_gonkarouter_api_key"        # Your secret key from gonkarouter.io
)

# 2. Simulate retrieving official government text from your database
retrieved_government_policy = """
POLICY-ID 402: Small Business Grants are available to local citizens with 
fewer than 10 employees. Applications must be submitted through portal.gov 
before October 31st.
"""

user_question = "Can a shop with 5 employees get the small business grant, and when is the deadline?"

# 3. Combine your official data with the user's question (The Prompt)
messages = [
    {
        "role": "system", 
        "content": (
            "You are an official government assistant. Answer the user's question "
            "ONLY using the provided context. If the answer cannot be found in the context, "
            "politely say you do not know."
        )
    },
    {
        "role": "user", 
        "content": f"Context:\n{retrieved_government_policy}\n\nQuestion: {user_question}"
    }
]

# 4. Route it to a fast, cost-efficient model via GonkaRouter
response = client.chat.completions.create(
    model="deepseek/DeepSeek-V4-Flash-0731",  # Choose your model from Gonka's catalog
    messages=messages,
    temperature=0.0                           # Keep temperature at 0.0 for strict factual accuracy
)

print(response.choices.message.content)
Use code with caution.Step 4: Hook it up to a Frontend UITo make this useful for everyday citizens, build a simple chat window website using framework systems like Streamlit (for quick Python prototyping) or Next.js/React for a public web application.
