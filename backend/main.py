from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field 

from rag import knowledge_base
from gonka_client import ask_gonka

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

@app.post("/ask")
def ask(req: AskRequest):
    context_entries = knowledge_base.retrieve(req.question, top_k=3)

    # --- SAFETY GATE (New!) ---
    if not context_entries:
        return {
            "answer": "I could not find sufficiently relevant official government information in the current knowledge base. Please try a MyKad, passport, or driving licence question.",
            "confidence": 0,
            "source": None,
            "retrieved": [],
        }

    result = ask_gonka(req.question, context_entries, target_lang=req.language)
    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "source": result["source"],
        "gonka_request_id": result["gonka_request_id"],
        "model": result["model"],
        "retrieved": context_entries,
    }


@app.get("/health")
def health():
    return {"status": "ok", "kb_entries": len(knowledge_base.entries)}
