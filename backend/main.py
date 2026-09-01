from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    question: str
    language: str = "en"  # "ms" | "en" | "zh"


@app.post("/ask")
def ask(req: AskRequest):
    # 1. Retrieval — find the most relevant official-source entries
    context_entries = knowledge_base.retrieve(req.question, top_k=3)

    # 2. Reasoning — mandatory Gonka Router call turns context into an answer
    result = ask_gonka(req.question, context_entries, target_lang=req.language)

    return {
        "answer": result["answer"],
        "confidence": result["confidence"],
        "source": result["source"],
        "gonka_request_id": result["gonka_request_id"],
        "model": result["model"],
        "retrieved": context_entries,  # useful for debugging / judges' curiosity
    }


@app.get("/health")
def health():
    return {"status": "ok", "kb_entries": len(knowledge_base.entries)}
