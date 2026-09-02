"""
Retrieval layer for the government service assistant, backed by LangChain
+ a local Chroma vector store built by `ingest.py` from files in data/.

This does NOT call any LLM — it only finds the most relevant chunks for a
user's question. The actual "reasoning" step (turning chunks into an
answer) happens in gonka_client.py, which is the mandatory Gonka Router
call.

Run `python ingest.py` at least once before starting the backend, and
again any time you change files in data/.
"""

from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

PERSIST_DIR = Path(__file__).parent / "chroma_db"

# Must match the model used in ingest.py — a mismatch here would mean
# queries and stored chunks are embedded differently, silently breaking
# retrieval quality.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"


class KnowledgeBase:
    def __init__(self, persist_directory: Path = PERSIST_DIR):
        if not persist_directory.exists():
            raise RuntimeError(
                f"No vector store found at {persist_directory}. "
                "Run `python ingest.py` first to build it from data/."
            )

        self.embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
        self.store = Chroma(
            persist_directory=str(persist_directory),
            embedding_function=self.embeddings,
        )

    @property
    def entries(self):
        """Kept so /health's `kb_entries` count still works — returns the
        number of stored chunks, not source documents."""
        return self.store.get()["ids"]

    def retrieve(self, query: str, top_k: int = 3):
        results = self.store.similarity_search_with_relevance_scores(query, k=top_k)

        formatted = []
        for doc, score in results:
            source = doc.metadata.get("source", "unknown")
            formatted.append({
                "id": f"{source}#{doc.metadata.get('page', 0)}",
                "category": doc.metadata.get("category", "general"),
                # NOTE: key kept as "text_ms" for compatibility with the
                # existing gonka_client.py, even though chunks may now
                # come from any-language source documents, not just Malay.
                "text_ms": doc.page_content,
                "source": source,
                "similarity": float(score),
            })
        return formatted


# Singleton instance loaded once when the backend starts
knowledge_base = KnowledgeBase()


# --- Backward-compatible function-style interface -------------------------
# Some versions of main.py import these directly instead of using the
# `knowledge_base` object above. Both styles hit the same underlying store.

def retrieve_relevant_context(query: str, top_k: int = 3):
    return knowledge_base.retrieve(query, top_k=top_k)


def knowledge_base_status():
    return {"status": "ok", "kb_entries": len(knowledge_base.entries)}