"""
Retrieval layer for the government service assistant.

This does NOT call any LLM — it only finds the most relevant knowledge-base
entries for a user's question, in whatever language they typed it in.
The actual "reasoning" step (turning these entries into an answer) happens
in gonka_client.py, which is the mandatory Gonka Router call.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer, util

KB_PATH = Path(__file__).parent / "knowledge_base.json"

# Multilingual model: understands Malay, English and Chinese queries well
# enough to match them against Malay-language source documents.
_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"


class KnowledgeBase:
    def __init__(self, path: Path = KB_PATH):
        with open(path, "r", encoding="utf-8") as f:
            self.entries = json.load(f)

        self.model = SentenceTransformer(_MODEL_NAME)

        # Embed each entry once at startup using its Malay text + keywords,
        # so queries in any of the 3 languages can still match it.
        corpus_texts = [
            f"{e['text_ms']} {' '.join(e.get('keywords', []))}"
            for e in self.entries
        ]
        self.corpus_embeddings = self.model.encode(
            corpus_texts, convert_to_tensor=True
        )

    def retrieve(self, query: str, top_k: int = 3):
        query_embedding = self.model.encode(query, convert_to_tensor=True)
        scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]
        top_results = scores.topk(min(top_k, len(self.entries)))

        results = []
        for score, idx in zip(top_results.values, top_results.indices):
            entry = self.entries[int(idx)]
            results.append({
                "id": entry["id"],
                "category": entry["category"],
                "text_ms": entry["text_ms"],
                "source": entry["source"],
                "similarity": float(score),
            })
        return results


# Singleton instance loaded once when the backend starts
knowledge_base = KnowledgeBase()
