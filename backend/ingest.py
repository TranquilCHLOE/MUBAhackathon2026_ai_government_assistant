"""
Run this again any time you add, remove, or edit files in data/:

    python ingest.py
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

DATA_DIR = Path(__file__).parent / "data"
PERSIST_DIR = Path(__file__).parent / "chroma_db"

# Same multilingual model the old sentence-transformers rag.py used —
# understands Malay, English, and Chinese well enough to match queries
# in any of the 3 languages against source documents.
EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def load_documents():
    """Load every .txt/.pdf under data/ as LangChain Documents, tagging
    each with a human-readable 'source' in its metadata (for citing in
    answers). .txt files are expected to start with a 'Source: <name>'
    line; if that's missing, the filename is used instead."""
    docs = []
    for path in sorted(DATA_DIR.glob("**/*")):
        if path.suffix.lower() == ".txt":
            loaded = TextLoader(str(path), encoding="utf-8").load()
        elif path.suffix.lower() == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
        else:
            continue

        for doc in loaded:
            first_line = doc.page_content.strip().splitlines()[0] if doc.page_content.strip() else ""
            if first_line.lower().startswith("source:"):
                doc.metadata["source"] = first_line.split(":", 1)[1].strip()
            else:
                doc.metadata.setdefault("source", path.stem)
            docs.append(doc)

    return docs


def main():
    docs = load_documents()
    if not docs:
        raise SystemExit(
            f"No .txt or .pdf files found in {DATA_DIR}. "
            "Add your government handbooks/FAQs/guidelines there first."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)

    print(f"Loaded {len(docs)} document(s), split into {len(chunks)} chunk(s).")
    print(f"Embedding with '{EMBEDDING_MODEL}' (first run downloads the model)...")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=str(PERSIST_DIR),
        # Force cosine distance so similarity_search_with_relevance_scores()
        # in rag.py returns proper 0-1 scores. Without this, Chroma defaults
        # to raw L2 distance, which similarity_search_with_relevance_scores()
        # still tries to convert as if it were cosine — producing negative
        # scores like -13.36 (and the "must be between 0 and 1" warning).
        # Those negative scores never pass main.py's `> max_similarity`
        # check (which starts at 0.0), so max_similarity is always 0.0 and
        # every query wrongly falls through to the web-scrape fallback.
        collection_metadata={"hnsw:space": "cosine"},
    )

    print(f"Done. Vector store persisted to {PERSIST_DIR}")


if __name__ == "__main__":
    main()