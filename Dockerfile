FROM python:3.12-slim

# Some ML deps (tokenizers, chromadb's hnswlib) occasionally need to
# compile from source if no prebuilt wheel matches the platform — keep
# build-essential around for that, then drop apt's cache to keep the
# image lean.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first so Docker can cache this layer across
# rebuilds where only application code changes.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code — includes the pre-built backend/chroma_db/ vector store and
# backend/data/ source documents, so no `ingest.py` run is needed at
# container start.
COPY backend/ ./backend/

# Hugging Face Spaces expects the app to listen on port 7860; Railway,
# Fly.io, and Cloud Run all inject their own $PORT and override this
# default automatically, so this one image works unchanged on any of
# them.
ENV PORT=7860
EXPOSE 7860

# GONKA_API_KEY, GONKA_BASE_URL, GONKA_MODEL, GONKA_MODEL_2, and
# FIRECRAWL_API_KEY are NOT baked into the image — set them as
# secrets/env vars in whichever platform you deploy to.
CMD ["sh", "-c", "cd backend && uvicorn main:app --host 0.0.0.0 --port ${PORT:-7860}"]