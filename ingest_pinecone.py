#!/usr/bin/env python3
"""
ingest_pinecone.py — Ingest documents into a Pinecone vector index.

Reads text files from a local directory, splits them into chunks,
embeds them with Ollama (mxbai-embed-large), and upserts into Pinecone.

The index is created automatically if it doesn't exist (serverless, cosine).

Usage:
    # Set your API key in .env or as env var:
    export PINECONE_API_KEY="pcsk_..."

    # Run:
    python ingest_pinecone.py

    # Or with custom settings:
    PINECONE_INDEX_NAME=my-index DOCS_DIR=/path/to/docs python ingest_pinecone.py

Requirements:
    pip install pinecone langchain-ollama langchain-text-splitters langchain-community
"""

import os
import sys
import time
from collections import defaultdict
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from pinecone import Pinecone, ServerlessSpec
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, TextLoader

# ─── Configuration ────────────────────────────────────────────────────────────

PINECONE_API_KEY    = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "rag-wiki-bench")
PINECONE_CLOUD      = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION     = os.getenv("PINECONE_REGION", "us-east-1")
PINECONE_NAMESPACE  = os.getenv("PINECONE_NAMESPACE", "")

DOCS_DIR            = os.getenv("DOCS_DIR", "/Users/sstprk/Desktop/sample_docs")
EMBED_MODEL         = os.getenv("EMBED_MODEL", "mxbai-embed-large:latest")
EMBED_DIMENSION     = 1024   # mxbai-embed-large dimension
CHUNK_SIZE          = 500
CHUNK_OVERLAP       = 50
BATCH_SIZE          = 50     # vectors per upsert batch

# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    if not PINECONE_API_KEY:
        print("❌ PINECONE_API_KEY not set. Add it to .env or export it.")
        sys.exit(1)

    docs_path = Path(DOCS_DIR)
    if not docs_path.exists():
        print(f"❌ Documents directory not found: {DOCS_DIR}")
        sys.exit(1)

    print("=" * 60)
    print("Pinecone Document Ingestion")
    print("=" * 60)

    # ── Step 1: Load documents ────────────────────────────────────────────────
    print(f"\n[1/5] Loading documents from {DOCS_DIR}...")
    loader = DirectoryLoader(
        DOCS_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    docs = loader.load()
    print(f"  Loaded {len(docs)} document(s)")

    if not docs:
        print("❌ No .txt files found. Nothing to ingest.")
        sys.exit(1)

    for doc in docs:
        print(f"  • {Path(doc.metadata.get('source', '')).name}")

    # ── Step 2: Split into chunks ─────────────────────────────────────────────
    print(f"\n[2/5] Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)

    # Inject metadata
    counts = defaultdict(int)
    for chunk in chunks:
        source = chunk.metadata.get("source", "unknown")
        doc_id = source
        doc_title = Path(source).stem
        chunk.metadata["doc_id"] = doc_id
        chunk.metadata["doc_title"] = doc_title
        chunk.metadata["doc_path"] = source
        chunk.metadata["chunk_index"] = counts[doc_id]
        chunk.metadata["total_chunks"] = 0  # will fill after
        counts[doc_id] += 1

    # Fill total_chunks
    for chunk in chunks:
        chunk.metadata["total_chunks"] = counts[chunk.metadata["doc_id"]]

    print(f"  {len(chunks)} chunks from {len(counts)} documents")

    # ── Step 3: Embed all chunks ──────────────────────────────────────────────
    print(f"\n[3/5] Embedding chunks with {EMBED_MODEL}...")
    embed_model = OllamaEmbeddings(model=EMBED_MODEL)

    texts = [c.page_content for c in chunks]
    t0 = time.perf_counter()
    vectors = embed_model.embed_documents(texts)
    embed_time = time.perf_counter() - t0
    print(f"  Embedded {len(vectors)} chunks in {embed_time:.1f}s "
          f"({len(vectors)/embed_time:.1f} chunks/s)")
    print(f"  Embedding dimension: {len(vectors[0])}")

    # ── Step 4: Create/connect to Pinecone index ─────────────────────────────
    print(f"\n[4/5] Connecting to Pinecone index '{PINECONE_INDEX_NAME}'...")
    pc = Pinecone(api_key=PINECONE_API_KEY)

    if not pc.has_index(PINECONE_INDEX_NAME):
        print(f"  Index not found — creating '{PINECONE_INDEX_NAME}' "
              f"(dim={EMBED_DIMENSION}, metric=cosine, {PINECONE_CLOUD}/{PINECONE_REGION})...")
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBED_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        )
        # Wait for the index to be ready
        print("  Waiting for index to be ready...", end="", flush=True)
        while not pc.describe_index(PINECONE_INDEX_NAME).status.get("ready", False):
            time.sleep(1)
            print(".", end="", flush=True)
        print(" ready!")
    else:
        print(f"  Index '{PINECONE_INDEX_NAME}' already exists.")

    index = pc.Index(PINECONE_INDEX_NAME)
    stats = index.describe_index_stats()
    print(f"  Current vectors in index: {stats.get('total_vector_count', 0)}")

    # ── Step 5: Upsert vectors ────────────────────────────────────────────────
    print(f"\n[5/5] Upserting {len(chunks)} vectors (batch_size={BATCH_SIZE})...")

    records = []
    for i, (chunk, vector) in enumerate(zip(chunks, vectors)):
        meta = chunk.metadata
        # Pinecone metadata values must be str, int, float, bool, or list of str
        record_metadata = {
            "text":            chunk.page_content,
            "doc_id":          meta["doc_id"],
            "doc_title":       meta["doc_title"],
            "doc_path":        meta["doc_path"],
            "chunk_index":     meta["chunk_index"],
            "total_chunks":    meta["total_chunks"],
        }
        # Section heading from splitter (if present)
        if meta.get("section_heading"):
            record_metadata["section_heading"] = meta["section_heading"]

        vec_id = f"{meta['doc_id']}::chunk-{meta['chunk_index']}"
        # Sanitise the ID (Pinecone IDs must be ASCII, max 512 chars)
        vec_id = "".join(c if c.isalnum() or c in "-_:." else "_" for c in vec_id)[:512]

        records.append({
            "id":       vec_id,
            "values":   vector,
            "metadata": record_metadata,
        })

    # Upsert in batches
    namespace = PINECONE_NAMESPACE or None
    total_upserted = 0
    for batch_start in range(0, len(records), BATCH_SIZE):
        batch = records[batch_start : batch_start + BATCH_SIZE]
        index.upsert(vectors=batch, namespace=namespace)
        total_upserted += len(batch)
        print(f"  Upserted {total_upserted}/{len(records)} vectors...")

    # Final stats
    time.sleep(2)  # brief pause for index to update stats
    stats = index.describe_index_stats()
    print(f"\n{'=' * 60}")
    print(f"✅ Ingestion complete!")
    print(f"  Documents:  {len(docs)}")
    print(f"  Chunks:     {len(chunks)}")
    print(f"  Index:      {PINECONE_INDEX_NAME}")
    print(f"  Namespace:  {PINECONE_NAMESPACE or '(default)'}")
    print(f"  Vectors:    {stats.get('total_vector_count', '?')}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
