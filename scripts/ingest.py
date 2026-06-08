"""
ingest.py — Run this ONCE to chunk the Mom Test book and store embeddings in Supabase.
Usage: python scripts/ingest.py
"""

import os
import re
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(dotenv_path=Path(__file__).parent.parent / "backend" / ".env")

SUPABASE_URL = os.environ["SUPABASE_URL"]
SUPABASE_KEY = os.environ["SUPABASE_SERVICE_KEY"]
BOOK_PATH = Path(__file__).parent.parent / "backend" / "mom-test.md"
CHUNK_SIZE = 400        # words per chunk
CHUNK_OVERLAP = 80      # word overlap between chunks


def load_book(path: Path) -> str:
    text = path.read_text(encoding="utf-8")
    # Remove page numbers (lone digits on a line)
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    # Collapse excess blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_into_chunks(text: str, chunk_size: int, overlap: int) -> list[dict]:
    words = text.split()
    chunks = []
    i = 0
    chunk_index = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words)
        chunks.append({
            "chunk_index": chunk_index,
            "text": chunk_text,
            "word_count": len(chunk_words),
        })
        chunk_index += 1
        i += chunk_size - overlap
    return chunks


def get_embeddings(texts: list[str]) -> list[list[float]]:
    """Generate embeddings locally using sentence-transformers (free, no API key)."""
    from sentence_transformers import SentenceTransformer
    print("Loading embedding model (downloads once ~80MB)...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    print(f"Generating embeddings for {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True, batch_size=32)
    return embeddings.tolist()


def store_in_supabase(chunks: list[dict], embeddings: list[list[float]]):
    """Store chunks + embeddings in Supabase pgvector."""
    import httpx

    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
    }

    # First: create the table if it doesn't exist (run via Supabase SQL editor)
    print("\n⚠️  Make sure you've run the SQL setup in Supabase (see README).\n")

    rows = []
    for chunk, embedding in zip(chunks, embeddings):
        rows.append({
            "chunk_index": chunk["chunk_index"],
            "content": chunk["text"],
            "embedding": embedding,
        })

    # Insert in batches of 50
    batch_size = 50
    total = len(rows)
    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        resp = httpx.post(
            f"{SUPABASE_URL}/rest/v1/mom_test_chunks",
            headers=headers,
            json=batch,
            timeout=60,
        )
        if resp.status_code not in (200, 201):
            print(f"Error inserting batch {start}: {resp.text}")
        else:
            print(f"Inserted chunks {start}–{start + len(batch) - 1} / {total}")

    print(f"\n✅ Done! {total} chunks stored in Supabase.")


def main():
    print("📖 Loading Mom Test book...")
    text = load_book(BOOK_PATH)
    print(f"Book loaded: {len(text.split())} words")

    print("✂️  Chunking...")
    chunks = split_into_chunks(text, CHUNK_SIZE, CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks")

    texts = [c["text"] for c in chunks]
    embeddings = get_embeddings(texts)

    print("☁️  Storing in Supabase...")
    store_in_supabase(chunks, embeddings)


if __name__ == "__main__":
    main()
