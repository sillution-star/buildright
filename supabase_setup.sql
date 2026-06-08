-- Run this in your Supabase SQL Editor (supabase.com > your project > SQL Editor)
-- Do this BEFORE running ingest.py

-- 1. Enable the pgvector extension
create extension if not exists vector;

-- 2. Create the chunks table
create table if not exists mom_test_chunks (
  id bigserial primary key,
  chunk_index integer not null,
  content text not null,
  embedding vector(384)  -- 384 dimensions = all-MiniLM-L6-v2 output size
);

-- 3. Create an index for fast similarity search
create index if not exists mom_test_chunks_embedding_idx
  on mom_test_chunks
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 50);

-- 4. Create the match function (called by the backend for RAG retrieval)
create or replace function match_mom_test_chunks(
  query_embedding vector(384),
  match_count int default 5
)
returns table (
  id bigint,
  chunk_index integer,
  content text,
  similarity float
)
language sql stable
as $$
  select
    id,
    chunk_index,
    content,
    1 - (embedding <=> query_embedding) as similarity
  from mom_test_chunks
  order by embedding <=> query_embedding
  limit match_count;
$$;

-- Done. Now run: python scripts/ingest.py
