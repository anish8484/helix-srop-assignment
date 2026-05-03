"""
RAG ingest CLI.

Usage:
    python -m app.rag.ingest --path docs/
    python -m app.rag.ingest --path docs/ --chunk-size 512 --chunk-overlap 64

Reads markdown files, chunks them, embeds, and writes to the vector store.
"""
import argparse
import asyncio
import hashlib
import re
from pathlib import Path
from typing import Any

import yaml
import google.generativeai as genai
import chromadb
from app.settings import settings

def chunk_markdown(text: str, max_chars: int = 800) -> list[str]:
    """
    Split markdown text into coherent chunks by heading and sentence boundaries.
    """
    # Split on any markdown heading (## or ###)
    sections = re.split(r'\n(?=#{2,3} )', text)
    chunks = []
    
    def chunk_sentences(text: str, max_chars: int = 512, overlap_sentences: int = 1) -> list[str]:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current, current_len = [], [], 0
        for sentence in sentences:
            if current_len + len(sentence) > max_chars and current:
                chunks.append(" ".join(current))
                current = current[-overlap_sentences:]  # keep last N sentences as overlap
                current_len = sum(len(s) for s in current)
            current.append(sentence)
            current_len += len(sentence)
        if current:
            chunks.append(" ".join(current))
        return chunks

    for section in sections:
        section = section.strip()
        if not section:
            continue
            
        if len(section) <= max_chars:
            chunks.append(section)
        else:
            # Sub-chunk long sections by sentence
            chunks.extend(chunk_sentences(section, max_chars=max_chars))
            
    return [c for c in chunks if c.strip()]


def extract_metadata(file_path: Path, text: str) -> tuple[dict[str, Any], str]:
    """
    Extract metadata from a markdown file's frontmatter.
    """
    match = re.match(r'^---\n(.*?)\n---\n', text, re.DOTALL)
    if not match:
        return {"source": file_path.name}, text
        
    try:
        metadata = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        metadata = {}
        
    metadata["source"] = file_path.name
    body = text[match.end():]
    return metadata, body


def make_chunk_id(file_name: str, chunk_index: int) -> str:
    raw = f"{file_name}::{chunk_index}"
    return "chunk_" + hashlib.sha256(raw.encode()).hexdigest()[:16]


async def ingest_directory(docs_path: Path, chunk_size: int, chunk_overlap: int) -> None:
    """
    Walk docs_path, chunk and embed every .md file, upsert into vector store.
    """
    if not settings.google_api_key:
        print("Error: GOOGLE_API_KEY not set in environment or .env")
        return

    genai.configure(api_key=settings.google_api_key)
    
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(
        name="helix_docs",
        metadata={"hnsw:space": "cosine"}
    )

    md_files = list(docs_path.rglob("*.md"))
    print(f"Found {len(md_files)} markdown files in {docs_path}")

    for file_path in md_files:
        text = file_path.read_text(encoding="utf-8")
        metadata, body = extract_metadata(file_path, text)
        chunks = chunk_markdown(body, max_chars=chunk_size)
        print(f"  {file_path.name}: {len(chunks)} chunks")
        
        if not chunks:
            continue

        # Embed chunks in one batch
        try:
            result = genai.embed_content(
                model="models/text-embedding-004",
                content=chunks,
                task_type="retrieval_document",
            )
            embeddings = result["embeddings"]
            
            ids = [make_chunk_id(file_path.name, i) for i in range(len(chunks))]
            metadatas = [metadata for _ in chunks]
            
            collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
        except Exception as e:
            print(f"    Error embedding {file_path.name}: {e}")

    print("Ingest complete.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest docs into the vector store")
    parser.add_argument("--path", type=Path, required=True, help="Directory containing .md files")
    parser.add_argument("--chunk-size", type=int, default=800)
    parser.add_argument("--chunk-overlap", type=int, default=100) # Unused in my heading-aware impl but kept for CLI compatibility
    args = parser.parse_args()

    asyncio.run(ingest_directory(args.path, args.chunk_size, args.chunk_overlap))


if __name__ == "__main__":
    main()
