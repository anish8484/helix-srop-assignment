import chromadb
import google.generativeai as genai
from pydantic import BaseModel, Field
from app.settings import settings

class DocChunk(BaseModel):
    chunk_id: str
    score: float
    content: str
    metadata: dict

async def search_docs(query: str, k: int = 5, product_area: str | None = None) -> list[DocChunk]:
    """
    Search the Helix product documentation for the given query.
    Returns the most relevant chunks with their similarity scores and chunk IDs.
    
    Args:
        query: The user's question or search terms.
        k: Number of results to return (default 5).
        product_area: Optional filter for a specific product area (e.g., 'security', 'billing').
    """
    genai.configure(api_key=settings.google_api_key)
    
    # Embed the query
    result = genai.embed_content(
        model="models/text-embedding-004",
        content=query,
        task_type="retrieval_query",
    )
    query_embedding = result["embedding"]

    # Connect to Chroma
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    collection = chroma_client.get_or_create_collection(name="helix_docs")

    # Build metadata filter
    where = {"product_area": product_area} if product_area else None

    # Query Chroma
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
    )

    chunks = []
    if not results["ids"] or not results["ids"][0]:
        return []

    for chunk_id, distance, doc, meta in zip(
        results["ids"][0],
        results["distances"][0],
        results["documents"][0],
        results["metadatas"][0],
    ):
        # Convert cosine distance to score (1 - distance)
        score = round(1.0 - distance, 4)
        chunks.append(DocChunk(
            chunk_id=chunk_id,
            score=score,
            content=doc,
            metadata=meta,
        ))

    # Sort by score descending
    chunks.sort(key=lambda x: x.score, reverse=True)
    return chunks

def format_chunks_for_agent(chunks: list[DocChunk]) -> str:
    """Format chunks into a string for the LLM prompt."""
    if not chunks:
        return "No relevant documentation found."
        
    parts = []
    for chunk in chunks:
        parts.append(
            f"[{chunk.chunk_id}] (score: {chunk.score:.2f}, source: {chunk.metadata.get('source')})\n"
            f"{chunk.content}"
        )
    return "\n\n---\n\n".join(parts)
