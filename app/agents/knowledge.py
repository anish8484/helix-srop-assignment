from google.adk.agents import LlmAgent
from app.agents.tools.search_docs import search_docs, format_chunks_for_agent
from app.settings import settings

async def knowledge_search_tool(query: str) -> str:
    """
    Search the product documentation to answer technical or feature questions.
    """
    chunks = await search_docs(query, k=5)
    return format_chunks_for_agent(chunks)

knowledge_agent = LlmAgent(
    name="knowledge",
    model=settings.adk_model,
    instruction="""
You are the Helix Knowledge Specialist.
Your job is to answer questions about Helix products, features, and technical processes using the documentation.

Rules:
1. Always use the 'knowledge_search_tool' to find answers in the documentation.
2. Answer ONLY using the provided context chunks.
3. Always cite the chunk ID for every claim (e.g., "According to [chunk_abc123]...").
4. If the context doesn't contain the answer, say "I'm sorry, I don't have documentation on that." Do not hallucinate.
5. Be concise and technical.
""",
    tools=[knowledge_search_tool],
)
