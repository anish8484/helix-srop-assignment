import asyncio
import uuid
from unittest.mock import AsyncMock, MagicMock
import httpx
from fastapi.testclient import TestClient
from app.main import app
from app.db.session import init_db, get_db
from app.srop.pipeline import PipelineResult
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from app.db.models import Base

# Setup in-memory test DB
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"
test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)

async def override_get_db():
    async with TestSessionLocal() as session:
        yield session

app.dependency_overrides[get_db] = override_get_db

async def run_demo():
    # 1. Initialize DB
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    
    print("--- 1. Initializing DB ---")
    
    # Mock ADK Runner
    from google.adk.runners import InMemoryRunner
    
    class MockEvent:
        def __init__(self, type, content=None, author=None, tool_name=None, tool_args=None):
            self.type = type
            self.content = content
            self.author = author
            self.tool_name = tool_name
            self.tool_args = tool_args
        
        def is_final_response(self):
            return self.type == "final_response"

    class MockStream:
        def __init__(self, message):
            self.message = message
            if "rotate" in message.lower():
                self.events = [
                    MockEvent("tool_call", tool_name="knowledge_search_tool", tool_args={"query": message}),
                    MockEvent("tool_result", tool_name="knowledge_search_tool", content="According to [chunk_abc123]..."),
                    MockEvent("final_response", author="knowledge", content=MagicMock(parts=[MagicMock(text="To rotate a deploy key, use the rotate-key command. According to [chunk_abc123].")]))
                ]
            elif "plan" in message.lower():
                self.events = [
                    MockEvent("final_response", author="account", content=MagicMock(parts=[MagicMock(text="Your current plan tier is pro.")]))
                ]
            else:
                self.events = [
                    MockEvent("final_response", author="smalltalk", content=MagicMock(parts=[MagicMock(text="Hello! How can I help you today?")]))
                ]

        async def __aiter__(self):
            for event in self.events:
                yield event

    async def mock_run_async(self, **kwargs):
        new_message = kwargs.get("new_message", {})
        content = new_message.get("parts", [{}])[0].get("text", "")
        return MockStream(content)

    InMemoryRunner.run_async = mock_run_async

    # Use TestClient to call endpoints
    client = TestClient(app)

    # 2. Create Session
    print("\n--- 2. Creating Session ---")
    resp = client.post("/v1/sessions", json={"user_id": "demo_user", "plan_tier": "pro"})
    session_data = resp.json()
    session_id = session_data["session_id"]
    print(f"Session Created: {session_id}")

    # 3. Chat Turn 1: Knowledge
    print("\n--- 3. Chat Turn 1 (Knowledge) ---")
    print("User: How do I rotate a deploy key?")
    resp = client.post(f"/v1/chat/{session_id}", json={"content": "How do I rotate a deploy key?"})
    chat_data = resp.json()
    print(f"Assistant ({chat_data['routed_to']}): {chat_data['reply']}")
    trace_id_1 = chat_data["trace_id"]

    # 4. Chat Turn 2: Account (Context persistence check)
    print("\n--- 4. Chat Turn 2 (Account) ---")
    print("User: What is my plan tier?")
    resp = client.post(f"/v1/chat/{session_id}", json={"content": "What is my plan tier?"})
    chat_data = resp.json()
    print(f"Assistant ({chat_data['routed_to']}): {chat_data['reply']}")
    trace_id_2 = chat_data["trace_id"]

    # 5. Check Trace
    print("\n--- 5. Checking Trace for Turn 1 ---")
    resp = client.get(f"/v1/traces/{trace_id_1}")
    trace_data = resp.json()
    print(f"Trace ID: {trace_data['trace_id']}")
    print(f"Routed To: {trace_data['routed_to']}")
    print(f"Tool Calls: {trace_data['tool_calls']}")
    print(f"Retrieved Chunks: {trace_data['retrieved_chunk_ids']}")

    print("\n--- Demo Complete ---")

if __name__ == "__main__":
    asyncio.run(run_demo())
