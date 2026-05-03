"""
Test fixtures.

Key fixtures:
- `client`: async test client with in-memory SQLite DB
- `mock_adk`: patches the ADK root agent so tests don't hit the real LLM
- `seeded_db`: DB with a test user and session pre-created
"""
import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.db.models import Base
from app.db.session import get_db
from app.main import app


TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DATABASE_URL, echo=False)
TestSessionLocal = async_sessionmaker(test_engine, expire_on_commit=False)


@pytest_asyncio.fixture(autouse=True)
async def setup_test_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture
async def db() -> AsyncSession:
    async with TestSessionLocal() as session:
        yield session


@pytest_asyncio.fixture
async def client(db):
    """Async test client with DB overridden to in-memory SQLite."""
    app.dependency_overrides[get_db] = lambda: db
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c
    app.dependency_overrides.clear()


from unittest.mock import MagicMock
from app.srop.pipeline import PipelineResult

@pytest.fixture
def mock_adk(monkeypatch):
    """
    Patch the ADK pipeline so tests don't call the real LLM.
    We'll mock the 'run' function in pipeline.py for simplicity in this integration test.
    """
    async def mock_run(session_id, message, db):
        if "rotate" in message.lower():
            return PipelineResult(
                content="To rotate a deploy key, use the rotate-key command. According to [chunk_abc123].",
                routed_to="knowledge",
                trace_id="test-trace-001",
            )
        elif "plan" in message.lower():
            # Simulate agent knowing the plan_tier from context
            return PipelineResult(
                content="Your current plan tier is pro.",
                routed_to="account",
                trace_id="test-trace-002",
            )
        return PipelineResult(
            content="Hello! How can I help you?",
            routed_to="smalltalk",
            trace_id="test-trace-003",
        )

    # We also need to mock the trace record in DB if we mock pipeline.run
    # Or better, we mock the ADK Runner so pipeline.run actually executes.
    # Let's try mocking the ADK Runner.run_async instead.
    
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
            self.events = []
            if "rotate" in message.lower():
                self.events = [
                    MockEvent("tool_call", tool_name="knowledge_search_tool", tool_args={"query": message}),
                    MockEvent("tool_result", tool_name="knowledge_search_tool", content="According to [chunk_abc123]..."),
                    MockEvent("final_response", author="knowledge", content=MagicMock(parts=[MagicMock(text="To rotate a deploy key, use the command. According to [chunk_abc123].")]))
                ]
            elif "plan" in message.lower():
                self.events = [
                    MockEvent("final_response", author="account", content=MagicMock(parts=[MagicMock(text="Your plan tier is pro.")]))
                ]
            else:
                self.events = [
                    MockEvent("final_response", author="smalltalk", content=MagicMock(parts=[MagicMock(text="Hello!")]))
                ]

        async def __aiter__(self):
            for event in self.events:
                yield event

    async def mock_run_async(self, **kwargs):
        new_message = kwargs.get("new_message", {})
        content = new_message.get("parts", [{}])[0].get("text", "")
        return MockStream(content)

    monkeypatch.setattr("google.adk.runners.InMemoryRunner.run_async", mock_run_async)
