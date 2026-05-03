import asyncio
import time
import uuid
from dataclasses import dataclass
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.agents.orchestrator import root_agent
from app.db.models import AgentTrace, Message, Session as SessionModel
from app.srop.state import SessionState
from app.settings import settings
from app.api.errors import UpstreamTimeoutError, SessionNotFoundError
from google.adk.runners import InMemoryRunner

@dataclass
class PipelineResult:
    content: str
    routed_to: str
    trace_id: str

async def run(session_id: str, user_message: str, db: AsyncSession) -> PipelineResult:
    # 1. Load session from DB
    result = await db.execute(
        select(SessionModel).where(SessionModel.session_id == session_id)
    )
    db_session = result.scalar_one_or_none()
    if not db_session:
        raise SessionNotFoundError(f"Session {session_id} not found")

    # 2. Re-hydrate SessionState
    state = SessionState.from_db_dict(db_session.state)
    state.turn_count += 1
    
    # 3. Dynamic instruction with state
    instruction_with_context = f"""
    Current session context:
    - user_id: {state.user_id}
    - plan_tier: {state.plan_tier}
    - last_agent: {state.last_agent or 'none'}
    - turn_count: {state.turn_count}
    """
    
    # 4. Prepare runner
    runner = InMemoryRunner(agent=root_agent)
    trace_id = str(uuid.uuid4())
    start_time = time.time()
    
    routed_to = "smalltalk"
    tool_calls = []
    retrieved_chunk_ids = []
    final_content = ""

    # 5. Run ADK with timeout
    try:
        response_stream = await asyncio.wait_for(
            runner.run_async(
                user_id=state.user_id,
                session_id=session_id,
                new_message={"role": "user", "parts": [{"text": user_message}]},
            ),
            timeout=settings.llm_timeout_seconds
        )
        
        async for event in response_stream:
            # Capture tool calls
            if event.type == "tool_call":
                tool_calls.append({
                    "tool_name": event.tool_name,
                    "args": event.tool_args,
                })
            
            if event.type == "tool_result":
                for call in tool_calls:
                    if call["tool_name"] == event.tool_name:
                        call["result"] = event.content
                        if event.tool_name == "knowledge_search_tool":
                            ids = re.findall(r'\[(chunk_[a-f0-9]+)\]', str(event.content))
                            retrieved_chunk_ids.extend(ids)

            if event.is_final_response():
                routed_to = event.author or "smalltalk"
                final_content = event.content.parts[0].text

    except asyncio.TimeoutError:
        raise UpstreamTimeoutError(f"LLM did not respond within {settings.llm_timeout_seconds}s")
    except Exception as e:
        print(f"Pipeline error: {e}")
        raise e

    latency_ms = int((time.time() - start_time) * 1000)

    # 6. Update state
    state.last_agent = routed_to # type: ignore
    
    # 7. Persist Trace
    trace = AgentTrace(
        trace_id=trace_id,
        session_id=session_id,
        routed_to=routed_to,
        tool_calls=tool_calls,
        retrieved_chunk_ids=list(set(retrieved_chunk_ids)),
        latency_ms=latency_ms
    )
    db.add(trace)

    # 8. Persist Messages
    user_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="user",
        content=user_message,
        trace_id=trace_id
    )
    assistant_msg = Message(
        message_id=str(uuid.uuid4()),
        session_id=session_id,
        role="assistant",
        content=final_content,
        trace_id=trace_id
    )
    db.add(user_msg)
    db.add(assistant_msg)

    # 9. Persist Updated Session State
    await db.execute(
        update(SessionModel)
        .where(SessionModel.session_id == session_id)
        .values(state=state.to_db_dict(), updated_at=db_session.updated_at) # SQLAlchemy handles onupdate
    )
    
    await db.commit()

    return PipelineResult(
        content=final_content,
        routed_to=routed_to,
        trace_id=trace_id
    )

import re # needed for chunk ID extraction
