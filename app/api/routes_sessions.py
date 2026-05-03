import uuid
from typing import Literal

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import User, Session as SessionModel
from app.db.session import get_db
from app.srop.state import SessionState

router = APIRouter(tags=["sessions"])


class CreateSessionRequest(BaseModel):
    user_id: str = Field(..., min_length=1)
    plan_tier: Literal["free", "pro", "enterprise"] = "free"


class CreateSessionResponse(BaseModel):
    session_id: str
    user_id: str


@router.post("/sessions", response_model=CreateSessionResponse)
async def create_session(
    body: CreateSessionRequest,
    db: AsyncSession = Depends(get_db),
) -> CreateSessionResponse:
    """
    Create a new session. Upsert the user if not seen before.
    Initialize SessionState and persist to DB.
    """
    # 1. Upsert User
    stmt = sqlite_insert(User).values(
        user_id=body.user_id,
        plan_tier=body.plan_tier
    ).on_conflict_do_update(
        index_elements=["user_id"],
        set_={"plan_tier": body.plan_tier}
    )
    await db.execute(stmt)

    # 2. Initialize SessionState
    initial_state = SessionState(
        user_id=body.user_id,
        plan_tier=body.plan_tier
    )

    # 3. Create Session
    session_id = str(uuid.uuid4())
    new_session = SessionModel(
        session_id=session_id,
        user_id=body.user_id,
        state=initial_state.to_db_dict()
    )
    db.add(new_session)
    
    await db.commit()

    return CreateSessionResponse(
        session_id=session_id,
        user_id=body.user_id
    )
