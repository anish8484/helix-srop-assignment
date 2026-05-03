from google.adk.agents import LlmAgent
from app.settings import settings

async def get_recent_builds(user_id: str, limit: int = 3) -> str:
    """
    Look up the most recent builds for a user.
    """
    # Mock data for demonstration
    builds = [
        {"id": "b_789", "status": "failed", "branch": "main", "created_at": "2024-03-20T10:00:00Z"},
        {"id": "b_456", "status": "failed", "branch": "feat-x", "created_at": "2024-03-19T15:30:00Z"},
        {"id": "b_123", "status": "success", "branch": "main", "created_at": "2024-03-18T09:00:00Z"},
    ]
    
    result = f"Recent {limit} builds for user {user_id}:\n"
    for b in builds[:limit]:
        result += f"- {b['id']}: {b['status']} on {b['branch']} ({b['created_at']})\n"
    return result

async def get_account_status(user_id: str) -> str:
    """
    Get the current status and plan tier of a user's account.
    """
    # Mock data for demonstration
    return f"Account status for {user_id}: Active. Plan Tier: Enterprise. Billing cycle: Monthly."

account_agent = LlmAgent(
    name="account",
    model=settings.adk_model,
    instruction="""
You are the Helix Account Specialist.
Your job is to help users with information about their account, builds, and usage.

Rules:
1. Use 'get_recent_builds' to show build history.
2. Use 'get_account_status' for plan and status questions.
3. If you don't have enough information (like a user_id), ask the user for it or check the provided context.
4. Be professional and helpful.
""",
    tools=[get_recent_builds, get_account_status],
)
