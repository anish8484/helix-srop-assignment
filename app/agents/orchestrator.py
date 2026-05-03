from google.adk.agents import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from app.agents.knowledge import knowledge_agent
from app.agents.account import account_agent
from app.settings import settings

ROOT_INSTRUCTION = """
You are the Helix Support Concierge — a routing agent.
Call the correct specialist tool based on the user's intent.

Intent → tool:
- HOW to do something, WHAT something is, docs/feature questions → knowledge_agent
- Their account, builds, status, usage → account_agent
- Greetings or off-topic → respond directly, no tool call

Always call a tool when intent matches. Never answer knowledge or account questions yourself.
User context will be in the system message — use it.
"""

knowledge_tool = AgentTool(agent=knowledge_agent)
account_tool   = AgentTool(agent=account_agent)

root_agent = LlmAgent(
    name="srop_root",
    model=settings.adk_model,
    instruction=ROOT_INSTRUCTION,
    tools=[knowledge_tool, account_tool],
)
