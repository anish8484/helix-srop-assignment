from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from app.api import routes_sessions, routes_chat, routes_traces
from app.db.session import init_db
from app.obs.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()
    await init_db()
    yield


app = FastAPI(title="Helix SROP", version="0.1.0", lifespan=lifespan)

app.include_router(routes_sessions.router, prefix="/v1")
app.include_router(routes_chat.router, prefix="/v1")
app.include_router(routes_traces.router, prefix="/v1")


@app.get("/healthz")
async def healthz() -> dict:
    return {"status": "ok"}


from fastapi.responses import HTMLResponse
from pathlib import Path

@app.get("/", response_class=HTMLResponse)
async def get_ui():
    ui_path = Path(__file__).parent.parent / "chat_ui.html"
    return ui_path.read_text()

from app.api.errors import HelixError, helix_error_handler

# Register exception handlers for HelixError subclasses
app.add_exception_handler(HelixError, helix_error_handler)
