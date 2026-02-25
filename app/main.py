import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from .model_manager import ModelManager
from .routes import router as api_router

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).parent / "templates"


@asynccontextmanager
async def lifespan(app: FastAPI):
    mm = ModelManager()
    app.state.model_manager = mm
    # Auto-load in background so uvicorn finishes binding the port first
    asyncio.create_task(mm.get_ready())
    logger.info("Startup complete — model loading in background.")
    yield
    logger.info("Shutting down — unloading model ...")
    await mm.unload()


app = FastAPI(
    title="Voice Clone TTS",
    description="Text-to-speech API using Qwen3-TTS with Dave's cloned voice. "
                "The model loads on demand and unloads after 15 minutes of inactivity.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount("/static", StaticFiles(directory=str(TEMPLATES_DIR)), name="static")
app.include_router(api_router)


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def landing_page(request: Request):
    return HTMLResponse((TEMPLATES_DIR / "index.html").read_text())
