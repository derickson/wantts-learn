import base64
import io

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from . import config

router = APIRouter(prefix="/api")


class GenerateRequest(BaseModel):
    text: str
    language: str = Field(default="English")
    voice: str = Field(default=config.DEFAULT_VOICE, description="Voice character to use (e.g. 'dave', 'claire')")


class GenerateJsonResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    format: str = "wav"
    text: str


@router.post("/generate", summary="Generate speech audio", response_class=StreamingResponse)
async def generate(req: GenerateRequest, request: Request):
    """Generate speech from text and return a WAV file."""
    if req.voice not in config.VOICES:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unknown voice '{req.voice}'. Available voices: {list(config.VOICES.keys())}"},
        )
    mm = request.app.state.model_manager
    await mm.ensure_loaded()
    wav_bytes, sr = await mm.generate(req.text, req.language, req.voice)
    return StreamingResponse(
        io.BytesIO(wav_bytes),
        media_type="audio/wav",
        headers={"Content-Disposition": f'attachment; filename="output.wav"'},
    )


@router.post("/generate/json", summary="Generate speech (base64 JSON)", response_model=GenerateJsonResponse)
async def generate_json(req: GenerateRequest, request: Request):
    """Generate speech from text and return base64-encoded audio with metadata."""
    if req.voice not in config.VOICES:
        return JSONResponse(
            status_code=400,
            content={"detail": f"Unknown voice '{req.voice}'. Available voices: {list(config.VOICES.keys())}"},
        )
    mm = request.app.state.model_manager
    await mm.ensure_loaded()
    wav_bytes, sr = await mm.generate(req.text, req.language, req.voice)
    return GenerateJsonResponse(
        audio_base64=base64.b64encode(wav_bytes).decode(),
        sample_rate=sr,
        text=req.text,
    )


@router.get("/getready", summary="Warm up model")
async def get_ready(request: Request):
    """Load the model and create voice prompt if not already loaded."""
    mm = request.app.state.model_manager
    await mm.get_ready()
    return {"status": "ready", "model_loaded": mm.is_loaded}


@router.get("/status", summary="Service status")
async def status(request: Request):
    """Check whether the model is loaded and GPU memory usage."""
    mm = request.app.state.model_manager
    return {
        "model_loaded": mm.is_loaded,
        "gpu": mm.gpu_memory_info(),
    }


@router.post("/unload", summary="Unload model")
async def unload(request: Request):
    """Manually unload the model and free GPU memory."""
    mm = request.app.state.model_manager
    await mm.unload()
    return {"status": "unloaded", "model_loaded": mm.is_loaded}
