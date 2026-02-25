import asyncio
import io
import logging

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

from . import config

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.model = None
        self.voice_prompt = None
        self.sample_rate: int | None = None
        self._lock = asyncio.Lock()
        self._idle_timer: asyncio.Task | None = None

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def _load_sync(self):
        logger.info("Loading model %s ...", config.MODEL_NAME)
        self.model = Qwen3TTSModel.from_pretrained(
            config.MODEL_NAME,
            device_map=config.DEVICE,
            dtype=config.DTYPE,
            attn_implementation=config.ATTN_IMPL,
        )
        logger.info("Creating voice clone prompt ...")
        self.voice_prompt = self.model.create_voice_clone_prompt(
            ref_audio=config.REF_AUDIO_PATH,
            ref_text=config.REF_TEXT,
        )
        logger.info("Model and voice prompt ready.")

    def _unload_sync(self):
        logger.info("Unloading model ...")
        del self.model
        self.model = None
        self.voice_prompt = None
        self.sample_rate = None
        torch.cuda.empty_cache()
        logger.info("Model unloaded, GPU memory freed.")

    async def load(self):
        async with self._lock:
            if self.model is not None:
                return
            await asyncio.to_thread(self._load_sync)
            self._reset_idle_timer()

    async def unload(self):
        async with self._lock:
            if self.model is None:
                return
            if self._idle_timer and not self._idle_timer.done():
                self._idle_timer.cancel()
                self._idle_timer = None
            await asyncio.to_thread(self._unload_sync)

    def _reset_idle_timer(self):
        if self._idle_timer and not self._idle_timer.done():
            self._idle_timer.cancel()
        self._idle_timer = asyncio.create_task(self._idle_countdown())

    async def _idle_countdown(self):
        try:
            await asyncio.sleep(config.IDLE_TIMEOUT_SECONDS)
            logger.info("Idle timeout reached, unloading model ...")
            async with self._lock:
                if self.model is not None:
                    await asyncio.to_thread(self._unload_sync)
        except asyncio.CancelledError:
            pass

    async def ensure_loaded(self):
        if self.model is None:
            await self.load()
        else:
            self._reset_idle_timer()

    async def get_ready(self):
        await self.ensure_loaded()

    def _generate_sync(self, text: str, language: str) -> tuple[bytes, int]:
        wavs, sr = self.model.generate_voice_clone(
            text=[text],
            language=[language],
            voice_clone_prompt=self.voice_prompt,
        )
        buf = io.BytesIO()
        sf.write(buf, wavs[0], sr, format="WAV")
        buf.seek(0)
        return buf.read(), sr

    async def generate(self, text: str, language: str = "English") -> tuple[bytes, int]:
        async with self._lock:
            if self.model is None:
                raise RuntimeError("Model is not loaded")
            wav_bytes, sr = await asyncio.to_thread(self._generate_sync, text, language)
            self.sample_rate = sr
            self._reset_idle_timer()
            return wav_bytes, sr

    def gpu_memory_info(self) -> dict:
        if not torch.cuda.is_available():
            return {"available": False}
        allocated = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        total = torch.cuda.get_device_properties(0).total_memory / 1e9
        return {
            "allocated_gb": round(allocated, 2),
            "reserved_gb": round(reserved, 2),
            "total_gb": round(total, 2),
        }
