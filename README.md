# Voice Clone TTS

A FastAPI service for text-to-speech using [Qwen3-TTS-12Hz-1.7B-Base](https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base) with voice cloning. The model clones a reference voice from an audio sample and generates speech in that voice.

## Features

- **Multiple voice characters** — ships with Dave and Claire, selectable via API parameter
- **Voice cloning** from reference audio samples
- **Automatic model lifecycle** — loads on demand, unloads after 15 minutes idle to free GPU memory
- **REST API** with Swagger UI (`/docs`) and ReDoc (`/redoc`)
- **Web UI** — landing page with voice selector, text input, audio playback, and download

## Requirements

- NVIDIA GPU with ~5 GB VRAM
- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (for Docker)

## Quick Start (Docker)

```bash
docker compose up --build
```

The first build takes 10-15 minutes (flash-attn compilation). On first run the HuggingFace model (~3.4 GB) downloads automatically — subsequent starts are fast since the model is cached in a named volume.

- **Web UI**: http://localhost:8335
- **Swagger docs**: http://localhost:8335/docs

### Verify it's working

```bash
curl http://localhost:8335/api/status

# Generate with default voice (Dave)
curl -X POST http://localhost:8335/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world"}' -o test.wav

# Generate with Claire's voice
curl -X POST http://localhost:8335/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "claire"}' -o claire.wav
```

## Make Commands

A `Makefile` is provided for common operations:

| Command | Description |
|---------|-------------|
| `make start` | Start the service in the background |
| `make stop` | Stop the service |
| `make restart` | Restart without rebuilding |
| `make redeploy` | Stop, rebuild, and start (full redeploy) |
| `make build` | Build the Docker image without starting |
| `make status` | Show container state and model/GPU status |
| `make logs` | Tail container logs |
| `make shell` | Open a bash shell in the running container |

## Local Development (without Docker)

Requires Python 3.12+ and CUDA-compatible PyTorch.

```bash
python -m venv venv
source venv/bin/activate

pip install qwen-tts soundfile torch
pip install fastapi 'uvicorn[standard]' python-multipart

# Optional: faster inference with flash attention
pip install flash-attn --no-build-isolation
```

```bash
source venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8335
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/generate` | Generate speech, returns WAV file |
| `POST` | `/api/generate/json` | Generate speech, returns base64 JSON |
| `GET` | `/api/getready` | Warm up / preload the model |
| `GET` | `/api/status` | Check if model is loaded + GPU memory |
| `POST` | `/api/unload` | Manually unload model and free GPU |

### Generate speech

```bash
curl -X POST http://localhost:8335/api/generate \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "English", "voice": "dave"}' \
  -o output.wav
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `text` | string | *(required)* | Text to synthesize |
| `language` | string | `"English"` | Language of the text |
| `voice` | string | `"dave"` | Voice character (`dave`, `claire`) |

### Generate speech (JSON response)

```bash
curl -X POST http://localhost:8335/api/generate/json \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "voice": "claire"}'
```

Returns:
```json
{
  "audio_base64": "...",
  "sample_rate": 24000,
  "format": "wav",
  "text": "Hello world"
}
```

### Check status

```bash
curl http://localhost:8335/api/status
```

## Adding New Voices

1. Place a reference audio file (`.mp3`, `.m4a`, `.wav`) in the project root.
2. Add an entry to the `VOICES` dict in `app/config.py`:
   ```python
   "newname": {
       "ref_audio": "./newname.wav",
       "ref_text": "Exact transcript of the reference audio...",
   },
   ```
3. Mount the file in `docker-compose.yml`:
   ```yaml
   volumes:
     - ./newname.wav:/app/newname.wav
   ```
4. Restart the service. The new voice is available via `"voice": "newname"` in API requests.

## Project Structure

```
app/
├── __init__.py
├── config.py             # Model name, voice definitions, timeouts, device
├── model_manager.py      # Async model lifecycle: load, unload, idle timer
├── routes.py             # API endpoint handlers
├── main.py               # FastAPI app, lifespan, landing page
└── templates/
    └── index.html        # Web UI
Dockerfile
docker-compose.yml
DaveSample.m4a            # Dave's reference voice sample
claire.mp3                # Claire's reference voice sample
```

## Configuration

Edit `app/config.py` to change:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `Qwen/Qwen3-TTS-12Hz-1.7B-Base` | HuggingFace model ID |
| `VOICES` | `{"dave": ..., "claire": ...}` | Voice character definitions (ref audio + transcript) |
| `DEFAULT_VOICE` | `"dave"` | Default voice when none specified |
| `IDLE_TIMEOUT_SECONDS` | `900` | Seconds before auto-unloading (15 min) |
| `DEVICE` | `cuda:0` | Torch device |
| `DTYPE` | `torch.bfloat16` | Model precision |
