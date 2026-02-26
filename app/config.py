import torch

MODEL_NAME = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
VOICES = {
    "dave": {
        "ref_audio": "./DaveSample.m4a",
        "ref_text": (
            "Hey everybody this is Dave Erickson and I wanted to record a sample of my voice. "
            "This is so that I can clone my voice. It's important that i have the ability to "
            "create a digital twin of myself. I am expressive and I speak in an uh eloquent "
            "and uh maybe a little bit  informal tone. Thanks"
        ),
    },
    "claire": {
        "ref_audio": "./claire.mp3",
        "ref_text": (
            "Wow, I am really embarrassed! I've never been to such a wild party and I think, "
            "... well, maybe I could get used to this kind of luxury treatment."
        ),
    },
}
DEFAULT_VOICE = "dave"
IDLE_TIMEOUT_SECONDS = 900  # 15 minutes
DEVICE = "cuda:0"
DTYPE = torch.bfloat16

try:
    import flash_attn  # noqa: F401
    ATTN_IMPL = "flash_attention_2"
except ImportError:
    ATTN_IMPL = "eager"
