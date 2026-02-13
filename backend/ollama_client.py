from __future__ import annotations

from typing import Optional, Iterator, Dict, Any
import os
import time

from groq import Groq

# Keep these for backward compatibility (main.py reads OLLAMA_MODEL for /health)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")

# Optional CPU thread tuning for Ollama (kept for compatibility; not used by Groq)
_OLLAMA_NUM_THREAD = os.getenv("OLLAMA_NUM_THREAD")  # e.g., "4" or "8"

# Groq client
_GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not _GROQ_API_KEY:
    raise RuntimeError("GROQ_API_KEY not set")

# Let you override Groq model without changing code
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

_CLIENT = Groq(api_key=_GROQ_API_KEY)


def _payload(
    prompt: str,
    *,
    model: str,
    stream: bool,
    temperature: float,
    num_predict: int,
) -> Dict[str, Any]:
    """
    Preserved for compatibility with the original file structure.
    Not used for Groq, but kept so you 'don't change logic/layout' in the module.
    """
    options: Dict[str, Any] = {
        "temperature": temperature,
        "num_predict": num_predict,
        "num_ctx": 4096,
    }
    if _OLLAMA_NUM_THREAD and _OLLAMA_NUM_THREAD.isdigit():
        options["num_thread"] = int(_OLLAMA_NUM_THREAD)

    return {
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "keep_alive": "10m",
        "options": options,
    }


def generate(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout_s: int = 180,
    num_predict: int = 350,  # same default
    temperature: float = 0.4,
) -> str:
    # Keep same "model or OLLAMA_MODEL" logic; map it to Groq's model internally
    _ = model or OLLAMA_MODEL  # preserved behavior
    groq_model = GROQ_MODEL

    completion = _CLIENT.chat.completions.create(
        model=groq_model,
        messages=[
            # Keep this minimal; your real instructions are already inside `prompt`
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=num_predict,
        stream=False,
    )

    return (completion.choices[0].message.content or "").strip()


def generate_stream(
    prompt: str,
    *,
    model: Optional[str] = None,
    timeout_s: int = 180,
    num_predict: int = 350,  # same default
    temperature: float = 0.4,
    # SAME buffering knobs as your original (smooth stream)
    flush_chars: int = 200,
    flush_interval_s: float = 0.25,
) -> Iterator[str]:
    """
    Streaming generator. Yields partial text chunks.
    Preserves your buffering behavior: emits chunks every N chars or every T seconds.
    """
    _ = model or OLLAMA_MODEL  # preserved behavior
    groq_model = GROQ_MODEL

    stream = _CLIENT.chat.completions.create(
        model=groq_model,
        messages=[
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=num_predict,
        stream=True,
    )

    buffer = ""
    last_flush = time.time()

    for event in stream:
        delta = event.choices[0].delta.content or ""
        if delta:
            buffer += delta

        now = time.time()
        if buffer and (len(buffer) >= flush_chars or (now - last_flush) >= flush_interval_s):
            yield buffer
            buffer = ""
            last_flush = now

    if buffer:
        yield buffer