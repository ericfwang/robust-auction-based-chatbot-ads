"""Shared LLM call cache. Everyone on the team uses this — never call APIs directly."""
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Optional

# Lazy imports so unit tests / type-checkers don't require API SDKs to be installed
_genai = None
_anthropic_client = None


def _get_genai():
    global _genai
    if _genai is None:
        import google.generativeai as genai
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not set")
        genai.configure(api_key=api_key)
        _genai = genai
    return _genai


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise RuntimeError("ANTHROPIC_API_KEY not set")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def _hash(model: str, prompt: str, system: str = "") -> str:
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(b"\x00")
    h.update(system.encode())
    h.update(b"\x00")
    h.update(prompt.encode())
    return h.hexdigest()


def _load_cache_index(cache_path: Path) -> dict[str, str]:
    """Load existing cache into a dict. Skips malformed lines."""
    if not cache_path.exists():
        return {}
    index = {}
    with cache_path.open() as f:
        for line in f:
            try:
                row = json.loads(line)
                index[row["key"]] = row["response"]
            except (json.JSONDecodeError, KeyError):
                continue
    return index


_CACHE_INDEX: dict[str, str] = {}
_CACHE_LOADED_FROM: Optional[Path] = None


def _ensure_loaded(cache_path: Path):
    global _CACHE_INDEX, _CACHE_LOADED_FROM
    if _CACHE_LOADED_FROM != cache_path:
        _CACHE_INDEX = _load_cache_index(cache_path)
        _CACHE_LOADED_FROM = cache_path


def _append_cache(cache_path: Path, key: str, response: str, model: str, prompt_preview: str):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("a") as f:
        f.write(json.dumps({
            "key": key,
            "model": model,
            "prompt_preview": prompt_preview[:200],
            "response": response,
            "ts": time.time(),
        }) + "\n")
    _CACHE_INDEX[key] = response


def cached_llm(
    prompt: str,
    model: str = "gemini-2.5-flash",
    system: str = "",
    cache_path: Optional[Path] = None,
    temperature: float = 0.0,
    max_retries: int = 3,
) -> str:
    """Call an LLM with on-disk caching.

    Cache key = sha256(model || system || prompt). Misses make a fresh call;
    hits return the stored response. Cache file is JSONL at cache_path.
    """
    if cache_path is None:
        from config import LLM_CACHE_PATH
        cache_path = LLM_CACHE_PATH

    cache_path = Path(cache_path)
    _ensure_loaded(cache_path)

    key = _hash(model, prompt, system)
    if key in _CACHE_INDEX:
        return _CACHE_INDEX[key]

    last_err = None
    for attempt in range(max_retries):
        try:
            if model.startswith("gemini"):
                response_text = _call_gemini(model, prompt, system, temperature)
            elif model.startswith("claude"):
                response_text = _call_claude(model, prompt, system, temperature)
            else:
                raise ValueError(f"Unsupported model: {model}")
            _append_cache(cache_path, key, response_text, model, prompt)
            return response_text
        except Exception as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"LLM call failed after {max_retries} retries: {last_err}")


def _call_gemini(model: str, prompt: str, system: str, temperature: float) -> str:
    genai = _get_genai()
    full_prompt = f"{system}\n\n{prompt}" if system else prompt
    m = genai.GenerativeModel(model)
    response = m.generate_content(
        full_prompt,
        generation_config={"temperature": temperature},
    )
    return response.text


def _call_claude(model: str, prompt: str, system: str, temperature: float) -> str:
    client = _get_anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=2048,
        temperature=temperature,
        system=system or "You are a careful evaluator.",
        messages=[{"role": "user", "content": prompt}],
    )
    return response.content[0].text


def cache_stats(cache_path: Optional[Path] = None) -> dict:
    """Quick stats on the cache. Useful for the team to monitor cost."""
    if cache_path is None:
        from config import LLM_CACHE_PATH
        cache_path = LLM_CACHE_PATH
    _ensure_loaded(Path(cache_path))
    return {"n_cached_calls": len(_CACHE_INDEX), "cache_path": str(cache_path)}
