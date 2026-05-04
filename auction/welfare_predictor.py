"""Non-LLM welfare predictor.

Two signals combined: embedding distance to sensitive-topic anchor prompts, plus a
keyword rule stack. Output is a 0–1 estimated welfare loss (1 = ad would do max harm).

This runs INSIDE the auction at reserve-setting time. Production ad systems can't afford
an LLM call per auction — embedding lookups + regex are the realistic alternative. The
proposal frames this as a realism upgrade, not a compromise.
"""
import re
from typing import Optional

import numpy as np


_anchor_emb_cache: Optional[np.ndarray] = None
_embedder_cache = None


def _get_embedder():
    """Process-wide cached SentenceTransformer. One load per session — avoids the noisy
    transformers 'UNEXPECTED' load report repeating on every welfare-predictor call."""
    global _embedder_cache
    if _embedder_cache is None:
        from sentence_transformers import SentenceTransformer
        from config import EMBED_MODEL
        _embedder_cache = SentenceTransformer(EMBED_MODEL)
    return _embedder_cache


def _get_anchor_embeddings() -> np.ndarray:
    """L2-normalized embeddings of the sensitive anchor prompts. Cached."""
    global _anchor_emb_cache
    if _anchor_emb_cache is None:
        from config import SENSITIVE_ANCHOR_PROMPTS
        emb = _get_embedder().encode(SENSITIVE_ANCHOR_PROMPTS, convert_to_numpy=True)
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
        _anchor_emb_cache = emb / norms
    return _anchor_emb_cache


def _keyword_hit(prompt: str) -> float:
    from config import SENSITIVE_KEYWORDS
    p = prompt.lower()
    n_hits = sum(1 for kw in SENSITIVE_KEYWORDS if re.search(rf"\b{re.escape(kw)}\b", p))
    if n_hits == 0:
        return 0.0
    if n_hits == 1:
        return 0.7
    return 1.0


def _anchor_similarity(prompt: str) -> float:
    """Max cosine similarity to any sensitive anchor, mapped to [0, 1]."""
    anchors = _get_anchor_embeddings()
    q = _get_embedder().encode([prompt], convert_to_numpy=True)[0]
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = anchors @ q
    max_sim = float(sims.max())
    return max(0.0, max_sim)


def predict_welfare_loss(prompt: str) -> float:
    """Returns estimated welfare loss in [0, 1]. Higher = ad would degrade answer more.

    Combines:
        1. Max cosine similarity to a curated set of sensitive-topic anchor prompts,
           with the empirical noise floor (~0.30) subtracted to prevent false positives
           on commercial prompts that share generic English structure with the anchors.
        2. Keyword rule stack on the prompt text.

    The two signals are combined by max() — either raises an alarm independently.
    """
    from config import ANCHOR_NOISE_FLOOR
    rule = _keyword_hit(prompt)
    anchor = _anchor_similarity(prompt)
    anchor_signal = max(0.0, anchor - ANCHOR_NOISE_FLOOR)
    return float(min(1.0, max(rule, anchor_signal)))


def predict_welfare_loss_batch(prompts: list[str]) -> np.ndarray:
    """Vectorized version for evaluating many prompts at once."""
    from config import ANCHOR_NOISE_FLOOR
    anchors = _get_anchor_embeddings()
    q = _get_embedder().encode(prompts, convert_to_numpy=True)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    anchor_sim = (q @ anchors.T).max(axis=1)  # (n_prompts,)
    anchor_signal = np.clip(anchor_sim - ANCHOR_NOISE_FLOOR, 0.0, 1.0)
    rule = np.array([_keyword_hit(p) for p in prompts])
    return np.minimum(1.0, np.maximum(rule, anchor_signal))
