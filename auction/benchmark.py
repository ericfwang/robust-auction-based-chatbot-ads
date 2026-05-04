"""Benchmark loader.

Default benchmark:
- **Prompts:** 50 hand-written chatbot prompts with paired clean answers (in seed_prompts.jsonl).
  40 commercial across 10 Electronics sub-categories + 10 sensitive prompts (some plausibly
  retrievable, to stress-test the welfare reserve). Pre-written so no LLM calls are needed
  to seed the benchmark.
- **Products:** 300 real Amazon Reviews 2023 Electronics products (cached on first download).

Optional LLM expansion functions are available if you want to grow the prompt set.
"""
import json
from pathlib import Path
from typing import Optional

from .cached_llm import cached_llm
from .types import Product

# ---------- Loaders ----------

def _seed_dir() -> Path:
    from config import DATA_DIR
    return Path(DATA_DIR)


def load_seed_prompts(path: Optional[Path] = None) -> list[dict]:
    """Load the hand-curated benchmark prompts.

    Returns a list of dicts: {id, category, is_sensitive, prompt, clean_answer}.
    The clean_answer field is pre-written so no LLM calls are needed to use the benchmark.
    """
    p = Path(path) if path else _seed_dir() / "seed_prompts.jsonl"
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def load_amazon_products_cached(
    n: int = 10000,
    cache_filename: Optional[str] = None,
    force_refresh: bool = False,
) -> list[Product]:
    """Load n Amazon Electronics products, caching to disk so subsequent calls are instant.

    First call: downloads ~2x n raw products (since some get filtered for missing data),
    keeps the first n with valid metadata, saves to data/<cache_filename>.
    Subsequent calls: reads from the saved JSONL.

    Default n=10000 — matches the original proposal claim. First-run download takes
    5-10 minutes; cached afterwards. Memory footprint of the embedding index ≈ 15 MB.

    Set force_refresh=True to re-download (use this if you change the dataset).
    """
    if cache_filename is None:
        cache_filename = f"amazon_electronics_{n}.jsonl"
    cache_path = _seed_dir() / cache_filename
    if cache_path.exists() and not force_refresh:
        return _load_products_from_jsonl(cache_path)

    # Lazy import — only required on first download
    from .data_pipeline import load_amazon_products
    raw = load_amazon_products(n=n * 2)  # over-fetch since some may be filtered
    products = raw[:n]
    save_products(products, cache_path)
    return products


def _load_products_from_jsonl(path: Path) -> list[Product]:
    products = []
    with path.open() as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            products.append(Product(
                id=row["id"],
                title=row["title"],
                description=row["description"],
                price=float(row["price"]),
                category=row.get("category", ""),
                features=row.get("features", []),
                image_url=row.get("image_url"),
            ))
    return products


def load_seed_products(path: Optional[Path] = None) -> list[Product]:
    """Fallback: load 30 hand-curated products (only used if Amazon download fails).

    Kept for offline development and as a back-pocket if Amazon Reviews 2023 is briefly
    unreachable from Colab. The primary product feed is load_amazon_products_cached().
    """
    p = Path(path) if path else _seed_dir() / "seed_products.jsonl"
    if not p.exists():
        return []
    return _load_products_from_jsonl(p)


# ---------- Clean answer generation (only needed if you LLM-expand the prompt set) ----------

CLEAN_ANSWER_SYSTEM = """You are a helpful chatbot assistant. Answer the user's question
concisely (3-5 sentences). If the user is asking about products, give a balanced recommendation
that names categories or specific examples without naming specific brands. Be specific and useful, not vague."""


def generate_clean_answer(prompt: str) -> str:
    """One LLM call per prompt to produce the canonical 'no-ad' answer. Cached.

    Only needed for prompts you've added via LLM expansion — the seed prompts already
    have hand-written clean_answer fields.
    """
    return cached_llm(prompt, system=CLEAN_ANSWER_SYSTEM, temperature=0.3).strip()


def fill_missing_clean_answers(prompts: list[dict], progress: bool = True) -> list[dict]:
    """Generate clean answers only for prompts missing one. Pre-written ones are kept as-is."""
    iterator = prompts
    if progress:
        try:
            from tqdm.auto import tqdm
            iterator = tqdm(prompts, desc="filling clean answers")
        except ImportError:
            pass
    out = []
    for row in iterator:
        if row.get("clean_answer"):
            out.append(row)
        else:
            out.append({**row, "clean_answer": generate_clean_answer(row["prompt"])})
    return out


# ---------- Optional: expand the benchmark via LLM ----------

PROMPT_EXPANDER_SYSTEM = """You generate realistic chatbot prompts that a user might type into
ChatGPT or Claude when seeking a product recommendation.

Style: conversational (not search-query style), 1-2 sentences, natural phrasing. Include
specific context like budget, intended use, or a constraint.

Output format: one prompt per line, no numbering, no quotes, no extra commentary."""


def generate_prompts_llm(category: str, n: int = 10, examples: Optional[list[str]] = None) -> list[dict]:
    """Generate n new chatbot-style prompts for a category. Each prompt gets a fresh id.

    examples: optional list of seed prompts to anchor the style.
    """
    user_msg = f"Generate {n} chatbot prompts for the **{category}** product category.\n"
    if examples:
        user_msg += "\nExamples of the style we want:\n"
        for ex in examples[:3]:
            user_msg += f"- {ex}\n"
    response = cached_llm(user_msg, system=PROMPT_EXPANDER_SYSTEM, temperature=0.7)
    lines = [line.strip().lstrip("-* ").strip() for line in response.splitlines()]
    prompts = [line for line in lines if line and len(line.split()) >= 5][:n]
    return [
        {"id": f"gen_{category}_{i:03d}", "category": category, "is_sensitive": False, "prompt": p}
        for i, p in enumerate(prompts)
    ]


PRODUCT_EXPANDER_SYSTEM = """You generate realistic product entries for an e-commerce catalog.

Each entry should have a real-sounding (or actual) product name, a 1-2 sentence description,
a realistic USD price, and 3-4 short feature bullets.

Output format: one product per line as compact JSON with keys: title, description, price (number),
features (list of 3-4 strings). No numbering, no extra commentary."""


def generate_products_llm(category: str, n: int = 10, examples: Optional[list[Product]] = None) -> list[Product]:
    """Generate n new products for a category. Each product gets a fresh id."""
    user_msg = f"Generate {n} product entries for the **{category}** category.\n"
    if examples:
        user_msg += "\nExamples:\n"
        for p in examples[:2]:
            user_msg += f'- {{"title": "{p.title}", "description": "{p.description[:100]}", "price": {p.price}, "features": {p.features[:3]}}}\n'
    response = cached_llm(user_msg, system=PRODUCT_EXPANDER_SYSTEM, temperature=0.7)
    products = []
    for i, line in enumerate(response.splitlines()):
        line = line.strip().lstrip("-* ").strip()
        if not line or not line.startswith("{"):
            continue
        try:
            row = json.loads(line)
            products.append(Product(
                id=f"gen_{category}_{i:03d}",
                title=row["title"],
                description=row["description"],
                price=float(row["price"]),
                category=category,
                features=row.get("features", [])[:4],
                image_url=None,
            ))
        except (json.JSONDecodeError, KeyError, ValueError, TypeError):
            continue
        if len(products) >= n:
            break
    return products


# ---------- Diagnostic: is the prompt benchmark a good fit for the loaded product feed? ----------

def diagnose_retrieval_quality(
    prompts: list[dict],
    products: list,
    product_index,
    k: int = 5,
    good_threshold: float = 0.55,
    poor_threshold: float = 0.40,
):
    """Compute per-prompt retrieval quality stats. Returns a pandas DataFrame.

    For each prompt, reports the top-1 and mean-top-k similarity to the product feed.
    Flags prompts as 'good', 'marginal', or 'poor' so you can quickly spot which prompts
    will produce uninteresting auctions because retrieval finds nothing relevant.

    What to do with the results:
        - All 'good' or 'marginal': run the benchmark.
        - A handful 'poor' (<5): tolerable; you can drop them or accept that the welfare
          reserve will gate them anyway.
        - Many 'poor' (>10): increase n in load_amazon_products_cached(n=1000) and
          force_refresh=True, OR regenerate the affected prompts using
          regenerate_prompt_for_category(category, sample_products).
    """
    import numpy as np
    import pandas as pd
    from .data_pipeline import _get_embedder

    embedder = _get_embedder()
    texts = [p['prompt'] for p in prompts]
    q = embedder.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    sims = q @ product_index.T  # (n_prompts, n_products), cosine since normalized
    top_idx = np.argsort(-sims, axis=1)[:, :k]
    rows = []
    for i, p in enumerate(prompts):
        top1_sim = float(sims[i, top_idx[i, 0]])
        topk_mean = float(sims[i, top_idx[i]].mean())
        top1_product = products[top_idx[i, 0]]
        if top1_sim >= good_threshold:
            quality = 'good'
        elif top1_sim >= poor_threshold:
            quality = 'marginal'
        else:
            quality = 'poor'
        rows.append({
            'id': p['id'],
            'category': p['category'],
            'is_sens': p['is_sensitive'],
            'top1_sim': round(top1_sim, 3),
            f'top{k}_mean_sim': round(topk_mean, 3),
            'quality': quality,
            'prompt': p['prompt'][:60],
            'top1_match': top1_product.title[:55],
        })
    df = pd.DataFrame(rows)
    return df


def diagnose_summary(diag_df) -> dict:
    """One-line summary of diagnostic results. Returns dict with counts."""
    counts = diag_df['quality'].value_counts().to_dict()
    return {
        'good': counts.get('good', 0),
        'marginal': counts.get('marginal', 0),
        'poor': counts.get('poor', 0),
        'total': len(diag_df),
    }


# ---------- Regenerate a prompt anchored to actual products in the feed ----------

PROMPT_REGEN_SYSTEM = """You generate a single realistic chatbot prompt that a user might type
into ChatGPT or Claude when looking for a product recommendation.

You will be shown a list of products that DO exist in the catalog. Generate a prompt that one or
more of these products would be a good answer to. The prompt should be conversational (not search
query style), 1-2 sentences, and include specific context like budget or use case.

Output format: just the prompt text. No preamble, no quotes, no explanations."""


def regenerate_prompt_for_category(
    category: str,
    sample_products: list,
    n: int = 1,
) -> list[dict]:
    """Generate n new prompts anchored to the actual products available in the feed.

    Use this when diagnose_retrieval_quality flags a prompt as 'poor' and you want a
    replacement that will retrieve actual matching products from the loaded Amazon sample.

    Each call costs one cached Gemini Flash call.
    """
    out = []
    for i in range(n):
        product_summary = "\n".join(
            f"- {p.title[:80]} (${p.price:.2f})" for p in sample_products[:8]
        )
        user_msg = (
            f"Category: {category}\n\n"
            f"Products available in this category:\n{product_summary}\n\n"
            f"Generate one chatbot-style prompt for this category."
        )
        new_prompt = cached_llm(user_msg, system=PROMPT_REGEN_SYSTEM, temperature=0.7).strip()
        # Also generate a clean answer for it
        clean = generate_clean_answer(new_prompt)
        out.append({
            "id": f"regen_{category}_{i:03d}",
            "category": category,
            "is_sensitive": False,
            "prompt": new_prompt,
            "clean_answer": clean,
        })
    return out


# ---------- Save ----------

def save_prompts(prompts: list[dict], path: Path):
    """Write prompts to JSONL. Useful when you've expanded via LLM and want to commit the result."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for row in prompts:
            f.write(json.dumps(row) + "\n")


def save_products(products: list[Product], path: Path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for p in products:
            f.write(json.dumps(p.__dict__) + "\n")
