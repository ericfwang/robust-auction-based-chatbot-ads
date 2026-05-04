"""Data pipeline: WildChat filter, Amazon products, embed, retrieve."""
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from .types import Product
from .llm_components import has_commercial_intent

# Lazy import for sentence_transformers (heavy)
_embedder = None


def _get_embedder():
    global _embedder
    if _embedder is None:
        from sentence_transformers import SentenceTransformer
        from config import EMBED_MODEL
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


# ---------- WildChat: load + filter for commercial intent ----------

def load_wildchat_prompts(n_to_scan: int = 5000) -> list[str]:
    """Load the first n WildChat conversations and extract first user-turn prompts.

    n_to_scan controls how many raw conversations we scan to find the 1k commercial-intent
    prompts after filtering. Default 5000 → assumes ~20% commercial-intent rate.
    """
    from datasets import load_dataset
    from config import WILDCHAT_DATASET

    ds = load_dataset(WILDCHAT_DATASET, split="train", streaming=True)
    prompts = []
    seen = set()
    for example in ds:
        if len(prompts) >= n_to_scan:
            break
        # Each example has a "conversation" field — list of turn dicts
        try:
            first_turn = example["conversation"][0]
            if first_turn.get("role") != "user":
                continue
            text = first_turn.get("content", "").strip()
            if example.get("language", "English") != "English":
                continue
            if not (5 <= len(text.split()) <= 200):
                continue
            if text in seen:
                continue
            seen.add(text)
            prompts.append(text)
        except (KeyError, IndexError, TypeError):
            continue
    return prompts


def filter_commercial_intent(
    prompts: list[str],
    target_n: int = 1000,
    progress: bool = True,
) -> list[str]:
    """Apply the LLM commercial-intent filter to prompts. Stops once target_n found.

    Cached, so re-runs are cheap.
    """
    out = []
    iterator = tqdm(prompts, desc="filtering for intent") if progress else prompts
    for p in iterator:
        if has_commercial_intent(p):
            out.append(p)
            if len(out) >= target_n:
                break
    return out


def save_prompts(prompts: list[str], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for p in prompts:
            f.write(json.dumps({"prompt": p}) + "\n")


def load_prompts(path: Path) -> list[str]:
    with Path(path).open() as f:
        return [json.loads(line)["prompt"] for line in f]


# ---------- Spot-check helper for the Day 1 gate ----------

def spot_check_filter(filtered_prompts: list[str], n: int = 100) -> pd.DataFrame:
    """Returns a sample of filtered prompts for human spot-checking.

    Eric reads through these and decides if the filter is clean enough to lock the 1k benchmark.
    Add a `keep` column manually in a spreadsheet, then compute precision = sum(keep)/n.
    """
    sample = filtered_prompts[:n]
    return pd.DataFrame({"prompt": sample, "keep_TF": ["" for _ in sample]})


# ---------- Amazon Reviews 2023: load + sample ----------

def _to_list(x) -> list:
    """Coerce parquet-loaded values to lists. Pandas/pyarrow may return numpy arrays."""
    if x is None:
        return []
    if isinstance(x, list):
        return x
    try:
        return list(x)
    except TypeError:
        return []


def _to_str(x) -> str:
    """Coerce a value (which may be a list of strings, a string, or None) to a single string."""
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    items = _to_list(x)
    return " ".join(str(i) for i in items if i is not None)


def load_amazon_products(n: int = 10_000, category: Optional[str] = None, max_shards: int = 10) -> list[Product]:
    """Load n Amazon products with non-null price/description from the given category.

    Reads the auto-converted parquet shards from the McAuley-Lab/Amazon-Reviews-2023
    repo (e.g. raw_meta_Electronics/full-00000-of-00010.parquet). One shard typically
    contains ~160k products, so loading the first shard is enough for n up to ~150k
    after filtering for valid metadata.

    Bypasses datasets.load_dataset since newer versions of the datasets library no
    longer support the script-based loading this dataset originally used.
    """
    import pandas as pd
    from huggingface_hub import hf_hub_download
    from config import AMAZON_DATASET, AMAZON_CATEGORY

    cat_raw = category or AMAZON_CATEGORY
    # Strip legacy "raw_meta_" prefix if present, e.g. "raw_meta_Electronics" -> "Electronics"
    cat = cat_raw[len("raw_meta_"):] if cat_raw.startswith("raw_meta_") else cat_raw

    products = []
    for shard_idx in range(max_shards):
        if len(products) >= n:
            break
        filename = f"raw_meta_{cat}/full-{shard_idx:05d}-of-00010.parquet"
        local_path = hf_hub_download(
            repo_id=AMAZON_DATASET,
            filename=filename,
            repo_type="dataset",
        )
        df = pd.read_parquet(local_path)
        for row in df.to_dict(orient="records"):
            if len(products) >= n:
                break
            try:
                price = row.get("price")
                if price is None or price == "" or (isinstance(price, str) and price.lower() == "none"):
                    continue
                try:
                    price_f = float(price)
                except (ValueError, TypeError):
                    continue
                if price_f <= 0:
                    continue
                title = (row.get("title") or "").strip()
                desc = _to_str(row.get("description"))[:500].strip()
                if not title or not desc:
                    continue
                features = [f for f in _to_list(row.get("features"))[:8] if isinstance(f, str)]
                cats_raw = _to_list(row.get("categories"))
                if cats_raw and isinstance(cats_raw[0], (list, tuple)) or hasattr(cats_raw[0] if cats_raw else None, "__iter__") and not isinstance(cats_raw[0], str):
                    cat_str = " > ".join(str(c) for c in _to_list(cats_raw[0]))
                elif cats_raw:
                    cat_str = " > ".join(str(c) for c in cats_raw)
                else:
                    cat_str = cat
                images_raw = row.get("images")
                image_url = None
                if isinstance(images_raw, dict):
                    large = _to_list(images_raw.get("large"))
                    if large:
                        image_url = str(large[0])
                elif isinstance(images_raw, list) and images_raw:
                    first = images_raw[0]
                    if isinstance(first, dict):
                        image_url = first.get("large") or first.get("hi_res") or first.get("thumb")
                products.append(Product(
                    id=row.get("parent_asin") or row.get("asin") or f"prod_{len(products)}",
                    title=title[:200],
                    description=desc,
                    price=price_f,
                    category=cat_str,
                    features=features,
                    image_url=image_url,
                ))
            except Exception:
                continue
    return products


def save_products(products: list[Product], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("w") as f:
        for p in products:
            f.write(json.dumps(p.__dict__) + "\n")


def load_products(path: Path) -> list[Product]:
    with Path(path).open() as f:
        return [Product(**json.loads(line)) for line in f]


# ---------- Embedding index ----------

def _product_text(p: Product) -> str:
    """The text we embed for retrieval. Title + first 200 chars of description + category."""
    feat = " ".join(p.features[:3])
    return f"{p.title}. {p.description[:300]} {feat} (category: {p.category})"


def build_product_index(products: list[Product]) -> np.ndarray:
    """Returns an (n_products, dim) matrix of L2-normalized embeddings."""
    embedder = _get_embedder()
    texts = [_product_text(p) for p in products]
    emb = embedder.encode(texts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


def embed_prompts(prompts: list[str]) -> np.ndarray:
    """Returns an (n_prompts, dim) matrix of L2-normalized embeddings."""
    embedder = _get_embedder()
    emb = embedder.encode(prompts, batch_size=64, show_progress_bar=True, convert_to_numpy=True)
    norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12
    return emb / norms


def save_embeddings(emb: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, emb)


def load_embeddings(path: Path) -> np.ndarray:
    return np.load(path)


# ---------- Retrieval ----------

def retrieve(
    prompt: str,
    product_index: np.ndarray,
    products: list[Product],
    k: int = 20,
) -> list[tuple[Product, float]]:
    """Return top-k (Product, similarity) for the given prompt. Similarity in [-1, 1]."""
    embedder = _get_embedder()
    q = embedder.encode([prompt], convert_to_numpy=True)[0]
    q = q / (np.linalg.norm(q) + 1e-12)
    sims = product_index @ q  # cosine since both are normalized
    top_idx = np.argsort(-sims)[:k]
    return [(products[i], float(sims[i])) for i in top_idx]


def relevance(prompt: str, product: Product, product_index: Optional[np.ndarray] = None,
              products: Optional[list[Product]] = None) -> float:
    """Cosine similarity between prompt and product, mapped to [0, 1].

    If product_index + products list provided, looks up the cached embedding to avoid recompute.
    Otherwise re-embeds. Mapping: cos in [-1, 1] → (cos + 1) / 2 in [0, 1].
    """
    embedder = _get_embedder()
    q = embedder.encode([prompt], convert_to_numpy=True)[0]
    q = q / (np.linalg.norm(q) + 1e-12)
    if product_index is not None and products is not None:
        try:
            i = next(j for j, p in enumerate(products) if p.id == product.id)
            cos = float(product_index[i] @ q)
        except StopIteration:
            cos = _embed_single_cosine(product, q)
    else:
        cos = _embed_single_cosine(product, q)
    return (cos + 1.0) / 2.0


def _embed_single_cosine(product: Product, q_normalized: np.ndarray) -> float:
    embedder = _get_embedder()
    pe = embedder.encode([_product_text(product)], convert_to_numpy=True)[0]
    pe = pe / (np.linalg.norm(pe) + 1e-12)
    return float(pe @ q_normalized)
