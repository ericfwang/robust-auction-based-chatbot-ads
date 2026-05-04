"""Auction mechanism: synthetic advertiser generation, VCG single-slot clearing, welfare reserve.

Bid model (per the proposal):
    bid_i = price_i × margin_i × conversion_rate_i × relevance(i, prompt) + ε

Auction score:
    score_i = bid_i × relevance_i × (1 − predicted_welfare_loss)

For a single slot, VCG = GSP = second-price. Winner pays the bid that would have made
the runner-up tie. Reserve gates whether anything clears at all.
"""
from typing import Callable, Optional

import numpy as np

from .types import Ad, AdServeRecord, Advertiser, AuctionResult, Product


# ---------- Synthetic advertiser pool ----------

def generate_advertisers(
    products: list[Product],
    n: int = 100,
    seed: int = 42,
) -> list[Advertiser]:
    """Sample n products and assign each a synthetic advertiser with margin/conversion priors."""
    from config import MARGIN_ALPHA, MARGIN_BETA, CONV_ALPHA, CONV_BETA

    rng = np.random.default_rng(seed)
    if n > len(products):
        n = len(products)
    idx = rng.choice(len(products), size=n, replace=False)
    advertisers = []
    for i, j in enumerate(idx):
        p = products[int(j)]
        margin = float(rng.beta(MARGIN_ALPHA, MARGIN_BETA))
        conv = float(rng.beta(CONV_ALPHA, CONV_BETA))
        ad = Ad(
            advertiser_id=f"adv_{i:04d}",
            product_id=p.id,
            copy=f"{p.title}. {p.description[:200]}",  # honest copy by default
            landing_page=f"{p.title}. {p.description}",
        )
        advertisers.append(Advertiser(
            id=f"adv_{i:04d}",
            product=p,
            margin=margin,
            conversion_rate=conv,
            ad=ad,
        ))
    return advertisers


# ---------- Bid generation ----------

def compute_bid(
    advertiser: Advertiser,
    relevance: float,
    rng: Optional[np.random.Generator] = None,
    noise_sigma: Optional[float] = None,
) -> float:
    """Per-prompt bid for one advertiser. See module docstring for formula."""
    from config import BID_NOISE_SIGMA
    if rng is None:
        rng = np.random.default_rng()
    if noise_sigma is None:
        noise_sigma = BID_NOISE_SIGMA
    base = advertiser.product.price * advertiser.margin * advertiser.conversion_rate * relevance
    noise = float(rng.lognormal(mean=0.0, sigma=noise_sigma * 0.1))  # multiplicative
    return float(base * noise)


# ---------- Single-slot VCG with welfare-priced reserve ----------

def run_auction(
    prompt: str,
    candidate_products: list[Product],
    advertisers: list[Advertiser],
    relevance_fn: Callable[[str, Product], float],
    welfare_loss_fn: Callable[[str], float],
    reserve_alpha: Optional[float] = None,
    seed: Optional[int] = None,
) -> AuctionResult:
    """Run a single-slot VCG auction over advertisers whose product appears in candidate_products.

    Args:
        prompt: the user prompt being auctioned.
        candidate_products: products retrieved by embedding similarity (the candidate pool).
        advertisers: full advertiser pool. Only those with a product in candidate_products bid.
        relevance_fn: (prompt, product) -> float in [0, 1].
        welfare_loss_fn: (prompt) -> float in [0, 1]. Used both for the reserve and to discount scores.
        reserve_alpha: reserve = reserve_alpha × welfare_loss × max_possible_score_baseline.
                       If None, uses config.RESERVE_ALPHA.
        seed: random seed for bid noise.

    Returns:
        AuctionResult with the winning advertiser/product, clearing price (in bid units),
        relevance score, predicted welfare loss, reserve, and bookkeeping.
    """
    from config import RESERVE_ALPHA
    if reserve_alpha is None:
        reserve_alpha = RESERVE_ALPHA

    rng = np.random.default_rng(seed)

    # Map advertiser → product.id, filter to candidates
    candidate_ids = {p.id for p in candidate_products}
    eligible = [a for a in advertisers if a.product.id in candidate_ids]

    welfare_loss = float(welfare_loss_fn(prompt))

    # Score each eligible advertiser: score = bid × relevance × (1 − welfare_loss)
    rows = []
    for adv in eligible:
        rel = float(relevance_fn(prompt, adv.product))
        bid = compute_bid(adv, rel, rng=rng)
        score = bid * rel * (1.0 - welfare_loss)
        rows.append((adv, rel, bid, score))

    # Reserve in score units. Tie reserve to welfare loss: high welfare loss → high reserve.
    # The constant is the median expected score across the pool when welfare_loss=0.
    # We use a simple fixed multiplier on welfare_loss; calibrate empirically.
    reserve_score = reserve_alpha * welfare_loss

    if not rows:
        return AuctionResult(
            winner_advertiser_id=None, winner_product_id=None,
            clearing_price=0.0, winner_relevance=0.0,
            predicted_welfare_loss=welfare_loss, reserve=reserve_score,
            winner_score=0.0, runner_up_score=0.0,
            n_candidates=0, n_above_reserve=0,
        )

    rows.sort(key=lambda r: -r[3])
    above = [r for r in rows if r[3] >= reserve_score]
    if not above:
        return AuctionResult(
            winner_advertiser_id=None, winner_product_id=None,
            clearing_price=0.0, winner_relevance=0.0,
            predicted_welfare_loss=welfare_loss, reserve=reserve_score,
            winner_score=rows[0][3], runner_up_score=rows[1][3] if len(rows) > 1 else 0.0,
            n_candidates=len(rows), n_above_reserve=0,
        )

    winner_adv, winner_rel, winner_bid, winner_score = above[0]
    runner_up_score = above[1][3] if len(above) > 1 else reserve_score

    # VCG single-slot clearing: the bid that would have made winner's score equal to runner-up's.
    # winner_score_at_clearing = clearing_bid × winner_rel × (1 − welfare_loss) = runner_up_score
    # → clearing_bid = runner_up_score / (winner_rel × (1 − welfare_loss))
    denom = winner_rel * (1.0 - welfare_loss)
    if denom <= 1e-9:
        clearing_price = winner_bid  # degenerate; fall back to bid
    else:
        clearing_price = float(runner_up_score / denom)

    return AuctionResult(
        winner_advertiser_id=winner_adv.id,
        winner_product_id=winner_adv.product.id,
        clearing_price=clearing_price,
        winner_relevance=winner_rel,
        predicted_welfare_loss=welfare_loss,
        reserve=reserve_score,
        winner_score=winner_score,
        runner_up_score=runner_up_score,
        n_candidates=len(rows),
        n_above_reserve=len(above),
    )


# ---------- Baselines for comparison ----------

def run_pure_relevance(
    prompt: str,
    candidate_products: list[Product],
    advertisers: list[Advertiser],
    relevance_fn: Callable[[str, Product], float],
) -> AuctionResult:
    """Baseline 1: pick the most-relevant ad regardless of bid. No auction, no reserve."""
    candidate_ids = {p.id for p in candidate_products}
    eligible = [a for a in advertisers if a.product.id in candidate_ids]
    if not eligible:
        return AuctionResult(None, None, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0, 0)
    scored = [(a, float(relevance_fn(prompt, a.product))) for a in eligible]
    scored.sort(key=lambda r: -r[1])
    winner, rel = scored[0]
    return AuctionResult(
        winner_advertiser_id=winner.id,
        winner_product_id=winner.product.id,
        clearing_price=0.0,  # no payment in pure-relevance baseline
        winner_relevance=rel,
        predicted_welfare_loss=0.0,
        reserve=0.0,
        winner_score=rel,
        runner_up_score=scored[1][1] if len(scored) > 1 else 0.0,
        n_candidates=len(eligible),
        n_above_reserve=len(eligible),
    )


def run_revenue_max(
    prompt: str,
    candidate_products: list[Product],
    advertisers: list[Advertiser],
    relevance_fn: Callable[[str, Product], float],
    seed: Optional[int] = None,
) -> AuctionResult:
    """Baseline 2: VCG, no welfare reserve. Reserve_alpha = 0."""
    return run_auction(
        prompt=prompt,
        candidate_products=candidate_products,
        advertisers=advertisers,
        relevance_fn=relevance_fn,
        welfare_loss_fn=lambda _: 0.0,
        reserve_alpha=0.0,
        seed=seed,
    )


# ---------- High-level entry point: prompt → AdServeRecord ----------

def serve_ad(
    prompt: str,
    *,
    products: list[Product],
    product_index: np.ndarray,
    advertisers: list[Advertiser],
    variant: str = "defended",
    clean_answer: str = "",
    prompt_id: str = "live",
    is_sensitive: bool = False,
    is_borderline: bool = False,
    category: str = "",
    k: int = 20,
    seed: Optional[int] = None,
) -> AdServeRecord:
    """End-to-end single-prompt entry point: retrieve → auction → rewrite → AdServeRecord.

    Use for the Gradio live demo and for the full benchmark loop. Wraps all the steps
    Alex/Swetha and Wes would otherwise have to assemble themselves.

    Args:
        prompt: the user prompt
        products, product_index: the loaded Amazon product feed and its embedding index
        advertisers: the synthetic advertiser pool (one per product)
        variant: "defended" (welfare-priced reserve), "revmax" (no reserve), or "pure_relevance"
        clean_answer: the no-ad answer for this prompt; used by the rewriter
        prompt_id, is_sensitive, is_borderline, category: bookkeeping carried into the record
        k: retrieval depth (default 20 — sponsored-search-like depth)
        seed: random seed for bid noise
    """
    from .data_pipeline import retrieve, relevance
    from .welfare_predictor import predict_welfare_loss
    from .llm_components import rewrite_with_ad
    from functools import partial

    rel_fn = partial(relevance, product_index=product_index, products=products)
    candidates = [prod for prod, _ in retrieve(prompt, product_index, products, k=k)]

    if variant == "defended":
        result = run_auction(prompt, candidates, advertisers, rel_fn, predict_welfare_loss, seed=seed)
    elif variant == "revmax":
        result = run_revenue_max(prompt, candidates, advertisers, rel_fn, seed=seed)
    elif variant == "pure_relevance":
        result = run_pure_relevance(prompt, candidates, advertisers, rel_fn)
    else:
        raise ValueError(f"Unknown variant: {variant!r} (use defended | revmax | pure_relevance)")

    winner_adv: Optional[Advertiser] = None
    winner_product: Optional[Product] = None
    rewritten: Optional[str] = None
    if result.winner_advertiser_id:
        winner_adv = next((a for a in advertisers if a.id == result.winner_advertiser_id), None)
        if winner_adv is not None:
            winner_product = winner_adv.product
            if clean_answer:
                rewritten = rewrite_with_ad(clean_answer, winner_adv.ad, winner_product)

    return AdServeRecord(
        prompt_id=prompt_id,
        prompt=prompt,
        clean_answer=clean_answer,
        is_sensitive=is_sensitive,
        is_borderline=is_borderline,
        category=category,
        mechanism_variant=variant,
        winner_advertiser_id=result.winner_advertiser_id,
        winner_product_id=result.winner_product_id,
        winner_product_title=winner_product.title if winner_product else None,
        winner_product_price=winner_product.price if winner_product else None,
        winner_product_image_url=winner_product.image_url if winner_product else None,
        winner_ad_copy=winner_adv.ad.copy if winner_adv else None,
        winner_landing_page=winner_adv.ad.landing_page if winner_adv else None,
        clearing_price=result.clearing_price,
        relevance_score=result.winner_relevance,
        predicted_welfare_loss=result.predicted_welfare_loss,
        reserve=result.reserve,
        rewritten_answer=rewritten,
        candidate_product_ids=[c.id for c in candidates],
        n_candidates=result.n_candidates,
        n_above_reserve=result.n_above_reserve,
    )


# ---------- Demo: render an AdServeRecord as an HTML product card ----------

_AD_CARD_CSS = """
<style>
.ad-card { font-family: -apple-system, system-ui, sans-serif; border: 1px solid #e0e0e0;
    border-radius: 8px; padding: 16px; max-width: 360px; background: #fafafa; margin: 8px 0; }
.ad-card .badge { display: inline-block; background: #fff3cd; color: #5d4400; font-size: 11px;
    font-weight: 600; padding: 2px 8px; border-radius: 4px; margin-bottom: 8px; letter-spacing: 0.5px; }
.ad-card img { max-width: 100%; max-height: 180px; display: block; margin: 8px auto; object-fit: contain; }
.ad-card .title { font-size: 15px; font-weight: 600; color: #111; margin-top: 8px; }
.ad-card .price { font-size: 18px; font-weight: 700; color: #b12704; margin: 6px 0; }
.ad-card .copy { font-size: 13px; color: #444; line-height: 1.4; margin-top: 8px; }
.ad-card .disclosure { font-size: 11px; color: #888; font-style: italic; margin-top: 12px;
    padding-top: 8px; border-top: 1px solid #eee; }
.no-ad { font-family: system-ui; color: #888; font-style: italic; padding: 16px; }
</style>
"""


def render_ad_card_html(record: AdServeRecord, include_css: bool = True) -> str:
    """Render an AdServeRecord as an HTML product card. Drop-in for Gradio HTML component."""
    if not record.winner_product_id:
        body = '<div class="no-ad">No sponsored result for this query (welfare reserve gated, or no eligible bidder).</div>'
    else:
        img = (
            f'<img src="{record.winner_product_image_url}" alt="{record.winner_product_title}" />'
            if record.winner_product_image_url else ""
        )
        title = (record.winner_product_title or "").replace("<", "&lt;").replace(">", "&gt;")
        copy = (record.winner_ad_copy or "").replace("<", "&lt;").replace(">", "&gt;")
        price = f"${record.winner_product_price:.2f}" if record.winner_product_price is not None else ""
        body = (
            f'<div class="ad-card">'
            f'<div class="badge">SPONSORED</div>'
            f'{img}'
            f'<div class="title">{title}</div>'
            f'<div class="price">{price}</div>'
            f'<div class="copy">{copy}</div>'
            f'<div class="disclosure">This is a paid placement. The recommendation above is independent.</div>'
            f'</div>'
        )
    return (_AD_CARD_CSS + body) if include_css else body
