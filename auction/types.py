"""Shared data contracts. Locked at kickoff — change with team alignment only."""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Product:
    id: str
    title: str
    description: str
    price: float
    category: str
    features: list[str] = field(default_factory=list)
    image_url: Optional[str] = None


@dataclass
class Ad:
    advertiser_id: str
    product_id: str
    copy: str          # the actual ad text
    landing_page: str  # text of the product's landing page (used by gaming defenses)


@dataclass
class Advertiser:
    id: str
    product: Product
    margin: float           # 0–1, fraction of price kept as margin
    conversion_rate: float  # 0–1, expected conversion per impression
    ad: Ad


@dataclass
class AuctionResult:
    winner_advertiser_id: Optional[str]
    winner_product_id: Optional[str]
    clearing_price: float           # in bid-equivalent dollars
    winner_relevance: float         # 0–1
    predicted_welfare_loss: float   # 0–1
    reserve: float                  # in score units
    winner_score: float             # in score units (bid × relevance × (1 − wl))
    runner_up_score: float
    rewritten_answer: Optional[str] = None
    n_candidates: int = 0
    n_above_reserve: int = 0


@dataclass
class AdServeRecord:
    """The handoff schema between Eric's pipeline and downstream eval/demo work.

    One AdServeRecord per (prompt, mechanism_variant) — both Wes (gaming + relevance eval)
    and Alex/Swetha (welfare judge + visual mocks) consume CSV exports of these.
    """
    # Prompt context
    prompt_id: str
    prompt: str
    clean_answer: str
    is_sensitive: bool
    is_borderline: bool          # subset of sensitive where welfare predictor has real work
    category: str

    # Mechanism applied
    mechanism_variant: str       # "defended" | "revmax" | "pure_relevance"

    # Winner (None if no ad cleared)
    winner_advertiser_id: Optional[str]
    winner_product_id: Optional[str]
    winner_product_title: Optional[str]
    winner_product_price: Optional[float]
    winner_product_image_url: Optional[str]   # for Alex/Swetha visual mocks
    winner_ad_copy: Optional[str]             # for Wes adversarial attacks
    winner_landing_page: Optional[str]        # for Wes landing-page-consistency defense

    # Auction outputs
    clearing_price: float
    relevance_score: float
    predicted_welfare_loss: float
    reserve: float

    # Rendered output (None if no ad cleared)
    rewritten_answer: Optional[str]           # input to Alex/Swetha's Claude welfare judge

    # Bookkeeping
    candidate_product_ids: list[str]          # so Wes can attack candidates, not just winners
    n_candidates: int
    n_above_reserve: int
