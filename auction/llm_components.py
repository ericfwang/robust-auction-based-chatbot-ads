"""LLM-backed steps in the pipeline: intent filter, slot detector, draft answer, rewriter."""
from .cached_llm import cached_llm
from .types import Ad, Product

# ---------- Commercial-intent filter ----------

INTENT_FILTER_SYSTEM = """You are classifying chatbot prompts for advertising suitability.

A prompt has COMMERCIAL INTENT if it expresses purchase interest, asks for a product/service
recommendation, or seeks advice on what to buy. Examples: "best wireless headphones for running",
"recommend a laptop under $1000", "what gaming console should I get my kid".

A prompt does NOT have commercial intent if it asks for help, information, education, opinion,
emotional support, code, writing, or anything else not directly tied to a purchasing decision.

Respond with exactly one word: YES or NO."""


def has_commercial_intent(prompt: str) -> bool:
    """True if the prompt expresses purchase intent. Caches results."""
    response = cached_llm(prompt, system=INTENT_FILTER_SYSTEM, temperature=0.0)
    return response.strip().upper().startswith("YES")


# ---------- Slot detector ----------

SLOT_DETECTOR_SYSTEM = """You are deciding whether it would be appropriate to show a product
advertisement alongside the assistant's answer to a chatbot prompt.

Say YES only if (a) the prompt is about a purchase decision or product category, AND (b) showing
a product ad would be helpful and contextually appropriate.

Say NO if the prompt is sensitive (health, mental health, legal, financial distress, grief,
relationship issues), purely informational without purchase intent, or in any way where ads
would feel intrusive or exploitative.

Respond with exactly one word: YES or NO."""


def is_appropriate_slot(prompt: str) -> bool:
    """True if showing an ad on this prompt is appropriate. Caches results."""
    response = cached_llm(prompt, system=SLOT_DETECTOR_SYSTEM, temperature=0.0)
    return response.strip().upper().startswith("YES")


# ---------- Draft answer ----------

DRAFT_ANSWER_SYSTEM = """You are a helpful chatbot assistant. Answer the user's question
concisely (3–5 sentences) and helpfully. If the user is asking about products, give a balanced
recommendation without naming specific brands unless the user explicitly asks for them."""


def draft_answer(prompt: str) -> str:
    """Generate the chatbot's clean (no-ad) answer to the prompt. Caches results."""
    return cached_llm(prompt, system=DRAFT_ANSWER_SYSTEM, temperature=0.3).strip()


# ---------- Rewriter (template-based, deterministic) ----------

def rewrite_with_ad(clean_answer: str, ad: Ad, product: Product) -> str:
    """Place the winning ad in a disclosed sponsored container below the answer.

    Template-based rather than LLM-generated for cost, latency, and reproducibility.
    Real ad systems do not call an LLM per impression to format the disclosure.
    """
    container = (
        "\n\n---\n"
        "**Sponsored**\n"
        f"**{product.title}** — ${product.price:.2f}\n"
        f"{ad.copy}\n"
        "*This is a paid placement. The recommendation above is independent.*"
    )
    return clean_answer + container


# ---------- Optional: LLM rewriter (more flexible, costs more) ----------

LLM_REWRITER_SYSTEM = """You will be given a chatbot's answer and a sponsored product to feature
below it. Output the original answer unchanged, then a clearly-labeled "Sponsored" section
that briefly describes the product and its price. Do NOT modify the original answer. Do NOT
present the sponsored content as a recommendation from the assistant."""


def rewrite_with_ad_llm(clean_answer: str, ad: Ad, product: Product) -> str:
    """LLM-based rewriter. Use only if you want stylistic flexibility; costs an LLM call."""
    user_msg = (
        f"ORIGINAL ANSWER:\n{clean_answer}\n\n"
        f"SPONSORED PRODUCT: {product.title} (${product.price:.2f})\n"
        f"AD COPY: {ad.copy}\n"
    )
    return cached_llm(user_msg, system=LLM_REWRITER_SYSTEM, temperature=0.0)
