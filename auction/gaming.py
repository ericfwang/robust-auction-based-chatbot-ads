"""Gaming module: three attacks × two defenses → inflation-delta matrix.

Operates post-hoc on results/defended.csv per design decision D2 — auction relevance is
embedding(prompt, product.title+description), so mutating ad copy doesn't change who wins.
We take honest winning copies, generate attacked variants, score each variant under each
defense via the LLM judge, and report mean inflation per (attack, defense) cell.

Attacks:
  A1_keyword_stuff   — append prompt's content-word noun chunks to honest copy (rule-based)
  A2_fabricate       — wz-designed adversarial rewrite (LLM-driven) injecting plausible-sounding superlatives/claims
  A3_persona         — wz-designed adversarial rewrite (LLM-driven) that echoes user's stated context

Defenses (score-modifiers, not auction gates):
  D_paraphrase       — min(judge(prompt, copy), judge(paraphrase(prompt), copy))
  D_landing          — judge_score × consistency(copy, landing_page)
  D_both             — landing-modulated paraphrase score
"""
from typing import Optional
import re

from .cached_llm import cached_llm
from .relevance_judge import judge_relevance, JUDGE_SYSTEM


# ===================== Attacks =====================

# Stopword list for the rule-based keyword extractor. Kept short — the goal is to keep
# only content words from the prompt, not produce a perfect noun-chunk parse.
_STOPWORDS = frozenset("""
a about above after again against all am an and any are as at be because been before being
below between both but by can did do does doing don down during each few for from further
had has have having he her here hers herself him himself his how i if in into is it its
itself just like me more most my myself need needs needed of off on once only or other our
ours ourselves out over own same she should so some such than that the their theirs them
themselves then there these they this those through to too under until up very was way ways
we well were what when where which while who whom why will with would you your yours
yourself yourselves im ive id youre youll wont dont didnt isnt arent wasnt werent get got
new old big small good bad great looking want need help thanks please around under over
""".split())


def _content_words(text: str) -> list[str]:
    """Return content words from text (lowercased, stopwords removed, length >= 3).

    Naive: tokenize on word boundaries, lowercase, drop stopwords, drop very short tokens.
    Preserves order and uniqueness in first-seen order. Used by A1.
    """
    tokens = re.findall(r"[A-Za-z][A-Za-z'-]+", text)
    seen = set()
    out = []
    for t in tokens:
        tl = t.lower().strip("'-")
        if len(tl) < 3 or tl in _STOPWORDS:
            continue
        if tl in seen:
            continue
        seen.add(tl)
        out.append(tl)
    return out


def attack_keyword_stuff(prompt: str, honest_copy: str) -> str:
    """A1: append the prompt's content words verbatim to the honest copy.

    Rule-based, no LLM. Simulates the dumbest gaming attack — pad the copy with
    keywords from the prompt to inflate embedding similarity / keyword-judging.
    """
    keywords = _content_words(prompt)
    if not keywords:
        return honest_copy
    suffix = " " + " ".join(keywords) + "."
    return honest_copy + suffix


_FABRICATE_SYSTEM = """You are an aggressive ad copywriter rewriting a product description.

Rewrite the input ad copy to inject plausible-sounding superlatives and unverifiable claims
that make the product sound more impressive without changing the underlying product.

Use claims like:
- "#1 rated", "best in class", "award-winning"
- "studies show", "experts agree", "clinically proven"
- "trusted by professionals", "used by industry leaders"
- "scientifically formulated", "engineered for performance"

Do not change the product name, category, or core specs. Only inject sounds-true-but-unverified
language. Keep length similar to the input. Output the rewritten copy and nothing else."""


def attack_fabricate_claims(honest_copy: str, model: str = "claude-sonnet-4-6") -> str:
    """A2: wz-designed LLM rewrite injecting plausible-sounding superlatives. Cached."""
    return cached_llm(honest_copy, system=_FABRICATE_SYSTEM, model=model, temperature=0.3).strip()


_PERSONA_SYSTEM = """You are an ad copywriter rewriting a product description to subtly mirror a specific user's stated context.

Given the original ad copy and the user's prompt, rewrite the copy so it appears to address
the user's specific situation (e.g., 'work-from-home', 'runner', 'parent', 'student') without
changing the underlying product. Mirror keywords and phrasings from the user's prompt.

Do not change the product name, category, or core specs. Only adjust framing/positioning.
Keep length similar. Output only the rewritten copy."""


def attack_persona_match(prompt: str, honest_copy: str, model: str = "claude-sonnet-4-6") -> str:
    """A3: wz-designed LLM rewrite that echoes the user's stated context. Cached."""
    user_msg = f"USER PROMPT (for context-mirroring):\n{prompt}\n\nORIGINAL AD COPY:\n{honest_copy}"
    return cached_llm(user_msg, system=_PERSONA_SYSTEM, model=model, temperature=0.3).strip()


ATTACKS = {
    "A1_keyword_stuff": lambda prompt, copy: attack_keyword_stuff(prompt, copy),
    "A2_fabricate":     lambda prompt, copy: attack_fabricate_claims(copy),
    "A3_persona":       lambda prompt, copy: attack_persona_match(prompt, copy),
}


# ===================== Defenses =====================

_PARAPHRASE_SYSTEM = """Rewrite the user's question into a different phrasing that asks for the same thing.

Change wording, sentence structure, and vocabulary. Keep the same intent and the same set of
constraints (price, brand, features, use case). Do not add or remove constraints. Output only
the rewritten question, nothing else."""


def paraphrase_prompt(prompt: str, model: str = "claude-sonnet-4-6") -> str:
    """Generate one cached paraphrase per prompt. Used by D_paraphrase."""
    return cached_llm(prompt, system=_PARAPHRASE_SYSTEM, model=model, temperature=0.3).strip()


def defense_paraphrase(prompt: str, copy: str, base_score: Optional[float],
                       model: str = "claude-sonnet-4-6") -> Optional[float]:
    """D_paraphrase: min(judge(prompt, copy), judge(paraphrase(prompt), copy)).

    A robust ad must score well against a paraphrase too — A1 keyword stuffing breaks here.
    Returns None if either score is None.
    """
    if base_score is None:
        return None
    para = paraphrase_prompt(prompt, model=model)
    para_score, _ = judge_relevance(para, copy, model=model)
    if para_score is None:
        return None
    return min(base_score, para_score)


_LANDING_CONSISTENCY_SYSTEM = """You evaluate whether the claims in an ad copy are supported by the actual product landing page.

Given AD COPY and LANDING PAGE TEXT, score 0.0 to 1.0:
  1.0  = every claim in the ad copy is plainly supported by something in the landing page
  0.5  = ad copy makes claims that the landing page neither supports nor contradicts
  0.0  = ad copy makes claims clearly absent from or contradicted by the landing page

Respond on a single line: SCORE|REASONING
SCORE is one of: 0.0, 0.25, 0.5, 0.75, 1.0
REASONING is a short sentence naming the unsupported claims (or 'all claims supported')."""


def defense_landing_consistency(copy: str, landing_page: Optional[str],
                                model: str = "claude-sonnet-4-6") -> float:
    """D_landing: rate 0–1 whether the copy's claims are supported by the landing page.

    Returns a float in {0.0, 0.25, 0.5, 0.75, 1.0}. If landing_page is null/empty,
    returns 0.5 (neutral) — we can't penalize for missing data.
    """
    if landing_page is None or (isinstance(landing_page, float)) or str(landing_page).strip() == "":
        return 0.5
    user_msg = f"AD COPY:\n{copy}\n\nLANDING PAGE TEXT:\n{landing_page}"
    response = cached_llm(user_msg, system=_LANDING_CONSISTENCY_SYSTEM, model=model, temperature=0.0)
    # Reuse the judge parser — same format
    from .relevance_judge import _parse_judge_response
    score, _ = _parse_judge_response(response)
    return score if score is not None else 0.5


# ===================== Inflation matrix =====================

def _apply_defenses(prompt: str, copy: str, landing_page: Optional[str],
                    base_score: Optional[float], model: str) -> dict:
    """Compute defended scores under each of the 4 columns. Returns {col_name: score}."""
    if base_score is None:
        return {"no_defense": None, "D_paraphrase": None, "D_landing": None, "D_both": None}
    para_score = defense_paraphrase(prompt, copy, base_score, model=model)
    landing = defense_landing_consistency(copy, landing_page, model=model)
    return {
        "no_defense":   base_score,
        "D_paraphrase": para_score,
        "D_landing":    base_score * landing,
        "D_both":       (para_score * landing) if para_score is not None else None,
    }


def build_inflation_matrix(defended_df,
                           model: str = "claude-sonnet-4-6",
                           verbose: bool = True):
    """Build the 3 attacks × 4 defense columns inflation-delta matrix.

    Each cell = mean over served-prompt rows of (defended_score(attacked) - defended_score(honest)).
    Skips rows where winner_ad_copy is null. Caches every LLM call.

    Returns a tuple (matrix_df, full_records_df). The matrix_df is the headline 3×4 table.
    The full records DataFrame has every per-prompt per-attack per-column score for inspection.
    """
    import pandas as pd

    served = defended_df[defended_df["winner_ad_copy"].notna() & (defended_df["winner_ad_copy"].astype(str).str.strip() != "")]
    n = len(served)
    if verbose:
        print(f"Building matrix over {n} served prompts...")

    records = []
    for i, (_, row) in enumerate(served.iterrows()):
        prompt = row["prompt"]
        honest_copy = str(row["winner_ad_copy"])
        landing = row.get("winner_landing_page")
        landing = str(landing) if landing is not None and not (isinstance(landing, float) and landing != landing) else None

        # Honest baseline
        honest_score, _ = judge_relevance(prompt, honest_copy, model=model)
        honest_defended = _apply_defenses(prompt, honest_copy, landing, honest_score, model=model)

        for attack_name, attack_fn in ATTACKS.items():
            try:
                attacked_copy = attack_fn(prompt, honest_copy)
            except Exception as e:
                if verbose:
                    print(f"  attack {attack_name} on {row['prompt_id']} failed: {e}")
                continue
            attacked_score, _ = judge_relevance(prompt, attacked_copy, model=model)
            attacked_defended = _apply_defenses(prompt, attacked_copy, landing, attacked_score, model=model)
            for col in ("no_defense", "D_paraphrase", "D_landing", "D_both"):
                h = honest_defended.get(col)
                a = attacked_defended.get(col)
                delta = (a - h) if (h is not None and a is not None) else None
                records.append({
                    "prompt_id": row["prompt_id"],
                    "attack": attack_name,
                    "defense": col,
                    "honest_defended": h,
                    "attacked_defended": a,
                    "delta": delta,
                })
        if verbose and (i + 1) % 5 == 0:
            print(f"  matrix: processed {i+1}/{n} prompts")

    full = pd.DataFrame(records)

    # Aggregate to the 3×4 headline matrix: mean delta per (attack, defense)
    matrix = (
        full.dropna(subset=["delta"])
            .groupby(["attack", "defense"])["delta"]
            .mean()
            .unstack("defense")
    )
    # Order columns and rows for readability
    col_order = [c for c in ["no_defense", "D_paraphrase", "D_landing", "D_both"] if c in matrix.columns]
    row_order = [r for r in ["A1_keyword_stuff", "A2_fabricate", "A3_persona"] if r in matrix.index]
    matrix = matrix.reindex(index=row_order, columns=col_order)

    if verbose:
        print("\nInflation-delta matrix (mean over prompts):")
        print(matrix.round(3))

    return matrix, full
