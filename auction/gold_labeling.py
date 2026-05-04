"""Gemini cross-check + adjudication merge for the gold relevance label set.

1. `gemini_crosscheck_labels`: ask Gemini 2.5 Pro the same labeling question for each
   row, via cached_llm. Independent of wz (the primary labeler) and the Sonnet judge,
   so wz-vs-Gemini agreement is a meaningful sanity check on the gold set.

2. `merge_with_wz_labels`: join the wz and Gemini label sets, flag rows where the
   two labelers disagree by ≥ 0.25 (those need adjudication), and produce the final
   gold-set CSV shape.
"""
from typing import Optional
import re

from .cached_llm import cached_llm

# Gemini sees the same rubric wz used (data/labeling_protocol.md, condensed).
# Identical to JUDGE_SYSTEM in spirit but framed as "label" rather than "judge" so the
# resulting score functions as an independent gold-label, not a second judge.
GEMINI_LABELER_SYSTEM = """You are producing gold-standard relevance labels for evaluating an LLM ad-relevance judge.

For each (USER PROMPT, SERVED AD COPY) pair, score how well the served ad addresses the
prompt. Your label will be compared to a separate LLM judge's label to compute Cohen's
kappa. Be careful and consistent.

Score scale (use exactly one):
  1.0  = Directly relevant. Right category AND right feature constraints. User would say "exactly what I asked for."
  0.75 = Mostly relevant. Right category and most features; one secondary constraint missed (modest price overshoot, "wireless" requested but wired offered, etc.).
  0.5  = Partial fit. Right category but misses a hard constraint, OR adjacent category that could plausibly substitute.
  0.25 = Weak fit. Wrong subcategory but same broad domain, OR right category but missing the central use case.
  0.0  = Irrelevant. Wrong domain, or no useful match.

Score relevance only. Do not consider: aesthetics, persuasiveness, ad copy quality, or
whether the ad ethically *should* have served. Only: does the offered product address the
user's stated need?

Respond on a single line in this exact format:
SCORE|REASONING

Where SCORE is exactly one of: 0.0, 0.25, 0.5, 0.75, 1.0
And REASONING is one short sentence (under 30 words) naming the matched/missed constraints."""


_VALID_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _parse_label_response(text: str) -> tuple[Optional[float], str]:
    """Same parser shape as relevance_judge._parse_judge_response. Duplicated here to keep
    the module self-contained — the judge's parser could be reused, but mirroring it locally
    avoids accidental tight coupling between the labeler and the judge.
    """
    text = (text or "").strip()
    if "|" in text:
        head, _, tail = text.partition("|")
        try:
            val = float(head.strip())
            if val in _VALID_SCORES:
                return val, tail.strip()
        except ValueError:
            pass
    m = re.search(r"\b(0\.0|0\.25|0\.5|0\.75|1\.0|0|1)\b", text)
    if m:
        val = float(m.group(1))
        if val in _VALID_SCORES:
            return val, text[:200]
    return None, text[:200]


def gemini_crosscheck_labels(
    defended_df,
    model: str = "gemini-2.5-pro",
    copy_col: str = "winner_ad_copy",
    prompt_col: str = "prompt",
    id_col: str = "prompt_id",
    verbose: bool = True,
):
    """Run Gemini over every (prompt, ad_copy) row. Returns DataFrame with gemini_label, gemini_reasoning.

    Rows where copy_col is null get gemini_label=None and gemini_reasoning='<no ad served>'
    — same convention as the wz labeler, so the join is clean.
    """
    import pandas as pd

    rows = []
    n_total = len(defended_df)
    for i, (_, row) in enumerate(defended_df.iterrows()):
        rid = row[id_col]
        prompt = row[prompt_col]
        ad_copy = row.get(copy_col)
        if ad_copy is None or (isinstance(ad_copy, float) and pd.isna(ad_copy)) or str(ad_copy).strip() == "":
            rows.append({
                "prompt_id": rid,
                "gemini_label": None,
                "gemini_reasoning": "<no ad served>",
            })
            continue
        user_msg = f"USER PROMPT:\n{prompt}\n\nSERVED AD COPY:\n{ad_copy}"
        try:
            response = cached_llm(user_msg, system=GEMINI_LABELER_SYSTEM, model=model, temperature=0.0)
        except Exception as e:
            rows.append({
                "prompt_id": rid,
                "gemini_label": None,
                "gemini_reasoning": f"<error: {str(e)[:80]}>",
            })
            continue
        score, reasoning = _parse_label_response(response)
        rows.append({
            "prompt_id": rid,
            "gemini_label": score,
            "gemini_reasoning": reasoning,
        })
        if verbose and (i + 1) % 10 == 0:
            print(f"  gemini-labeled {i+1}/{n_total}")
    df = pd.DataFrame(rows)
    if verbose:
        scored = df["gemini_label"].dropna()
        if len(scored) > 0:
            print(f"Gemini: {len(scored)}/{len(df)} scored, mean={scored.mean():.3f}")
    return df


def merge_with_wz_labels(wz_df, gemini_df, disagree_threshold: float = 0.25):
    """Join wz + Gemini labels and flag disagreements.

    Args:
        wz_df: DataFrame with columns [prompt_id, prompt, winner_ad_copy, wz_label, wz_reasoning]
        gemini_df: DataFrame with columns [prompt_id, gemini_label, gemini_reasoning]
        disagree_threshold: |wz_label - gemini_label| >= this triggers needs_adjudication=True.

    Returns DataFrame with: prompt_id, prompt, winner_ad_copy, wz_label, wz_reasoning,
    gemini_label, gemini_reasoning, needs_adjudication.

    Adjudication itself (committing the final gold_label + adjudication_note) is done by
    wz after this merge — not in code, since it requires reading both reasonings.
    """
    import pandas as pd

    merged = wz_df.merge(gemini_df, on="prompt_id", how="left")

    def _disagree(row):
        a, b = row.get("wz_label"), row.get("gemini_label")
        if a is None or b is None or pd.isna(a) or pd.isna(b):
            # If either side is null, only flag for adjudication if exactly one is null
            # (the other claims a score) — that itself is a disagreement.
            return (a is None or pd.isna(a)) != (b is None or pd.isna(b))
        return abs(float(a) - float(b)) >= disagree_threshold

    merged["needs_adjudication"] = merged.apply(_disagree, axis=1)
    return merged
