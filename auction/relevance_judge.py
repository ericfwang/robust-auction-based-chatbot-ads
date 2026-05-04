"""Relevance LLM judge for Wes's eval workstream.

Given (prompt, ad_copy), the judge produces a relevance score in
{0.0, 0.25, 0.5, 0.75, 1.0} plus a short reasoning. Mirrors the parser/loop
pattern from modules/validator.py so the team only has to learn one shape.

Cost: ~50 cached calls per CSV per defense-mode. Reruns are free.
"""
from typing import Optional, Union

import re

from .cached_llm import cached_llm

# The judge sees the same rubric wz used (data/labeling_protocol.md).
# This keeps the comparison apples-to-apples — labeler and judge against one rubric.
JUDGE_SYSTEM = """You evaluate whether a sponsored product ad is relevant to a user's prompt.

The user wrote a prompt asking for product help. An ad system selected and showed them
a sponsored product. Your job: score how well the served ad addresses the prompt.

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
And REASONING is one short sentence (under 30 words) naming the matched/missed constraints.

Example outputs:
1.0|Right category (over-ear headphones) with the stated key features (ANC, dual-device pairing, under-$400 budget).
0.5|Right category (security camera) but misses the central "video doorbell" form factor the user asked for.
0.25|Same broad domain (audio) but wrong product class (speakers offered when over-ear headphones were requested)."""


_VALID_SCORES = (0.0, 0.25, 0.5, 0.75, 1.0)


def _parse_judge_response(text: str) -> tuple[Optional[float], str]:
    """Parse 'SCORE|REASONING'. Returns (score, reasoning) or (None, raw_text) on failure.

    Mirrors modules/validator.py:_parse_validator_response — first try the strict format,
    then a regex fallback that grabs the first valid score literal in the response.
    """
    text = (text or "").strip()
    # Some cached responses include the requested placeholder before the actual answer,
    # e.g. "SCORE|REASONING\n0.75|...". Prefer the real scored line.
    for line in text.splitlines():
        line = line.strip()
        m = re.match(r"^(0(?:\.0|\.25|\.5|\.75)?|1(?:\.0)?)\s*\|\s*(.+)$", line)
        if not m:
            m = re.match(r"^SCORE\s*:\s*(0(?:\.0|\.25|\.5|\.75)?|1(?:\.0)?)\s*\|\s*(.+)$", line)
        if m:
            val = float(m.group(1))
            if val in _VALID_SCORES:
                return val, _clean_reasoning(m.group(2))

    if "|" in text:
        head, _, tail = text.partition("|")
        try:
            val = float(head.strip())
            if val in _VALID_SCORES:
                return val, _clean_reasoning(tail)
        except ValueError:
            pass
    # Fallback: find the first valid score literal anywhere in the response.
    m = re.search(r"\b(0\.0|0\.25|0\.5|0\.75|1\.0|0|1)\s*\|\s*([^\n]+)", text)
    if m:
        raw = m.group(1)
        val = float(raw)
        # Map bare 0/1 to 0.0/1.0
        if val in _VALID_SCORES:
            return val, _clean_reasoning(m.group(2))
    return None, _clean_reasoning(text[:200])


def _clean_reasoning(text: str) -> str:
    """Remove prompt-format boilerplate from judge reasoning strings."""
    text = (text or "").strip()
    text = re.sub(r"^SCORE\s*\|\s*REASONING\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^SCORE\s*:\s*", "", text, flags=re.IGNORECASE).strip()
    text = re.sub(r"^(0(?:\.0|\.25|\.5|\.75)?|1(?:\.0)?)\s*\|\s*", "", text).strip()
    return " ".join(text.split())


def judge_relevance(
    prompt: str,
    ad_copy: str,
    model: str = "claude-sonnet-4-6",
) -> tuple[Optional[float], str]:
    """Score a single (prompt, ad_copy) pair. Returns (score, reasoning).

    If the first attempt's response can't be parsed (e.g., the LLM returns the literal
    placeholder 'SCORE|REASONING' instead of substituting), automatically retry with a
    one-line prefix that makes the cache key differ — this avoids a permanently-bad
    cached response sticking.
    """
    user_msg = f"USER PROMPT:\n{prompt}\n\nSERVED AD COPY:\n{ad_copy}"
    response = cached_llm(user_msg, system=JUDGE_SYSTEM, model=model, temperature=0.0)
    score, reasoning = _parse_judge_response(response)
    if score is None:
        # Retry once with an extra instruction at the top — different prompt = different
        # cache key, so we get a fresh call. Most parse failures are the model returning
        # the literal SCORE token; the emphatic retry usually fixes it.
        retry_system = JUDGE_SYSTEM + "\n\nIMPORTANT: Replace the literal word SCORE with one of the five numeric values. Do not output the word SCORE itself."
        response = cached_llm(user_msg, system=retry_system, model=model, temperature=0.0)
        score, reasoning = _parse_judge_response(response)
    return score, reasoning


def judge_dataframe(
    records_df,
    copy_col: str = "winner_ad_copy",
    prompt_col: str = "prompt",
    id_col: str = "prompt_id",
    model: str = "claude-sonnet-4-6",
    verbose: bool = True,
):
    """Run the judge over every row of records_df. Mirrors validate_with_llm structure.

    Rows where copy_col is null (no ad served) get judge_score=None and reasoning='<no ad served>'.

    Returns a DataFrame with: id, prompt, ad_copy, judge_score, judge_reasoning.
    """
    import pandas as pd

    rows = []
    n_total = len(records_df)
    for i, (_, row) in enumerate(records_df.iterrows()):
        ad_copy = row.get(copy_col)
        prompt = row[prompt_col]
        rid = row[id_col]
        if ad_copy is None or (isinstance(ad_copy, float) and pd.isna(ad_copy)) or str(ad_copy).strip() == "":
            rows.append({
                "id": rid,
                "prompt": prompt,
                "ad_copy": None,
                "judge_score": None,
                "judge_reasoning": "<no ad served>",
            })
            continue
        try:
            score, reasoning = judge_relevance(prompt, str(ad_copy), model=model)
        except Exception as e:
            rows.append({
                "id": rid,
                "prompt": prompt,
                "ad_copy": str(ad_copy)[:200],
                "judge_score": None,
                "judge_reasoning": f"<error: {str(e)[:80]}>",
            })
            continue
        rows.append({
            "id": rid,
            "prompt": prompt,
            "ad_copy": str(ad_copy)[:200],
            "judge_score": score,
            "judge_reasoning": reasoning,
        })
        if verbose and (i + 1) % 10 == 0:
            print(f"  judged {i+1}/{n_total}")
    df = pd.DataFrame(rows)
    if verbose:
        scored = df["judge_score"].dropna()
        print(f"Judge: {len(scored)}/{len(df)} scored, mean={scored.mean():.3f}, std={scored.std():.3f}")
        print(f"  score distribution: {dict(scored.value_counts().sort_index())}")
    return df


def cohens_kappa_against_gold(judge_df, gold_df, judge_score_col: str = "judge_score", gold_score_col: str = "gold_label"):
    """Cohen's κ between the judge and the gold set.

    Drops rows where either side is null/N/A so the κ is computed on the comparable subset.
    Returns: {n_compared, kappa_unweighted, kappa_linear, mean_judge, mean_gold,
              n_dropped_no_judge, n_dropped_no_gold}.

    Uses sklearn if available; falls back to a hand-rolled implementation otherwise so
    the function works in environments without sklearn.
    """
    import pandas as pd
    merged = judge_df[["id", judge_score_col]].merge(
        gold_df[["prompt_id", gold_score_col]].rename(columns={"prompt_id": "id"}),
        on="id",
        how="outer",
    )
    n_no_judge = merged[judge_score_col].isna().sum()
    n_no_gold = merged[gold_score_col].isna().sum()

    def _coerce(x):
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, str):
            s = x.strip()
            if s.upper() in ("N/A", "NA", "NAN", ""):
                return None
            try:
                return float(s)
            except ValueError:
                return None
        return float(x)

    merged["_j"] = merged[judge_score_col].map(_coerce)
    merged["_g"] = merged[gold_score_col].map(_coerce)
    cmp = merged.dropna(subset=["_j", "_g"])
    if len(cmp) == 0:
        return {"n_compared": 0, "kappa_unweighted": None, "kappa_linear": None,
                "mean_judge": None, "mean_gold": None,
                "n_dropped_no_judge": int(n_no_judge), "n_dropped_no_gold": int(n_no_gold)}

    j = cmp["_j"].tolist()
    g = cmp["_g"].tolist()

    # sklearn's cohen_kappa_score rejects continuous targets; bucket to ints (×4) so the
    # same label set still rounds to discrete categories without losing the 0.25 grid.
    j_disc = [int(round(x * 4)) for x in j]
    g_disc = [int(round(x * 4)) for x in g]

    try:
        from sklearn.metrics import cohen_kappa_score
        kappa_u = cohen_kappa_score(g_disc, j_disc)
        kappa_l = cohen_kappa_score(g_disc, j_disc, weights="linear")
    except ImportError:
        kappa_u = _kappa(g_disc, j_disc, weighted=False)
        kappa_l = _kappa(g_disc, j_disc, weighted=True)

    return {
        "n_compared": len(cmp),
        "kappa_unweighted": float(kappa_u),
        "kappa_linear": float(kappa_l),
        "mean_judge": float(sum(j) / len(j)),
        "mean_gold": float(sum(g) / len(g)),
        "n_dropped_no_judge": int(n_no_judge),
        "n_dropped_no_gold": int(n_no_gold),
    }


def _kappa(a: list, b: list, weighted: bool = False) -> float:
    """Hand-rolled Cohen's κ for the no-sklearn fallback path. Linear weights when weighted=True."""
    cats = sorted(set(a) | set(b))
    n = len(a)
    idx = {c: i for i, c in enumerate(cats)}
    obs = [[0] * len(cats) for _ in cats]
    for x, y in zip(a, b):
        obs[idx[x]][idx[y]] += 1
    row_tot = [sum(r) for r in obs]
    col_tot = [sum(obs[i][j] for i in range(len(cats))) for j in range(len(cats))]

    def w(i, j):
        if not weighted:
            return 0.0 if i == j else 1.0
        max_d = max(1, len(cats) - 1)
        return abs(i - j) / max_d

    obs_disagree = sum(w(i, j) * obs[i][j] for i in range(len(cats)) for j in range(len(cats)))
    exp_disagree = sum(w(i, j) * (row_tot[i] * col_tot[j] / n) for i in range(len(cats)) for j in range(len(cats)))
    if exp_disagree == 0:
        return 1.0
    return 1.0 - obs_disagree / exp_disagree


def correlate_with_embedding(judge_df, records_df,
                             judge_score_col: str = "judge_score",
                             embed_col: str = "relevance_score",
                             id_col_judge: str = "id",
                             id_col_records: str = "prompt_id"):
    """Spearman + Pearson between judge_score and the auction's embedding-relevance.

    Drops rows where either is null. Returns: {n, spearman, pearson}.
    """
    import pandas as pd
    merged = judge_df[[id_col_judge, judge_score_col]].merge(
        records_df[[id_col_records, embed_col]].rename(columns={id_col_records: id_col_judge}),
        on=id_col_judge,
        how="inner",
    )
    cmp = merged.dropna(subset=[judge_score_col, embed_col])
    if len(cmp) < 3:
        return {"n": len(cmp), "spearman": None, "pearson": None}
    try:
        from scipy.stats import spearmanr, pearsonr
        s, _ = spearmanr(cmp[judge_score_col], cmp[embed_col])
        p, _ = pearsonr(cmp[judge_score_col], cmp[embed_col])
        return {"n": len(cmp), "spearman": float(s), "pearson": float(p)}
    except ImportError:
        return {"n": len(cmp), "spearman": None, "pearson": None,
                "_note": "scipy not available; install scipy for correlation."}
