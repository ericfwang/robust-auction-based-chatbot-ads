# Robust Auction-Based Chatbot Ads

A welfare-aware ad auction for chatbot answers, with empirical robustness
evaluation against advertiser gaming. Final project for OIT 277 (Stanford GSB).

## What it does

1. **Welfare-aware placement.** A VCG auction with a welfare-priced reserve
   that suppresses ads on sensitive prompts — mental health, grief, financial
   distress, medical diagnoses — where commercial intent is misaligned with
   user need. The reserve is set as a multiple of a non-LLM welfare-loss
   estimate so the mechanism degrades gracefully without an LLM in the hot
   path.
2. **Robustness to gaming.** Three advertiser attacks (keyword stuffing,
   fabricated claims, persona/context impersonation) evaluated against two
   defenses (paraphrase robustness, landing-page consistency), reported as a
   3 × 4 inflation-delta matrix. Relevance scores come from a Sonnet 4.6
   judge calibrated against a hand-labeled gold set.

## Layout

```
.
├── auction/            # core package
│   ├── types.py             # Product, Ad, Advertiser, AuctionResult
│   ├── cached_llm.py        # disk-cached Gemini / Claude calls
│   ├── data_pipeline.py     # Amazon Reviews loader, embeddings, retrieval
│   ├── benchmark.py         # benchmark loader (prompts + products)
│   ├── llm_components.py    # slot detector, ad rewriter
│   ├── welfare_predictor.py # non-LLM welfare-loss estimator
│   ├── mechanism.py         # VCG with welfare-priced reserve
│   ├── validator.py         # answer-quality validation
│   ├── relevance_judge.py   # Sonnet 4.6 ad-relevance judge
│   ├── gold_labeling.py     # gold-set labeling and adjudication helpers
│   └── gaming.py            # attack/defense inflation matrix
├── notebooks/
│   ├── 01_smoke_test.ipynb           # end-to-end pipeline check
│   ├── 02_full_run.ipynb             # 1k-prompt benchmark
│   ├── 03_predictor_validation.ipynb # welfare predictor stress test
│   ├── 04_gaming_eval.ipynb          # attack × defense inflation matrix
│   └── 05_gradio_demo.ipynb          # live demo UI
├── config.py           # all hyperparameters
└── requirements.txt
```

## Stack

Python · Gemini 2.5 Flash (in-mechanism LLM calls) · Claude Sonnet 4.6
(welfare and relevance judges) · `sentence-transformers/all-MiniLM-L6-v2`
for retrieval embeddings · HuggingFace datasets (Amazon Reviews 2023,
WildChat-1M) · Gradio for the demo UI.

## Contributors

- **Alex Wurm**
- **Eric Wang**
- **Swetha Srinivasan**
- **Wesley Zhao**

## Note

Data files, evaluation CSVs, and internal team-coordination notes are
excluded from this public repository. The code is published for portfolio
purposes and is not configured to reproduce the full benchmark end-to-end
without those artifacts.
