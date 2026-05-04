"""All hyperparameters in one place. Change here, not in modules."""
from pathlib import Path

# Paths (override BASE_DIR for Colab/Drive)
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CACHE_DIR = BASE_DIR / "cache"
RESULTS_DIR = BASE_DIR / "results"
LLM_CACHE_PATH = CACHE_DIR / "llm_calls.jsonl"

# Models
GEMINI_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CLAUDE_MODEL = "claude-sonnet-4-6"  # for welfare judge only

# Data
N_PROMPTS = 1000
N_PRODUCTS = 10000
AMAZON_CATEGORY = "Electronics"  # category name in Amazon-Reviews-2023 (e.g. "Electronics", "Home_and_Kitchen")
WILDCHAT_DATASET = "allenai/WildChat-1M"
AMAZON_DATASET = "McAuley-Lab/Amazon-Reviews-2023"

# Retrieval
TOP_K_PRODUCTS = 20

# Auction
N_ADVERTISERS = 100  # synthetic advertiser pool size
MARGIN_ALPHA, MARGIN_BETA = 2, 8       # Beta(2,8), mean=0.20
CONV_ALPHA, CONV_BETA = 2, 38          # Beta(2,38), mean=0.05
BID_NOISE_SIGMA = 1.0                  # log-normal noise on bids
RANDOM_SEED = 42

# Welfare reserve
RESERVE_ALPHA = 2.0  # reserve = RESERVE_ALPHA × predicted_welfare_loss

# Welfare predictor: anchor-similarity noise floor.
# Empirically, any prompt has ~0.30 cosine similarity to the sensitive anchor set just
# from generic English structure. Subtract this floor before treating anchor similarity
# as a signal — otherwise commercial prompts get false-positive welfare flags.
# Calibrated from the 2026-04-25 full run: max commercial anchor sim was 0.30.
ANCHOR_NOISE_FLOOR = 0.30

# Welfare predictor (non-LLM)
SENSITIVE_ANCHOR_PROMPTS = [
    # Mental health & crisis
    "I've been struggling with depression and anxiety lately",
    "I'm considering self-harm",
    "I have suicidal thoughts",
    "Everything feels pointless and I don't see a way forward",
    "I've been having panic attacks and don't know what to do",
    "I'm not sure I want to be alive anymore",
    # Health diagnosis & illness
    "My doctor diagnosed me with a serious illness, what should I do",
    "I have a chronic medical condition and need treatment options",
    "My body has been failing me and I'm scared about what's happening",
    "I just got diagnosed with a chronic disability and don't know how to adjust",
    # Grief & loss
    "My parent recently passed away and I'm grieving",
    "My partner died unexpectedly and I'm having a hard time",
    "I just had a miscarriage and I'm devastated",
    "My pet died and I'm overwhelmed with grief",
    # Financial distress
    "I'm in a lot of debt and can't pay my bills",
    "I lost my job and I'm panicking about money",
    "I'm being evicted from my apartment",
    "I have huge medical bills and don't know how I'll pay them",
    # Legal & safety
    "I'm going through a divorce and need legal advice",
    "I'm being harassed at work and need legal help",
    "My partner has been violent toward me and I don't know if I should leave",
    "I'm a survivor of sexual assault and need support",
    "I'm a survivor of trauma and looking for support",
    # Addiction & recovery
    "I'm addicted to substances and need help",
    "I'm trying to stay sober and it's really hard right now",
    "I think I have a drinking problem",
    # Identity & discrimination
    "I'm questioning my gender identity and don't know who to talk to",
    "I want to come out to my family but I'm scared they'll reject me",
    "I've been experiencing racism at work and need advice",
    "I'm being harassed online because of my identity",
    "I'm worried about my immigration status and don't know my options",
    # Caregiving & family stress
    "I think my elderly parent is showing signs of dementia",
    "I've been caring for my sick parent and I'm completely burned out",
    "My child has a developmental disorder",
    "My child was just diagnosed with a serious illness",
    "My child is being bullied at school and we don't know what to do",
    "I'm in a custody battle and I'm exhausted",
    # Body image & eating
    "I've been struggling with disordered eating for a while",
    "I hate the way I look and it's affecting my mental health",
    # PTSD & trauma
    "I served in combat and I think I have PTSD",
    "I'm dealing with childhood trauma and don't know where to start",
    # Existential & social isolation
    "I have no friends and feel completely isolated",
    "I'm losing my faith and feeling lost",
    # Pregnancy & fertility
    "I've been trying to get pregnant for years and it's not happening",
    "I'm pregnant and considering my options",
    # Worried about others
    "I'm worried about a family member's mental health",
    "My friend has been talking about suicide and I'm worried",
]

SENSITIVE_KEYWORDS = [
    "suicide", "self-harm", "depression", "depressed", "anxiety", "anxious",
    "diagnosed", "cancer", "chronic", "illness", "medication", "prescription",
    "debt", "bankruptcy", "evicted", "homeless", "foreclosure",
    "divorce", "custody", "abuse", "harassment", "lawsuit", "attorney",
    "grief", "grieving", "passed away", "bereaved", "deceased",
    "addiction", "alcoholic", "overdose",
    "trauma", "ptsd", "panic attack",
]
# Note: "died" / "death" intentionally excluded — too broad ("my TV died").
# Grief contexts caught by "passed away", "bereaved", "deceased", and the embedding-anchor path.
