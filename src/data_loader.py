# """
# data_loader.py
# --------------
# Loads TruthfulQA (MCQ1) and returns a clean, flat pandas DataFrame.

# Two modes:
#   1. From cache  — reads data/raw/truthfulqa_mc1.csv (fast, no network needed)
#   2. From HuggingFace — downloads both configs, joins on question, saves cache, and returns DataFrame. (slow, network needed)
# """

# import ast
# import pandas as pd
# from datasets import load_dataset as hf_load_dataset
# import config

# def _flatten_mc1(mc_row: dict, gen_row: dict) -> dict:
#     choices = mc_row["mc1_targets"]["choices"]
#     labels  = mc_row["mc1_targets"]["labels"]
#     correct = [c for c, l in zip(choices, labels) if l == 1]
#     return {
#         "question"      : mc_row["question"],
#         "category"      : gen_row["category"],
#         "correct_answer": correct[0] if correct else None,
#         "all_choices"   : choices,       
#         "num_choices"   : len(choices),
#     }


# def _is_high_risk(category: str) -> bool:
#     cat_lower = category.lower()
#     return any(kw in cat_lower for kw in config.HIGH_RISK_KEYWORDS)


# def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
#     """Add columns that are useful downstream but not in the raw data."""
#     df["high_risk"] = df["category"].apply(_is_high_risk)
#     return df


# def load_truthfulqa(use_cache: bool = True) -> pd.DataFrame:
#     if use_cache and config.TRUTHFULQA_CSV.exists():
#         print(f"[data_loader] Loading from cache: {config.TRUTHFULQA_CSV}")
#         df = _load_from_csv(config.TRUTHFULQA_CSV)
#     else:
#         print("[data_loader] Downloading from HuggingFace...")
#         df = _load_from_huggingface()
#         _save_to_csv(df, config.TRUTHFULQA_CSV)
#         print(f"[data_loader] Cache saved → {config.TRUTHFULQA_CSV}")

#     df = _add_derived_columns(df)
#     print(f"[data_loader] Loaded {len(df)} questions across "
#           f"{df['category'].nunique()} categories.")
#     return df


# def get_categories(df: pd.DataFrame) -> list[str]:
#     return sorted(df["category"].unique().tolist())


# def get_questions_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
#     return df[df["category"] == category].reset_index(drop=True)


# def _load_from_huggingface() -> pd.DataFrame:
#     mc_dataset  = hf_load_dataset(
#         config.HF_DATASET_NAME,
#         config.HF_MC_CONFIG,
#         split=config.HF_SPLIT,
#     )
#     gen_dataset = hf_load_dataset(
#         config.HF_DATASET_NAME,
#         config.HF_GEN_CONFIG,
#         split=config.HF_SPLIT,
#     )
#     question_to_gen = {row["question"]: row for row in gen_dataset}
#     missing = [
#         row["question"] for row in mc_dataset
#         if row["question"] not in question_to_gen
#     ]
#     if missing:
#         raise ValueError(
#             f"{len(missing)} questions in multiple_choice config have no match "
#             f"in generation config. First missing: '{missing[0]}'"
#         )
#     records = [
#         _flatten_mc1(mc_row, question_to_gen[mc_row["question"]])
#         for mc_row in mc_dataset
#     ]
#     return pd.DataFrame(records)


# def _save_to_csv(df: pd.DataFrame, path) -> None:
#     df_save = df.copy()
#     df_save["all_choices"] = df_save["all_choices"].apply(
#         lambda choices: " | ".join(choices)
#     )
#     df_save.to_csv(path, index=False)


# def _load_from_csv(path) -> pd.DataFrame:
#     df = pd.read_csv(path)
#     df["all_choices"] = df["all_choices"].apply(
#         lambda s: [c.strip() for c in s.split(" | ")] if isinstance(s, str) else []
#     )
#     return df

"""
data_loader.py
--------------
Loads TruthfulQA (MCQ1) and returns a clean, flat pandas DataFrame.

Two modes:
  1. From cache  — reads data/raw/truthfulqa_mc1.csv (fast, no network needed)
  2. From HuggingFace — downloads both configs, joins on question, saves cache

Every downstream module (model_runner, evaluator, analyzer) calls load_dataset()
and gets back the same consistent DataFrame schema.

DataFrame schema
----------------
question        str   — the question text
category        str   — one of 38 topic categories (from generation config)
correct_answer  str   — the single correct answer text (mc1_targets label==1)
all_choices     list  — all answer choice strings for this question
num_choices     int   — number of choices (varies: 2–8)
high_risk       bool  — True if category is a known hallucination hotspot
"""

import ast
import pandas as pd
from datasets import load_dataset as hf_load_dataset

import config


# ── Internal helpers ───────────────────────────────────────────────────────────

def _flatten_mc1(mc_row: dict, gen_row: dict) -> dict:
    """
    Merge one row from the multiple_choice config with its matching row
    from the generation config to produce a single flat record.
    """
    choices = mc_row["mc1_targets"]["choices"]
    labels  = mc_row["mc1_targets"]["labels"]
    correct = [c for c, l in zip(choices, labels) if l == 1]
    return {
        "question"      : mc_row["question"],
        "category"      : gen_row["category"],
        "correct_answer": correct[0] if correct else None,
        "all_choices"   : choices,        # kept as a Python list
        "num_choices"   : len(choices),
    }


def _is_high_risk(category: str) -> bool:
    """Return True if the category name contains a high-risk keyword."""
    cat_lower = category.lower()
    return any(kw in cat_lower for kw in config.HIGH_RISK_KEYWORDS)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add columns that are useful downstream but not in the raw data."""
    df["high_risk"] = df["category"].apply(_is_high_risk)
    return df


# ── Public API ─────────────────────────────────────────────────────────────────

def load_truthfulqa(use_cache: bool = True) -> pd.DataFrame:
    """
    Load TruthfulQA MCQ1 and return a clean DataFrame.

    Parameters
    ----------
    use_cache : bool
        If True (default) and the CSV cache exists, load from disk.
        If False, always re-download from HuggingFace and overwrite cache.

    Returns
    -------
    pd.DataFrame with columns:
        question, category, correct_answer, all_choices (list),
        num_choices, high_risk
    """
    if use_cache and config.TRUTHFULQA_CSV.exists():
        print(f"[data_loader] Loading from cache: {config.TRUTHFULQA_CSV}")
        df = _load_from_csv(config.TRUTHFULQA_CSV)
    else:
        print("[data_loader] Downloading from HuggingFace...")
        df = _load_from_huggingface()
        _save_to_csv(df, config.TRUTHFULQA_CSV)
        print(f"[data_loader] Cache saved → {config.TRUTHFULQA_CSV}")

    df = _add_derived_columns(df)
    print(f"[data_loader] Loaded {len(df)} questions across "
          f"{df['category'].nunique()} categories.")
    return df


def get_categories(df: pd.DataFrame) -> list[str]:
    """Return sorted list of all unique category names."""
    return sorted(df["category"].unique().tolist())


def get_questions_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    """Return subset of df filtered to a single category."""
    return df[df["category"] == category].reset_index(drop=True)


# ── Private I/O helpers ────────────────────────────────────────────────────────

def _load_from_huggingface() -> pd.DataFrame:
    """
    Download both HuggingFace configs, join on question text, return DataFrame.

    The multiple_choice config has mc1_targets but no category.
    The generation config has category but no mc1_targets.
    We join them on the question string (which is identical in both).
    """
    mc_dataset  = hf_load_dataset(
        config.HF_DATASET_NAME,
        config.HF_MC_CONFIG,
        split=config.HF_SPLIT,
    )
    gen_dataset = hf_load_dataset(
        config.HF_DATASET_NAME,
        config.HF_GEN_CONFIG,
        split=config.HF_SPLIT,
    )

    # Build two lookup dicts:
    #   1. exact match on question text
    #   2. normalised match (lowercase, no spaces) as fallback
    # Needed because some questions have minor typos between the two configs
    # e.g. 'atarot card' in mc vs 'a tarot card' in gen
    question_to_gen   = {row["question"]: row for row in gen_dataset}
    normalised_to_gen = {
        row["question"].lower().replace(" ", ""): row
        for row in gen_dataset
    }

    records = []
    skipped = []
    for mc_row in mc_dataset:
        q = mc_row["question"]

        if q in question_to_gen:
            gen_row = question_to_gen[q]
        else:
            q_norm = q.lower().replace(" ", "")
            if q_norm in normalised_to_gen:
                gen_row = normalised_to_gen[q_norm]
                print(f"[data_loader] Fuzzy matched: '{q}'")
            else:
                skipped.append(q)
                gen_row = {"category": "Unknown"}

        records.append(_flatten_mc1(mc_row, gen_row))

    if skipped:
        print(f"[data_loader] Warning: {len(skipped)} questions could not be "
              f"matched and assigned category='Unknown': {skipped}")

    return pd.DataFrame(records)


def _save_to_csv(df: pd.DataFrame, path) -> None:
    """
    Save DataFrame to CSV. all_choices (a list) is serialised as a
    pipe-separated string so it survives the CSV round-trip cleanly.
    """
    df_save = df.copy()
    df_save["all_choices"] = df_save["all_choices"].apply(
        lambda choices: " | ".join(choices)
    )
    df_save.to_csv(path, index=False)


def _load_from_csv(path) -> pd.DataFrame:
    """
    Read the cached CSV and restore all_choices back to a Python list.
    """
    df = pd.read_csv(path)
    df["all_choices"] = df["all_choices"].apply(
        lambda s: [c.strip() for c in s.split(" | ")] if isinstance(s, str) else []
    )
    return df