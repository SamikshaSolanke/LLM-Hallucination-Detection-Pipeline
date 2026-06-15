"""
data_loader.py
--------------
Loads TruthfulQA (MCQ1) and returns a clean, flat pandas DataFrame.

Two modes:
  1. From cache  — reads data/raw/truthfulqa_mc1.csv (fast, no network needed)
  2. From HuggingFace — downloads both configs, joins on question, saves cache

Every downstream module (model_runner, evaluator, analyzer) calls load_dataset()
and gets back the same consistent DataFrame schema.
"""

import ast
import pandas as pd
from datasets import load_dataset as hf_load_dataset
import config


def _flatten_mc1(mc_row: dict, gen_row: dict) -> dict:
    choices = mc_row["mc1_targets"]["choices"]
    labels  = mc_row["mc1_targets"]["labels"]
    correct = [c for c, l in zip(choices, labels) if l == 1]
    return {
        "question"      : mc_row["question"],
        "category"      : gen_row["category"],
        "correct_answer": correct[0] if correct else None,
        "all_choices"   : choices,        
        "num_choices"   : len(choices),
    }


def _is_high_risk(category: str) -> bool:
    cat_lower = category.lower()
    return any(kw in cat_lower for kw in config.HIGH_RISK_KEYWORDS)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    df["high_risk"] = df["category"].apply(_is_high_risk)
    return df


def load_truthfulqa(use_cache: bool = True) -> pd.DataFrame:
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
    return sorted(df["category"].unique().tolist())


def get_questions_by_category(df: pd.DataFrame, category: str) -> pd.DataFrame:
    return df[df["category"] == category].reset_index(drop=True)


def _load_from_huggingface() -> pd.DataFrame:
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
    df_save = df.copy()
    df_save["all_choices"] = df_save["all_choices"].apply(
        lambda choices: " | ".join(choices)
    )
    df_save.to_csv(path, index=False)


def _load_from_csv(path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["all_choices"] = df["all_choices"].apply(
        lambda s: [c.strip() for c in s.split(" | ")] if isinstance(s, str) else []
    )
    return df