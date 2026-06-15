"""
evaluator.py
------------
Scores model responses from model_runner.py for hallucination.

Why not use DeepEval's HallucinationMetric directly?
-----------------------------------------------------
DeepEval's HallucinationMetric is designed for free-text RAG outputs — it
needs a "context" passage and checks if the model's response contradicts it.
TruthfulQA MCQ1 is a structured multiple-choice task: the model picks one
letter, we map it to an answer text, and we compare it against a known
correct answer. That comparison is deterministic — no LLM judge needed.

So our evaluation layer does two things:
  1. is_correct      — exact match between model_answer and correct_answer
                       (already computed in model_runner.py, re-verified here)
  2. hallucination   — a question is "hallucinated" when the model is WRONG.
                       We define hallucination score as:
                         0.0 = correct (no hallucination)
                         1.0 = wrong   (hallucinated)
                       This binary score is what DeepEval's framework would
                       produce for a closed-book MCQ task.
"""

import json
import pandas as pd
from pathlib import Path
import config
from model_runner import load_results, results_to_dataframe


def score_result(result: dict) -> dict:
    scored = result.copy()

    if result["error"] is not None or result["model_answer"] is None:
        scored["hallucination_score"] = None
        scored["hallucinated"] = None
        scored["eval_status"] = "skipped"
        return scored

    is_correct = result["model_answer"] == result["correct_answer"]
    h_score = 0.0 if is_correct else 1.0

    scored["hallucination_score"] = h_score
    scored["hallucinated"] = h_score >= config.HALLUCINATION_THRESHOLD
    scored["eval_status"] = "evaluated"

    return scored


def evaluate(results: list[dict]) -> list[dict]:
    scored = [score_result(r) for r in results]

    total = len(scored)
    evaluated = sum(1 for r in scored if r["eval_status"] == "evaluated")
    skipped = sum(1 for r in scored if r["eval_status"] == "skipped")
    hallucinated = sum(
        1 for r in scored
        if r["hallucinated"] is True
    )

    print(f"[evaluator] Total        : {total}")
    print(f"[evaluator] Evaluated    : {evaluated}")
    print(f"[evaluator] Skipped      : {skipped}  (API errors)")
    print(f"[evaluator] Hallucinated : {hallucinated} / {evaluated}  "
          f"({hallucinated / evaluated * 100:.1f}%)" if evaluated else "")

    return scored


def evaluate_both_models() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n── Evaluating Gemini Flash ──")
    flash_results = load_results("flash")
    flash_scored  = evaluate(flash_results)
    df_flash      = results_to_dataframe(flash_scored)
    df_flash["model"] = "gemini-1.5-flash"

    print("\n── Evaluating Gemini Pro ──")
    pro_results = load_results("pro")
    pro_scored  = evaluate(pro_results)
    df_pro      = results_to_dataframe(pro_scored)
    df_pro["model"] = "gemini-1.5-pro"

    return df_flash, df_pro


def save_scored_results(df_flash: pd.DataFrame, df_pro: pd.DataFrame) -> None:
    combined = pd.concat([df_flash, df_pro], ignore_index=True)

    combined["all_choices"] = combined["all_choices"].apply(
        lambda x: x if isinstance(x, list) else []
    )

    records = combined.to_dict(orient="records")
    with open(config.SCORED_RESULTS_JSON, "w") as f:
        json.dump(records, f, indent=2)

    print(f"\n[evaluator] Scored results saved → {config.SCORED_RESULTS_JSON}")


def load_scored_results() -> pd.DataFrame:
    if not config.SCORED_RESULTS_JSON.exists():
        raise FileNotFoundError(
            f"Scored results not found at {config.SCORED_RESULTS_JSON}. "
            "Run evaluate_both_models() and save_scored_results() first."
        )
    with open(config.SCORED_RESULTS_JSON, "r") as f:
        records = json.load(f)
        
    return pd.DataFrame(records)