"""
main.py
-------
End-to-end entry point for the Hallucination Detection Pipeline.

Run order
---------
  1. Load TruthfulQA dataset           (data_loader)
  2. Run Gemini Flash on all questions  (model_runner)
  3. Run Gemini Pro on all questions    (model_runner)
  4. Score both sets of responses       (evaluator)
  5. Run category-level analysis        (analyzer)
  6. Generate charts + HTML report      (visualizer)

Usage
-----
  # Full run (both models from scratch)
  python main.py

  # Skip model inference — re-run analysis + charts only
  python main.py --skip-inference

  # Run only one model (useful for testing)
  python main.py --model flash
  python main.py --model pro
"""

import argparse
import time
import config
from data_loader  import load_truthfulqa
from model_runner import run_model, load_results, results_to_dataframe
from evaluator    import evaluate, evaluate_both_models, save_scored_results, load_scored_results
from analyzer     import build_report
from visualizer   import generate_all_charts, generate_html_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hallucination Detection Pipeline — TruthfulQA × Gemini"
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip model API calls and load cached responses from data/results/",
    )
    parser.add_argument(
        "--model",
        choices=["flash", "pro", "both"],
        default="both",
        help="Which model(s) to run inference on (default: both)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Re-download TruthfulQA from HuggingFace even if CSV cache exists",
    )
    return parser.parse_args()


def step_load_data(use_cache: bool) -> object:
    print("\n" + "=" * 60)
    print("STEP 1 — Load TruthfulQA Dataset")
    print("=" * 60)
    df = load_truthfulqa(use_cache=use_cache)
    print(f"  {len(df)} questions · {df['category'].nunique()} categories")
    return df


def step_run_inference(df: object, model: str) -> None:
    print("\n" + "=" * 60)
    print("STEP 2 — Run Model Inference")
    print("=" * 60)

    path_map = {
        "flash": config.FLASH_RESPONSES_JSON,
        "pro" : config.PRO_RESPONSES_JSON,
    }

    models_to_run = ["flash", "pro"] if model == "both" else [model]

    for model_key in models_to_run:
        print(f"\n  ── {config.MODELS[model_key]} ──")
        run_model(
            model_key = model_key,
            df = df,
            output_path = path_map[model_key],
            resume = True,       
        )
        if model_key == "flash" and len(models_to_run) > 1:
            print("\n  [main] Pausing 5s between models to avoid rate limits...")
            time.sleep(5)


def step_evaluate() -> object:
    print("\n" + "=" * 60)
    print("STEP 3 — Score Responses (Hallucination Detection)")
    print("=" * 60)

    df_flash, df_pro = evaluate_both_models()
    save_scored_results(df_flash, df_pro)
    return load_scored_results()


def step_analyze(df_scored: object) -> dict:
    print("\n" + "=" * 60)
    print("STEP 4 — Category-Level Analysis")
    print("=" * 60)
    return build_report(df_scored)


def step_visualize(report: dict) -> None:
    print("\n" + "=" * 60)
    print("STEP 5 — Generate Charts & HTML Report")
    print("=" * 60)
    chart_paths = generate_all_charts(report)
    report_path = generate_html_report(report, chart_paths)
    print(f"\n  ✓ Open your report: {report_path}")


def main() -> None:
    args = parse_args()

    print("\n" + "█" * 60)
    print("  HALLUCINATION DETECTION PIPELINE")
    print("  TruthfulQA × Gemini Flash & Pro")
    print("█" * 60)

    start = time.time()

    # Step 1 — Load data
    df = step_load_data(use_cache=not args.no_cache)

    # Step 2 — Model inference (skip if --skip-inference)
    if args.skip_inference:
        print("\n[main] --skip-inference set. Loading cached responses...")
        for model_key, path in [("flash", config.FLASH_RESPONSES_JSON),
                                 ("pro",   config.PRO_RESPONSES_JSON)]:
            if not path.exists():
                raise FileNotFoundError(
                    f"No cached responses found for '{model_key}' at {path}.\n"
                    f"Run without --skip-inference first."
                )
    else:
        step_run_inference(df, args.model)

    # Step 3 — Evaluate
    df_scored = step_evaluate()

    # Step 4 — Analyze
    report = step_analyze(df_scored)

    # Step 5 — Visualize
    step_visualize(report)

    elapsed = time.time() - start
    print(f"\n{'=' * 60}")
    print(f"  Pipeline complete in {elapsed:.1f}s")
    print(f"  Charts  → {config.CHARTS_DIR}")
    print(f"  Report  → {config.REPORTS_DIR / 'hallucination_report.html'}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()