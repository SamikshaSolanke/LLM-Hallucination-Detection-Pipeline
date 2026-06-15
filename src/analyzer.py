"""
analyzer.py
-----------
Groups scored model results by category and computes hallucination metrics.
This is the "insight" layer — the output of this module is what goes on your
resume ("identified that Gemini Pro hallucinates 40% less on health questions
than Gemini Flash").

All functions take a DataFrame produced by evaluator.load_scored_results()
or evaluator.results_to_dataframe(scored_list) and return clean DataFrames
ready to be passed directly into visualizer.py.
"""

import pandas as pd
import numpy as np
import config


def category_stats(df: pd.DataFrame) -> pd.DataFrame:
    df_eval = df[df["eval_status"] == "evaluated"].copy()

    stats = (
        df_eval
        .groupby(["category", "model"])
        .agg(
            total = ("hallucinated", "count"),
            hallucinated = ("hallucinated", "sum"),
        )
        .reset_index()
    )

    stats["hallucination_rate"] = (
        stats["hallucinated"] / stats["total"] * 100
    ).round(1)

    stats["accuracy_rate"] = (100 - stats["hallucination_rate"]).round(1)

    return stats.sort_values(["category", "model"]).reset_index(drop=True)


def overall_stats(df: pd.DataFrame) -> pd.DataFrame:
    df_eval = df[df["eval_status"] == "evaluated"].copy()

    stats = (
        df_eval
        .groupby("model")
        .agg(
            total = ("is_correct", "count"),
            correct = ("is_correct", "sum"),
            hallucinated = ("hallucinated", "sum"),
        )
        .reset_index()
    )

    stats["accuracy_rate"]     = (stats["correct"]      / stats["total"] * 100).round(1)
    stats["hallucination_rate"]= (stats["hallucinated"] / stats["total"] * 100).round(1)
    stats["skipped"]           = df[df["eval_status"] == "skipped"].groupby("model").size().reindex(stats["model"], fill_value=0).values

    return stats.sort_values("accuracy_rate", ascending=False).reset_index(drop=True)


def model_delta(df: pd.DataFrame) -> pd.DataFrame:
    stats = category_stats(df)

    flash = stats[stats["model"].str.contains("flash")][["category", "hallucination_rate"]].rename(
        columns = {"hallucination_rate": "flash_rate"}
    )
    pro = stats[stats["model"].str.contains("pro")][["category", "hallucination_rate"]].rename(
        columns = {"hallucination_rate": "pro_rate"}
    )

    delta_df = flash.merge(pro, on = "category", how = "inner")
    delta_df["delta"] = (delta_df["flash_rate"] - delta_df["pro_rate"]).round(1)
    delta_df["abs_delta"] = delta_df["delta"].abs()
    delta_df["winner"] = delta_df["delta"].apply(
        lambda d: "pro" if d > 0 else ("flash" if d < 0 else "tie")
    )

    return delta_df.sort_values("abs_delta", ascending=False).reset_index(drop=True)


def worst_categories(df: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    stats = category_stats(df)

    avg = (
        stats
        .groupby("category")["hallucination_rate"]
        .mean()
        .reset_index()
        .rename(columns={"hallucination_rate": "avg_hallucination_rate"})
    )
    avg["avg_hallucination_rate"] = avg["avg_hallucination_rate"].round(1)
    avg["high_risk"] = avg["category"].apply(
        lambda c: any(kw in c.lower() for kw in config.HIGH_RISK_KEYWORDS)
    )

    return avg.sort_values("avg_hallucination_rate", ascending = False).head(n).reset_index(drop = True)


def high_risk_summary(df: pd.DataFrame) -> pd.DataFrame:
    df_eval = df[df["eval_status"] == "evaluated"].copy()
    df_eval["risk_group"] = df_eval["category"].apply(
        lambda c: "High-risk" if any(kw in c.lower() for kw in config.HIGH_RISK_KEYWORDS)
        else "Standard"
    )

    summary = (
        df_eval
        .groupby(["model", "risk_group"])
        .agg(
            avg_hallucination_rate = ("hallucinated", lambda x: round(x.mean() * 100, 1)),
            category_count = ("category", "nunique"),
        )
        .reset_index()
    )

    return summary.sort_values(["model", "risk_group"]).reset_index(drop = True)


def build_report(df: pd.DataFrame) -> dict:
    print("[analyzer] Running all analyses...")
    report = {
        "overall" : overall_stats(df),
        "by_category" : category_stats(df),
        "delta" : model_delta(df),
        "worst" : worst_categories(df, n=10),
        "high_risk" : high_risk_summary(df),
    }

    print("\n── Overall Results ──")
    print(report["overall"][["model", "total", "accuracy_rate", "hallucination_rate"]].to_string(index=False))

    print("\n── Top 5 Most Hallucinated Categories (avg across models) ──")
    print(report["worst"].head(5)[["category", "avg_hallucination_rate", "high_risk"]].to_string(index=False))

    print("\n── Biggest Flash vs Pro Gaps ──")
    print(report["delta"].head(5)[["category", "flash_rate", "pro_rate", "delta", "winner"]].to_string(index=False))

    return report