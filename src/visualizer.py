"""
visualizer.py
-------------
Generates all charts and the final HTML report from analyzer.py output.

Charts produced
---------------
  1. hallucination_by_category.png  — hallucination rate per category, both models
  2. model_comparison.png           — Flash vs Pro side-by-side overall
  3. delta_chart.png                — per-category delta (where Pro beats Flash most)
  4. high_risk_comparison.png       — high-risk vs standard category breakdown
  5. worst_categories.png           — top 10 worst categories (avg across models)

HTML report
-----------
  hallucination_report.html         — standalone report embedding all charts,
                                      summary tables, and key findings.
                                      Open in any browser, no server needed.
"""

import base64
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import pandas as pd
import numpy as np
import config

sns.set_theme(style="whitegrid", font_scale=1.1)
plt.rcParams["figure.dpi"]    = 130
plt.rcParams["savefig.dpi"]   = 130
plt.rcParams["font.family"]   = "sans-serif"

FLASH_COLOR = "#4C72B0"
PRO_COLOR   = "#DD8452"
RISK_COLORS = {"High-risk": "#e05c5c", "Standard": "#5c87e0"}


def plot_hallucination_by_category(by_category: pd.DataFrame) -> Path:
    categories = sorted(by_category["category"].unique())
    n          = len(categories)
    y          = np.arange(n)
    bar_h      = 0.35

    flash_rates = []
    pro_rates   = []

    for cat in categories:
        subset = by_category[by_category["category"] == cat]
        f = subset[subset["model"].str.contains("flash")]["hallucination_rate"]
        p = subset[subset["model"].str.contains("pro")]["hallucination_rate"]
        flash_rates.append(f.values[0] if len(f) else 0)
        pro_rates.append(p.values[0]   if len(p) else 0)

    fig, ax = plt.subplots(figsize=(13, max(8, n * 0.45)))

    ax.barh(y + bar_h / 2, flash_rates, bar_h, label = "Gemini Flash", color = FLASH_COLOR, alpha = 0.88)
    ax.barh(y - bar_h / 2, pro_rates,   bar_h, label = "Gemini Pro",   color = PRO_COLOR,   alpha = 0.88)

    ax.set_yticks(y)
    ax.set_yticklabels(categories, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Hallucination Rate (%)")
    ax.set_title("Hallucination Rate by Category — Flash vs Pro", fontsize = 14, pad = 14)
    ax.axvline(50, color = "grey", linestyle="--", linewidth = 0.8, alpha = 0.5, label = "50% line")
    ax.legend(loc = "lower right")
    sns.despine(left = True)
    plt.tight_layout()

    out = config.CHARTS_DIR / "hallucination_by_category.png"
    plt.savefig(out, bbox_inches = "tight")
    plt.close()
    print(f"[visualizer] Saved → {out}")

    return out


def plot_model_comparison(overall: pd.DataFrame) -> Path:
    models = overall["model"].tolist()
    acc_rates = overall["accuracy_rate"].tolist()
    hall_rates = overall["hallucination_rate"].tolist()

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize = (9, 5))
    bars1 = ax.bar(x - width / 2, acc_rates,  width, label = "Accuracy (%)", color = FLASH_COLOR, alpha = 0.88)
    bars2 = ax.bar(x + width / 2, hall_rates, width, label = "Hallucination (%)", color = PRO_COLOR,   alpha = 0.88)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("gemini-1.5-", "Gemini ").title() for m in models], fontsize = 11)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Overall Model Performance — Flash vs Pro", fontsize = 14, pad = 14)
    ax.legend()

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha = "center", va = "bottom", fontsize=10)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f"{bar.get_height():.1f}%", ha = "center", va = "bottom", fontsize=10)

    sns.despine()
    plt.tight_layout()

    out = config.CHARTS_DIR / "model_comparison.png"
    plt.savefig(out, bbox_inches = "tight")
    plt.close()
    print(f"[visualizer] Saved → {out}")

    return out


def plot_delta_chart(delta: pd.DataFrame, top_n: int = 20) -> Path:
    df = delta.head(top_n).copy()
    df = df.sort_values("delta")

    colors = [PRO_COLOR if d > 0 else FLASH_COLOR for d in df["delta"]]

    fig, ax = plt.subplots(figsize=(11, max(6, len(df) * 0.45)))
    bars = ax.barh(df["category"], df["delta"], color = colors, alpha = 0.88, edgecolor = "white")

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Hallucination Rate Delta (Flash − Pro)  %")
    ax.set_title(f"Flash vs Pro — Biggest Category Gaps (Top {top_n})", fontsize = 14, pad = 14)

    legend_elements = [
        mpatches.Patch(color = PRO_COLOR,   label = "Pro wins (Flash hallucinates more)"),
        mpatches.Patch(color = FLASH_COLOR, label = "Flash wins (Pro hallucinates more)"),
    ]
    ax.legend(handles=legend_elements, loc = "lower right", fontsize = 9)
    sns.despine(left = True)
    plt.tight_layout()

    out = config.CHARTS_DIR / "delta_chart.png"
    plt.savefig(out, bbox_inches = "tight")
    plt.close()
    print(f"[visualizer] Saved → {out}")

    return out


def plot_high_risk_comparison(high_risk: pd.DataFrame) -> Path:
    models = high_risk["model"].unique()
    risk_groups = ["High-risk", "Standard"]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize = (9, 5))

    for i, rg in enumerate(risk_groups):
        subset = high_risk[high_risk["risk_group"] == rg]
        rates = [
            subset[subset["model"] == m]["avg_hallucination_rate"].values[0]
            if m in subset["model"].values else 0
            for m in models
        ]
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, rates, width, label = rg,
                      color = RISK_COLORS[rg], alpha = 0.88)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
                    f"{bar.get_height():.1f}%", ha = "center", va = "bottom", fontsize = 10)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("gemini-1.5-", "Gemini ").title() for m in models], fontsize = 11)
    ax.set_ylabel("Avg Hallucination Rate (%)")
    ax.set_title("High-Risk vs Standard Categories — Hallucination Rate", fontsize = 13, pad = 14)
    ax.set_ylim(0, 110)
    ax.legend()
    sns.despine()
    plt.tight_layout()

    out = config.CHARTS_DIR / "high_risk_comparison.png"
    plt.savefig(out, bbox_inches = "tight")
    plt.close()
    print(f"[visualizer] Saved → {out}")

    return out


def plot_worst_categories(worst: pd.DataFrame) -> Path:
    df = worst.copy()
    colors = [RISK_COLORS["High-risk"] if r else RISK_COLORS["Standard"]
              for r in df["high_risk"]]

    fig, ax = plt.subplots(figsize = (11, 6))
    bars = ax.barh(df["category"], df["avg_hallucination_rate"],
                   color = colors, alpha = 0.88, edgecolor = "white")

    for bar, val in zip(bars, df["avg_hallucination_rate"]):
        ax.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{val:.1f}%", va = "center", ha = "left", fontsize = 9)

    legend_elements = [
        mpatches.Patch(color = RISK_COLORS["High-risk"], label = "High-risk category"),
        mpatches.Patch(color = RISK_COLORS["Standard"],  label = "Standard category"),
    ]
    ax.legend(handles = legend_elements, loc = "lower right", fontsize = 9)
    ax.invert_yaxis()
    ax.set_xlabel("Avg Hallucination Rate (%) — both models")
    ax.set_title("Top 10 Most Hallucinated Categories", fontsize = 14, pad = 14)
    sns.despine(left = True)
    plt.tight_layout()

    out = config.CHARTS_DIR / "worst_categories.png"
    plt.savefig(out, bbox_inches = "tight")
    plt.close()
    print(f"[visualizer] Saved → {out}")

    return out


def _img_to_base64(path: Path) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _df_to_html_table(df: pd.DataFrame, cols: list[str]) -> str:
    return df[cols].to_html(index = False, border = 0, classes = "summary-table")


def generate_html_report(report: dict, chart_paths: dict) -> Path:
    overall = report["overall"]
    worst = report["worst"]
    delta = report["delta"]
    high_risk = report["high_risk"]

    flash_row = overall[overall["model"].str.contains("flash")].iloc[0]
    pro_row = overall[overall["model"].str.contains("pro")].iloc[0]

    def _img_tag(key: str) -> str:
        if key not in chart_paths or not chart_paths[key].exists():
            return f"<p><em>Chart not found: {key}</em></p>"
        b64 = _img_to_base64(chart_paths[key])
        return f'<img src = "data:image/png;base64,{b64}" style = "width:100%;max-width:900px;">'

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hallucination Detection Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'Segoe UI', Arial, sans-serif; background: #f7f8fa;
          color: #1a1a2e; line-height: 1.6; padding: 0 0 60px 0; }}
  header {{ background: #1a1a2e; color: #fff; padding: 40px 60px; }}
  header h1 {{ font-size: 2rem; font-weight: 700; letter-spacing: -0.5px; }}
  header p  {{ color: #a0aec0; margin-top: 6px; font-size: 0.95rem; }}
  .container {{ max-width: 1000px; margin: 0 auto; padding: 0 30px; }}
  h2 {{ font-size: 1.3rem; font-weight: 700; margin: 48px 0 16px;
        color: #1a1a2e; border-left: 4px solid #4C72B0;
        padding-left: 12px; }}
  h3 {{ font-size: 1rem; font-weight: 600; margin: 24px 0 10px; color: #444; }}
  .kpi-grid {{ display: grid; grid-template-columns: repeat(4, 1fr);
               gap: 16px; margin: 28px 0; }}
  .kpi {{ background: #fff; border-radius: 10px; padding: 20px 18px;
          box-shadow: 0 1px 4px rgba(0,0,0,0.07); text-align: center; }}
  .kpi .val {{ font-size: 2rem; font-weight: 800; color: #4C72B0; }}
  .kpi .val.red {{ color: #e05c5c; }}
  .kpi .val.green {{ color: #38a169; }}
  .kpi .lbl {{ font-size: 0.78rem; color: #718096; margin-top: 4px; }}
  .chart-box {{ background: #fff; border-radius: 10px; padding: 24px;
                box-shadow: 0 1px 4px rgba(0,0,0,0.07); margin-bottom: 28px; }}
  .chart-box img {{ display: block; margin: 0 auto; }}
  .summary-table {{ width: 100%; border-collapse: collapse; font-size: 0.88rem; }}
  .summary-table th {{ background: #1a1a2e; color: #fff; padding: 9px 14px;
                       text-align: left; font-weight: 600; }}
  .summary-table td {{ padding: 8px 14px; border-bottom: 1px solid #e2e8f0; }}
  .summary-table tr:hover td {{ background: #eef2ff; }}
  .tag {{ display: inline-block; padding: 2px 8px; border-radius: 12px;
          font-size: 0.75rem; font-weight: 600; }}
  .tag.risk {{ background: #fed7d7; color: #c53030; }}
  .tag.std  {{ background: #bee3f8; color: #2b6cb0; }}
  footer {{ text-align: center; color: #a0aec0; font-size: 0.8rem; margin-top: 60px; }}
</style>
</head>
<body>
<header>
  <div class="container">
    <h1>Hallucination Detection Pipeline</h1>
    <p>TruthfulQA MCQ1 · 817 questions · 38 categories · Gemini Flash vs Pro</p>
  </div>
</header>

<div class="container">

  <h2>Overall Results</h2>
  <div class="kpi-grid">
    <div class="kpi">
      <div class="val green">{flash_row['accuracy_rate']}%</div>
      <div class="lbl">Flash Accuracy</div>
    </div>
    <div class="kpi">
      <div class="val red">{flash_row['hallucination_rate']}%</div>
      <div class="lbl">Flash Hallucination Rate</div>
    </div>
    <div class="kpi">
      <div class="val green">{pro_row['accuracy_rate']}%</div>
      <div class="lbl">Pro Accuracy</div>
    </div>
    <div class="kpi">
      <div class="val red">{pro_row['hallucination_rate']}%</div>
      <div class="lbl">Pro Hallucination Rate</div>
    </div>
  </div>

  <h2>Model Comparison</h2>
  <div class="chart-box">{_img_tag("model_comparison")}</div>

  <h2>Hallucination Rate by Category</h2>
  <div class="chart-box">{_img_tag("hallucination_by_category")}</div>

  <h2>Top 10 Most Hallucinated Categories</h2>
  <div class="chart-box">{_img_tag("worst_categories")}</div>
  <div class="chart-box">
    <table class="summary-table">
      <tr><th>Category</th><th>Avg Hallucination Rate</th><th>Risk</th></tr>
      {"".join(
          f"<tr><td>{r['category']}</td>"
          f"<td>{r['avg_hallucination_rate']}%</td>"
          f"<td><span class='tag {'risk' if r['high_risk'] else 'std'}'>"
          f"{'High-risk' if r['high_risk'] else 'Standard'}</span></td></tr>"
          for _, r in worst.iterrows()
      )}
    </table>
  </div>

  <h2>Flash vs Pro — Per-Category Gap</h2>
  <div class="chart-box">{_img_tag("delta_chart")}</div>
  <div class="chart-box">
    <table class="summary-table">
      <tr><th>Category</th><th>Flash Rate</th><th>Pro Rate</th><th>Delta</th><th>Winner</th></tr>
      {"".join(
          f"<tr><td>{r['category']}</td>"
          f"<td>{r['flash_rate']}%</td>"
          f"<td>{r['pro_rate']}%</td>"
          f"<td>{r['delta']:+.1f}%</td>"
          f"<td>{r['winner'].title()}</td></tr>"
          for _, r in delta.head(15).iterrows()
      )}
    </table>
  </div>

  <h2>High-Risk vs Standard Categories</h2>
  <div class="chart-box">{_img_tag("high_risk_comparison")}</div>

</div>

<footer>
  <div class="container">
    <p>Hallucination Detection Pipeline · TruthfulQA · Gemini Flash & Pro</p>
  </div>
</footer>
</body>
</html>"""

    out = config.REPORTS_DIR / "hallucination_report.html"
    with open(out, "w", encoding = "utf-8") as f:
        f.write(html)

    print(f"[visualizer] HTML report saved → {out}")
    return out


def generate_all_charts(report: dict) -> dict:
    print("[visualizer] Generating charts...")
    return {
        "hallucination_by_category": plot_hallucination_by_category(report["by_category"]),
        "model_comparison"         : plot_model_comparison(report["overall"]),
        "delta_chart"              : plot_delta_chart(report["delta"]),
        "high_risk_comparison"     : plot_high_risk_comparison(report["high_risk"]),
        "worst_categories"         : plot_worst_categories(report["worst"]),
    }