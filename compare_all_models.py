"""
compare_all_models.py - Rank SVM, Decision Tree, Random Forest and XGBoost
==========================================================================
Run this AFTER the four model scripts have finished:
    py SVM.py
    py Decision_Tree.py
    py Random_Forest.py
    py XGBoost.py
    py compare_all_models.py

INSTALL (run once):
    py -m pip install numpy pandas scipy scikit-learn xgboost matplotlib joblib

It reads the metrics saved in results/<model>/ (it does NOT retrain or retune
anything) and writes to results/comparison/ :
    model_comparison.csv, comparison_report.md, comparison_chart.png

Ranking rule: rank models by TEST accuracy and by TEST macro F1 (1 = best),
average the two ranks, break ties with macro F1, then accuracy.

HONEST LIMITATION: the MATLAB labelling rules are unknown, so a ceiling below
95% may be a property of the data, not of the models.
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = BASE_DIR / "results"
OUT_DIR = RESULTS_ROOT / "comparison"
MODELS = ["SVM", "Decision_Tree", "Random_Forest", "XGBoost"]
TARGET_ACC = 0.95


def read_json(path):
    if not path.exists():
        sys.exit(f"ERROR: {path} not found. Run the model script for this model first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in MODELS:
        folder = RESULTS_ROOT / name
        train = read_json(folder / "train_metrics.json")
        val = read_json(folder / "validation_metrics.json")
        test = read_json(folder / "test_metrics.json")
        params = read_json(folder / "best_params.json")
        n = test["n_samples"]
        acc = test["accuracy"]
        rows.append({
            "Model": name,
            "Feature_Set": params["selected_feature_set"],
            "Test_Accuracy": acc,
            "Acc_95pct_CI_halfwidth": 1.96 * np.sqrt(acc * (1 - acc) / n),
            "Test_Macro_F1": test["macro_f1"],
            "Test_Macro_Precision": test["macro_precision"],
            "Test_Macro_Recall": test["macro_recall"],
            "Test_Weighted_F1": test["weighted_f1"],
            "Test_V2I_F1": test["per_class"]["V2I"]["f1"],
            "Test_V2V_F1": test["per_class"]["V2V"]["f1"],
            "Val_Accuracy": val["accuracy"],
            "Val_Macro_F1": val["macro_f1"],
            "Train_Accuracy": train["accuracy"],
            "Train_Test_Acc_Gap": train["accuracy"] - acc,
            "Val_Test_Macro_F1_Gap": val["macro_f1"] - test["macro_f1"],
            "CV_Macro_F1": params["cv_macro_f1"],
            "Quick_Mode": params.get("quick_mode", False),
        })
    df = pd.DataFrame(rows)

    # ---- ranking: test accuracy + test macro F1 ----
    df["Rank_Accuracy"] = df["Test_Accuracy"].rank(ascending=False, method="min")
    df["Rank_Macro_F1"] = df["Test_Macro_F1"].rank(ascending=False, method="min")
    df["Avg_Rank"] = (df["Rank_Accuracy"] + df["Rank_Macro_F1"]) / 2
    df = df.sort_values(["Avg_Rank", "Test_Macro_F1", "Test_Accuracy"],
                        ascending=[True, False, False]).reset_index(drop=True)
    df.insert(0, "Final_Rank", range(1, len(df) + 1))
    df.to_csv(OUT_DIR / "model_comparison.csv", index=False)

    # ---- console table ----
    show = df[["Final_Rank", "Model", "Feature_Set", "Test_Accuracy", "Test_Macro_F1",
               "Test_Weighted_F1", "Test_V2I_F1", "Test_V2V_F1", "Train_Test_Acc_Gap"]]
    print("\nMODEL RANKING (test set)")
    print(show.round(4).to_string(index=False))

    # ---- recommendation ----
    best = df.iloc[0]
    lines = []
    lines.append(f"Recommended model: {best['Model']} "
                 f"(test accuracy {best['Test_Accuracy']:.4f}, test macro F1 {best['Test_Macro_F1']:.4f}, "
                 f"features: {best['Feature_Set']}).")
    if len(df) > 1:
        second = df.iloc[1]
        diff = best["Test_Accuracy"] - second["Test_Accuracy"]
        margin = best["Acc_95pct_CI_halfwidth"] + second["Acc_95pct_CI_halfwidth"]
        if diff < margin / 2:
            lines.append(f"The gap to {second['Model']} ({diff * 100:.2f} percentage points) is small compared "
                         f"with the sampling uncertainty of an 1,800-row test set (about +/- "
                         f"{best['Acc_95pct_CI_halfwidth'] * 100:.1f} points), so the top models are "
                         f"practically tied. Prefer the one with the smaller train-test gap, "
                         f"or the simpler / faster one.")
    if best["Test_Accuracy"] >= TARGET_ACC:
        lines.append(f"The {TARGET_ACC:.0%} accuracy target is met on the untouched test set.")
    else:
        lines.append(f"The {TARGET_ACC:.0%} accuracy target is NOT met (best = {best['Test_Accuracy']:.2%}). "
                     f"This is reported as it is. The MATLAB labelling rules are unknown, so the labels may "
                     f"contain noise or depend on information that is not among the 7 features.")
    if best["Train_Test_Acc_Gap"] > 0.05:
        lines.append(f"Warning: train-test accuracy gap is {best['Train_Test_Acc_Gap']:.3f}, "
                     f"which suggests some overfitting.")
    lines.append(f"Minority classes: the recommended model's V2I F1 = {best['Test_V2I_F1']:.4f} and "
                 f"V2V F1 = {best['Test_V2V_F1']:.4f}. Check results/{best['Model']}/test_confusion_matrix.png "
                 f"to see which classes they are confused with.")
    if df["Quick_Mode"].any():
        lines.append("WARNING: at least one model was run in VEC_QUICK mode (tiny search). "
                     "Re-run it normally before using these numbers in your report.")

    print("\nFINAL RECOMMENDATION")
    for line in lines:
        print(" - " + line)

    # ---- markdown report ----
    md = ["# Model comparison (test set)\n",
          "Models were tuned on the training set (stratified 5-fold CV, macro F1) and selected on the "
          "validation set. The test set was used once per model for the final scores below.\n",
          "Limitation: the MATLAB label-generation rules are unknown; no perfect predictability is assumed.\n",
          "## Ranking\n",
          "| Rank | Model | Features | Accuracy | Macro F1 | Weighted F1 | V2I F1 | V2V F1 | Train-Test gap |",
          "|---|---|---|---|---|---|---|---|---|"]
    for _, r in df.iterrows():
        md.append(f"| {int(r['Final_Rank'])} | {r['Model']} | {r['Feature_Set']} | {r['Test_Accuracy']:.4f} | "
                  f"{r['Test_Macro_F1']:.4f} | {r['Test_Weighted_F1']:.4f} | {r['Test_V2I_F1']:.4f} | "
                  f"{r['Test_V2V_F1']:.4f} | {r['Train_Test_Acc_Gap']:.4f} |")
    md.append("\n## Recommendation\n")
    md.extend(f"- {line}" for line in lines)
    (OUT_DIR / "comparison_report.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # ---- chart ----
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    x = np.arange(len(df))
    w = 0.38
    axes[0].bar(x - w / 2, df["Test_Accuracy"], w, label="Accuracy")
    axes[0].bar(x + w / 2, df["Test_Macro_F1"], w, label="Macro F1")
    axes[0].axhline(TARGET_ACC, color="red", linestyle="--", linewidth=1, label="95% target")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(df["Model"], rotation=15)
    axes[0].set_ylim(max(0, df[["Test_Accuracy", "Test_Macro_F1"]].min().min() - 0.1), 1.0)
    axes[0].set_title("Test accuracy and macro F1")
    axes[0].legend()
    axes[1].bar(x - w / 2, df["Test_V2I_F1"], w, label="V2I F1")
    axes[1].bar(x + w / 2, df["Test_V2V_F1"], w, label="V2V F1")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(df["Model"], rotation=15)
    axes[1].set_ylim(max(0, df[["Test_V2I_F1", "Test_V2V_F1"]].min().min() - 0.1), 1.0)
    axes[1].set_title("Minority-class F1 (test)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "comparison_chart.png", dpi=150)
    plt.close(fig)

    print(f"\nSaved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
