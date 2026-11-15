"""
XGBoost.py - XGBoost for Dynamic Task Offloading in Vehicular Edge Computing
=======================================================================================
Predicts Offloading_Decision (Local / V2V / V2I / V2C) from 7 input features.

INSTALL (run once in Command Prompt or PowerShell):
    py -m pip install numpy pandas scipy scikit-learn xgboost matplotlib joblib

RUN (from the folder that contains this file and the 'dataset' folder):
    py XGBoost.py

Quick test of the code (tiny search):
    PowerShell :  $env:VEC_QUICK="1"; py XGBoost.py
    cmd.exe    :  set VEC_QUICK=1 && py XGBoost.py

Results are written to:  results/XGBoost/

Class imbalance: XGBoost has no class_weight option, so we compare two runs per
feature set - no weighting vs "balanced" per-row sample weights - and the
VALIDATION set decides which is better.

HONEST LIMITATION: the exact MATLAB rules that created the labels are NOT known.
Nothing here assumes the labels are perfectly predictable from the 7 features.
If the labels contain noise or hidden variables, accuracy above 95% may not be
reachable, and this script will simply report the true number.

Rules followed: tuning on TRAIN only (stratified 5-fold CV, macro F1) ->
model/feature-set choice on VALIDATION -> TEST used once at the end.
"""

# =====================================================================
# PART 1 - SHARED SETUP (identical in all four model scripts so that
#          preprocessing and evaluation are consistent everywhere)
# =====================================================================
import json
import os
import sys
import time
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")  # draw plots to files only (no window needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight

SEED = 42  # fixed seed -> reproducible results

# Quick smoke-test mode: set VEC_QUICK=1 to run a tiny search (a minute or so).
#   PowerShell:  $env:VEC_QUICK="1"; py XGBoost.py
QUICK = os.environ.get("VEC_QUICK", "0") == "1"

# ---- Folders ----------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "dataset"
if not DATA_DIR.exists():  # fall back to the folder you ran the command from
    DATA_DIR = Path.cwd() / "dataset"
RESULTS_ROOT = BASE_DIR / "results"

# ---- Column names and classes ----------------------------------------
FEATURES = [
    "Task_Size",           # FLOPs
    "Deadline",            # ms
    "Data_Size",           # MB
    "Latency",             # ms
    "Bandwidth",           # Mbps
    "Processing_Power",    # FLOPs/sec
    "Energy_Consumption",  # W
]
TARGET = "Offloading_Decision"
CLASSES = ["Local", "V2V", "V2I", "V2C"]  # fixed order -> label 0,1,2,3
CLASS_TO_INT = {name: i for i, name in enumerate(CLASSES)}

# ---- Engineered features (physically meaningful) ----------------------
ENG_FEATURES = [
    "Local_Comp_Time_ms",    # Task_Size / Processing_Power * 1000
    "Transmission_Time_ms",  # Data_Size * 8 / Bandwidth * 1000
    "Comm_Time_ms",          # Latency + Transmission_Time_ms
    "Comp_Deadline_Ratio",   # Local_Comp_Time_ms / Deadline
    "Tx_Deadline_Ratio",     # Transmission_Time_ms / Deadline
]
# NOT added on purpose: "Task_Size / Processing_Power" and "Data_Size / Bandwidth".
# They are the same numbers as the two time features above, only divided by
# 1000, so they would be 100% redundant (identical after scaling, and trees
# split on them identically).

FEATURE_SETS = {
    "original": FEATURES,
    "engineered": FEATURES + ENG_FEATURES,
}


# =====================================================================
# PART 2 - LOAD + VALIDATE DATA
# =====================================================================
def load_split(filename, split_name):
    """Read one CSV and check columns, numbers, missing values and labels."""
    path = DATA_DIR / filename
    if not path.exists():
        sys.exit(f"ERROR: cannot find {path}\n"
                 f"Put the CSVs inside a folder called 'dataset' next to this script.")
    df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]

    missing_cols = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing_cols:
        sys.exit(f"ERROR [{split_name}]: missing columns {missing_cols}.\n"
                 f"Columns found: {list(df.columns)}")
    extra = [c for c in df.columns if c not in FEATURES + [TARGET]]
    if extra:
        print(f"  note [{split_name}]: ignoring extra columns {extra}")
    df = df[FEATURES + [TARGET]].copy()

    # Missing values (stop instead of silently guessing)
    n_missing = df.isna().sum()
    if n_missing.sum() > 0:
        sys.exit(f"ERROR [{split_name}]: missing values found:\n{n_missing[n_missing > 0]}")

    # Features must be numeric and finite
    for col in FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    bad = df[FEATURES].isna().sum()
    if bad.sum() > 0:
        sys.exit(f"ERROR [{split_name}]: non-numeric values in:\n{bad[bad > 0]}")
    if not np.isfinite(df[FEATURES].to_numpy()).all():
        sys.exit(f"ERROR [{split_name}]: infinite values found in the features.")

    # Labels: trim spaces, ignore upper/lower case, reject unknown labels
    canon = {c.lower(): c for c in CLASSES}
    cleaned = df[TARGET].astype(str).str.strip().str.lower().map(canon)
    if cleaned.isna().any():
        unknown = df.loc[cleaned.isna(), TARGET].unique()
        sys.exit(f"ERROR [{split_name}]: unknown labels {list(unknown)}. Expected {CLASSES}")
    df[TARGET] = cleaned

    # Warn (do not stop) about values that would break the physics formulas
    for col in ["Processing_Power", "Bandwidth", "Deadline"]:
        n_bad = int((df[col] <= 0).sum())
        if n_bad:
            print(f"  WARNING [{split_name}]: {n_bad} rows have {col} <= 0 "
                  f"(engineered ratios will be capped for those rows)")
    return df


def load_all_data():
    print("Loading data from:", DATA_DIR)
    train = load_split("Training_Dataset.csv", "train")
    val = load_split("Validation_Dataset.csv", "validation")
    test = load_split("Testing_Dataset.csv", "test")
    print(f"  rows -> train {len(train)}, validation {len(val)}, test {len(test)} "
          f"(expected 8400 / 1800 / 1800)")
    for name, d in [("train", train), ("validation", val), ("test", test)]:
        counts = d[TARGET].value_counts().reindex(CLASSES).fillna(0).astype(int)
        print(f"  {name:<10} class counts: " + ", ".join(f"{k}={v}" for k, v in counts.items()))
    # Information only: identical feature rows appearing in two splits
    key = FEATURES
    for a_name, a, b_name, b in [("train", train, "test", test), ("train", train, "validation", val)]:
        overlap = len(a[key].drop_duplicates().merge(b[key].drop_duplicates(), how="inner"))
        if overlap:
            print(f"  note: {overlap} identical feature rows appear in both {a_name} and {b_name}")
    return train, val, test


# =====================================================================
# PART 3 - FEATURE ENGINEERING
# =====================================================================
def _safe_div(numerator, denominator):
    """Divide, but return NaN where the denominator is zero or negative."""
    return numerator / denominator.where(denominator > 0)


def add_engineered_features(df):
    """Adds the physically meaningful features. Uses ONLY the 7 inputs.
    Units: FLOPs / (FLOPs/s) = seconds -> x1000 = ms
           MB * 8 = Mb;  Mb / Mbps = seconds -> x1000 = ms"""
    out = df.copy()
    out["Local_Comp_Time_ms"] = _safe_div(df["Task_Size"], df["Processing_Power"]) * 1000.0
    out["Transmission_Time_ms"] = _safe_div(df["Data_Size"] * 8.0, df["Bandwidth"]) * 1000.0
    # Assumption: Latency (ms) is a delay added on top of the transfer time.
    out["Comm_Time_ms"] = out["Transmission_Time_ms"] + df["Latency"]
    out["Comp_Deadline_Ratio"] = _safe_div(out["Local_Comp_Time_ms"], df["Deadline"])
    out["Tx_Deadline_Ratio"] = _safe_div(out["Transmission_Time_ms"], df["Deadline"])
    return out


def prepare_data(train, val, test):
    """Engineer features for all splits. Any invalid (NaN) engineered value is
    filled with the TRAINING maximum of that column, so no test info is used."""
    train_e, val_e, test_e = (add_engineered_features(d) for d in (train, val, test))
    for col in ENG_FEATURES:
        fill = train_e[col].max()
        fill = 0.0 if pd.isna(fill) else fill
        for name, d in [("train", train_e), ("validation", val_e), ("test", test_e)]:
            n_nan = int(d[col].isna().sum())
            if n_nan:
                print(f"  note: {n_nan} invalid values of {col} in {name} set -> filled with {fill:.4g}")
                d[col] = d[col].fillna(fill)
    print("\nEngineered feature summary (training set):")
    print(train_e[ENG_FEATURES].describe().loc[["min", "50%", "max"]].round(3).to_string())
    return train_e, val_e, test_e


# =====================================================================
# PART 4 - METRICS AND SAVING
# =====================================================================
def to_python(obj):
    """Make numpy types JSON-friendly."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return str(obj)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=to_python)


def compute_metrics(y_true, y_pred):
    labels = list(range(len(CLASSES)))
    mp, mr, mf, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0)
    pp, pr, pf, ps = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    return {
        "n_samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(mp),
        "macro_recall": float(mr),
        "macro_f1": float(mf),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "per_class": {
            CLASSES[i]: {"precision": float(pp[i]), "recall": float(pr[i]),
                         "f1": float(pf[i]), "support": int(ps[i])}
            for i in labels
        },
    }


def save_evaluation(out_dir, split, y_true, y_pred, title):
    """Saves metrics (json), classification report (txt), confusion matrix (csv + png)."""
    labels = list(range(len(CLASSES)))
    metrics = compute_metrics(y_true, y_pred)
    save_json(metrics, out_dir / f"{split}_metrics.json")

    report = classification_report(y_true, y_pred, labels=labels, target_names=CLASSES,
                                   digits=4, zero_division=0)
    with open(out_dir / f"{split}_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(f"{title} - {split} set\n\n{report}\n")

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    pd.DataFrame(cm, index=[f"true_{c}" for c in CLASSES],
                 columns=[f"pred_{c}" for c in CLASSES]).to_csv(
        out_dir / f"{split}_confusion_matrix.csv")

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(labels)
    ax.set_yticks(labels)
    ax.set_xticklabels(CLASSES)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{title} - {split}")
    for i in labels:
        for j in labels:
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(out_dir / f"{split}_confusion_matrix.png", dpi=150)
    plt.close(fig)
    return metrics, report


def save_feature_importance(out_dir, model, columns):
    """Only for tree-based models (Decision Tree, Random Forest, XGBoost)."""
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=columns).sort_values(ascending=False)
    imp.to_csv(out_dir / "feature_importance.csv", header=["importance"])
    print("\nTop feature importances:")
    print(imp.head(8).round(4).to_string())


# =====================================================================
# PART 5 - THE EXPERIMENT RUNNER (same logic for every model)
# =====================================================================
def run_experiment(model_name, make_search, weight_options=("none",)):
    """
    For every feature set (original 7 / engineered) and every weighting option:
      1) tune with RandomizedSearchCV on TRAINING data only (stratified 5-fold, macro F1)
      2) score the tuned model on the VALIDATION set
    Then pick the best candidate by VALIDATION macro F1 (tie -> validation accuracy).
    The TEST set is used exactly once, at the very end, to report the final result.
    """
    t0 = time.time()
    out_dir = RESULTS_ROOT / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    train, val, test = load_all_data()
    train, val, test = prepare_data(train, val, test)
    y_train = train[TARGET].map(CLASS_TO_INT).to_numpy()
    y_val = val[TARGET].map(CLASS_TO_INT).to_numpy()
    y_test = test[TARGET].map(CLASS_TO_INT).to_numpy()

    candidates = []
    for fs_name, cols in FEATURE_SETS.items():
        for weighting in weight_options:
            print(f"\n--- Tuning {model_name} | features = {fs_name} ({len(cols)}) | "
                  f"weighting = {weighting} ---")
            search = make_search(weighting)
            fit_kwargs = {}
            if weighting == "balanced":  # per-row weights (used by XGBoost)
                fit_kwargs["sample_weight"] = compute_sample_weight("balanced", y_train)
            search.fit(train[cols], y_train, **fit_kwargs)

            best_idx = search.best_index_
            cv_mean = float(search.cv_results_["mean_test_score"][best_idx])
            cv_std = float(search.cv_results_["std_test_score"][best_idx])
            model = search.best_estimator_
            val_m = compute_metrics(y_val, model.predict(val[cols]))
            print(f"  CV macro F1 = {cv_mean:.4f} (+/- {cv_std:.4f}) | "
                  f"validation: acc = {val_m['accuracy']:.4f}, macro F1 = {val_m['macro_f1']:.4f}")
            candidates.append({
                "feature_set": fs_name, "weighting": weighting, "columns": cols,
                "model": model, "best_params": search.best_params_,
                "cv_macro_f1": cv_mean, "cv_std": cv_std, "val": val_m,
            })

    # ---- candidate comparison table (validation only) ----
    table = pd.DataFrame([{
        "feature_set": c["feature_set"], "weighting": c["weighting"],
        "cv_macro_f1": c["cv_macro_f1"], "cv_std": c["cv_std"],
        "val_accuracy": c["val"]["accuracy"], "val_macro_f1": c["val"]["macro_f1"],
        "val_V2I_f1": c["val"]["per_class"]["V2I"]["f1"],
        "val_V2V_f1": c["val"]["per_class"]["V2V"]["f1"],
    } for c in candidates])
    table.to_csv(out_dir / "candidate_comparison_validation.csv", index=False)
    print("\nCandidate comparison (validation set):")
    print(table.round(4).to_string(index=False))

    # ---- model selection on VALIDATION data ----
    best = max(candidates, key=lambda c: (round(c["val"]["macro_f1"], 6),
                                          c["val"]["accuracy"], -len(c["columns"])))
    cols, model = best["columns"], best["model"]
    print(f"\nSELECTED (by validation macro F1): features = {best['feature_set']}, "
          f"weighting = {best['weighting']}")

    # ---- save the tuned model + parameters ----
    joblib.dump({"model": model, "feature_columns": cols, "classes": CLASSES,
                 "feature_set": best["feature_set"]}, out_dir / f"{model_name}_best_model.joblib")
    save_json({"model": model_name, "selected_feature_set": best["feature_set"],
               "weighting": best["weighting"], "feature_columns": cols,
               "best_params": best["best_params"], "cv_macro_f1": best["cv_macro_f1"],
               "cv_macro_f1_std": best["cv_std"], "quick_mode": QUICK, "seed": SEED},
              out_dir / "best_params.json")

    # ---- reports: train (to check overfitting), validation, then TEST (once) ----
    train_m, _ = save_evaluation(out_dir, "train", y_train, model.predict(train[cols]), model_name)
    val_m, _ = save_evaluation(out_dir, "validation", y_val, model.predict(val[cols]), model_name)
    final_model = model.steps[-1][1] if hasattr(model, "steps") else model
    save_feature_importance(out_dir, final_model, cols)

    print("\n" + "=" * 62)
    print("FINAL TEST EVALUATION (test set used once, no tuning on it)")
    print("=" * 62)
    test_m, test_report = save_evaluation(out_dir, "test", y_test, model.predict(test[cols]), model_name)
    print(test_report)
    print(f"Accuracy       : train {train_m['accuracy']:.4f} | validation {val_m['accuracy']:.4f} "
          f"| TEST {test_m['accuracy']:.4f}")
    print(f"Macro F1       : train {train_m['macro_f1']:.4f} | validation {val_m['macro_f1']:.4f} "
          f"| TEST {test_m['macro_f1']:.4f}")
    print(f"Macro precision: {test_m['macro_precision']:.4f} | Macro recall: {test_m['macro_recall']:.4f} "
          f"| Weighted F1: {test_m['weighted_f1']:.4f}")
    print(f"V2I F1 = {test_m['per_class']['V2I']['f1']:.4f} | V2V F1 = {test_m['per_class']['V2V']['f1']:.4f}")
    gap = train_m["accuracy"] - test_m["accuracy"]
    print(f"Train-test accuracy gap: {gap:.4f}" + ("  (large gap -> possible overfitting)" if gap > 0.05 else ""))
    if test_m["accuracy"] < 0.95:
        print("NOTE: test accuracy is below 95%. This is reported honestly. The MATLAB labelling "
              "rules are unknown, so the labels may contain noise or depend on information that "
              "is not in the 7 features.")
    print(f"\nAll files saved in: {out_dir}")
    print(f"Total time: {(time.time() - t0) / 60:.1f} minutes")


def make_cv():
    return StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)


# =====================================================================
# PART 6 - XGBOOST-SPECIFIC SEARCH
# =====================================================================
from xgboost import XGBClassifier

N_ITER = 4 if QUICK else 30


def make_search(weighting):
    space = {
        "n_estimators": randint(100, 601),
        "max_depth": randint(3, 11),
        "learning_rate": loguniform(0.02, 0.3),
        "min_child_weight": randint(1, 11),
        "subsample": uniform(0.6, 0.4),          # 0.6 .. 1.0
        "colsample_bytree": uniform(0.6, 0.4),   # 0.6 .. 1.0
        "gamma": uniform(0.0, 2.0),
        "reg_alpha": loguniform(1e-3, 1e1),
        "reg_lambda": loguniform(1e-1, 1e1),
    }
    base = XGBClassifier(
        objective="multi:softprob", tree_method="hist", eval_metric="mlogloss",
        random_state=SEED, n_jobs=1,
    )
    return RandomizedSearchCV(
        base, space, n_iter=N_ITER, scoring="f1_macro", cv=make_cv(),
        n_jobs=-1, random_state=SEED, refit=True, verbose=1,
    )


if __name__ == "__main__":  # this guard is REQUIRED on Windows for parallel jobs
    # "none" = no weighting, "balanced" = balanced sample weights (helps minority classes)
    run_experiment("XGBoost", make_search, weight_options=("none", "balanced"))
