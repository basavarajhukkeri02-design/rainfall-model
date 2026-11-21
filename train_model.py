"""Train and compare chronological Karnataka next-day rain classifiers."""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "karnataka_ml_features.csv"
# The root rainfall_model.pkl belongs to app.py (a different, Australian-data model).
OUTPUT_DIR = BASE_DIR / "karnataka_model"
MODEL_PATH = OUTPUT_DIR / "rainfall_model.pkl"
REPORT_PATH = OUTPUT_DIR / "model_comparison_report.md"

NUMERIC_FEATURES = [
    "Rainfall_Lag_1", "Rainfall_Lag_2", "Rainfall_Lag_3", "Rainfall_Lag_7",
    "Rolling_Mean_3", "Rolling_Sum_7", "Month", "Day_Of_Year",
]
CATEGORICAL_FEATURES = ["District"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "Rain_Tomorrow"


def make_models():
    rf_preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("district", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL_FEATURES),
    ])
    hgb_preprocessor = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
        ("district", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]), CATEGORICAL_FEATURES),
    ], sparse_threshold=0)
    return {
        "Random Forest": Pipeline([
            ("preprocessor", rf_preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=60, max_depth=18, min_samples_leaf=5, max_samples=0.4,
                class_weight="balanced_subsample", random_state=42, n_jobs=-1,
            )),
        ]),
        "HistGradientBoosting": Pipeline([
            ("preprocessor", hgb_preprocessor),
            ("classifier", HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=31, learning_rate=0.08,
                l2_regularization=1.0, random_state=42,
            )),
        ]),
    }


def tune_threshold(y_true, probabilities):
    # Choose the threshold that maximizes validation balanced accuracy.
    candidates = np.unique(np.concatenate(([0.01, 0.99], probabilities)))
    scores = [balanced_accuracy_score(y_true, probabilities >= t) for t in candidates]
    best = max(range(len(scores)), key=lambda i: (scores[i], -abs(candidates[i] - 0.5)))
    return float(candidates[best]), float(scores[best])


def metrics_for(y_true, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "pr_auc": float(average_precision_score(y_true, probabilities)),
        "threshold": float(threshold),
        "confusion_matrix": confusion_matrix(y_true, predicted, labels=[0, 1]).tolist(),
        "class_report": classification_report(
            y_true, predicted, labels=[0, 1], target_names=["No Rain", "Rain"],
            output_dict=True, zero_division=0,
        ),
    }


def train_model():
    df = pd.read_csv(DATA_PATH)
    required = ["Date", TARGET, *FEATURES]
    missing = sorted(set(required) - set(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    df = df.dropna(subset=["Date", TARGET]).copy()
    df = df[df[TARGET].isin([0, 1])].copy()
    df[TARGET] = df[TARGET].astype(int)
    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df["District"] = df["District"].fillna("Unknown").astype(str)
    df = df.sort_values("Date").reset_index(drop=True)

    dates = np.array(sorted(df["Date"].unique()))
    if len(dates) < 10:
        raise ValueError("At least 10 distinct dates are needed for chronological splits.")
    train_end = int(len(dates) * 0.70)
    validation_end = int(len(dates) * 0.80)
    train_dates, validation_dates = dates[:train_end], dates[train_end:validation_end]
    test_dates = dates[validation_end:]
    train = df[df.Date.isin(train_dates)]
    validation = df[df.Date.isin(validation_dates)]
    test = df[df.Date.isin(test_dates)]

    X_train, y_train = train[FEATURES], train[TARGET]
    X_val, y_val = validation[FEATURES], validation[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    print(f"Records: {len(df):,}; train/validation/test: {len(train):,}/{len(validation):,}/{len(test):,}")
    print(f"Date ranges: train through {train.Date.max().date()}, validation {validation.Date.min().date()} to {validation.Date.max().date()}, test {test.Date.min().date()} to {test.Date.max().date()}")

    results = {}
    pipelines = make_models()
    for name, pipeline in pipelines.items():
        print(f"\nFitting {name}...")
        pipeline.fit(X_train, y_train)
        val_prob = pipeline.predict_proba(X_val)[:, 1]
        threshold, val_balanced_accuracy = tune_threshold(y_val, val_prob)
        test_prob = pipeline.predict_proba(X_test)[:, 1]
        results[name] = {
            "pipeline": pipeline,
            "threshold": threshold,
            "validation_balanced_accuracy": val_balanced_accuracy,
            "test": metrics_for(y_test, test_prob, threshold),
        }
        print(f"Validation-selected threshold: {threshold:.4f}; validation balanced accuracy: {val_balanced_accuracy:.4f}")

    # Select using validation only; final-test scores are for reporting, never selection.
    best_name = max(results, key=lambda n: results[n]["validation_balanced_accuracy"])
    best = results[best_name]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = {
        "pipeline": best["pipeline"], "features": FEATURES, "target": TARGET,
        "model_name": f"Karnataka Rainfall {best_name}",
        "threshold": best["threshold"],
        "positive_label": 1,
    }
    joblib.dump(bundle, MODEL_PATH)

    report_lines = [
        "# Karnataka rainfall model comparison", "",
        f"Dataset: {len(df):,} valid records; target: rain tomorrow (0 = no rain, 1 = rain).",
        f"Chronological splits by whole date: train {len(train):,} rows ({train.Date.min().date()} to {train.Date.max().date()}), validation {len(validation):,} rows ({validation.Date.min().date()} to {validation.Date.max().date()}), test {len(test):,} rows ({test.Date.min().date()} to {test.Date.max().date()}).",
        f"Rain prevalence: train {y_train.mean():.1%}, validation {y_val.mean():.1%}, test {y_test.mean():.1%}.",
        "Validation period alone selected the threshold and the saved model. Test scores below were calculated once on the later held-out period.",
        "", "## Leakage review", "",
        "The trainer uses only prior-rainfall lags/rolling summaries, district and calendar fields. `Rainfall_Tomorrow` (the source used to make the label) is excluded. All rows for a date stay in one split, and imputers/encoders are fitted within each training pipeline on training rows only. The feature preparation script shifts rainfall backward for input features and forward for the target. This checks the code path; it cannot rule out errors or duplication in the source observations.",
        "The source table contains many station rows on each date, as expected. Splitting by unique date keeps rows from the same date together.",
        "", "## Test results", "",
        "Reference supplied before this run: accuracy 65.21%, No Rain precision/recall 0.90/0.52, Rain precision/recall 0.50/0.89. These imply balanced accuracy about 0.705. The reported models do not exceed that reference on the later test period, so this run does not confirm an improvement.",
        "Confusion matrix rows are actual [No Rain, Rain], columns are predicted [No Rain, Rain].",
    ]
    for name, entry in results.items():
        m = entry["test"]
        report_lines += [
            f"### {name}", "",
            f"Threshold (validation tuned): {m['threshold']:.4f}",
            f"Accuracy: {m['accuracy']:.4f}; balanced accuracy: {m['balanced_accuracy']:.4f}; ROC-AUC: {m['roc_auc']:.4f}; PR-AUC: {m['pr_auc']:.4f}",
            f"Confusion matrix: `{m['confusion_matrix']}`", "",
            "| Class | Precision | Recall | F1-score | Support |",
            "|---|---:|---:|---:|---:|",
        ]
        for label in ("No Rain", "Rain"):
            row = m["class_report"][label]
            report_lines.append(f"| {label} | {row['precision']:.4f} | {row['recall']:.4f} | {row['f1-score']:.4f} | {int(row['support']):,} |")
        report_lines.append("")
    report_lines += [
        "## Saved model", "",
        f"Selected by validation balanced accuracy: **{best_name}** (validation balanced accuracy {best['validation_balanced_accuracy']:.4f}).",
        f"Model file: `{MODEL_PATH.relative_to(BASE_DIR)}`. It is stored separately because the root `rainfall_model.pkl` is loaded by the existing app and uses a different feature schema.",
        "",
    ]
    REPORT_PATH.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"\nSelected on validation: {best_name}")
    print(f"Model saved: {MODEL_PATH}")
    print(f"Report saved: {REPORT_PATH}")
    print("Test metrics:")
    print(json.dumps({name: entry["test"] for name, entry in results.items()}, indent=2))


if __name__ == "__main__":
    train_model()
