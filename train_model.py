"""Train and evaluate the AgriRain rainfall classifier from weatherAUS.csv."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
)
from sklearn.pipeline import Pipeline


BASE_DIR = Path(__file__).resolve().parent
DATA_CANDIDATES = [BASE_DIR / "data" / "weatherAUS.csv", BASE_DIR / "weatherAUS.csv"]
MODEL_PATH = BASE_DIR / "rainfall_model.pkl"
FEATURES = ["MinTemp", "MaxTemp", "Rainfall", "Humidity3pm"]


def load_clean_data(data_path):
    """Load required columns, normalize labels, and keep valid dated records."""
    columns = FEATURES + ["RainTomorrow", "Date"]
    try:
        data = pd.read_csv(data_path, usecols=columns)
    except ValueError as error:
        raise SystemExit(f"Dataset is missing required columns: {columns}") from error

    for feature in FEATURES:
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["RainTomorrow"] = data["RainTomorrow"].astype("string").str.strip().str.lower()
    data = data.replace([float("inf"), float("-inf")], pd.NA)
    data = data[data["RainTomorrow"].isin(["yes", "no"])].copy()
    data = data.dropna(subset=["Date"])
    data["target"] = data["RainTomorrow"].map({"no": 0, "yes": 1}).astype("int8")
    data = data.dropna(subset=["target"])
    if len(data) < 100 or data["target"].nunique() != 2:
        raise SystemExit("Dataset has too few valid examples or only one target class.")
    return data


def chronological_split(data):
    """Hold out all observations from the newest 20% of unique dates."""
    dates = sorted(data["Date"].unique())
    split_index = max(1, int(len(dates) * 0.8))
    if split_index >= len(dates):
        raise SystemExit("Dataset does not contain enough distinct dates for a time-based holdout.")
    test_start = dates[split_index]
    train_rows = data["Date"] < test_start
    test_rows = ~train_rows
    X = data[FEATURES]
    y = data["target"]
    X_train, X_test = X.loc[train_rows], X.loc[test_rows]
    y_train, y_test = y.loc[train_rows], y.loc[test_rows]
    if y_train.nunique() < 2 or y_test.nunique() < 2:
        raise SystemExit("Chronological train/test split must contain both target classes.")
    return X_train, X_test, y_train, y_test, test_start


def build_pipeline():
    """Create preprocessing and classifier together to preserve feature handling."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=160, max_depth=16, min_samples_leaf=3,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )),
    ])


def main():
    data_path = next((path for path in DATA_CANDIDATES if path.is_file()), None)
    if data_path is None:
        raise SystemExit(
            "Dataset not found. Download the public Australian Weather Dataset "
            "(weatherAUS.csv) and put it in the data folder. See README.md."
        )
    data = load_clean_data(data_path)
    X_train, X_test, y_train, y_test, test_start = chronological_split(data)
    evaluation_pipeline = build_pipeline()
    evaluation_pipeline.fit(X_train, y_train)
    predictions = evaluation_pipeline.predict(X_test)

    print(f"Dataset: {data_path}")
    print(f"Valid rows: {len(data):,}; train: {len(X_train):,}; test: {len(X_test):,}")
    print(f"Chronological test period: {pd.Timestamp(test_start).date()} to {data['Date'].max().date()}")
    print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
    print("Classification report (0 = No rain, 1 = Rain):")
    print(classification_report(y_test, predictions, target_names=["No rain", "Rain"], zero_division=0))
    print("Confusion matrix [true rows, predicted columns]:")
    print(confusion_matrix(y_test, predictions, labels=[0, 1]))
    # Evaluate on unseen later dates, then refit the deployment model on every
    # valid labelled row so the saved model can use all available observations.
    pipeline = build_pipeline()
    pipeline.fit(data[FEATURES], data["target"])
    joblib.dump({
        "pipeline": pipeline,
        "features": FEATURES,
        "target_mapping": {"no": 0, "yes": 1},
        "training_dataset": data_path.name,
        "evaluation_method": "chronological holdout by date",
        "test_start_date": str(pd.Timestamp(test_start).date()),
        "test_end_date": str(data["Date"].max().date()),
    }, MODEL_PATH, compress=3)
    print(f"Saved model bundle: {MODEL_PATH}")
    print("LIMITATION: WeatherAUS contains Australian observations. Metrics are a held-out Australian test only; this is not validation for Karnataka.")


if __name__ == "__main__":
    main()
