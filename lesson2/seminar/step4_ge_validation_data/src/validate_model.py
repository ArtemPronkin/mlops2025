import json
import pickle
import sys

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["total_bill", "size"]
TARGET_COLUMN = "high_tip"
METRICS_PATH = "metrics/metrics.json"
ACCURACY_TOLERANCE = 1e-12


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_metrics():
    try:
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"Metrics file not found: {METRICS_PATH}")
        sys.exit(1)


def validate_model():
    params = load_params()
    metrics = load_metrics()

    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    df = pd.read_csv("data/processed/dataset.csv")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    y_pred = model.predict(X_test)
    accuracy = float(accuracy_score(y_test, y_pred))
    recorded_accuracy = float(metrics["accuracy"])

    if abs(accuracy - recorded_accuracy) > ACCURACY_TOLERANCE:
        print(
            "Metric mismatch for accuracy: "
            f"actual={accuracy:.12f}, recorded={recorded_accuracy:.12f}"
        )
        sys.exit(1)

    if int(len(df)) != int(metrics["rows"]):
        print(f"Metric mismatch for rows: actual={len(df)}, recorded={metrics['rows']}")
        sys.exit(1)

    if int(len(y_test)) != int(metrics["test_rows"]):
        print(
            "Metric mismatch for test_rows: "
            f"actual={len(y_test)}, recorded={metrics['test_rows']}"
        )
        sys.exit(1)

    accuracy_min = float(params["accuracy_min"])
    if accuracy < accuracy_min:
        print(
            f"Model validation failed: accuracy={accuracy:.4f} "
            f"is below accuracy_min={accuracy_min:.4f}"
        )
        sys.exit(1)

    print(
        f"Model validation passed: accuracy={accuracy:.4f}, "
        f"accuracy_min={accuracy_min:.4f}"
    )


if __name__ == "__main__":
    validate_model()
