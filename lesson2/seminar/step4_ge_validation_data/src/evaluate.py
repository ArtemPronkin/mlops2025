import json
import os
import pickle

import pandas as pd
import yaml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["total_bill", "size"]
TARGET_COLUMN = "high_tip"
METRICS_PATH = "metrics/metrics.json"


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def evaluate_model():
    params = load_params()

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
    metrics = {
        "accuracy": accuracy,
        "rows": int(len(df)),
        "test_rows": int(len(y_test)),
    }

    os.makedirs("metrics", exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, sort_keys=True)
        f.write("\n")

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Metrics written to {METRICS_PATH}")


if __name__ == "__main__":
    evaluate_model()
