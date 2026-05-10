import os
import pickle

import pandas as pd
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = ["total_bill", "size"]
TARGET_COLUMN = "high_tip"


def load_params():
    with open("params.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def train_model():
    params = load_params()

    df = pd.read_csv("data/processed/dataset.csv")

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, _, y_train, _ = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["seed"]
    )

    model = LogisticRegression(random_state=params["seed"])
    model.fit(X_train, y_train)

    os.makedirs("models", exist_ok=True)
    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)


if __name__ == "__main__":
    train_model()
