# Step 4 Homework: Data and Model Validation

ML pipeline with DVC, Great Expectations data validation, metrics tracking, and
a fail-fast model quality gate.

## Setup

```bash
poetry install
poetry run pre-commit install
poetry run dvc init --subdir
poetry run dvc remote add -d local ../../.dvcstore
```

## Run

```bash
poetry run pre-commit run --all-files
poetry run dvc repro
poetry run dvc metrics show
```

Pipeline stages:

1. `get_data` downloads the tips dataset.
2. `validate_data` checks raw data with Great Expectations and writes
   `reports/validation/index.html`.
3. `preprocess` creates the `high_tip` target.
4. `train` trains `LogisticRegression`.
5. `evaluate` writes `metrics/metrics.json`.
6. `validate_model` checks that recorded metrics match the model output and
   that `accuracy >= accuracy_min`.

The default data produces accuracy around `0.8367`; the configured minimum is
`accuracy_min: 0.8`.
