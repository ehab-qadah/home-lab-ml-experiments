"""Dataset catalog — the selectable options for this experiment.

Every entry is verified against the Kaggle API (`kaggle datasets files -d <ref>`), so a
typo in a ref fails at pull time with a clear error instead of a mystery 403.

`leaky` lists columns that are only known *after* the outcome of interest is observed.
They are kept out of the model by default (see `drop_columns` in the config) because a
model trained on them scores beautifully and is useless in production.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    kaggle_ref: str
    description: str
    target: str
    task: str                      # binary | multiclass | regression
    positive_label: str | None = None
    files: tuple[str, ...] = ()
    leaky: tuple[str, ...] = ()
    test_size: float = 0.2         # only used when the dataset ships no split
    notes: str = ""


CATALOG: dict[str, DatasetSpec] = {
    "bank-marketing": DatasetSpec(
        key="bank-marketing",
        kaggle_ref="prakharrathi25/banking-dataset-marketing-targets",
        description="UCI Bank Marketing — predict whether a client subscribes to a term deposit.",
        target="y",
        task="binary",
        positive_label="yes",
        files=("train.csv", "test.csv"),
        leaky=("duration",),
        notes=(
            "Ships its own train/test split (22512/7509). `duration` is the call length, only "
            "known after the call ends — it leaks the label, so it is dropped by default. "
            "Class imbalance ~11.7% positives, handled with pos_weight."
        ),
    ),
    "titanic": DatasetSpec(
        key="titanic",
        kaggle_ref="yasserh/titanic-dataset",
        description="Titanic survival — the classic binary classification starter.",
        target="Survived",
        task="binary",
        positive_label="1",
        files=("Titanic-Dataset.csv",),
        leaky=(),
        notes="PassengerId/Name/Ticket are high-cardinality ids; drop them for a clean baseline.",
    ),
    "iris": DatasetSpec(
        key="iris",
        kaggle_ref="uciml/iris",
        description="Iris species — tiny multiclass sanity check for the pipeline.",
        target="Species",
        task="multiclass",
        files=("Iris.csv",),
        leaky=("Id",),
        notes="150 rows: use it to prove the plumbing, never to judge a model.",
    ),
    "wine-quality": DatasetSpec(
        key="wine-quality",
        kaggle_ref="uciml/red-wine-quality-cortez-et-al-2009",
        description="Red wine quality (0-10 score) — multiclass with a heavy middle class.",
        target="quality",
        task="multiclass",
        files=("winequality-red.csv",),
        leaky=(),
        notes="Semicolon separated. Scores 3-8; macro F1 is more honest than accuracy here.",
    ),
    "telco-churn": DatasetSpec(
        key="telco-churn",
        kaggle_ref="blastchar/telco-customer-churn",
        description="Telco customer churn — binary with mixed types and a numeric-as-string column.",
        target="Churn",
        task="binary",
        positive_label="Yes",
        files=("WA_Fn-UseC_-Telco-Customer-Churn.csv",),
        leaky=(),
        notes="`TotalCharges` arrives as a string with blanks; the preprocessor coerces it.",
    ),
}

DEFAULT_DATASET = "bank-marketing"


def get(key: str) -> DatasetSpec:
    if key not in CATALOG:
        raise KeyError(f"unknown dataset {key!r}; choose from: {', '.join(sorted(CATALOG))}")
    return CATALOG[key]


def describe_all() -> str:
    rows = [f"{'key':<16}{'task':<12}{'target':<14}ref"]
    for spec in CATALOG.values():
        rows.append(f"{spec.key:<16}{spec.task:<12}{spec.target:<14}{spec.kaggle_ref}")
    return "\n".join(rows)
