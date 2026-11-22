"""Frame -> tensors, with a preprocessor that is serialised alongside the model.

The single rule this module enforces: whatever happens to the training frame must be
reproducible from `preprocessor.joblib` alone at serving time. Nothing is allowed to be
computed ad hoc in the predictor.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler


@dataclass
class Split:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]


def feature_columns(df: pd.DataFrame, target: str, drop: tuple[str, ...] | list[str]) -> list[str]:
    dropped = set(drop) | {target}
    return [c for c in df.columns if c not in dropped]


def infer_column_types(df: pd.DataFrame, features: list[str]) -> tuple[list[str], list[str]]:
    """Numeric vs categorical, with the usual 'number stored as text' guard.

    `TotalCharges` in the telco dataset is numeric-as-string with blanks; treating it as
    categorical would one-hot thousands of distinct values, so attempt a coercion first.
    """
    numeric, categorical = [], []
    for col in features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric.append(col)
            continue
        coerced = pd.to_numeric(s, errors="coerce")
        if coerced.notna().mean() >= 0.95:      # mostly numeric -> treat as numeric
            numeric.append(col)
        else:
            categorical.append(col)
    return numeric, categorical


def build_preprocessor(df: pd.DataFrame, features: list[str]) -> ColumnTransformer:
    numeric, categorical = infer_column_types(df, features)
    num_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    cat_pipe = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=0.01)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def encode_target(y: pd.Series, task: str, positive_label: str | None):
    """Return (encoded array, classes/label-encoder or None)."""
    if task == "regression":
        return y.astype(float).to_numpy(), None
    if task == "binary":
        values = y.astype(str).unique()
        if positive_label is None:
            raise ValueError("binary task needs positive_label in the catalog")
        mapping = {v: (1.0 if v == positive_label else 0.0) for v in values}
        unmapped = [v for v in values if v not in (positive_label,)]
        if len(unmapped) > 1:
            raise ValueError(f"binary target has more than two values: {sorted(values)}")
        return y.astype(str).map(mapping).to_numpy(dtype=np.float32), None
    le = LabelEncoder()
    return le.fit_transform(y.astype(str)).astype(np.int64), le


def prepare(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    task: str,
    drop: tuple[str, ...] | list[str],
    positive_label: str | None,
) -> tuple[Split, Split, ColumnTransformer, LabelEncoder | None, list[str]]:
    features = feature_columns(train_df, target, drop)
    pre = build_preprocessor(train_df, features)

    Xtr = pre.fit_transform(train_df[features]).astype(np.float32)
    Xte = pre.transform(test_df[features]).astype(np.float32)

    ytr, le = encode_target(train_df[target], task, positive_label)
    if le is not None:
        yte = le.transform(test_df[target].astype(str)).astype(np.int64)
    else:
        yte = encode_target(test_df[target], task, positive_label)[0]

    names = list(pre.get_feature_names_out())
    return Split(Xtr, ytr, names), Split(Xte, yte, names), pre, le, features


def save_preprocessor(pre: ColumnTransformer, le: LabelEncoder | None, features: list[str],
                      path: Path, extra: dict | None = None) -> None:
    joblib.dump({"preprocessor": pre, "label_encoder": le, "input_features": features,
                 "extra": extra or {}}, path)


def schema_for(features: list[str], df: pd.DataFrame) -> dict:
    """Input contract published with the model, so the predictor can reject bad rows."""
    cols = {}
    for c in features:
        sample = df[c]
        cols[c] = {
            "dtype": str(sample.dtype),
            "numeric": bool(pd.api.types.is_numeric_dtype(sample)),
            "nullable": bool(sample.isna().any()),
            "example": (None if sample.dropna().empty else
                        (sample.dropna().iloc[0].item()
                         if hasattr(sample.dropna().iloc[0], "item") else
                         str(sample.dropna().iloc[0]))),
        }
    return {"features": cols}


def write_schema(schema: dict, path: Path) -> None:
    path.write_text(json.dumps(schema, indent=2))
