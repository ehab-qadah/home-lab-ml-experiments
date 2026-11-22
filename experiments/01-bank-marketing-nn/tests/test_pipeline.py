"""Fast tests that catch the mistakes this pipeline can actually make.

    python -m pytest -q
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from src import catalog, data, model as model_mod


def _frame(n: int = 200, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "age": rng.integers(18, 70, n),
        "job": rng.choice(["admin.", "technician", "blue-collar"], n),
        "balance": rng.normal(1000, 500, n),
        "duration": rng.integers(10, 600, n),
        "y": rng.choice(["yes", "no"], n),
    })


def test_catalog_refs_are_unique_and_complete():
    refs = [s.kaggle_ref for s in catalog.CATALOG.values()]
    assert len(refs) == len(set(refs))
    for spec in catalog.CATALOG.values():
        assert spec.files, f"{spec.key} declares no files"
        if spec.task == "binary":
            assert spec.positive_label, f"{spec.key} is binary but has no positive_label"


def test_leaky_column_is_excluded_by_default():
    spec = catalog.get("bank-marketing")
    features = data.feature_columns(_frame(), spec.target, spec.leaky)
    assert "duration" not in features
    assert "y" not in features


def test_preprocessor_is_deterministic_and_survives_pickle(tmp_path):
    df = _frame()
    features = data.feature_columns(df, "y", ("duration",))
    pre = data.build_preprocessor(df, features)
    a = pre.fit_transform(df[features])
    b = pre.fit_transform(df[features])
    assert np.allclose(a, b), "preprocessing is not deterministic"

    import joblib
    path = tmp_path / "pre.joblib"
    data.save_preprocessor(pre, None, features, path)
    loaded = joblib.load(path)["preprocessor"]
    assert np.allclose(loaded.transform(df[features]), pre.transform(df[features]))


def test_binary_target_maps_positive_label_to_one():
    df = _frame()
    y, _ = data.encode_target(df["y"], "binary", "yes")
    assert set(np.unique(y)) <= {0.0, 1.0}
    assert y[df["y"].eq("yes").to_numpy()].min() == 1.0


def test_model_forward_shape_and_torchscript_roundtrip(tmp_path):
    net = model_mod.build(in_dim=17, task="binary", n_classes=None, cfg={"hidden": [16, 8]})
    out = net(torch.zeros(4, 17))
    assert out.shape == (4, 1)

    path = tmp_path / "model.pt"
    model_mod.export_torchscript(net, 17, path)
    reloaded = torch.jit.load(str(path))
    with torch.no_grad():
        assert reloaded(torch.zeros(3, 17)).shape == (3, 1)


def test_multiclass_head_has_one_logit_per_class():
    net = model_mod.build(in_dim=5, task="multiclass", n_classes=3, cfg={"hidden": [8]})
    assert net(torch.zeros(2, 5)).shape == (2, 3)
