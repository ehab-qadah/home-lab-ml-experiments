"""Verify a live endpoint against the local bundle, over many rows.

A smoke test that returns labels proves the plumbing. This proves the *model*: it scores
the same rows locally from the shipped bundle and from the deployed endpoint, then demands
they agree, and reports the AUC the endpoint actually achieves.

    python -m src.verify --url http://192.168.10.121 --host-header <hdr> \
        --dataset bank-marketing --model-dir artifacts/<run_id> --rows 500
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from . import catalog
from .infer_client import _post


def _local_scores(model_dir: str, rows: pd.DataFrame) -> np.ndarray:
    bundle = joblib.load(f"{model_dir}/preprocessor.joblib")
    model = torch.jit.load(f"{model_dir}/model.pt", map_location="cpu").eval()
    X = bundle["preprocessor"].transform(rows[bundle["input_features"]]).astype(np.float32)
    with torch.no_grad():
        logits = model(torch.from_numpy(X))
    return torch.sigmoid(logits.squeeze(-1)).numpy()


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://192.168.10.121")
    ap.add_argument("--host-header", default=None)
    ap.add_argument("--dataset", default="bank-marketing")
    ap.add_argument("--model-dir", required=True, help="local bundle, e.g. artifacts/<run_id>")
    ap.add_argument("--rows", type=int, default=500)
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--datasets-root", default="datasets")
    args = ap.parse_args(argv)

    spec = catalog.get(args.dataset)
    df = pd.read_parquet(f"{args.datasets_root}/{args.dataset}/test.parquet").head(args.rows)
    X_rows = df.drop(columns=[spec.target])
    truth = (df[spec.target].astype(str) == str(spec.positive_label)).astype(int).to_numpy()

    remote: list[float] = []
    for i in range(0, len(X_rows), args.batch):
        chunk = X_rows.iloc[i:i + args.batch].to_dict("records")
        payload = {"inputs": [{"name": "rows", "shape": [len(chunk), 1], "datatype": "BYTES",
                               "data": [json.dumps(r) for r in chunk]}]}
        try:
            out = _post(f"{args.url}/v2/models/{args.dataset}/infer", payload, args.host_header)
        except urllib.error.HTTPError as exc:
            print(f"HTTP {exc.code}: {exc.read().decode()[:400]}", file=sys.stderr)
            return 1
        remote.extend(json.loads(x)["score"] for x in out["outputs"][0]["data"])

    remote_arr = np.array(remote)
    local_arr = _local_scores(args.model_dir, X_rows)

    print(f"rows            : {len(remote)}")
    print(f"max |Δ| vs local: {np.abs(remote_arr - local_arr).max():.2e}")
    print(f"endpoint  auc   : {roc_auc_score(truth, remote_arr):.4f}  "
          f"pr_auc={average_precision_score(truth, remote_arr):.4f}")
    print(f"local     auc   : {roc_auc_score(truth, local_arr):.4f}")
    acc = accuracy_score(truth, (remote_arr >= 0.5).astype(int))
    print(f"endpoint  acc@0.5: {acc:.4f}  (positives in sample: {int(truth.sum())})")

    ok = np.allclose(remote_arr, local_arr, atol=1e-5)
    print("VERDICT:", "endpoint matches the bundle" if ok else "MISMATCH — serving != trained")
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
