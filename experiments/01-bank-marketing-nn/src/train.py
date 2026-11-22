"""Train the MLP, log everything to MLflow, and publish a deployable bundle to S3.

    python -m src.train --config configs/bank-marketing.yaml
    python -m src.train --config configs/bank-marketing.yaml --epochs 2      # smoke

The bundle written to `s3://models/<dataset>/<run_id>/` is exactly what the predictor
loads, so "what we trained" and "what is serving" cannot drift: the serving manifest
records the run id.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from . import catalog, data, model as model_mod


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _metrics(task: str, y_true, y_prob, y_pred) -> dict[str, float]:
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_prob))),
            "mae": float(mean_absolute_error(y_true, y_prob)),
        }
    if task == "binary":
        p = y_prob.reshape(-1)
        return {
            "roc_auc": float(roc_auc_score(y_true, p)),
            "pr_auc": float(average_precision_score(y_true, p)),
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "f1": float(f1_score(y_true, y_pred)),
            "precision": float(precision_recall_fscore_support(y_true, y_pred, average="binary")[0]),
            "recall": float(precision_recall_fscore_support(y_true, y_pred, average="binary")[1]),
        }
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted")),
    }


def _predict(model: nn.Module, X: np.ndarray, task: str, batch: int = 4096):
    model.eval()
    outs = []
    with torch.no_grad():
        for i in range(0, len(X), batch):
            xb = torch.from_numpy(X[i:i + batch])
            outs.append(model(xb))
    logits = torch.cat(outs)
    if task == "regression":
        return logits.squeeze(-1).numpy(), None
    if task == "binary":
        prob = torch.sigmoid(logits.squeeze(-1)).numpy()
        return prob, (prob >= 0.5).astype(int)
    prob = torch.softmax(logits, dim=-1).numpy()
    return prob, prob.argmax(axis=1)


def _upload_bundle(bundle: Path, s3_uri: str) -> None:
    """Push the bundle to S3 with boto3, using the platform's SeaweedFS endpoint."""
    import boto3

    assert s3_uri.startswith("s3://"), s3_uri
    bucket, _, prefix = s3_uri[5:].partition("/")
    client = boto3.client("s3", endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"))
    for path in sorted(bundle.rglob("*")):
        if path.is_file():
            key = f"{prefix}/{path.relative_to(bundle).as_posix()}"
            client.upload_file(str(path), bucket, key)
            print(f"[train] uploaded s3://{bucket}/{key}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="train the PyTorch model and log to MLflow")
    ap.add_argument("--config", default="configs/bank-marketing.yaml")
    ap.add_argument("--dataset", default=None, help="override the config's dataset key")
    ap.add_argument("--datasets-root", default="datasets")
    ap.add_argument("--epochs", type=int, default=None, help="override for smoke runs")
    ap.add_argument("--no-upload", action="store_true", help="skip the S3 bundle upload")
    ap.add_argument("--bundle-dir", default="artifacts", help="local bundle output root")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    key = args.dataset or cfg["dataset"]
    spec = catalog.get(key)
    drop = tuple(cfg.get("drop_columns", spec.leaky))
    epochs = args.epochs or int(cfg["training"]["epochs"])
    seed = int(cfg.get("seed", 42))
    _seed_everything(seed)

    data_dir = Path(args.datasets_root) / key
    if not (data_dir / "train.parquet").exists():
        raise SystemExit(f"[train] no data at {data_dir} — run: python -m src.pull --dataset {key}")
    train_df = pd.read_parquet(data_dir / "train.parquet")
    test_df = pd.read_parquet(data_dir / "test.parquet")
    manifest = json.loads((data_dir / "manifest.json").read_text())

    split_tr, split_te, pre, le, features = data.prepare(
        train_df, test_df, spec.target, spec.task, drop, spec.positive_label
    )
    n_classes = int(len(le.classes_)) if le is not None else None
    net = model_mod.build(split_tr.X.shape[1], spec.task, n_classes, cfg["model"])

    # class imbalance: weight the positive class instead of resampling, so the
    # predicted probabilities stay calibrated enough to rank on
    pos_weight = None
    if spec.task == "binary" and cfg["training"].get("pos_weight", "auto") == "auto":
        pos = float(split_tr.y.sum())
        neg = float(len(split_tr.y) - pos)
        if pos > 0:
            pos_weight = torch.tensor([neg / pos])
            print(f"[train] pos_weight={neg / pos:.3f} ({int(pos)} positives / {int(neg)} negatives)")

    loss_fn = (nn.BCEWithLogitsLoss(pos_weight=pos_weight) if spec.task == "binary"
               else nn.CrossEntropyLoss() if spec.task == "multiclass"
               else nn.MSELoss())
    opt = torch.optim.AdamW(net.parameters(), lr=float(cfg["training"]["lr"]),
                            weight_decay=float(cfg["training"].get("weight_decay", 0.0)))
    ds = TensorDataset(torch.from_numpy(split_tr.X), torch.from_numpy(split_tr.y))
    dl = DataLoader(ds, batch_size=int(cfg["training"]["batch_size"]), shuffle=True)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000"))
    mlflow.set_experiment(cfg["experiment"])

    with mlflow.start_run(run_name=cfg.get("run_name")) as run:
        mlflow.log_params({
            "dataset": key, "task": spec.task, "epochs": epochs, "seed": seed,
            "lr": cfg["training"]["lr"], "batch_size": cfg["training"]["batch_size"],
            "hidden": str(cfg["model"].get("hidden")), "dropout": cfg["model"].get("dropout"),
            "dropped_columns": str(list(drop)), "n_features": split_tr.X.shape[1],
            "train_rows": len(train_df), "test_rows": len(test_df),
            "data_sha256": json.dumps(manifest["raw_files"], sort_keys=True),
        })
        mlflow.set_tags({"dataset_ref": spec.kaggle_ref, "split_origin": manifest["split_origin"]})

        best_auc, best_state, patience = -1.0, None, int(cfg["training"].get("patience", 5))
        patience_left = patience
        for epoch in range(1, epochs + 1):
            net.train()
            total = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                out = net(xb)
                loss = loss_fn(out.squeeze(-1), yb.float()) if spec.task in ("binary", "regression") \
                    else loss_fn(out, yb)
                loss.backward()
                opt.step()
                total += float(loss.item()) * len(xb)
            train_loss = total / len(ds)

            val_prob, val_pred = _predict(net, split_te.X, spec.task)
            val = _metrics(spec.task, split_te.y, val_prob, val_pred)
            mlflow.log_metrics({"train_loss": train_loss, **{f"val_{k}": v for k, v in val.items()}},
                               step=epoch)
            watch = val.get("roc_auc", val.get("f1_macro", -val.get("rmse", 0.0)))
            print(f"[train] epoch {epoch:>2}/{epochs} loss={train_loss:.4f} "
                  + " ".join(f"{k}={v:.4f}" for k, v in val.items()))
            if watch > best_auc:
                best_auc, best_state = watch, {k: v.clone() for k, v in net.state_dict().items()}
                patience_left = patience
            else:
                patience_left -= 1
                if patience_left <= 0:
                    print(f"[train] early stop at epoch {epoch} (no improvement in {patience})")
                    break

        if best_state is not None:
            net.load_state_dict(best_state)
        final_prob, final_pred = _predict(net, split_te.X, spec.task)
        final = _metrics(spec.task, split_te.y, final_prob, final_pred)
        mlflow.log_metrics({f"test_{k}": v for k, v in final.items()})
        if spec.task in ("binary", "multiclass"):
            mlflow.log_metric("test_confusion_sum", float(confusion_matrix(split_te.y, final_pred).sum()))
        print("[train] holdout: " + " ".join(f"{k}={v:.4f}" for k, v in final.items()))

        bundle = Path(args.bundle_dir) / run.info.run_id
        bundle.mkdir(parents=True, exist_ok=True)
        model_mod.export_torchscript(net, split_tr.X.shape[1], bundle / "model.pt")
        data.save_preprocessor(pre, le, features, bundle / "preprocessor.joblib", extra={
            "task": spec.task, "target": spec.target, "positive_label": spec.positive_label,
            "drop_columns": list(drop), "dataset": key,
        })
        # class order is a contract: the predictor reads index 1 as the positive class.
        # For binary that means [negative, positive] — writing it the other way round
        # silently inverts every label while leaving the scores looking sane.
        if le is not None:
            classes = [str(c) for c in le.classes_]
        else:
            negatives = sorted({str(v) for v in train_df[spec.target].unique()}
                               - {str(spec.positive_label)})
            classes = [negatives[0] if negatives else "0", str(spec.positive_label)]
        schema = data.schema_for(features, train_df)
        schema.update({"task": spec.task, "target": spec.target,
                       "positive_label": spec.positive_label,
                       "classes": classes,
                       "class_order": "binary: [negative, positive]; multiclass: sorted labels",
                       "feature_names": split_tr.feature_names})
        (bundle / "schema.json").write_text(json.dumps(schema, indent=2))
        (bundle / "metrics.json").write_text(json.dumps({"test": final, "run_id": run.info.run_id}, indent=2))
        (bundle / "manifest.json").write_text(json.dumps(manifest, indent=2))
        (bundle / "config_snapshot.yaml").write_text(yaml.safe_dump(cfg))
        (bundle / "input_features.json").write_text(json.dumps(features, indent=2))

        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
        s3_uri = f"s3://{cfg['serving']['s3_bucket']}/{key}/{run.info.run_id}"
        if not args.no_upload:
            _upload_bundle(bundle, s3_uri)
        print(f"[train] run_id={run.info.run_id}")
        print(f"[train] bundle={bundle}")
        print(f"[train] serving uri={s3_uri}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
