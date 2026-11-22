"""KServe custom predictor speaking the Open Inference Protocol (V2).

Why a custom container instead of a built-in runtime: the model needs a *tabular*
preprocessing step (impute + scale + one-hot) that must be byte-identical to training.
Shipping the fitted preprocessor next to a TorchScript model is the smallest thing that
keeps serving honest; `sklearn`/`mlserver` runtimes would force a second serialisation
format and a second place for the two to drift.

Payloads
--------
V2, rows as JSON strings (datatype BYTES) — works with any V2 client:

    POST /v2/models/{name}/infer
    {"inputs": [{"name": "rows", "shape": [1, 1], "datatype": "BYTES",
                 "data": ["{\"age\": 44, \"job\": \"admin.\", ...}"]}]}

Convenience endpoint for humans and curl:

    POST /predict
    {"instances": [{"age": 44, "job": "admin.", ...}]}
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import uvicorn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("predictor")

MODEL_DIR = Path(os.environ.get("MODEL_DIR", "/mnt/models"))
MODEL_NAME = os.environ.get("MODEL_NAME", "model")
S3_MODEL_URI = os.environ.get("MODEL_S3_URI", "").strip()

STATE: dict = {}


def _fetch_from_s3(uri: str, dest: Path) -> None:
    """Pull the bundle if it is not already baked into the image."""
    import boto3

    bucket, _, prefix = uri[5:].partition("/")
    client = boto3.client("s3", endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL") or None)
    dest.mkdir(parents=True, exist_ok=True)
    paginator = client.get_paginator("list_objects_v2")
    found = 0
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):].lstrip("/")
            if not rel:
                continue
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            client.download_file(bucket, obj["Key"], str(target))
            found += 1
    if not found:
        raise RuntimeError(f"no objects under {uri} — wrong bucket/prefix?")
    log.info("downloaded %d object(s) from %s", found, uri)


def load() -> None:
    if S3_MODEL_URI and not (MODEL_DIR / "model.pt").exists():
        log.info("fetching model bundle from %s", S3_MODEL_URI)
        _fetch_from_s3(S3_MODEL_URI, MODEL_DIR)

    required = ["model.pt", "preprocessor.joblib", "schema.json", "input_features.json"]
    missing = [f for f in required if not (MODEL_DIR / f).exists()]
    if missing:
        raise RuntimeError(f"model bundle incomplete in {MODEL_DIR}: missing {missing}")

    STATE["model"] = torch.jit.load(str(MODEL_DIR / "model.pt"), map_location="cpu").eval()
    bundle = joblib.load(MODEL_DIR / "preprocessor.joblib")
    STATE["pre"] = bundle["preprocessor"]
    STATE["le"] = bundle["label_encoder"]
    STATE["features"] = bundle["input_features"]
    STATE["schema"] = json.loads((MODEL_DIR / "schema.json").read_text())
    STATE["task"] = STATE["schema"].get("task", "binary")
    STATE["classes"] = STATE["schema"].get("classes", [])
    log.info("loaded model: task=%s features=%d classes=%s",
             STATE["task"], len(STATE["features"]), STATE["classes"])


def _predict_rows(rows: list[dict]) -> list[dict]:
    features = STATE["features"]
    missing = sorted({f for row in rows for f in features if f not in row})
    if missing:
        raise HTTPException(status_code=400,
                            detail=f"missing required feature(s): {missing}")
    frame = pd.DataFrame(rows)[features]
    X = STATE["pre"].transform(frame).astype(np.float32)
    with torch.no_grad():
        logits = STATE["model"](torch.from_numpy(X))

    task = STATE["task"]
    if task == "binary":
        prob = torch.sigmoid(logits.squeeze(-1)).numpy()
        pos, neg = STATE["classes"][1] if len(STATE["classes"]) > 1 else "1", \
                   STATE["classes"][0] if STATE["classes"] else "0"
        return [{"score": float(p), "label": pos if p >= 0.5 else neg,
                 "proba": {pos: float(p), neg: float(1 - p)}} for p in prob]
    if task == "multiclass":
        prob = torch.softmax(logits, dim=-1).numpy()
        classes = STATE["classes"] or [str(i) for i in range(prob.shape[1])]
        out = []
        for row in prob:
            idx = int(row.argmax())
            out.append({"label": str(classes[idx]), "score": float(row[idx]),
                        "proba": {str(c): float(p) for c, p in zip(classes, row)}})
        return out
    pred = logits.squeeze(-1).numpy()
    return [{"score": float(p), "label": float(p)} for p in pred]


app = FastAPI(title="torch-nn-predictor", version="1.0")


@app.on_event("startup")
def _startup() -> None:
    load()


@app.get("/v2/health/live")
@app.get("/v2/health/ready")
def health():
    return JSONResponse(status_code=200, content={})


@app.get("/v2/models/{name}")
def metadata(name: str):
    return {
        "name": name, "versions": ["1"], "platform": "pytorch-torchscript",
        "inputs": [{"name": "rows", "datatype": "BYTES", "shape": [-1]}],
        "outputs": [{"name": "predictions", "datatype": "BYTES", "shape": [-1]}],
    }


@app.post("/v2/models/{name}/infer")
async def infer(name: str, request: Request):
    body = await request.json()
    inputs = body.get("inputs") or []
    if not inputs:
        raise HTTPException(status_code=400, detail="no inputs")
    inp = inputs[0]
    raw = inp.get("data", [])
    try:
        rows = [json.loads(item) if isinstance(item, str) else dict(item) for item in raw]
    except Exception as exc:
        raise HTTPException(status_code=400,
                            detail=f"each data element must be a JSON object string: {exc}") from exc
    preds = _predict_rows(rows)
    return {"model_name": name, "id": body.get("id", ""),
            "outputs": [{"name": "predictions", "shape": [len(preds)], "datatype": "BYTES",
                         "data": [json.dumps(p) for p in preds]}]}


@app.post("/predict")
async def predict(request: Request):
    body = await request.json()
    rows = body.get("instances")
    if not isinstance(rows, list) or not rows:
        raise HTTPException(status_code=400, detail="expected {\"instances\": [ {...} ]}")
    return {"predictions": _predict_rows(rows)}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", "8080")))
