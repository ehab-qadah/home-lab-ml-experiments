"""Call a V2 endpoint with a real row, so a deployment can be checked from a shell.

    python -m src.infer_client --url http://host:8080 --dataset bank-marketing
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

from . import catalog


def _post(url: str, payload: dict, host_header: str | None = None) -> dict:
    headers = {"Content-Type": "application/json", "User-Agent": "curl/8.5.0"}
    if host_header:
        headers["Host"] = host_header          # route through Envoy without needing DNS
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--dataset", default=catalog.DEFAULT_DATASET)
    ap.add_argument("--rows", type=int, default=3)
    ap.add_argument("--datasets-root", default="datasets")
    ap.add_argument("--host-header", default=None,
                    help="Host header to send (routes through the Envoy gateway without DNS)")
    args = ap.parse_args(argv)

    spec = catalog.get(args.dataset)
    df = pd.read_parquet(Path(args.datasets_root) / args.dataset / "test.parquet")
    rows = df.head(args.rows).drop(columns=[spec.target]).to_dict("records")
    truth = df.head(args.rows)[spec.target].tolist()

    payload = {"inputs": [{"name": "rows", "shape": [len(rows), 1], "datatype": "BYTES",
                           "data": [json.dumps(r) for r in rows]}]}
    try:
        out = _post(f"{args.url}/v2/models/{args.dataset}/infer", payload, args.host_header)
    except urllib.error.HTTPError as exc:
        print(f"HTTP {exc.code}: {exc.read().decode()[:400]}", file=sys.stderr)
        return 1
    preds = [json.loads(x) for x in out["outputs"][0]["data"]]
    for pred, actual in zip(preds, truth):
        print(f"  predicted={pred['label']!r:>10} score={pred['score']:.4f}   actual={actual!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
