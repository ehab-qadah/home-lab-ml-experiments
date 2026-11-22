"""Pull a dataset from Kaggle, normalise it to parquet, and write a manifest.

    python -m src.pull --list
    python -m src.pull --dataset bank-marketing

Design notes:
- The raw files are hashed and kept, so a training run can be traced to exact bytes.
- When the dataset ships its own split (bank-marketing does) that split is preserved —
  re-splitting would silently change every number anyone has ever reported.
- The separator is detected, not assumed: UCI-derived Kaggle CSVs are frequently
  semicolon-separated and a comma read turns the whole file into one column.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import urllib.error
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from . import catalog


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _read_any(path: Path) -> pd.DataFrame:
    """Read a CSV whose separator we do not trust."""
    best = None
    for sep in (",", ";", "\t"):
        try:
            df = pd.read_csv(path, sep=sep)
        except Exception:
            continue
        if best is None or df.shape[1] > best.shape[1]:
            best = df
    if best is None:
        raise ValueError(f"could not parse {path} with any known separator")
    return best


def _api_token() -> str:
    """Resolve the Kaggle token: env first, then the token file, then legacy kaggle.json.

    Reading the file directly keeps this image free of the `kaggle` CLI, which needs an
    undeclared `six` dependency (kaggle 2.2.5) and would fail at import in a clean env.
    """
    tok = os.environ.get("KAGGLE_API_TOKEN", "").strip()
    if tok:
        return tok
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR") or (Path.home() / ".kaggle"))
    for name in ("access_token", "kaggle.json"):
        path = cfg / name
        if not path.exists():
            continue
        text = path.read_text().strip()
        if name == "kaggle.json":
            data = json.loads(text)
            return data.get("key") or data.get("api_token") or ""
        return text
    raise SystemExit(
        f"[pull] no Kaggle credentials found. Looked in {cfg} for access_token / kaggle.json, "
        "and KAGGLE_API_TOKEN is unset. Mount the token directory and set KAGGLE_CONFIG_DIR."
    )


def _download(spec: catalog.DatasetSpec, dest: Path) -> None:
    """Fetch the dataset zip through the API and extract it.

    Kaggle answers `/datasets/download/<ref>` with a redirect to signed storage; urllib
    follows it, and the Authorization header is not forwarded to the other host, which is
    exactly what the signed URL wants.
    """
    dest.mkdir(parents=True, exist_ok=True)
    token = _api_token()
    url = f"https://www.kaggle.com/api/v1/datasets/download/{spec.kaggle_ref}"
    print(f"[pull] GET {url}")
    req = urllib.request.Request(url, headers={
        "Authorization": f"Bearer {token}",
        # Kaggle 404s urllib's default UA, so borrow a familiar one
        "User-Agent": "curl/8.5.0",
    })
    archive = dest / "dataset.zip"
    try:
        with urllib.request.urlopen(req, timeout=180) as resp, archive.open("wb") as fh:
            shutil.copyfileobj(resp, fh)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:300]
        raise SystemExit(
            f"[pull] download failed: HTTP {exc.code} for {spec.kaggle_ref}\n  {detail}\n"
            "  401/403 -> token invalid or lacks access; 404 -> the ref does not exist."
        ) from exc

    with zipfile.ZipFile(archive) as zf:
        zf.extractall(dest)
    archive.unlink()
    print(f"[pull] extracted {len(list(dest.iterdir()))} entry(ies) into {dest}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="pull + normalise a Kaggle dataset")
    ap.add_argument("--list", action="store_true", help="show the selectable datasets and exit")
    ap.add_argument("--dataset", default=catalog.DEFAULT_DATASET, help="catalog key")
    ap.add_argument("--out", default="datasets", help="datasets root (default: datasets/)")
    ap.add_argument("--force", action="store_true", help="re-download even if present")
    args = ap.parse_args(argv)

    if args.list:
        print(catalog.describe_all())
        return 0

    spec = catalog.get(args.dataset)
    out_dir = Path(args.out) / spec.key
    parquet_train = out_dir / "train.parquet"
    if parquet_train.exists() and not args.force:
        print(f"[pull] {parquet_train} already exists — use --force to re-download")
        return 0

    raw_dir = out_dir / "raw"
    if raw_dir.exists() and args.force:
        shutil.rmtree(raw_dir)
    _download(spec, raw_dir)

    wanted = [raw_dir / f for f in spec.files]
    missing = [p.name for p in wanted if not p.exists()]
    if missing:
        found = sorted(p.name for p in raw_dir.glob("*"))
        raise SystemExit(f"[pull] expected {spec.files}, missing {missing}; found {found}")

    hashes = {p.name: _sha256(p) for p in wanted}
    frames = {p.stem: _read_any(p) for p in wanted}

    # preserve a shipped split; otherwise split here
    if "train" in frames and "test" in frames:
        train_df, test_df = frames["train"], frames["test"]
        split_origin = "shipped"
    else:
        df = pd.concat(frames.values(), ignore_index=True)
        from sklearn.model_selection import train_test_split

        strat = df[spec.target] if spec.task in ("binary", "multiclass") else None
        train_df, test_df = train_test_split(
            df, test_size=spec.test_size, random_state=42, stratify=strat
        )
        split_origin = f"stratified {1 - spec.test_size:.0%}/{spec.test_size:.0%}"

    if spec.target not in train_df.columns:
        raise SystemExit(f"[pull] target {spec.target!r} not in columns: {list(train_df.columns)}")

    out_dir.mkdir(parents=True, exist_ok=True)
    train_df.to_parquet(parquet_train, index=False)
    test_df.to_parquet(out_dir / "test.parquet", index=False)

    manifest = {
        "dataset": spec.key,
        "kaggle_ref": spec.kaggle_ref,
        "pulled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "split_origin": split_origin,
        "target": spec.target,
        "task": spec.task,
        "positive_label": spec.positive_label,
        "leaky_columns": list(spec.leaky),
        "raw_files": {name: {"sha256": h, "bytes": (raw_dir / name).stat().st_size}
                      for name, h in hashes.items()},
        "rows": {"train": int(len(train_df)), "test": int(len(test_df))},
        "columns": list(train_df.columns),
        "target_distribution": {
            str(k): int(v) for k, v in train_df[spec.target].value_counts().items()
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"[pull] {spec.key}: train={len(train_df)} test={len(test_df)} "
          f"({split_origin}), {len(train_df.columns)} columns")
    print(f"[pull] -> {parquet_train}")
    print(f"[pull] target distribution: {manifest['target_distribution']}")
    if spec.leaky:
        print(f"[pull] note: {spec.leaky} leak the label — the config drops them by default")
    return 0


if __name__ == "__main__":
    sys.exit(main())
