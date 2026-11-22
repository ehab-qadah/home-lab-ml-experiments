# home-lab-ml-experiments

Experiments, notebooks and training specs for the self-hosted ML platform on `shuqba`.

## What this plugs into

| Piece | Where |
|---|---|
| Tracking + registry | MLflow 3 — `http://mlflow:5000` from inside `ehabnet`, https://mlflow.shuqba.co in a browser (Pocket ID gate) |
| Artifacts | `s3://mlflow` on SeaweedFS, endpoint `http://seaweedfs:8333` |
| Notebooks | https://jupyter.shuqba.co — tokenless, Pocket ID is the gate; work dir `/home/jovyan/work` |
| Datasets | Kaggle token already provisioned on host, VM and cluster; `kagglehub` cache persists at `/mnt/cache/ml/kagglehub-cache` |
| Training | Kubeflow Trainer `TrainJob` on the `ml-node` k3s guest (`192.168.10.121`) |
| Serving | KServe `InferenceService` (Standard mode) behind Envoy Gateway |

## Conventions

- **Never hard-code the tracking URI.** Use `os.environ["MLFLOW_TRACKING_URI"]`, which the notebook
  image already exports. The MLflow UI is behind SSO, so code that logs must run inside `ehabnet`.
- **No dataset bytes in git.** Pull with a script and cache on the pool; see `.gitignore`. Small
  fixture files can be force-added deliberately.
- **No credentials in the repo.** Tokens live under `/mnt/cache/appdata/ml-platform/` on the host and
  are mounted read-only into the consumers.
- **One experiment per directory**, named `<slug>-<yyyymmdd>` if it is a re-run, so MLflow experiment
  names and directory names stay greppable against each other.

## Layout

```
experiments/ # one directory per experiment, each self-contained
```

| # | Experiment | What it shows |
|---|---|---|
| 01 | [`01-bank-marketing-nn`](experiments/01-bank-marketing-nn) | PyTorch MLP on Kaggle Bank Marketing: pull → train → MLflow → KServe V2 endpoint |

Each experiment owns its own `src/`, `configs/`, `k8s/` and numbered `scripts/`, and its
README records the decisions and the traps hit while building it. Dataset bytes are pulled
into that experiment's `datasets/` directory and ignored by git; the manifest records the
sha256 of every raw file so a run traces back to exact bytes.

## Not in this repo

The platform itself — MLflow, S3, Jupyter, k3s, KServe, Trainer — lives in
`Herms_work_repo/ml-platform` and is documented in `/opt/data/docs/shuqba-full-setup.md` (§4.9e–4.9i).
