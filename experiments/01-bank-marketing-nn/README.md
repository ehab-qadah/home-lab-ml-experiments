# 01 — Bank Marketing, PyTorch MLP, end to end

Predict whether a bank client subscribes to a term deposit, from pull to a live KServe
endpoint. Everything here runs on the `shuqba` ML platform: MLflow 3 for tracking,
SeaweedFS for artifacts, k3s + KServe (Standard mode) in the `ml-node` guest for serving.

Dataset: [`prakharrathi25/banking-dataset-marketing-targets`](https://www.kaggle.com/datasets/prakharrathi25/banking-dataset-marketing-targets)
(UCI Bank Marketing) — **45,211 train / 4,521 test rows, 17 columns, 11.7% positives**,
shipped split preserved.

## Result (CPU, 30 epochs, ~13 s)

| metric | holdout |
|---|---|
| ROC AUC | **0.8389** |
| PR AUC | 0.4779 |
| accuracy @0.5 | 0.8184 |
| F1 @0.5 | 0.4552 |
| precision / recall | 0.3479 / 0.6583 |

`pos_weight` is why recall dwarfs precision: the loss is weighted ~7.5:1 toward the
minority class, which is the right trade when a false negative costs a lost customer and a
false positive costs a phone call. Raise the decision threshold if precision matters more.

Serving is verified, not assumed: `src/verify.py` scores 500 holdout rows **twice** — once
from the local bundle, once through the deployed endpoint — and requires agreement.

```
max |Δ| vs local : 0.00e+00
endpoint auc     : 0.8638  (500-row subset; full holdout 0.8389)
local auc        : 0.8638
VERDICT          : endpoint matches the bundle
```

Latency through Envoy Gateway: **4–7 ms** per single-row request, `/v2/health/ready` 1.7 ms.

## The pipeline

```
        ┌──────────────┐   ┌──────────────┐   ┌─────────────┐   ┌──────────────────┐
kaggle →│ src/pull.py  │→ │ src/train.py │→  │  MLflow     │→  │ s3://models/...  │
        │ parquet+sha  │   │ TorchScript  │   │ run + metrics│   │ bundle for serve │
        └──────────────┘   └──────────────┘   └─────────────┘   └────────┬─────────┘
                                                                        ↓
                                                          KServe InferenceService
                                                          (custom V2 predictor)
```

| Stage | Command |
|---|---|
| build image | `scripts/00_build.sh` |
| pull dataset | `scripts/01_pull.sh [dataset]` |
| train | `scripts/02_train.sh [--epochs N]` |
| ship image to k3s | `scripts/03_import_image.sh` |
| deploy | `scripts/04_deploy.sh <mlflow_run_id>` |
| smoke (labels) | `scripts/05_smoke.sh` |
| verify (endpoint == bundle) | `scripts/06_verify.sh <run_id>` |
| tests | `docker run --rm -v $PWD:/app -w /app shuqba/torch-nn-exp pytest -q` |

Selecting a different dataset is a flag, not a fork — `--dataset` reads `src/catalog.py`,
which carries `titanic`, `iris`, `wine-quality` and `telco-churn` alongside the default
`bank-marketing`:

```bash
docker run --rm -v "$PWD:/app" -w /app shuqba/torch-nn-exp python -m src.pull --list
```

## Decisions that matter

**`duration` is dropped, deliberately.** It is the length of the last phone call — known
only *after* the call, and the single strongest predictor in the file. Keeping it gives
~0.93 ROC AUC on a model that can only be run after you have already made the call.
The config drops it (`drop_columns: [duration]`); remove that line to reproduce the leaky
number on purpose.

**The shipped train/test split is preserved.** Re-splitting would make every reported
number incomparable with everyone else's work on this dataset. `manifest.json` records
`split_origin: shipped` so it is never a mystery.

**Imbalance is handled with `pos_weight`, not resampling.** ~11.7% positives. Weighting
the loss keeps predicted probabilities usable for ranking; oversampling would distort them
and undersampling would throw away 88% of the rows.

**One image for train and serve.** The predictor needs the same torch build and the same
fitted preprocessor as training. Two images would be two places for `sklearn` versions to
drift, and a drift there is silent — the model just gets slightly wrong inputs.

**A custom V2 predictor, not a built-in runtime.** The input is a mixed-type table that
must go through `impute → scale → one-hot` exactly as in training. `sklearn`/`MLServer`
runtimes would need a second serialisation of that transform. Instead the bundle carries
`model.pt` (TorchScript) + `preprocessor.joblib`, and the predictor applies the same
fitted object.

**The model is fetched from S3 at pod startup, not baked into the image.** Shipping a new
model means changing `MODEL_S3_URI` in the `InferenceService`, not rebuilding and
re-importing an image into the guest's containerd.

**The image is imported, not pulled.** There is no registry on this box: `docker save |
ssh | k3s ctr images import`. That is also why `imagePullPolicy: IfNotPresent` — a pull
would fail.

## Serving contract

V2, one JSON object per row (datatype `BYTES`) — works with any V2 client:

```bash
curl -s http://192.168.10.121/v2/models/bank-marketing/infer \
  -H 'Host: bank-marketing-ml-serving.kserve.shuqba.co' \
  -H 'Content-Type: application/json' \
  -d '{"inputs":[{"name":"rows","shape":[1,1],"datatype":"BYTES",
       "data":["{\"age\":44,\"job\":\"admin.\",\"marital\":\"married\",\"education\":\"tertiary\",\"default\":\"no\",\"balance\":1506,\"housing\":\"yes\",\"loan\":\"no\",\"contact\":\"cellular\",\"day\":5,\"month\":\"may\",\"campaign\":1,\"pdays\":-1,\"previous\":0,\"poutcome\":\"unknown\"}"]}]}'
```

```json
{"outputs":[{"name":"predictions","datatype":"BYTES",
  "data":["{\"score\":0.0812,\"label\":\"no\",\"proba\":{\"yes\":0.0812,\"no\":0.9188}}"]}]}
```

`POST /predict` with `{"instances":[{...}]}` does the same thing for humans. Missing
features are rejected with a `400` naming them, rather than silently imputed — a caller
sending the wrong schema should find out immediately.

## Layout

```
configs/       hyperparameters + which dataset + serving bucket
src/catalog.py dataset options (verified Kaggle refs, targets, leaky columns)
src/pull.py    download → parquet + manifest.json (sha256 of every raw file)
src/data.py    feature typing, ColumnTransformer, target encoding, schema
src/model.py   configurable MLP + TorchScript export
src/train.py   training loop, early stopping, MLflow logging, S3 bundle
src/predictor.py  FastAPI V2 server (KServe custom predictor)
src/deploy.py  renders k8s/inferenceservice.yaml, applies, waits for Ready
src/infer_client.py  call a live endpoint with real holdout rows
k8s/           InferenceService template (placeholders rendered by src/deploy.py)
scripts/       numbered shell wrappers for this box
tests/         fast invariants (determinism, leaky-column exclusion, TorchScript roundtrip)
```

## Gotchas found while building this

- **A binary class order is a contract, and getting it backwards is silent.** The first
  version wrote `classes: [positive, negative]` while the predictor read index 1 as the
  positive class — so every label came out inverted while the scores looked perfectly
  reasonable (0.87 → "no"). The label smoke test surfaced it only because a 0.33 score was
  labelled "yes". `classes` is now `[negative, positive]` for binary and `schema.json`
  states the convention; `src/verify.py` compares scores, which is what actually catches
  this class of bug.
- **`--env-file` is read by the docker CLI, not resolved on the host.** Inside the Hermes
  container it must be `/host/mnt/...`; `-v` paths are the opposite (host-absolute, no
  `/host`). Mixing them gives `no such file or directory` from docker itself.
- **The `kaggle` CLI is not in this image, and `kaggle` 2.2.5 needs an undeclared `six`.**
  `src/pull.py` talks to the API directly with the token file instead — one less dependency
  and no packaging bug to work around.
- **`mlflow.log_artifact` from a container needs the tracking Host allowlist**: the client
  talks to `http://mlflow:5000`, so `mlflow:5000` must be in `MLFLOW_ALLOWED_HOSTS` or every
  call returns `403 Invalid host header` (see the platform runbook §4.9e).
- **There is no registry on this box** — the image reaches the guest via `docker save |
  ssh | k3s ctr images import`, which is also why the `InferenceService` sets
  `imagePullPolicy: IfNotPresent`.
- **`kubectl exec` during a rollout can land on the terminating pod.** Reading the class
  order from the "first" pod returned the *old* ordering and made a correct deploy look
  broken; check pod age and prefer `grep -v Terminating`.
- **Kaggle's `datasets/files` call is the cheap existence check** — it fails fast with a
  clear message on a bad ref, unlike a download that half-succeeds.
- **`TotalCharges`-style columns** (telco) are numeric stored as text with blanks; treating
  them as categorical one-hot-encodes thousands of levels. `data.infer_column_types` coerces
  and only falls back to categorical below 95% numeric.
- **CPU-only training is fine here**: ~22k rows × ~50 features trains in seconds per epoch,
  and the GPU story on this box is a dead card plus an x1 slot (runbook §4.9g). Nothing in
  this pipeline needs CUDA, which is why the image installs CPU wheels.
