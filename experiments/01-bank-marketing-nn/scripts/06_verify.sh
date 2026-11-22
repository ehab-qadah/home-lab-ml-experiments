#!/usr/bin/env bash
# The strong check: score the same holdout rows locally from the bundle and remotely from
# the endpoint, then require the two to agree. A label-only smoke test cannot catch a
# silently inverted class order; this can.
source "$(dirname "$0")/_common.sh"
run_id="${1:?usage: 06_verify.sh <mlflow_run_id>}"
host_hdr="${DATASET}-ml-serving.kserve.shuqba.co"
url="${GATEWAY:-http://192.168.10.121}"

docker run --rm -v "${HOST_DIR}:/app" -w /app --user 10000:10000 "${IMAGE}" \
  python -m src.verify --url "${url}" --host-header "${host_hdr}" --dataset "${DATASET}" \
  --model-dir "artifacts/${run_id}" --rows "${ROWS:-500}"
