#!/usr/bin/env bash
# Train + log to MLflow + publish the serving bundle to s3://models/<dataset>/<run_id>.
# On ehabnet so `mlflow` and `seaweedfs` resolve; as uid 10000 so artifacts stay writable.
source "$(dirname "$0")/_common.sh"
extra=("$@")

docker run --rm --network ehabnet --user 10000:10000 \
  -v "${HOST_DIR}:/app" -w /app --env-file "${PLATFORM_ENV}" \
  -e MLFLOW_TRACKING_URI=http://mlflow:5000 \
  "${IMAGE}" python -m src.train --config "configs/${DATASET}.yaml" "${extra[@]}"

docker run --rm -v "${HOST_DIR}:/app" alpine chown -R 10000:10000 /app/artifacts 2>/dev/null || true
