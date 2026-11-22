#!/usr/bin/env bash
# Pull the dataset and normalise it to parquet + manifest.
# Runs as root because the Kaggle token file is 0600 on another uid; ownership of the
# output is handed back to 10000 afterwards so the agent and the trainer can write there.
source "$(dirname "$0")/_common.sh"
ds="${1:-${DATASET}}"

docker run --rm -v "${HOST_DIR}:/app" -v "${KAGGLE_DIR}:/kaggle:ro" \
  -e KAGGLE_CONFIG_DIR=/kaggle -w /app "${IMAGE}" \
  python -m src.pull --dataset "${ds}"

docker run --rm -v "${HOST_DIR}:/app" alpine chown -R 10000:10000 /app/datasets
echo "[pull] datasets owned by uid 10000"
