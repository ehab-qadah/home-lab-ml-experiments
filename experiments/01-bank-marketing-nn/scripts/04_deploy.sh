#!/usr/bin/env bash
# Render + apply the InferenceService, then wait for Ready.
# Usage: ./04_deploy.sh <mlflow_run_id>
source "$(dirname "$0")/_common.sh"
run_id="${1:?usage: 04_deploy.sh <mlflow_run_id>}"

python3 -m src.deploy --dataset "${DATASET}" --run-id "${run_id}" --image "${IMAGE}" \
  --ssh-target "${VM}" --ssh-key "${VM_KEY}"
