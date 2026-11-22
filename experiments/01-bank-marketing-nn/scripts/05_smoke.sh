#!/usr/bin/env bash
# Ask the deployed model for predictions on real holdout rows, through Envoy Gateway.
# Uses the gateway IP with a Host header so the test does not depend on DNS.
source "$(dirname "$0")/_common.sh"
host_hdr="${DATASET}-ml-serving.kserve.shuqba.co"
url="${GATEWAY:-http://192.168.10.121}"

# inside the image: it has pandas, and the gateway is reachable from any bridge network
docker run --rm -v "${HOST_DIR}:/app" -w /app --user 10000:10000 "${IMAGE}" \
  python -m src.infer_client --url "${url}" --host-header "${host_hdr}" \
  --dataset "${DATASET}" --rows "${ROWS:-5}"
