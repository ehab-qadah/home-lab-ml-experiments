#!/usr/bin/env bash
# Build the experiment image (pull + train + serve all use it).
source "$(dirname "$0")/_common.sh"
docker build -t "${IMAGE}" "${HOST_DIR}"
echo "[build] ${IMAGE}"
docker image inspect "${IMAGE}" --format '  size={{.Size}} created={{.Created}}'
