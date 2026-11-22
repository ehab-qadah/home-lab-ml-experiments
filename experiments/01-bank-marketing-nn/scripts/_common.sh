#!/usr/bin/env bash
# Shared settings for this experiment's scripts.
# NOTE: docker paths are resolved on the *host*, so HOST_DIR is the pool path behind
# /opt/data (which is a bind of /mnt/cache/appdata/hermes), not the container path.
set -euo pipefail

HOST_DIR="${HOST_DIR:-/mnt/cache/appdata/hermes/Herms_work_repo/home-lab-ml-experiments/experiments/01-bank-marketing-nn}"
IMAGE="${IMAGE:-shuqba/torch-nn-exp:latest}"
DATASET="${DATASET:-bank-marketing}"

VM="${VM:-ehab@192.168.10.121}"
VM_KEY="${VM_KEY:-/opt/data/cache/scratch/ml_node_key}"
KUBECTL="ssh -i ${VM_KEY} -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null ${VM} sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl"

KAGGLE_DIR="${KAGGLE_DIR:-/mnt/cache/appdata/ml-platform/kaggle}"
# --env-file is read by the docker CLI (which runs inside the Hermes container), so it needs
# the /host view of the pool path; -v paths are resolved on the host and must NOT have it.
PLATFORM_ENV="${PLATFORM_ENV:-/host/mnt/cache/appdata/hindsight-proxy/ml-platform.env}"
