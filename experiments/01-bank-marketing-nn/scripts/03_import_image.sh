#!/usr/bin/env bash
# Make the image available to the k3s containerd inside the guest.
# There is no registry on this box: docker save | ssh | k3s ctr images import.
source "$(dirname "$0")/_common.sh"
echo "[import] ${IMAGE} -> ${VM}"
docker save "${IMAGE}" | ssh -i "${VM_KEY}" -o StrictHostKeyChecking=no \
  -o UserKnownHostsFile=/dev/null "${VM}" 'sudo k3s ctr images import -'
${KUBECTL} get nodes -o wide | tail -2
