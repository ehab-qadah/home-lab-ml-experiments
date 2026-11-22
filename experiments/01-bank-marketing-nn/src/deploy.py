"""Render the manifests, push them to the cluster, and wait for Ready.

Runs from the Hermes host (it needs `docker` to read the S3 credential file and `ssh` to
reach the k3s guest). Credentials are piped over stdin and never appear in argv or logs.

    python -m src.deploy --run-id <mlflow_run_id> --image shuqba/torch-nn-exp:<tag>
    python -m src.deploy --run-id <id> --dry-run          # print YAML only
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

TEMPLATE = Path(__file__).resolve().parent.parent / "k8s" / "inferenceservice.yaml"


def _read_s3_credentials(s3_json_host_path: str) -> tuple[str, str]:
    """Read the SeaweedFS identities through a throwaway container (the file is root-only)."""
    host_dir = str(Path(s3_json_host_path).parent)
    out = subprocess.run(
        ["docker", "run", "--rm", "-v", f"{host_dir}:/s:ro", "alpine",
         "cat", f"/s/{Path(s3_json_host_path).name}"],
        capture_output=True, text=True, check=True,
    ).stdout
    identity = json.loads(out)["identities"][0]
    creds = identity["credentials"][0]
    return creds["accessKey"], creds["secretKey"]


def _kubectl(args: list[str], ssh_target: str, ssh_key: str, stdin: str | None = None,
             timeout: int = 300) -> str:
    remote = "sudo KUBECONFIG=/etc/rancher/k3s/k3s.yaml kubectl " + " ".join(args)
    proc = subprocess.run(
        ["ssh", "-i", ssh_key, "-o", "StrictHostKeyChecking=no",
         "-o", "UserKnownHostsFile=/dev/null", ssh_target, remote],
        input=stdin, capture_output=True, text=True, timeout=timeout,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"kubectl {' '.join(args)} failed:\n{proc.stderr.strip()[:800]}")
    return proc.stdout


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="deploy the predictor to KServe")
    ap.add_argument("--dataset", default="bank-marketing")
    ap.add_argument("--run-id", required=True, help="MLflow run id whose bundle to serve")
    ap.add_argument("--image", default="shuqba/torch-nn-exp:latest")
    ap.add_argument("--namespace", default="ml-serving")
    ap.add_argument("--bucket", default="models")
    ap.add_argument("--s3-endpoint", default="http://192.168.10.18:8333")
    ap.add_argument("--s3-json", default="/mnt/cache/ml/seaweedfs/s3.json")
    ap.add_argument("--ssh-target", default="ehab@192.168.10.121")
    ap.add_argument("--ssh-key", default="/opt/data/cache/scratch/ml_node_key")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    manifest = (TEMPLATE.read_text()
                .replace("__NAME__", args.dataset)
                .replace("__NAMESPACE__", args.namespace)
                .replace("__IMAGE__", args.image)
                .replace("__MODEL_S3_URI__", f"s3://{args.bucket}/{args.dataset}/{args.run_id}")
                .replace("__S3_ENDPOINT__", args.s3_endpoint))

    if args.dry_run:
        print(manifest)
        return 0

    access_key, secret_key = _read_s3_credentials(args.s3_json)
    # Hand-rolled YAML: a JSON string is a valid YAML scalar, so json.dumps gives correct
    # quoting without pulling pyyaml into whatever python runs this.
    secret = (
        "apiVersion: v1\nkind: Secret\ntype: Opaque\n"
        f"metadata:\n  name: s3-models-creds\n  namespace: {args.namespace}\n"
        f"stringData:\n  access_key: {json.dumps(access_key)}\n"
        f"  secret_key: {json.dumps(secret_key)}\n"
    )

    _kubectl(["create", "namespace", args.namespace, "--dry-run=client", "-o", "yaml"],
             args.ssh_target, args.ssh_key, timeout=120)
    ns_yaml = _kubectl(["create", "namespace", args.namespace, "--dry-run=client", "-o", "yaml"],
                       args.ssh_target, args.ssh_key, timeout=120)
    _kubectl(["apply", "-f", "-"], args.ssh_target, args.ssh_key, stdin=ns_yaml)
    _kubectl(["apply", "-f", "-"], args.ssh_target, args.ssh_key, stdin=secret)
    print("[deploy] namespace + s3 secret applied (values not echoed)")

    out = _kubectl(["apply", "-f", "-"], args.ssh_target, args.ssh_key, stdin=manifest)
    print("[deploy]", out.strip().replace("\n", " "))

    _kubectl(["wait", "--for=condition=Ready", f"inferenceservice/{args.dataset}",
              "-n", args.namespace, "--timeout=420s"], args.ssh_target, args.ssh_key, timeout=460)
    print(f"[deploy] {args.dataset} is Ready in {args.namespace}")
    url = _kubectl(["get", "inferenceservice", args.dataset, "-n", args.namespace,
                    "-o", "jsonpath={.status.url}"], args.ssh_target, args.ssh_key, timeout=120)
    print(f"[deploy] url={url.strip() or '(none reported)'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
