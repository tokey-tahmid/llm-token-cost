#!/bin/bash -l
#SBATCH --job-name=ltc-provision
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x-%j.out

# Pull the serving image, build the harness venv, fetch the public-domain
# sources and generate the workloads. Runs on any node with internet access.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/slurm/common.sh"
source configs/versions.env
mkdir -p "$CONTAINER_DIR" logs
OUT=logs/provision-$SLURM_JOB_ID.txt

SIF=$CONTAINER_DIR/$VLLM_SIF
[[ -s $SIF ]] || apptainer pull --force "$SIF" "docker://$VLLM_IMAGE"
echo "$VLLM_IMAGE -> $VLLM_SIF sha256:$(sha256sum "$SIF" | cut -c1-16)" > "$OUT"
if [[ -n "${SGLANG_IMAGE:-}" ]]; then
    SIF=$CONTAINER_DIR/$SGLANG_SIF
    # pull by digest: repo@sha256:... (the tag in SGLANG_IMAGE is for the record)
    [[ -s $SIF ]] || apptainer pull --force "$SIF" "docker://${SGLANG_IMAGE%%:*}@${SGLANG_IMAGE##*@}"
    echo "$SGLANG_IMAGE -> $SGLANG_SIF sha256:$(sha256sum "$SIF" | cut -c1-16)" >> "$OUT"
    run_container "$SIF" python3 -c "import sglang; print('sglang', sglang.__version__)" >> "$OUT" 2>&1 || true
fi

if [[ ! -x $VENV/bin/python ]]; then
    python3 -m venv "$VENV"
    "$VENV/bin/pip" install --quiet --upgrade pip
fi
"$VENV/bin/pip" install --quiet httpx pyyaml numpy pandas matplotlib
"$VENV/bin/python" -c "import sys, httpx; print('venv', sys.version.split()[0], 'httpx', httpx.__version__)" >> "$OUT"

"$VENV/bin/python" workloads/fetch_sources.py > /dev/null
"$VENV/bin/python" workloads/make_workloads.py workloads/sources --out workloads | tail -1 >> "$OUT"
"$VENV/bin/python" - <<'EOF' >> "$OUT"
import json
m = json.load(open("workloads/workloads_meta.json"))
print("workload checksums:", m.get("sha256"))
EOF
cat "$OUT"
