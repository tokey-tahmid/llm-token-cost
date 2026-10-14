#!/bin/bash -l
#SBATCH --job-name=ltc-sweep
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
# (memory follows CPUs on this partition: request all of them so weights staged in node tmpfs fit)
#SBATCH --time=01:30:00
#SBATCH --output=logs/%x-%j.out

# One serving configuration, full sweep: chat and agent workloads, concurrency
# levels, 3 reps, warm-up, unique prompts, GPU power sampled every second.
# Configured through the environment (see scripts/submit_sweeps.sh):
#   LABEL MODEL REVISION TP GPUS VLLM_ARGS EXTRA_BODY ALLOW IGNORE
#   CHAT_CONC AGENT_CONC REPS
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/slurm/common.sh"
source configs/versions.env
SIF_VLLM=$CONTAINER_DIR/$VLLM_SIF

: "${LABEL:?}" "${MODEL:?}" "${REVISION:?}" "${TP:?}"
GPUS=${GPUS:-$(seq -s, 0 $((TP - 1)))}
CHAT_CONC=${CHAT_CONC:-"1 4 16 64 256"}
AGENT_CONC=${AGENT_CONC:-"1 4 16 64 128"}
REPS=${REPS:-3}
EXTRA_BODY=${EXTRA_BODY:-'{"ignore_eos": true}'}
ALLOW=${ALLOW:-'["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken"]'}
IGNORE=${IGNORE:-'["original/*", "metal/*"]'}

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$LABEL-$SLURM_JOB_ID
R=results/raw/$RUN_ID
mkdir -p "$R"
PY=$VENV/bin/python
echo "start $(date -u +%FT%TZ) label=$LABEL model=$MODEL@$REVISION tp=$TP gpus=$GPUS" | tee "$R/job.txt"

# 1. weights -> node tmpfs (only the files serving needs)
t0=$(date +%s)
LOCAL=$(run_container "$SIF_VLLM" python3 -c "
from huggingface_hub import snapshot_download
import json
print(snapshot_download('$MODEL', revision='$REVISION', allow_patterns=json.loads('''$ALLOW'''),
                        ignore_patterns=json.loads('''$IGNORE''')))" | tail -1)
echo "download_s=$(( $(date +%s) - t0 )) size=$(du -sh "$LOCAL" | cut -f1)" | tee -a "$R/job.txt"

# 2. server
export CUDA_VISIBLE_DEVICES=$GPUS
t0=$(date +%s)
# shellcheck disable=SC2086
HF_HUB_OFFLINE=1 run_container "$SIF_VLLM" vllm serve "$LOCAL" --served-model-name "$MODEL" --host 127.0.0.1 --port 8000 \
    --tensor-parallel-size "$TP" --seed 1234 --enable-prefix-caching --no-enable-log-requests \
    ${VLLM_ARGS:-} > "$R/vllm.log" 2>&1 &
SRV=$!
nvidia-smi --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used --format=csv,noheader,nounits -l 1 \
    > "$R/power.csv" 2>/dev/null &
SMI=$!
trap 'kill $SRV $SMI 2>/dev/null || true' EXIT
wait_ready http://127.0.0.1:8000 1500 $SRV
echo "startup_s=$(( $(date +%s) - t0 ))" | tee -a "$R/job.txt"

cat > "$R/config.yaml" <<EOF
run_id: "$RUN_ID"
label: "$LABEL"
host_label: "${HOST_LABEL:-a100-40gb-sxm-node}"
billing: allocated cluster time (no per-hour bill; priced at public list rates in analysis)
engine: {name: vLLM, image: "$VLLM_IMAGE", tp: $TP, gpus: "$GPUS", args: "${VLLM_ARGS:-}"}
model: {repo: "$MODEL", revision: "$REVISION"}
command: "vllm serve <snapshot of $MODEL@$REVISION> --tensor-parallel-size $TP ${VLLM_ARGS:-}"
workloads: {chat: "workloads/chat_1k.jsonl", agent: "workloads/agent_8k.jsonl",
            chat_conc: "$CHAT_CONC", agent_conc: "$AGENT_CONC", reps: $REPS, extra_body: '$EXTRA_BODY'}
notes: fixed output lengths (ignore_eos); unique prompts per request (--fresh); prefix caching on (vLLM default)
EOF
$PY bench/meta.py --base-url http://127.0.0.1:8000 --config "$R/config.yaml" --out "$R/run.json"

# 3. sweeps
run() {  # workload, concurrency levels
    $PY bench/loadgen.py --base-url http://127.0.0.1:8000/v1 --model "$MODEL" \
        --prompts "workloads/$1.jsonl" --concurrency $2 --requests 8 --requests-per-conc 4 \
        --warmup 4 --warmup-per-conc --reps "$REPS" --fresh --max-tokens 400 \
        --extra-body "$EXTRA_BODY" --label "$LABEL-$1" --out "$R/raw_$1.jsonl"
    curl -s http://127.0.0.1:8000/metrics > "$R/metrics_after_$1.prom" || true
}
run chat_1k "$CHAT_CONC"
run agent_8k "$AGENT_CONC"
$PY bench/summarize.py "$R"/raw_*.jsonl --out "$R/summary.csv"
echo "status=complete done=$(date -u +%FT%TZ)" | tee -a "$R/job.txt"
