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
#   CHAT_CONC AGENT_CONC REPS   (an empty level list skips that workload)
#   LONG_CONC levels for long_8k1k (8K in, 1K out, the InferenceX shape); unset = not run
#   ENGINE    vllm (default) or sglang; VLLM_ARGS then holds that engine's extra arguments
#   RUN_TAG   optional, e.g. knee: extra concurrency levels for an existing config;
#             analysis/cost.py merges them and keeps the first run of a repeated level;
#             RUN_TAG=smoke marks a short check run that the analysis skips
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/slurm/common.sh"
source configs/versions.env
SIF_VLLM=$CONTAINER_DIR/$VLLM_SIF
ENGINE=${ENGINE:-vllm}

: "${LABEL:?}" "${MODEL:?}" "${REVISION:?}" "${TP:?}"
GPUS=${GPUS:-$(seq -s, 0 $((TP - 1)))}
CHAT_CONC=${CHAT_CONC-"1 4 16 64 256"}
AGENT_CONC=${AGENT_CONC-"1 4 16 64 128"}
REPS=${REPS:-3}
EXTRA_BODY=${EXTRA_BODY:-'{"ignore_eos": true}'}
ALLOW=${ALLOW:-'["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model", "*.tiktoken"]'}
IGNORE=${IGNORE:-'["original/*", "metal/*"]'}

RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$LABEL${RUN_TAG:+-$RUN_TAG}-$SLURM_JOB_ID
R=results/raw/$RUN_ID
mkdir -p "$R"
PY=$VENV/bin/python
echo "start $(date -u +%FT%TZ) label=$LABEL model=$MODEL@$REVISION tp=$TP gpus=$GPUS engine=$ENGINE" | tee "$R/job.txt"

# 1. weights -> node tmpfs (only the files serving needs)
t0=$(date +%s)
LOCAL=$(run_container "$SIF_VLLM" python3 -c "
from huggingface_hub import snapshot_download
import json
print(snapshot_download('$MODEL', revision='$REVISION', allow_patterns=json.loads('''$ALLOW'''),
                        ignore_patterns=json.loads('''$IGNORE''')))" | tail -1)
echo "download_s=$(( $(date +%s) - t0 )) size=$(du -shL "$LOCAL" | cut -f1)" | tee -a "$R/job.txt"

# 2. server
export CUDA_VISIBLE_DEVICES=$GPUS
t0=$(date +%s)
if [[ $ENGINE == sglang ]]; then  # prefix caching (radix cache) is on by default
    SIF_SRV=$CONTAINER_DIR/$SGLANG_SIF; IMAGE=$SGLANG_IMAGE; ENGINE_NAME=SGLang; LOG=$R/server.log
    SERVE=(python3 -m sglang.launch_server --model-path "$LOCAL" --served-model-name "$MODEL" --host 127.0.0.1
           --port 8000 --tp-size "$TP" --random-seed 1234 --enable-metrics)
else
    SIF_SRV=$SIF_VLLM; IMAGE=$VLLM_IMAGE; ENGINE_NAME=vLLM; LOG=$R/vllm.log
    SERVE=(vllm serve "$LOCAL" --served-model-name "$MODEL" --host 127.0.0.1 --port 8000
           --tensor-parallel-size "$TP" --seed 1234 --enable-prefix-caching --no-enable-log-requests)
fi
# shellcheck disable=SC2086
HF_HUB_OFFLINE=1 run_container "$SIF_SRV" "${SERVE[@]}" ${VLLM_ARGS:-} > "$LOG" 2>&1 &
SRV=$!
# power.csv: UTC epoch, gpu index, board watts, utilization %, memory MiB (one line per GPU per second)
( while true; do t=$(date +%s.%N); nvidia-smi --query-gpu=index,power.draw,utilization.gpu,memory.used \
      --format=csv,noheader,nounits | sed "s/^/$t, /"; sleep 1; done ) > "$R/power.csv" 2>/dev/null &
SMI=$!
trap 'kill $SRV $SMI 2>/dev/null || true' EXIT
wait_ready http://127.0.0.1:8000 1500 $SRV
echo "startup_s=$(( $(date +%s) - t0 ))" | tee -a "$R/job.txt"

cat > "$R/config.yaml" <<EOF
run_id: "$RUN_ID"
label: "$LABEL"
host_label: "${HOST_LABEL:-a100-40gb-sxm-node}"
billing: allocated cluster time (no per-hour bill; priced at public list rates in analysis)
engine: {name: $ENGINE_NAME, image: "$IMAGE", tp: $TP, gpus: "$GPUS", args: '${VLLM_ARGS:-}'}
model: {repo: "$MODEL", revision: "$REVISION"}
command: '${SERVE[*]/$LOCAL/<snapshot of $MODEL@$REVISION>} ${VLLM_ARGS:-}'
workloads: {chat: "workloads/chat_1k.jsonl", agent: "workloads/agent_8k.jsonl",
            long: "workloads/long_8k1k.jsonl", chat_conc: "$CHAT_CONC", agent_conc: "$AGENT_CONC",
            long_conc: "${LONG_CONC:-}", reps: $REPS, extra_body: '$EXTRA_BODY'}
notes: fixed output lengths (ignore_eos); unique prompts per request (--fresh); prefix caching on
EOF
$PY bench/meta.py --base-url http://127.0.0.1:8000 --config "$R/config.yaml" --out "$R/run.json"

# idle power: the server is ready and loaded, no requests (analysis/cost.py run_facts)
t_idle=$(date +%s.%N); sleep 60
echo "idle_from=$t_idle idle_to=$(date +%s.%N)" | tee -a "$R/job.txt"

# 3. sweeps
run() {  # workload, concurrency levels
    [[ -z "${2// }" ]] && return 0
    $PY bench/loadgen.py --base-url http://127.0.0.1:8000/v1 --model "$MODEL" \
        --prompts "workloads/$1.jsonl" --concurrency $2 --requests 8 --requests-per-conc 4 \
        --warmup 4 --warmup-per-conc --reps "$REPS" --fresh --max-tokens 400 \
        --extra-body "$EXTRA_BODY" --label "$LABEL-$1" --out "$R/raw_$1.jsonl"
    curl -s http://127.0.0.1:8000/metrics > "$R/metrics_after_$1.prom" || true
}
run chat_1k "$CHAT_CONC"
run agent_8k "$AGENT_CONC"
if [[ -n "${LONG_CONC:-}" ]]; then  # generated deterministically; chat and agent files stay byte-identical
    [[ -s workloads/long_8k1k.jsonl ]] || $PY workloads/make_workloads.py workloads/sources --out workloads > /dev/null
    run long_8k1k "$LONG_CONC"
fi
$PY bench/summarize.py "$R"/raw_*.jsonl --out "$R/summary.csv"
echo "status=complete done=$(date -u +%FT%TZ)" | tee -a "$R/job.txt"
