#!/bin/bash -l
#SBATCH --job-name=ltc-probe
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
# (memory follows CPUs on this partition: request all of them so weights staged in node tmpfs fit)
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Hardware facts + an end-to-end harness check: vLLM serving a small model on
# one GPU, the load generator at three concurrency levels.
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/slurm/common.sh"
source configs/versions.env
R=results/probe/$SLURM_JOB_ID
mkdir -p "$R"
{
  nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit,pcie.link.gen.max --format=csv
  nvidia-smi topo -m
  lscpu | grep -E "Model name|Socket|Core\(s\)|Thread|NUMA node\(s\)"
  free -g | head -2
} > "$R/hardware.txt" 2>&1
{ gcc -O3 -march=native -fopenmp bench/membw.c -o "$SCRATCH/membw" &&
  OMP_NUM_THREADS=$(( $(nproc) / 2 )) OMP_PROC_BIND=spread OMP_PLACES=cores "$SCRATCH/membw" 4096; } > "$R/membw.txt" 2>&1 || true

export CUDA_VISIBLE_DEVICES=0
run_container "$CONTAINER_DIR/$VLLM_SIF" vllm serve Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8000 \
    --max-model-len 8192 --gpu-memory-utilization 0.85 > "$R/vllm.log" 2>&1 &
SRV=$!
trap 'kill $SRV 2>/dev/null || true' EXIT
wait_ready http://127.0.0.1:8000 900 $SRV
"$VENV/bin/python" bench/loadgen.py --base-url http://127.0.0.1:8000/v1 --model Qwen/Qwen3-0.6B \
    --prompts workloads/chat_1k.jsonl --concurrency 1 8 32 --requests 32 --warmup 8 --reps 2 --fresh \
    --max-tokens 128 --extra-body '{"chat_template_kwargs": {"enable_thinking": false}}' \
    --label probe-0.6b --out "$R/raw.jsonl"
"$VENV/bin/python" bench/summarize.py "$R/raw.jsonl" --out "$R/summary.csv"
