# Sourced by every job script. Everything site-specific (module setup, container
# and venv locations, node-local scratch) comes from site.env; copy
# site.env.example and fill it in. Run sbatch from the repository root.
REPO=${SLURM_SUBMIT_DIR:?run sbatch from the repository root}
cd "$REPO"
set +eu
# shellcheck disable=SC1090
source "${SITE_ENV:-$REPO/site.env}"
eval "${SITE_SETUP:-}"
set -eu
: "${CONTAINER_DIR:?set in site.env}" "${VENV:?set in site.env}" "${SCRATCH:?set in site.env}"
mkdir -p "$SCRATCH"/{hf,home,tmp,cache} logs results/raw
export HF_HOME=$SCRATCH/hf
export APPTAINER_CACHEDIR=${APPTAINER_CACHEDIR:-$CONTAINER_DIR/cache}
export APPTAINER_TMPDIR=${APPTAINER_TMPDIR:-$SCRATCH/tmp}

# Run a command in a container with the GPUs, a clean environment, and all
# write caches on node-local scratch (shared filesystems are often read-only
# inside the container unless bound rw).
run_container() {
    local sif=$1; shift
    local gpu_env=()
    [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]] && gpu_env=(--env "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES")
    apptainer exec --nv --cleanenv "${gpu_env[@]}" \
        --bind "$REPO:$REPO:rw" --bind "$SCRATCH:$SCRATCH:rw" --bind "$CONTAINER_DIR:$CONTAINER_DIR:ro" \
        --home "$SCRATCH/home" --env TMPDIR="$SCRATCH/tmp" --env HF_HOME="$HF_HOME" \
        --env XDG_CACHE_HOME="$SCRATCH/cache" --env TORCHINDUCTOR_CACHE_DIR="$SCRATCH/cache/inductor" \
        --env TRITON_CACHE_DIR="$SCRATCH/cache/triton" --env VLLM_CACHE_ROOT="$SCRATCH/cache/vllm" \
        ${CONTAINER_EXTRA_ARGS:-} "$sif" "$@"
}

# Block until an OpenAI-compatible server answers, or fail.
wait_ready() {
    local url=$1 timeout_s=${2:-1500} pid=${3:-} t=0
    until curl -sf -m 5 "$url/health" >/dev/null 2>&1 || curl -sf -m 5 "$url/v1/models" >/dev/null 2>&1; do
        if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then echo "server exited before ready" >&2; return 1; fi
        if (( t >= timeout_s )); then echo "server not ready after ${timeout_s}s" >&2; return 1; fi
        sleep 5; t=$((t + 5))
    done
    echo "ready after ${t}s"
}
