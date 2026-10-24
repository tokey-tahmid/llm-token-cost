#!/bin/bash -l
#SBATCH --job-name=ltc-repro-harmony
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=128
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x-%j.out

# Minimal reproduction of the gpt-oss + ignore_eos streaming failure (docs/notes.md, 2026-10-15),
# on the vLLM version the study used (v0.31.0) and on the newest release (v0.32.0), with
# gpt-oss-20b on one GPU. For each version, the same streaming requests run twice: with
# ignore_eos, and with min_tokens = max_tokens (the workaround). Each request is classified by
# how its stream ended. The newer image, its pull cache and the model stay in node memory, so
# nothing new lands on shared storage.
#   scripts/submit.sh slurm/repro_harmony_ignore_eos.sbatch --qos=test
set -euo pipefail
source "${SLURM_SUBMIT_DIR:?}/slurm/common.sh"
source configs/versions.env
MODEL20=openai/gpt-oss-20b REV20=6cee5e81ee83917806bbde320786a8fb61efebee
NEW_IMAGE=vllm/vllm-openai:v0.32.0
R=results/raw/$(date -u +%Y%m%dT%H%M%SZ)-repro-harmony-$SLURM_JOB_ID
mkdir -p "$R"
PY=$VENV/bin/python
export CUDA_VISIBLE_DEVICES=0

NEW_SIF=$SCRATCH/vllm-openai_v0.32.0.sif
APPTAINER_CACHEDIR=$SCRATCH/apptainer-cache APPTAINER_TMPDIR=$SCRATCH/tmp \
    apptainer pull --force "$NEW_SIF" "docker://$NEW_IMAGE" > "$R/pull.log" 2>&1
LOCAL=$(run_container "$CONTAINER_DIR/$VLLM_SIF" python3 -c "
from huggingface_hub import snapshot_download
print(snapshot_download('$MODEL20', revision='$REV20', allow_patterns=['*.json', '*.safetensors', '*.txt', '*.jinja', '*.tiktoken'],
                        ignore_patterns=['original/*', 'metal/*']))" | tail -1)

cat > "$R/probe.py" <<'EOF'
"""Send streaming chat requests and classify how each stream ended."""
import asyncio, collections, json, sys
import httpx
mode, n, PORT = sys.argv[1], int(sys.argv[2]), sys.argv[3]
QUESTIONS = ["Answer in one sentence: what is the capital of France?", "Name three primary colours.",
             "What is 12 times 12?", "Give one word that rhymes with cat.", "Summarize the plot of Moby-Dick in two sentences."]
async def one(client, i):
    body = {"model": "m", "stream": True, "max_tokens": 300, "reasoning_effort": "low",
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": QUESTIONS[i % len(QUESTIONS)]}]}
    body |= {"ignore_eos": True} if mode == "ignore_eos" else {"min_tokens": 300}
    url = f"http://127.0.0.1:{PORT}/v1/chat/completions"
    finish, usage, error = None, None, None
    async with client.stream("POST", url, json=body) as r:
        status = r.status_code
        buf = b""
        async for raw in r.aiter_bytes():  # split on b"\n" only (aiter_lines also splits at U+2028)
            buf += raw
            *lines, buf = buf.split(b"\n")
            for line in (l.decode().strip() for l in lines):
                if not line.startswith("data:") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[5:])
                if ev.get("error") or ev.get("object") == "error":
                    error = str(ev.get("error", ev))[:160]
                for ch in ev.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                usage = ev.get("usage") or usage
    tokens = (usage or {}).get("completion_tokens")
    return status, ("error event" if error else "no finish reason" if finish is None else f"finish={finish}"), tokens, error
async def main():
    async with httpx.AsyncClient(timeout=300) as c:
        res = await asyncio.gather(*(one(c, i) for i in range(n)))
    outcomes = collections.Counter((s, o) for s, o, _, _ in res)
    tokens = [t for _, o, t, _ in res if t is not None]
    print(json.dumps({"mode": mode, "requests": n, "outcomes": {f"HTTP {s}, {o}": k for (s, o), k in outcomes.items()},
                      "completion_tokens_min_max": [min(tokens), max(tokens)] if tokens else None,
                      "example_error": next((e for *_, e in res if e), None)}))
asyncio.run(main())
EOF

# each version on its own port, and the next starts only once the previous server is gone;
# the version is read from the server that answers (GET /version), not from the image
port=8000
for sif in "$CONTAINER_DIR/$VLLM_SIF" "$NEW_SIF"; do
    tag=$(basename "$sif" .sif)
    HF_HUB_OFFLINE=1 run_container "$sif" vllm serve "$LOCAL" --served-model-name m --host 127.0.0.1 --port $port \
        --max-model-len 8192 --seed 1234 > "$R/server_$tag.log" 2>&1 &
    SRV=$!
    wait_ready http://127.0.0.1:$port 900 $SRV
    echo "$tag answered by server version $(curl -s http://127.0.0.1:$port/version)" | tee -a "$R/results.txt"
    for mode in ignore_eos min_tokens; do $PY "$R/probe.py" "$mode" 40 $port | tee -a "$R/results.txt"; done
    echo "server log lines mentioning the parser error: $(grep -c -E 'HarmonyError|Unexpected token' "$R/server_$tag.log" || true)" | tee -a "$R/results.txt"
    pkill -TERM -f "vllm serve $LOCAL" || true
    for _ in $(seq 60); do pgrep -f "vllm serve $LOCAL" >/dev/null || break; sleep 2; done
    pkill -KILL -f "vllm serve $LOCAL" 2>/dev/null || true
    port=$((port + 1))
done
echo "status=complete" >> "$R/results.txt"
