# llm-token-cost

**What does a million tokens cost when you serve an open model yourself, and at what load does that beat today's API prices?**

This repository measures how the cost per token changes with load when you serve open large language models yourself on NVIDIA A100 GPUs, and it compares that cost with what API providers charge for exactly the same requests. The answer comes as a break-even point for each model and workload, together with a formula that you can rerun with your own GPU price and your own traffic.

## What I found

The measurements are complete, and every number below traces back to a raw run under `results/raw/`. The dated findings, corrections and caveats are in `docs/notes.md`, and the tables and figures are in `results/tables/` and `results/figures/`.

The clearest result is that agents are where self-hosting wins. An agent-style request repeats a long prompt prefix, which the server keeps in its cache (78% of the prompt tokens came from the cache on four GPUs), while most API offers bill that prefix at full price on each request (4 of the 24 offers for gpt-oss-120b discount cached input, and the cheapest one doesn't). As a result, gpt-oss-120b on two GPUs, within the latency target and at the middle GPU rental rate, costs $0.84 per million output tokens, which is just under the cheapest provider's $0.86 for the same requests.

For chat, four GPUs serve 48 users within the latency target at $0.35 per million output tokens at the middle rate. That beats the median provider comfortably, but it only ties the cheapest provider at the lowest rental rate. However, for long prompts where nothing repeats, the API wins at every rental rate, because prompt processing pushes the time to the first token past the target after only a few users.

The serving engine also matters. SGLang's 4-bit path for gpt-oss on A100 plateaued at around 400 output tokens a second, while vLLM kept scaling, and SGLang's own log says that this path "is not fully optimized yet". On Qwen3.8-27B, however, SGLang matched vLLM within the latency target once its state cache was sized correctly.

Measured against the thresholds that I registered before collecting any data, the first hypothesis (one user costs at least 20 times the API) is false, the second is true for Qwen but false for gpt-oss, the third (holding the latency target at least doubles the cost) is mixed, and the fourth (the API is always cheaper for agents) is false for gpt-oss. The exact ratios for each hypothesis are in `docs/notes.md`.

## The question, made falsifiable

I fixed these thresholds before any data was collected.

1. With one user, the self-hosted cost per million output tokens is at least 20 times the API price.
2. Self-hosting Qwen3.8-27B beats its API output price from 16 or fewer concurrent users on one A100, while gpt-oss-120b never beats its API price on A100s.
3. Keeping every user at 20 tokens a second or more at least doubles the cost per token, compared with batching for maximum throughput.
4. For an agent-style workload with a long shared prefix, the API is cheaper at every load tested.

## What it measures

For every request, the load generator records the time to the first token, the time per output token, the gaps between streamed chunks and the end-to-end latency, using the token counts that the engine itself reports. For every configuration, it reports the median, 95th and 99th percentiles across three repetitions after a warm-up, together with the total throughput and the throughput that stays within a latency target. In addition, GPU power is sampled every second.

The latency target is that, for 95% of requests, the first token arrives within two seconds and the text then keeps coming at 20 tokens a second or more. The cost follows from the measured throughput.

```
cost per 1M output tokens = price per GPU-hour × GPUs ÷ (output tokens/s × 3600) × 10^6
```

The GPU time was allocated cluster time with no hourly bill, so it's priced at a dated range of public list prices for the same GPU. The API comparison uses each provider's bill for the same requests, with the input tokens included, from a price snapshot taken on the same day.

| Axis | Values |
|---|---|
| Models | Qwen3.8-27B (BF16 on 2 GPUs; FP8 weight-only on 1 GPU), gpt-oss-120b (MXFP4 on 2 and 4 GPUs), all pinned by revision |
| Engines | vLLM v0.31.0 for every configuration; SGLang v0.5.21 for gpt-oss-120b on 4 GPUs (chat) and Qwen3.8-27B BF16 on 2 GPUs |
| Workloads | chat (about 1,000 tokens in, 400 out), agent-style (about 7,300 in, mostly a shared prefix, 300 out), long prompts (about 7,200 in, nothing shared, 1,024 out) |
| Load | closed-loop concurrency from 1 to 256 users, with extra levels around each configuration's latency limit |

Every request uses a prompt that hasn't been used earlier in the run, so the prefix cache only helps where the workload is designed to share a prefix. Output lengths are fixed so that token counts are comparable across configurations, using `ignore_eos` for Qwen and `min_tokens` equal to `max_tokens` for gpt-oss, because `ignore_eos` lets gpt-oss run past its end token, which crashes vLLM's streaming parser for that model.

## Hardware

Each node has four NVIDIA A100-SXM4-40GB GPUs, with every pair of GPUs connected by NVLink, two AMD EPYC 7452 processors with 64 cores in total, and 503 GiB of memory. The nodes are diskless, so the model weights are staged in node memory before each run.

## Layout

```
bench/       load generator, summarizer, run provenance, memory-bandwidth test
workloads/   public-domain sources (Project Gutenberg) -> prompt sets, with checksums
configs/     pinned image, serving configurations, prices, published GPU figures
slurm/       job scripts (provision, probe, sweep); site settings live in site.env
scripts/     submission helpers, public-content check
analysis/    tables and figures, generated from results/raw only
results/raw/ one folder per run: run.json, config.yaml, raw JSONL, power, metrics; append-only
docs/notes.md  findings, surprises and corrections, dated
```

## Reproduce

```bash
cp site.env.example site.env              # account, partition, module setup, paths
scripts/submit.sh slurm/provision.sbatch  # image, venv, workloads
scripts/submit.sh slurm/probe.sbatch      # hardware facts + end-to-end check
scripts/submit_sweeps.sh                  # one job per serving configuration
```

The tables and figures regenerate from `results/raw/` with one command.

```bash
python analysis/cost.py && python analysis/figures.py   # results/tables/, results/figures/
```

The prices live in `configs/prices.yaml`, each with its source and date.

## Rules this repository follows

1. There is no invented data. Nothing under `results/` comes from anything but a real run, and failed or out-of-memory configurations are recorded as results rather than dropped. Runs that turned out to be faulty are listed with the reason in `results/excluded_runs.yaml`.
2. Every run records its provenance, which means the git commit, the Slurm job ID, the GPU model, the driver and CUDA versions, the engine image, the model revision, and the full configuration and command.
3. Every level is repeated at least three times, the warm-up is recorded but excluded, and medians are reported with their spread.

## License

The code is under the MIT license, and the workload texts come from Project Gutenberg and are in the public domain.
