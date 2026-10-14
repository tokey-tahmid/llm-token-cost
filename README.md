# llm-token-cost

**What does a million tokens cost when you serve an open model yourself, and at what load does that beat today's API prices?**

This repository measures cost-per-token curves for self-hosted LLM serving on NVIDIA A100 GPUs, and compares them with published API prices for the same models. It answers with a break-even number per model, and with a formula you can rerun at your own GPU price and utilization.

> **Status: not yet measured.** The harness, workloads and job scripts are in place. No result in this README exists until a run lands under `results/raw/`; every number here will trace to a committed raw artifact.

## The question, made falsifiable

Thresholds were fixed before any data was collected:

- **H1:** with one user, self-hosted cost per million output tokens is at least 20× the API price.
- **H2:** self-hosting Qwen3.8-27B beats its API output price from ≤ 16 concurrent users on one A100, while gpt-oss-120b never beats its API price on A100s.
- **H3:** keeping every user at ≥ 20 tok/s at least doubles the cost per token, compared with max-throughput batching.
- **H4:** for an agent-style workload (~8K tokens in, 300 out, shared prefix), the API is cheaper at every load tested.

## What it measures

- **Per request:** time to first token, time per output token, inter-chunk latency and end-to-end latency, using engine-reported token counts.
- **Per configuration:** median, p95 and p99 across three repetitions after warm-up, plus aggregate throughput and goodput under a latency target. GPU power is sampled every second.
- **Cost:**

  ```
  cost per 1M output tokens = price per GPU-hour × GPUs ÷ (output tokens/s × 3600) × 10^6
  ```

  The GPU time was allocated cluster time with no per-hour bill, so it's priced at a dated range of public list prices for the same GPU. API prices are list prices recorded on the same day.

| Axis | Values |
|---|---|
| Models | Qwen3.8-27B (BF16, TP=2; FP8 weight-only, TP=1), gpt-oss-120b (MXFP4, TP=2 and TP=4), all pinned by revision |
| Engine | vLLM v0.31.0, pinned container image |
| Workloads | chat (~1K tokens in, 400 out), agent-style (~7.5K in with a 5.5K shared system prefix, 300 out) |
| Load | closed-loop concurrency 1, 4, 16, 64, 256 (agent: 1, 4, 16, 64, 128) |

Every request uses a prompt not used earlier in the run, so the prefix cache only helps where the workload is designed to share a prefix. Output lengths are fixed (`ignore_eos`), so token counts are comparable across configurations.

## Hardware

4× NVIDIA A100-SXM4-40GB per node, all GPU pairs connected by NVLink (NV4), 2× AMD EPYC 7452 (64 cores), 503 GiB RAM, diskless nodes with model weights staged in node memory.

## Layout

```
bench/       load generator, summarizer, run provenance, memory-bandwidth test
workloads/   public-domain sources (Project Gutenberg) -> prompt sets, with checksums
configs/     pinned image, serving configurations
slurm/       job scripts (provision, probe, sweep); site settings live in site.env
scripts/     submission helpers, public-content check
analysis/    tables and figures, generated from results/raw only
results/raw/ one folder per run: run.json, config.yaml, raw JSONL, power, metrics; append-only
docs/notes.md  findings and surprises, dated
```

## Reproduce

```bash
cp site.env.example site.env              # account, partition, module setup, paths
scripts/submit.sh slurm/provision.sbatch  # image, venv, workloads
scripts/submit.sh slurm/probe.sbatch      # hardware facts + end-to-end check
scripts/submit_sweeps.sh                  # one job per serving configuration
```

Analysis and figures are regenerated from `results/raw/` with one command, added with the first results.

## Rules this repository follows

- **No invented data.** Nothing under `results/` is produced by anything but a real run. Failed or out-of-memory configurations are recorded as results, not dropped.
- **Provenance per run:** git SHA, Slurm job id, GPU model, driver and CUDA, engine image, model revision, full configuration and command.
- **Statistics:** at least 3 repetitions, warm-up recorded but excluded, medians reported with min–max spread.

## License

Code: MIT. Workload texts: Project Gutenberg, public domain.
