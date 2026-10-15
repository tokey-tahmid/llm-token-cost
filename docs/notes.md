# Notes

Findings, surprises and corrections, dated. Numbers come from `results/tables/` unless a log is named.

## 2026-10-15 · gpt-oss-120b, first sweeps

Runs: `20261015T003247Z-gptoss120b-tp2-5341553` (2 GPUs), `20261015T011211Z-gptoss120b-tp4-5341554` (4 GPUs). A100-SXM4-40GB, vLLM v0.31.0, 3 reps per level, zero request errors.

**Two GPUs fit the model but leave almost no room for users.** With the weights loaded, vLLM had 1.41 GiB per GPU for KV cache on 2 GPUs: 65,396 tokens in total, about 4 concurrent 16K-token requests. On 4 GPUs it had 17.74 GiB per GPU: 1,642,391 tokens, 25 times more with twice the GPUs (`vllm.log.gz`, KV cache size lines). The consequence is queueing, not slower decoding:
- agent workload (about 7.1K prompt tokens), 2 GPUs: throughput peaks at 378 tok/s at 16 users and falls to 276 at 64, with median time to first token 18.6 s at 64 users and 51 s at 128;
- chat at 256 users, 2 GPUs: median time to first token 36 s, while per-user decode speed still reads 38 tok/s. Decode speed alone hides this, so the figures mark every level that misses the SLO with a hollow marker.

**"The API price" is a range, and the verdict depends on where in it you buy.** OpenRouter listed 24 providers for gpt-oss-120b on 2026-10-15 (21 live), with output prices from $0.15 to $0.95 per 1M tokens (`results/prices/2026-10-15-openrouter-*.json`). The price named in the brief ($0.037 in, $0.17 out, recorded 2026-10-14) is one of the cheapest endpoints. Several of the cheapest serve 4-bit weights, as our runs do (gpt-oss ships in MXFP4). Every API figure below is the bill for the same requests, input tokens included, per 1M output tokens:

| Config, workload | Highest level within SLO | Throughput | Self-hosted, low / mid GPU rate | API: named | cheapest | median |
|---|---|---|---|---|---|---|
| 2 GPUs, chat | 16 users | 1,118 tok/s | $0.32 / $0.50 | $0.26 | $0.22 | $0.75 |
| 4 GPUs, chat | 48 users (knee run) | 3,171 tok/s | $0.23 / $0.35 | $0.26 | $0.22 | $0.75 |
| 2 GPUs, agent | 4 users | 251 tok/s | $1.43 / $2.21 | $1.05 | $0.86 | $2.88 |
| 4 GPUs, agent | 16 users | 646 tok/s | $1.11 / $1.72 | $1.05 | $0.86 | $2.88 |

Per month, renting the GPUs 24/7 (`monthly_breakeven.csv`, capacity = highest throughput within the SLO):
- **against the cheapest provider, self-hosting never wins, but 4 GPUs on chat come within 1%:** at the lowest GPU rate, break-even needs 8.41B output tokens a month and the node serves 8.33B within the SLO;
- **against the price named in the brief, 4 GPUs on chat win at the lowest GPU rate** (break-even 7.21B, capacity 8.33B), not at the mid rate (11.2B). So hypothesis H2, "gpt-oss-120b never beats its API price on A100s", is false at the lowest rate, and only once the sweep resolved the SLO boundary (see the knee run below);
- **against the median provider, both configurations win on both workloads** at the low and mid GPU rates. 4 GPUs, chat, mid rate: break-even 3.9B, capacity 8.3B. Agent, the hardest case: 1.0B against 1.7B;
- at the highest GPU rate, nothing wins against anyone.

The video reports all three readings. Providers are not named or ranked on screen (brief, risks).

Prompt caching barely moves the API side: most endpoints list no cache-read price, so crediting the share of the prompt our server served from cache (78% on 4 GPUs) lowers the median agent bill only from $2.88 to $2.78.

**If users accept a wait, even the cheapest API is beaten.** 4 GPUs, chat, 256 users: $0.19 per 1M output tokens at the mid rate, below the cheapest provider's $0.22, but with p95 time to first token at 8.3 s and 30 tok/s per user. The verdict depends on the latency target and on which provider you compare with, more than on the GPU price.

**The SLO boundary decides the verdict, so it was measured finely** (knee run `20261015T090434Z-gptoss120b-tp4-knee-5343438`, another node). 4 GPUs, chat: 24, 32 and 48 users all meet the SLO (p95 first token 0.89, 1.17, 1.70 s); 48 users give 3,171 tok/s, 84% more than the 16-user level that bounded capacity before. Agent: 24 users miss by 6 ms (p95 first token 2.006 s), so 16 users remain the agent capacity; a threshold that sharp is worth saying on screen. The repeated levels reproduce across nodes within 1-5% on throughput and about 2% on p95 first token (`anchor_repeats.csv`). The 2-GPU knee run is queued.

**Energy is a small part of the cost.** Board power of the GPUs in use (not the host), averaged over each level's requests: 4 GPUs at 256 chat users drew 1,387 W, 0.23 J per output token, about €0.02 per 1M output tokens at €0.30/kWh. At one user the same node needs 3.5 J per token.

## 2026-10-15 · Analysis corrections, before any figure was published

- **Power timestamps.** The power sampler of these two runs wrote node-local time without a zone. Converting with `astype("int64")` assumed nanoseconds, but pandas parsed milliseconds, so every window came out empty. Fixed with a Timedelta conversion, and the sampler now writes UTC epoch seconds.
- **Power summed over GPUs.** Each GPU's sample carries its own millisecond timestamp, so grouping by timestamp averaged GPUs instead of summing them, and the 2-GPU run also counted the two idle GPUs. Now: only the configuration's GPUs, mean over time per GPU, then summed.
- **API comparison per level.** The per-level figure first compared against the API's output-token price alone ($0.17), while the self-hosted cost includes the GPU time spent on the prompt. It now uses the API's bill for the same requests per output token. The monthly break-even already used this blended price.

## 2026-10-15 · Failed attempts (kept in `results/raw/`, no summary.csv)

- `qwen27b-bf16-tp2-5341551`: the run's config.yaml was not valid YAML (the engine arguments contain JSON). Arguments and command lines are now single-quoted.
- `qwen27b-fp8-tp1-5341552`: out of memory on one 40 GB GPU. CUDA-graph profiling with the default sequence limit wanted 12.25 GiB on top of the weights.
- `qwen27b-fp8-tp1-5343417`: with the limit at 128, vLLM refused to start. The model is a hybrid and each running sequence holds one Mamba state block; 109 fit. The limit is now 96.
