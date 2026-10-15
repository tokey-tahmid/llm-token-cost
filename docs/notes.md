# Notes

Findings, surprises and corrections, dated. Numbers come from `results/tables/` unless a log is named.

## 2026-10-15 · Results after the rerun

Runs (all with fixed output lengths verified: 0 incomplete outputs, errors under 0.1%):
`…-gptoss120b-tp4-5343917`, `…-gptoss120b-tp2-5343918`, `…-gptoss120b-tp4-long-5343919`, `…-qwen27b-bf16-tp2-5343416` plus its knee run `…-5343924`, and `…-qwen27b-fp8-tp1-5343452`, which hit its time limit after chat and agent levels 1-16 (agent 64 and 128 not measured; 16 already misses the SLO by 8 s). A100-SXM4-40GB, vLLM v0.31.0, 3 reps per level. The 7 remaining errors were client-side: the HTTP library also split lines at U+2028 inside JSON (fixed in `bench/loadgen.py`).

**Capacity within the SLO and cost per 1M output tokens** (API = the bill for the same requests, input included, at the cheapest and median live provider on 2026-10-15):

| Config, workload | Users within SLO | Throughput | Self-hosted, low / mid GPU rate | API cheapest / median |
|---|---|---|---|---|
| gpt-oss-120b, 4 GPUs, chat | 48 | 3,216 tok/s | $0.22 / $0.35 | $0.22 / $0.75 |
| gpt-oss-120b, 2 GPUs, chat | 32 | 1,634 tok/s | $0.22 / $0.34 | $0.22 / $0.75 |
| gpt-oss-120b, 4 GPUs, agent | 16 | 1,211 tok/s | $0.59 / $0.92 | $0.86 / $2.88 |
| gpt-oss-120b, 2 GPUs, agent | 12 | 659 tok/s | $0.54 / $0.84 | $0.86 / $2.88 |
| gpt-oss-120b, 4 GPUs, 8K in / 1K out | 4 | 515 tok/s | $1.39 / $2.16 | $0.36 / $1.21 |
| Qwen3.8-27B BF16, 2 GPUs, chat | 12 | 383 tok/s | $0.93 / $1.45 | $0.55 / $2.87 |
| Qwen3.8-27B BF16, 2 GPUs, agent | 6 | 187 tok/s | $1.91 / $2.97 | $0.99 / $7.40 |
| Qwen3.8-27B FP8, 1 GPU, chat | 6 (knee run) | 193 tok/s | $0.93 / $1.44 | $0.55 / $2.86 |
| Qwen3.8-27B FP8, 1 GPU, agent | 2 (knee run) | 68 tok/s | $2.65 / $4.11 | $0.99 / $7.37 |

**Agents are where self-hosting wins.** The agent prompt repeats a 6K-token prefix that the server caches (78% of prompt tokens served from cache on 4 GPUs, 59% on 2, 75% for Qwen), while every API provider bills it on each request. gpt-oss on 2 GPUs beats the cheapest provider within the SLO even at the mid GPU rate ($0.84 vs $0.86); on 4 GPUs, at the low rate. The 8K/1K workload has no shared prefix and the opposite result: prefill keeps time to first token over 2 s beyond 4 users, and the API wins at every rate.

**Per month** (`monthly_breakeven.csv`, renting 24/7 against capacity within the SLO):
- against the median provider, every configuration wins at the low and mid GPU rate except the 8K/1K workload; gpt-oss wins even at the highest rate;
- against the cheapest provider, gpt-oss chat wins only at the lowest rate and by a hair (4 GPUs: break-even 8.40B output tokens a month, capacity 8.45B); gpt-oss agent wins at the low rate (2 GPUs also at the mid rate); Qwen never does.

**Hypotheses as registered in the brief** (against the price it named: gpt-oss $0.037 in / $0.17 out, Qwen $0.04 / $2.30):
- H1, one user costs ≥ 20× the API: **false**. Chat at one user costs 3× to 19× the named price at the mid rate (gpt-oss 4 GPUs 18.8×, Qwen FP8 2.9×). Against the cheapest provider, 12× to 24×.
- H2, Qwen beats its API price from ≤ 16 users on one A100: **true** (FP8, one GPU, 4 users within the SLO: $1.99 against $2.40 for the same requests). gpt-oss never beats its API price on A100s: **false**; 4 GPUs at 48 chat users and the lowest rate cost $0.22 against $0.26.
- H3, holding the SLO at least doubles the cost per token: **mixed**. Ratio of the cheapest level within the SLO to the cheapest level at all: gpt-oss chat 1.3× (2 GPUs) and 1.9× (4 GPUs), gpt-oss agent 1.2× and 2.0×, Qwen 3.0× to 3.3× (FP8 agent 5.5×), 8K/1K 3.7×.
- H4, the API is cheaper for agents at every load: **false for gpt-oss** (above); for Qwen, false only beyond the SLO (64 users: $0.59 at the low rate against $0.99).

**Energy** (GPU board power only): 4 GPUs drew 706 W at 1 chat user and 1,389 W at 256; 3.1 to 0.23 J per output token, €0.26 to €0.02 per 1M tokens at €0.30/kWh. Idle with the model loaded: 61 to 63 W per GPU (60 s window, `run_facts.csv`).

**The engine matters as much as the GPU price, at least for gpt-oss on A100** (`…-gptoss120b-tp4-sglang-5344814`, SGLang v0.5.21, engine defaults apart from context length). One user: 142 tok/s against vLLM's 225. From 16 users up, SGLang plateaued near 400 tok/s while vLLM reached 3,680 at 64 users; p95 time to first token was 12 s at 16 users. SGLang's log warns that its MXFP4 path "is not fully optimized yet", and its own decode-step throughput (1,282 tok/s at 16 running requests) shows that prompt processing, admitted one request at a time, is the bottleneck. The run hit its 90-minute limit during the 256-user chat level (1 rep there); the agent workload did not run. This is a statement about SGLang's MXFP4 path on Ampere, not about SGLang in general. Tuning effort for gpt-oss: none beyond defaults, for either engine.

**SGLang on Qwen BF16, 2 GPUs: tuning needed to compare at all** (smoke runs kept in `results/raw`). With defaults, SGLang caps this hybrid model at 4 running requests: it keeps 5 Mamba state slots per request for prefix caching and sized the cache at 23 slots (smoke 5345106: one user 43 tok/s, as vLLM; 16 users 159 tok/s, p95 first token 31 s). Raising the cache to 160 slots with 0.88 static memory left no room for the KV cache and the server refused to start (5345146). 80 slots allow 16 running requests and work (5345158). The state stays float32, as the model's config sets it. vLLM needed no such tuning.

**With that tuning, SGLang matches or beats vLLM on Qwen within the SLO** (`…-qwen27b-bf16-tp2-sglang-5345174`; hit its 2-hour limit in the agent 128-user level, agent 64 has 2 reps). Up to 12 users, SGLang delivered 3-10% more throughput on both workloads with slightly lower p95 first-token times: chat holds the SLO to 12 users at 400 tok/s (vLLM 383), agent to 8 users at 250 tok/s (vLLM's 8-user level missed by 0.14 s, leaving it at 6 users, 187 tok/s). From 16 users, SGLang's 16-request cap queues the rest (chat 16: p95 first token 11.1 s vs 2.05 s). So for Qwen the cost verdict does not depend on the engine; for gpt-oss on A100 it does, because SGLang's MXFP4 path is unoptimized there.

**Qwen FP8 knee run** (`…-qwen27b-fp8-tp1-knee-5344815`): one GPU holds the SLO to 6 chat users (193 tok/s, p95 first token 1.97 s) and 2 agent users; repeated levels match the main run within 0.5%.

**Against published newer GPUs** (labelled, not our measurement: `configs/published_gpu_benchmarks.yaml`). On the 8K/1K shape at about 50 tok/s per user, 4 A100s deliver 3,112 input + output tokens per second per GPU (32 users; p95 first token 8.4 s, a limit the published runs do not apply). SemiAnalysis InferenceX lists 8,973 for H100 and 44,489 for B200 at 50 tok/s per user (vLLM, FP4). Different team, versions and tuning.

## 2026-10-15 · Correction: the first gpt-oss-120b results are withdrawn

Every gpt-oss-120b run so far asked for a fixed output length with `ignore_eos`. gpt-oss ends an answer with a harmony end token; ignoring it let generation continue, and vLLM's streaming parser then raised `HarmonyError: Unexpected token 200002 while expecting start token 200006` (2,025 times in one run's log). The stream ended without a finish reason or usage. Share of requests cut short: about 58% on the agent workload, 59% on the 8K/1K workload, 3% on chat. The client ignored the in-stream error event, so the runs looked clean.

Consequences: those requests left the server early, so it ran lighter than the nominal user count, and their tokens were not counted. The throughput, latency and cost numbers in the gpt-oss section below are therefore not valid, the agent ones least of all; the verdicts drawn from them are withdrawn until the rerun. Still valid, because they do not depend on the requests: KV cache sizes (startup log), idle power, and the provider price snapshot.

Fix (commit `fa7c295`): gpt-oss requests use `min_tokens = max_tokens` instead of `ignore_eos`; the load generator records in-stream error events; the summary counts incomplete outputs per level and `cost.py` warns about them. The four affected runs are listed in `results/excluded_runs.yaml` and kept as records. Qwen3.8 runs are unaffected (0 incomplete). Short checks on vLLM and SGLang come first, then full reruns.

## 2026-10-15 · gpt-oss-120b, first sweeps (withdrawn, see the correction above)

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
