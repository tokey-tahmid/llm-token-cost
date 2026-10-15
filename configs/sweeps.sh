# Serving configurations for the sweep (sourced by scripts/submit_sweeps.sh).
# label | model | revision | TP | engine args | request extra body | walltime | engine (default vllm)
QWEN=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
QWEN_FP8=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
GPTOSS=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
NO_MM='--limit-mm-per-prompt {"image":0,"video":0}'
QWEN_BODY='{"ignore_eos": true, "chat_template_kwargs": {"enable_thinking": false}}'
# gpt-oss: ignore_eos lets generation run past the harmony end token, which crashes vLLM's
# streaming parser (HarmonyError, ~58% of agent requests cut short in jobs 5341553/54, 5343438/39,
# 5343670). min_tokens = max_tokens masks the stop tokens instead (loadgen fills in the number).
OSS_BODY='{"min_tokens": "max_tokens", "reasoning_effort": "low"}'
SWEEPS=(
  "qwen27b-bf16-tp2|Qwen/Qwen3.8-27B|$QWEN|2|--max-model-len 16384 --gpu-memory-utilization 0.92 $NO_MM|$QWEN_BODY|01:40:00"
  # one 40 GB GPU: the default sequence limit makes CUDA-graph profiling allocate 12.25 GiB of
  # per-sequence state on top of 28 GiB of weights (OOM, job 5341552). The model is a hybrid:
  # each running sequence also holds one Mamba state block, and 109 fit (job 5343417 refused
  # 128). 96 leaves margin; requests beyond it queue, which shows up in TTFT
  "qwen27b-fp8-tp1|Qwen/Qwen3.8-27B-FP8|$QWEN_FP8|1|--max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 96 $NO_MM|$QWEN_BODY|01:50:00"
  "gptoss120b-tp2|openai/gpt-oss-120b|$GPTOSS|2|--max-model-len 16384 --gpu-memory-utilization 0.94|$OSS_BODY|01:20:00"
  "gptoss120b-tp4|openai/gpt-oss-120b|$GPTOSS|4|--max-model-len 16384 --gpu-memory-utilization 0.92|$OSS_BODY|01:10:00"
  # robustness check (brief 001): the headline configuration on SGLang, engine defaults apart from context length
  "gptoss120b-tp4-sglang|openai/gpt-oss-120b|$GPTOSS|4|--context-length 16384|$OSS_BODY|01:10:00|sglang"
  # gpt-oss's MXFP4 path in SGLang is unoptimized on A100 (its own warning; job 5344814), so the
  # engine check also runs on a path SGLang optimizes: dense BF16
  # SGLang's defaults cap this hybrid model at 4 running requests (Mamba state cache: 23 slots, 5 per
  # request; smoke 5345106). Tuning, documented: static memory 0.88 (vLLM ran 0.92) and 160 state
  # slots (32 requests); numerics at defaults.
  "qwen27b-bf16-tp2-sglang|Qwen/Qwen3.8-27B|$QWEN|2|--context-length 16384 --mem-fraction-static 0.88 --max-mamba-cache-size 160|$QWEN_BODY|01:40:00|sglang"
)
