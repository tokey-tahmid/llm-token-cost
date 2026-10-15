# Serving configurations for the sweep (sourced by scripts/submit_sweeps.sh).
# label | model | revision | TP | vllm args | request extra body | walltime
QWEN=1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0
QWEN_FP8=017b9c7af6b5689d5dd426a76e0bc077eb5ca20a
GPTOSS=b5c939de8f754692c1647ca79fbf85e8c1e70f8a
NO_MM='--limit-mm-per-prompt {"image":0,"video":0}'
QWEN_BODY='{"ignore_eos": true, "chat_template_kwargs": {"enable_thinking": false}}'
OSS_BODY='{"ignore_eos": true, "reasoning_effort": "low"}'
SWEEPS=(
  "qwen27b-bf16-tp2|Qwen/Qwen3.8-27B|$QWEN|2|--max-model-len 16384 --gpu-memory-utilization 0.92 $NO_MM|$QWEN_BODY|01:40:00"
  # one 40 GB GPU: the default sequence limit makes CUDA-graph profiling allocate 12.25 GiB of
  # per-sequence state on top of 28 GiB of weights (OOM, job 5341552); 128 fits, more queues
  "qwen27b-fp8-tp1|Qwen/Qwen3.8-27B-FP8|$QWEN_FP8|1|--max-model-len 16384 --gpu-memory-utilization 0.92 --max-num-seqs 128 $NO_MM|$QWEN_BODY|01:50:00"
  "gptoss120b-tp2|openai/gpt-oss-120b|$GPTOSS|2|--max-model-len 16384 --gpu-memory-utilization 0.94|$OSS_BODY|01:20:00"
  "gptoss120b-tp4|openai/gpt-oss-120b|$GPTOSS|4|--max-model-len 16384 --gpu-memory-utilization 0.92|$OSS_BODY|01:10:00"
)
