"""Closed-loop load generator for OpenAI-compatible streaming endpoints.

    python bench/loadgen.py --base-url http://127.0.0.1:8000/v1 \
        --prompts workloads/chat_1k.jsonl --concurrency 1 4 \
        --requests 16 --warmup 4 --reps 3 --fresh --out results/raw/<run_id>/raw_chat.jsonl

One JSONL line per request (schema: results/README.md). Timing is client-side
with a monotonic clock; warm-up requests are written too, marked
`warmup: true`, so nothing is silently dropped.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import itertools
import json
import time
from pathlib import Path

import httpx


def load_prompts(path: str) -> list[dict]:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    shared: dict[str, str] = {}
    for i, r in enumerate(rows):
        r.setdefault("id", f"p{i:04d}")
        if "messages" not in r:
            r["messages"] = [{"role": "user", "content": r.pop("prompt")}]
        if "system_file" in r:  # shared prefix stored once next to the JSONL
            f = r.pop("system_file")
            if f not in shared:
                shared[f] = (Path(path).parent / f).read_text()
            r["messages"] = [{"role": "system", "content": shared[f]}] + r["messages"]
    return rows


async def one_request(client: httpx.AsyncClient, url: str, body: dict, save_text: bool) -> dict:
    rec: dict = {"t_wall": time.time(), "chunk_t": [], "error": None}
    t0 = time.perf_counter()
    first = first_content = None
    text, reasoning = [], []
    try:
        async with client.stream("POST", url, json=body) as r:
            if r.status_code != 200:
                rec["error"] = f"HTTP {r.status_code}: {(await r.aread())[:300]!r}"
                return rec | {"e2e_s": time.perf_counter() - t0}
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                ev = json.loads(data)
                if ev.get("usage"):
                    rec["usage"] = ev["usage"]
                for ch in ev.get("choices") or []:
                    d = ch.get("delta") or {}
                    c = d.get("content") or ""
                    rc = d.get("reasoning_content") or d.get("reasoning") or ""
                    if c or rc:
                        t = time.perf_counter() - t0
                        rec["chunk_t"].append(round(t, 5))
                        first = t if first is None else first
                        if c and first_content is None:
                            first_content = t
                        text.append(c)
                        reasoning.append(rc)
                    if ch.get("finish_reason"):
                        rec["finish_reason"] = ch["finish_reason"]
    except Exception as ex:  # record, don't retry: failures are results
        rec["error"] = f"{type(ex).__name__}: {ex}"[:300]
    out = "".join(text)
    rec |= {"e2e_s": round(time.perf_counter() - t0, 5),
            "ttft_s": None if first is None else round(first, 5),
            "ttft_content_s": None if first_content is None else round(first_content, 5),
            "n_chunks": len(rec["chunk_t"]),
            "out_chars": len(out), "reasoning_chars": len("".join(reasoning)),
            "out_sha256": hashlib.sha256(out.encode()).hexdigest()[:16]}
    if save_text:
        rec["text"] = out
        rec["reasoning"] = "".join(reasoning)
    return rec


async def run_level(a, prompts: list[dict], conc: int, rep: int, sink, start: int) -> int:
    """Run warm-up + measured requests; returns the next unused prompt index (--fresh)."""
    url = a.base_url.rstrip("/") + "/chat/completions"
    limits = httpx.Limits(max_connections=conc + 2, max_keepalive_connections=conc + 2)
    async with httpx.AsyncClient(timeout=httpx.Timeout(a.timeout, connect=30), limits=limits,
                                 headers={"Authorization": f"Bearer {a.api_key}"}) as client:
        for phase, n in (("warmup", a.warmup), ("measure", a.requests)):
            # --fresh: every request gets a prompt not used earlier in this run (no repeat
            # served from the prefix cache); otherwise rotate the start per rep
            first = start if a.fresh else rep * 7 % len(prompts)
            order = itertools.islice(itertools.cycle(prompts), first, first + n)
            if a.fresh:
                if start + n > len(prompts):
                    print(f"  warning: prompt set exhausted ({len(prompts)}); prompts repeat from here")
                start += n
            queue: asyncio.Queue = asyncio.Queue()
            for i, p in enumerate(order):
                queue.put_nowait((i, p))

            async def worker(w: int):
                while not queue.empty():
                    i, p = queue.get_nowait()
                    body = {"model": a.model, "messages": p["messages"], "stream": True,
                            "stream_options": {"include_usage": True},
                            "max_tokens": p.get("max_tokens", a.max_tokens),
                            "temperature": a.temperature, "seed": a.seed + i}
                    body |= a.extra_body
                    rec = await one_request(client, url, body, a.save_text)
                    rec |= {"run_id": a.run_id, "label": a.label, "concurrency": conc,
                            "rep": rep, "seq": i, "worker": w, "prompt_id": p["id"],
                            "warmup": phase == "warmup", "max_tokens": body["max_tokens"]}
                    sink.write(json.dumps(rec) + "\n")
                    sink.flush()

            t = time.perf_counter()
            await asyncio.gather(*(worker(w) for w in range(conc)))
            print(f"  c={conc} rep={rep} {phase}: {n} req in {time.perf_counter() - t:.1f}s")
    return start


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--model", default="default")
    ap.add_argument("--api-key", default="none")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--concurrency", type=int, nargs="+", default=[1])
    ap.add_argument("--requests", type=int, default=16, help="measured requests per (level, rep)")
    ap.add_argument("--requests-per-conc", type=float, default=0,
                    help="if set, measured requests = max(--requests, this x concurrency)")
    ap.add_argument("--warmup-per-conc", action="store_true", help="warm-up requests = max(--warmup, concurrency)")
    ap.add_argument("--fresh", action="store_true", help="never reuse a prompt within the run")
    ap.add_argument("--warmup", type=int, default=4)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--extra-body", type=json.loads, default={}, help='JSON merged into each request')
    ap.add_argument("--save-text", action="store_true", help="keep outputs (quality evals)")
    ap.add_argument("--label", default="", help="config label, e.g. strata-iq3s-ctx1k")
    ap.add_argument("--run-id", default=time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    prompts = load_prompts(a.prompts)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    nreq, nwarm = a.requests, a.warmup
    nxt = 0
    with open(a.out, "a") as sink:
        for conc in a.concurrency:
            a.requests = max(nreq, int(a.requests_per_conc * conc)) if a.requests_per_conc else nreq
            a.warmup = max(nwarm, conc) if a.warmup_per_conc else nwarm
            for rep in range(1, a.reps + 1):
                nxt = asyncio.run(run_level(a, prompts, conc, rep, sink, nxt))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
