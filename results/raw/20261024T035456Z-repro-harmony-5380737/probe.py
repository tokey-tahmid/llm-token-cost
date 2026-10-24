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
