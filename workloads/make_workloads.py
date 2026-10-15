"""Build the benchmark workloads from public-domain books (Project Gutenberg).

    python workloads/fetch_sources.py && python workloads/make_workloads.py workloads/sources --out workloads

chat_1k.jsonl   instruction first, then a ~3.4K-char passage (~1K tokens in), 400 out
agent_8k.jsonl  shared ~6K-token system prefix (tool schemas + reference text), then a
                task + ~6K-char passage (~1.6K tokens), 300 out
long_8k1k.jsonl instruction, then a 29K-char passage (~8K tokens in), 1,024 out, no shared prefix:
                the shape of SemiAnalysis InferenceX's published "8k/1k" runs, so our A100 numbers
                can sit next to their H100/B200 numbers on the same workload shape. Passages start
                every 9,700 chars (overlapping text, distinct starts) so a sweep never repeats a prompt.

The task comes before the passage so prompts diverge early: only the agent
workload's system prefix is meant to hit the prefix cache. Prompts are unique
within a run (loadgen --fresh), so no request is served from a cached repeat.
Token counts are approximate here; the engine's usage report is the measurement.
"""
from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

CHAT_TASKS = [
    "Summarize the following passage in about 250 words, then list three questions it leaves open.",
    "Explain what happens in the following passage to someone who has not read the book, in plain modern English.",
    "Describe the characters who appear in the following passage and what each one wants.",
    "Rewrite the following passage as a short news report, keeping every factual detail.",
    "List the key events of the following passage in order, one line each, then give a one-paragraph verdict on its tone.",
]
AGENT_TASKS = [
    "Use the tools if needed. Produce a structured research note on the excerpt below: themes, people, places.",
    "Use the tools if needed. Extract every named person and place from the excerpt below as a JSON list.",
    "Use the tools if needed. Write a three-step plan to fact-check the excerpt below against the reference library.",
    "Use the tools if needed. Compare the excerpt below with the reference library and note one similarity and one difference.",
    "Use the tools if needed. Draft a short index entry for the excerpt below, with a title and five keywords.",
    "Use the tools if needed. Classify the excerpt below by genre and mood, and justify each label in one sentence.",
]
TOOLS = [
    {"name": "search_documents", "description": "Full-text search over the reference library.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                    "required": ["query"]}},
    {"name": "read_document", "description": "Read a document from the library by id.",
     "parameters": {"type": "object", "properties": {"doc_id": {"type": "string"}, "offset": {"type": "integer"}},
                    "required": ["doc_id"]}},
    {"name": "write_note", "description": "Save a research note.",
     "parameters": {"type": "object", "properties": {"title": {"type": "string"}, "body": {"type": "string"},
                                                     "tags": {"type": "array", "items": {"type": "string"}}},
                    "required": ["title", "body"]}},
    {"name": "lookup_entity", "description": "Look up a person or place in the catalogue.",
     "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "kind": {"enum": ["person", "place"]}},
                    "required": ["name"]}},
]


def body(path: Path) -> str:
    t = path.read_text(encoding="utf-8", errors="ignore").replace("\r\n", "\n")
    s = re.search(r"\*\*\* ?START OF (THE|THIS) PROJECT GUTENBERG EBOOK.*?\*\*\*", t)
    e = re.search(r"\*\*\* ?END OF (THE|THIS) PROJECT GUTENBERG EBOOK", t)
    t = t[s.end() if s else 0: e.start() if e else len(t)]
    return re.sub(r"\n{3,}", "\n\n", t).strip()


def passages(texts: list[str], chars: int) -> list[str]:
    out = []
    for t in texts:
        paras = [p.strip() for p in t.split("\n\n") if len(p.strip()) > 40]
        cur = ""
        for p in paras:
            cur = f"{cur}\n\n{p}" if cur else p
            if len(cur) >= chars:
                out.append(cur[: int(chars * 1.15)])
                cur = ""
    return out


def windows(texts: list[str], chars: int, stride: int) -> list[str]:
    """Exactly `chars` characters starting every `stride` characters, at paragraph starts."""
    out = []
    for t in texts:
        starts = [0] + [m.end() for m in re.finditer(r"\n\n", t)]
        nxt = 0
        for st in starts:
            if st >= nxt and st + chars <= len(t):
                out.append(t[st: st + chars])
                nxt = st + stride
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory with Gutenberg pg*.txt files")
    ap.add_argument("--out", default="workloads")
    a = ap.parse_args()
    files = sorted(Path(a.src).glob("pg*.txt"))
    texts = [body(f) for f in files]
    rng = random.Random(20261014)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    chat = [(t, p) for p in passages(texts, 3400) for t in CHAT_TASKS]
    rng.shuffle(chat)
    with open(out / "chat_1k.jsonl", "w") as f:
        for i, (task, p) in enumerate(chat):
            f.write(json.dumps({"id": f"chat{i:05d}", "max_tokens": 400,
                                "messages": [{"role": "user", "content": f"{task}\n\n---\n{p}"}]}) + "\n")

    library = passages(texts[-2:], 22000)[0]  # fixed reference text for the shared prefix
    system = ("You are a careful research assistant working over a reference library. "
              "You may call these tools (JSON schema):\n" + json.dumps(TOOLS, indent=1) +
              "\n\nReference library, document LIB-001:\n" + library)
    agent = [(t, p) for p in passages(texts[:-2], 6000) for t in AGENT_TASKS]
    rng.shuffle(agent)
    (out / "agent_system.txt").write_text(system)
    with open(out / "agent_8k.jsonl", "w") as f:
        for i, (task, p) in enumerate(agent):
            f.write(json.dumps({"id": f"agent{i:05d}", "max_tokens": 300, "system_file": "agent_system.txt",
                                "messages": [{"role": "user", "content": f"{task}\n\n---\n{p}"}]}) + "\n")
    # own RNG, generated last: chat_1k and agent_8k stay byte-identical to earlier runs
    rng_long = random.Random(20261015)
    long = [(t, p) for p in windows(texts, 29000, 9700) for t in CHAT_TASKS]
    rng_long.shuffle(long)
    with open(out / "long_8k1k.jsonl", "w") as f:
        for i, (task, p) in enumerate(long):
            f.write(json.dumps({"id": f"long{i:05d}", "max_tokens": 1024,
                                "messages": [{"role": "user", "content": f"{task}\n\n---\n{p}"}]}) + "\n")
    meta = {"sources": [f.name for f in files], "chat_prompts": len(chat), "agent_prompts": len(agent),
            "long_prompts": len(long),
            "system_prefix_chars": len(system), "license": "Project Gutenberg texts (public domain); task text and tool schemas are ours",
            "seed": 20261014}
    import hashlib
    meta["sha256"] = {n: hashlib.sha256((out / n).read_bytes()).hexdigest()[:16]
                      for n in ("chat_1k.jsonl", "agent_8k.jsonl", "agent_system.txt", "long_8k1k.jsonl")}
    (out / "workloads_meta.json").write_text(json.dumps(meta, indent=1))
    print(meta)


if __name__ == "__main__":
    main()
