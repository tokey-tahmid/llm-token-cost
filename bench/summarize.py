"""Summarize loadgen JSONL into tidy per-(label, concurrency, rep) stats.

    python bench/summarize.py results/raw/<run_id>/raw_*.jsonl --out results/raw/<run_id>/summary.csv

Per rep: median/p95/p99 of TTFT, TPOT and chunk ITL; throughput over the
measured window. Across reps: median of the per-rep medians, with min–max
spread. Warm-up and errored requests are excluded and
counted.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd


def per_request(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    u = df.get("usage")
    df["completion_tokens"] = [x.get("completion_tokens") if isinstance(x, dict) else np.nan
                               for x in (u if u is not None else [None] * len(df))]
    df["prompt_tokens"] = [x.get("prompt_tokens") if isinstance(x, dict) else np.nan
                           for x in (u if u is not None else [None] * len(df))]
    # TPOT: decode time per token after the first one (needs engine-reported usage)
    df["tpot_s"] = (df["e2e_s"] - df["ttft_s"]) / (df["completion_tokens"] - 1)
    df.loc[df["completion_tokens"] <= 1, "tpot_s"] = np.nan
    df["tokens_per_chunk"] = df["completion_tokens"] / df["n_chunks"].replace(0, np.nan)
    df["t_end_wall"] = df["t_wall"] + df["e2e_s"]
    return df


def q(x: pd.Series, p: float) -> float:
    x = x.dropna()
    return float(np.percentile(x, p)) if len(x) else np.nan


def summarize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    m = df[~df["warmup"]]
    rows = []
    for (label, conc, rep), g in m.groupby(["label", "concurrency", "rep"]):
        ok = g[g["error"].isna()]
        # an output shorter than requested, or without the engine's usage report, means the
        # request did not run as designed (fixed output length); count it, never hide it
        want = ok["max_tokens"] if "max_tokens" in ok else pd.Series(np.nan, index=ok.index)
        incomplete = int((ok["completion_tokens"].isna() | (ok["completion_tokens"] < want)).sum())
        itl = np.concatenate([np.diff(c) for c in ok["chunk_t"] if len(c) > 1] or [np.array([])])
        window = ok["t_end_wall"].max() - ok["t_wall"].min() if len(ok) else np.nan
        rows.append({
            "label": label, "concurrency": conc, "rep": rep,
            "n_ok": len(ok), "n_err": int(g["error"].notna().sum()), "n_incomplete": incomplete,
            "prompt_tokens_med": ok["prompt_tokens"].median(),
            "completion_tokens_med": ok["completion_tokens"].median(),
            "ttft_med_s": q(ok["ttft_s"], 50), "ttft_p95_s": q(ok["ttft_s"], 95), "ttft_p99_s": q(ok["ttft_s"], 99),
            "tpot_med_ms": q(ok["tpot_s"], 50) * 1e3, "tpot_p95_ms": q(ok["tpot_s"], 95) * 1e3,
            "tpot_p99_ms": q(ok["tpot_s"], 99) * 1e3,
            "itl_med_ms": q(pd.Series(itl), 50) * 1e3, "itl_p95_ms": q(pd.Series(itl), 95) * 1e3,
            "itl_p99_ms": q(pd.Series(itl), 99) * 1e3,
            "tokens_per_chunk_med": ok["tokens_per_chunk"].median(),
            "decode_tok_s_user": 1.0 / q(ok["tpot_s"], 50) if len(ok) else np.nan,
            "throughput_out_tok_s": ok["completion_tokens"].sum() / window if window else np.nan,
            "e2e_med_s": q(ok["e2e_s"], 50),
        })
    per_rep = pd.DataFrame(rows)
    agg_cols = ["ttft_med_s", "ttft_p95_s", "tpot_med_ms", "tpot_p95_ms", "itl_p99_ms",
                "decode_tok_s_user", "throughput_out_tok_s"]
    g = per_rep.groupby(["label", "concurrency"])
    across = g[agg_cols].median().add_suffix("")
    lo, hi = g[agg_cols].min().add_suffix("_min"), g[agg_cols].max().add_suffix("_max")
    across = across.join(lo).join(hi).join(g["rep"].count().rename("reps")).join(
        g["n_err"].sum().rename("errors")).join(g["n_incomplete"].sum().rename("incomplete")).join(
        (g["n_ok"].sum() + g["n_err"].sum()).rename("requests")).reset_index()
    return per_rep, across


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", help="CSV for the across-rep summary; per-rep goes next to it")
    a = ap.parse_args()
    read = lambda f: gzip.open(f, "rt").read() if str(f).endswith(".gz") else Path(f).read_text()
    recs = [json.loads(l) for f in a.files for l in read(f).splitlines() if l.strip()]  # .jsonl or .jsonl.gz
    df = per_request(pd.DataFrame(recs))
    per_rep, across = summarize(df)
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.precision", 3):
        print(across[["label", "concurrency", "reps", "errors", "ttft_med_s", "ttft_p95_s",
                      "tpot_med_ms", "tpot_p95_ms", "decode_tok_s_user", "decode_tok_s_user_min",
                      "decode_tok_s_user_max", "throughput_out_tok_s"]].to_string(index=False))
    if a.out:
        across.to_csv(a.out, index=False)
        per_rep.to_csv(Path(a.out).with_name(Path(a.out).stem + "_per_rep.csv"), index=False)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
