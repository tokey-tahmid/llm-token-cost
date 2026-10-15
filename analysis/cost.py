"""Cost per token from measured sweeps: per request, per million tokens, per month.

    python analysis/cost.py                      # reads results/raw/*, writes results/tables/
    python analysis/cost.py --raw tests/fixtures/HARNESS_SELFTEST --out /tmp/x   # harness check only

Inputs per run folder: summary.csv (bench/summarize.py), config.yaml, power.csv, raw_*.jsonl[.gz].
Prices: configs/prices.yaml (dated, sourced). Nothing here estimates a measurement:
throughput, latency and power come from the run; prices are labelled list prices.

Two framings:
  per request  self-hosted GPU time per request at each load, priced at the list-price
               range, against the API list price for the same input/output tokens
  per month    renting the GPUs full-time is a fixed cost; the API cost grows with volume.
               Break-even volume = fixed cost / API price per token; it only counts if the
               measured capacity within the SLO can serve that volume.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_runs(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(p for p in raw_dir.iterdir() if (p / "summary.csv").exists()):
        cfg = yaml.safe_load((run / "config.yaml").read_text()) if (run / "config.yaml").exists() else {}
        s = pd.read_csv(run / "summary.csv")
        per_rep = pd.read_csv(run / "summary_per_rep.csv") if (run / "summary_per_rep.csv").exists() else None
        if per_rep is not None:  # token counts per request, for per-request pricing
            tok = per_rep.groupby(["label", "concurrency"])[["prompt_tokens_med", "completion_tokens_med"]].median()
            s = s.merge(tok, left_on=["label", "concurrency"], right_index=True, how="left")
        eng = cfg.get("engine", {}) or {}
        s["run_id"] = run.name
        s["model"] = (cfg.get("model") or {}).get("repo")
        s["tp"] = int(eng.get("tp") or 1)
        s["config"] = cfg.get("label", run.name)
        s["workload"] = s["label"].str.rsplit("-", n=1).str[-1].str.replace("_1k", "").str.replace("_8k", "")
        s["power_w"] = [mean_power(run, r) for r in s.itertuples()]
        rows.append(s)
    if not rows:
        raise SystemExit(f"no runs with summary.csv under {raw_dir}")
    return pd.concat(rows, ignore_index=True)


def split_repeats(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Knee runs (RUN_TAG in slurm/sweep.sbatch) add concurrency levels to a config and
    repeat a few levels as anchors. The first run of each (config, workload, concurrency)
    is the reported value; every row of a repeated level goes to the second frame, a
    cross-run (and usually cross-node) reproducibility check."""
    key = ["config", "workload", "concurrency"]
    df = df.sort_values("run_id", kind="stable")
    return df[~df.duplicated(key, keep="first")].reset_index(drop=True), df[df.duplicated(key, keep=False)]


def mean_power(run: Path, r) -> float:
    """Mean board power (all GPUs in the config) over this (label, concurrency) window."""
    pw, raw = run / "power.csv", list(run.glob(f"raw_*{r.label.rsplit('-', 1)[-1]}*.jsonl*"))
    if not pw.exists() or not raw:
        return np.nan
    p = pd.read_csv(pw, header=None, names=["ts", "gpu", "watts", "util", "mem"], skipinitialspace=True)
    gpus = job_gpus(run)              # the node's other GPUs are sampled too; count only this config's
    if gpus:
        p = p[p["gpu"].isin(gpus)]
    epoch = pd.to_numeric(p["ts"], errors="coerce")
    if epoch.notna().all():          # current format: UTC epoch seconds
        p["t"] = epoch
    else:                            # earlier format: nvidia-smi local timestamp, no zone
        naive = pd.to_datetime(p["ts"], format="%Y/%m/%d %H:%M:%S.%f", errors="coerce")
        # seconds via Timedelta: astype("int64") depends on the parsed resolution (ns, us or ms)
        p["t"] = (naive - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        p = p.dropna(subset=["t"])
        p["t"] -= local_offset_s(run, p["t"].min())
    read = lambda f: gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text()  # committed runs are gzipped
    recs = [json.loads(l) for f in raw for l in read(f).splitlines() if l.strip()]
    w = [x for x in recs if x.get("concurrency") == r.concurrency and not x.get("warmup")]
    if not w:
        return np.nan
    t0, t1 = min(x["t_wall"] for x in w), max(x["t_wall"] + x["e2e_s"] for x in w)
    win = p[(p["t"] >= t0) & (p["t"] <= t1)]
    if win.empty:
        return np.nan
    # mean over time per GPU, then sum over GPUs (each GPU's row carries its own ms timestamp)
    return float(win.groupby("gpu")["watts"].mean().sum())


def job_gpus(run: Path) -> list[int]:
    """GPU indices the server ran on, from job.txt's gpus=0,1 (empty if not recorded)."""
    job = run / "job.txt"
    if not job.exists():
        return []
    w = next((w for w in job.read_text().split() if w.startswith("gpus=")), "")
    return [int(g) for g in w.split("=", 1)[-1].split(",") if g.strip().isdigit()]


def local_offset_s(run: Path, first_sample_naive_s: float) -> float:
    """nvidia-smi stamps power samples in the node's local time without a zone; the
    request log uses UTC epochs. job.txt records the job start in UTC and the
    weight download time; power sampling starts right after the download, so the
    offset is (first sample - (start + download)) rounded to the nearest quarter
    hour (time zones are whole quarter hours)."""
    job = run / "job.txt"
    if not job.exists():
        return 0.0
    words = job.read_text().split()
    start = next((w for w in words if w.endswith("Z") and "T" in w), None)
    if not start:
        return 0.0
    dl = next((float(w.split("=", 1)[1]) for w in words if w.startswith("download_s=")), 0.0)
    utc = pd.Timestamp(start).timestamp() + dl
    return round((first_sample_naive_s - utc) / 900) * 900.0


def price_table(df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    gp, api = prices["gpu_list_prices"], prices["api_list_prices"]["models"]
    slo = prices["slo"]
    out = df.copy()
    for k in ("low", "mid", "high"):
        usd_h = gp[k]["usd_per_gpu_hour"] * out["tp"]
        out[f"usd_per_m_out_{k}"] = usd_h / (out["throughput_out_tok_s"] * 3600) * 1e6
        req_per_h = out["throughput_out_tok_s"] * 3600 / out["completion_tokens_med"]
        out[f"usd_per_req_self_{k}"] = usd_h / req_per_h
    p_in = out["model"].map(lambda m: api.get(m, {}).get("input", np.nan))
    p_out = out["model"].map(lambda m: api.get(m, {}).get("output", np.nan))
    out["usd_per_req_api"] = (out["prompt_tokens_med"] * p_in + out["completion_tokens_med"] * p_out) / 1e6
    out["api_usd_per_m_out"] = p_out  # output-token list price alone
    # what the API bills for the same requests (input + output) per 1M output tokens: the
    # like-for-like comparison, since the self-hosted GPU time also covers the prefill
    out["api_usd_per_m_out_blended"] = out["usd_per_req_api"] / out["completion_tokens_med"] * 1e6
    out["within_slo"] = (out["ttft_p95_s"] <= slo["ttft_p95_s"]) & (out["tpot_p95_ms"] <= slo["tpot_p95_ms"])
    eur_kwh = prices["electricity"]["eur_per_kwh"]
    out["joules_per_out_token"] = out["power_w"] / out["throughput_out_tok_s"]
    out["eur_electricity_per_m_out"] = out["joules_per_out_token"] * 1e6 / 3.6e6 * eur_kwh
    return out


def monthly(df: pd.DataFrame, prices: dict) -> pd.DataFrame:
    """Per config and workload: capacity within SLO, fixed monthly cost, break-even volume."""
    gp, api, hours = prices["gpu_list_prices"], prices["api_list_prices"]["models"], prices["hours_per_month"]
    rows = []
    for (config, workload), g in df.groupby(["config", "workload"]):
        ok = g[g["within_slo"]]
        cap = ok["throughput_out_tok_s"].max() if len(ok) else 0.0
        m = g["model"].iloc[0]
        tok_in, tok_out = g["prompt_tokens_med"].median(), g["completion_tokens_med"].median()
        # API price per output token, carrying the input tokens each output token comes with
        blend = (tok_in * api[m]["input"] + tok_out * api[m]["output"]) / tok_out / 1e6
        row = {"config": config, "workload": workload, "model": m, "tp": int(g["tp"].iloc[0]),
               "capacity_out_tok_s_within_slo": cap,
               "capacity_out_tok_per_month": cap * 3600 * hours,
               "api_usd_per_out_token_blended": blend}
        for k in ("low", "mid", "high"):
            fixed = gp[k]["usd_per_gpu_hour"] * row["tp"] * hours
            row[f"fixed_usd_per_month_{k}"] = fixed
            be = fixed / blend
            row[f"breakeven_out_tok_per_month_{k}"] = be
            row[f"self_host_can_win_{k}"] = bool(cap and be <= row["capacity_out_tok_per_month"])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "results/raw"))
    ap.add_argument("--prices", default=str(ROOT / "configs/prices.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/tables"))
    a = ap.parse_args()
    prices = yaml.safe_load(Path(a.prices).read_text())
    df, repeats = split_repeats(load_runs(Path(a.raw)))
    df = price_table(df, prices)
    mon = monthly(df, prices)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cost_per_level.csv", index=False)
    mon.to_csv(out / "monthly_breakeven.csv", index=False)
    if len(repeats):
        cols = ["config", "workload", "concurrency", "run_id", "throughput_out_tok_s", "ttft_p95_s", "tpot_p95_ms", "power_w"]
        repeats[cols].sort_values(cols[:4]).to_csv(out / "anchor_repeats.csv", index=False)
        print(repeats[cols].sort_values(cols[:4]).to_string(index=False), "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 3):
        print(df[["config", "workload", "concurrency", "throughput_out_tok_s", "decode_tok_s_user", "within_slo",
                  "usd_per_m_out_low", "usd_per_m_out_mid", "usd_per_m_out_high", "api_usd_per_m_out_blended",
                  "power_w"]].to_string(index=False))
        print()
        print(mon.to_string(index=False))
    print(f"\nwrote {out}/cost_per_level.csv and monthly_breakeven.csv "
          f"(GPU prices as of {prices['gpu_list_prices']['as_of']}, API prices as of {prices['api_list_prices']['as_of']})")


if __name__ == "__main__":
    main()
