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
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
# workload file stem -> short name, in the order slurm/sweep.sbatch runs them
WORKLOADS = {"chat_1k": "chat", "agent_8k": "agent", "long_8k1k": "long"}


def load_runs(raw_dir: Path) -> pd.DataFrame:
    rows = []
    for run in sorted(p for p in raw_dir.iterdir() if (p / "summary.csv").exists() and "-smoke-" not in p.name):
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
        s["engine"] = eng.get("name", "vLLM")
        s["workload"] = s["label"].str.rsplit("-", n=1).str[-1].map(WORKLOADS)
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
    p = load_power(run)
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


def load_power(run: Path) -> pd.DataFrame:
    """power.csv with a UTC epoch column t, restricted to the configuration's GPUs."""
    p = pd.read_csv(run / "power.csv", header=None, names=["ts", "gpu", "watts", "util", "mem"], skipinitialspace=True)
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
    return p


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


def prom_total(path: Path, name: str) -> float:
    """Sum of a Prometheus counter over its label sets (NaN if absent)."""
    if not path.exists():
        return np.nan
    vals = [float(l.rsplit(" ", 1)[1]) for l in path.read_text().splitlines()
            if l.startswith(name + "{") or l.startswith(name + " ")]
    return sum(vals) if vals else np.nan


def run_facts(run: Path) -> list[dict]:
    """Per run and workload: KV cache size, prefix-cache hit rate, idle power.
    Counters are cumulative over the server's life and the workloads run in the order
    below, so each workload's share is the difference from the previous snapshot.
    Idle power: the config's GPUs between server ready and the first request (an
    explicit idle window, idle_from/idle_to in job.txt, when the run recorded one)."""
    cfg = yaml.safe_load((run / "config.yaml").read_text()) if (run / "config.yaml").exists() else {}
    logf = next(iter(sorted(run.glob("vllm.log*")) + sorted(run.glob("server.log*"))), None)
    kv = np.nan
    if logf:  # vLLM: "GPU KV cache size: N tokens"; SGLang: "max_total_num_tokens=N"
        text = gzip.open(logf, "rt").read() if logf.suffix == ".gz" else logf.read_text()
        m = re.search(r"GPU KV cache size: ([0-9,]+) tokens|max_total_num_tokens=([0-9]+)", text)
        kv = float((m.group(1) or m.group(2)).replace(",", "")) if m else np.nan
    rows, prev_q, prev_h = [], 0.0, 0.0
    for wl in WORKLOADS:
        prom = run / f"metrics_after_{wl}.prom"
        if not prom.exists():
            continue
        q, h = prom_total(prom, "vllm:prefix_cache_queries_total"), prom_total(prom, "vllm:prefix_cache_hits_total")
        rows.append({"run_id": run.name, "config": cfg.get("label", run.name), "workload": WORKLOADS[wl],
                     "kv_cache_tokens": kv, "prefix_cache_hit_rate": (h - prev_h) / (q - prev_q) if q > prev_q else np.nan})
        if not np.isnan(q):
            prev_q, prev_h = q, h
    idle_w, n = idle_power(run)
    for r in rows:
        r["idle_power_w"], r["idle_samples_per_gpu"] = idle_w, n
    return rows


def idle_power(run: Path) -> tuple[float, int]:
    job, pw = run / "job.txt", run / "power.csv"
    if not (job.exists() and pw.exists()):
        return np.nan, 0
    kv = dict(w.split("=", 1) for w in job.read_text().split() if "=" in w)
    if "idle_from" in kv:
        t0, t1 = float(kv["idle_from"]), float(kv["idle_to"])
    else:
        words = job.read_text().split()
        start = next((w for w in words if w.endswith("Z") and "T" in w), None)
        raws = sorted(run.glob("raw_*.jsonl*"))
        if not (start and "startup_s" in kv and raws):
            return np.nan, 0
        t0 = pd.Timestamp(start).timestamp() + float(kv.get("download_s", 0)) + float(kv["startup_s"]) + 2
        read = lambda f: gzip.open(f, "rt").read() if f.suffix == ".gz" else f.read_text()
        t1 = min(json.loads(l)["t_wall"] for f in raws for l in read(f).splitlines() if l.strip()) - 1
    p = load_power(run)
    w = p[(p["t"] >= t0) & (p["t"] <= t1)]
    if w.empty:
        return np.nan, 0
    return float(w.groupby("gpu")["watts"].mean().sum()), int(w.groupby("gpu").size().min())


def api_endpoints(prices_dir: Path, route: str, day: str | None = None) -> pd.DataFrame:
    """Live endpoints of one model from a dated snapshot (scripts/fetch_api_prices.py), USD per
    token. Live: status 0 and, when reported, at least 95% uptime over the last 30 minutes."""
    files = sorted(prices_dir.glob(f"*-openrouter-{route.replace('/', '--')}.json"))
    if day:
        files = [f for f in files if f.name.startswith(day)]
    if not files:
        return pd.DataFrame()
    eps = json.loads(files[-1].read_text()).get("data", {}).get("endpoints", [])
    num = lambda p, k: float(p[k]) if p.get(k) not in (None, "") else np.nan
    rows = [{"provider": e.get("provider_name"), "tag": e.get("tag"), "quantization": e.get("quantization"),
             "input": num(e["pricing"], "prompt"), "output": num(e["pricing"], "completion"),
             "cache_read": num(e["pricing"], "input_cache_read"), "snapshot": files[-1].name}
            for e in eps if e.get("status", 0) == 0 and (e.get("uptime_last_30m") is None or e["uptime_last_30m"] >= 95)]
    return pd.DataFrame(rows)


def api_bills(ep: pd.DataFrame, tok_in: float, tok_out: float, hit: float = 0.0) -> pd.Series:
    """Each endpoint's bill for one request, per 1M output tokens. hit: share of the prompt billed
    at the endpoint's cache-read price (its input price when it lists none)."""
    cache = ep["cache_read"].fillna(ep["input"])
    bill = tok_in * ((1 - hit) * ep["input"] + hit * cache) + tok_out * ep["output"]
    return bill / tok_out * 1e6


def price_table(df: pd.DataFrame, prices: dict, endpoints: dict, hits: dict) -> pd.DataFrame:
    gp, api = prices["gpu_list_prices"], prices["api_list_prices"]["models"]
    slo = prices["slo"]
    out = df.copy()
    for k in ("low", "mid", "high"):
        usd_h = gp[k]["usd_per_gpu_hour"] * out["tp"]
        out[f"usd_per_m_out_{k}"] = usd_h / (out["throughput_out_tok_s"] * 3600) * 1e6
        req_per_h = out["throughput_out_tok_s"] * 3600 / out["completion_tokens_med"]
        out[f"usd_per_req_self_{k}"] = usd_h / req_per_h
    # the price named in the brief (one route, recorded before the runs): input + output for the same requests
    p_in = out["model"].map(lambda m: api.get(m, {}).get("input", np.nan))
    p_out = out["model"].map(lambda m: api.get(m, {}).get("output", np.nan))
    out["usd_per_req_api"] = (out["prompt_tokens_med"] * p_in + out["completion_tokens_med"] * p_out) / 1e6
    out["api_usd_per_m_out"] = p_out  # output-token list price alone
    out["api_usd_per_m_out_named"] = out["usd_per_req_api"] / out["completion_tokens_med"] * 1e6
    # the market: every live provider's bill for the same requests, per 1M output tokens
    for col in ("cheapest", "median", "priciest", "median_cached"):
        out[f"api_usd_per_m_out_{col}"] = np.nan
    out["api_endpoints"] = 0
    for i, r in out.iterrows():
        ep = endpoints.get(r["model"], pd.DataFrame())
        if ep.empty:
            continue
        b = api_bills(ep, r["prompt_tokens_med"], r["completion_tokens_med"])
        out.loc[i, ["api_usd_per_m_out_cheapest", "api_usd_per_m_out_median", "api_usd_per_m_out_priciest"]] = \
            [b.min(), b.median(), b.max()]
        h = hits.get((r["config"], r["workload"]), 0.0)
        out.loc[i, "api_usd_per_m_out_median_cached"] = api_bills(ep, r["prompt_tokens_med"], r["completion_tokens_med"],
                                                                  0.0 if np.isnan(h) else h).median()
        out.loc[i, "api_endpoints"] = len(ep)
    # input + output tokens per GPU per second: the unit published GPU comparisons (InferenceX) use
    out["total_tok_s_per_gpu"] = (out["throughput_out_tok_s"] * (out["prompt_tokens_med"] + out["completion_tokens_med"])
                                  / out["completion_tokens_med"] / out["tp"])
    out["within_slo"] = (out["ttft_p95_s"] <= slo["ttft_p95_s"]) & (out["tpot_p95_ms"] <= slo["tpot_p95_ms"])
    eur_kwh = prices["electricity"]["eur_per_kwh"]
    out["joules_per_out_token"] = out["power_w"] / out["throughput_out_tok_s"]
    out["eur_electricity_per_m_out"] = out["joules_per_out_token"] * 1e6 / 3.6e6 * eur_kwh
    return out


def monthly(df: pd.DataFrame, prices: dict, endpoints: dict) -> pd.DataFrame:
    """Per config, workload and API reference: capacity within the SLO, monthly rent of the GPUs,
    break-even volume, and whether that volume fits in the capacity. API references: the price
    named in the brief, and the cheapest and median live provider for the workload's token mix."""
    gp, api, hours = prices["gpu_list_prices"], prices["api_list_prices"]["models"], prices["hours_per_month"]
    rows = []
    for (config, workload), g in df.groupby(["config", "workload"]):
        ok = g[g["within_slo"]]
        cap = ok["throughput_out_tok_s"].max() if len(ok) else 0.0
        m = g["model"].iloc[0]
        tok_in, tok_out = g["prompt_tokens_med"].median(), g["completion_tokens_med"].median()
        refs = {"named": (tok_in * api[m]["input"] + tok_out * api[m]["output"]) / tok_out / 1e6}
        ep = endpoints.get(m, pd.DataFrame())
        if not ep.empty:
            b = api_bills(ep, tok_in, tok_out) / 1e6
            refs |= {"cheapest": b.min(), "median": b.median()}
        for ref, blend in refs.items():
            row = {"config": config, "workload": workload, "model": m, "tp": int(g["tp"].iloc[0]), "api_ref": ref,
                   "capacity_out_tok_s_within_slo": cap, "capacity_out_tok_per_month": cap * 3600 * hours,
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
    ap.add_argument("--api-prices", default=str(ROOT / "results/prices"), help="dated provider snapshots")
    ap.add_argument("--api-day", default=None, help="snapshot date YYYY-MM-DD (default: latest)")
    a = ap.parse_args()
    prices = yaml.safe_load(Path(a.prices).read_text())
    df, repeats = split_repeats(load_runs(Path(a.raw)))
    facts = pd.DataFrame([f for run in sorted(Path(a.raw).iterdir())
                          if (run / "summary.csv").exists() and "-smoke-" not in run.name for f in run_facts(run)])
    hits = {(r.config, r.workload): r.prefix_cache_hit_rate for r in facts.itertuples()} if len(facts) else {}
    routes = {m: v["route"] for m, v in prices["api_list_prices"]["models"].items()}
    endpoints = {m: api_endpoints(Path(a.api_prices), route, a.api_day) for m, route in routes.items()}
    df = price_table(df, prices, endpoints, hits)
    mon = monthly(df, prices, endpoints)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "cost_per_level.csv", index=False)
    mon.to_csv(out / "monthly_breakeven.csv", index=False)
    facts.to_csv(out / "run_facts.csv", index=False)
    print(facts.round(3).to_string(index=False), "\n")
    if len(repeats):
        cols = ["config", "workload", "concurrency", "run_id", "throughput_out_tok_s", "ttft_p95_s", "tpot_p95_ms", "power_w"]
        repeats[cols].sort_values(cols[:4]).to_csv(out / "anchor_repeats.csv", index=False)
        print(repeats[cols].sort_values(cols[:4]).to_string(index=False), "\n")
    with pd.option_context("display.width", 220, "display.max_columns", 20, "display.precision", 3):
        print(df[["config", "workload", "concurrency", "throughput_out_tok_s", "decode_tok_s_user", "within_slo",
                  "usd_per_m_out_low", "usd_per_m_out_mid", "usd_per_m_out_high", "api_usd_per_m_out_named",
                  "api_usd_per_m_out_cheapest", "api_usd_per_m_out_median", "api_usd_per_m_out_median_cached"]].to_string(index=False))
        print()
        print(mon.to_string(index=False))
    snaps = sorted({e["snapshot"].iloc[0] for e in endpoints.values() if not e.empty})
    print(f"\nwrote {out}/cost_per_level.csv and monthly_breakeven.csv (GPU prices as of "
          f"{prices['gpu_list_prices']['as_of']}; named API prices as of {prices['api_list_prices']['as_of']}; "
          f"provider snapshots {snaps})")


if __name__ == "__main__":
    main()
