"""Figures for the token-cost results, regenerated from results/tables only.

    python analysis/cost.py && python analysis/figures.py          # results/figures/*.png
    python analysis/figures.py --tables /tmp/x --out /tmp/x/fig     # harness check

1. cost_per_m_<workload>.png   cost per 1M output tokens vs concurrency; band = list-price
                               range, line = mid price; dashed = API list price
2. monthly_<workload>.png      monthly cost vs monthly output volume: API grows with volume,
                               a rented node is flat up to its measured capacity within the SLO
3. user_speed_<workload>.png   per-user decode speed vs concurrency, against the 20 tok/s line
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import style  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def footer(prices: dict, extra: str = "") -> str:
    g, a = prices["gpu_list_prices"], prices["api_list_prices"]
    return (f"A100-SXM4-40GB · vLLM · medians of 3 reps · measured\n"
            f"GPU time priced at public list rates ${g['low']['usd_per_gpu_hour']}–{g['high']['usd_per_gpu_hour']}"
            f"/GPU-h ({g['as_of']}) · API list prices ({a['as_of']}){extra}")


def concurrency_axis(ax, values) -> None:
    """Log2 x axis labelled with the measured concurrency levels as plain numbers."""
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_locator(FixedLocator(sorted(set(values))))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(NullLocator())


def dollar_log_axis(ax) -> None:
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.2f}" if v < 1 else f"${v:,.0f}"))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ""))


def cost_per_m(df, prices, workload, out):
    d = df[df["workload"] == workload]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.84, top=0.8, bottom=0.2)
    for i, (cfg, g) in enumerate(d.groupby("config", sort=False)):
        g = g.sort_values("concurrency")
        c = style.SLOTS[i]
        ax.fill_between(g["concurrency"], g["usd_per_m_out_low"], g["usd_per_m_out_high"], color=c, alpha=0.12, lw=0)
        ax.plot(g["concurrency"], g["usd_per_m_out_mid"], color=c, marker="o", label=cfg)
        last = g.iloc[-1]
        ax.annotate(cfg, (last["concurrency"], last["usd_per_m_out_mid"]), xytext=(12, 0),
                    textcoords="offset points", va="center", color=style.ink("ink2"), fontsize=22)
    for m, g in d.groupby("model"):
        api = g["api_usd_per_m_out"].iloc[0]
        ax.axhline(api, color=style.ink("ink2"), lw=2, ls=(0, (6, 4)))
        ax.annotate(f"API: {m.split('/')[-1]} ${api:.2f}/M", (d["concurrency"].min(), api), xytext=(4, 8),
                    textcoords="offset points", color=style.ink("ink2"), fontsize=22)
    concurrency_axis(ax, d["concurrency"])
    dollar_log_axis(ax)
    ax.set_xlabel("Concurrent users (log scale)")
    ax.set_ylabel("USD per 1M output tokens (log scale)")
    style.titles(fig, ax, f"What a million output tokens costs, {workload} workload",
                 "Band: cheapest to priciest public A100 rate. Dashed: API list price.")
    style.source_line(fig, footer(prices))
    fig.savefig(out / f"cost_per_m_{workload}.png")
    plt.close(fig)


def monthly(mon, prices, workload, out):
    d = mon[mon["workload"] == workload]
    if d.empty:
        return
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.84, top=0.8, bottom=0.2)
    vmax = max(d["capacity_out_tok_per_month"].max(), d["breakeven_out_tok_per_month_mid"].max()) * 1.3
    x = np.linspace(0, vmax, 200)
    for i, r in enumerate(d.itertuples()):
        c = style.SLOTS[i]
        ax.plot(x / 1e9, x * r.api_usd_per_out_token_blended, color=c, ls=(0, (6, 4)), lw=2)
        cap = r.capacity_out_tok_per_month
        ax.plot([0, cap / 1e9], [r.fixed_usd_per_month_mid] * 2, color=c, lw=4)
        ax.plot(cap / 1e9, r.fixed_usd_per_month_mid, "o", color=c)
        ax.annotate(f"{r.config}: node rented 24/7 (mid price), up to measured capacity",
                    (cap / 1e9, r.fixed_usd_per_month_mid), xytext=(10, 10), textcoords="offset points",
                    color=style.ink("ink2"), fontsize=20)
    ax.set_xlabel("Output tokens per month (billions)")
    ax.set_ylabel("USD per month")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    style.titles(fig, ax, f"When does renting a node beat the API? ({workload})",
                 "Solid: rented node, flat until it runs out of capacity within the SLO. Dashed: API cost at that volume.")
    style.source_line(fig, footer(prices, "\ncapacity = max throughput meeting TTFT p95 ≤ 2 s and ≥ 20 tok/s per user"))
    fig.savefig(out / f"monthly_{workload}.png")
    plt.close(fig)


def user_speed(df, prices, workload, out):
    d = df[df["workload"] == workload]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.84, top=0.8, bottom=0.2)
    for i, (cfg, g) in enumerate(d.groupby("config", sort=False)):
        g = g.sort_values("concurrency")
        ax.plot(g["concurrency"], g["decode_tok_s_user"], color=style.SLOTS[i], marker="o", label=cfg)
        last = g.iloc[-1]
        ax.annotate(cfg, (last["concurrency"], last["decode_tok_s_user"]), xytext=(12, 0), textcoords="offset points",
                    va="center", color=style.ink("ink2"), fontsize=22)
    slo = 1000.0 / prices["slo"]["tpot_p95_ms"]
    ax.axhline(slo, color=style.STATUS["warning"], lw=2)
    ax.annotate(f"usable: {slo:.0f} tok/s per user", (d["concurrency"].min(), slo), xytext=(4, 8),
                textcoords="offset points", color=style.ink("ink2"), fontsize=22)
    concurrency_axis(ax, d["concurrency"])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Concurrent users (log scale)")
    ax.set_ylabel("Decode speed per user (tok/s)")
    style.titles(fig, ax, f"More users, slower each ({workload})", "Median time per output token, converted to tok/s")
    style.source_line(fig, footer(prices))
    fig.savefig(out / f"user_speed_{workload}.png")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tables", default=str(ROOT / "results/tables"))
    ap.add_argument("--prices", default=str(ROOT / "configs/prices.yaml"))
    ap.add_argument("--out", default=str(ROOT / "results/figures"))
    a = ap.parse_args()
    prices = yaml.safe_load(Path(a.prices).read_text())
    df = pd.read_csv(Path(a.tables) / "cost_per_level.csv")
    mon = pd.read_csv(Path(a.tables) / "monthly_breakeven.csv")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    style.use("dark")
    for w in sorted(df["workload"].unique()):
        cost_per_m(df, prices, w, out)
        monthly(mon, prices, w, out)
        user_speed(df, prices, w, out)
    print("wrote", sorted(p.name for p in out.glob("*.png")))


if __name__ == "__main__":
    main()
