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
# entity colors: a configuration keeps its palette slot in every chart (docs: style slots 1-4)
SLOT = {"qwen27b-bf16-tp2": 0, "qwen27b-fp8-tp1": 1, "gptoss120b-tp2": 2, "gptoss120b-tp4": 3}


def color(cfg: str) -> str:
    if cfg not in SLOT:  # a new configuration takes the next free slot, in order of first use
        SLOT[cfg] = len(SLOT)
    return style.SLOTS[SLOT[cfg]]


API_DAY = ""  # provider snapshot date, set in main from the tables


def footer(prices: dict, extra: str = "") -> str:
    g = prices["gpu_list_prices"]
    return (f"A100-SXM4-40GB · vLLM · medians of 3 reps · measured\n"
            f"GPU time priced at public list rates ${g['low']['usd_per_gpu_hour']}–{g['high']['usd_per_gpu_hour']}"
            f"/GPU-h ({g['as_of']}) · API: every live provider's list price for this model ({API_DAY}){extra}")


def pretty(cfg: str, model: str, tp: int) -> str:
    """Viewer-facing config name: model, precision when the label names one, GPU count."""
    name = model.split("/")[-1]
    prec = next((p.upper() for p in ("bf16", "fp8", "int4", "awq", "gptq") if p in cfg.lower()), "")
    eng = ", SGLang" if "sglang" in cfg.lower() else ""
    return f"{name}{' ' + prec if prec else ''}, {tp} GPU{'s' if tp > 1 else ''}{eng}"


def end_labels(ax, items, min_gap_frac=0.06, log=True) -> None:
    """Labels right of each line's last point, items = (x, y, text, color). Called after the
    axis limits are final; labels closer than min_gap_frac of the y range are pushed apart."""
    lo, hi = ax.get_ylim()
    f, inv = (np.log10, lambda v: 10 ** v) if log else ((lambda v: v), (lambda v: v))
    gap, placed = (f(hi) - f(lo)) * min_gap_frac, []
    for x, y, text, color in sorted(items, key=lambda t: t[1]):
        yy = f(y) if not placed else max(f(y), placed[-1] + gap)
        placed.append(yy)
        ax.annotate(text, (x, y), xytext=(14, inv(yy)), textcoords=("offset points", "data"),
                    va="center", color=color, fontsize=22, annotation_clip=False)


def slo_markers(ax, g, y, color) -> None:
    """Filled marker: level meets the SLO. Hollow: it does not (usually requests queueing)."""
    ok = g["within_slo"].astype(bool)
    ax.plot(g.loc[ok, "concurrency"], g.loc[ok, y], "o", ls="none", ms=12, mfc=color, mec=color, mew=3)
    ax.plot(g.loc[~ok, "concurrency"], g.loc[~ok, y], "o", ls="none", ms=12, mfc=style.ink("surface"), mec=color, mew=3)


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
    fig.subplots_adjust(left=0.1, right=0.78, top=0.8, bottom=0.2)
    labels = []
    for cfg, g in d.groupby("config", sort=False):
        g = g.sort_values("concurrency")
        c = color(cfg)
        ax.fill_between(g["concurrency"], g["usd_per_m_out_low"], g["usd_per_m_out_high"], color=c, alpha=0.12, lw=0)
        ax.plot(g["concurrency"], g["usd_per_m_out_mid"], color=c)
        slo_markers(ax, g, "usd_per_m_out_mid", c)
        last = g.iloc[-1]
        labels.append((last["concurrency"], last["usd_per_m_out_mid"], pretty(cfg, last["model"], last["tp"]), c))
    xmin, xmax = d["concurrency"].min(), d["concurrency"].max()
    for m, g in d.groupby("model"):  # API bill for the same requests, input tokens included
        for col, what, dash in (("api_usd_per_m_out_median", "median provider", (0, (6, 4))),
                                ("api_usd_per_m_out_cheapest", "cheapest provider", (0, (2, 3)))):
            api = g[col].median()
            if np.isnan(api):
                continue
            ax.hlines(api, xmin, xmax, color=style.ink("ink2"), lw=2, ls=dash)
            ax.annotate(f"API, {what}: ${api:.2f}/M", (xmin, api), xytext=(4, -10), va="top",
                        textcoords="offset points", color=style.ink("ink2"), fontsize=22)
    concurrency_axis(ax, d["concurrency"])
    dollar_log_axis(ax)
    ax.set_xlabel("Concurrent users (log scale)")
    ax.set_ylabel("USD per 1M output tokens (log scale)")
    end_labels(ax, labels)
    style.titles(fig, ax, f"What a million output tokens costs, {workload} workload",
                 "Line: mid A100 rate, band: low to high. Dashed: API bill for the same requests. Hollow: misses SLO.")
    style.source_line(fig, footer(prices, "\nSLO: first token within 2 s and at least 20 tok/s per user, both at p95"))
    fig.savefig(out / f"cost_per_m_{workload}.png")
    plt.close(fig)


def monthly(mon, prices, workload, out):
    for m, d in mon[mon["workload"] == workload].groupby("model"):
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=0.1, right=0.84, top=0.8, bottom=0.2)
        cfgs = d.drop_duplicates("config")
        refs = d.drop_duplicates("api_ref").set_index("api_ref")["api_usd_per_out_token_blended"]
        vmax = cfgs["capacity_out_tok_per_month"].max() * 1.6
        ytop = max(cfgs["fixed_usd_per_month_mid"].max() * 1.6, 1)
        x = np.linspace(0, vmax, 200)
        # the upper left is always clear (API lines start at the origin), so the legend goes there
        for r in cfgs.itertuples():
            c = color(r.config)
            cap = r.capacity_out_tok_per_month
            ax.plot([0, cap / 1e9], [r.fixed_usd_per_month_mid] * 2, color=c, lw=4,
                    label=f"{pretty(r.config, r.model, r.tp)}: ${r.fixed_usd_per_month_mid:,.0f}/month, "
                          f"up to {cap / 1e9:.1f}B tokens")
            ax.plot(cap / 1e9, r.fixed_usd_per_month_mid, "o", color=c)
        for ref, what, dash in (("median", "median provider", (0, (6, 4))), ("cheapest", "cheapest provider", (0, (2, 3)))):
            if ref in refs:
                ax.plot(x / 1e9, x * refs[ref], color=style.ink("ink2"), ls=dash, lw=2,
                        label=f"API, {what}: ${refs[ref] * 1e6:.2f} per 1M output tokens, input included")
        leg = ax.legend(loc="upper left", fontsize=21, handlelength=2.6, borderaxespad=0.3, frameon=True,
                        facecolor=style.ink("surface"), edgecolor=style.ink("surface"), framealpha=1)
        leg.set_zorder(5)  # opaque: lines pass behind it
        ax.set_xlabel("Output tokens per month (billions)")
        ax.set_ylabel("USD per month")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:,.0f}"))
        ax.set_xlim(0, vmax / 1e9)
        ax.set_ylim(0, ytop)
        style.titles(fig, ax, f"When does renting GPUs beat the API? {m.split('/')[-1]}, {workload}",
                     "Solid: GPUs rented 24/7 at the mid rate, flat up to their capacity within the SLO.")
        style.source_line(fig, footer(prices, "\ncapacity = max measured throughput with TTFT p95 ≤ 2 s and ≥ 20 tok/s per user (p95)"))
        fig.savefig(out / f"monthly_{workload}_{m.split('/')[-1]}.png")
        plt.close(fig)


def user_speed(df, prices, workload, out):
    d = df[df["workload"] == workload]
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=0.1, right=0.78, top=0.8, bottom=0.2)
    labels = []
    for cfg, g in d.groupby("config", sort=False):
        g = g.sort_values("concurrency")
        ax.plot(g["concurrency"], g["decode_tok_s_user"], color=color(cfg))
        slo_markers(ax, g, "decode_tok_s_user", color(cfg))
        last = g.iloc[-1]
        labels.append((last["concurrency"], last["decode_tok_s_user"], pretty(cfg, last["model"], last["tp"]),
                       color(cfg)))
    slo = 1000.0 / prices["slo"]["tpot_p95_ms"]
    ax.axhline(slo, color=style.ink("muted"), lw=2)  # neutral: amber is an entity slot
    ax.annotate(f"usable: {slo:.0f} tok/s per user", (d["concurrency"].min(), slo), xytext=(4, 8),
                textcoords="offset points", color=style.ink("ink2"), fontsize=22)
    concurrency_axis(ax, d["concurrency"])
    ax.set_ylim(bottom=0)
    ax.set_xlabel("Concurrent users (log scale)")
    ax.set_ylabel("Decode speed per user (tok/s)")
    end_labels(ax, labels, log=False)
    style.titles(fig, ax, f"More users, slower each ({workload})", "Median time per output token as tok/s. Hollow: misses the SLO.")
    style.source_line(fig, footer(prices, "\nSLO: first token within 2 s and at least 20 tok/s per user, both at p95"))
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
    global API_DAY
    snaps = sorted(Path(a.tables).parent.glob("prices/*-openrouter-*.json"))
    API_DAY = snaps[-1].name[:10] if snaps else "no snapshot"
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
