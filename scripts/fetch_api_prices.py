"""Snapshot every provider's price for the benchmarked models (OpenRouter's public endpoint list).

    python scripts/fetch_api_prices.py              # results/prices/<UTC date>-openrouter-<model>.json

The same open model is sold by many providers at prices that differ several times
over, so "the API price" is a distribution. analysis/cost.py compares against the
cheapest, median and priciest live endpoint for each workload's token mix. Raw
responses are kept, dated, so every figure can name the prices it used.
"""
from __future__ import annotations

import datetime as dt
import json
import urllib.request
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    prices = yaml.safe_load((ROOT / "configs/prices.yaml").read_text())
    routes = sorted({m["route"] for m in prices["api_list_prices"]["models"].values()})
    day = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d")
    out = ROOT / "results/prices"
    out.mkdir(parents=True, exist_ok=True)
    for route in routes:
        url = f"https://openrouter.ai/api/v1/models/{route}/endpoints"
        with urllib.request.urlopen(url, timeout=30) as r:
            data = json.load(r)
        data["_retrieved_utc"] = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        data["_url"] = url
        f = out / f"{day}-openrouter-{route.replace('/', '--')}.json"
        f.write_text(json.dumps(data, indent=1, sort_keys=True))
        eps = data.get("data", {}).get("endpoints", [])
        outp = sorted(float(e["pricing"]["completion"]) * 1e6 for e in eps)
        print(f"{route}: {len(eps)} endpoints, output ${outp[0]:.3g}-{outp[-1]:.3g}/M -> {f.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
