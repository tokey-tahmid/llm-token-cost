#!/bin/bash
# Prepare pulled results for commit:
#  1. redact site-private identifiers (scripts/scrub_private.py, patterns in .private-patterns)
#  2. compress the bulky per-run files: raw request logs (per-chunk timings) and server logs.
#     bench/summarize.py and analysis/cost.py read .gz directly.
#   scripts/pack_results.sh            # every run under results/
set -euo pipefail
cd "$(dirname "$0")/.."
find results -type f ! -name '*.png' -print0 | xargs -0 python3 scripts/scrub_private.py
find results -type f \( -name 'raw_*.jsonl' -o -name '*.log' \) -print0 | xargs -0 -r gzip -9 -f -v
