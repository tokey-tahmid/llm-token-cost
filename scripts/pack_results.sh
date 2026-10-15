#!/bin/bash
# Prepare pulled results for commit:
#  1. redact site-private identifiers (scripts/scrub_private.py, patterns in .private-patterns)
#  2. compress the bulky per-run files: raw request logs (per-chunk timings) and server logs.
#     bench/summarize.py and analysis/cost.py read .gz directly.
#   scripts/pack_results.sh                       # every finished run (job.txt has status=complete)
#   scripts/pack_results.sh results/raw/<run> ... # named runs, e.g. failed ones kept as records
# A run still in progress is never packed: deploy.sh pull-results skips packed folders, so a
# partial folder packed now would never be completed.
set -euo pipefail
cd "$(dirname "$0")/.."
dirs=()
if [[ $# -gt 0 ]]; then
    dirs=("$@")
else
    while IFS= read -r f; do dirs+=("$(dirname "$f")"); done < <(grep -l "status=complete" results/*/*/job.txt 2>/dev/null)
fi
[[ ${#dirs[@]} -eq 0 ]] && { echo "nothing to pack"; exit 0; }
find "${dirs[@]}" -type f ! -name '*.png' -print0 | xargs -0 python3 scripts/scrub_private.py
find "${dirs[@]}" -type f \( -name 'raw_*.jsonl' -o -name '*.log' \) -print0 | xargs -0 -r gzip -9 -f -v
