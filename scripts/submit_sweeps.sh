#!/bin/bash
# Submit one sweep job per serving configuration in configs/sweeps.sh.
#   scripts/submit_sweeps.sh [label ...]      (default: all)
#   DEP=afterok:<jobid> scripts/submit_sweeps.sh   (hold until a probe succeeds)
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/sweeps.sh
for row in "${SWEEPS[@]}"; do
    IFS='|' read -r label model rev tp args body wall <<< "$row"
    if [[ $# -gt 0 && " $* " != *" $label "* ]]; then continue; fi
    LABEL=$label MODEL=$model REVISION=$rev TP=$tp VLLM_ARGS=$args EXTRA_BODY=$body \
        scripts/submit.sh slurm/sweep.sbatch --parsable --job-name="ltc-$label" --time="$wall" \
        --export=ALL ${DEP:+--dependency=$DEP}
done
