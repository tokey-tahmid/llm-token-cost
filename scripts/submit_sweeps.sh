#!/bin/bash
# Submit one sweep job per serving configuration in configs/sweeps.sh.
#   scripts/submit_sweeps.sh [label ...]      (default: all)
#   DEP=afterok:<jobid> scripts/submit_sweeps.sh   (hold until a probe succeeds)
#   RUN_TAG=knee CHAT_CONC="16 24 32 48 64" AGENT_CONC="..." scripts/submit_sweeps.sh <label>
#                                              (extra levels for one config, merged in analysis)
#   SBATCH_EXTRA="--qos=test --time=00:30:00" RUN_TAG=smoke CHAT_CONC=1 AGENT_CONC=1 REPS=1 \
#       scripts/submit_sweeps.sh <label>      (check a new engine or model before its full sweep)
set -euo pipefail
cd "$(dirname "$0")/.."
source configs/sweeps.sh
for row in "${SWEEPS[@]}"; do
    IFS='|' read -r label model rev tp args body wall engine <<< "$row"
    if [[ $# -gt 0 && " $* " != *" $label "* ]]; then continue; fi
    LABEL=$label MODEL=$model REVISION=$rev TP=$tp VLLM_ARGS=$args EXTRA_BODY=$body ENGINE=${engine:-vllm} \
        scripts/submit.sh slurm/sweep.sbatch --parsable --job-name="ltc-$label" --time="$wall" \
        --export=ALL ${DEP:+--dependency=$DEP} ${SBATCH_EXTRA:-}
done
