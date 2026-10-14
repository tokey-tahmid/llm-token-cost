#!/bin/bash
# Submit a job script with the account/partition/QOS from site.env, refusing
# duplicates (a job with the same name already queued or running).
#   scripts/submit.sh slurm/<job>.sbatch [extra sbatch args]
set -euo pipefail
cd "$(dirname "$0")/.."
set +u; source "${SITE_ENV:-site.env}"; set -u
JOB=$1; shift
name=$(sed -n 's/^#SBATCH --job-name=//p' "$JOB")
for a in "$@"; do [[ $a == --job-name=* ]] && name=${a#*=}; done
if squeue -h -u "$USER" -o "%j" | grep -qx "$name"; then
    echo "refused: a job named $name is already queued or running" >&2; exit 1
fi
sbatch --account="$SLURM_ACCOUNT" --partition="$SLURM_PARTITION" ${SLURM_QOS:+--qos=$SLURM_QOS} "$@" "$JOB"
