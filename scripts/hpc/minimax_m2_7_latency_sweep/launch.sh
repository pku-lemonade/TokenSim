#!/usr/bin/env bash
set -eo pipefail

WORKDIR=${1:?usage: launch.sh WORKDIR RUN_ROOT}
RUN_ROOT=${2:?usage: launch.sh WORKDIR RUN_ROOT}

source /etc/profile
module load anaconda3/2022.10
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate tokensim11
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/tokensim11-mpl
set -u

mkdir -p "$RUN_ROOT"/{logs,meta,progress,queues,results,slurm,stderr}
python "$WORKDIR/scripts/hpc/minimax_m2_7_latency_sweep/prepare.py" \
    --workdir "$WORKDIR" \
    --run-root "$RUN_ROOT"
sbatch \
    --export=ALL,WORKDIR="$WORKDIR",RUN_ROOT="$RUN_ROOT" \
    "$WORKDIR/scripts/hpc/minimax_m2_7_latency_sweep/submit.sh"
