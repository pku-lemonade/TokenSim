#!/usr/bin/env bash
#SBATCH -p c96
#SBATCH -N 1
#SBATCH --ntasks=96
#SBATCH --ntasks-per-node=96
#SBATCH --cpus-per-task=1
#SBATCH --exclusive
#SBATCH --array=0-3%4
#SBATCH -J minimax-m2.7-sweep
#SBATCH -o slurm/%x_%A_%a.out
#SBATCH -e slurm/%x_%A_%a.err

set -eo pipefail

: "${WORKDIR:?WORKDIR is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

source /etc/profile
module load anaconda3/2022.10
source /public/software/anaconda3/etc/profile.d/conda.sh
conda activate tokensim11
export PYTHONNOUSERSITE=1
export MPLCONFIGDIR=/tmp/tokensim11-mpl
set -u
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

traces=(traceA traceB thinking coder)
trace_name=${traces[${SLURM_ARRAY_TASK_ID:?}]}
export TRACE_NAME="$trace_name"
mkdir -p "$RUN_ROOT"/{logs,meta,progress,results,slurm,stderr}
printf 'job\tarray_task\ttrace\tnode\tcores\n%s\t%s\t%s\t%s\t96\n' \
    "$SLURM_JOB_ID" "$SLURM_ARRAY_TASK_ID" "$TRACE_NAME" "${SLURMD_NODENAME:-unknown}" \
    > "$RUN_ROOT/meta/job_${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}.tsv"

srun \
    --ntasks=96 \
    --ntasks-per-node=96 \
    --cpus-per-task=1 \
    --cpu-bind=cores \
    bash "$WORKDIR/scripts/hpc/minimax_m2_7_latency_sweep/run_core.sh"
