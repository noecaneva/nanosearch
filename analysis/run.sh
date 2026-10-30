#!/bin/bash
# Both analyses, CPU only (BM25 re-runs), on an es0 node instead of the login node.
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out analysis/run.sh
#SBATCH -J nanosearch-analysis
#SBATCH -p es0
#SBATCH -A pc_mlgeometry
#SBATCH -q es_debug
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 00:30:00
set -e
cd "$SLURM_SUBMIT_DIR"
source env.sh
source "$GRPOP_ROOT/envs/grpop/bin/activate"
python -m analysis.reasoning > analysis/out/reasoning.txt
python -m analysis.habit > analysis/out/habit.txt
