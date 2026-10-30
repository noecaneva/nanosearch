#!/bin/bash
# Exp 04 step 2: judge unit test, then the base model sampled like GRPO step 0 (500 train q x 8,
# temperature 1) and scored with the summary judge. One GPU, es_debug.
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/04-summary-judge/base_groups.sh
#SBATCH -J nanosearch-04base
#SBATCH -p es0
#SBATCH -A pc_mlgeometry
#SBATCH -q es_debug
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 0:50:00
set -e
cd "$SLURM_SUBMIT_DIR"
source env.sh
module load ai/vllm/0.19.0
source "$GRPOP_ROOT/envs/grpop/bin/activate"
python -m checks.judge_score
nanosearch/serve.sh 8000 > "$GRPOP_ROOT/logs/04base-serve-$SLURM_JOB_ID.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
timeout 900 bash -c "until curl -sf localhost:8000/v1/models > /dev/null; do sleep 5; done"
PYTHONPATH=. python experiments/04-summary-judge/base_groups.py
