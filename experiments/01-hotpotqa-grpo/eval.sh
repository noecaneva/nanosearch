#!/bin/bash
# Evaluate a trained adapter on the 200 eval questions (greedy), one GPU.
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/01-hotpotqa-grpo/eval.sh \
#          $SCRATCH/grpop/outputs/runs/run1/adapters/step0150 agent_grpo_run1 [--separate_results]
#SBATCH -J nanosearch-eval
#SBATCH -p es0
#SBATCH -A pc_mlgeometry
#SBATCH -q es_debug
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 0:30:00
set -e
cd "$SLURM_SUBMIT_DIR"
source env.sh
module load ai/vllm/0.19.0
source "$GRPOP_ROOT/envs/grpop/bin/activate"
nanosearch/serve.sh 8000 > "$GRPOP_ROOT/logs/eval-serve-$SLURM_JOB_ID.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
timeout 900 bash -c "until curl -sf localhost:8000/v1/models > /dev/null; do sleep 5; done"
python -m nanosearch.agent --lora_path "$1" --out_name "$2" "${@:3}"
