#!/bin/bash
# Prompted agent (no LoRA), greedy, on the first 200 train questions (where the prompt was tuned)
# and the 200 eval questions, in
# 4 layouts: {inline, separate message} x {reminder, no reminder}. One GPU, es_debug.
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/02-result-message/eval_prompted.sh
#SBATCH -J nanosearch-resmsg
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
python -m checks.env_mask
nanosearch/serve.sh 8000 > "$GRPOP_ROOT/logs/resmsg-serve-$SLURM_JOB_ID.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
timeout 900 bash -c "until curl -sf localhost:8000/v1/models > /dev/null; do sleep 5; done"
for split in "train --limit 200" eval; do
  A="python -m nanosearch.agent --split $split"
  $A --out_name resmsg_inline_rem
  $A --out_name resmsg_inline_norem --no_reminder
  $A --out_name resmsg_sep_rem --separate_results
  $A --out_name resmsg_sep_norem --separate_results --no_reminder
done
