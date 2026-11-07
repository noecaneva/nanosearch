#!/bin/bash
# Exp 07 runs 5-8 (+9): step-150 adapter of each seed in core (Rule 1) and in Exp 06's harness T
# (Rule 2), greedy, 200 eval questions, one vLLM server, adapters one at a time (--max-loras 2).
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/07-qwen3/evals.sh [s0 s1 ...]
#SBATCH -J nanosearch-07-evals
#SBATCH -p es0
#SBATCH -A pc_mlgeometry
#SBATCH -q es_debug
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 0:45:00
set -e
cd "$SLURM_SUBMIT_DIR"
source env.sh
module load ai/vllm/0.19.0
source "$GRPOP_ROOT/envs/grpop/bin/activate"
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
SEEDS=${@:-s0 s1}
for s in $SEEDS; do
  test -d "$GRPOP_ROOT/outputs/runs/q3_run2_$s/adapters/step0150" || { echo "missing adapter for $s"; exit 1; }
done

nanosearch/serve.sh 8000 "$MODEL" > "$GRPOP_ROOT/logs/exp07-evals-serve-$SLURM_JOB_ID.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
timeout 900 bash -c "until curl -sf localhost:8000/v1/models > /dev/null; do sleep 5; done"

for s in $SEEDS; do
  A="$GRPOP_ROOT/outputs/runs/q3_run2_$s/adapters/step0150"
  curl -sf localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"$s\", \"lora_path\": \"$A\"}"; echo
  echo "=== exp07_${s}_core"
  python -m nanosearch.agent --model "$MODEL" --lora $s --out_name "exp07_${s}_core" | tail -4
  echo "=== exp07_${s}_T"
  python -m nanosearch.agent --model "$MODEL" --lora $s --harness T --out_name "exp07_${s}_T" | tail -4
  curl -sf localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' \
    -d "{\"lora_name\": \"$s\"}"; echo
done
echo "ALL DONE"
