#!/bin/bash
# Exp 07: experiments/01-hotpotqa-grpo/run.sh with the model passed to serve.sh and the trainer
# (MODEL env var, default Qwen/Qwen3-1.7B). Run2 recipe, two seeds:
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/07-qwen3/run.sh q3_run2_s0 --lr 5e-5 --w_step 0 --seed 0
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/07-qwen3/run.sh q3_run2_s1 --lr 5e-5 --w_step 0 --seed 1
#SBATCH -J nanosearch-07
#SBATCH -p es0
#SBATCH -A pc_mlgeometry
#SBATCH -q es_normal
#SBATCH --gres=gpu:4
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 8:00:00
set -e
cd "$SLURM_SUBMIT_DIR"
source env.sh
# wandb-core did not start within its default 30 s in 4 of 6 starts on 2026-10-23, all 3 with
# 3 servers busy on 8 cores (ServicePollForTokenError; a node-local TMPDIR did not help).
export WANDB__SERVICE_WAIT=300
module load ai/vllm/0.19.0
source "$GRPOP_ROOT/envs/grpop/bin/activate"
RUN=$1; shift
MODEL=${MODEL:-Qwen/Qwen3-1.7B}
NGPU=$(nvidia-smi -L | wc -l)
NTRAIN=${NTRAIN:-1}
NSERVE=$((NGPU - NTRAIN))
URLS=()
for ((g = 0; g < NSERVE; g++)); do
  CUDA_VISIBLE_DEVICES=$g nanosearch/serve.sh $((8000 + g)) "$MODEL" > "$GRPOP_ROOT/logs/$RUN-serve$g.log" 2>&1 &
  URLS+=("http://localhost:$((8000 + g))")
done
trap 'kill $(jobs -p) 2>/dev/null' EXIT
for u in "${URLS[@]}"; do
  timeout 900 bash -c "until curl -sf $u/v1/models > /dev/null; do sleep 5; done"
done
export CUDA_VISIBLE_DEVICES=$(seq -s, $NSERVE $((NGPU - 1)))
if ((NTRAIN == 1)); then
  python -m nanosearch.grpo --model "$MODEL" --run_name "$RUN" --urls "${URLS[@]}" "$@"
else
  # torch.distributed.run = torchrun, but with the venv's python (the module's torchrun has no peft)
  python -m torch.distributed.run --standalone --nproc_per_node=$NTRAIN -m nanosearch.grpo --model "$MODEL" --run_name "$RUN" --urls "${URLS[@]}" "$@"
fi
