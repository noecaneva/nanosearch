#!/bin/bash
# Multi-GPU trainer check, one es_debug node: checks/ddp_grad.py (1 GPU vs 2 GPUs), then 5 GRPO
# steps with 3 servers + 1 trainer and with 2 + 2, for timing.
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/01-hotpotqa-grpo/ddp_bench.sh
#SBATCH -J nanosearch-ddp
#SBATCH -p es0
#SBATCH -A pc_mlgeometry
#SBATCH -q es_debug
#SBATCH --gres=gpu:4
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 0:59:00
cd "$SLURM_SUBMIT_DIR"
source env.sh
module load ai/vllm/0.19.0
source "$GRPOP_ROOT/envs/grpop/bin/activate"
O=$GRPOP_ROOT/outputs/ddp_check
mkdir -p $O
set -x
CUDA_VISIBLE_DEVICES=0 python -m checks.ddp_grad --out $O/ddp1.pt || exit 1
CUDA_VISIBLE_DEVICES=1 python -m checks.ddp_grad --out $O/ddp1b.pt || exit 1
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc_per_node=2 -m checks.ddp_grad --out $O/ddp2.pt || exit 1
python -m checks.ddp_grad --compare $O/ddp1.pt $O/ddp1b.pt $O/ddp2.pt | tee $O/compare.txt
grep -q PASS $O/compare.txt || exit 1
A="--lr 5e-5 --w_step 0 --steps 5"
NTRAIN=1 bash experiments/01-hotpotqa-grpo/run.sh ddpbench_3p1 $A
sleep 20
NTRAIN=2 bash experiments/01-hotpotqa-grpo/run.sh ddpbench_2p2 $A
