#!/bin/bash
# Exp 08: lexical ladder + grid on eval (200) and train, then the tables. CPU work only; the GPU is
# requested because es0 jobs are GPU jobs (same request as Exp 07 evals.sh).
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/08-lexical-ladder/run.sh
#SBATCH -J nanosearch-08-ladder
#SBATCH -p es0
#SBATCH -A pc_mlgeometry
#SBATCH -q es_debug
#SBATCH --gres=gpu:1
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 0:30:00
set -e
cd "$SLURM_SUBMIT_DIR"
source env.sh
module load ai/vllm/0.19.0   # the grpop venv is built on this module's Python
source "$GRPOP_ROOT/envs/grpop/bin/activate"
python experiments/08-lexical-ladder/ladder.py selftest
for split in eval train; do
  python experiments/08-lexical-ladder/ladder.py run --split $split   # exits if BM25 is not reproduced
  python experiments/08-lexical-ladder/analyze.py --split $split
done
