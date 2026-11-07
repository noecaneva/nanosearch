#!/bin/bash
# Exp 07 day-0 checks on one 2080 Ti (README, "Day-0 checks"), in this order:
#   0. download Qwen3-1.7B; prompt checks (Qwen2.5 prompts byte-identical to core-v1, Qwen3 gets
#      enable_thinking=False and the empty <think> block in its generation prompt)
#   4. prompted evals in core and in the Exp 02 layout (runs 1 and 2: format-fixed baseline)
#   3. step-0 variance check, 500 train q x G=8 at temperature 1
#   2. one trainer step on the real batch (fit, NaN check), vLLM stopped meanwhile (both do not fit)
#   1. vLLM serves Qwen3 with the step-1 adapter and answers requests through the agent
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/07-qwen3/day0.sh [model]
#SBATCH -J nanosearch-07-day0
#SBATCH -p es0
#SBATCH -A pc_mlgeometry
#SBATCH -q es_debug
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 1:00:00
set -e
cd "$SLURM_SUBMIT_DIR"
source env.sh
module load ai/vllm/0.19.0
source "$GRPOP_ROOT/envs/grpop/bin/activate"
MODEL=${1:-Qwen/Qwen3-1.7B}
export MODEL
SECONDS=0
python -c "from huggingface_hub import snapshot_download; print(snapshot_download('$MODEL'))"
echo "download done at ${SECONDS}s"

python - <<'PY'
import os
from transformers import AutoTokenizer
from nanosearch.env import CORE, SYSTEM_PROMPT, T, SearchEnv, parse_turn
q25 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
d = SearchEnv(q25, None)
assert d.template_kwargs == {}, d.template_kwargs
msgs = [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": "Question: Q?"}]
assert d.prompt_ids("Q?") == q25.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
sep = SearchEnv(q25, None, separate_results=True)
assert (sep.close_turn, sep.open_turn) == ("<|im_end|>\n<|im_start|>user\n", "<|im_end|>\n<|im_start|>assistant\n")
q3 = AutoTokenizer.from_pretrained(os.environ["MODEL"])
e = SearchEnv(q3, None)
assert e.template_kwargs == {"enable_thinking": False}, e.template_kwargs
p = q3.decode(e.prompt_ids("Q?"))
assert p.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"), repr(p[-80:])
assert SYSTEM_PROMPT in p and "<search>" in p
assert q3.vocab_size == q25.vocab_size, (q3.vocab_size, q25.vocab_size)
print("tokenizer lengths (base vocab + added tokens):", len(q25), len(q3))
e2 = SearchEnv(q3, None, separate_results=True)
assert e2.close_turn == "<|im_end|>\n<|im_start|>user\n", repr(e2.close_turn)
assert e2.open_turn == "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", repr(e2.open_turn)
t = SearchEnv(q3, None, harness=T, separate_results=True)
assert T.system in q3.decode(t.prompt_ids("Q?")) and parse_turn("x <query>a</query>", T) == ("search", "a")
think = q3.decode(q3.encode("<think>\nhi\n</think>\nok", add_special_tokens=False), skip_special_tokens=True)
print("think tags survive skip_special_tokens:", "<think>" in think and "</think>" in think, repr(think))
assert "<think>" in think, "think tags are special tokens: think_turns would undercount"
print("Qwen3 generation prompt tail:", repr(p[-120:]))
print("prompt check PASS")
PY
echo "prompt check done at ${SECONDS}s"

serve() {
  nanosearch/serve.sh 8000 "$MODEL" > "$GRPOP_ROOT/logs/exp07-day0-serve-$SLURM_JOB_ID-$1.log" 2>&1 &
  SERVE=$!
  timeout 900 bash -c "until curl -sf localhost:8000/v1/models > /dev/null; do sleep 5; done"
  echo "server $1 up at ${SECONDS}s"
}
stop_serve() {
  kill $SERVE; wait $SERVE 2>/dev/null || true
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 1000 ]; do sleep 2; done
  echo "server stopped at ${SECONDS}s"
}
trap 'kill $(jobs -p) 2>/dev/null' EXIT

serve a
echo "=== run 1: prompted, core"
python -m nanosearch.agent --model "$MODEL" --out_name exp07_prompted_core | tail -4
echo "=== run 2: prompted, Exp 02 layout"
python -m nanosearch.agent --model "$MODEL" --separate_results --out_name exp07_prompted_sep | tail -4
echo "evals done at ${SECONDS}s"
echo "=== check 3: step-0 groups"
PYTHONPATH=. python experiments/07-qwen3/day0_groups.py --model "$MODEL"
echo "groups done at ${SECONDS}s"
stop_serve

echo "=== check 2: one trainer step"
PYTHONPATH=. python experiments/07-qwen3/day0_train_step.py --model "$MODEL"
echo "trainer step done at ${SECONDS}s"

echo "=== check 1: serve with the step-1 adapter"
serve b
python -m nanosearch.agent --model "$MODEL" --lora_path "$GRPOP_ROOT/outputs/exp07_day0_step/adapter_step1" \
  --limit 16 --out_name exp07_day0_lora | tail -4
echo "ALL DONE at ${SECONDS}s"
