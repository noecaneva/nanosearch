#!/bin/bash
# Exp 06: run2 adapters (steps 10, 50, 150) and the prompted model in harnesses core / W / T, greedy,
# 200 eval questions, eight evals on one vLLM server (each eval ~30-60 s, server start ~1-2 min).
#   sbatch -o $SCRATCH/grpop/logs/%x-%j.out experiments/06-harness-transfer/evals.sh
#SBATCH -J nanosearch-06
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
A=$GRPOP_ROOT/outputs/runs/run2_wstep0/adapters
for s in step0010 step0050 step0150; do test -d $A/$s || { echo "missing $A/$s"; exit 1; }; done

# Pre-registered checks (README): defaults byte-identical to core-v1; W keeps the tokens; T renames
# them and the old tokens are not actions in T.
python - <<'PY'
from transformers import AutoTokenizer
from nanosearch.env import ACTION_RE, CORE, REMINDER, STOP, SYSTEM_PROMPT, T, W, SearchEnv, parse_turn
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
assert STOP == ["</search>", "<done/>", "<result>"] and ACTION_RE.pattern == r"<search>(.*?)</search>|<done/>"
assert REMINDER == "Do not answer the question. End this turn with <search>...</search> or <done/>.\n"
d = SearchEnv(tok, None)
assert d.harness is CORE and d.system == SYSTEM_PROMPT and d.prompt_ids("Q?") == SearchEnv(tok, None, harness=CORE).prompt_ids("Q?")
assert d.result_block.__func__ is SearchEnv.result_block and CORE.reminder == REMINDER and CORE.stop == STOP
assert "<search>" in W.system and "<done/>" in W.system and "<query>" not in W.system and "<stop/>" not in W.system
assert W.stop == STOP and W.reminder == REMINDER
assert "<query>" in T.system and "<stop/>" in T.system and "<search>" not in T.system and "<done/>" not in T.system
assert T.stop == ["</query>", "<stop/>", "<result>"]
assert T.reminder == "Do not answer the question. End this turn with <query>...</query> or <stop/>.\n"
assert parse_turn("I need X. <search>x</search>", T) == ("malformed", "no_action")
assert parse_turn("I have both. <done/>", T) == ("malformed", "no_action")
assert parse_turn("I need X. <query>x</query>", T) == ("search", "x") and parse_turn("ok <stop/>", T) == ("done", None)
assert parse_turn("I need X. <query>x</query>") == ("malformed", "no_action")  # core default unchanged
assert parse_turn("I need X. <search>x</search>") == ("search", "x")
t = SearchEnv(tok, None, harness=T, separate_results=True)
assert T.system in tok.decode(t.prompt_ids("Q?")) and "<search>" not in tok.decode(t.prompt_ids("Q?"))
print("prompt check PASS")
PY

nanosearch/serve.sh 8000 > "$GRPOP_ROOT/logs/exp06-serve-$SLURM_JOB_ID.log" 2>&1 &
trap 'kill $(jobs -p) 2>/dev/null' EXIT
timeout 900 bash -c "until curl -sf localhost:8000/v1/models > /dev/null; do sleep 5; done"

load() { curl -sf localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
           -d "{\"lora_name\": \"$1\", \"lora_path\": \"$A/$1\"}"; echo; }
unload() { curl -sf localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' \
             -d "{\"lora_name\": \"$1\"}"; echo; }
ev() { echo "=== exp06_$1"; python -m nanosearch.agent --out_name "exp06_$1" "${@:2}" | tail -3; }

# adapters one at a time (serve.sh has --max-loras 2)
load step0150
ev step0150_W    --lora step0150 --harness W
ev step0150_T    --lora step0150 --harness T
ev step0150_core --lora step0150            # same-job rerun; noise floor vs agent_grpo_run2
unload step0150
load step0010
ev step0010_core --lora step0010
ev step0010_T    --lora step0010 --harness T
unload step0010
load step0050
ev step0050_core --lora step0050
ev step0050_T    --lora step0050 --harness T
unload step0050
ev prompted_T --harness T                   # control; core exists: agent_prompted
echo "ALL DONE"
