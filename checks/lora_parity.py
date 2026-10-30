"""Day-0 parity: vLLM's LoRA path must match HF+peft, or GRPO's ratio is meaningless.

Scores one fixed text with vLLM (prompt_logprobs, base and adapter) and with HF+peft in fp32 on
CPU, and reports max/mean |delta logp| per token.
"""

import json
import os
import sys

import requests
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
BASE = "http://localhost:8011"
LORA = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.environ["GRPOP_ROOT"], "outputs",
                                                          "day0_dummy_lora")
TEXT = ("Question: Which magazine was started first, Arthur's Magazine or First for Women?\n"
        "<search>Arthur's Magazine founding year</search>")


def vllm_logprobs(model, ids):
    r = requests.post(f"{BASE}/v1/completions", json={
        "model": model, "prompt": ids, "max_tokens": 1, "temperature": 0,
        "prompt_logprobs": 0})
    r.raise_for_status()
    plp = r.json()["choices"][0]["prompt_logprobs"]
    # Entry i is a dict {token_id: {"logprob": ...}} for prompt token i (None for i=0).
    return [plp[i][str(ids[i])]["logprob"] for i in range(1, len(ids))]


def hf_logprobs(model, ids):
    with torch.no_grad():
        logits = model(torch.tensor([ids])).logits[0, :-1].float()
    lp = torch.log_softmax(logits, -1)
    return lp.gather(1, torch.tensor(ids[1:])[:, None])[:, 0].tolist()


def diff(a, b):
    d = [abs(x - y) for x, y in zip(a, b)]
    return {"max": round(max(d), 4), "mean": round(sum(d) / len(d), 4)}


tok = AutoTokenizer.from_pretrained(MODEL)
ids = tok(TEXT)["input_ids"]
v_base, v_lora = vllm_logprobs(MODEL, ids), vllm_logprobs("policy", ids)
base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.float32)
h_base = hf_logprobs(base, ids)
peft_model = PeftModel.from_pretrained(base, LORA)
h_lora = hf_logprobs(peft_model, ids)
res = {
    "n_tokens": len(ids) - 1,
    "vllm_base_vs_hf_base": diff(v_base, h_base),
    "vllm_lora_vs_hf_lora": diff(v_lora, h_lora),
    "hf_lora_vs_hf_base (adapter effect)": diff(h_lora, h_base),
    "vllm_lora_vs_vllm_base (adapter effect)": diff(v_lora, v_base),
    "sum_logp": {"vllm_base": sum(v_base), "hf_base": sum(h_base),
                 "vllm_lora": sum(v_lora), "hf_lora": sum(h_lora)},
}
print(json.dumps(res, indent=2))
with open(os.path.join(os.environ["GRPOP_ROOT"], "outputs", "day0_lora_parity.json"), "w") as f:
    json.dump(res, f, indent=2)
