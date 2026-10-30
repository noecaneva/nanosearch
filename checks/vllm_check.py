"""Day-0 check: vLLM on a 2080 Ti (sm_75) serves Qwen2.5-1.5B-Instruct in fp16 and accepts a
LoRA adapter at runtime, the way the GRPO loop will refresh the policy each step.

Expects a server started with VLLM_ALLOW_RUNTIME_LORA_UPDATING=True and --enable-lora.

  1. one chat request against the base model
  2. build a random (non-zero) r=16 LoRA with peft, load it via /v1/load_lora_adapter,
     query it, then unload + reload it and time the swap
  3. throughput: N concurrent sampled completions (ignore_eos, fixed length), base and LoRA
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

MODEL = "Qwen/Qwen2.5-1.5B-Instruct"


def make_dummy_lora(path, rank=16):
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    if os.path.exists(os.path.join(path, "adapter_config.json")):
        return
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16)
    cfg = LoraConfig(r=rank, lora_alpha=2 * rank, init_lora_weights=False,
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                     "gate_proj", "up_proj", "down_proj"])
    model = get_peft_model(model, cfg)
    # Random init on both A and B so the adapter visibly changes outputs; scale it down
    # so the model stays coherent enough to read.
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "lora_B" in n:
                p.mul_(0.01)
    model.print_trainable_parameters()
    model.save_pretrained(path)


def chat(base, model, prompt, max_tokens=64, temperature=0.0):
    r = requests.post(f"{base}/v1/chat/completions", json={
        "model": model, "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens, "temperature": temperature}, timeout=600)
    r.raise_for_status()
    return r.json()


def completion(base, model, prompt, max_tokens, seed):
    r = requests.post(f"{base}/v1/completions", json={
        "model": model, "prompt": prompt, "max_tokens": max_tokens, "temperature": 1.0,
        "seed": seed, "ignore_eos": True}, timeout=600)
    r.raise_for_status()
    return r.json()["usage"]["completion_tokens"]


def throughput(base, model, n, max_tokens):
    prompt = "Write a detailed explanation of how search engines rank documents.\n"
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        toks = list(ex.map(lambda i: completion(base, model, prompt, max_tokens, i), range(n)))
    dt = time.time() - t0
    return {"model": model, "n_requests": n, "max_tokens": max_tokens,
            "generated_tokens": sum(toks), "seconds": round(dt, 2),
            "tokens_per_s": round(sum(toks) / dt, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://localhost:8011")
    ap.add_argument("--lora_dir", default=os.path.join(os.environ["GRPOP_ROOT"], "outputs",
                                                       "day0_dummy_lora"))
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--out", default=os.path.join(os.environ["GRPOP_ROOT"], "outputs",
                                                  "day0_vllm_check.json"))
    args = ap.parse_args()
    res = {}

    r = chat(args.base, MODEL, "What is the capital of Switzerland? Answer in one word.")
    res["base_answer"] = r["choices"][0]["message"]["content"]
    print("base:", res["base_answer"])

    make_dummy_lora(args.lora_dir)
    timings = []
    for i in range(3):
        t0 = time.time()
        requests.post(f"{args.base}/v1/unload_lora_adapter", json={"lora_name": "policy"})
        r = requests.post(f"{args.base}/v1/load_lora_adapter",
                          json={"lora_name": "policy", "lora_path": args.lora_dir})
        r.raise_for_status()
        timings.append(round(time.time() - t0, 3))
    res["lora_swap_seconds"] = timings
    r = chat(args.base, "policy", "What is the capital of Switzerland? Answer in one word.")
    res["lora_answer"] = r["choices"][0]["message"]["content"]
    print("lora:", res["lora_answer"], "| swap s:", timings)

    res["throughput"] = [throughput(args.base, MODEL, args.n, args.max_tokens),
                         throughput(args.base, "policy", args.n, args.max_tokens)]
    for t in res["throughput"]:
        print(t)
    with open(args.out, "w") as f:
        json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
