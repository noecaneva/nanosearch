"""GRPO from scratch: LoRA policy trained by HF/peft, rollouts sampled by a vLLM server.

One step:
  1. save the current adapter and load it into vLLM under a fresh name (weight sync by reload)
  2. sample n_prompts train questions, G episodes each, temperature 1
  3. reward each episode; advantage A_i = (r_i - mean_group) / (std_group + 1e-4);
     drop groups with zero reward variance (they carry no gradient)
  4. old log-probs in a no-grad pass, then `minibatches` optimizer steps on the clipped surrogate
     min(rho * A, clip(rho, 1-0.2, 1+0.2) * A), averaged over all model tokens in the minibatch
     (token-level normalization, not per rollout). beta = 0: no KL term, no reference model.

Precision (2080 Ti, no bf16): frozen fp16 base, fp32 LoRA weights, fp16 autocast + GradScaler,
log-probs in fp32 computed only at model-token positions, lm_head in 256-position chunks.
Only tokens with mask 1 (model-generated, built by SearchEnv) get loss.

vLLM and the trainer do not fit together on one 10.5 GB card: use
experiments/01-hotpotqa-grpo/run.sh, which serves on some GPUs and trains on the rest.

Several trainer GPUs (torchrun, NTRAIN=2 in run.sh): rank 0 generates, scores and broadcasts the
kept episodes; every rank takes every world-th episode of each minibatch and the LoRA gradients are
summed across ranks, so each update equals the one-GPU update (checks/ddp_grad.py).
"""

import argparse
import datetime
import json
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
import torch
import torch.distributed as dist
from torch.utils.checkpoint import checkpoint

from nanosearch import DATA_DIR, OUT_DIR
from nanosearch.agent import MODEL, VLLMClient, score
from nanosearch.baselines import load_questions
from nanosearch.env import W_STEP, SearchEnv, judge_score, reward, unsupported_titles
from nanosearch.indexes import get_retriever

CHUNK = 256
CLIP = 0.2
ADV_EPS = 1e-4


def build_policy(lr):
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float16,
                                                 attn_implementation="sdpa").cuda()
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()  # HF only checkpoints in train mode; Qwen2.5 has no dropout, so this is safe
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.0, task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        p.data = p.data.float()
    return model, params, torch.optim.AdamW(params, lr=lr, weight_decay=0.0)


def _chunk_logp(w, h, t):
    logits = (h @ w.T).float()
    return logits.log_softmax(-1).gather(-1, t[:, None])[:, 0]


def token_logprobs(model, ids, mask):
    """fp32 log-probs of the mask-1 tokens of one sequence."""
    base = model.get_base_model()
    x = torch.tensor([ids], device="cuda")
    sel = torch.tensor(mask[1:], device="cuda").bool()
    with torch.autocast("cuda", dtype=torch.float16):
        h = base.model(input_ids=x).last_hidden_state[0, :-1][sel]
        tgt = x[0, 1:][sel]
        return torch.cat([checkpoint(_chunk_logp, base.lm_head.weight, h[i:i + CHUNK],
                                     tgt[i:i + CHUNK], use_reentrant=False)
                          for i in range(0, len(tgt), CHUNK)])


def sync_adapter(model, urls, step, run_dir, prev):
    path = os.path.join(run_dir, "adapters", f"step{step:04d}")
    model.save_pretrained(path)
    name = f"{os.path.basename(run_dir)}-s{step}"
    for url in urls:
        if prev:
            requests.post(f"{url}/v1/unload_lora_adapter", json={"lora_name": prev}).raise_for_status()
        requests.post(f"{url}/v1/load_lora_adapter",
                      json={"lora_name": name, "lora_path": path}).raise_for_status()
    return name


def advantages(rewards, G):
    r = np.array(rewards).reshape(-1, G)
    std = r.std(1, keepdims=True)
    adv = (r - r.mean(1, keepdims=True)) / (std + ADV_EPS)
    keep = (std[:, 0] > 1e-6).repeat(G)
    assert np.allclose(adv.sum(1), 0, atol=1e-3)
    return adv.reshape(-1), keep


def train_on(model, params, opt, scaler, eps, minibatches, rank=0, world=1):
    """eps arrive shuffled; with world > 1 every rank gets the same eps and does its share."""
    mbs = np.array_split(np.arange(len(eps)), minibatches)
    mine = [i for mb in mbs for i in mb[rank::world]]
    with torch.no_grad():
        old = {i: token_logprobs(model, eps[i]["ids"], eps[i]["mask"]) for i in mine}
    stats = {"clip_frac": [], "ratio_max": [], "grad_norm": [], "loss": []}
    for mb in mbs:
        opt.zero_grad(set_to_none=True)
        n_tok = sum(sum(eps[i]["mask"][1:]) for i in mb)  # whole minibatch, all ranks
        for i in mb[rank::world]:
            lp = token_logprobs(model, eps[i]["ids"], eps[i]["mask"])
            ratio = torch.exp(lp - old[i])
            a = eps[i]["adv"]
            obj = torch.minimum(ratio * a, ratio.clamp(1 - CLIP, 1 + CLIP) * a)
            loss = -obj.sum() / n_tok
            scaler.scale(loss).backward()
            stats["loss"].append(loss.item())
            stats["clip_frac"].append(((ratio - 1).abs() > CLIP).float().mean().item())
            stats["ratio_max"].append(ratio.max().item())
        if world > 1:
            for p in params:
                if p.grad is None:
                    p.grad = torch.zeros_like(p)
                dist.all_reduce(p.grad)
        scaler.unscale_(opt)
        stats["grad_norm"].append(torch.nn.utils.clip_grad_norm_(params, 1.0).item())
        scaler.step(opt)
        scaler.update()
    stats["neg_lp"] = [float(o.neg().sum()) for o in old.values()]
    stats["n_old"] = [len(o) for o in old.values()]
    if world > 1:
        parts = [None] * world
        dist.all_gather_object(parts, stats)
        stats = {k: sum((p[k] for p in parts), []) for k in stats}
        stats["grad_norm"] = parts[0]["grad_norm"]  # same on every rank (summed grads)
    neg_lp = sum(stats["neg_lp"]) / sum(stats["n_old"])
    return {"loss": float(np.sum(stats["loss"]) / minibatches),
            "clip_frac": float(np.mean(stats["clip_frac"])),
            "ratio_max": float(np.max(stats["ratio_max"])),
            "grad_norm": float(np.mean(stats["grad_norm"])),
            "neg_logp": neg_lp, "grad_scale": scaler.get_scale()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_name", required=True)
    ap.add_argument("--urls", nargs="+", default=["http://localhost:8000"])
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--n_prompts", type=int, default=16)
    ap.add_argument("--G", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--minibatches", type=int, default=2)
    ap.add_argument("--w_step", type=float, default=W_STEP, help="training reward only; eval keeps core-v1")
    ap.add_argument("--overfit", type=int, default=0, help="train on this many fixed prompts")
    ap.add_argument("--train_file", default="train.jsonl", help="in DATA_DIR; train3000.jsonl = Exp 03")
    ap.add_argument("--separate_results", action="store_true",
                    help="result block in its own user message (Exp 02); default inline, core-v1")
    ap.add_argument("--reward", default="core", choices=["core", "judge", "mix"],
                    help="training reward base (Exp 04): core = recall@9, judge = judge_score of the "
                         "final turn, mix = 0.5 recall + 0.5 judge; format bonus and w_step as core")
    ap.add_argument("--workers", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl", timeout=datetime.timedelta(minutes=30))
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    model, params, opt = build_policy(args.lr)
    scaler = torch.amp.GradScaler("cuda")
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)
    if rank > 0:  # train on what rank 0 broadcasts, until it sends None
        while True:
            msg = [None]
            dist.broadcast_object_list(msg, 0)
            if msg[0] is None:
                break
            if msg[0]:
                train_on(model, params, opt, scaler, msg[0], args.minibatches, rank, world)
        dist.destroy_process_group()
        return
    run_dir = os.path.join(OUT_DIR, "runs", args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w") as f:
        json.dump({**vars(args), "trainer_gpus": world}, f, indent=2)

    import wandb
    from transformers import AutoTokenizer
    wandb.init(project="nanosearch", name=args.run_name, config=vars(args), dir=run_dir,
               mode=os.environ.get("WANDB_MODE", "offline"))
    env = SearchEnv(AutoTokenizer.from_pretrained(MODEL), get_retriever("bm25"),
                    separate_results=args.separate_results)
    questions = load_questions(os.path.join(DATA_DIR, args.train_file))
    if args.overfit:
        questions = random.sample(questions, args.overfit)
    log = open(os.path.join(run_dir, "metrics.jsonl"), "a")
    rollouts = open(os.path.join(run_dir, "rollouts.jsonl"), "a")
    adapter = None

    for step in range(args.steps):
        t0 = time.time()
        adapter = sync_adapter(model, args.urls, step, run_dir, adapter)
        client = VLLMClient(args.urls, adapter, temperature=1.0, seed=None)
        prompts = (questions if args.overfit else random.sample(questions, args.n_prompts))
        jobs = [q for q in prompts for _ in range(args.G)]
        with ThreadPoolExecutor(args.workers) as pool:
            eps = list(pool.map(lambda q: env.run(q["question"], client), jobs))
        t_gen = time.time() - t0
        scores = [score(q, e, args.w_step) for q, e in zip(jobs, eps)]
        for q, e, s in zip(jobs, eps, scores):  # judge metrics, logged for every reward mode
            titles = [env.docs[d]["title"] for d in q["gold_doc_ids"]]
            s["judge"] = judge_score(e["turns"], titles)
            s["n_named"], s["n_unsupported"] = unsupported_titles(
                e["turns"], titles, [env.docs[d]["title"] for d in e["retrieved"]])
            if args.reward != "core":
                base = s["judge"] if args.reward == "judge" else 0.5 * (s["recall@9"] + s["judge"])
                s["reward"] = reward(base, e["format_valid"], e["n_searches"], args.w_step)
        adv, keep = advantages([s["reward"] for s in scores], args.G)
        for e, a in zip(eps, adv):
            e["adv"] = float(a)
        train_eps = [{f: e[f] for f in ("ids", "mask", "adv")} for e, k in zip(eps, keep) if k]
        random.shuffle(train_eps)
        t1 = time.time()
        torch.cuda.reset_peak_memory_stats()
        if world > 1:
            dist.broadcast_object_list([train_eps], 0)
        stats = (train_on(model, params, opt, scaler, train_eps, args.minibatches, rank, world)
                 if train_eps else {})
        m = {"step": step,
             **{k: float(np.mean([s[k] for s in scores]))
                for k in ("reward", "recall@9", "all_gold@9", "judge")},
             # share of gold titles named in the final turn that no retrieved passage carries
             "unsupported": (sum(s["n_unsupported"] for s in scores)
                             / max(1, sum(s["n_named"] for s in scores))),
             "titles_named": float(np.mean([s["n_named"] for s in scores])),
             "reward_std": float(np.std([s["reward"] for s in scores])),
             "format_valid": float(np.mean([e["format_valid"] for e in eps])),
             "n_searches": float(np.mean([e["n_searches"] for e in eps])),
             "n_generated": float(np.mean([e["n_generated"] for e in eps])),
             "query_words": float(np.mean([len(q.split()) for e in eps for q in e["queries"]] or [0])),
             "frac_groups_filtered": 1 - keep.mean(),
             "model_tokens": int(sum(sum(e["mask"]) for e in eps)),
             "t_gen": t_gen, "t_train": time.time() - t1,
             "peak_mem_gb": torch.cuda.max_memory_allocated() / 2**30, **stats}
        print(json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in m.items()}),
              flush=True)
        log.write(json.dumps(m) + "\n")
        log.flush()
        wandb.log(m, step=step)
        for q, e, s in list(zip(jobs, eps, scores))[:args.G]:  # first group, for the hacking log
            rollouts.write(json.dumps({"step": step, "question": q["question"], "reward": s["reward"],
                                       "recall@9": s["recall@9"], "judge": s["judge"],
                                       "queries": e["queries"], "retrieved": e["retrieved"],
                                       "end": e["end"],
                                       "turns": [t["text"] for t in e["turns"]]}) + "\n")
        rollouts.flush()
    sync_adapter(model, args.urls, args.steps, run_dir, adapter)
    wandb.finish()
    if world > 1:
        dist.broadcast_object_list([None], 0)
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
