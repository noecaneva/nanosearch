"""Check that train_on on 2 GPUs gives the same update as on 1 GPU.

Fake episodes (fixed seed, real tokens, mask 1 on random spans, random advantages), 7 of them so
the 2 minibatches split unevenly across ranks. Each run saves the LoRA weights after one train_on
call (SGD, so the update is lr x the clipped gradient) and its stats; --compare reports the relative
difference of each run's update to the first one's. A second 1-GPU run gives the noise floor.

The lr is small on purpose: with a large first step (SGD lr 1, job 27232562) the second minibatch
clips ~35% of tokens, and rounding noise moving tokens across the clip boundary gave 0.5-1.9%
differences between jobs, which hides real errors.

  python -m checks.ddp_grad --out $TMPDIR/ddp1.pt
  python -m checks.ddp_grad --out $TMPDIR/ddp1b.pt
  python -m torch.distributed.run --standalone --nproc_per_node=2 -m checks.ddp_grad --out $TMPDIR/ddp2.pt
  python -m checks.ddp_grad --compare $TMPDIR/ddp1.pt $TMPDIR/ddp1b.pt $TMPDIR/ddp2.pt
"""

import argparse
import os
import random

import torch
import torch.distributed as dist

from nanosearch.grpo import build_policy, train_on


def fake_episodes(n=7, seed=0):
    rng = random.Random(seed)
    eps = []
    for _ in range(n):
        length = rng.randint(300, 900)
        ids = [rng.randrange(1000, 30000) for _ in range(length)]
        mask = [0] * length
        for _ in range(3):
            a = rng.randrange(100, length - 50)
            for j in range(a, min(length, a + rng.randint(10, 60))):
                mask[j] = 1
        eps.append({"ids": ids, "mask": mask, "adv": rng.uniform(-1.5, 1.5)})
    return eps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out")
    ap.add_argument("--compare", nargs="+", help="reference run first, then the others")
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()
    if args.compare:
        runs = [torch.load(p) for p in args.compare]
        deltas = [torch.cat([(r["after"][k] - r["before"][k]).flatten() for k in r["after"]])
                  for r in runs]
        rels = [float((d - deltas[0]).norm() / deltas[0].norm()) for d in deltas[1:]]
        for path, r, rel in zip(args.compare[1:], runs[1:], rels):
            print({"run": os.path.basename(path), "rel_update_diff": rel, "stats": r["stats"]})
        print({"run": os.path.basename(args.compare[0]), "reference": True, "stats": runs[0]["stats"]})
        print("PASS" if max(rels) < 1e-2 else "FAIL")
        return
    rank, world = int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1))
    if world > 1:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
    torch.manual_seed(0)
    model, params, _ = build_policy(1e-4)
    opt = torch.optim.SGD(params, lr=args.lr)  # update = -lr x clipped grad; Adam's 1st step ~ sign(grad)
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    # B starts at zero, so one step only moves B; give B a nonzero start so A gets gradient too
    with torch.no_grad():
        for n, p in zip(names, params):
            if "lora_B" in n:
                p.normal_(0, 1e-3)
        before = {n: p.detach().cpu().clone() for n, p in zip(names, params)}
    stats = train_on(model, params, opt, torch.amp.GradScaler("cuda"), fake_episodes(), 2, rank, world)
    if rank == 0:
        after = {n: p.detach().cpu().clone() for n, p in zip(names, params)}
        torch.save({"before": before, "after": after, "stats": stats}, args.out)
        print(stats)
    if world > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
