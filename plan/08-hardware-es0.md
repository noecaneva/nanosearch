# 08. Running on es0: 4x RTX 2080 Ti

Target partition: es0, the free no-SU-charge nodes. 4 GPUs per node, 10.5 GiB usable VRAM each,
94 GB host RAM.

**Verdict: the project fits, with four changes.** The binding constraint is not the 10.5 GB, it is
that the RTX 2080 Ti is Turing, compute capability 7.5, and Turing has no bfloat16.

## Change 1: fp16 everywhere, and compute log-probs in fp32

Turing supports fp16 tensor cores but not bf16, which needs Ampere or newer. Every default in the
modern training stack assumes bf16, so this has to be set explicitly in the trainer, in vLLM, and in
the model load.

fp16 has a much narrower dynamic range than bf16, which matters more for RL than for supervised
training. The GRPO ratio is

```
rho = exp(logp_new - logp_old)
```

and it amplifies any error in either log-probability. Two mandatory precautions:

- **Cast logits to fp32 before `log_softmax`.** Never compute log-probs in fp16. This is cheap and
  it removes the main source of ratio noise.
- **Use standard AMP with fp32 master weights and a `GradScaler`.** Watch for the scaler repeatedly
  halving its scale, which means overflow and is the fp16 failure signature.

Log the ratio distribution every step. On bf16 hardware you would probably skip this; here it is how
you find out that the run is numerically sick rather than badly tuned.

## Change 2: LoRA instead of full fine-tuning

Full fine-tuning 1.5B needs roughly 24 GB just for weights, gradients and Adam state. It does not
fit, and no amount of batch-size reduction changes that.

LoRA does fit with room to spare:

| Component | Memory |
|---|---|
| Frozen base, 1.5B in fp16 | ~3.1 GB |
| LoRA adapters, r=16 on attention and MLP projections, roughly 18M params, with fp32 master, gradients and Adam | ~0.3 GB |
| Activations, batch 1, 2048 tokens, gradient checkpointing on | ~1 to 2 GB |
| Headroom | ~4 GB |

Keep LoRA constant across the model-size ablation. Comparing LoRA-1.5B against full-fine-tuned 0.5B
would confound rank with scale and the result would mean nothing.

**The sneaky memory cost is the logits, not the weights.** Qwen2.5 has a 151,936-token vocabulary.
One sequence of 2048 positions in fp32 is 2048 x 151936 x 4 bytes, about 1.2 GB, for a single
sequence. Chunk the log-prob computation over the sequence dimension, 256 positions at a time, or
this alone will run you out of memory on a card this size. Budget half a day for discovering this if
you do not do it up front.

## Change 3: split the four GPUs, do not co-locate

vLLM and the trainer cannot share a 10.5 GB card. Split them across the node instead, which is the
realistic production topology anyway.

```
GPU 0, 1  ->  two independent vLLM servers, rollout generation
GPU 2, 3  ->  trainer, DDP under torchrun
```

**Run two independent vLLM instances, not tensor parallel across two GPUs.** These are consumer
cards with no NVLink in a cluster chassis, so tensor parallelism would run over PCIe and be slower
than data parallelism. The model fits on one card, so shard the prompts across two servers instead
and merge the rollouts.

This still gives you genuine `torchrun` multi-GPU DDP on the training side, which was the point of
having it in the plan.

Generation dominates wall-clock in GRPO, so if the trainer sits idle, rebalance to 3 servers and 1
trainer. Measure before deciding.

## Change 4: attention backend and vLLM version

Flash Attention 2 requires Ampere. On Turing you get PyTorch SDPA's memory-efficient backend for
training, and xformers for vLLM, which is what it selects automatically on sm_75.

**Day-0 check, do this before anything else.** vLLM has been narrowing support for older
architectures across releases, and the current version may or may not still build and run on sm_75.
Verify by serving Qwen2.5-1.5B-Instruct with `--dtype float16` and sending one request. If the
current release refuses, pin an older version known to support Turing. If no workable version
exists, fall back to batched `model.generate` with HuggingFace, accept roughly an order of magnitude
less generation throughput, and say so in the write-up as a hardware limitation rather than a design
choice.

Do not discover this on day 7.

## Revised scale

| Dimension | Original | On es0 |
|---|---|---|
| Policy model | Qwen2.5-1.5B-Instruct | Unchanged, with LoRA r=16 |
| Ablation model | Qwen2.5-0.5B-Instruct | Unchanged, also LoRA r=16 |
| Precision | bf16 | fp16, fp32 log-probs |
| Training | Full fine-tune | LoRA |
| Group size G | 8 | 8, unchanged |
| Prompts per step | 16 | 8 to 16, tune to memory |
| Episode token budget | 2048 | Unchanged |
| Corpus | ~30,000 passages | Unchanged, indexes live in host RAM |

The corpus and both indexes sit in the 94 GB of host RAM without difficulty. FAISS flat over 30,000
passages with a small embedder is well under 100 MB. Nothing about the data side is constrained by
this hardware.

## Expected throughput

Rough arithmetic, to be replaced by measurement on day 1.

An episode generates roughly 450 tokens across three turns. One GRPO step at 16 prompts and G=8 is
128 episodes, so about 58,000 generated tokens. Two 2080 Tis serving a 1.5B model in fp16 at useful
batch sizes should give low thousands of tokens per second aggregate, putting generation at roughly
30 seconds per step, and a full step somewhere near a minute once training is included.

Three hundred steps is then about five hours. That is a comfortable overnight run, and it means you
can afford several full training runs inside the fourteen days rather than one.

## The free-partition tax

A no-SU-charge partition normally comes with preemption, short wall-clock limits, or both. Design
for it from the start:

- Checkpoint LoRA adapters and optimizer state every N steps. Adapters are small, so this is cheap.
- Make training resumable from a checkpoint with one flag, and test the resume path on day 5, not
  the first time a job is killed.
- Keep the wandb run id stable across resumes so a preempted run reads as one curve.
- Develop against a 200-question subset on a login node or a single GPU, and reserve full-node jobs
  for real runs.

Check the actual wall-clock limit and preemption policy before committing to a start date. If jobs
cap at a few hours, the checkpoint interval matters more than anything else in this file.

## What this hardware costs you scientifically

State these in the write-up's limitations section rather than leaving them for a reader to infer.

LoRA rather than full fine-tuning, so the result speaks to adapter-based RL and not to full-parameter
RL. fp16 rather than bf16, which is a genuine numerical difference worth one sentence. And no
seed-variance study unless time allows, since each configuration costs a full run.

None of these invalidate the project. All of them are the kind of constraint a real team works under,
and naming them precisely reads considerably better than pretending the run happened on an H100.
