# nanosearch: a testbed for small search agents (repo dir: grpop)

- Read `plan/10-handoff-2026-10-19.md` first (state + dated LOG), then `plan/09-short-plan.md` (active plan).
- Lawrencium cluster: load the `lawrencium-hpc` skill before cluster decisions. No compute on the login node;
  run work in es0 jobs (`-A pc_mlgeometry -q es_debug|es_normal`).
- Core code in `nanosearch/` (frozen as core-v1, data in `$GRPOP_ROOT/core-v1`); experiments in
  `experiments/NN-*/`; every result goes into `RESULTS.md`. Simplicity first.
- `source env.sh` first. Code stays in this repo; data, indexes, outputs, logs, venv and all caches go to
  `$SCRATCH/grpop` (home quota is tight).
- vLLM on 2080 Ti needs `--dtype float16 --attention-backend TRITON_ATTN --generation-config vllm`.
- Log every run in the handoff file's LOG. Write negative results down.
- Keep separate from the kinet project: don't read or write its files, sessions or memory.
