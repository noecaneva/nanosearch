#!/bin/bash
# Serve the policy with vLLM on one 2080 Ti. Usage: [GPU_UTIL=0.35] nanosearch/serve.sh [port] [model]
# The flags after --max-model-len are required on sm_75 with the ai/vllm/0.19.0 module
# (no sm_75 CUDA kernels in that build; see plan/10-handoff LOG 2026-10-19).
PORT=${1:-8000}
MODEL=${2:-Qwen/Qwen2.5-1.5B-Instruct}
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True
exec vllm serve "$MODEL" --port "$PORT" --dtype float16 \
  --enable-lora --max-lora-rank 16 --max-loras 2 \
  --gpu-memory-utilization ${GPU_UTIL:-0.85} --max-model-len 4096 \
  --attention-backend TRITON_ATTN --generation-config vllm
