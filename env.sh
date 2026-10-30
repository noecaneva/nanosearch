# Source this before any grpop work: keeps all heavy files on $SCRATCH, not $HOME.
export GRPOP_ROOT=$SCRATCH/grpop
export HF_HOME=$GRPOP_ROOT/hf_cache
export HF_DATASETS_CACHE=$HF_HOME/datasets
export SENTENCE_TRANSFORMERS_HOME=$HF_HOME/st
export PIP_CACHE_DIR=$GRPOP_ROOT/pip_cache
export XDG_CACHE_HOME=$GRPOP_ROOT/xdg_cache
export TRITON_CACHE_DIR=$GRPOP_ROOT/xdg_cache/triton
export TORCH_HOME=$GRPOP_ROOT/xdg_cache/torch
export VLLM_CACHE_ROOT=$GRPOP_ROOT/xdg_cache/vllm
export TORCHINDUCTOR_CACHE_DIR=$GRPOP_ROOT/xdg_cache/inductor
export WANDB_DIR=$GRPOP_ROOT/outputs
export WANDB_CACHE_DIR=$GRPOP_ROOT/xdg_cache/wandb
export TMPDIR=$GRPOP_ROOT/tmp
