#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH="huggyllama/llama-7b"
SPLIT="e5-base_train"
DATA_DIR="data/generation_tasks"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
torchrun --nproc_per_node ${PROC_PER_NODE} src/inference/gen_llm_scores.py \
    --llm_model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --fp16 \
    --search_split "${SPLIT}" \
    --search_topk 32 \
    --llm_batch_size_per_device 16 \
    --max_train_samples 200000 \
    --output_dir "${DATA_DIR}" \
    --data_dir "${DATA_DIR}" \
    --report_to none "$@"
