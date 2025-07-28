#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

MODEL_NAME_OR_PATH=e5-base
SPLIT="train"

OUTPUT_DIR=data/e5-base

DATA_DIR=data/tasks

PYTHONPATH=src/ python -u src/inference/search_topk.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --do_search \
    --fp16 \
    --search_split "${SPLIT}" \
    --search_topk 300 \
    --dataloader_num_workers 1 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --logging_dir "${OUTPUT_DIR}" \
