#!/usr/bin/env bash

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

# MODEL_NAME_OR_PATH="bm25"

SPLIT="test"

OUTPUT_DIR=data/bm25

DATA_DIR=data/tasks

PYTHONPATH=src/ python -u src/inference/bm25s_search_topk.py \
    --do_search \
    --fp16 \
    --search_split "${SPLIT}" \
    --search_topk 100 \
    --dataloader_num_workers 1 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --logging_dir "${OUTPUT_DIR}" \
    # --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    # --report_to tensorboard "$@"
