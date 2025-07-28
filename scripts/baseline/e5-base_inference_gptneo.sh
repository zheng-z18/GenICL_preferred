#!/usr/bin/env bash

set -x
set -e
DIR="$( cd "$( dirname "$0" )" && cd ../../ && pwd )"
echo "working directory: ${DIR}"

GPU_ID=1

MODEL_NAME_OR_PATH=e5-base
LLM_MODEL_NAME_OR_PATH="EleutherAI/gpt-neo-2.7B"

DATA_DIR="${DIR}/data/tasks"
N_SHOTS=8
EVAL_TASKS=("snli") #"rte" "sst2" "ag_news" "boolq" "qnli" "mrpc" "snli"
OUTPUT_DIR="${DIR}/output_baseline/inference_$(basename "${MODEL_NAME_OR_PATH}")/${EVAL_TASKS[@]}/$(basename "${LLM_MODEL_NAME_OR_PATH}")"
SEED=1234
LLM_Batch_Size_per_device=8
LLM_Max_input_length=1024
LLM_Max_decode_length=64
N_PREFIX_TOKENS=10
LLM_EVAL_SPLIT=test

CUDA_VISIBLE_DEVICES="${GPU_ID}" PYTHONPATH=src/ python -u src/inference/e5_topk_generate_few_shot_prompt.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}"\
    --n_prefix_tokens "${N_PREFIX_TOKENS}" \
    --llm_batch_size_per_device "${LLM_Batch_Size_per_device}" \
    --seed "${SEED}" \
    --fp16 False \
    --llm_eval_tasks "${EVAL_TASKS[@]}" \
    --llm_eval_split "${LLM_EVAL_SPLIT}" \
    --llm_k_shot "${N_SHOTS}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \

CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/main_eval.py \
    --model_name_or_path "${MODEL_NAME_OR_PATH}" \
    --seed "${SEED}" \
    --do_llm_eval \
    --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
    --llm_batch_size_per_device "${LLM_Batch_Size_per_device}" \
    --llm_eval_tasks "${EVAL_TASKS[@]}" \
    --llm_eval_split "${LLM_EVAL_SPLIT}" \
    --llm_k_shot "${N_SHOTS}" \
    --llm_max_input_length "${LLM_Max_input_length}" \
    --llm_max_decode_length "${LLM_Max_decode_length}" \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --overwrite_output_dir \
    --disable_tqdm True \
    --report_to none "$@" \
    --fp16 False \


MODEL_FOLDER=$(basename "${LLM_MODEL_NAME_OR_PATH}")
FOLDER_PATH="${OUTPUT_DIR}/${MODEL_FOLDER}"
OUTPUT_CSV_PATH="${OUTPUT_DIR}/metrics_tasks.csv"
PYTHONPATH=src/ python src/generate_csv.py \
    --folder_path "${FOLDER_PATH}" \
    --output_path "${OUTPUT_CSV_PATH}"