#!/usr/bin/env bash

set -x
set -e
DIR="$( cd "$( dirname "$0" )" && cd ../../ && pwd )"
echo "working directory: ${DIR}"


GPU_IDS="0"
PROC_PER_NODE=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
MODEL_NAME_OR_PATH=random
LLM_MODEL_NAME_OR_PATH="huggyllama/llama-7b"

DATA_DIR="${DIR}/data/tasks"
N_SHOTS=8
# EVAL_TASKS=("rte" "sst2" "ag_news" "boolq" "qnli" "mrpc" "snli") #"e2e_nlg" "snli" "wsc" "qqp" "dart" "mrpc" "yelp" "arc_c" "paws" "natural_questions" "gigaword" "aeslc" "arc_e" "multirc" 
EVAL_TASKS=("mnli_m" "mnli_mm") #"arc_c" "arc_e" "wsc" 
SEEDS=(1234)
LLM_BATCH_SIZE_PER_DEVICE=4
LLM_MAX_INPUT_LENGTH=1024
LLM_MAX_DECODE_LENGTH=64
N_PREFIX_TOKENS=10
LLM_EVAL_SPLIT=test

for SEED in "${SEEDS[@]}"; do
    echo "Using seed: ${SEED}"

    for TASK in "${EVAL_TASKS[@]}"; do
        echo "Processing task: ${TASK}"

        OUTPUT_DIR="${DIR}/output_baseline/inference_$(basename "${MODEL_NAME_OR_PATH}")_seed${SEED}/${TASK}/$(basename "${LLM_MODEL_NAME_OR_PATH}")"
        echo "Output directory: ${OUTPUT_DIR}"

        CUDA_VISIBLE_DEVICES=$GPU_IDS python src/inference/random_generate_few_shot_prompt.py \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
            --n_prefix_tokens "${N_PREFIX_TOKENS}" \
            --llm_batch_size_per_device "${LLM_BATCH_SIZE_PER_DEVICE}" \
            --seed "${SEED}" \
            --fp16 False \
            --llm_eval_tasks "${TASK}" \
            --llm_eval_split "${LLM_EVAL_SPLIT}" \
            --llm_k_shot "${N_SHOTS}" \
            --output_dir "${OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}"

        MASTER_PORT=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
        CUDA_VISIBLE_DEVICES=$GPU_IDS torchrun --nproc_per_node=${PROC_PER_NODE} --master_port ${MASTER_PORT} src/main_eval.py \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --seed "${SEED}" \
            --do_llm_eval \
            --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
            --llm_batch_size_per_device "${LLM_BATCH_SIZE_PER_DEVICE}" \
            --llm_eval_tasks "${TASK}" \
            --llm_eval_split "${LLM_EVAL_SPLIT}" \
            --llm_k_shot "${N_SHOTS}" \
            --llm_max_input_length "${LLM_MAX_INPUT_LENGTH}" \
            --llm_max_decode_length "${LLM_MAX_DECODE_LENGTH}" \
            --output_dir "${OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}" \
            --overwrite_output_dir \
            --disable_tqdm True \
            --report_to none "$@" \
            --fp16 False

        MODEL_FOLDER=$(basename "${LLM_MODEL_NAME_OR_PATH}")
        FOLDER_PATH="${OUTPUT_DIR}/${MODEL_FOLDER}"
        OUTPUT_CSV_PATH="${OUTPUT_DIR}/metrics_tasks.csv"
        PYTHONPATH=src/ python src/generate_csv.py \
            --folder_path "${FOLDER_PATH}" \
            --output_path "${OUTPUT_CSV_PATH}"

        echo "Completed task: ${TASK}"
        echo "----------------------------------------"
    done

done

echo "All tasks have been processed for all seeds."
