#!/usr/bin/env bash

set -x
set -e
DIR="$( cd "$( dirname "$0" )" && cd ../../ && pwd )"
echo "working directory: ${DIR}"

GPU_IDS="0"
PROC_PER_NODE=$(echo "$GPU_IDS" | awk -F',' '{print NF}')
# EVAL_TASKS=("rte" "sst2" "ag_news" "boolq" "qnli" "mrpc" "snli") #"e2e_nlg" "snli" "wsc" "qqp" "dart" "mrpc" "arc_c" "paws" "natural_questions" "gigaword" "aeslc" "arc_e" "multirc" 
EVAL_TASKS=("mnli_m" "mnli_mm") #"arc_c" "arc_e" "wsc" 
MODEL_NAME_OR_PATH=bm25
LLM_MODEL_NAME_OR_PATH="/huggyllama/llama-7b"
LLM_Batch_Size_per_device=4

SEEDS=(1234)
N_SHOTS=8
DATA_DIR="${DIR}/data/tasks"
LLM_EVAL_SPLIT=test
LLM_Max_input_length=1024
LLM_Max_decode_length=64
N_PREFIX_TOKENS=10

for SEED in "${SEEDS[@]}"; do
    echo "Using seed: ${SEED}"

    for TASK_NAME in "${EVAL_TASKS[@]}"; do
        echo "Processing task: ${TASK_NAME}"

        OUTPUT_DIR="${DIR}/output_baseline/inference_$(basename "${MODEL_NAME_OR_PATH}")_seed${SEED}/${TASK_NAME}/$(basename "${LLM_MODEL_NAME_OR_PATH}")"

        CUDA_VISIBLE_DEVICES=$GPU_IDS python src/inference/e5_topk_generate_few_shot_prompt.py \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
            --n_prefix_tokens "${N_PREFIX_TOKENS}" \
            --llm_batch_size_per_device "${LLM_Batch_Size_per_device}" \
            --seed "${SEED}" \
            --fp16 False \
            --llm_eval_tasks "${TASK_NAME}" \
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
            --llm_batch_size_per_device "${LLM_Batch_Size_per_device}" \
            --llm_eval_tasks "${TASK_NAME}" \
            --llm_eval_split "${LLM_EVAL_SPLIT}" \
            --llm_k_shot "${N_SHOTS}" \
            --llm_max_input_length "${LLM_Max_input_length}" \
            --llm_max_decode_length "${LLM_Max_decode_length}" \
            --output_dir "${OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}" \
            --overwrite_output_dir \
            --disable_tqdm True \
            --report_to none "$@"

        FOLDER_PATH="${OUTPUT_DIR}/$(basename "${LLM_MODEL_NAME_OR_PATH}")"
        OUTPUT_CSV_PATH="${OUTPUT_DIR}/metrics_tasks.csv"
        PYTHONPATH=src/ python src/generate_csv.py \
            --folder_path "${FOLDER_PATH}" \
            --output_path "${OUTPUT_CSV_PATH}"

        echo "Completed task: ${TASK_NAME}"
        echo "----------------------------------------"
    done
done

echo "All tasks have been processed for all seeds."
