#!/usr/bin/env bash

set -x
set -e
DIR="$( cd "$( dirname "$0" )" && cd ../../ && pwd )"
echo "working directory: ${DIR}"

EVAL_TASKS=("rte" "sst2" "ag_news" "boolq" "qnli" "mrpc" "snli")
MODEL_NAME_OR_PATH=sbert
LLM_MODEL_NAME_OR_PATH="EleutherAI/gpt-neo-2.7B"
LLM_Batch_Size_per_device=8

SEEDS=(87 91011)
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

        OUTPUT_DIR="${DIR}/output_baseline/inference_$(basename "${MODEL_NAME_OR_PATH}")_seed${SEED}/${TASK}/$(basename "${LLM_MODEL_NAME_OR_PATH}")"

        python src/inference/e5_topk_generate_few_shot_prompt.py \
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

        PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
        MASTER_PORT=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
        torchrun --nproc_per_node=${PROC_PER_NODE} --master_port ${MASTER_PORT} src/main_eval.py \
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
