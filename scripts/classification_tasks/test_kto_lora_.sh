#!/usr/bin/env bash

set -x
# set -e
DIR="$( cd "$( dirname "$0" )" && cd ../../../ && pwd )"
echo "Working directory: ${DIR}"
CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"

GPU_IDS=(1)
GPU_COUNT=${#GPU_IDS[@]}

N_SHOTS=8
CHECKPOINTS=(10000 16000 18000 20000)
EVAL_TASKS=("dart")
N_PREFIX_TOKENS=10
LLM_EVAL_SPLIT=test
DATA_DIR="${DIR}/data/e5-base"
LLM_MODEL_NAME_OR_PATH="huggyllama/llama-7b"

SEED=1234
BATCH_SIZE_DS=128
BATCH_SIZE=4
LLM_Max_input_length=1024
LLM_Max_decode_length=64
GRADIENT_ACCUMULATION_STEPS=1
DATALOADER_NUM_WORKERS=1
INVERSE=False

TEST_FILE_NAME=test

process_task() {
    local EVAL_TASK=$1
    local GPU_ID=$2

    echo "========================================"
    echo "Processing Evaluation Task: ${EVAL_TASK} on GPU ${GPU_ID}"
    echo "========================================"
    TASK_OUTPUT_DIR="${DIR}/output_kto_theta_lora/${EVAL_TASK}/llama-7b"

    for CHECKPOINT in "${CHECKPOINTS[@]}"; do
        echo "----------------------------------------"
        echo "Processing checkpoint-${CHECKPOINT} for task ${EVAL_TASK} on GPU ${GPU_ID}"
        echo "----------------------------------------"

        Embed_File="${TASK_OUTPUT_DIR}/train/checkpoint-${CHECKPOINT}"
        OUTPUT_DIR="${TASK_OUTPUT_DIR}/${TEST_FILE_NAME}/checkpoint-${CHECKPOINT}"

        echo "Output directory: ${OUTPUT_DIR}"
        echo "Embedding file: ${Embed_File}"
        mkdir -p "${OUTPUT_DIR}"

        MASTER_PORT=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
        accelerate launch --num_processes 1 --gpu_ids "${GPU_ID}" --main_process_port "${MASTER_PORT}" src/test_kto_lora.py \
            --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
            --gradient_checkpointing True \
            --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
            --per_device_train_batch_size "${BATCH_SIZE_DS}" \
            --per_device_eval_batch_size "${BATCH_SIZE_DS}" \
            --llm_eval_tasks "${EVAL_TASK}" \
            --prefix_embed_file "${Embed_File}" \
            --seed "${SEED}" \
            --n_prefix_tokens "${N_PREFIX_TOKENS}" \
            --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
            --output_dir "${OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}" \
            --remove_unused_columns False \
            --overwrite_output_dir \
            --disable_tqdm True \
            --fp16 \
            --llm_k_shot "${N_SHOTS}" \
            --llm_eval_split "${LLM_EVAL_SPLIT}" \
            --inverse "${INVERSE}"

        MASTER_PORT=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
        CUDA_VISIBLE_DEVICES="${GPU_ID}" python src/main_eval.py \
            --model_name_or_path "${MODEL_NAME_OR_PATH}" \
            --seed "${SEED}" \
            --do_llm_eval \
            --llm_model_name_or_path "${LLM_MODEL_NAME_OR_PATH}" \
            --llm_batch_size_per_device "${BATCH_SIZE}" \
            --llm_eval_tasks "${EVAL_TASK}" \
            --llm_eval_split "${LLM_EVAL_SPLIT}" \
            --llm_k_shot "${N_SHOTS}" \
            --llm_max_input_length "${LLM_Max_input_length}" \
            --llm_max_decode_length "${LLM_Max_decode_length}" \
            --output_dir "${OUTPUT_DIR}" \
            --data_dir "${DATA_DIR}" \
            --overwrite_output_dir \
            --disable_tqdm True \
            --report_to none "$@" \
            --fp16

        MODEL_FOLDER=$(basename "${LLM_MODEL_NAME_OR_PATH}")
        FOLDER_PATH="${OUTPUT_DIR}/${MODEL_FOLDER}"
        OUTPUT_CSV_PATH="${OUTPUT_DIR}/${MODEL_FOLDER}-metrics.csv"
        PYTHONPATH=src/ python src/generate_csv.py \
            --folder_path "${FOLDER_PATH}" \
            --output_path "${OUTPUT_CSV_PATH}"

        echo "CSV generated at: ${OUTPUT_CSV_PATH}"
    done

    FOLDER_DIR="${TASK_OUTPUT_DIR}/${TEST_FILE_NAME}"
    MODEL_NAME="$(basename "${LLM_MODEL_NAME_OR_PATH}")"
    python tools/inference_aggregate/aggregate_all_metric.py --folder_path "${FOLDER_DIR}" --model_name "${MODEL_NAME}"

    echo "Completed processing for evaluation task: ${EVAL_TASK}"
}

for index in "${!EVAL_TASKS[@]}"; do
    EVAL_TASK=${EVAL_TASKS[index]}
    GPU_ID=${GPU_IDS[index % GPU_COUNT]}
    process_task "${EVAL_TASK}" "${GPU_ID}" &
done

wait

echo "All evaluation tasks processed successfully."
