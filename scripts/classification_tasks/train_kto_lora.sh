#!/usr/bin/env bash
set -x
DIR="$( cd "$( dirname "$0" )" && cd ../../../ && pwd )"
echo "Working directory: ${DIR}"

GPU_ID=0,1
GPU_COUNT=$(echo "$GPU_ID" | awk -F',' '{print NF}')

BATCH_SIZES=(16)
EVAL_TASKS=("hellaswag")

N_PREFIX_TOKENS=10
LEARNING_RATE=5e-4
POSITIVE=(0 1)
NEGATIVE=(-1 -2)
P_FORMAT=$(printf "P%s_N%s" "$(IFS=; echo "${POSITIVE[*]}")" "$(IFS=; echo "${NEGATIVE[*]}")")
Train_File="${DIR}/data/tasks/llama-7b_e5-base_train.jsonl.gz"
MODEL_name_or_path="huggyllama/llama-7b"
WARMUP_STEPS=3000
SEED=987
MAX_STEPS=30000
SAVE_STEPS=2000
EVAL_STEPS=2000
LOGGING_STEPS=1000
GRADIENT_ACCUMULATION_STEPS=1
DATALOADER_NUM_WORKERS=1

INVERSE=False
MODEL_NAME=$(basename "$MODEL_name_or_path")
DATA_DIR="${DIR}/data/tasks"

for BATCH_SIZE in "${BATCH_SIZES[@]}"; do
    echo "---------------------------------------------"
    echo "Starting training with BATCH_SIZE=${BATCH_SIZE}"
    echo "---------------------------------------------"
    
    for TASK in "${EVAL_TASKS[@]}"; do
        echo "Training for Evaluation Task: ${TASK}"
        
        OUTPUT_DIR="output_kto_lora/${P_FORMAT}/${TASK}/${MODEL_NAME}_bs$((BATCH_SIZE * GPU_COUNT))_lr${LEARNING_RATE}/train"
        Logging_Dir="${OUTPUT_DIR}"
        
        echo "Output Directory: ${OUTPUT_DIR}"
        echo "Logging Directory: ${Logging_Dir}"
        
        MASTER_PORT=$(python3 -c "import socket; s=socket.socket(socket.AF_INET, socket.SOCK_STREAM); s.bind(('', 0)); print(s.getsockname()[1]); s.close()")
        accelerate launch --config_file deepspeed.yaml --gpu_ids "${GPU_ID}" --num_processes ${GPU_COUNT} --main_process_port ${MASTER_PORT} src/train_kto_lora.py \
            --model_name_or_path "${MODEL_name_or_path}" \
            --gradient_checkpointing True \
            --per_device_train_batch_size "${BATCH_SIZE}" \
            --per_device_eval_batch_size "${BATCH_SIZE}" \
            --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}" \
            --llm_eval_tasks "${TASK}" \
            --do_train \
            --seed "${SEED}" \
            --topk_as_positive "${POSITIVE[@]}" \
            --bottomk_as_negative "${NEGATIVE[@]}" \
            --train_file "${Train_File}" \
            --n_prefix_tokens "${N_PREFIX_TOKENS}" \
            --dataloader_num_workers "${DATALOADER_NUM_WORKERS}" \
            --learning_rate "${LEARNING_RATE}" \
            --warmup_steps "${WARMUP_STEPS}" \
            --logging_steps "${LOGGING_STEPS}" \
            --output_dir "${OUTPUT_DIR}" \
            --evaluation_strategy no \
            --eval_steps "${EVAL_STEPS}" \
            --data_dir "${DATA_DIR}" \
            --max_steps "${MAX_STEPS}" \
            --save_strategy steps \
            --save_steps "${SAVE_STEPS}" \
            --remove_unused_columns False \
            --overwrite_output_dir \
            --disable_tqdm True \
            --logging_dir "${Logging_Dir}" \
            --report_to tensorboard "$@" \
            --fp16 \
            --inverse "${INVERSE}"

        echo "Training completed for Task: ${TASK} with BATCH_SIZE=${BATCH_SIZE}"
        echo "---------------------------------------------"
    done
done

echo "All training runs completed."
