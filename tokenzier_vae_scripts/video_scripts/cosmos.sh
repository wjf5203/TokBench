#!/bin/bash

CUDA_VISIBLE_DEVICES='4,5,6,7'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


DATAS=("ds" "ch3")

MODEL_NAMES=('CosmosCV4x8x8' 'CosmosCV8x16x16' 'CosmosDV4x8x8' 'CosmosDV8x16x16')  

SHORT_SIZES=(480 256)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATA in "${DATAS[@]}"; do
        for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
            echo "Running model: $MODEL_NAME with short_size: $SHORT_SIZE on dataset $DATA"
            
            for IDX in $(seq 0 $((CHUNKS-1))); do
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python cosmos_rec.py \
                    --video_path /path/to/TokBench/videos/text_data/$DATA  \
                    --save_path /path/to/video_reconstruction_results/$MODEL_NAME/text_data/$DATA \
                    --model $MODEL_NAME  \
                    --short_size $SHORT_SIZE \
                    --num_chunks $CHUNKS \
                    --chunk_idx $IDX &
            done
            
            wait 
            echo "Completed model: $MODEL_NAME with short_size: $SHORT_SIZE on dataset $DATA"
        done
    done
done

DATAS=("face_clip_3s")

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATA in "${DATAS[@]}"; do
        for SHORT_SIZE in "${SHORT_SIZES[@]}"; do
            echo "Running model: $MODEL_NAME with short_size: $SHORT_SIZE on dataset $DATA"
            
            for IDX in $(seq 0 $((CHUNKS-1))); do
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python cosmos_rec.py \
                    --video_path /path/to/TokBench/videos/text_data/$DATA  \
                    --save_path /path/to/video_reconstruction_results/$MODEL_NAME/text_data/$DATA \
                    --model $MODEL_NAME  \
                    --short_size $SHORT_SIZE \
                    --num_chunks $CHUNKS \
                    --chunk_idx $IDX &
            done
            
            wait 
            echo "Completed model: $MODEL_NAME with short_size: $SHORT_SIZE on dataset $DATA"
        done
    done
done