#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}


DATAS=("ic13" "ic15" "textocr" "tt" "cord" "docvqa" "infograph" "sroie")

MODEL_NAME="resize"


PADDING_SIZES=(256 512 1024)

for DATA in "${DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "Running model: $MODEL_NAME with padding_size: $PADDING_SIZE on dataset $DATA"
        
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python resize_rec.py \
                --image_path /path/to/TokBench/images/text_data/$DATA \
                --save_path /path/to/image_reconstruction_results/$MODEL_NAME/text_data/$DATA \
                --padding_size $PADDING_SIZE \
                --num_chunks $CHUNKS \
                --chunk_idx $IDX &
        done
        wait 
    done
done


DATAS=("wflw") 

for DATA in "${DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "Running model: $MODEL_NAME with padding_size: $PADDING_SIZE on dataset $DATA"
        
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python resize_rec.py \
                --image_path /path/to/TokBench/images/face_data/$DATA \
                --save_path /path/to/image_reconstruction_results/$MODEL_NAME/face_data/$DATA \
                --padding_size $PADDING_SIZE \
                --num_chunks $CHUNKS \
                --chunk_idx $IDX &
        done
        wait 
    done
done