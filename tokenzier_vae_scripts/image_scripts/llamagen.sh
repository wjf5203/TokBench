#!/bin/bash

CUDA_VISIBLE_DEVICES='0,1,2,3'

gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

CHUNKS=${#GPULIST[@]}




MODEL_NAMES=("llamagen_vq16" "llamagen_vq8" )  


# text
DATAS=("ic13" "ic15" "textocr" "tt" "cord" "docvqa" "infograph" "sroie")

PADDING_SIZES=(256 512 1024)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
    for DATA in "${DATAS[@]}"; do
        for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
            echo "Running model: $MODEL_NAME with padding_size: $PADDING_SIZE on dataset $DATA"
            
            for IDX in $(seq 0 $((CHUNKS-1))); do
                CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python llamagen_rec.py \
                    --image_path /path/to/TokBench/images/text_data/$DATA \
                    --model_name $MODEL_NAME \
                    --save_path /path/to/image_reconstruction_results/$MODEL_NAME/text_data/$DATA \
                    --padding_size $PADDING_SIZE \
                    --num_chunks $CHUNKS \
                    --chunk_idx $IDX &
            done
            
            wait 
            echo "Completed model: $MODEL_NAME with padding_size: $PADDING_SIZE on dataset $DATA"
        done
    done
done


# face
DATAS=("wflw") 

for DATA in "${DATAS[@]}"; do
    for PADDING_SIZE in "${PADDING_SIZES[@]}"; do
        echo "Running model: $MODEL_NAME with padding_size: $PADDING_SIZE on dataset $DATA"
        
        for IDX in $(seq 0 $((CHUNKS-1))); do
            CUDA_VISIBLE_DEVICES=${GPULIST[$IDX]} python resize_rec.py \
                --image_path /path/to/TokBench/images/face_data/$DATA \
                --model_name $MODEL_NAME \
                --save_path /path/to/image_reconstruction_results/$MODEL_NAME/face_data/$DATA \
                --padding_size $PADDING_SIZE \
                --num_chunks $CHUNKS \
                --chunk_idx $IDX &
        done
        wait 
    done
done


