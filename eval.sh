#!/bin/bash

tokenizer_name="tokenizer1"
recon_result="path/to/reconstruction_results/"


dataset_names=("ic13" "ic15" "tt" "textocr" "cord" "sroie" "infograph" "docvqa")
for dataset_name in "${dataset_names[@]}"
do
  python eval_text.py  \
  --img_folder "${recon_result}/text_data/${dataset_name}/" \
  --gt_path "TokBench/annotations/text_${dataset_name}.json" \
  --dataset "${dataset_name}" \
  --batch_size 64 \
  --method_name $tokenizer_name \
  --setting 256  &\
done
wait


python eval_face.py  \
    --original_image_path TokBench/images/face_data/ \
    --reconstruction_image_path "${recon_result}/face_data/" \
    --tokenizer $tokenizer_name \
    --meta_path TokBench/annotations/face_meta.json \
    --setting 256 


python compute_all_metrics.py  --setting 256 