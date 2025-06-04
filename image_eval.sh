#!/bin/bash

tokenizer_name="resize"
recon_result="image_reconstruction_results"
resolution=256

dataset_names=("ic13" "ic15" "tt" "textocr" "cord" "sroie" "infograph" "docvqa")
for dataset_name in "${dataset_names[@]}"
do
  CUDA_VISIBLE_DEVICES=6 python eval_text.py  \
  --img_folder "${recon_result}/${tokenizer_name}/text_data/${dataset_name}_${resolution}/" \
  --gt_path "TokBench/annotations/text_${dataset_name}.json" \
  --dataset "${dataset_name}" \
  --data_type "image" \
  --batch_size 64 \
  --method_name $tokenizer_name \
  --setting $resolution  \
  --save_dir  image_outputs  &\
done
# wait


CUDA_VISIBLE_DEVICES=7 python eval_face.py  \
    --original_image_path TokBench/images/face_data/wflw \
    --reconstruction_image_path "${recon_result}/${tokenizer_name}/face_data/wflw_${resolution}/" \
    --tokenizer $tokenizer_name \
    --data_type "image" \
    --meta_path TokBench/annotations/face_meta.json \
    --setting $resolution \
    --save_dir  image_outputs  


python compute_all_metrics.py  --setting $resolution --data_type image  --output_path  image_outputs